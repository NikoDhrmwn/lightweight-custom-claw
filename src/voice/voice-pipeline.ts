/**
 * LiteClaw — Discord Voice Pipeline Orchestrator
 *
 * Coordinates the full real-time speech loop:
 * User speaking -> Opus stream -> Decoded PCM -> VAD -> ASR -> LLM -> Sentence Buffer -> TTS -> Opus playback.
 * Implements user interruption/barge-in detection and sentence-by-sentence TTS queueing.
 */

import { Readable } from 'stream';
import { EventEmitter } from 'events';
import {
  AudioPlayer,
  AudioPlayerStatus,
  createAudioPlayer,
  createAudioResource,
  EndBehaviorType,
  joinVoiceChannel,
  NoSubscriberBehavior,
  StreamType,
  VoiceConnection,
} from '@discordjs/voice';
import * as prism from 'prism-media';
import { getConfig } from '../config.js';
import { createLogger } from '../logger.js';
import { AgentEngine } from '../core/engine.js';
import { LLMMessage } from '../core/llm.js';
import { discordToASR } from './audio-utils.js';
import { VoiceActivityDetector, SpeechSegment } from './vad.js';
import { ASRClient } from './asr-client.js';
import { TTSClient } from './tts-client.js';
import { SentenceBuffer } from './sentence-buffer.js';

const log = createLogger('voice:pipeline');

interface UserVoiceState {
  userId: string;
  username: string;
  vad: VoiceActivityDetector;
  audioStream?: any;
  decoderStream?: any;
}

export class VoicePipeline extends EventEmitter {
  private player: AudioPlayer;
  private connection: VoiceConnection;
  private engine: AgentEngine;
  private asrClient: ASRClient;
  private ttsClient: TTSClient;
  private sentenceBuffer: SentenceBuffer;

  private userStates = new Map<string, UserVoiceState>();
  private playbackQueue: Buffer[] = [];
  private voiceHistory: LLMMessage[] = [];
  
  private isAgentSpeaking = false;
  private isProcessingLlm = false;
  private shouldInterrupt = false;
  private currentLlmAbortController: AbortController | null = null;

  private config: any;
  private triggerMode: 'always' | 'wake-word';
  private wakeName: string;
  private maxResponseTokens: number;

  constructor(connection: VoiceConnection, engine: AgentEngine) {
    super();
    this.connection = connection;
    this.engine = engine;

    this.config = getConfig();
    const voiceConfig = this.config.voice || {};
    this.triggerMode = voiceConfig.triggerMode || 'always';
    this.wakeName = (voiceConfig.wakeName || this.config.agent?.name || 'liteclaw').toLowerCase();
    this.maxResponseTokens = voiceConfig.maxResponseTokens || 150;

    this.asrClient = new ASRClient();
    this.ttsClient = new TTSClient();
    
    // Configure sentence buffer for conversational pacing
    this.sentenceBuffer = new SentenceBuffer({
      maxChars: 180,
      minChars: 8,
      stripMarkdown: true,
    });

    // Create the audio player for playback
    this.player = createAudioPlayer({
      behaviors: {
        noSubscriber: NoSubscriberBehavior.Play,
      },
    });

    this.connection.subscribe(this.player);

    this.setupPlayerListeners();
    this.setupSentenceBufferListeners();
  }

  /**
   * Set up audio player event handlers to manage playback queueing.
   */
  private setupPlayerListeners(): void {
    this.player.on(AudioPlayerStatus.Idle, () => {
      log.debug('Audio player idle, checking queue...');
      this.playNextInQueue();
    });

    this.player.on('error', (err) => {
      log.error({ error: err.message }, 'Audio player error');
      this.playNextInQueue();
    });
  }

  /**
   * Set up sentence buffer to dispatch sentences to TTS immediately.
   */
  private setupSentenceBufferListeners(): void {
    this.sentenceBuffer.on('sentence', async (text: string, index: number) => {
      if (this.shouldInterrupt) return;
      log.info({ index, text }, 'Sentence boundary detected, sending to TTS...');
      
      try {
        const ttsResult = await this.ttsClient.synthesize(text);
        if (this.shouldInterrupt) return;

        if (ttsResult.pcm.length > 0) {
          this.enqueueAudio(ttsResult.pcm);
        }
      } catch (err: any) {
        log.error({ error: err.message }, 'Failed to synthesize sentence');
      }
    });

    this.sentenceBuffer.on('done', (remaining) => {
      log.debug({ remaining }, 'Sentence buffer parsing finished');
    });
  }

  /**
   * Starts listening to voice events in the Discord connection.
   */
  start(): void {
    log.info('Starting Discord Voice Pipeline...');
    
    this.connection.receiver.speaking.on('start', (userId) => {
      this.handleUserStartSpeaking(userId).catch((err) => {
        log.error({ error: err.message, userId }, 'Error handling user speech start');
      });
    });
  }

  /**
   * Cleans up all listeners and user states.
   */
  stop(): void {
    log.info('Stopping Discord Voice Pipeline...');
    this.interrupt();
    
    for (const [userId, state] of this.userStates) {
      this.cleanupUserState(userId);
    }
    
    this.userStates.clear();
    this.player.stop(true);
  }

  /**
   * Interrupt the agent's current output (Barge-in / User speaking).
   */
  private interrupt(): void {
    if (this.isAgentSpeaking || this.isProcessingLlm) {
      log.info('Interruption triggered: stopping playback and LLM stream');
    }

    this.shouldInterrupt = true;
    
    if (this.currentLlmAbortController) {
      this.currentLlmAbortController.abort();
      this.currentLlmAbortController = null;
    }

    this.sentenceBuffer.reset();
    this.playbackQueue = [];
    this.player.stop(true);
    this.isAgentSpeaking = false;
    this.isProcessingLlm = false;
  }

  /**
   * Subscribes to user audio, decodes to PCM, and runs VAD.
   */
  private async handleUserStartSpeaking(userId: string): Promise<void> {
    // If agent is speaking, interrupt immediately when user speaks! (Barge-in)
    this.interrupt();

    let state = this.userStates.get(userId);
    if (!state) {
      let username = 'User';
      try {
        const client = (this.connection as any).client;
        if (client) {
          const user = await client.users.fetch(userId);
          username = user?.username || 'User';
        }
      } catch {
        // Fallback if client is unavailable
      }

      const vadConfig = this.config.voice?.vad || {};
      const vad = new VoiceActivityDetector({
        silenceDurationMs: vadConfig.silenceDurationMs ?? 300,
        energyThreshold: vadConfig.energyThreshold ?? 0.01,
      });

      state = { userId, username, vad };
      this.userStates.set(userId, state);

      // Listen to VAD events
      vad.on('speechStart', () => {
        log.debug({ username }, 'User speech start detected by VAD');
        this.interrupt();
      });

      vad.on('speechEnd', (segment: SpeechSegment) => {
        this.handleUserSpeechSegment(state!, segment.audio).catch((err) => {
          log.error({ error: err.message, username }, 'Failed to process speech segment');
        });
      });
    }

    // Subscribe to the Opus stream from Discord
    if (!state.audioStream) {
      const audioStream = this.connection.receiver.subscribe(userId, {
        end: {
          behavior: EndBehaviorType.AfterSilence,
          duration: 300,
        },
      });

      // Decode Opus to 48kHz Stereo PCM
      const decoderStream = new prism.opus.Decoder({
        rate: 48000,
        channels: 2,
        frameSize: 960,
      });

      state.audioStream = audioStream;
      state.decoderStream = decoderStream;

      audioStream.pipe(decoderStream);

      decoderStream.on('data', (chunk: Buffer) => {
        // Resample 48kHz stereo to 16kHz mono for VAD
        const asrChunk = discordToASR(chunk);
        state!.vad.processChunk(asrChunk);
      });

      audioStream.on('end', () => {
        this.cleanupUserState(userId);
      });

      audioStream.on('error', (err) => {
        log.error({ error: err.message, username: state!.username }, 'Audio receive stream error');
        this.cleanupUserState(userId);
      });
    }
  }

  private cleanupUserState(userId: string): void {
    const state = this.userStates.get(userId);
    if (state) {
      if (state.decoderStream) {
        state.decoderStream.removeAllListeners();
        state.decoderStream.destroy();
      }
      if (state.audioStream) {
        state.audioStream.removeAllListeners();
        state.audioStream.destroy();
      }
      state.audioStream = undefined;
      state.decoderStream = undefined;
    }
  }

  /**
   * Processes a complete voice segment from VAD: transcribes via ASR and queries LLM if triggered.
   */
  private async handleUserSpeechSegment(state: UserVoiceState, speechPcm: Buffer): Promise<void> {
    log.info({ username: state.username, pcmBytes: speechPcm.length }, 'User speech segment completed, running ASR...');

    const asrResult = await this.asrClient.transcribe(speechPcm);
    const text = asrResult.text.trim();

    if (!text) {
      log.debug('ASR returned empty transcript. Ignoring.');
      return;
    }

    log.info({ username: state.username, transcript: text }, 'Speech transcribed');

    // Add user message to history
    this.voiceHistory.push({ role: 'user', name: state.username, content: text });
    
    // Keep history size small to prevent bloating the context
    if (this.voiceHistory.length > 20) {
      this.voiceHistory.shift();
    }

    // Check voice trigger mode
    let shouldRespond = false;
    if (this.triggerMode === 'always') {
      shouldRespond = true;
    } else if (this.triggerMode === 'wake-word') {
      if (text.toLowerCase().includes(this.wakeName)) {
        shouldRespond = true;
      } else {
        log.info({ wakeName: this.wakeName }, 'Wake name not found in transcript. Ignoring.');
      }
    }

    if (shouldRespond) {
      await this.generateAgentResponse();
    }
  }

  /**
   * Generates response from LLM, streams tokens to SentenceBuffer, and synthesizes speech.
   */
  private async generateAgentResponse(): Promise<void> {
    this.interrupt();
    this.shouldInterrupt = false;
    this.isProcessingLlm = true;

    this.currentLlmAbortController = new AbortController();

    log.info('Generating LLM response for voice...');

    // System prompt tailored for real-time voice chat
    const agentName = this.config.agent?.name || 'LiteClaw';
    const voiceSystemPrompt = `You are ${agentName}, a conversational voice assistant.
Having a real-time voice chat. Keep your response extremely brief, conversational, and simple (1-2 sentences max).
Never use list points, markdown, URLs, or long lists. Speak naturally, directly answering the user.`;

    try {
      const messages: LLMMessage[] = [
        { role: 'system', content: voiceSystemPrompt },
        ...this.voiceHistory,
      ];

      // Stream chat completion
      const generator = this.engine.getLLMClient().streamChat(
        messages,
        [], // Empty tools list -> disable tools for real-time voice chat to avoid latency
        {
          maxTokens: this.maxResponseTokens,
          disableReasoning: true, // Disable reasoning model blocks to minimize first-token latency
        }
      );

      this.sentenceBuffer.reset();

      let fullResponse = '';

      for await (const chunk of generator) {
        if (this.shouldInterrupt) {
          log.info('LLM response stream interrupted');
          break;
        }

        if (chunk.type === 'content' && chunk.content) {
          fullResponse += chunk.content;
          this.sentenceBuffer.addToken(chunk.content);
        }
      }

      this.sentenceBuffer.finish();
      this.isProcessingLlm = false;

      if (!this.shouldInterrupt && fullResponse.trim()) {
        log.info({ fullResponse }, 'LLM response generation finished');
        // Record agent response in history
        this.voiceHistory.push({ role: 'assistant', content: fullResponse.trim() });
        if (this.voiceHistory.length > 20) {
          this.voiceHistory.shift();
        }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        log.debug('LLM request aborted');
      } else {
        log.error({ error: err.message }, 'Failed to generate agent response');
      }
      this.isProcessingLlm = false;
    }
  }

  /**
   * Enqueues PCM audio and starts playback if player is idle.
   */
  private enqueueAudio(pcmBuffer: Buffer): void {
    if (this.shouldInterrupt) return;
    
    this.playbackQueue.push(pcmBuffer);
    
    if (!this.isAgentSpeaking) {
      this.playNextInQueue();
    }
  }

  /**
   * Plays the next PCM buffer in the queue.
   */
  private playNextInQueue(): void {
    if (this.shouldInterrupt) {
      this.isAgentSpeaking = false;
      return;
    }

    if (this.playbackQueue.length === 0) {
      log.debug('Playback queue empty. Agent finished speaking.');
      this.isAgentSpeaking = false;
      return;
    }

    this.isAgentSpeaking = true;
    const pcm = this.playbackQueue.shift()!;

    log.debug({ size: pcm.length }, 'Playing next synthesized audio chunk in Discord');

    try {
      const stream = Readable.from(pcm);

      // Create Opus encoder
      const encoder = new prism.opus.Encoder({
        rate: 48000,
        channels: 2,
        frameSize: 960,
      });

      const opusStream = stream.pipe(encoder);

      const resource = createAudioResource(opusStream, {
        inputType: StreamType.Opus,
      });

      this.player.play(resource);
    } catch (err: any) {
      log.error({ error: err.message }, 'Failed to play audio chunk');
      this.playNextInQueue();
    }
  }
}
