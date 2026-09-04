/**
 * LiteClaw — WebUI Voice Session Orchestrator
 *
 * Coordinates real-time voice chat with the WebUI.
 * Receives 16kHz mono PCM chunks from the browser,
 * feeds them to VAD, triggers ASR on speech completion,
 * prompts the LLM, feeds tokens to Sentence Buffer,
 * synthesizes speech with TTS, and streams WAV chunks back.
 */

import { WebSocket } from 'ws';
import { AgentEngine } from '../core/engine.js';
import { VoiceActivityDetector, SpeechSegment } from './vad.js';
import { ASRClient } from './asr-client.js';
import { TTSClient } from './tts-client.js';
import { SentenceBuffer } from './sentence-buffer.js';
import { getConfig } from '../config.js';
import { createLogger } from '../logger.js';
import { LLMMessage } from '../core/llm.js';

const log = createLogger('voice:webui');

export class WebUIVoiceSession {
  private ws: WebSocket;
  private engine: AgentEngine;
  private vad: VoiceActivityDetector;
  private asrClient: ASRClient;
  private ttsClient: TTSClient;
  private sentenceBuffer: SentenceBuffer;

  private mode: 'vad' | 'push-to-talk' = 'vad';
  private sessionKey = 'webui:default';
  
  private voiceHistory: LLMMessage[] = [];
  private isProcessingLlm = false;
  private shouldInterrupt = false;
  private currentLlmAbortController: AbortController | null = null;
  private accumulatedPttAudio: Buffer[] = [];

  constructor(ws: WebSocket, engine: AgentEngine, options: { mode: 'vad' | 'push-to-talk'; sessionKey: string }) {
    this.ws = ws;
    this.engine = engine;
    this.mode = options.mode;
    this.sessionKey = options.sessionKey;

    const config = getConfig();
    const voiceConfig = config.voice || {};
    const vadConfig = voiceConfig.vad || {};

    this.vad = new VoiceActivityDetector({
      silenceDurationMs: vadConfig.silenceDurationMs ?? 300,
      energyThreshold: vadConfig.energyThreshold ?? 0.01,
      sampleRate: 16000,
      channels: 1,
    });

    this.asrClient = new ASRClient();
    this.ttsClient = new TTSClient();
    this.sentenceBuffer = new SentenceBuffer({
      maxChars: 180,
      minChars: 8,
      stripMarkdown: true,
    });

    this.setupListeners();
  }

  private setupListeners(): void {
    // VAD Speech Start -> User started speaking -> Interrupt immediately!
    this.vad.on('speechStart', () => {
      log.debug('VAD Speech start detected, interrupting agent');
      this.interrupt();
      this.sendState('listening');
    });

    // VAD Speech End -> User finished speaking -> Transcribe and answer
    this.vad.on('speechEnd', async (segment) => {
      log.debug(`VAD Speech end detected: ${segment.durationMs}ms`);
      this.sendState('thinking');
      await this.handleUserAudioSegment(segment.audio);
    });

    // Sentence Buffer -> dispatch text to TTS as they are ready
    this.sentenceBuffer.on('sentence', async (text, index) => {
      if (this.shouldInterrupt) return;
      log.info({ index, text }, 'WebUI Sentence boundary detected, running TTS...');

      try {
        const ttsResult = await this.ttsClient.synthesize(text);
        if (this.shouldInterrupt) return;

        if (ttsResult.wav && this.ws.readyState === WebSocket.OPEN) {
          log.info({ size: ttsResult.wav.length }, 'Sending voice audio chunk to WebUI');
          this.ws.send(JSON.stringify({
            type: 'voice_audio_chunk',
            audio: ttsResult.wav.toString('base64'),
            text,
          }));
        }
      } catch (err: any) {
        log.error({ error: err.message }, 'Failed to synthesize text for WebUI');
      }
    });
  }

  private sendState(state: 'idle' | 'listening' | 'thinking' | 'speaking'): void {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'voice_state', state }));
    }
  }

  private interrupt(): void {
    log.info('Interrupting WebUI voice session response');
    this.shouldInterrupt = true;

    if (this.currentLlmAbortController) {
      this.currentLlmAbortController.abort();
      this.currentLlmAbortController = null;
    }

    this.sentenceBuffer.reset();
    this.isProcessingLlm = false;

    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'voice_interrupt' }));
    }
    this.sendState('listening');
  }

  /**
   * Handle incoming raw 16kHz mono PCM chunks from browser.
   */
  handleAudioChunk(pcm: Buffer): void {
    if (this.isProcessingLlm) {
      this.interrupt();
    }

    if (this.mode === 'vad') {
      this.vad.processChunk(pcm);
    } else {
      this.accumulatedPttAudio.push(pcm);
    }
  }

  async stopPushToTalk(): Promise<void> {
    if (this.mode !== 'push-to-talk') return;
    this.sendState('thinking');
    const audio = Buffer.concat(this.accumulatedPttAudio);
    this.accumulatedPttAudio = [];
    await this.handleUserAudioSegment(audio);
  }

  private async handleUserAudioSegment(speechPcm: Buffer): Promise<void> {
    const asrResult = await this.asrClient.transcribe(speechPcm);
    const text = asrResult.text.trim();

    if (!text) {
      log.debug('ASR returned empty transcript for WebUI. Ignoring.');
      this.sendState('idle');
      return;
    }

    log.info({ transcript: text }, 'WebUI Speech transcribed');

    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'voice_transcription',
        role: 'user',
        content: text,
      }));
    }

    // Add user message to history
    this.voiceHistory.push({ role: 'user', content: text });
    if (this.voiceHistory.length > 20) {
      this.voiceHistory.shift();
    }

    // Now run LLM response
    await this.generateAgentResponse();
  }

  private async generateAgentResponse(): Promise<void> {
    this.shouldInterrupt = false;
    this.isProcessingLlm = true;
    this.sendState('thinking');

    this.currentLlmAbortController = new AbortController();

    const config = getConfig();
    const agentName = config.agent?.name || 'LiteClaw';
    const maxResponseTokens = config.voice?.maxResponseTokens || 150;

    const voiceSystemPrompt = `You are ${agentName}, a conversational voice assistant.
Having a real-time voice chat. Keep your response extremely brief, conversational, and simple (1-2 sentences max).
Never use list points, markdown, URLs, or long lists. Speak naturally, directly answering the user.`;

    try {
      const messages: LLMMessage[] = [
        { role: 'system', content: voiceSystemPrompt },
        ...this.voiceHistory,
      ];

      const generator = this.engine.getLLMClient().streamChat(
        messages,
        [],
        {
          maxTokens: maxResponseTokens,
          disableReasoning: true,
          signal: this.currentLlmAbortController.signal,
        }
      );

      this.sentenceBuffer.reset();
      let fullResponse = '';
      let startedSpeaking = false;

      for await (const chunk of generator) {
        if (this.shouldInterrupt) {
          log.info('WebUI LLM response stream interrupted');
          break;
        }

        if (chunk.type === 'content' && chunk.content) {
          fullResponse += chunk.content;
          this.sentenceBuffer.addToken(chunk.content);
          
          if (!startedSpeaking) {
            startedSpeaking = true;
            this.sendState('speaking');
          }
        }
      }

      this.sentenceBuffer.finish();
      this.isProcessingLlm = false;

      if (!this.shouldInterrupt && fullResponse.trim()) {
        log.info({ fullResponse }, 'WebUI LLM response finished');
        this.voiceHistory.push({ role: 'assistant', content: fullResponse.trim() });
        if (this.voiceHistory.length > 20) {
          this.voiceHistory.shift();
        }

        if (this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({
            type: 'voice_transcription',
            role: 'assistant',
            content: fullResponse.trim(),
          }));
        }
      }

      // If we finished speaking and didn't get interrupted, change state to idle
      if (!this.shouldInterrupt) {
        this.sendState('idle');
      }

    } catch (err: any) {
      if (err.name === 'AbortError') {
        log.debug('WebUI LLM request aborted');
      } else {
        log.error({ error: err.message }, 'Failed to generate WebUI agent response');
      }
      this.isProcessingLlm = false;
      this.sendState('idle');
    }
  }

  close(): void {
    log.info('Closing WebUI voice session');
    this.interrupt();
    this.vad.removeAllListeners();
    this.sentenceBuffer.removeAllListeners();
  }
}
