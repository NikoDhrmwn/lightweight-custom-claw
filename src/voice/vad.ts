/**
 * LiteClaw — Voice Activity Detection (VAD)
 *
 * Energy-based VAD that detects when a user starts and stops speaking.
 * Emits speech segments as complete PCM buffers for downstream ASR.
 *
 * Design:
 * - Uses RMS energy threshold to detect speech vs. silence
 * - Requires `minSpeechMs` of continuous speech before triggering
 * - Waits for `silenceDurationMs` of silence before finalizing an utterance
 * - Outputs trimmed speech segments ready for ASR processing
 */

import { EventEmitter } from 'events';
import { calculateRMSEnergy, pcmDurationMs } from './audio-utils.js';
import { createLogger } from '../logger.js';

const log = createLogger('voice:vad');

// ─── Types ────────────────────────────────────────────────────────────

export interface VADOptions {
  /** RMS energy threshold to consider as speech (0-1). Default: 0.01 */
  energyThreshold?: number;
  /** Milliseconds of silence to wait before ending an utterance. Default: 300 */
  silenceDurationMs?: number;
  /** Minimum speech duration to trigger (avoids noise blips). Default: 100 */
  minSpeechMs?: number;
  /** Maximum utterance duration before forced cut. Default: 30000 (30s) */
  maxUtteranceMs?: number;
  /** Sample rate of input audio. Default: 16000 */
  sampleRate?: number;
  /** Number of channels. Default: 1 */
  channels?: number;
}

export interface SpeechSegment {
  /** The raw PCM audio data of the speech segment */
  audio: Buffer;
  /** Duration in milliseconds */
  durationMs: number;
  /** Timestamp when speech started */
  startedAt: number;
  /** Timestamp when speech ended */
  endedAt: number;
}

// ─── VAD Event Types ──────────────────────────────────────────────────

export interface VADEvents {
  /** Emitted when speech is detected (user starts talking) */
  speechStart: [];
  /** Emitted when a complete speech segment is ready */
  speechEnd: [segment: SpeechSegment];
  /** Emitted periodically with current energy level for monitoring */
  energy: [rms: number, isSpeech: boolean];
}

// ─── VAD Implementation ──────────────────────────────────────────────

export class VoiceActivityDetector extends EventEmitter {
  private energyThreshold: number;
  private silenceDurationMs: number;
  private minSpeechMs: number;
  private maxUtteranceMs: number;
  private sampleRate: number;
  private channels: number;

  /** Whether we are currently in a "speaking" state */
  private isSpeaking = false;
  /** Accumulated speech audio chunks */
  private speechChunks: Buffer[] = [];
  /** Total duration of accumulated speech in ms */
  private speechDurationMs = 0;
  /** Timestamp when current speech started */
  private speechStartTime = 0;
  /** Duration of continuous silence in ms */
  private silenceDurationAccum = 0;

  constructor(options: VADOptions = {}) {
    super();
    this.energyThreshold = options.energyThreshold ?? 0.01;
    this.silenceDurationMs = options.silenceDurationMs ?? 300;
    this.minSpeechMs = options.minSpeechMs ?? 100;
    this.maxUtteranceMs = options.maxUtteranceMs ?? 30000;
    this.sampleRate = options.sampleRate ?? 16000;
    this.channels = options.channels ?? 1;
  }

  /**
   * Feed a chunk of PCM audio into the VAD.
   * Call this repeatedly with incoming audio data.
   */
  processChunk(pcm: Buffer): void {
    const energy = calculateRMSEnergy(pcm);
    const chunkDurationMs = pcmDurationMs(pcm, this.sampleRate, this.channels);
    const isSpeech = energy > this.energyThreshold;

    this.emit('energy', energy, isSpeech);

    if (isSpeech) {
      this.silenceDurationAccum = 0;

      if (!this.isSpeaking) {
        // Speech just started
        this.isSpeaking = true;
        this.speechChunks = [];
        this.speechDurationMs = 0;
        this.speechStartTime = Date.now();
        log.debug({ energy: energy.toFixed(4) }, 'Speech started');
        this.emit('speechStart');
      }

      this.speechChunks.push(pcm);
      this.speechDurationMs += chunkDurationMs;

      // Check max utterance length
      if (this.speechDurationMs >= this.maxUtteranceMs) {
        log.debug({ durationMs: this.speechDurationMs }, 'Max utterance duration reached, forcing end');
        this.finalizeSpeech();
      }
    } else {
      // Silence
      if (this.isSpeaking) {
        this.silenceDurationAccum += chunkDurationMs;
        // Still include the silence chunk in the buffer (natural trailing silence)
        this.speechChunks.push(pcm);
        this.speechDurationMs += chunkDurationMs;

        if (this.silenceDurationAccum >= this.silenceDurationMs) {
          // Enough silence to consider the utterance complete
          this.finalizeSpeech();
        }
      }
    }
  }

  /**
   * Finalize the current speech segment and emit it.
   */
  private finalizeSpeech(): void {
    if (!this.isSpeaking) return;

    this.isSpeaking = false;
    const now = Date.now();

    // Check minimum speech duration
    if (this.speechDurationMs < this.minSpeechMs) {
      log.debug({ durationMs: this.speechDurationMs }, 'Speech too short, discarding');
      this.speechChunks = [];
      this.speechDurationMs = 0;
      this.silenceDurationAccum = 0;
      return;
    }

    const audio = Buffer.concat(this.speechChunks);
    const segment: SpeechSegment = {
      audio,
      durationMs: this.speechDurationMs,
      startedAt: this.speechStartTime,
      endedAt: now,
    };

    log.debug({
      durationMs: Math.round(this.speechDurationMs),
      audioBytes: audio.length,
    }, 'Speech segment complete');

    this.speechChunks = [];
    this.speechDurationMs = 0;
    this.silenceDurationAccum = 0;

    this.emit('speechEnd', segment);
  }

  /**
   * Force-end any in-progress speech (e.g., on disconnect).
   */
  flush(): void {
    if (this.isSpeaking && this.speechChunks.length > 0) {
      this.finalizeSpeech();
    }
  }

  /**
   * Reset all internal state.
   */
  reset(): void {
    this.isSpeaking = false;
    this.speechChunks = [];
    this.speechDurationMs = 0;
    this.silenceDurationAccum = 0;
    this.speechStartTime = 0;
  }

  /**
   * Update VAD options at runtime.
   */
  updateOptions(options: Partial<VADOptions>): void {
    if (options.energyThreshold !== undefined) this.energyThreshold = options.energyThreshold;
    if (options.silenceDurationMs !== undefined) this.silenceDurationMs = options.silenceDurationMs;
    if (options.minSpeechMs !== undefined) this.minSpeechMs = options.minSpeechMs;
    if (options.maxUtteranceMs !== undefined) this.maxUtteranceMs = options.maxUtteranceMs;
  }
}
