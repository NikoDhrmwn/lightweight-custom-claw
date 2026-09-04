/**
 * LiteClaw — Nemotron ASR Client
 *
 * Communicates with the local Python ASR server.
 * Resamples audio to 16kHz mono, wraps it in a WAV container,
 * and sends it to the ASR API for rapid transcription.
 */

import { getConfig } from '../config.js';
import { createLogger } from '../logger.js';
import { buildWav, ASR_SAMPLE_RATE } from './audio-utils.js';

const log = createLogger('voice:asr');

export interface ASRResult {
  text: string;
  language: string;
  durationMs: number;
}

export class ASRClient {
  private serverUrl: string;

  constructor() {
    const config = getConfig();
    const voiceConfig = (config as any).voice || {};
    const asrConfig = voiceConfig.asr || {};
    this.serverUrl = asrConfig.serverUrl || 'http://localhost:8089/transcribe';
  }

  /**
   * Transcribe a 16kHz mono 16-bit LE PCM audio buffer.
   */
  async transcribe(pcmBuffer: Buffer, language: string = 'en'): Promise<ASRResult> {
    if (pcmBuffer.length === 0) {
      return { text: '', language, durationMs: 0 };
    }

    const startTime = Date.now();
    try {
      // Build WAV file in memory
      const wavBuffer = buildWav(pcmBuffer, ASR_SAMPLE_RATE, 1, 16);

      log.debug({ size: wavBuffer.length }, 'Sending audio to ASR server...');

      // Call local ASR server
      const response = await fetch(this.serverUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'audio/wav',
          'X-Language': language,
        },
        body: wavBuffer,
      });

      if (!response.ok) {
        throw new Error(`ASR server returned status ${response.status}: ${response.statusText}`);
      }

      const result = await response.json() as any;
      const duration = Date.now() - startTime;

      log.info(
        {
          text: result.text,
          lang: result.language,
          serverMs: result.duration_ms || 0,
          totalMs: duration,
        },
        'ASR transcription completed'
      );

      return {
        text: (result.text || '').trim(),
        language: result.language || language,
        durationMs: duration,
      };
    } catch (err: any) {
      log.error({ error: err.message, url: this.serverUrl }, 'ASR transcription failed');
      return { text: '', language, durationMs: Date.now() - startTime };
    }
  }
}
