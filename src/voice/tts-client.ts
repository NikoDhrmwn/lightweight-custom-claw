/**
 * LiteClaw — OmniVoice TTS Client
 *
 * Communicates with the local OmniVoice.cpp / OpenAI-compatible TTS server.
 * Sends text sentences and converts the resulting 24kHz WAV to Discord-compatible 48kHz stereo PCM.
 */

import { getConfig } from '../config.js';
import { createLogger } from '../logger.js';
import { extractPCMFromWav, ttsToDiscord } from './audio-utils.js';

const log = createLogger('voice:tts');

export interface TTSResult {
  pcm: Buffer;        // 48kHz stereo PCM
  durationMs: number;
  wav?: Buffer;       // Raw WAV buffer (e.g. for WebUI playback)
}

export class TTSClient {
  private serverUrl: string;
  private voiceRef: string;
  private modelName: string;

  constructor() {
    const config = getConfig();
    const voiceConfig = (config as any).voice || {};
    const ttsConfig = voiceConfig.tts || {};
    this.serverUrl = ttsConfig.serverUrl || 'http://localhost:8090/v1/audio/speech';
    this.voiceRef = ttsConfig.voiceRef || 'auto';
    this.modelName = ttsConfig.modelName || 'omnivoice';
  }

  /**
   * Synthesize text to 48kHz stereo PCM for Discord playback.
   */
  async synthesize(text: string): Promise<TTSResult> {
    const startTime = Date.now();
    if (!text || !text.trim()) {
      return { pcm: Buffer.alloc(0), durationMs: 0 };
    }

    try {
      log.debug({ text }, 'Requesting TTS synthesis...');

      const response = await fetch(this.serverUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer none', // OpenAI compatibility expects header
        },
        body: JSON.stringify({
          model: this.modelName,
          input: text,
          voice: this.voiceRef,
          response_format: 'wav',
        }),
      });

      if (!response.ok) {
        throw new Error(`TTS server returned status ${response.status}: ${response.statusText}`);
      }

      // Read response as binary buffer
      const arrayBuffer = await response.arrayBuffer();
      const wavBuffer = Buffer.from(arrayBuffer);

      if (wavBuffer.length === 0) {
        throw new Error('TTS server returned empty audio buffer');
      }

      // Extract raw 24kHz mono PCM from WAV
      const rawPcm = extractPCMFromWav(wavBuffer);

      // Resample 24kHz mono to 48kHz stereo
      const discordPcm = ttsToDiscord(rawPcm);

      const duration = Date.now() - startTime;
      log.info(
        {
          textLength: text.length,
          wavSize: wavBuffer.length,
          pcmSize: discordPcm.length,
          totalMs: duration,
        },
        'TTS synthesis completed'
      );

      return {
        pcm: discordPcm,
        durationMs: duration,
        wav: wavBuffer,
      };
    } catch (err: any) {
      log.error({ error: err.message, url: this.serverUrl }, 'TTS synthesis failed');
      return { pcm: Buffer.alloc(0), durationMs: Date.now() - startTime };
    }
  }
}
