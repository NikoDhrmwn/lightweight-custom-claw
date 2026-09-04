/**
 * LiteClaw — Voice Audio Utilities
 *
 * PCM resampling, ring buffers, WAV header parsing,
 * and format conversion for the voice pipeline.
 *
 * Discord sends/receives: 48kHz, stereo, 16-bit LE PCM
 * Nemotron ASR expects:   16kHz, mono, 16-bit LE PCM
 * OmniVoice TTS outputs:  24kHz, mono, 16-bit LE PCM (in WAV container)
 */

import { createLogger } from '../logger.js';

const log = createLogger('voice:audio');

// ─── Constants ────────────────────────────────────────────────────────

/** Discord audio format */
export const DISCORD_SAMPLE_RATE = 48000;
export const DISCORD_CHANNELS = 2;
export const DISCORD_BIT_DEPTH = 16;
export const DISCORD_FRAME_SIZE_MS = 20; // 20ms Opus frames
export const DISCORD_FRAME_SAMPLES = (DISCORD_SAMPLE_RATE * DISCORD_FRAME_SIZE_MS) / 1000;

/** ASR (Nemotron) expects 16kHz mono */
export const ASR_SAMPLE_RATE = 16000;
export const ASR_CHANNELS = 1;

/** TTS (OmniVoice) outputs 24kHz mono */
export const TTS_SAMPLE_RATE = 24000;
export const TTS_CHANNELS = 1;

// ─── PCM Resampling (simple linear interpolation) ─────────────────────

/**
 * Resample 16-bit LE PCM audio between sample rates.
 * Uses linear interpolation — acceptable quality for voice.
 */
export function resamplePCM(
  input: Buffer,
  fromRate: number,
  toRate: number,
  fromChannels: number = 1,
  toChannels: number = 1,
): Buffer {
  if (fromRate === toRate && fromChannels === toChannels) {
    return Buffer.from(input);
  }

  const bytesPerSample = 2; // 16-bit
  const inputSamples = input.length / (bytesPerSample * fromChannels);
  const ratio = toRate / fromRate;
  const outputSamples = Math.floor(inputSamples * ratio);
  const output = Buffer.alloc(outputSamples * bytesPerSample * toChannels);

  for (let i = 0; i < outputSamples; i++) {
    const srcPos = i / ratio;
    const srcIndex = Math.floor(srcPos);
    const frac = srcPos - srcIndex;

    // Read source sample (mix to mono if needed)
    const readSample = (idx: number): number => {
      if (idx >= inputSamples) idx = inputSamples - 1;
      if (idx < 0) idx = 0;
      if (fromChannels === 1) {
        return input.readInt16LE(idx * bytesPerSample);
      }
      // Mix stereo to mono: average left and right
      let sum = 0;
      for (let ch = 0; ch < fromChannels; ch++) {
        sum += input.readInt16LE((idx * fromChannels + ch) * bytesPerSample);
      }
      return Math.round(sum / fromChannels);
    };

    // Linear interpolation
    const s0 = readSample(srcIndex);
    const s1 = readSample(srcIndex + 1);
    const interpolated = Math.round(s0 + (s1 - s0) * frac);
    const clamped = Math.max(-32768, Math.min(32767, interpolated));

    // Write to output (duplicate to stereo if needed)
    for (let ch = 0; ch < toChannels; ch++) {
      output.writeInt16LE(clamped, (i * toChannels + ch) * bytesPerSample);
    }
  }

  return output;
}

/**
 * Shortcut: Discord 48kHz stereo → ASR 16kHz mono
 */
export function discordToASR(pcm: Buffer): Buffer {
  return resamplePCM(pcm, DISCORD_SAMPLE_RATE, ASR_SAMPLE_RATE, DISCORD_CHANNELS, ASR_CHANNELS);
}

/**
 * Shortcut: TTS 24kHz mono → Discord 48kHz stereo
 */
export function ttsToDiscord(pcm: Buffer): Buffer {
  return resamplePCM(pcm, TTS_SAMPLE_RATE, DISCORD_SAMPLE_RATE, TTS_CHANNELS, DISCORD_CHANNELS);
}

// ─── WAV Parsing ──────────────────────────────────────────────────────

export interface WavHeader {
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
  dataOffset: number;
  dataSize: number;
}

/**
 * Parse a WAV file header and return metadata + data offset.
 */
export function parseWavHeader(buffer: Buffer): WavHeader {
  if (buffer.length < 44) {
    throw new Error('Buffer too small to be a WAV file');
  }

  const riff = buffer.toString('ascii', 0, 4);
  if (riff !== 'RIFF') {
    throw new Error(`Invalid WAV: expected RIFF header, got "${riff}"`);
  }

  const wave = buffer.toString('ascii', 8, 12);
  if (wave !== 'WAVE') {
    throw new Error(`Invalid WAV: expected WAVE format, got "${wave}"`);
  }

  // Find fmt chunk
  let offset = 12;
  let channels = 1;
  let sampleRate = 24000;
  let bitsPerSample = 16;

  while (offset < buffer.length - 8) {
    const chunkId = buffer.toString('ascii', offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);

    if (chunkId === 'fmt ') {
      channels = buffer.readUInt16LE(offset + 10);
      sampleRate = buffer.readUInt32LE(offset + 12);
      bitsPerSample = buffer.readUInt16LE(offset + 22);
    }

    if (chunkId === 'data') {
      return {
        sampleRate,
        channels,
        bitsPerSample,
        dataOffset: offset + 8,
        dataSize: chunkSize,
      };
    }

    offset += 8 + chunkSize;
    // Align to even boundary
    if (chunkSize % 2 !== 0) offset++;
  }

  throw new Error('No data chunk found in WAV');
}

/**
 * Extract raw PCM data from a WAV buffer.
 */
export function extractPCMFromWav(wavBuffer: Buffer): Buffer {
  const header = parseWavHeader(wavBuffer);
  return wavBuffer.subarray(header.dataOffset, header.dataOffset + header.dataSize);
}

/**
 * Build a WAV file from raw PCM data.
 */
export function buildWav(pcm: Buffer, sampleRate: number, channels: number = 1, bitsPerSample: number = 16): Buffer {
  const byteRate = sampleRate * channels * (bitsPerSample / 8);
  const blockAlign = channels * (bitsPerSample / 8);
  const dataSize = pcm.length;
  const fileSize = 36 + dataSize;

  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(fileSize, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16);         // fmt chunk size
  header.writeUInt16LE(1, 20);          // PCM format
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(bitsPerSample, 34);
  header.write('data', 36);
  header.writeUInt32LE(dataSize, 40);

  return Buffer.concat([header, pcm]);
}

// ─── Ring Buffer ──────────────────────────────────────────────────────

/**
 * A fixed-size ring buffer for accumulating audio samples.
 * Used to buffer incoming Discord audio before sending to ASR.
 */
export class AudioRingBuffer {
  private buffer: Buffer;
  private writePos = 0;
  private readPos = 0;
  private _available = 0;

  constructor(private maxBytes: number) {
    this.buffer = Buffer.alloc(maxBytes);
  }

  /** Number of bytes available to read. */
  get available(): number {
    return this._available;
  }

  /** Write data into the ring buffer. Overwrites oldest data if full. */
  write(data: Buffer): void {
    for (let i = 0; i < data.length; i++) {
      this.buffer[this.writePos] = data[i];
      this.writePos = (this.writePos + 1) % this.maxBytes;
      if (this._available < this.maxBytes) {
        this._available++;
      } else {
        // Overwriting old data — advance read pointer
        this.readPos = (this.readPos + 1) % this.maxBytes;
      }
    }
  }

  /** Read up to `count` bytes from the buffer. */
  read(count: number): Buffer {
    const toRead = Math.min(count, this._available);
    const out = Buffer.alloc(toRead);
    for (let i = 0; i < toRead; i++) {
      out[i] = this.buffer[this.readPos];
      this.readPos = (this.readPos + 1) % this.maxBytes;
    }
    this._available -= toRead;
    return out;
  }

  /** Peek at `count` bytes without consuming them. */
  peek(count: number): Buffer {
    const toRead = Math.min(count, this._available);
    const out = Buffer.alloc(toRead);
    let pos = this.readPos;
    for (let i = 0; i < toRead; i++) {
      out[i] = this.buffer[pos];
      pos = (pos + 1) % this.maxBytes;
    }
    return out;
  }

  /** Clear all buffered data. */
  clear(): void {
    this.writePos = 0;
    this.readPos = 0;
    this._available = 0;
  }
}

// ─── Energy Calculation ───────────────────────────────────────────────

/**
 * Calculate RMS energy of a 16-bit PCM buffer.
 * Returns a value between 0 and 1.
 */
export function calculateRMSEnergy(pcm: Buffer): number {
  const samples = pcm.length / 2;
  if (samples === 0) return 0;

  let sumSquares = 0;
  for (let i = 0; i < pcm.length; i += 2) {
    const sample = pcm.readInt16LE(i) / 32768;
    sumSquares += sample * sample;
  }

  return Math.sqrt(sumSquares / samples);
}

/**
 * Calculate the duration of a PCM buffer in milliseconds.
 */
export function pcmDurationMs(pcm: Buffer, sampleRate: number, channels: number = 1): number {
  const bytesPerSample = 2; // 16-bit
  const samples = pcm.length / (bytesPerSample * channels);
  return (samples / sampleRate) * 1000;
}
