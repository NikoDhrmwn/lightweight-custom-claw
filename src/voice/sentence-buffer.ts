/**
 * LiteClaw — Sentence Buffer
 *
 * Buffers streaming LLM tokens and emits complete sentences.
 * This is the key component that enables overlapping LLM + TTS:
 * as soon as a sentence boundary is detected, it's dispatched to TTS
 * while the LLM continues generating.
 */

import { EventEmitter } from 'events';
import { createLogger } from '../logger.js';

const log = createLogger('voice:sentence-buffer');

// ─── Types ────────────────────────────────────────────────────────────

export interface SentenceBufferOptions {
  /** Maximum characters before forcing a flush (even without a boundary). Default: 200 */
  maxChars?: number;
  /** Minimum characters for a sentence to be emitted. Default: 10 */
  minChars?: number;
  /** Whether to strip markdown formatting from output. Default: true */
  stripMarkdown?: boolean;
}

export interface SentenceBufferEvents {
  /** Emitted when a complete sentence is ready for TTS */
  sentence: [text: string, index: number];
  /** Emitted when the buffer is flushed (end of generation) */
  done: [remaining: string | null];
}

// ─── Sentence Boundary Detection ──────────────────────────────────────

/**
 * Characters that mark the end of a sentence in conversational speech.
 */
const SENTENCE_TERMINATORS = new Set(['.', '!', '?', '…']);

/**
 * Patterns that should NOT be treated as sentence boundaries.
 * E.g., "Dr.", "Mr.", "e.g.", "3.14", "file.txt"
 */
const FALSE_BOUNDARY_PATTERNS = [
  /\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|a\.m|p\.m)\.\s*$/i,
  /\d\.\s*$/,                // "3." — number followed by period
  /\.\w+\s*$/,               // ".txt", ".js" — file extensions
  /(?:https?|ftp):\/\/\S*$/, // URLs
];

/**
 * Check if the buffer ends at a genuine sentence boundary.
 */
function isSentenceBoundary(buffer: string): boolean {
  const trimmed = buffer.trimEnd();
  if (trimmed.length === 0) return false;

  const lastChar = trimmed[trimmed.length - 1];
  if (!SENTENCE_TERMINATORS.has(lastChar)) return false;

  // Check for false positives
  for (const pattern of FALSE_BOUNDARY_PATTERNS) {
    if (pattern.test(trimmed)) return false;
  }

  return true;
}

/**
 * Strip common markdown formatting for cleaner TTS output.
 */
function stripMarkdown(text: string): string {
  return text
    // Remove bold/italic markers
    .replace(/\*{1,3}([^*]+)\*{1,3}/g, '$1')
    .replace(/_{1,3}([^_]+)_{1,3}/g, '$1')
    // Remove inline code
    .replace(/`([^`]+)`/g, '$1')
    // Remove links [text](url) → text
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    // Remove headers
    .replace(/^#{1,6}\s+/gm, '')
    // Remove bullet points
    .replace(/^[\s]*[-*+]\s+/gm, '')
    // Remove numbered lists
    .replace(/^[\s]*\d+\.\s+/gm, '')
    // Collapse multiple spaces
    .replace(/\s+/g, ' ')
    .trim();
}

// ─── Sentence Buffer ──────────────────────────────────────────────────

export class SentenceBuffer extends EventEmitter {
  private buffer = '';
  private sentenceIndex = 0;
  private maxChars: number;
  private minChars: number;
  private shouldStripMarkdown: boolean;
  private isFinished = false;

  constructor(options: SentenceBufferOptions = {}) {
    super();
    this.maxChars = options.maxChars ?? 200;
    this.minChars = options.minChars ?? 10;
    this.shouldStripMarkdown = options.stripMarkdown ?? true;
  }

  /**
   * Feed a token (or chunk of tokens) from the LLM stream.
   */
  addToken(token: string): void {
    if (this.isFinished) return;

    this.buffer += token;

    // Check if we have a sentence boundary
    if (isSentenceBoundary(this.buffer) && this.buffer.trim().length >= this.minChars) {
      this.emitSentence();
      return;
    }

    // Force flush if buffer is getting too long (safety valve)
    if (this.buffer.trim().length >= this.maxChars) {
      // Try to find the last natural break point (comma, semicolon, etc.)
      const breakPoints = [', ', '; ', ' — ', ' - ', ': '];
      let lastBreak = -1;
      for (const bp of breakPoints) {
        const idx = this.buffer.lastIndexOf(bp);
        if (idx > this.minChars) {
          lastBreak = idx + bp.length;
        }
      }

      if (lastBreak > 0) {
        const sentence = this.buffer.slice(0, lastBreak);
        this.buffer = this.buffer.slice(lastBreak);
        this.emitProcessed(sentence);
      } else {
        // No good break point — flush the whole thing
        this.emitSentence();
      }
    }
  }

  /**
   * Signal that the LLM has finished generating.
   * Flushes any remaining buffered text.
   */
  finish(): void {
    if (this.isFinished) return;
    this.isFinished = true;

    const remaining = this.buffer.trim();
    if (remaining.length > 0) {
      this.emitProcessed(remaining);
    }

    this.emit('done', remaining.length > 0 ? remaining : null);
    this.buffer = '';
  }

  /**
   * Reset the buffer for a new generation.
   */
  reset(): void {
    this.buffer = '';
    this.sentenceIndex = 0;
    this.isFinished = false;
  }

  private emitSentence(): void {
    const text = this.buffer.trim();
    this.buffer = '';
    if (text.length > 0) {
      this.emitProcessed(text);
    }
  }

  private emitProcessed(text: string): void {
    let processed = text.trim();
    if (this.shouldStripMarkdown) {
      processed = stripMarkdown(processed);
    }
    if (processed.length > 0) {
      this.emit('sentence', processed, this.sentenceIndex);
      this.sentenceIndex++;
      log.debug({
        index: this.sentenceIndex - 1,
        chars: processed.length,
        preview: processed.slice(0, 60),
      }, 'Sentence emitted');
    }
  }

  /** Get the number of sentences emitted so far. */
  get count(): number {
    return this.sentenceIndex;
  }
}
