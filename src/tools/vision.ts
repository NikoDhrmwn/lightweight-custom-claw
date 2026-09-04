/**
 * LiteClaw — Vision & Image Inspection Tools
 *
 * Allows the agent to inspect photos, screenshots, diagrams, receipts,
 * and stickers locally using Microsoft Florence-2-large.
 * Also provides image preprocessing (resize/compress) for multimodal pipelines.
 */

import { existsSync } from 'fs';
import { toolRegistry, ToolResult } from '../core/tools.js';
import { resolveFlexiblePath } from '../core/workspace.js';
import { visionService } from '../core/vision.js';
import { getConfig } from '../config.js';
import { createLogger } from '../logger.js';

const log = createLogger('vision');

/**
 * Preprocess an image for the LLM.
 * - Resize to maxDimensionPx (default: 1024)
 * - Convert to JPEG for consistent base64
 * - Returns data URI string
 */
export async function preprocessImage(
  input: Buffer | string,
  maxDimension?: number
): Promise<string> {
  const config = getConfig();
  const maxPx = maxDimension ?? config.tools?.vision?.maxDimensionPx ?? 1024;

  try {
    const sharp = await import('sharp').then(m => m.default).catch(() => null);

    if (sharp && Buffer.isBuffer(input)) {
      const meta = await sharp(input).metadata();
      const width = meta.width ?? 0;
      const height = meta.height ?? 0;

      let processed = sharp(input);

      if (width > maxPx || height > maxPx) {
        processed = processed.resize(maxPx, maxPx, { fit: 'inside', withoutEnlargement: true });
      }

      const buffer = await processed.jpeg({ quality: 85 }).toBuffer();
      return `data:image/jpeg;base64,${buffer.toString('base64')}`;
    }
  } catch (err: any) {
    log.warn({ error: err.message }, 'Sharp processing failed, using raw image');
  }

  if (typeof input === 'string') {
    if (input.startsWith('data:')) return input;
    return `data:image/jpeg;base64,${input}`;
  }

  return `data:image/jpeg;base64,${input.toString('base64')}`;
}

/**
 * Check if a message contains image content.
 */
export function hasImageContent(content: any): boolean {
  if (Array.isArray(content)) {
    return content.some(part => part.type === 'image_url');
  }
  return false;
}

// ─── inspect_image Tool ──────────────────────────────────────────────

toolRegistry.register({
  name: 'inspect_image',
  description: 'Inspect, describe, and transcribe (OCR) any image, photo, receipt, diagram, sticker, or screenshot using local Florence-2 Large.',
  category: 'vision',
  parameters: [
    {
      name: 'path',
      type: 'string',
      description: 'Path to the image file (supports "downloads/receipt.jpg", "C:\\...", or relative paths).',
      required: true,
    },
  ],
  usageNotes: [
    'Use this whenever the user asks you to check an image, read a receipt, inspect a photo, or transcribe text from an image on disk.',
    'Supports JPEG, PNG, WebP, BMP, and GIF.'
  ],
  examples: [
    { userIntent: 'inspect receipt in downloads', arguments: { path: 'downloads/receipt.jpg' } },
    { userIntent: 'read text in screenshot', arguments: { path: 'screenshot.png' } },
  ],
  keywords: ['image', 'picture', 'photo', 'receipt', 'screenshot', 'diagram', 'sticker', 'inspect image', 'read image', 'ocr image', 'transcribe image'],
  handler: async (args, context): Promise<ToolResult> => {
    const rawPath = args.path as string | undefined;
    if (!rawPath) {
      return { success: false, output: 'No image path specified.' };
    }

    let filePath: string;
    try {
      filePath = resolveFlexiblePath(rawPath, context.workingDir).absolute;
    } catch (err: any) {
      return { success: false, output: `Path resolution error: ${err.message}` };
    }

    if (!existsSync(filePath)) {
      return { success: false, output: `Image file not found: ${filePath}` };
    }

    try {
      const analysis = await visionService.analyzeImage(filePath);
      return {
        success: true,
        output: analysis.formattedContext,
        filePath,
      };
    } catch (err: any) {
      return {
        success: false,
        output: `Failed to inspect image with Florence-2: ${err.message}`,
      };
    }
  },
});
