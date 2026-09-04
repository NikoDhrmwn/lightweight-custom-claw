/**
 * LiteClaw — Local Vision & Motion Engine (Florence-2 Large)
 *
 * Provides on-device image, moving sticker, and animated GIF understanding
 * using Microsoft's Florence-2-large (770M) via Transformers.js & ONNX Runtime.
 *
 * For moving stickers (animated WebP) and animated GIFs, it extracts the representative
 * subject frame and frames it explicitly as a single continuous motion loop, preventing
 * the model or agent from hallucinating multiple separate subjects or a collage.
 *
 * Runs 100% locally on CPU/GPU without external API keys or cloud dependencies.
 */

import { existsSync, writeFileSync, unlinkSync, mkdirSync, readFileSync } from 'fs';
import { join } from 'path';
import { execSync } from 'child_process';
import { createLogger } from '../logger.js';
import { getStateDir } from '../config.js';

const log = createLogger('vision');

export interface VisionAnalysis {
  caption: string;
  ocrText?: string;
  isAnimated: boolean;
  formattedContext: string;
}

export class FlorenceVisionService {
  private static instance: FlorenceVisionService | null = null;
  private model: any = null;
  private processor: any = null;
  private tokenizer: any = null;
  private rawImageCls: any = null;
  private isInitializing = false;
  private initPromise: Promise<void> | null = null;

  public modelId = 'onnx-community/Florence-2-large';
  public dtype: 'q4' | 'q8' | 'fp32' = 'q4';

  private constructor() {}

  static getInstance(): FlorenceVisionService {
    if (!FlorenceVisionService.instance) {
      FlorenceVisionService.instance = new FlorenceVisionService();
    }
    return FlorenceVisionService.instance;
  }

  /**
   * Lazy initialization of the Florence-2 model.
   * Loads from local cache if previously downloaded.
   */
  async ensureInitialized(): Promise<void> {
    if (this.model && this.processor && this.tokenizer) return;

    if (this.isInitializing && this.initPromise) {
      return this.initPromise;
    }

    this.isInitializing = true;
    this.initPromise = (async () => {
      try {
        log.info({ modelId: this.modelId, dtype: this.dtype }, 'Loading Florence-2 Large vision model...');
        const startTime = Date.now();

        const {
          Florence2ForConditionalGeneration,
          AutoProcessor,
          AutoTokenizer,
          RawImage,
        } = await import('@huggingface/transformers');

        this.rawImageCls = RawImage;
        this.model = await Florence2ForConditionalGeneration.from_pretrained(this.modelId, {
          dtype: this.dtype,
        });
        this.processor = await AutoProcessor.from_pretrained(this.modelId);
        this.tokenizer = await AutoTokenizer.from_pretrained(this.modelId);

        log.info(
          { elapsedMs: Date.now() - startTime },
          'Florence-2 Large vision model loaded successfully'
        );
      } catch (err: any) {
        log.error({ error: err.message }, 'Failed to initialize Florence-2 model');
        throw err;
      } finally {
        this.isInitializing = false;
      }
    })();

    return this.initPromise;
  }

  /**
   * Detect if a buffer contains animated media (animated WebP, animated GIF, or MP4/video).
   */
  isAnimatedMedia(buffer: Buffer): boolean {
    if (!buffer || buffer.length < 16) return false;

    // 1. MP4 / MOV video (WhatsApp videoMessage / gifPlayback)
    if (buffer.length > 8 && buffer.slice(4, 8).toString('ascii') === 'ftyp') {
      return true;
    }

    // 2. Animated GIF (GIF87a / GIF89a)
    if (buffer.slice(0, 6).toString('ascii').startsWith('GIF')) {
      return true;
    }

    // 3. Animated WebP (Stickers): Contains 'RIFF' + 'WEBP' + 'ANIM' chunk header
    if (
      buffer.slice(0, 4).toString('ascii') === 'RIFF' &&
      buffer.slice(8, 12).toString('ascii') === 'WEBP' &&
      buffer.includes(Buffer.from('ANIM'))
    ) {
      return true;
    }

    // 4. WebM / Matroska video
    if (buffer.slice(0, 4).equals(Buffer.from([0x1a, 0x45, 0xdf, 0xa3]))) {
      return true;
    }

    return false;
  }

  /**
   * Convert image / sticker / video / GIF buffer into a RawImage suitable for Florence-2.
   * For animated stickers & GIFs, extracts the clear primary subject frame so Florence-2
   * perceives 1 single subject with full resolution instead of a confusing grid.
   */
  async prepareRawImage(imageSource: Buffer | string): Promise<{ rawImage: any; isAnimated: boolean }> {
    await this.ensureInitialized();

    let buffer: Buffer;
    if (typeof imageSource === 'string') {
      buffer = readFileSync(imageSource);
    } else {
      buffer = imageSource;
    }

    const animated = this.isAnimatedMedia(buffer);
    const tempDir = join(getStateDir(), 'temp');
    if (!existsSync(tempDir)) mkdirSync(tempDir, { recursive: true });

    if (animated) {
      // 1. For animated WebP stickers & GIFs: extract primary frame using Sharp
      try {
        const sharp = (await import('sharp')).default;
        const primaryBuf = await sharp(buffer, { page: 0 }).png().toBuffer();
        const rawImage = await this.rawImageCls.fromBlob(new Blob([new Uint8Array(primaryBuf)]));
        return { rawImage, isAnimated: true };
      } catch (e: any) {
        log.debug({ error: e.message }, 'Sharp primary frame extraction failed, trying ffmpeg');
      }

      // 2. For video files (MP4, MKV), extract frame using FFmpeg with proper .mp4 extension
      const tempIn = join(tempDir, `anim_in_${Date.now()}.mp4`);
      const tempOut = join(tempDir, `anim_frame_${Date.now()}.jpg`);
      writeFileSync(tempIn, buffer);
      try {
        execSync(`ffmpeg -y -i "${tempIn}" -vframes 1 "${tempOut}"`, { windowsHide: true, stdio: 'pipe' });
        const frameBuf = readFileSync(tempOut);
        const rawImage = await this.rawImageCls.fromBlob(new Blob([new Uint8Array(frameBuf)]));
        return { rawImage, isAnimated: true };
      } catch (e: any) {
        log.warn({ error: e.message }, 'FFmpeg frame extraction failed');
      } finally {
        if (existsSync(tempIn)) try { unlinkSync(tempIn); } catch {}
        if (existsSync(tempOut)) try { unlinkSync(tempOut); } catch {}
      }
    }

    // Static image or fallback
    try {
      const rawImage = await this.rawImageCls.fromBlob(new Blob([new Uint8Array(buffer)]));
      return { rawImage, isAnimated: false };
    } catch {
      // Fallback: convert static image to standard JPEG via ffmpeg
      const tempIn = join(tempDir, `static_in_${Date.now()}.jpg`);
      const tempOut = join(tempDir, `static_out_${Date.now()}.jpg`);
      writeFileSync(tempIn, buffer);
      try {
        execSync(`ffmpeg -y -i "${tempIn}" -vframes 1 "${tempOut}"`, { windowsHide: true, stdio: 'pipe' });
        const frameBuf = readFileSync(tempOut);
        const rawImage = await this.rawImageCls.fromBlob(new Blob([new Uint8Array(frameBuf)]));
        return { rawImage, isAnimated: false };
      } finally {
        if (existsSync(tempIn)) try { unlinkSync(tempIn); } catch {}
        if (existsSync(tempOut)) try { unlinkSync(tempOut); } catch {}
      }
    }
  }

  /**
   * Analyze an image, moving sticker, or animated GIF and return detailed description and OCR.
   */
  async analyzeImage(imageSource: Buffer | string): Promise<VisionAnalysis> {
    await this.ensureInitialized();
    const { rawImage, isAnimated } = await this.prepareRawImage(imageSource);

    log.info(
      { width: rawImage.width, height: rawImage.height, isAnimated },
      'Analyzing visual input with Florence-2 Large'
    );

    // 1. Run Detailed Caption
    const captionPrompt = '<MORE_DETAILED_CAPTION>';
    const capInputs = await this.processor(rawImage, captionPrompt);
    const capOutputs = await this.model.generate({
      ...capInputs,
      max_new_tokens: 512,
    });
    const capGenText = this.tokenizer.batch_decode(capOutputs, { skip_special_tokens: false })[0];
    const capParsed = this.processor.post_process_generation(capGenText, captionPrompt, [rawImage.height, rawImage.width]);
    const caption = capParsed['<MORE_DETAILED_CAPTION>'] || '(No description generated)';

    // 2. Run OCR Extraction
    const ocrPrompt = '<OCR>';
    const ocrInputs = await this.processor(rawImage, ocrPrompt);
    const ocrOutputs = await this.model.generate({
      ...ocrInputs,
      max_new_tokens: 1024,
    });
    const ocrGenText = this.tokenizer.batch_decode(ocrOutputs, { skip_special_tokens: false })[0];
    const ocrParsed = this.processor.post_process_generation(ocrGenText, ocrPrompt, [rawImage.height, rawImage.width]);
    const ocrText = (ocrParsed['<OCR>'] || '').trim();

    // Format visual summary
    const parts: string[] = [];
    if (isAnimated) {
      parts.push('🎬 [Received Moving Sticker / Animated GIF Reaction]');
      parts.push(`• Subject & Expression: ${caption}`);
      parts.push('• Animation: This is ONE continuous animation of this single subject playing in a moving loop.');
      if (ocrText && ocrText.length > 3) {
        parts.push(`• Text on sticker/animation: "${ocrText}"`);
      }
      parts.push('• IMPORTANT FOR AGENT: The user sent 1 animated reaction sticker or GIF of a single subject. Do NOT refer to it as multiple images or multiple subjects. React naturally to what the subject is expressing or doing (e.g. nodding, squinting, laughing, dancing, etc.)!');
    } else {
      parts.push('🖼️ [Visual Analysis — Florence-2 Large]');
      if (caption) {
        parts.push(`• Description: ${caption}`);
      }
      if (ocrText && ocrText.length > 5) {
        parts.push(`• Visible Text / OCR:\n${ocrText}`);
      }
    }

    const formattedContext = parts.join('\n');

    return {
      caption,
      ocrText: ocrText || undefined,
      isAnimated,
      formattedContext,
    };
  }
}

export const visionService = FlorenceVisionService.getInstance();
