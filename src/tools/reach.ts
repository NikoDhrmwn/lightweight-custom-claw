/**
 * LiteClaw — Agent Reach Tools
 *
 * Multi-platform internet capability suite adapted from Agent-Reach:
 * - reach_doctor: Environment health check across 15 platforms
 * - web_extract: Clean markdown reader powered by Jina Reader (r.jina.ai)
 * - reach_read: Multi-platform content reader (YouTube, V2EX, Reddit, GitHub, Bilibili)
 * - reach_search: Cross-platform search (YouTube, GitHub, V2EX, Web)
 * - reach_transcribe: Audio transcription using Whisper (Groq, OpenAI, local)
 */

import { execFile, execSync } from 'child_process';
import { promisify } from 'util';
import { existsSync, readFileSync, unlinkSync, readdirSync, writeFileSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { toolRegistry, ToolResult } from '../core/tools.js';
import { getConfig } from '../config.js';
import { createLogger } from '../logger.js';

const log = createLogger('reach');
const execFileAsync = promisify(execFile);

// Helper: check command availability
function isCommandAvailable(command: string): boolean {
  try {
    const isWindows = process.platform === 'win32';
    const checkCmd = isWindows ? `where.exe ${command}` : `which ${command}`;
    execSync(checkCmd, { stdio: 'ignore', timeout: 2000 });
    return true;
  } catch {
    return false;
  }
}

// Helper: parse VTT subtitles into clean text
function parseVttToText(vttContent: string): string {
  const lines = vttContent.split(/\r?\n/);
  const textLines: string[] = [];
  let lastLine = '';

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('WEBVTT') || trimmed.startsWith('NOTE') || /^\d+$/.test(trimmed)) {
      continue;
    }
    // Skip timestamp lines: 00:00:00.000 --> 00:00:05.000
    if (/^\d{2}:\d{2}/.test(trimmed) && trimmed.includes('-->')) {
      continue;
    }
    // Remove inline VTT tags like <c.color> or <00:00:00.000>
    const clean = trimmed.replace(/<[^>]+>/g, '').trim();
    if (clean && clean !== lastLine) {
      textLines.push(clean);
      lastLine = clean;
    }
  }

  return textLines.join('\n');
}

// ─── 1. reach_doctor ──────────────────────────────────────────────────────────

toolRegistry.register({
  name: 'reach_doctor',
  description: 'Check availability and active backends across 15 internet platforms (Twitter/X, Reddit, YouTube, Bilibili, GitHub, XiaoHongShu, LinkedIn, V2EX, Xueqiu, Facebook, Instagram, RSS, Exa, Whisper, Jina Reader).',
  category: 'web',
  parameters: [
    { name: 'detailed', type: 'boolean', description: 'Include verbose probe output and configuration tips' },
  ],
  usageNotes: [
    'Use this before executing multi-platform research to see which channels are active.',
    'Reports status as ok (available), warn (needs login/key), or missing (tool not installed).',
  ],
  keywords: ['doctor', 'agent reach', 'platforms', 'check', 'status', 'health', 'availability', 'inspect'],
  handler: async (args): Promise<ToolResult> => {
    const config = getConfig();
    const env = process.env;
    const detailed = Boolean(args.detailed);

    // Platform probes
    const platforms: Record<string, {
      name: string;
      category: string;
      tier: number; // 0=zero-config, 1=needs key/login, 2=complex setup
      status: 'ok' | 'warn' | 'missing';
      backend: string;
      message: string;
    }> = {};

    // YouTube
    const hasYtDlp = isCommandAvailable('yt-dlp');
    platforms['youtube'] = {
      name: 'YouTube',
      category: 'video',
      tier: 0,
      status: hasYtDlp ? 'ok' : 'missing',
      backend: hasYtDlp ? 'yt-dlp' : 'none',
      message: hasYtDlp ? 'yt-dlp installed (subtitles & metadata ready)' : 'yt-dlp not found in PATH',
    };

    // GitHub
    const hasGh = isCommandAvailable('gh');
    const hasGhToken = Boolean(env.GITHUB_PERSONAL_ACCESS_TOKEN || env.GH_TOKEN || env.GITHUB_TOKEN);
    platforms['github'] = {
      name: 'GitHub',
      category: 'dev',
      tier: 0,
      status: (hasGh || hasGhToken) ? 'ok' : 'warn',
      backend: hasGh ? 'gh CLI' : (hasGhToken ? 'REST API (token)' : 'Public API (rate-limited)'),
      message: hasGh ? 'gh CLI available' : (hasGhToken ? 'Token configured' : 'Unauthenticated (rate limit: 60/hr)'),
    };

    // Jina Reader / Universal Web
    platforms['web'] = {
      name: 'Universal Web Reader',
      category: 'web',
      tier: 0,
      status: 'ok',
      backend: 'Jina Reader (r.jina.ai)',
      message: 'Zero-config Markdown reader active',
    };

    // V2EX
    platforms['v2ex'] = {
      name: 'V2EX',
      category: 'social',
      tier: 0,
      status: 'ok',
      backend: 'Public HTTPS JSON API',
      message: 'Zero-config hot topics and discussions active',
    };

    // Bilibili
    const hasBili = isCommandAvailable('bili');
    platforms['bilibili'] = {
      name: 'Bilibili',
      category: 'video',
      tier: hasBili ? 0 : 1,
      status: hasBili ? 'ok' : 'warn',
      backend: hasBili ? 'bili-cli' : 'Web API / Jina fallback',
      message: hasBili ? 'bili-cli installed' : 'Needs bili-cli for full access (platform anti-bot 412 active)',
    };

    // Reddit
    const hasOpenCli = isCommandAvailable('opencli');
    const hasRdt = isCommandAvailable('rdt');
    platforms['reddit'] = {
      name: 'Reddit',
      category: 'social',
      tier: 1,
      status: (hasOpenCli || hasRdt) ? 'ok' : 'warn',
      backend: hasOpenCli ? 'OpenCLI' : (hasRdt ? 'rdt-cli' : 'Jina Reader fallback'),
      message: (hasOpenCli || hasRdt) ? 'Desktop/CLI backend available' : 'Using Jina Reader fallback (read-only)',
    };

    // Twitter / X
    const hasTwitterCli = isCommandAvailable('twitter');
    const hasTwitterToken = Boolean(env.TWITTER_AUTH_TOKEN && env.TWITTER_CT0);
    platforms['twitter'] = {
      name: 'Twitter / X',
      category: 'social',
      tier: 1,
      status: (hasTwitterCli || hasTwitterToken || hasOpenCli) ? 'ok' : 'warn',
      backend: hasTwitterCli ? 'twitter-cli' : (hasOpenCli ? 'OpenCLI' : (hasTwitterToken ? 'Token auth' : 'Jina Reader fallback')),
      message: (hasTwitterCli || hasTwitterToken || hasOpenCli) ? 'Authenticated backend ready' : 'Requires cookies or OpenCLI for search',
    };

    // XiaoHongShu
    platforms['xiaohongshu'] = {
      name: 'XiaoHongShu',
      category: 'social',
      tier: 2,
      status: hasOpenCli ? 'ok' : 'warn',
      backend: hasOpenCli ? 'OpenCLI (Chrome session)' : 'Cookie-Editor manual export',
      message: hasOpenCli ? 'OpenCLI Chrome bridge connected' : 'Requires OpenCLI Chrome extension or cookies',
    };

    // LinkedIn
    platforms['linkedin'] = {
      name: 'LinkedIn',
      category: 'career',
      tier: 1,
      status: 'ok',
      backend: 'Jina Reader / Public Profile fetcher',
      message: 'Public profiles readable via Jina Reader',
    };

    // Xueqiu (Finance)
    platforms['xueqiu'] = {
      name: 'Xueqiu (Stocks & Finance)',
      category: 'finance',
      tier: 1,
      status: hasOpenCli ? 'ok' : 'warn',
      backend: hasOpenCli ? 'OpenCLI' : 'Public web search fallback',
      message: hasOpenCli ? 'OpenCLI session ready' : 'Requires OpenCLI or session cookie for quotes',
    };

    // Exa AI Search
    const hasExaKey = Boolean(config.tools?.reach?.exaApiKey || env.EXA_API_KEY);
    platforms['exa'] = {
      name: 'Exa AI Search',
      category: 'search',
      tier: 1,
      status: hasExaKey ? 'ok' : 'warn',
      backend: hasExaKey ? 'Exa API' : 'None (using SearXNG/Tavily/Jina)',
      message: hasExaKey ? 'Exa search key configured' : 'Configure EXA_API_KEY for Exa neural search',
    };

    // Whisper Audio Transcription
    const hasGroq = Boolean(config.tools?.reach?.groqApiKey || env.GROQ_API_KEY);
    const hasOpenAI = Boolean(env.OPENAI_API_KEY);
    const hasWhisperCli = isCommandAvailable('whisper');
    platforms['whisper'] = {
      name: 'Audio Transcription (Whisper)',
      category: 'audio',
      tier: 0,
      status: (hasGroq || hasOpenAI || hasWhisperCli) ? 'ok' : 'warn',
      backend: hasGroq ? 'Groq Cloud (whisper-large-v3-turbo)' : (hasOpenAI ? 'OpenAI Whisper' : (hasWhisperCli ? 'Local Whisper CLI' : 'none')),
      message: (hasGroq || hasOpenAI || hasWhisperCli)
        ? `Active: ${hasGroq ? 'Groq Whisper (fast & free)' : hasOpenAI ? 'OpenAI Whisper' : 'Local Whisper'}`
        : 'Set GROQ_API_KEY (free at console.groq.com) for video/podcast transcription',
    };

    // Format output report
    const lines: string[] = [
      '# Agent Reach — Platform Availability Report',
      '',
      'Legend: [OK] Ready  [WARN] Needs login/key/fallback  [MISSING] Tool missing',
      '',
      '| Platform | Category | Status | Active Backend | Notes |',
      '|---|---|---|---|---|',
    ];

    for (const [_, p] of Object.entries(platforms)) {
      const statusBadge = p.status === 'ok' ? 'OK' : p.status === 'warn' ? 'WARN' : 'MISSING';
      lines.push(`| **${p.name}** | ${p.category} | ${statusBadge} | ${p.backend} | ${p.message} |`);
    }

    if (detailed) {
      lines.push(
        '',
        '### Upstream Tools Status',
        `- yt-dlp: ${hasYtDlp ? 'Installed' : 'Not found'}`,
        `- gh CLI: ${hasGh ? 'Installed' : 'Not found'}`,
        `- OpenCLI: ${hasOpenCli ? 'Installed' : 'Not found'}`,
        `- Groq API Key: ${hasGroq ? 'Configured' : 'Not configured'}`,
        `- Exa API Key: ${hasExaKey ? 'Configured' : 'Not configured'}`,
        `- Python: ${isCommandAvailable('python') ? 'Available' : 'Not found'}`
      );
    }

    return {
      success: true,
      output: lines.join('\n'),
    };
  },
});

// ─── 2. web_extract (Jina Reader) ─────────────────────────────────────────────

toolRegistry.register({
  name: 'web_extract',
  description: 'Extract clean, LLM-ready markdown content from any webpage, article, blog, Reddit post, or documentation using Jina Reader (r.jina.ai). Stips ads and boilerplates.',
  category: 'web',
  parameters: [
    { name: 'url', type: 'string', description: 'The URL to extract content from', required: true },
    { name: 'format', type: 'string', description: 'Output format: "markdown" (default) or "text"' },
    { name: 'retainImages', type: 'boolean', description: 'Whether to retain image links in markdown output (default: false)' },
    { name: 'timeoutMs', type: 'number', description: 'Extraction timeout in milliseconds (default: 20000)' },
  ],
  usageNotes: [
    'Use this instead of web_fetch when you need high-quality article/documentation markdown without HTML noise.',
    'Works on almost any public webpage including articles, tech docs, Reddit posts, and Twitter threads.',
    'Automatically falls back to direct fetch if Jina Reader is unreachable.',
  ],
  examples: [
    { userIntent: 'read article content', arguments: { url: 'https://news.ycombinator.com', format: 'markdown' } },
  ],
  keywords: ['extract', 'jina', 'markdown', 'article', 'read', 'scrape', 'page', 'content', 'reader'],
  handler: async (args): Promise<ToolResult> => {
    const url = String(args.url ?? '').trim();
    if (!url) return { success: false, output: 'No URL specified' };

    const format = args.format === 'text' ? 'text' : 'markdown';
    const retainImages = Boolean(args.retainImages);
    const timeoutMs = Number(args.timeoutMs) || 20000;
    const config = getConfig();
    const apiKey = config.tools?.reach?.jinaApiKey || process.env.JINA_API_KEY;

    // Build Jina Reader URL
    const jinaUrl = `https://r.jina.ai/${url}`;
    const headers: Record<string, string> = {
      'Accept': format === 'text' ? 'text/plain' : 'text/markdown',
      'X-No-Cache': 'true',
    };
    if (!retainImages) {
      headers['X-Retain-Images'] = 'none';
    }
    if (apiKey) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }

    try {
      log.debug({ url, jinaUrl }, 'Extracting content via Jina Reader');
      const resp = await fetch(jinaUrl, {
        headers,
        signal: AbortSignal.timeout(timeoutMs),
      });

      if (resp.ok) {
        const text = await resp.text();
        if (text && text.length > 50 && !text.includes('Target URL returned error 412')) {
          return {
            success: true,
            output: text,
          };
        }
      }
      log.warn({ status: resp.status }, 'Jina Reader returned non-OK status, falling back to direct fetch');
    } catch (err: any) {
      log.warn({ error: err.message }, 'Jina Reader failed or timed out, falling back to direct fetch');
    }

    // Direct fallback
    try {
      const resp = await fetch(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 LiteClaw/1.0',
          'Accept': 'text/html,text/plain,application/json',
        },
        signal: AbortSignal.timeout(timeoutMs),
      });

      if (!resp.ok) {
        return { success: false, output: `Failed to fetch page: HTTP ${resp.status} ${resp.statusText}` };
      }

      const html = await resp.text();
      // Basic HTML to clean text conversion
      const cleaned = html
        .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
        .replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '')
        .replace(/<[^>]+>/g, ' ')
        .replace(/&nbsp;/g, ' ')
        .replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/\s{2,}/g, ' ')
        .replace(/\n\s*\n/g, '\n\n')
        .trim();

      return {
        success: true,
        output: cleaned.slice(0, 15000),
      };
    } catch (err: any) {
      return { success: false, output: `Extraction failed: ${err.message}` };
    }
  },
});

// ─── 3. reach_read ───────────────────────────────────────────────────────────

toolRegistry.register({
  name: 'reach_read',
  description: 'Smart content reader for 15 platforms: YouTube (subtitles & metadata), V2EX (topics & replies), Reddit, GitHub, Bilibili, and Twitter/X.',
  category: 'web',
  parameters: [
    { name: 'url', type: 'string', description: 'Platform URL to read (YouTube, V2EX, Reddit, GitHub, Bilibili, Twitter, etc.)', required: true },
    { name: 'subtitles', type: 'boolean', description: 'For YouTube videos, whether to extract subtitles/transcript (default: true)' },
    { name: 'maxItems', type: 'number', description: 'Maximum comments/replies/items to return (default: 10)' },
  ],
  usageNotes: [
    'For YouTube: extracts video title, duration, view count, description, and full speech transcript.',
    'For V2EX: fetches topic content, author, and all community replies via public API.',
    'For Reddit / Twitter: extracts clean discussion threads via Jina Reader.',
    'For GitHub: inspects repos, issues, and PRs.',
  ],
  keywords: ['read', 'youtube', 'subtitles', 'transcript', 'v2ex', 'reddit', 'github', 'bilibili', 'twitter', 'social'],
  handler: async (args, context): Promise<ToolResult> => {
    const rawUrl = String(args.url ?? '').trim();
    if (!rawUrl) return { success: false, output: 'No URL specified' };

    const getSubtitles = args.subtitles !== false;
    const maxItems = Math.min(Number(args.maxItems) || 10, 50);

    // 1. YouTube handling
    if (/youtube\.com\/watch|youtu\.be\/|youtube\.com\/shorts/i.test(rawUrl)) {
      if (!isCommandAvailable('yt-dlp')) {
        return { success: false, output: 'yt-dlp is required for YouTube extraction but is not installed.' };
      }

      try {
        // Fetch metadata
        const { stdout: metaJson } = await execFileAsync('yt-dlp', [
          '--dump-json',
          '--no-playlist',
          rawUrl,
        ], { timeout: 25000 });

        const meta = JSON.parse(metaJson);
        const videoInfo = [
          `# ${meta.title ?? 'YouTube Video'}`,
          `- **Channel**: ${meta.uploader ?? meta.channel ?? 'Unknown'}`,
          `- **Duration**: ${meta.duration_string ?? `${meta.duration}s`}`,
          `- **Views**: ${meta.view_count?.toLocaleString() ?? 'Unknown'}`,
          `- **Upload Date**: ${meta.upload_date ?? 'Unknown'}`,
          `- **URL**: ${meta.webpage_url ?? rawUrl}`,
          '',
          `### Description`,
          (meta.description ?? '').slice(0, 1500),
        ];

        // Fetch subtitles if requested
        if (getSubtitles) {
          const tempDir = tmpdir();
          const tempPrefix = join(tempDir, `yt_sub_${Date.now()}`);

          try {
            await execFileAsync('yt-dlp', [
              '--write-sub',
              '--write-auto-sub',
              '--sub-lang', 'en,zh-Hans,id,es,ja',
              '--skip-download',
              '-o', `${tempPrefix}.%(ext)s`,
              rawUrl,
            ], { timeout: 25000 });

            // Look for generated vtt file
            const files = readdirSync(tempDir);
            const vttFile = files.find(f => f.startsWith(`yt_sub_${Date.now()}`.slice(0, 10)) && f.endsWith('.vtt'));

            if (vttFile) {
              const vttPath = join(tempDir, vttFile);
              const vttContent = readFileSync(vttPath, 'utf-8');
              const transcript = parseVttToText(vttContent);
              try { unlinkSync(vttPath); } catch {}

              videoInfo.push(
                '',
                '### Subtitles / Speech Transcript',
                transcript ? transcript.slice(0, 12000) : 'No speech text detected in subtitles.'
              );
            } else {
              videoInfo.push('', '*(No subtitles available for this video)*');
            }
          } catch (subErr: any) {
            log.debug({ error: subErr.message }, 'Failed to download subtitles, skipping transcript');
            videoInfo.push('', '*(Subtitles could not be extracted)*');
          }
        }

        return {
          success: true,
          output: videoInfo.join('\n'),
        };
      } catch (err: any) {
        return { success: false, output: `YouTube extraction failed: ${err.message}` };
      }
    }

    // 2. V2EX topic handling
    const v2exMatch = rawUrl.match(/v2ex\.com\/t\/(\d+)/i);
    if (v2exMatch) {
      const topicId = v2exMatch[1];
      try {
        const [topicResp, repliesResp] = await Promise.all([
          fetch(`https://www.v2ex.com/api/topics/show.json?id=${topicId}`, {
            headers: { 'User-Agent': 'LiteClaw-AgentReach/1.0' },
          }),
          fetch(`https://www.v2ex.com/api/replies/show.json?topic_id=${topicId}&page_size=${maxItems}`, {
            headers: { 'User-Agent': 'LiteClaw-AgentReach/1.0' },
          }),
        ]);

        if (!topicResp.ok) {
          return { success: false, output: `V2EX topic API error: HTTP ${topicResp.status}` };
        }

        const topicData = (await topicResp.json()) as any[];
        const topic = Array.isArray(topicData) ? topicData[0] : topicData;
        const replies = (repliesResp.ok ? await repliesResp.json() : []) as any[];

        const lines = [
          `# ${topic?.title ?? 'V2EX Topic'}`,
          `- **Author**: ${topic?.member?.username ?? 'Unknown'}`,
          `- **Node**: ${topic?.node?.title ?? 'General'}`,
          `- **URL**: ${topic?.url ?? rawUrl}`,
          `- **Replies**: ${topic?.replies ?? 0}`,
          '',
          `### Topic Content`,
          topic?.content ?? '(No content)',
          '',
          `### Community Replies (Top ${Math.min(replies.length, maxItems)})`,
        ];

        for (const [idx, r] of replies.slice(0, maxItems).entries()) {
          lines.push(`**${idx + 1}. ${r.member?.username}**: ${r.content}`);
        }

        return { success: true, output: lines.join('\n') };
      } catch (err: any) {
        return { success: false, output: `V2EX read failed: ${err.message}` };
      }
    }

    // 3. Fallback: Jina Reader for Reddit, Twitter, GitHub, and general web
    const extractTool = toolRegistry.get('web_extract');
    if (extractTool) {
      return extractTool.handler({ url: rawUrl }, context);
    }
    return { success: false, output: 'web_extract tool not registered' };
  },
});

// ─── 4. reach_search ──────────────────────────────────────────────────────────

toolRegistry.register({
  name: 'reach_search',
  description: 'Cross-platform search across internet platforms: YouTube videos, GitHub repositories, V2EX topics, or neural web search.',
  category: 'web',
  parameters: [
    { name: 'query', type: 'string', description: 'Search query', required: true },
    { name: 'platform', type: 'string', description: 'Target platform: "youtube" | "github" | "v2ex" | "web" (default: "web")' },
    { name: 'limit', type: 'number', description: 'Maximum results to return (default: 5)' },
  ],
  usageNotes: [
    'Use platform="youtube" to find video titles, channels, durations, and URLs via yt-dlp.',
    'Use platform="github" to search open source repositories and documentation.',
    'Use platform="v2ex" to search developer community topics.',
    'Use platform="web" for general internet search.',
  ],
  keywords: ['reach search', 'search', 'youtube', 'github', 'v2ex', 'find', 'videos', 'repos'],
  handler: async (args, context): Promise<ToolResult> => {
    const query = String(args.query ?? '').trim();
    if (!query) return { success: false, output: 'No search query specified' };

    const platform = String(args.platform ?? 'web').toLowerCase();
    const limit = Math.min(Number(args.limit) || 5, 20);

    // 1. YouTube Search via yt-dlp
    if (platform === 'youtube') {
      if (!isCommandAvailable('yt-dlp')) {
        return { success: false, output: 'yt-dlp is required for YouTube search but is not installed.' };
      }

      try {
        const { stdout } = await execFileAsync('yt-dlp', [
          '--dump-json',
          '--flat-playlist',
          `ytsearch${limit}:${query}`,
        ], { timeout: 30000 });

        const items = stdout.trim().split('\n').filter(Boolean).map(line => {
          try { return JSON.parse(line); } catch { return null; }
        }).filter(Boolean);

        const lines = [`# YouTube Search Results for "${query}"`, ''];
        for (const [idx, item] of items.entries()) {
          const videoUrl = item.webpage_url ?? (item.url?.startsWith('http') ? item.url : `https://www.youtube.com/watch?v=${item.id ?? item.url}`);
          lines.push(
            `### ${idx + 1}. [${item.title}](${videoUrl})`,
            `- **Channel**: ${item.uploader ?? item.channel ?? 'Unknown'}`,
            `- **Duration**: ${item.duration_string ?? (item.duration ? `${item.duration}s` : 'Unknown')}`,
            `- **Views**: ${item.view_count?.toLocaleString() ?? 'Unknown'}`,
            `- **Snippet**: ${(item.description ?? '').slice(0, 200).replace(/\n/g, ' ')}...`,
            ''
          );
        }

        return { success: true, output: lines.join('\n') };
      } catch (err: any) {
        return { success: false, output: `YouTube search failed: ${err.message}` };
      }
    }

    // 2. GitHub Search
    if (platform === 'github') {
      if (isCommandAvailable('gh')) {
        try {
          const { stdout } = await execFileAsync('gh', [
            'search', 'repos', query,
            '--limit', String(limit),
            '--json', 'fullName,description,stargazersCount,url,updatedAt'
          ], { timeout: 15000 });

          const repos = JSON.parse(stdout);
          const lines = [`# GitHub Repositories for "${query}"`, ''];
          for (const [idx, r] of repos.entries()) {
            lines.push(
              `### ${idx + 1}. [${r.fullName}](${r.url}) ⭐ ${r.stargazersCount?.toLocaleString()}`,
              `- **Description**: ${r.description ?? 'No description'}`,
              `- **Updated**: ${r.updatedAt?.slice(0, 10)}`,
              ''
            );
          }
          return { success: true, output: lines.join('\n') };
        } catch {}
      }

      // Public API fallback
      try {
        const resp = await fetch(`https://api.github.com/search/repositories?q=${encodeURIComponent(query)}&per_page=${limit}`, {
          headers: {
            'User-Agent': 'LiteClaw-AgentReach/1.0',
            'Accept': 'application/vnd.github.v3+json',
          },
        });

        if (resp.ok) {
          const data = await resp.json() as any;
          const items = data.items ?? [];
          const lines = [`# GitHub Repositories for "${query}"`, ''];
          for (const [idx, r] of items.entries()) {
            lines.push(
              `### ${idx + 1}. [${r.full_name}](${r.html_url}) ⭐ ${r.stargazers_count?.toLocaleString()}`,
              `- **Description**: ${r.description ?? 'No description'}`,
              ''
            );
          }
          return { success: true, output: lines.join('\n') };
        }
      } catch (err: any) {
        return { success: false, output: `GitHub search failed: ${err.message}` };
      }
    }

    // 3. V2EX Hot / Node search
    if (platform === 'v2ex') {
      try {
        const resp = await fetch('https://www.v2ex.com/api/topics/hot.json', {
          headers: { 'User-Agent': 'LiteClaw-AgentReach/1.0' },
        });

        if (resp.ok) {
          const topics = await resp.json() as any[];
          const queryLower = query.toLowerCase();
          const filtered = topics.filter(t =>
            (t.title ?? '').toLowerCase().includes(queryLower) ||
            (t.content ?? '').toLowerCase().includes(queryLower)
          );

          const list = filtered.length > 0 ? filtered : topics.slice(0, limit);
          const lines = [`# V2EX Topics (${filtered.length > 0 ? `Matched "${query}"` : 'Hot Topics'})`, ''];
          for (const [idx, t] of list.slice(0, limit).entries()) {
            lines.push(`### ${idx + 1}. [${t.title}](${t.url})`);
            lines.push(`- **Author**: ${t.member?.username} | **Replies**: ${t.replies} | **Node**: ${t.node?.title}`);
            lines.push(`- **Snippet**: ${(t.content ?? '').slice(0, 150).replace(/\n/g, ' ')}...`, '');
          }
          return { success: true, output: lines.join('\n') };
        }
      } catch (err: any) {
        return { success: false, output: `V2EX search failed: ${err.message}` };
      }
    }

    // 4. Default: delegate to standard web_search tool
    const webSearchTool = toolRegistry.get('web_search');
    if (webSearchTool) {
      return webSearchTool.handler({ query, maxResults: limit }, context);
    }

    return { success: false, output: 'Web search tool not available.' };
  },
});

// ─── 5. reach_transcribe ──────────────────────────────────────────────────────

toolRegistry.register({
  name: 'reach_transcribe',
  description: 'Transcribe audio from a media URL (YouTube, podcast MP3) or local audio/video file using Whisper (Groq Cloud, OpenAI, or local Whisper).',
  category: 'web',
  parameters: [
    { name: 'urlOrPath', type: 'string', description: 'URL of video/audio or local file path to transcribe', required: true },
    { name: 'prompt', type: 'string', description: 'Optional context, technical glossary, or prompt to guide spelling' },
  ],
  usageNotes: [
    'Ideal for YouTube videos or podcasts that lack subtitles.',
    'Uses Groq Cloud Whisper (whisper-large-v3-turbo) when GROQ_API_KEY is configured (free & very fast).',
    'Falls back to OpenAI Whisper API or local whisper CLI.',
  ],
  keywords: ['transcribe', 'whisper', 'audio', 'podcast', 'speech to text', 'voice to text', 'video transcript'],
  handler: async (args): Promise<ToolResult> => {
    const target = String(args.urlOrPath ?? '').trim();
    if (!target) return { success: false, output: 'No audio URL or file path specified' };

    const config = getConfig();
    const groqKey = config.tools?.reach?.groqApiKey || process.env.GROQ_API_KEY;
    const openAiKey = process.env.OPENAI_API_KEY;
    const prompt = args.prompt ? String(args.prompt) : undefined;

    let localAudioPath = target;
    let cleanupNeeded = false;

    // If target is a URL, download audio using yt-dlp
    if (/^https?:\/\//i.test(target)) {
      if (!isCommandAvailable('yt-dlp')) {
        return { success: false, output: 'yt-dlp is required to download media audio for transcription, but is not installed.' };
      }

      const tempDir = tmpdir();
      const tempAudio = join(tempDir, `reach_audio_${Date.now()}.mp3`);
      try {
        log.info({ target }, 'Downloading audio via yt-dlp for transcription');
        await execFileAsync('yt-dlp', [
          '-x',
          '--audio-format', 'mp3',
          '--audio-quality', '5',
          '-o', tempAudio,
          target,
        ], { timeout: 60000 });

        if (!existsSync(tempAudio)) {
          return { success: false, output: 'Failed to extract audio track from media URL.' };
        }

        localAudioPath = tempAudio;
        cleanupNeeded = true;
      } catch (err: any) {
        return { success: false, output: `Failed to download audio for transcription: ${err.message}` };
      }
    }

    try {
      if (!existsSync(localAudioPath)) {
        return { success: false, output: `Audio file not found: ${localAudioPath}` };
      }

      // 1. Groq Cloud Whisper (fast & free tier)
      if (groqKey) {
        log.info('Using Groq Cloud Whisper API for transcription');
        const fileBuffer = readFileSync(localAudioPath);
        const blob = new Blob([fileBuffer], { type: 'audio/mp3' });
        const formData = new FormData();
        formData.append('file', blob, 'audio.mp3');
        formData.append('model', 'whisper-large-v3-turbo');
        formData.append('response_format', 'verbose_json');
        if (prompt) formData.append('prompt', prompt);

        const resp = await fetch('https://api.groq.com/openai/v1/audio/transcriptions', {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${groqKey}` },
          body: formData,
          signal: AbortSignal.timeout(90000),
        });

        if (resp.ok) {
          const result = await resp.json() as any;
          return {
            success: true,
            output: result.text ?? 'Empty transcription result.',
          };
        }
        log.warn({ status: resp.status }, 'Groq Whisper failed, trying fallback');
      }

      // 2. OpenAI Whisper
      if (openAiKey) {
        log.info('Using OpenAI Whisper API for transcription');
        const fileBuffer = readFileSync(localAudioPath);
        const blob = new Blob([fileBuffer], { type: 'audio/mp3' });
        const formData = new FormData();
        formData.append('file', blob, 'audio.mp3');
        formData.append('model', 'whisper-1');
        if (prompt) formData.append('prompt', prompt);

        const resp = await fetch('https://api.openai.com/v1/audio/transcriptions', {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${openAiKey}` },
          body: formData,
          signal: AbortSignal.timeout(90000),
        });

        if (resp.ok) {
          const result = await resp.json() as any;
          return {
            success: true,
            output: result.text ?? 'Empty transcription result.',
          };
        }
      }

      // 3. Local Whisper CLI
      if (isCommandAvailable('whisper')) {
        log.info('Using local whisper CLI for transcription');
        const { stdout } = await execFileAsync('whisper', [
          localAudioPath,
          '--model', 'base',
          '--output_format', 'txt',
          '--output_dir', tmpdir(),
        ], { timeout: 120000 });

        return {
          success: true,
          output: stdout || 'Transcription complete.',
        };
      }

      return {
        success: false,
        output: 'No Whisper provider available. Please set GROQ_API_KEY in your .env file (free at console.groq.com) or configure OPENAI_API_KEY.',
      };
    } finally {
      if (cleanupNeeded) {
        try { unlinkSync(localAudioPath); } catch {}
      }
    }
  },
});
