/**
 * LiteClaw — Persistent Personality Memory Manager
 *
 * Manages persistent long-term knowledge in MEMORY.md (facts, decisions, world knowledge)
 * and USER.md (user profile, preferences, working habits).
 * Stored in state directory or project config.
 */

import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import { getStateDir } from '../config.js';
import { createLogger } from '../logger.js';

const log = createLogger('personality_memory');

export type MemoryTarget = 'memory' | 'user';

export function getMemoryFilePath(target: MemoryTarget): string {
  const stateDir = getStateDir();
  const dir = join(stateDir, 'personality');
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }

  const filename = target === 'user' ? 'USER.md' : 'MEMORY.md';
  const stateFile = join(dir, filename);

  // If state file doesn't exist yet, check if project config has a template or existing file to seed from
  if (!existsSync(stateFile)) {
    const projectConfigDir = join(process.cwd(), 'config', 'personality');
    const projectSource = join(projectConfigDir, filename);
    const templateSource = join(projectConfigDir, target === 'user' ? 'USER.template.md' : 'MEMORY.template.md');

    if (existsSync(projectSource)) {
      writeFileSync(stateFile, readFileSync(projectSource, 'utf-8'), 'utf-8');
    } else if (existsSync(templateSource)) {
      writeFileSync(stateFile, readFileSync(templateSource, 'utf-8'), 'utf-8');
    } else {
      const defaultContent = target === 'user'
        ? `# USER.md - About Your Human\n\n_Keep track of user preferences, habits, and key context._\n`
        : `# MEMORY.md - Curated Long-Term Memory\n\n_Important facts, project decisions, and persistent knowledge learned across sessions._\n`;
      writeFileSync(stateFile, defaultContent, 'utf-8');
    }
  }

  return stateFile;
}

export function readMemoryFile(target: MemoryTarget): string {
  const path = getMemoryFilePath(target);
  try {
    return readFileSync(path, 'utf-8');
  } catch (err: any) {
    log.error({ error: err.message, target }, 'Failed to read memory file');
    return '';
  }
}

export function appendMemoryEntry(target: MemoryTarget, entry: string): void {
  const path = getMemoryFilePath(target);
  const current = readMemoryFile(target);
  const dateStr = new Date().toISOString().split('T')[0];
  const formatted = entry.startsWith('-') ? entry : `- [${dateStr}] ${entry}`;
  const updated = current.trim() ? `${current.trim()}\n${formatted}\n` : `${formatted}\n`;
  writeFileSync(path, updated, 'utf-8');
  log.info({ target, path }, 'Appended entry to personality memory');
}

export function writeMemoryFile(target: MemoryTarget, content: string): void {
  const path = getMemoryFilePath(target);
  writeFileSync(path, content, 'utf-8');
  log.info({ target, path }, 'Updated personality memory file');
}

export function searchMemoryFiles(query: string): Array<{ target: MemoryTarget; line: string; lineNum: number }> {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  const results: Array<{ target: MemoryTarget; line: string; lineNum: number }> = [];

  for (const target of ['memory', 'user'] as MemoryTarget[]) {
    const content = readMemoryFile(target);
    const lines = content.split('\n');
    lines.forEach((line, index) => {
      const lower = line.toLowerCase();
      if (terms.some(t => lower.includes(t))) {
        results.push({ target, line: line.trim(), lineNum: index + 1 });
      }
    });
  }

  return results;
}
