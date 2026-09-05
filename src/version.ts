/**
 * LiteClaw — Version Info
 *
 * Single source of truth for LiteClaw version.
 * Automatically resolves from package.json at runtime.
 */

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { existsSync, readFileSync } from 'fs';

let cachedVersion: string | null = null;

export function getVersion(): string {
  if (cachedVersion) return cachedVersion;
  try {
    const currentDir = dirname(fileURLToPath(import.meta.url));
    // 1. Check parent directory: ../package.json (when running from dist/ or src/)
    const p1 = join(currentDir, '..', 'package.json');
    if (existsSync(p1)) {
      const pkg = JSON.parse(readFileSync(p1, 'utf-8'));
      if (pkg.version) {
        const v = String(pkg.version);
        cachedVersion = v;
        return v;
      }
    }
    // 2. Check current directory: ./package.json (when running from root)
    const p2 = join(currentDir, 'package.json');
    if (existsSync(p2)) {
      const pkg = JSON.parse(readFileSync(p2, 'utf-8'));
      if (pkg.version) {
        const v = String(pkg.version);
        cachedVersion = v;
        return v;
      }
    }
    // 3. Walk up directory tree to find root package.json if bundled/nested
    let dir = currentDir;
    for (let i = 0; i < 4; i++) {
      const parent = dirname(dir);
      if (parent === dir) break;
      dir = parent;
      const candidate = join(dir, 'package.json');
      if (existsSync(candidate)) {
        const pkg = JSON.parse(readFileSync(candidate, 'utf-8'));
        if (pkg.name === 'liteclaw' && pkg.version) {
          const v = String(pkg.version);
          cachedVersion = v;
          return v;
        }
      }
    }
  } catch {
    // ignore
  }
  return '1.0.1';
}

export const VERSION: string = getVersion();
