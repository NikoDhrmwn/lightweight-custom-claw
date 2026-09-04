/**
 * LiteClaw — Workspace & System Path Resolver
 *
 * Shared path resolver used by:
 *   - filesystem tools (read_file, write_file, edit_file, delete_file, list_dir, find_files)
 *   - send_file tool
 *   - exec tool (cwd)
 *   - workspace API / gateway
 *
 * Supports expanding user shortcuts (~, %USERPROFILE%, documents, downloads, desktop)
 * and allows secure access to external system folders when permitted.
 */

import { resolve, normalize, relative, isAbsolute, join } from 'path';
import { homedir } from 'os';
import { getConfig } from '../config.js';
import { createLogger } from '../logger.js';

const log = createLogger('workspace');

// ─── Types ───────────────────────────────────────────────────────────

export interface ResolvedPath {
  /** The final absolute path, safe to pass to fs / spawn */
  absolute: string;
  /** Path relative to workspace root */
  relative: string;
  /** True if path is outside the configured workspace root */
  isExternal?: boolean;
}

export class PathEscapeError extends Error {
  constructor(requestedPath: string, workspace: string) {
    super(
      `Path "${requestedPath}" escapes the workspace root "${workspace}". ` +
      `Set agent.allowAbsolutePaths: true in config or request owner confirmation.`
    );
    this.name = 'PathEscapeError';
  }
}

// ─── Path Expansion Helpers ──────────────────────────────────────────

/**
 * Expands system environment variables, home directory shortcuts (~),
 * and common user folder aliases.
 */
export function expandSystemPath(rawPath: string): string {
  if (!rawPath) return rawPath;
  let pathStr = rawPath.trim();

  // 1. Expand Windows environment variables like %USERPROFILE%, %APPDATA%, %TEMP%
  pathStr = pathStr.replace(/%([^%]+)%/g, (_, name) => {
    return process.env[name] || process.env[name.toUpperCase()] || `\${name}`;
  });

  // 2. Expand home directory ~ or ~/ or ~\
  if (pathStr === '~' || pathStr.startsWith('~/') || pathStr.startsWith('~\\')) {
    pathStr = join(homedir(), pathStr.slice(1));
  }

  // 3. Common user folder shortcuts and subpaths (e.g. "documents/notes.txt", "downloads/data.csv")
  const lowered = pathStr.toLowerCase().replace(/\\/g, '/');
  const checkPrefix = (prefix: string, targetDir: string): string | null => {
    if (lowered === prefix) return targetDir;
    if (lowered.startsWith(`${prefix}/`)) {
      return join(targetDir, pathStr.slice(prefix.length + 1));
    }
    return null;
  };

  const matched =
    checkPrefix('documents', join(homedir(), 'Documents')) ||
    checkPrefix('my documents', join(homedir(), 'Documents')) ||
    checkPrefix('~/documents', join(homedir(), 'Documents')) ||
    checkPrefix('downloads', join(homedir(), 'Downloads')) ||
    checkPrefix('~/downloads', join(homedir(), 'Downloads')) ||
    checkPrefix('desktop', join(homedir(), 'Desktop')) ||
    checkPrefix('~/desktop', join(homedir(), 'Desktop')) ||
    checkPrefix('pictures', join(homedir(), 'Pictures')) ||
    checkPrefix('~/pictures', join(homedir(), 'Pictures'));

  if (matched) return matched;

  return pathStr;
}

/**
 * Check if a path is outside the given workspace root.
 */
export function isPathOutsideWorkspace(candidatePath: string, workspaceRoot?: string): boolean {
  const root = normalize(workspaceRoot ?? getWorkspaceRoot());
  const candidate = normalize(candidatePath);
  const rel = relative(root, candidate);
  return rel.startsWith('..') || (isAbsolute(rel) && !candidate.toLowerCase().startsWith(root.toLowerCase()));
}

// ─── Public API ──────────────────────────────────────────────────────

/**
 * Resolve a user-supplied path safely within the workspace.
 *
 * @param userPath  — The raw path from the LLM / user
 * @param workspaceRoot — Override for the workspace root (defaults to config agent.workspace or cwd)
 * @param options — allowExternal: if true, allows accessing paths outside workspace
 * @returns ResolvedPath with absolute, relative, and isExternal forms
 */
export function resolveWorkspacePath(
  userPath: string,
  workspaceRoot?: string,
  options?: { allowExternal?: boolean }
): ResolvedPath {
  const config = getConfig();
  const root = normalize(workspaceRoot ?? config.agent?.workspace ?? process.cwd());
  const allowAbsolute = options?.allowExternal ?? (config.agent?.allowAbsolutePaths ?? true);

  const expanded = expandSystemPath(userPath);

  // Normalize candidate path
  let candidate: string;
  if (isAbsolute(expanded)) {
    candidate = normalize(expanded);
  } else {
    candidate = normalize(resolve(root, expanded));
  }

  const rel = relative(root, candidate);
  const isExternal = rel.startsWith('..') || (isAbsolute(rel) && !candidate.toLowerCase().startsWith(root.toLowerCase()));

  if (isExternal && !allowAbsolute) {
    throw new PathEscapeError(userPath, root);
  }

  log.debug({ userPath, resolved: candidate, workspace: root, isExternal }, 'Path resolved');

  return {
    absolute: candidate,
    relative: rel || '.',
    isExternal,
  };
}

/**
 * Flexible path resolver that always returns resolved path and notes whether
 * it lies outside the current workspace.
 */
export function resolveFlexiblePath(
  userPath: string,
  workspaceRoot?: string
): ResolvedPath & { isExternal: boolean } {
  const res = resolveWorkspacePath(userPath, workspaceRoot, { allowExternal: true });
  return {
    ...res,
    isExternal: !!res.isExternal,
  };
}

/**
 * Quick boolean check — does this path stay inside the workspace?
 */
export function isInsideWorkspace(
  userPath: string,
  workspaceRoot?: string,
): boolean {
  try {
    const res = resolveWorkspacePath(userPath, workspaceRoot, { allowExternal: false });
    return !res.isExternal;
  } catch {
    return false;
  }
}

/**
 * Get the resolved workspace root from config.
 */
export function getWorkspaceRoot(): string {
  const config = getConfig();
  return normalize(config.agent?.workspace ?? process.cwd());
}
