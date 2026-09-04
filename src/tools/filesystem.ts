/**
 * LiteClaw — Filesystem & File Management Tools
 * 
 * read_file, write_file, edit_file, delete_file (with confirmation),
 * list_dir, find_files, search_file_content, copy_file, move_file,
 * and send_file (sends to originating channel).
 *
 * Supports accessing external folders (e.g. Documents, Downloads, C:\)
 * with owner-verified safety checks for destructive operations.
 */

import {
  readFileSync,
  writeFileSync,
  unlinkSync,
  existsSync,
  statSync,
  readdirSync,
  copyFileSync,
  renameSync,
  mkdirSync,
} from 'fs';
import { join, basename, dirname, extname } from 'path';
import { toolRegistry, ToolContext, ToolResult } from '../core/tools.js';
import {
  resolveWorkspacePath,
  resolveFlexiblePath,
  expandSystemPath,
  isPathOutsideWorkspace,
  PathEscapeError,
} from '../core/workspace.js';
import { readFileFormatted } from '../core/file_processor.js';

// ─── read_file ───────────────────────────────────────────────────────

toolRegistry.register({
  name: 'read_file',
  description: 'Read the contents of a file. Returns the text content. Specify startLine/endLine for partial reads. Supports paths anywhere on the system (e.g. "documents/notes.txt", "C:\\Users\\...").',
  category: 'filesystem',
  parameters: [
    { name: 'path', type: 'string', description: 'Absolute or relative path to the file (supports ~, %USERPROFILE%, documents, C:\\...)', required: true },
    { name: 'startLine', type: 'number', description: 'Start line (1-indexed, optional)' },
    { name: 'endLine', type: 'number', description: 'End line (1-indexed, inclusive, optional)' },
    { name: 'lineNumbers', type: 'boolean', description: 'Include line numbers in output (optional, default: false)' },
  ],
  usageNotes: [
    'Use this when you already know the file path or filename you need to inspect.',
    'Works on files inside the workspace and in external folders (Documents, Downloads, etc.).',
    'If the file may be large, include startLine/endLine instead of reading the whole thing.',
    'Do not call list_dir first unless the path is genuinely unknown.'
  ],
  examples: [
    { userIntent: 'read package.json', arguments: { path: 'package.json' } },
    { userIntent: 'inspect file in Documents', arguments: { path: 'documents/notes.txt' } },
    { userIntent: 'read absolute C drive file', arguments: { path: 'C:\\Users\\elect\\Documents\\todo.txt' } },
  ],
  keywords: ['read', 'file', 'open', 'show', 'content', 'view', 'cat', 'type', 'display', 'look', 'check'],
  handler: async (args, context): Promise<ToolResult> => {
    let filePath: string;
    try {
      filePath = resolveFlexiblePath(args.path, context.workingDir).absolute;
    } catch (err: any) {
      return { success: false, output: `Path error: ${err.message}` };
    }

    if (!existsSync(filePath)) {
      return { success: false, output: `File not found: ${filePath}` };
    }

    try {
      const stat = statSync(filePath);
      if (stat.isDirectory()) {
        return { success: false, output: `Path is a directory. Use list_dir or find_files instead.` };
      }

      const ext = filePath.toLowerCase().substring(filePath.lastIndexOf('.'));

      // Check for structured document formats (Excel, PDF, Word)
      const docExtensions = ['.xlsx', '.xls', '.pdf', '.docx', '.doc'];
      if (docExtensions.includes(ext)) {
        try {
          const formattedDoc = await readFileFormatted(filePath);
          return {
            success: true,
            output: formattedDoc,
            filePath,
          };
        } catch (err: any) {
          return { success: false, output: `Error reading ${ext} document: ${err.message}` };
        }
      }

      // Size guard: don't read huge files into context
      if (stat.size > 500_000) {
        return { success: false, output: `File is too large (${(stat.size / 1024).toFixed(0)} KB). Use startLine/endLine for partial reads.` };
      }

      // Extension check for remaining binary formats
      const binaryExtensions = ['.exe', '.dll', '.so', '.dylib', '.bin', '.pptx', '.zip', '.gz', '.tar', '.7z', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.ico', '.mp3', '.mp4', '.mov'];
      if (binaryExtensions.includes(ext)) {
        return { success: false, output: `File '${args.path}' is a binary media/archive file (${ext}). LiteClaw cannot display raw binary data. Use 'send_file' if you need to share it.` };
      }

      let content = readFileSync(filePath, 'utf-8');

      // Basic binary content detection
      if (content.includes('\u0000')) {
        return { success: false, output: `File '${args.path}' contains binary data and cannot be read as text.` };
      }

      const allLines = content.split('\n');

      // Apply line range if specified
      let start = 0;
      let end = allLines.length;
      if (args.startLine || args.endLine) {
        start = Math.max(1, args.startLine ?? 1) - 1;
        end = Math.min(allLines.length, args.endLine ?? allLines.length);
      }

      const slice = allLines.slice(start, end);
      
      if (args.lineNumbers) {
        content = slice.map((line, idx) => {
          const lineNum = start + idx + 1;
          return `${lineNum.toString().padStart(4, ' ')} | ${line}`;
        }).join('\n');
      } else {
        content = slice.join('\n');
      }

      return {
        success: true,
        output: `File: ${filePath} (lines ${start + 1}-${end} of ${allLines.length})\n\n${content}`,
      };
    } catch (err: any) {
      return { success: false, output: `Error reading file: ${err.message}` };
    }
  },
});

// ─── write_file ──────────────────────────────────────────────────────

toolRegistry.register({
  name: 'write_file',
  description: 'Write content to a file. Creates the file if it does not exist, overwrites if it does. If the destination is outside the workspace, requests owner confirmation first.',
  category: 'filesystem',
  parameters: [
    { name: 'path', type: 'string', description: 'Path to the file to write', required: true },
    { name: 'content', type: 'string', description: 'Content to write to the file', required: true },
  ],
  usageNotes: [
    'Use this only after you have already determined the exact file path and full content to save.',
    'For edits to existing code, prefer edit_file over write_file to avoid accidental truncation.',
    'Writing outside the workspace requires confirmation.'
  ],
  examples: [
    { userIntent: 'create a notes file', arguments: { path: 'notes.txt', content: 'Hello' } },
  ],
  keywords: ['write', 'create', 'save', 'file', 'output', 'generate', 'put'],
  handler: async (args, context): Promise<ToolResult> => {
    let resolved;
    try {
      resolved = resolveFlexiblePath(args.path, context.workingDir);
    } catch (err: any) {
      return { success: false, output: `Path error: ${err.message}` };
    }

    const filePath = resolved.absolute;

    // If writing outside the workspace, require owner confirmation
    if (resolved.isExternal && context.requestConfirmation) {
      const confirmed = await context.requestConfirmation(
        `Tool wants to write to external system path outside workspace: "${filePath}"`
      );
      if (!confirmed) {
        return { success: false, output: 'Owner rejected writing to external path.' };
      }
    }

    try {
      const dir = dirname(filePath);
      if (!existsSync(dir)) {
        mkdirSync(dir, { recursive: true });
      }

      const existed = existsSync(filePath);
      writeFileSync(filePath, args.content, 'utf-8');
      const size = statSync(filePath).size;

      return {
        success: true,
        output: `${existed ? 'Updated' : 'Created'} file: ${filePath} (${size} bytes)`,
        filePath,
      };
    } catch (err: any) {
      return { success: false, output: `Error writing file: ${err.message}` };
    }
  },
});

// ─── edit_file ───────────────────────────────────────────────────────

toolRegistry.register({
  name: 'edit_file',
  description: 'Edit an existing file using search and replace blocks. If the target file is outside the workspace, requests owner confirmation first.',
  category: 'filesystem',
  parameters: [
    { name: 'path', type: 'string', description: 'Path to the file to edit', required: true },
    { 
      name: 'edits', 
      type: 'array', 
      description: 'List of edits to apply', 
      required: true,
      items: {
        type: 'object',
        properties: {
          search: { type: 'string', description: 'The exact string to find in the file' },
          replace: { type: 'string', description: 'The string to replace it with' }
        },
        required: ['search', 'replace']
      }
    },
  ],
  usageNotes: [
    'Use this for modifying existing files. It prevents accidental truncation.',
    'The "search" string MUST match exactly, including indentation and whitespace.',
    'If the search string matches multiple times, the tool will fail to ensure precision.'
  ],
  keywords: ['edit', 'modify', 'replace', 'fix', 'update', 'change', 'patch'],
  handler: async (args, context): Promise<ToolResult> => {
    let resolved;
    try {
      resolved = resolveFlexiblePath(args.path, context.workingDir);
    } catch (err: any) {
      return { success: false, output: `Path error: ${err.message}` };
    }

    const filePath = resolved.absolute;

    if (!existsSync(filePath)) {
      return { success: false, output: `File not found: ${filePath}` };
    }

    // Require confirmation if editing outside the workspace
    if (resolved.isExternal && context.requestConfirmation) {
      const confirmed = await context.requestConfirmation(
        `Tool wants to edit an external file outside workspace: "${filePath}"`
      );
      if (!confirmed) {
        return { success: false, output: 'Owner rejected editing external file.' };
      }
    }

    try {
      let content = readFileSync(filePath, 'utf-8');
      const edits = args.edits as { search: string, replace: string }[];

      for (const edit of edits) {
        const parts = content.split(edit.search);
        
        if (parts.length === 1) {
          return { 
            success: false, 
            output: `Search block not found in ${args.path}. Ensure whitespace and indentation match exactly.\n\nSearch attempt:\n${edit.search}` 
          };
        }
        
        if (parts.length > 2) {
          return { 
            success: false, 
            output: `Search block matches multiple times (${parts.length - 1}) in ${args.path}. Provide more context in the search string to make it unique.` 
          };
        }

        content = parts.join(edit.replace);
      }

      writeFileSync(filePath, content, 'utf-8');
      return {
        success: true,
        output: `Successfully applied ${edits.length} edits to ${filePath}`,
        filePath,
      };
    } catch (err: any) {
      return { success: false, output: `Error editing file: ${err.message}` };
    }
  },
});

// ─── delete_file ─────────────────────────────────────────────────────

toolRegistry.register({
  name: 'delete_file',
  description: 'Delete a file. Strictly requires owner confirmation before proceeding.',
  category: 'filesystem',
  parameters: [
    { name: 'path', type: 'string', description: 'Path to the file to delete', required: true },
  ],
  usageNotes: [
    'Use this only when the user clearly asked to remove a file.',
    'Always requires owner confirmation.'
  ],
  keywords: ['delete', 'remove', 'rm', 'del', 'erase', 'destroy', 'clean'],
  requiresConfirmation: true,
  handler: async (args, context): Promise<ToolResult> => {
    let filePath: string;
    try {
      filePath = resolveFlexiblePath(args.path, context.workingDir).absolute;
    } catch (err: any) {
      return { success: false, output: `Path error: ${err.message}` };
    }

    if (!existsSync(filePath)) {
      return { success: false, output: `File not found: ${filePath}` };
    }

    try {
      unlinkSync(filePath);
      return { success: true, output: `Deleted: ${filePath}` };
    } catch (err: any) {
      return { success: false, output: `Error deleting file: ${err.message}` };
    }
  },
});

// ─── list_dir ────────────────────────────────────────────────────────

toolRegistry.register({
  name: 'list_dir',
  description: 'List the contents of a directory, showing files and subdirectories with sizes. Works on any accessible directory (e.g. ".", "documents", "C:\\Users\\...").',
  category: 'filesystem',
  parameters: [
    { name: 'path', type: 'string', description: 'Path to the directory to list (supports ~, %USERPROFILE%, documents, C:\\...)', required: true },
  ],
  usageNotes: [
    'Use this when the user asks what files exist or to browse a directory.',
    'Works on workspace folders as well as external folders like Documents or Downloads.'
  ],
  examples: [
    { userIntent: 'what files are here', arguments: { path: '.' } },
    { userIntent: 'list my documents folder', arguments: { path: 'documents' } },
  ],
  keywords: ['list', 'directory', 'folder', 'dir', 'ls', 'tree', 'files', 'contents', 'what'],
  handler: async (args, context): Promise<ToolResult> => {
    let dirPath: string;
    try {
      dirPath = resolveFlexiblePath(args.path || '.', context.workingDir).absolute;
    } catch (err: any) {
      return { success: false, output: `Path error: ${err.message}` };
    }

    if (!existsSync(dirPath)) {
      return { success: false, output: `Directory not found: ${dirPath}` };
    }

    try {
      const entries = readdirSync(dirPath, { withFileTypes: true });
      const lines: string[] = [`Directory: ${dirPath}\n`];

      const dirs: string[] = [];
      const files: string[] = [];

      for (const entry of entries) {
        if (entry.name.startsWith('.') && entry.name !== '..') continue;

        if (entry.isDirectory()) {
          dirs.push(`  📁 ${entry.name}/`);
        } else {
          try {
            const stat = statSync(join(dirPath, entry.name));
            const sizeKB = (stat.size / 1024).toFixed(1);
            files.push(`  📄 ${entry.name}  (${sizeKB} KB)`);
          } catch {
            files.push(`  📄 ${entry.name}`);
          }
        }
      }

      lines.push(...dirs.sort(), ...files.sort());
      lines.push(`\nTotal: ${dirs.length} directories, ${files.length} files`);

      return { success: true, output: lines.join('\n') };
    } catch (err: any) {
      return { success: false, output: `Error listing directory: ${err.message}` };
    }
  },
});

// ─── find_files ──────────────────────────────────────────────────────

toolRegistry.register({
  name: 'find_files',
  description: 'Search for files by name, pattern, or extension across any directory (including external folders like Documents, Downloads, or C:\\).',
  category: 'filesystem',
  parameters: [
    { name: 'pattern', type: 'string', description: 'Pattern to search for (e.g. "*.pdf", "budget", "invoice*.xlsx", ".txt")', required: true },
    { name: 'path', type: 'string', description: 'Directory to search within (e.g. ".", "documents", "downloads", "C:\\Users\\..."). Defaults to workspace root.' },
    { name: 'recursive', type: 'boolean', description: 'Search subdirectories recursively (default: true).' },
    { name: 'max_depth', type: 'number', description: 'Maximum directory recursion depth (default: 5).' },
    { name: 'limit', type: 'number', description: 'Maximum number of results to return (default: 30).' },
    { name: 'file_type', type: 'string', description: 'Filter by "file", "directory", or "all" (default: "file").', enum: ['file', 'directory', 'all'] },
  ],
  keywords: ['find', 'search', 'locate', 'where is', 'list files', 'scan', 'documents', 'c drive', 'find file', 'look for file'],
  handler: async (args, context): Promise<ToolResult> => {
    let searchDir: string;
    try {
      searchDir = resolveFlexiblePath(args.path || '.', context.workingDir).absolute;
    } catch (err: any) {
      return { success: false, output: `Path error: ${err.message}` };
    }

    if (!existsSync(searchDir)) {
      return { success: false, output: `Directory not found: ${searchDir}` };
    }

    const pattern = typeof args.pattern === 'string' ? args.pattern.trim() : '*';
    const recursive = args.recursive !== false;
    const maxDepth = typeof args.max_depth === 'number' && args.max_depth >= 0 ? args.max_depth : 5;
    const limit = typeof args.limit === 'number' && args.limit > 0 ? args.limit : 30;
    const fileType = args.file_type || 'file';

    const results: Array<{ path: string; isDir: boolean; size: number; mtime: Date }> = [];

    try {
      searchDirectoryRecursive(searchDir, pattern, { recursive, maxDepth, limit, fileType }, 0, results);

      if (results.length === 0) {
        return {
          success: true,
          output: `No ${fileType === 'file' ? 'files' : fileType === 'directory' ? 'directories' : 'items'} found matching "${pattern}" in "${searchDir}".`,
        };
      }

      const lines = [`Found ${results.length} item(s) matching "${pattern}" in "${searchDir}":\n`];
      for (let i = 0; i < results.length; i++) {
        const item = results[i];
        const icon = item.isDir ? '📁' : '📄';
        const sizeStr = item.isDir ? '' : ` (${(item.size / 1024).toFixed(1)} KB)`;
        const dateStr = item.mtime.toLocaleDateString();
        lines.push(`${i + 1}. ${icon} ${item.path}${sizeStr} — ${dateStr}`);
      }

      return {
        success: true,
        output: lines.join('\n'),
      };
    } catch (err: any) {
      return { success: false, output: `Error searching directory: ${err.message}` };
    }
  },
});

// ─── search_file_content ─────────────────────────────────────────────

toolRegistry.register({
  name: 'search_file_content',
  description: 'Search for text or keywords inside files across a folder (similar to grep/ripgrep).',
  category: 'filesystem',
  parameters: [
    { name: 'query', type: 'string', description: 'The text or pattern to search for inside files.', required: true },
    { name: 'path', type: 'string', description: 'Directory to search within (defaults to workspace root).' },
    { name: 'extension', type: 'string', description: 'Optional file extension filter (e.g. ".txt", ".ts", ".md", ".json").' },
    { name: 'limit', type: 'number', description: 'Maximum number of matches to return (default: 20).' },
  ],
  keywords: ['grep', 'search inside', 'find text', 'search code', 'search text', 'content search', 'contain'],
  handler: async (args, context): Promise<ToolResult> => {
    let searchDir: string;
    try {
      searchDir = resolveFlexiblePath(args.path || '.', context.workingDir).absolute;
    } catch (err: any) {
      return { success: false, output: `Path error: ${err.message}` };
    }

    if (!existsSync(searchDir)) {
      return { success: false, output: `Directory not found: ${searchDir}` };
    }

    const query = typeof args.query === 'string' ? args.query : '';
    if (!query) {
      return { success: false, output: 'Query is required.' };
    }

    const extFilter = args.extension ? args.extension.toLowerCase().trim() : '';
    const limit = typeof args.limit === 'number' && args.limit > 0 ? args.limit : 20;

    const files: string[] = [];
    collectTextFiles(searchDir, extFilter, files, 0, 4, 100);

    const matches: Array<{ file: string; line: number; text: string }> = [];

    for (const file of files) {
      if (matches.length >= limit) break;
      try {
        const content = readFileSync(file, 'utf-8');
        if (content.includes('\u0000')) continue; // Skip binary
        const lines = content.split('\n');
        for (let idx = 0; idx < lines.length; idx++) {
          if (matches.length >= limit) break;
          if (lines[idx].toLowerCase().includes(query.toLowerCase())) {
            matches.push({
              file,
              line: idx + 1,
              text: lines[idx].trim(),
            });
          }
        }
      } catch {
        // Skip unreadable files
      }
    }

    if (matches.length === 0) {
      return {
        success: true,
        output: `No matches found for "${query}" across ${files.length} scanned files in "${searchDir}".`,
      };
    }

    const lines = [`Matches for "${query}" (${matches.length} found):\n`];
    for (const m of matches) {
      lines.push(`• ${m.file}:${m.line}\n  ${m.text}`);
    }

    return {
      success: true,
      output: lines.join('\n\n'),
    };
  },
});

// ─── copy_file ───────────────────────────────────────────────────────

toolRegistry.register({
  name: 'copy_file',
  description: 'Copy a file from a source path to a destination path. Works across folders and drives.',
  category: 'filesystem',
  parameters: [
    { name: 'source', type: 'string', description: 'Path to the source file.', required: true },
    { name: 'destination', type: 'string', description: 'Path to destination file or directory.', required: true },
    { name: 'overwrite', type: 'boolean', description: 'Whether to overwrite destination if it exists (default: false).' },
  ],
  keywords: ['copy', 'cp', 'duplicate', 'clone file'],
  handler: async (args, context): Promise<ToolResult> => {
    let srcPath: string;
    let dstPath: string;

    try {
      srcPath = resolveFlexiblePath(args.source, context.workingDir).absolute;
      dstPath = resolveFlexiblePath(args.destination, context.workingDir).absolute;
    } catch (err: any) {
      return { success: false, output: `Path error: ${err.message}` };
    }

    if (!existsSync(srcPath)) {
      return { success: false, output: `Source file not found: ${srcPath}` };
    }

    try {
      if (existsSync(dstPath) && statSync(dstPath).isDirectory()) {
        dstPath = join(dstPath, basename(srcPath));
      }

      if (existsSync(dstPath) && !args.overwrite) {
        return { success: false, output: `Destination file already exists: ${dstPath}. Pass overwrite: true to replace.` };
      }

      const dir = dirname(dstPath);
      if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

      copyFileSync(srcPath, dstPath);
      return {
        success: true,
        output: `Successfully copied "${srcPath}" to "${dstPath}".`,
      };
    } catch (err: any) {
      return { success: false, output: `Error copying file: ${err.message}` };
    }
  },
});

// ─── move_file ───────────────────────────────────────────────────────

toolRegistry.register({
  name: 'move_file',
  description: 'Move or rename a file from source to destination. If moving outside the workspace, requests owner confirmation.',
  category: 'filesystem',
  parameters: [
    { name: 'source', type: 'string', description: 'Path to the source file.', required: true },
    { name: 'destination', type: 'string', description: 'Path to destination file or directory.', required: true },
  ],
  keywords: ['move', 'mv', 'rename', 'relocate file'],
  handler: async (args, context): Promise<ToolResult> => {
    let srcPath: string;
    let dstPath: string;
    let dstResolved;

    try {
      srcPath = resolveFlexiblePath(args.source, context.workingDir).absolute;
      dstResolved = resolveFlexiblePath(args.destination, context.workingDir);
      dstPath = dstResolved.absolute;
    } catch (err: any) {
      return { success: false, output: `Path error: ${err.message}` };
    }

    if (!existsSync(srcPath)) {
      return { success: false, output: `Source file not found: ${srcPath}` };
    }

    // If destination is outside workspace, require confirmation
    if (dstResolved.isExternal && context.requestConfirmation) {
      const confirmed = await context.requestConfirmation(
        `Tool wants to move file to external system path: "${dstPath}"`
      );
      if (!confirmed) {
        return { success: false, output: 'Owner rejected moving file to external path.' };
      }
    }

    try {
      if (existsSync(dstPath) && statSync(dstPath).isDirectory()) {
        dstPath = join(dstPath, basename(srcPath));
      }

      const dir = dirname(dstPath);
      if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

      renameSync(srcPath, dstPath);
      return {
        success: true,
        output: `Successfully moved "${srcPath}" to "${dstPath}".`,
      };
    } catch (err: any) {
      return { success: false, output: `Error moving file: ${err.message}` };
    }
  },
});

// ─── send_file ───────────────────────────────────────────────────────

toolRegistry.register({
  name: 'send_file',
  description: 'Send a file to the user through the current chat channel (WhatsApp document/media, Discord attachment, or WebUI download). Supports files anywhere on the machine.',
  category: 'channel',
  parameters: [
    { name: 'path', type: 'string', description: 'Path to the file to send (supports ~, documents, C:\\...)', required: true },
    { name: 'fileName', type: 'string', description: 'Display name for the file (optional)' },
  ],
  usageNotes: [
    'Use this only when the user wants the actual file delivered back into the chat.',
    'Do not use this just to confirm a file exists.'
  ],
  keywords: ['send', 'share', 'attach', 'upload', 'deliver', 'transfer', 'give', 'download', 'export', 'output'],
  handler: async (args, context): Promise<ToolResult> => {
    let filePath: string;
    try {
      filePath = resolveFlexiblePath(args.path, context.workingDir).absolute;
    } catch (err: any) {
      return { success: false, output: `Path error: ${err.message}` };
    }

    if (!existsSync(filePath)) {
      return { success: false, output: `File not found: ${filePath}` };
    }

    if (!context.sendFile) {
      return {
        success: false,
        output: 'File sending is not available in the current channel.',
      };
    }

    try {
      await context.sendFile(filePath, args.fileName);
      return {
        success: true,
        output: `File sent: ${basename(filePath)}`,
        filePath,
      };
    } catch (err: any) {
      return { success: false, output: `Error sending file: ${err.message}` };
    }
  },
});

// ─── Directory Search Helpers ────────────────────────────────────────

function searchDirectoryRecursive(
  dir: string,
  pattern: string,
  options: { recursive: boolean; maxDepth: number; limit: number; fileType: string },
  currentDepth: number,
  results: Array<{ path: string; isDir: boolean; size: number; mtime: Date }>
): void {
  if (currentDepth > options.maxDepth || results.length >= options.limit) return;

  const skipNames = new Set([
    '.git', 'node_modules', '$recycle.bin', 'system volume information',
    '.vscode', '.idea', 'dist', 'build', '.next', 'appdata', '.cache',
    '__pycache__', '.tmp', 'temp'
  ]);

  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }

  const isMatch = (name: string): boolean => {
    const lowered = name.toLowerCase();
    const patLower = pattern.toLowerCase();
    if (patLower === '*' || patLower === '*.*') return true;
    if (patLower.startsWith('*.')) {
      return lowered.endsWith(patLower.slice(1));
    }
    if (patLower.startsWith('.')) {
      return lowered.endsWith(patLower);
    }
    if (patLower.includes('*')) {
      const escaped = patLower.split('*').map(s => s.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')).join('.*');
      return new RegExp(`^${escaped}$`, 'i').test(name);
    }
    return lowered.includes(patLower);
  };

  for (const entry of entries) {
    if (results.length >= options.limit) break;
    const nameLower = entry.name.toLowerCase();
    if (skipNames.has(nameLower) && currentDepth > 0) continue;

    const fullPath = join(dir, entry.name);
    const isDir = entry.isDirectory();

    const matchesType =
      options.fileType === 'all' ||
      (options.fileType === 'directory' && isDir) ||
      (options.fileType === 'file' && !isDir);

    if (matchesType && isMatch(entry.name)) {
      try {
        const stat = statSync(fullPath);
        results.push({ path: fullPath, isDir, size: stat.size, mtime: stat.mtime });
      } catch {
        results.push({ path: fullPath, isDir, size: 0, mtime: new Date() });
      }
    }

    if (isDir && options.recursive && !skipNames.has(nameLower)) {
      searchDirectoryRecursive(fullPath, pattern, options, currentDepth + 1, results);
    }
  }
}

function collectTextFiles(
  dir: string,
  extFilter: string,
  files: string[],
  currentDepth: number,
  maxDepth: number,
  limit: number
): void {
  if (currentDepth > maxDepth || files.length >= limit) return;
  const skipNames = new Set(['.git', 'node_modules', 'dist', 'build', '.next', 'appdata']);

  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }

  for (const entry of entries) {
    if (files.length >= limit) break;
    const nameLower = entry.name.toLowerCase();
    if (skipNames.has(nameLower)) continue;

    const fullPath = join(dir, entry.name);
    if (entry.isDirectory()) {
      collectTextFiles(fullPath, extFilter, files, currentDepth + 1, maxDepth, limit);
    } else if (entry.isFile()) {
      if (!extFilter || nameLower.endsWith(extFilter)) {
        files.push(fullPath);
      }
    }
  }
}
