/**
 * LiteClaw — Persistent Memory Tool
 *
 * Allows the agent to remember facts, user preferences, and project context across sessions,
 * persisted in MEMORY.md and USER.md.
 */

import { toolRegistry, ToolResult } from '../core/tools.js';
import {
  readMemoryFile,
  appendMemoryEntry,
  writeMemoryFile,
  searchMemoryFiles,
  MemoryTarget,
} from '../core/personality_memory.js';

toolRegistry.register({
  name: 'manage_memory',
  description: 'Manage persistent long-term memory across sessions. Use this to remember facts, user preferences, project context, or recall stored memories.',
  category: 'utility',
  parameters: [
    {
      name: 'action',
      type: 'string',
      description: 'The memory operation: "remember" (add new entry), "recall" (search memory), "view" (read whole file), or "update" (overwrite file content)',
      required: true,
      enum: ['remember', 'recall', 'view', 'update'],
    },
    {
      name: 'target',
      type: 'string',
      description: 'Target memory file: "memory" (general facts, decisions, system facts in MEMORY.md) or "user" (user identity, preferences, working habits in USER.md)',
      required: true,
      enum: ['memory', 'user'],
    },
    {
      name: 'content',
      type: 'string',
      description: 'The fact or note to save (for "remember"), full markdown content (for "update"), or search query (for "recall")',
      required: false,
    },
  ],
  usageNotes: [
    'Call remember when the user shares their name, preferred tools, workflows, or rules they want you to always follow.',
    'Call remember when key architectural or project decisions are made that should persist in future chats.',
    'Use recall when you need to search past recorded facts or user preferences.',
  ],
  examples: [
    { userIntent: 'remember that user prefers TypeScript over Python', arguments: { action: 'remember', target: 'user', content: 'Prefers TypeScript over Python for all backend tasks' } },
    { userIntent: 'remember database port', arguments: { action: 'remember', target: 'memory', content: 'Production Postgres port is 5433 on internal network' } },
    { userIntent: 'recall user preferences', arguments: { action: 'recall', target: 'user', content: 'TypeScript' } },
  ],
  keywords: ['memory', 'remember', 'recall', 'forget', 'preference', 'note', 'facts', 'user'],
  handler: async (args): Promise<ToolResult> => {
    const action = String(args.action ?? 'recall').toLowerCase();
    const target = (String(args.target ?? 'memory').toLowerCase() === 'user' ? 'user' : 'memory') as MemoryTarget;
    const content = String(args.content ?? '').trim();

    try {
      if (action === 'remember') {
        if (!content) {
          return { success: false, output: 'Missing "content" to remember.' };
        }
        appendMemoryEntry(target, content);
        return {
          success: true,
          output: `Successfully saved to ${target === 'user' ? 'USER.md' : 'MEMORY.md'}:\n${content}`,
        };
      }

      if (action === 'view') {
        const fullContent = readMemoryFile(target);
        return {
          success: true,
          output: `=== ${target === 'user' ? 'USER.md' : 'MEMORY.md'} ===\n\n${fullContent || '(empty)'}`,
        };
      }

      if (action === 'update') {
        if (!content) {
          return { success: false, output: 'Missing "content" to update.' };
        }
        writeMemoryFile(target, content);
        return {
          success: true,
          output: `Successfully updated ${target === 'user' ? 'USER.md' : 'MEMORY.md'}.`,
        };
      }

      if (action === 'recall') {
        if (!content) {
          const fullContent = readMemoryFile(target);
          return {
            success: true,
            output: `=== ${target === 'user' ? 'USER.md' : 'MEMORY.md'} ===\n\n${fullContent || '(empty)'}`,
          };
        }
        const matches = searchMemoryFiles(content);
        if (matches.length === 0) {
          return {
            success: true,
            output: `No matching memory entries found for query: "${content}".`,
          };
        }
        const formatted = matches
          .map(m => `[${m.target.toUpperCase()} L${m.lineNum}] ${m.line}`)
          .join('\n');
        return {
          success: true,
          output: `Found ${matches.length} memory match(es):\n${formatted}`,
        };
      }

      return { success: false, output: `Unknown action: "${action}"` };
    } catch (err: any) {
      return { success: false, output: `Memory error: ${err.message}` };
    }
  },
});
