/**
 * LiteClaw — Cross-Session Search Tool (FTS5)
 *
 * Allows the agent to search through conversation history across all sessions or within the current session
 * using SQLite FTS5 full-text indexing.
 */

import { toolRegistry, ToolResult } from '../core/tools.js';
import { getMemoryStore } from '../core/memory.js';

toolRegistry.register({
  name: 'search_history',
  description: 'Search past conversation history across all sessions or the current session using full-text search. Useful for recalling previous discussions, user requests, decisions, or code discussed in past chats.',
  category: 'session',
  parameters: [
    {
      name: 'query',
      type: 'string',
      description: 'Search keywords, phrases, or topics to look for in past messages',
      required: true,
    },
    {
      name: 'limit',
      type: 'number',
      description: 'Maximum number of past message excerpts to return (default: 8)',
      required: false,
    },
    {
      name: 'currentSessionOnly',
      type: 'boolean',
      description: 'If true, searches only the current session. If false (default), searches across all past sessions.',
      required: false,
    },
  ],
  usageNotes: [
    'Use this when the user asks "what did we discuss about X?", "do you remember when we worked on Y?", or when needing context from past conversations.',
    'Returns matching message snippets with timestamps and session identifiers.',
  ],
  examples: [
    { userIntent: 'search what we did yesterday with ffmpeg', arguments: { query: 'ffmpeg storyboard' } },
    { userIntent: 'check past database config discussed', arguments: { query: 'postgres port' } },
  ],
  keywords: ['history', 'search', 'recall', 'past', 'conversation', 'previous', 'remember', 'messages'],
  handler: async (args, context): Promise<ToolResult> => {
    const query = String(args.query ?? '').trim();
    if (!query) {
      return { success: false, output: 'Missing search "query".' };
    }

    const limit = Math.max(1, Math.min(25, Number(args.limit) || 8));
    const memory = getMemoryStore();
    const sessionKey = args.currentSessionOnly ? context.sessionKey : undefined;

    try {
      const results = memory.searchFTS(query, limit, sessionKey);
      if (results.length === 0) {
        return {
          success: true,
          output: `No conversation history found matching "${query}".`,
        };
      }

      const formatted = results.map(r => {
        const dateStr = new Date(r.timestamp).toLocaleString();
        const snippet = r.content.length > 300 ? `${r.content.slice(0, 300)}...` : r.content;
        return `• [${dateStr}] [${r.role.toUpperCase()}] (Session: ${r.sessionKey}):\n  ${snippet.replace(/\n/g, '\n  ')}`;
      }).join('\n\n');

      return {
        success: true,
        output: `Found ${results.length} conversation match(es) for "${query}":\n\n${formatted}`,
      };
    } catch (err: any) {
      return { success: false, output: `Search failed: ${err.message}` };
    }
  },
});
