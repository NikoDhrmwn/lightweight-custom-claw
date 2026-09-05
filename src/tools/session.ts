/**
 * LiteClaw — Session Management & Introspection Tools
 *
 * Allows the agent to inspect token usage, view active sessions,
 * read cross-session conversation history, clear sessions,
 * and dispatch cross-session notifications.
 */

import { toolRegistry, ToolResult } from '../core/tools.js';
import { getMemoryStore } from '../core/memory.js';
import { channelRegistry } from '../channels/registry.js';
import { ownerRegistry } from '../core/owner.js';

// ─── Token Metrics & Context Introspection ───────────────────────────

toolRegistry.register({
  name: 'get_session_metrics',
  description: 'Inspect token consumption, message count, and context window limits for the current or specified session.',
  category: 'session',
  parameters: [
    {
      name: 'session_key',
      type: 'string',
      description: 'Optional session key (e.g. "whatsapp:12345@s.whatsapp.net"). Defaults to current session.',
      required: false,
    },
  ],
  keywords: ['tokens', 'token usage', 'metrics', 'context limit', 'context window', 'how many tokens', 'budget', 'usage'],
  handler: async (args, context): Promise<ToolResult> => {
    const memory = getMemoryStore();
    const sessionKey = args.session_key?.trim() || context.sessionKey;

    if (!sessionKey) {
      return {
        success: false,
        output: 'No session key provided or available in context.',
      };
    }

    const metrics = memory.getSessionMetrics(sessionKey);

    const report = [
      `📊 Session Metrics for "${sessionKey}":`,
      `• Messages in history: ${metrics.messageCount}`,
      `• Images processed: ${metrics.imageCount}`,
      `• Estimated history tokens: ${metrics.estimatedTokens.toLocaleString()}`,
      `• Max model context limit: ${(metrics.maxContextTokens ?? 64000).toLocaleString()} tokens`,
      `• Context budget: ${(metrics.budgetTokens ?? 51200).toLocaleString()} tokens`,
      `• Context budget used: ${metrics.usagePct ?? 0}%`,
      `• Soft compaction threshold: ${(metrics.softThresholdTokens ?? 46080).toLocaleString()} tokens (${metrics.compactionThresholdPct ?? 90}%)`,
      `• Compaction status: ${metrics.isNearCompaction ? '⚠️ Approaching compaction' : '✅ Healthy'}`,
      `• Last activity: ${metrics.lastActivity ? new Date(metrics.lastActivity).toLocaleString() : 'No activity recorded'}`,
    ].join('\n');

    return {
      success: true,
      output: report,
    };
  },
});

// ─── List All Sessions ───────────────────────────────────────────────

toolRegistry.register({
  name: 'list_sessions',
  description: 'List active sessions across WhatsApp, Discord, and WebUI, including message count, tokens, and last activity.',
  category: 'session',
  parameters: [
    {
      name: 'limit',
      type: 'number',
      description: 'Maximum number of sessions to return (default: 15).',
      required: false,
    },
    {
      name: 'channel',
      type: 'string',
      description: 'Filter sessions by channel type: "whatsapp", "discord", or "webui".',
      enum: ['whatsapp', 'discord', 'webui'],
      required: false,
    },
  ],
  keywords: ['list sessions', 'sessions', 'all sessions', 'active chats', 'chats', 'cross session'],
  handler: async (args): Promise<ToolResult> => {
    const memory = getMemoryStore();
    const limit = typeof args.limit === 'number' && args.limit > 0 ? args.limit : 15;
    const filterChannel = args.channel?.toLowerCase()?.trim();

    let sessions = memory.listSessions();

    if (filterChannel) {
      sessions = sessions.filter(s => s.sessionKey.toLowerCase().startsWith(filterChannel));
    }

    sessions = sessions.slice(0, limit);

    if (sessions.length === 0) {
      return {
        success: true,
        output: filterChannel
          ? `No active sessions found matching channel "${filterChannel}".`
          : 'No active sessions found in memory store.',
      };
    }

    const lines = [`📋 Active Sessions (${sessions.length}):`];
    for (const s of sessions) {
      const lastAct = s.lastActivity ? new Date(s.lastActivity).toLocaleString() : 'Never';
      const label = s.sessionName ? ` "${s.sessionName}"` : (s.userIdentifier ? ` (${s.userIdentifier})` : '');
      lines.push(`• [${s.sessionKey}]${label} — ${s.messageCount} msgs, ~${(s.estimatedTokens ?? 0).toLocaleString()} tokens, last: ${lastAct}`);
    }

    return {
      success: true,
      output: lines.join('\n'),
    };
  },
});

// ─── Cross-Session History Inspection ────────────────────────────────

toolRegistry.register({
  name: 'get_session_history',
  description: 'Retrieve recent message history from any session. Use to recall what a user discussed in another channel or group.',
  category: 'session',
  parameters: [
    {
      name: 'session_key',
      type: 'string',
      description: 'The session key to inspect (e.g. "whatsapp:12345@s.whatsapp.net" or "discord:channel_id").',
      required: true,
    },
    {
      name: 'limit',
      type: 'number',
      description: 'Number of recent messages to retrieve (default: 10, max: 30).',
      required: false,
    },
  ],
  keywords: ['session history', 'read session', 'other chat', 'previous chat', 'other session', 'inspect conversation'],
  handler: async (args): Promise<ToolResult> => {
    const memory = getMemoryStore();
    const sessionKey = args.session_key?.trim();

    if (!sessionKey) {
      return {
        success: false,
        output: 'session_key is required.',
      };
    }

    const limit = Math.min(Math.max(Number(args.limit) || 10, 1), 30);
    const history = memory.getHistory(sessionKey, limit);

    if (history.length === 0) {
      return {
        success: true,
        output: `No history found for session "${sessionKey}".`,
      };
    }

    const lines = [`📜 Recent History for "${sessionKey}" (${history.length} messages):`];
    for (const msg of history) {
      const time = new Date(msg.timestamp).toLocaleTimeString();
      const preview = msg.content.length > 200 ? msg.content.slice(0, 200) + '...' : msg.content;
      lines.push(`[${time}] ${msg.role}: ${preview}`);
    }

    return {
      success: true,
      output: lines.join('\n'),
    };
  },
});

// ─── Clear Session ───────────────────────────────────────────────────

toolRegistry.register({
  name: 'clear_session',
  description: 'Clear all conversation history, summaries, and task plans for the current or specified session.',
  category: 'session',
  requiresConfirmation: true,
  parameters: [
    {
      name: 'session_key',
      type: 'string',
      description: 'The session key to clear. If omitted, defaults to the current session.',
      required: false,
    },
  ],
  keywords: ['clear session', 'reset session', 'clear chat', 'reset chat', 'wipe memory', 'delete history', 'fresh start'],
  handler: async (args, context): Promise<ToolResult> => {
    const memory = getMemoryStore();
    const targetSession = args.session_key?.trim() || context.sessionKey;

    if (!targetSession) {
      return {
        success: false,
        output: 'No session key provided to clear.',
      };
    }

    memory.clearSession(targetSession);

    return {
      success: true,
      output: `🗑 Session "${targetSession}" conversation history and task plans have been completely cleared.`,
    };
  },
});

// ─── Delete Last Messages ────────────────────────────────────────────

toolRegistry.register({
  name: 'delete_last_messages',
  description: 'Undo/delete the last N messages from a session history.',
  category: 'session',
  parameters: [
    {
      name: 'count',
      type: 'number',
      description: 'Number of recent messages to delete (default: 1).',
      required: false,
    },
    {
      name: 'session_key',
      type: 'string',
      description: 'Optional target session key. Defaults to current session.',
      required: false,
    },
  ],
  keywords: ['delete last', 'undo message', 'remove last message', 'rollback chat'],
  handler: async (args, context): Promise<ToolResult> => {
    const memory = getMemoryStore();
    const targetSession = args.session_key?.trim() || context.sessionKey;
    const count = Math.max(Number(args.count) || 1, 1);

    if (!targetSession) {
      return {
        success: false,
        output: 'No session key provided.',
      };
    }

    const removed = memory.deleteLastMessages(targetSession, count);

    return {
      success: true,
      output: `Deleted ${removed} message(s) from session "${targetSession}".`,
    };
  },
});

// ─── Cross-Session Messaging ─────────────────────────────────────────

toolRegistry.register({
  name: 'send_message_to_session',
  description: 'Send a message or notification directly to another channel or session (e.g. WhatsApp, Discord).',
  category: 'session',
  requiresConfirmation: true,
  parameters: [
    {
      name: 'session_key',
      type: 'string',
      description: 'Target session key (e.g., "whatsapp:62812345678@s.whatsapp.net" or "discord:1234567890").',
      required: true,
    },
    {
      name: 'message',
      type: 'string',
      description: 'The text message to deliver to the target session.',
      required: true,
    },
  ],
  keywords: ['send to session', 'cross session message', 'message other channel', 'notify chat', 'forward message'],
  handler: async (args): Promise<ToolResult> => {
    const sessionKey = args.session_key?.trim();
    const message = args.message?.trim();

    if (!sessionKey || !message) {
      return {
        success: false,
        output: 'Both session_key and message are required.',
      };
    }

    const colonIndex = sessionKey.indexOf(':');
    if (colonIndex === -1) {
      return {
        success: false,
        output: `Invalid session_key "${sessionKey}". Expected format like "whatsapp:<jid>" or "discord:<channelId>".`,
      };
    }

    const channelType = sessionKey.slice(0, colonIndex);
    const channelTarget = sessionKey.slice(colonIndex + 1);

    const sent = await channelRegistry.sendMessage(channelType, channelTarget, message);
    if (!sent) {
      return {
        success: false,
        output: `Failed to deliver message to ${channelType} target "${channelTarget}". Ensure the channel is active.`,
      };
    }

    return {
      success: true,
      output: `Message successfully dispatched to ${channelType} session "${sessionKey}".`,
    };
  },
});

// ─── Owner Management Tools ──────────────────────────────────────────

toolRegistry.register({
  name: 'get_owner',
  description: 'View the currently registered owner(s) who hold confirmation authority.',
  category: 'session',
  parameters: [
    {
      name: 'channel_type',
      type: 'string',
      description: 'Channel type to query (e.g. "whatsapp", "discord"). Defaults to current channel.',
      required: false,
    },
  ],
  keywords: ['owner', 'who is owner', 'admin', 'operator', 'registered owner'],
  handler: async (args, context): Promise<ToolResult> => {
    const channel = args.channel_type?.toLowerCase() || context.channelType || 'whatsapp';
    const owners = ownerRegistry.getOwners(channel);

    if (owners.length === 0) {
      return {
        success: true,
        output: `👑 No registered owner found for channel "${channel}". The first direct message sender can register as owner using register_owner or /register-owner.`,
      };
    }

    const list = owners.map(o => `• ${o.displayName ? `${o.displayName} ` : ''}(${o.ownerId}) - registered on ${new Date(o.createdAtMs).toLocaleDateString()}`).join('\n');
    return {
      success: true,
      output: `👑 Registered Owners for ${channel}:\n${list}`,
    };
  },
});

toolRegistry.register({
  name: 'register_owner',
  description: 'Register a user as an authorized instance owner who can confirm destructive operations and external file modifications.',
  category: 'session',
  parameters: [
    {
      name: 'owner_id',
      type: 'string',
      description: 'Identifier / phone number / JID of the owner. Defaults to the current user.',
      required: false,
    },
    {
      name: 'display_name',
      type: 'string',
      description: 'Optional display name / label for the owner.',
      required: false,
    },
    {
      name: 'channel_type',
      type: 'string',
      description: 'Channel type (defaults to current channel, e.g. "whatsapp").',
      required: false,
    },
  ],
  keywords: ['register owner', 'make me owner', 'claim owner', 'set owner', 'authorize owner'],
  handler: async (args, context): Promise<ToolResult> => {
    const channel = args.channel_type?.toLowerCase() || context.channelType || 'whatsapp';
    const ownerId = args.owner_id?.trim() || context.channelTarget || '';
    const displayName = args.display_name?.trim() || 'Owner';

    if (!ownerId) {
      return {
        success: false,
        output: 'Cannot register owner: No user identifier or channel target available.',
      };
    }

    const primary = ownerRegistry.getPrimaryOwner(channel);
    if (!primary && !ownerRegistry.hasAnyOwner(channel)) {
      // First registration -> Absolute Owner!
      ownerRegistry.registerOwner(channel, ownerId, displayName, true);
      return {
        success: true,
        output: `👑 Successfully registered "${displayName}" (${ownerId}) as the Absolute Owner for ${channel}.`,
      };
    }

    // If caller is not the Absolute Owner, require Absolute Owner confirmation
    if (context.channelTarget && !ownerRegistry.isPrimaryOwner(channel, context.channelTarget)) {
      if (context.requestConfirmation) {
        const approved = await context.requestConfirmation(
          `User "${displayName}" (${ownerId}) wants to be registered as an authorized owner. Approve?`
        );
        if (!approved) {
          return {
            success: false,
            output: '❌ Registration rejected by the Absolute Owner.',
          };
        }
      } else {
        return {
          success: false,
          output: '⛔ Permission denied: Only the Absolute Owner can authorize new owner registrations.',
        };
      }
    }

    const success = ownerRegistry.registerOwner(channel, ownerId, displayName, false);
    if (!success) {
      return {
        success: false,
        output: `Failed to register owner "${ownerId}".`,
      };
    }

    return {
      success: true,
      output: `👑 Successfully registered "${displayName}" (${ownerId}) as an authorized owner for ${channel}.`,
    };
  },
});

