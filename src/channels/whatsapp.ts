/**
 * LiteClaw — WhatsApp Channel
 *
 * Uses @whiskeysockets/baileys (same as OpenClaw).
 * Handles QR pairing, session persistence, interactive buttons
 * for confirmations, and file/media sending.
 */

import makeWASocket, {
  useMultiFileAuthState,
  DisconnectReason,
  fetchLatestBaileysVersion,
  makeCacheableSignalKeyStore,
  type WASocket,
  type BaileysEventMap,
} from '@whiskeysockets/baileys';
import { Boom } from '@hapi/boom';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import { join, basename, extname } from 'path';
import { lookup } from 'mime-types';
import { AgentEngine, AgentRequest, AgentStreamEvent } from '../core/engine.js';
import { buildWhatsAppConfirmation, ConfirmationManager } from '../core/confirmation.js';
import { getConfig, getStateDir } from '../config.js';
import { createLogger, createSilentLogger } from '../logger.js';
import { printStepDone, printStepWarn, printStepError } from '../logger.js';
import { preprocessImage } from '../tools/vision.js';
import {
  applyEventToChannelProgress,
  buildOutgoingMessages,
  createChannelProgressState,
  formatDurationShort,
  formatProgressPreview,
  formatProgressStatusLabel,
  formatTaskStatusIcon,
  getProgressCounts,
  type ChannelProgressState,
} from './progress.js';
import { unfurlUrl, downloadUnfurledMedia } from './utils.js';
import { channelRegistry } from './registry.js';
import { parseScheduleTime } from '../core/scheduler.js';
import { processFile } from '../core/file_processor.js';
import { ownerRegistry } from '../core/owner.js';
import { visionService } from '../core/vision.js';
import { readMemoryFile } from '../core/personality_memory.js';

const log = createLogger('whatsapp');
const baileysLog = createSilentLogger('baileys');

interface MentionTarget {
  id: string;
  label: string;
  aliases: string[];
}

interface WhatsAppProgressState extends ChannelProgressState {
  currentTaskLabel?: string;
  error?: string;
  messageKey?: any;
  lastUpdateAt?: number;
  isCreatingTracker?: boolean;
}

interface MessageQueueItem {
  jid: string;
  content: any;
  options?: any;
  retries: number;
  resolve: (val: any) => void;
  reject: (err: any) => void;
}

export class WhatsAppChannel {
  private sock: WASocket | null = null;
  private engine: AgentEngine;
  private confirmations: ConfirmationManager;
  private config: any;
  private sessionDir: string;
  private progresses = new Map<string, WhatsAppProgressState>();
  private groupNameCache = new Map<string, string>();
  private contactNameCache = new Map<string, string>();
  private messageQueue: MessageQueueItem[] = [];
  private isProcessingQueue = false;
  private isReconnecting = false;

  constructor(engine: AgentEngine, confirmations: ConfirmationManager) {
    this.engine = engine;
    this.confirmations = confirmations;
    this.config = getConfig().channels?.whatsapp ?? {};
    this.sessionDir = join(getStateDir(), 'whatsapp-session');

    if (!existsSync(this.sessionDir)) {
      mkdirSync(this.sessionDir, { recursive: true });
    }

    channelRegistry.register('whatsapp', {
      sendMessage: async (target: string, content: string, options?: any) => {
        return this.sendMessageWithRetry(target, { text: content, ...options });
      },
      sendPoll: async (target: string, poll: { name: string; options: string[]; selectableCount?: number }) => {
        const sent = await this.sendMessageWithRetry(target, {
          poll: {
            name: poll.name,
            values: poll.options,
            selectableCount: poll.selectableCount ?? 1,
          },
        });
        return sent?.key?.id ?? '';
      },
      sendEvent: async (target: string, event: {
        name: string;
        description?: string;
        startDate: Date;
        endDate?: Date;
        location?: string;
        call?: 'audio' | 'video';
      }) => {
        const sent = await this.sendMessageWithRetry(target, {
          event: {
            name: event.name,
            description: event.description,
            startDate: event.startDate,
            endDate: event.endDate,
            location: event.location ? { name: event.location, degreesLatitude: 0, degreesLongitude: 0 } : undefined,
            call: event.call,
          },
        });
        return sent?.key?.id ?? '';
      },
      sendFile: async (target: string, filePath: string, fileName?: string) => {
        return this.sendFile(target, filePath, fileName);
      },
      react: async (target: string, messageKey: any, emoji: string) => {
        if (this.sock && messageKey) {
          await this.sock.sendMessage(target, { react: { text: emoji, key: messageKey } });
        }
      },
    });

    this.setupConfirmationHandler();
  }

  async start(): Promise<void> {
    if (this.isReconnecting) return;

    try {
      const { state, saveCreds } = await useMultiFileAuthState(this.sessionDir);
      const { version } = await fetchLatestBaileysVersion();

      this.sock = makeWASocket({
        version,
        auth: {
          creds: state.creds,
          keys: makeCacheableSignalKeyStore(state.keys, baileysLog as any),
        },
        logger: baileysLog as any,
        generateHighQualityLinkPreview: false,
        getMessage: async () => ({ conversation: '' }),
      });

      // Save credentials on update
      this.sock.ev.on('creds.update', saveCreds);

      // Handle connection events
      this.sock.ev.on('connection.update', (update) => {
        const { connection, lastDisconnect, qr } = update;

        if (qr) {
          printStepWarn('WhatsApp not linked — scan QR code to pair');
        }

        if (connection === 'close') {
          const reason = (lastDisconnect?.error as Boom)?.output?.statusCode;
          const shouldReconnect = reason !== DisconnectReason.loggedOut;

          log.info({ reason, shouldReconnect }, 'WhatsApp connection closed');

          if (shouldReconnect) {
            this.isReconnecting = true;
            printStepWarn(`WhatsApp disconnected (reason: ${reason}), reconnecting in 5s...`);
            setTimeout(() => {
              this.isReconnecting = false;
              this.start();
            }, 5000);
          } else {
            printStepError('WhatsApp logged out. Run: liteclaw channels login --channel whatsapp');
          }
        }

        if (connection === 'open') {
          this.isReconnecting = false;
          const me = (this.sock as any)?.authState?.creds?.me || {};
          log.info({
            id: this.sock?.user?.id,
            lid: (this.sock?.user as any)?.lid,
            credsMe: me
          }, 'WhatsApp connected - Identity Details');
          printStepDone('WhatsApp connected');
        }
      });

      // Handle incoming messages
      this.sock.ev.on('messages.upsert', async ({ messages, type }) => {
        if (type !== 'notify') return;

        for (const msg of messages) {
          const unwrapped = unwrapMessage(msg.message);
          if (!unwrapped) continue;

          // Check for text commands first
          if (await this.handleTextCommand(msg, unwrapped)) continue;

          await this.handleMessage(msg, unwrapped);
        }
      });

      // Handle poll vote updates
      this.sock.ev.on('messages.update', async (updates) => {
        for (const update of updates) {
          if (update.update?.pollUpdates) {
            log.info({ pollMsg: update.key?.id, updates: update.update.pollUpdates }, 'WhatsApp poll update received');
          }
        }
      });
    } catch (err: any) {
      log.error({ error: err.message }, 'Failed to start WhatsApp channel');
      this.isReconnecting = true;
      setTimeout(() => {
        this.isReconnecting = false;
        this.start();
      }, 10000);
    }
  }

  private async handleTextCommand(msg: any, messageContent: any): Promise<boolean> {
    const jid = msg.key.remoteJid!;
    let text = '';

    if (messageContent?.conversation) {
      text = messageContent.conversation;
    } else if (messageContent?.extendedTextMessage?.text) {
      text = messageContent.extendedTextMessage.text;
    }

    text = text.trim();
    if (!text.startsWith('/')) return false;

    const [cmd, ...args] = text.slice(1).split(/\s+/);
    const command = cmd.toLowerCase();
    const sessionKey = `whatsapp:${jid}`;

    log.info({ command, from: jid }, 'WhatsApp command received');

    switch (command) {
      case 'help':
        await this.sendMessageWithRetry(jid, {
          text: `*LiteClaw Commands*\n\n` +
                `*/reset* or */clear* - Clear conversation history for this chat\n` +
                `*/clear <session>* - Clear a specific session\n` +
                `*/clear all* - Clear all sessions\n` +
                `*/status* - Show agent and system uptime status\n` +
                `*/tokens* or */usage* - Show token consumption & context limits\n` +
                `*/sessions* - List all active sessions\n` +
                `*/owner* - View registered owner\n` +
                `*/register-owner* - Register yourself as the instance owner\n` +
                `*/poll <question> | <opt1> | <opt2> ...* - Send a native WhatsApp poll\n` +
                `*/event <title> | <time> | [desc]* - Send a native WhatsApp event card\n` +
                `*/remind <time> | <message>* - Schedule an autonomous reminder\n` +
                `*/retry* - Re-run the last turn with a fresh attempt\n` +
                `*/undo* - Revert the last conversation exchange\n` +
                `*/stop* - Immediately stop currently executing agent task\n` +
                `*/memory* - View persistent facts (MEMORY.md) and profile (USER.md)\n` +
                `*/search <query>* - Search full-text conversation history (FTS5)\n` +
                `*/insights [days]* - View token usage and activity metrics\n` +
                `*/tasks* or */kanban* - View active Kanban task board\n` +
                `*/help* - Show this message`
        });
        return true;

      case 'retry': {
        const lastUser = this.engine.getMemory().getLastUserMessage(sessionKey);
        if (!lastUser) {
          await this.sendMessageWithRetry(jid, { text: '⚠️ No previous turn to retry.' });
          return true;
        }
        this.engine.getMemory().undoLastExchange(sessionKey);
        await this.sendMessageWithRetry(jid, { text: `🔄 *Retrying turn:* "${lastUser.content.slice(0, 100)}..."` });
        await this.handleMessage(msg, { conversation: lastUser.content });
        return true;
      }

      case 'undo': {
        const result = this.engine.getMemory().undoLastExchange(sessionKey);
        if (result.removedCount === 0) {
          await this.sendMessageWithRetry(jid, { text: '⚠️ No previous exchange found to undo.' });
        } else {
          await this.sendMessageWithRetry(jid, {
            text: `↩️ *Undid last exchange* (${result.removedCount} messages removed).\nPrevious message was: "${(result.undoneUserMessage || '').slice(0, 120)}..."`
          });
        }
        return true;
      }

      case 'stop': {
        const stopped = this.engine.abortSession(sessionKey);
        if (stopped) {
          await this.sendMessageWithRetry(jid, { text: '⏹️ *Current agent turn stopped.*' });
        } else {
          await this.sendMessageWithRetry(jid, { text: 'ℹ️ No active agent task is currently running.' });
        }
        return true;
      }

      case 'memory': {
        const mem = readMemoryFile('memory');
        const usr = readMemoryFile('user');
        await this.sendMessageWithRetry(jid, {
          text: `🧠 *Persistent Agent Memory*\n\n` +
                `*👤 USER.md*\n${usr.slice(0, 1000) || '(empty)'}\n\n` +
                `*📝 MEMORY.md*\n${mem.slice(0, 1000) || '(empty)'}`
        });
        return true;
      }

      case 'search': {
        const q = args.join(' ').trim();
        if (!q) {
          await this.sendMessageWithRetry(jid, { text: 'Usage: `/search <query>`' });
          return true;
        }
        const matches = this.engine.getMemory().searchFTS(q, 5);
        if (matches.length === 0) {
          await this.sendMessageWithRetry(jid, { text: `🔍 No history found for "${q}".` });
          return true;
        }
        const formatted = matches.map(m => `• [${new Date(m.timestamp).toLocaleDateString()}] *${m.role}*: ${m.content.slice(0, 150)}...`).join('\n\n');
        await this.sendMessageWithRetry(jid, { text: `🔍 *History Results for "${q}":*\n\n${formatted}` });
        return true;
      }

      case 'insights': {
        const days = Math.max(1, Math.min(90, Number(args[0]) || 7));
        const stats = this.engine.getMemory().getUsageStats(days);
        const topSess = stats.topSessions.map(s => `• \`${s.sessionKey.slice(0, 20)}\`: ${s.messageCount} msgs (~${s.estimatedTokens.toLocaleString()} tokens)`).join('\n');
        await this.sendMessageWithRetry(jid, {
          text: `📊 *Agent Usage Insights (Last ${days} Days)*\n\n` +
                `💬 *Total Messages:* ${stats.totalMessages.toLocaleString()} (${stats.userMessages} user, ${stats.assistantMessages} bot)\n` +
                `🔑 *Active Sessions:* ${stats.totalSessions}\n` +
                `🪙 *Estimated Tokens:* ~${stats.estimatedTokens.toLocaleString()}\n\n` +
                `*Top Active Sessions:*\n${topSess || '(none)'}`
        });
        return true;
      }

      case 'tasks':
      case 'kanban': {
        const userKey = sessionKey.split(':')[0] || 'default';
        const boards = this.engine.getMemory().listKanbanBoards(userKey);
        if (boards.length === 0) {
          await this.sendMessageWithRetry(jid, { text: '📋 No Kanban boards found. Tell the agent "create a task board for X".' });
          return true;
        }
        const board = boards[0];
        const cards = this.engine.getMemory().listKanbanCards(board.id);
        const formatted = cards.slice(0, 10).map(c => `• [${c.columnName.toUpperCase()}] *${c.title}* ${c.priority ? `(${c.priority})` : ''}`).join('\n');
        await this.sendMessageWithRetry(jid, {
          text: `📋 *Kanban Board: ${board.name}*\n\n${formatted || '(empty board)'}`
        });
        return true;
      }

      case 'reset':
      case 'clear': {
        if (args[0] === 'all') {
          const sessions = this.engine.getMemory().listSessions();
          for (const s of sessions) {
            this.engine.getMemory().clearSession(s.sessionKey);
          }
          await this.sendMessageWithRetry(jid, { text: `🗑 *All ${sessions.length} sessions cleared.* Starting fresh.` });
          return true;
        } else if (args[0] && args[0].includes(':')) {
          const target = args[0].trim();
          this.engine.getMemory().clearSession(target);
          await this.sendMessageWithRetry(jid, { text: `🗑 *Session "${target}" cleared.*` });
          return true;
        } else {
          const metrics = this.engine.getMemory().getSessionMetrics(sessionKey);
          this.engine.getMemory().clearSession(sessionKey);
          await this.sendMessageWithRetry(jid, {
            text: `🗑 *History cleared for this chat.*\nFreed ~${metrics.estimatedTokens.toLocaleString()} tokens across ${metrics.messageCount} messages.`
          });
          return true;
        }
      }

      case 'status': {
        const metrics = this.engine.getMemory().getSessionMetrics(sessionKey);
        const uptime = process.uptime();
        await this.sendMessageWithRetry(jid, {
          text: `*LiteClaw Status*\n\n` +
                `🤖 *Agent:* ${this.config.agent?.name || 'Molty'}\n` +
                `⏳ *Uptime:* ${formatDurationShort(uptime * 1000)}\n` +
                `💬 *Messages:* ${metrics.messageCount}\n` +
                `🪙 *Tokens used:* ~${metrics.estimatedTokens.toLocaleString()}\n` +
                `📅 *Last activity:* ${metrics.lastActivity ? new Date(metrics.lastActivity).toLocaleString() : 'never'}`
        });
        return true;
      }

      case 'tokens':
      case 'usage': {
        const metrics = this.engine.getMemory().getSessionMetrics(sessionKey);
        await this.sendMessageWithRetry(jid, {
          text: `*📊 Session Token & Context Metrics*\n\n` +
                `🔑 *Session:* \`${sessionKey}\`\n` +
                `💬 *Messages:* ${metrics.messageCount} (Images: ${metrics.imageCount})\n` +
                `🪙 *Estimated Tokens:* ~${metrics.estimatedTokens.toLocaleString()}\n` +
                `📦 *Context Budget:* ~${(metrics.budgetTokens ?? 51200).toLocaleString()} tokens (Max: ${(metrics.maxContextTokens ?? 64000).toLocaleString()})\n` +
                `📈 *Budget Used:* ${metrics.usagePct ?? 0}%\n` +
                `⚡ *Soft Compaction:* ${(metrics.softThresholdTokens ?? 46080).toLocaleString()} tokens (${metrics.compactionThresholdPct ?? 90}%)\n` +
                `🛡 *Compaction Status:* ${metrics.isNearCompaction ? '⚠️ Near compaction threshold' : '✅ Healthy'}`
        });
        return true;
      }

      case 'sessions': {
        const sessions = this.engine.getMemory().listSessions().slice(0, 15);
        if (sessions.length === 0) {
          await this.sendMessageWithRetry(jid, { text: '📋 *No active sessions recorded.*' });
          return true;
        }
        const lines = [`*📋 Active Sessions (${sessions.length})*:\n`];
        for (const s of sessions) {
          const isCurrent = s.sessionKey === sessionKey ? ' 👈 _(here)_' : '';
          const user = s.userIdentifier ? ` (${s.userIdentifier})` : '';
          const last = s.lastActivity ? new Date(s.lastActivity).toLocaleDateString() : 'Never';
          lines.push(`• \`${s.sessionKey}\`${user}${isCurrent}\n  💬 ${s.messageCount} msgs | 🪙 ~${(s.estimatedTokens ?? 0).toLocaleString()} tok | 📅 ${last}`);
        }
        await this.sendMessageWithRetry(jid, { text: lines.join('\n') });
        return true;
      }

      case 'poll': {
        const rawArgs = text.slice(cmd.length + 1).trim();
        const parts = rawArgs.split('|').map(p => p.trim()).filter(Boolean);
        if (parts.length < 3) {
          await this.sendMessageWithRetry(jid, {
            text: '⚠️ *Usage:* `/poll Question | Option 1 | Option 2 | [Option 3]...`'
          });
          return true;
        }
        const [question, ...options] = parts;
        await this.sendMessageWithRetry(jid, {
          poll: {
            name: question,
            values: options.slice(0, 12),
            selectableCount: 1,
          }
        });
        return true;
      }

      case 'event': {
        const rawArgs = text.slice(cmd.length + 1).trim();
        const parts = rawArgs.split('|').map(p => p.trim()).filter(Boolean);
        if (parts.length < 2) {
          await this.sendMessageWithRetry(jid, {
            text: '⚠️ *Usage:* `/event Title | Start Time (e.g. tomorrow at 15:00) | [Description]`'
          });
          return true;
        }
        const title = parts[0];
        const timeMs = parseScheduleTime(parts[1]);
        if (!timeMs) {
          await this.sendMessageWithRetry(jid, {
            text: `⚠️ Could not parse time "${parts[1]}". Example: "in 2 hours", "tomorrow at 10am".`
          });
          return true;
        }
        const desc = parts[2] || undefined;
        await this.sendMessageWithRetry(jid, {
          event: {
            name: title,
            description: desc,
            startDate: new Date(timeMs),
          }
        });
        return true;
      }

      case 'remind': {
        const rawArgs = text.slice(cmd.length + 1).trim();
        const pipeIdx = rawArgs.indexOf('|');
        let timeStr = '';
        let reminderText = '';
        if (pipeIdx !== -1) {
          timeStr = rawArgs.slice(0, pipeIdx).trim();
          reminderText = rawArgs.slice(pipeIdx + 1).trim();
        } else {
          const parts = rawArgs.split(/\s+/);
          timeStr = parts[0];
          reminderText = parts.slice(1).join(' ');
        }
        const timeMs = parseScheduleTime(timeStr);
        if (!timeMs || !reminderText) {
          await this.sendMessageWithRetry(jid, {
            text: '⚠️ *Usage:* `/remind <time> | <message>` (e.g. `/remind in 15m | Check server`)'
          });
          return true;
        }
        const task = this.engine.getMemory().createScheduledTask({
          sessionKey,
          channelType: 'whatsapp',
          channelTarget: jid,
          triggerAtMs: timeMs,
          taskType: 'reminder',
          payload: reminderText,
        });
        await this.sendMessageWithRetry(jid, {
          text: `⏰ *Reminder scheduled* for ${new Date(timeMs).toLocaleString()}:\n"${reminderText}"\n(ID: \`${task.id}\`)`
        });
        return true;
      }

      case 'owner': {
        const primary = ownerRegistry.getPrimaryOwner('whatsapp');
        const owners = ownerRegistry.getOwners('whatsapp');
        if (owners.length === 0 && !primary) {
          await this.sendMessageWithRetry(jid, {
            text: '👑 *No owner registered yet.*\nSend `/register-owner` to register yourself as the Absolute Owner.'
          });
        } else {
          const lines = ['👑 *Authorized Owners (WhatsApp):*'];
          if (primary) {
            lines.push(`• 🌟 *Absolute Owner:* ${primary.displayName ? `*${primary.displayName}* ` : ''}(\`${primary.ownerId}\`)`);
          }
          const secondary = owners.filter(o => !primary || ownerRegistry.normalizeId(o.ownerId) !== ownerRegistry.normalizeId(primary.ownerId));
          for (const s of secondary) {
            lines.push(`• 🛡️ *Owner:* ${s.displayName ? `*${s.displayName}* ` : ''}(\`${s.ownerId}\`)`);
          }
          await this.sendMessageWithRetry(jid, { text: lines.join('\n') });
        }
        return true;
      }

      case 'register-owner': {
        const senderJid = msg.key.participant || jid;
        const senderName = msg.pushName || senderJid.split('@')[0];
        const primary = ownerRegistry.getPrimaryOwner('whatsapp');

        // Case 1: No owner registered yet -> sender becomes Absolute Owner
        if (!primary && !ownerRegistry.hasAnyOwner('whatsapp')) {
          ownerRegistry.registerOwner('whatsapp', senderJid, senderName, true);
          await this.sendMessageWithRetry(jid, {
            text: `👑 *Success!* You are now registered as the *Absolute Owner* of this LiteClaw instance (@${senderName} - \`${senderJid}\`).\n\n` +
                  `Any future \`/register-owner\` attempts by other users will require your approval via message.`
          });
          return true;
        }

        // Case 2: Sender is already Absolute Owner
        if (primary && ownerRegistry.isPrimaryOwner('whatsapp', senderJid)) {
          await this.sendMessageWithRetry(jid, {
            text: `👑 You are already the *Absolute Owner* (@${senderName}).`
          });
          return true;
        }

        // Case 3: Sender is already a registered secondary owner
        if (ownerRegistry.isOwner('whatsapp', senderJid)) {
          await this.sendMessageWithRetry(jid, {
            text: `🛡️ You are already a registered owner (@${senderName}).`
          });
          return true;
        }

        // Case 4: Other user requesting owner access -> require approval from the Absolute Owner!
        if (!primary) {
          await this.sendMessageWithRetry(jid, {
            text: `⚠️ Cannot request owner access: No Absolute Owner found.`
          });
          return true;
        }

        const primaryTarget = primary.ownerId.includes('@') ? primary.ownerId : `${primary.ownerId}@s.whatsapp.net`;

        await this.sendMessageWithRetry(jid, {
          text: `⏳ *Registration Pending Approval:*\n` +
                `Your request to become an authorized owner has been forwarded to the Absolute Owner (@${primary.displayName || primary.ownerId.split('@')[0]}). ` +
                `You will be notified once they respond.`
        });

        // Request approval from the Absolute Owner
        void (async () => {
          try {
            const approved = await this.confirmations.requestConfirmation(
              'register_owner',
              `User @${senderName} (${senderJid}) wants to be registered as an authorized owner.`,
              'whatsapp',
              primaryTarget,
              {
                requesterId: senderJid,
                requiredOwner: true,
                timeoutMs: 120_000,
              }
            );

            if (approved) {
              ownerRegistry.registerOwner('whatsapp', senderJid, senderName, false);
              await this.sendMessageWithRetry(jid, {
                text: `🎉 *Registration Approved!*\n` +
                      `The Absolute Owner has approved your request. You (@${senderName}) are now an authorized owner.`
              });
              if (primaryTarget !== jid) {
                await this.sendMessageWithRetry(primaryTarget, {
                  text: `✅ User @${senderName} (\`${senderJid}\`) has been successfully registered as an authorized owner.`
                });
              }
            } else {
              await this.sendMessageWithRetry(jid, {
                text: `❌ *Registration Denied:*\nThe Absolute Owner rejected or did not respond to your registration request.`
              });
            }
          } catch (err: any) {
            log.error({ error: err.message }, 'Error during owner registration approval flow');
          }
        })();

        return true;
      }

      default:
        return false;
    }
  }

  private async resolveWhatsAppSessionInfo(
    jid: string,
    pushName?: string
  ): Promise<{ sessionName: string; isGroup: boolean; groupSubject?: string }> {
    const isGroup = jid.endsWith('@g.us');
    if (isGroup) {
      let subject = this.groupNameCache.get(jid);
      if (!subject && this.sock) {
        try {
          const meta = await this.sock.groupMetadata(jid);
          if (meta?.subject) {
            subject = meta.subject;
            this.groupNameCache.set(jid, subject);
            if (meta.participants) {
              for (const p of meta.participants) {
                const phone = p.id.split('@')[0].split(':')[0];
                if (!this.contactNameCache.has(phone)) {
                  this.contactNameCache.set(phone, phone);
                }
              }
            }
          }
        } catch (e: any) {
          log.warn({ jid, error: e.message }, 'Failed to fetch WhatsApp group metadata');
        }
      }
      const displayName = subject || `Group (${jid.split('@')[0]})`;
      return {
        sessionName: `WhatsApp > ${displayName}`,
        isGroup: true,
        groupSubject: subject,
      };
    }

    const userLabel = pushName || jid.split('@')[0].replace('@s.whatsapp.net', '');
    return {
      sessionName: `WhatsApp DM > ${userLabel}`,
      isGroup: false,
    };
  }

  private async handleMessage(msg: any, messageContent: any): Promise<void> {
    const jid = msg.key.remoteJid!;

    // Check allow policy
    const allowFrom = this.config.allowFrom ?? ['*'];
    if (!allowFrom.includes('*')) {
      const phone = jid.replace('@s.whatsapp.net', '');
      if (!allowFrom.some((pattern: string) => phone.includes(pattern))) {
        return;
      }
    }

    // Extract text content
    let content = '';

    if (messageContent?.conversation) {
      content = messageContent.conversation;
    } else if (messageContent?.extendedTextMessage?.text) {
      content = messageContent.extendedTextMessage.text;
    } else if (messageContent?.imageMessage?.caption) {
      content = messageContent.imageMessage.caption;
    }

    const isGroup = jid.endsWith('@g.us');
    const senderJid = msg.key.participant || jid;
    const senderPhone = senderJid.split('@')[0].split(':')[0];
    const senderLabel = msg.pushName || senderPhone;

    // Cache sender contact name
    if (msg.pushName) {
      this.contactNameCache.set(senderPhone, msg.pushName);
    }

    // If this is a direct message and no owner is registered yet, auto-register as owner
    if (!isGroup && !ownerRegistry.hasAnyOwner('whatsapp')) {
      ownerRegistry.registerOwner('whatsapp', senderJid, senderLabel);
      log.info({ senderJid, senderLabel }, 'Auto-registered first DM user as instance owner');
    }

    // Process incoming documents & media attachments
    const attachments: Array<{ name: string; dataUrl: string }> = [];
    const incomingDir = join(getStateDir(), 'incoming');
    const docMsg = messageContent?.documentMessage || messageContent?.documentWithCaptionMessage?.message?.documentMessage;

    if (docMsg) {
      try {
        if (!existsSync(incomingDir)) mkdirSync(incomingDir, { recursive: true });
        const fileName = docMsg.fileName || `document_${Date.now()}`;
        if (docMsg.caption) content = docMsg.caption;

        const buffer = await this.downloadMedia(msg);
        if (buffer) {
          const localPath = join(incomingDir, `${Date.now()}_${fileName}`);
          writeFileSync(localPath, buffer);

          const mime = docMsg.mimetype || lookup(fileName) || 'application/octet-stream';
          const dataUrl = `data:${mime};base64,${buffer.toString('base64')}`;
          attachments.push({ name: fileName, dataUrl });

          let extractedText = '';
          try {
            const processed = await processFile(fileName, dataUrl);
            extractedText = processed.content;
          } catch (e: any) {
            extractedText = `[Error parsing document: ${e.message}]`;
          }

          const docHeader = `📎 [Received Document: "${fileName}" saved to: ${localPath}]`;
          const docBody = extractedText ? `\n\n--- FILE: ${fileName} ---\n${extractedText}\n--- END FILE ---` : '';
          content = content ? `${content}\n\n${docHeader}${docBody}` : `${docHeader}${docBody}`;
        }
      } catch (err: any) {
        log.warn({ error: err.message }, 'Failed to download WhatsApp document');
      }
    }

    if (messageContent?.audioMessage) {
      try {
        if (!existsSync(incomingDir)) mkdirSync(incomingDir, { recursive: true });
        const buffer = await this.downloadMedia(msg);
        if (buffer) {
          const audioPath = join(incomingDir, `${Date.now()}_voice_note.ogg`);
          writeFileSync(audioPath, buffer);
          const audioHeader = `🎤 [Voice Note received and saved to: ${audioPath}]`;
          content = content ? `${content}\n\n${audioHeader}` : audioHeader;
        }
      } catch (err: any) {
        log.warn({ error: err.message }, 'Failed to download WhatsApp audio');
      }
    }

    // Process incoming image with Florence-2 Large
    if (messageContent?.imageMessage) {
      try {
        const buffer = await this.downloadMedia(msg);
        if (buffer) {
          try {
            const visionResult = await visionService.analyzeImage(buffer);
            content = content ? `${content}\n\n${visionResult.formattedContext}` : visionResult.formattedContext;
          } catch (e: any) {
            log.warn({ error: e.message }, 'Failed to analyze image with Florence-2');
          }
        }
      } catch (err: any) {
        log.warn({ error: err.message }, 'Failed to download WhatsApp image');
      }
    }

    // Process incoming sticker with Florence-2 Large
    if (messageContent?.stickerMessage) {
      try {
        const buffer = await this.downloadMedia(msg);
        if (buffer) {
          try {
            const visionResult = await visionService.analyzeImage(buffer);
            content = content ? `${content}\n\n${visionResult.formattedContext}` : visionResult.formattedContext;
          } catch (e: any) {
            log.warn({ error: e.message }, 'Failed to describe sticker with Florence-2');
          }
        }
      } catch (err: any) {
        log.warn({ error: err.message }, 'Failed to download WhatsApp sticker');
      }
    }

    // Process incoming video / animated GIF with Florence-2 Large
    if (messageContent?.videoMessage) {
      try {
        const buffer = await this.downloadMedia(msg);
        if (buffer) {
          try {
            const visionResult = await visionService.analyzeImage(buffer);
            content = content ? `${content}\n\n${visionResult.formattedContext}` : visionResult.formattedContext;
          } catch (e: any) {
            log.warn({ error: e.message }, 'Failed to analyze video/GIF with Florence-2');
          }
        }
      } catch (err: any) {
        log.warn({ error: err.message }, 'Failed to download WhatsApp video/GIF');
      }
    }

    // Extract tags & mentions
    const contextInfo =
      messageContent?.extendedTextMessage?.contextInfo ??
      messageContent?.imageMessage?.contextInfo ??
      messageContent?.videoMessage?.contextInfo ??
      messageContent?.documentMessage?.contextInfo;

    const mentionedJids: string[] = contextInfo?.mentionedJid ?? [];
    const taggedUsers: string[] = [];
    for (const mJid of mentionedJids) {
      const phone = mJid.split('@')[0].split(':')[0];
      const knownName = this.contactNameCache.get(phone);
      const tag = knownName ? `@${knownName} (${phone})` : `@${phone}`;
      if (!taggedUsers.includes(tag)) taggedUsers.push(tag);
      if (knownName && content) {
        const phoneRegex = new RegExp(`@${phone}\\b`, 'g');
        content = content.replace(phoneRegex, `@${knownName} (${phone})`);
      }
    }

    const mentionTargets = this.extractMentionTargets(msg, messageContent);
    const replyContext = this.extractReplyContext(messageContent);
    const wasMentioned = didMentionMe(messageContent, this.sock?.user?.id);
    const sessionInfo = await this.resolveWhatsAppSessionInfo(jid, msg.pushName);

    const images = await this.collectIncomingImages(msg, messageContent);

    if (!content && images.length === 0 && attachments.length === 0) return;

    log.info({
      from: jid.replace('@s.whatsapp.net', ''),
      contentLength: content.length,
      hasImages: images.length > 0,
      hasAttachments: attachments.length > 0,
    }, 'WhatsApp message received');

    // Build session key
    const sessionKey = `whatsapp:${jid}`;

    const request: AgentRequest = {
      message: buildStructuredIncomingMessage(
        {
          conversationLabel: sessionInfo.groupSubject
            ? `group: "${sessionInfo.groupSubject}"`
            : (sessionInfo.isGroup ? `group: ${jid.split('@')[0]}` : `DM: ${msg.pushName || jid.split('@')[0]}`),
          sender: {
            id: msg.key.participant || jid,
            label: msg.pushName || jid.split('@')[0],
            name: msg.pushName || jid.split('@')[0],
            jid: msg.key.participant || jid,
          },
          isGroupChat: sessionInfo.isGroup,
          wasMentioned,
          mentionTargets,
          taggedUsers,
          replyContext,
        },
        content || '(attachment received)'
      ),
      images: images.length > 0 ? images : undefined,
      attachments: attachments.length > 0 ? attachments : undefined,
      sessionKey,
      sessionName: sessionInfo.sessionName,
      isGroup: sessionInfo.isGroup,
      channelType: 'whatsapp',
      channelTarget: jid,
      userIdentifier: msg.pushName || jid.split('@')[0],
      messageKey: msg.key,
      sendFile: async (filePath: string, fileName?: string) => {
        await this.sendFile(jid, filePath, fileName);
      },
      sendInteractiveChoice: async (choiceReq) => {
        const sent = await this.sendMessageWithRetry(jid, {
          poll: {
            name: choiceReq.prompt,
            values: choiceReq.options,
            selectableCount: 1,
          },
        });
        return sent?.key?.id ?? '';
      },
      sendPoll: async (poll) => {
        const sent = await this.sendMessageWithRetry(jid, {
          poll: {
            name: poll.name,
            values: poll.options,
            selectableCount: poll.selectableCount ?? 1,
          },
        });
        return sent?.key?.id ?? '';
      },
      sendEvent: async (event) => {
        const sent = await this.sendMessageWithRetry(jid, {
          event: {
            name: event.name,
            description: event.description,
            startDate: event.startDate,
            endDate: event.endDate,
            location: event.location ? { name: event.location, degreesLatitude: 0, degreesLongitude: 0 } : undefined,
            call: event.call,
          },
        });
        return sent?.key?.id ?? '';
      },
      react: async (emoji) => {
        if (this.sock) {
          await this.sock.sendMessage(jid, { react: { text: emoji, key: msg.key } });
        }
      },
    };

    // Ignore group chat messages unless mentioned or replied to (but save to memory for context)
    const isGroupChat = jid.endsWith('@g.us');
    if (isGroupChat) {
      const selfJidRaw = this.sock?.user?.id || '';
      const selfLid = (this.sock?.user as any)?.lid || (this.sock as any)?.authState?.creds?.me?.lid || '';
      const selfJid = selfJidRaw.split(':')[0] + '@s.whatsapp.net';
      const myName = this.config.agent?.name || this.sock?.user?.name || 'Molty';

      const namePattern = new RegExp(`\\b${escapeRegex(myName)}\\b`, 'i');
      const isMentioned = didMentionMe(messageContent, selfJid) ||
                          didMentionMe(messageContent, selfJidRaw) ||
                          (selfLid && didMentionMe(messageContent, selfLid)) ||
                          content.toLowerCase().includes(`@${myName.toLowerCase()}`) ||
                          namePattern.test(content); // Handle informal mentions like "oi molty"

      const isReplyToMe = messageContent?.extendedTextMessage?.contextInfo?.participant === selfJid ||
                          messageContent?.extendedTextMessage?.contextInfo?.participant === selfJidRaw ||
                          (selfLid && normalizeJid(messageContent?.extendedTextMessage?.contextInfo?.participant || '') === normalizeJid(selfLid));

      log.info({
        isGroupChat, isMentioned, isReplyToMe,
        selfJid, selfLid, myName,
        exactText: content,
        mentionedJids: messageContent?.extendedTextMessage?.contextInfo?.mentionedJid
      }, 'WhatsApp group filter check VERY VERBOSE');

      if (!isMentioned && !isReplyToMe) {
        this.engine.saveMessageSilent(request);
        return;
      }
    }

    // Send read receipts if configured
    if (this.config.sendReadReceipts) {
      await this.sock?.readMessages([msg.key]);
    }

    // ── Continuous composing presence ──
    // WhatsApp shows "typing..." when we update presence
    const sendComposing = async () => {
      try {
        await this.sock?.sendPresenceUpdate('composing', jid);
      } catch { /* ignore */ }
    };

    await sendComposing();
    const typingInterval = setInterval(sendComposing, 5_000);

    // Process and accumulate response
    let fullContent = '';
    const showProgress = this.config.showToolProgress ?? true;
    let progress: WhatsAppProgressState | undefined;

    if (showProgress) {
      progress = createWhatsAppProgressState();
      this.progresses.set(sessionKey, progress);
    }

    try {
      for await (const event of this.engine.processRequest(request)) {
        if (progress) {
          applyEventToWhatsAppProgress(progress, event);
          await this.updateProgress(jid, progress);
        }

        switch (event.type) {
          case 'content':
            fullContent += event.content ?? '';
            break;
          case 'error':
            if (!progress) fullContent += `\n⚠ Error: ${event.error}`;
            break;
        }
      }

      // Final progress update
      if (progress) {
        progress.status = progress.status === 'error' ? 'error' : 'done';
        // Final update to the tracker (Outcome will show a preview)
        await this.updateProgress(jid, progress, fullContent);
        this.progresses.delete(sessionKey);
      }

      // Always send the full response as a separate message for readability
      if (fullContent.trim()) {
        await this.sendResponse(jid, fullContent, [], mentionTargets);
      }

      log.info({
        to: jid.replace('@s.whatsapp.net', ''),
        responseLength: fullContent.length,
      }, 'WhatsApp turn completed');

    } catch (err: any) {
      log.error({ error: err.message }, 'WhatsApp message handling error');
      await this.sendMessageWithRetry(jid, { text: `⚠ Error: ${err.message}` });
    } finally {
      clearInterval(typingInterval);
      // Clear composing state
      try {
        await this.sock?.sendPresenceUpdate('paused', jid);
      } catch { /* ignore */ }
    }
  }

  private async updateProgress(jid: string, progress: WhatsAppProgressState, finalContent?: string): Promise<void> {
    if (!this.sock) return;

    // Throttle updates to avoid rate limits (max 1 update per 1.5s)
    const now = Date.now();
    const isFinal = progress.status === 'done' || progress.status === 'error';
    if (!isFinal && progress.lastUpdateAt && now - progress.lastUpdateAt < 1500) {
      return;
    }
    progress.lastUpdateAt = now;

    const text = buildWhatsAppProgressMessage(progress, finalContent);

    try {
      if (!progress.messageKey) {
        if (progress.isCreatingTracker) return;
        progress.isCreatingTracker = true;

        try {
          const sent = await this.sendMessageWithRetry(jid, { text });
          progress.messageKey = sent?.key;
        } finally {
          progress.isCreatingTracker = false;
        }
      } else {
        await this.sendMessageWithRetry(jid, { edit: progress.messageKey, text });
      }
    } catch (err: any) {
      log.warn({ error: err.message }, 'Failed to update WhatsApp progress');
    }
  }

  private async sendMessageWithRetry(jid: string, content: any, options?: any): Promise<any> {
    return new Promise((resolve, reject) => {
      this.messageQueue.push({ jid, content, options, retries: 0, resolve, reject });
      void this.processQueue();
    });
  }

  private async processQueue(): Promise<void> {
    if (this.isProcessingQueue || !this.sock || this.messageQueue.length === 0) return;
    this.isProcessingQueue = true;

    try {
      while (this.messageQueue.length > 0) {
        const item = this.messageQueue.shift()!;
        try {
          const result = await this.sock.sendMessage(item.jid, item.content, item.options);
          item.resolve(result);
        } catch (err: any) {
          if (item.retries < 3) {
            item.retries++;
            log.warn({ error: err.message, retry: item.retries }, 'Message send failed, retrying...');
            this.messageQueue.unshift(item);
            await new Promise(r => setTimeout(r, 1000 * item.retries));
          } else {
            log.error({ error: err.message }, 'Message send failed after retries');
            item.reject(err);
          }
        }
        // Small delay between messages to be safe
        await new Promise(r => setTimeout(r, 500));
      }
    } finally {
      this.isProcessingQueue = false;
    }
  }

  private extractReplyContext(messageContent: any): string | null {
    const contextInfo =
      messageContent?.extendedTextMessage?.contextInfo ??
      messageContent?.imageMessage?.contextInfo ??
      messageContent?.videoMessage?.contextInfo ??
      messageContent?.documentMessage?.contextInfo;

    const quoted = contextInfo?.quotedMessage;
    if (!quoted) return null;

    const participant = contextInfo?.participant || contextInfo?.remoteJid || 'Quoted user';
    const quotedText =
      quoted?.conversation ??
      quoted?.extendedTextMessage?.text ??
      quoted?.imageMessage?.caption ??
      quoted?.videoMessage?.caption ??
      quoted?.documentMessage?.caption ??
      '(quoted media)';

    return quotedText ? formatReplyContext(participant, quotedText) : null;
  }

  private async sendResponse(
    jid: string,
    content: string,
    toolUpdates: string[],
    mentionTargets: MentionTarget[]
  ): Promise<void> {
    if (!this.sock) return;

    let finalContent = content;

    // Extract and send URLs as media if they contain gifs/videos
    const urls = content.match(/https?:\/\/[^\s<)\]]+/g);
    if (urls) {
      const uniqueUrls = [...new Set(urls)];
      for (const url of uniqueUrls) {
        try {
          const mediaUrl = await unfurlUrl(url);
          if (mediaUrl) {
            const downloaded = await downloadUnfurledMedia(mediaUrl);
            if (downloaded) {
              const { buffer, mimeType } = downloaded;
              let sentMedia = false;
              if (mimeType.startsWith('image/')) {
                await this.sendMessageWithRetry(jid, { image: buffer, mimetype: mimeType });
                sentMedia = true;
              } else if (mimeType.startsWith('video/')) {
                await this.sendMessageWithRetry(jid, { video: buffer, mimetype: mimeType });
                sentMedia = true;
              }
              
              if (sentMedia) {
                const esc = url.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                finalContent = finalContent.replace(new RegExp(`\\[[^\\]]*\\]\\(${esc}\\)`, 'g'), '');
                finalContent = finalContent.replace(new RegExp(esc, 'g'), '');
              }
            }
          }
        } catch (e) {
          log.warn({ url, error: String(e) }, 'Failed to unfurl/send media for WhatsApp');
        }
      }
    }

    finalContent = finalContent.trim();
    if (!finalContent && toolUpdates.length === 0) {
      return; // If the message was entirely a link/gif, don't send an empty text message
    }

    const messages = buildOutgoingMessages(finalContent, toolUpdates, {
      replyStyle: this.config.replyStyle ?? 'single',
      showToolProgress: this.config.showToolProgress ?? false,
      maxLen: 4000,
      format: 'whatsapp',
    });

    for (const chunk of messages) {
      const resolved = resolveWhatsAppMentions(chunk, mentionTargets, this.contactNameCache);
      await this.sendMessageWithRetry(jid, {
        text: resolved.content,
        mentions: resolved.jids,
      });
    }
  }

  private extractMentionTargets(msg: any, messageContent: any): MentionTarget[] {
    const jid = msg.key.remoteJid!;
    const contextInfo =
      messageContent?.extendedTextMessage?.contextInfo ??
      messageContent?.imageMessage?.contextInfo ??
      messageContent?.videoMessage?.contextInfo ??
      messageContent?.documentMessage?.contextInfo;

    const targets: MentionTarget[] = [];
    const senderJid = msg.key.participant || jid;
    const senderPhone = senderJid.split('@')[0].split(':')[0];
    const senderLabel = msg.pushName || senderPhone;
    targets.push(createMentionTarget(senderJid, senderLabel, senderPhone, msg.pushName));

    const quotedParticipant = contextInfo?.participant;
    if (quotedParticipant) {
      const phone = quotedParticipant.split('@')[0].split(':')[0];
      const knownName = this.contactNameCache.get(phone);
      targets.push(
        createMentionTarget(
          quotedParticipant,
          knownName || phone,
          phone,
          knownName
        )
      );
    }

    const mentioned = contextInfo?.mentionedJid ?? [];
    for (const mentionedJid of mentioned) {
      const phone = mentionedJid.split('@')[0].split(':')[0];
      const knownName = this.contactNameCache.get(phone);
      targets.push(
        createMentionTarget(
          mentionedJid,
          knownName || phone,
          phone,
          knownName
        )
      );
    }

    return dedupeMentionTargets(targets);
  }

  private async sendFile(jid: string, filePath: string, fileName?: string): Promise<void> {
    if (!this.sock || !existsSync(filePath)) return;

    const name = fileName ?? basename(filePath);
    const ext = extname(filePath).toLowerCase();
    const mimeType = lookup(ext) || 'application/octet-stream';
    const buffer = readFileSync(filePath);

    // Determine send type based on mime
    if (mimeType.startsWith('image/')) {
      await this.sock.sendMessage(jid, {
        image: buffer,
        caption: `📎 ${name}`,
        mimetype: mimeType,
      });
    } else if (mimeType.startsWith('video/')) {
      await this.sock.sendMessage(jid, {
        video: buffer,
        caption: `📎 ${name}`,
        mimetype: mimeType,
      });
    } else if (mimeType.startsWith('audio/')) {
      await this.sock.sendMessage(jid, {
        audio: buffer,
        mimetype: mimeType,
      });
    } else {
      // Send as document
      await this.sock.sendMessage(jid, {
        document: buffer,
        fileName: name,
        mimetype: mimeType,
      });
    }

    log.info({ file: name, jid }, 'Sent file via WhatsApp');
  }

  private async collectIncomingImages(msg: any, messageContent: any): Promise<string[]> {
    const images: string[] = [];

    if (messageContent?.imageMessage) {
      const current = await this.downloadMessageImage(msg, 'message');
      if (current) images.push(current);
    }

    const quotedImage = await this.downloadQuotedReplyImage(messageContent);
    if (quotedImage) images.push(quotedImage);

    return images;
  }

  private async downloadMessageImage(msg: any, reason: string): Promise<string | null> {
    try {
      const stream = await this.downloadMedia(msg);
      if (!stream) return null;
      return await preprocessImage(stream);
    } catch (err: any) {
      log.warn({ error: err.message, reason }, 'Failed to download WhatsApp image');
      return null;
    }
  }

  private async downloadQuotedReplyImage(messageContent: any): Promise<string | null> {
    const contextInfo =
      messageContent?.extendedTextMessage?.contextInfo ??
      messageContent?.imageMessage?.contextInfo ??
      messageContent?.videoMessage?.contextInfo ??
      messageContent?.documentMessage?.contextInfo;

    const quotedImage = contextInfo?.quotedMessage?.imageMessage;
    if (!quotedImage) return null;

    try {
      const { downloadContentFromMessage } = await import('@whiskeysockets/baileys');
      const stream = await downloadContentFromMessage(quotedImage, 'image');
      const chunks: Buffer[] = [];
      for await (const chunk of stream) {
        chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
      }
      if (chunks.length === 0) return null;
      return await preprocessImage(Buffer.concat(chunks));
    } catch (err: any) {
      log.warn({ error: err.message }, 'Failed to download quoted WhatsApp image');
      return null;
    }
  }

  private async downloadMedia(msg: any): Promise<Buffer | null> {
    try {
      const { downloadMediaMessage } = await import('@whiskeysockets/baileys');
      const buffer = await downloadMediaMessage(msg, 'buffer', {}) as Buffer;
      return buffer;
    } catch (err: any) {
      log.warn({ error: err.message }, 'Media download failed');
      return null;
    }
  }

  private setupConfirmationHandler(): void {
    this.confirmations.on('confirmation_request', async (conf) => {
      if (conf.channelType !== 'whatsapp' || !conf.channelTarget || !this.sock) return;

      const jid = conf.channelTarget;

      try {
        // Try interactive buttons first
        const payload = buildWhatsAppConfirmation(conf);
        await this.sendMessageWithRetry(jid, {
          text: `${payload.text}\n\nReply *yes* to confirm or *no* to cancel.`,
        } as any);

        // Set up a temporary listener for the response
        const handler = async ({ messages, type }: any) => {
          if (type !== 'notify') return;
          for (const respMsg of messages) {
            if (respMsg.key.remoteJid !== jid || respMsg.key.fromMe) continue;
            const text = (respMsg.message?.conversation ??
              respMsg.message?.extendedTextMessage?.text ?? '').toLowerCase().trim();

            const isYes = ['yes', 'y', 'confirm', '✅'].includes(text);
            const isNo = ['no', 'n', 'cancel', '❌'].includes(text);

            if (!isYes && !isNo) continue;

            const senderJid = respMsg.key.participant || respMsg.key.remoteJid || '';

            // Security check: verify owner status if required
            if (conf.requiredOwner) {
              if (conf.toolName === 'register_owner') {
                const isAbsOwner = ownerRegistry.isPrimaryOwner('whatsapp', senderJid);
                if (!isAbsOwner) {
                  log.warn({ senderJid }, 'Non-absolute owner attempted to approve register_owner');
                  await this.sendMessageWithRetry(jid, {
                    text: `⛔ *Permission Denied:*\nOnly the *Absolute Owner* can approve new owner registrations.`,
                  });
                  continue;
                }
              } else {
                const isOwnerUser = ownerRegistry.isOwner('whatsapp', senderJid);
                if (!isOwnerUser) {
                  log.warn({ senderJid, tool: conf.toolName }, 'Non-owner attempted confirmation');
                  await this.sendMessageWithRetry(jid, {
                    text: `⛔ *Permission Denied:*\nOnly an authorized owner can confirm \`${conf.toolName}\`.`,
                  });
                  continue;
                }
              }
            }

            if (isYes) {
              this.confirmations.resolveConfirmation(conf.id, true);
              this.sock?.ev.off('messages.upsert', handler);
              await this.sendMessageWithRetry(jid, { text: `✅ *Confirmed by owner.* Proceeding with \`${conf.toolName}\`...` });
            } else if (isNo) {
              this.confirmations.resolveConfirmation(conf.id, false);
              this.sock?.ev.off('messages.upsert', handler);
              await this.sendMessageWithRetry(jid, { text: `❌ *Cancelled by owner.*` });
            }
          }
        };

        this.sock.ev.on('messages.upsert', handler);

        // Auto-remove handler after timeout
        setTimeout(() => {
          this.sock?.ev.off('messages.upsert', handler);
        }, conf.timeoutMs);

      } catch (err: any) {
        log.error({ error: err.message }, 'Failed to send WhatsApp confirmation');
      }
    });
  }

  stop(): void {
    channelRegistry.unregister('whatsapp');
    this.sock?.end(undefined);
  }
}

// ─── Utilities ───────────────────────────────────────────────────────

function buildStructuredIncomingMessage(
  meta: {
    conversationLabel: string;
    sender: Record<string, string | undefined>;
    isGroupChat: boolean;
    wasMentioned: boolean;
    mentionTargets: MentionTarget[];
    taggedUsers?: string[];
    replyContext?: string | null;
  },
  content: string
): string {
  // Compact context header — one line for LLM context, won't clutter WebUI
  const senderLabel = meta.sender.name || meta.sender.label || 'unknown';
  const senderHandle = meta.sender.jid?.split('@')[0] || senderLabel;
  const chatType = meta.isGroupChat ? 'group' : 'DM';

  const parts: string[] = [
    `[context: whatsapp | ${chatType} | ${meta.conversationLabel} | sender: ${senderLabel} (${senderHandle})]`,
    `[bot mentioned: ${meta.wasMentioned ? 'yes' : 'no'}]`,
  ];

  if (meta.taggedUsers && meta.taggedUsers.length > 0) {
    parts.push(`[tagged users: ${meta.taggedUsers.join(', ')}]`);
  }

  if (meta.mentionTargets.length > 0) {
    const handles = meta.mentionTargets
      .slice(0, 8)
      .map(t => `@${t.aliases[0]} (${t.label})`)
      .join(', ');
    parts.push(`[participants: ${handles}]`);
  }

  if (meta.replyContext) {
    parts.push(meta.replyContext);
  }

  parts.push('', content);
  return parts.join('\n');
}

function formatReplyContext(author: string, content: string): string {
  return `[Reply context]\n${author}: ${content}\n[/Reply context]`;
}

// Normalize JIDs: strip device suffix (e.g., "123:10")
// and handle @s.whatsapp.net, @c.us, and @lid consistently
function normalizeJid(id: string): string {
  if (!id) return '';
  // Strip domain and device IDs
  return id.split(':')[0].split('@')[0];
}

function didMentionMe(messageContent: any, selfId?: string): boolean {
  if (!selfId) return false;

  const contextInfo =
    messageContent?.extendedTextMessage?.contextInfo ??
    messageContent?.imageMessage?.contextInfo ??
    messageContent?.videoMessage?.contextInfo ??
    messageContent?.documentMessage?.contextInfo;

  const mentioned: string[] = contextInfo?.mentionedJid ?? [];

  const normalizedSelf = normalizeJid(selfId);
  const matched = mentioned.some(m => normalizeJid(m) === normalizedSelf);

  if (!matched && mentioned.length > 0) {
    log.debug({
      selfId,
      normalizedSelf,
      mentioned: mentioned.map(m => `${m} -> ${normalizeJid(m)}`)
    }, 'No ID match in mentions');
  }

  return matched;
}

/**
 * Unwrap message content from Baileys wrappers (ephemeral, viewOnce, etc.)
 */
function unwrapMessage(msg: any): any {
  if (!msg) return null;

  if (msg.ephemeralMessage) return unwrapMessage(msg.ephemeralMessage.message);
  if (msg.viewOnceMessage) return unwrapMessage(msg.viewOnceMessage.message);
  if (msg.viewOnceMessageV2) return unwrapMessage(msg.viewOnceMessageV2.message);
  if (msg.viewOnceMessageV2Extension) return unwrapMessage(msg.viewOnceMessageV2Extension.message);

  return msg;
}

function createMentionTarget(id: string, ...labels: Array<string | undefined>): MentionTarget {
  const aliases = labels
    .flatMap(label => label ? buildNameAliases(label) : [])
    .filter(Boolean);
  return {
    id,
    label: labels.find(Boolean) ?? id,
    aliases: aliases.length > 0 ? aliases : [id],
  };
}

function dedupeMentionTargets(targets: MentionTarget[]): MentionTarget[] {
  const merged = new Map<string, MentionTarget>();
  for (const target of targets) {
    const existing = merged.get(target.id);
    if (!existing) {
      merged.set(target.id, target);
      continue;
    }
    existing.aliases = Array.from(new Set([...existing.aliases, ...target.aliases]));
  }
  return Array.from(merged.values());
}

function buildNameAliases(label: string): string[] {
  const clean = label.replace(/^@+/, '').trim();
  if (!clean) return [];

  const aliases = new Set<string>();
  aliases.add(clean);
  aliases.add(clean.toLowerCase());
  aliases.add(clean.replace(/\s+/g, '_'));
  aliases.add(clean.replace(/\s+/g, ''));
  aliases.add(clean.replace(/[^\p{L}\p{N}_ ]/gu, '').trim());
  aliases.add(clean.replace(/[^\p{L}\p{N}_ ]/gu, '').replace(/\s+/g, '_').trim());
  return Array.from(aliases).filter(Boolean);
}

function resolveWhatsAppMentions(
  text: string,
  targets: MentionTarget[],
  contactCache?: Map<string, string>
): { content: string; jids: string[] } {
  let content = text;
  const jids = new Set<string>();

  // 1. Detect explicit @phoneNumber (e.g. @628123456789)
  const phonePattern = /(^|[^\w])@(\d{7,16})(?=$|[^\w])/g;
  let match: RegExpExecArray | null;
  while ((match = phonePattern.exec(content)) !== null) {
    const phone = match[2];
    jids.add(`${phone}@s.whatsapp.net`);
  }

  // 2. Resolve known mentionTargets (aliases -> @phone)
  for (const target of targets) {
    const phone = target.id.split('@')[0].split(':')[0];
    const sortedAliases = [...target.aliases].sort((a, b) => b.length - a.length);
    for (const alias of sortedAliases) {
      const escaped = escapeRegex(alias.replace(/^@+/, ''));
      if (/^\d+$/.test(escaped)) continue; // Handled by digits regex
      const pattern = new RegExp(`(^|[^\\w])@${escaped}(?=$|[^\\w])`, 'giu');
      if (pattern.test(content)) {
        content = content.replace(pattern, (m, prefix) => {
          jids.add(`${phone}@s.whatsapp.net`);
          return `${prefix}@${phone}`;
        });
      }
    }
  }

  // 3. Resolve from contactCache (name -> @phone)
  if (contactCache) {
    for (const [phone, name] of contactCache.entries()) {
      if (!name || name === phone) continue;
      const escaped = escapeRegex(name.replace(/^@+/, ''));
      if (/^\d+$/.test(escaped)) continue;
      const pattern = new RegExp(`(^|[^\\w])@${escaped}(?=$|[^\\w])`, 'giu');
      if (pattern.test(content)) {
        content = content.replace(pattern, (m, prefix) => {
          jids.add(`${phone}@s.whatsapp.net`);
          return `${prefix}@${phone}`;
        });
      }
    }
  }

  return { content, jids: Array.from(jids) };
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function applyEventToWhatsAppProgress(progress: WhatsAppProgressState, event: AgentStreamEvent): void {
  applyEventToChannelProgress(progress, event, 'whatsapp');
}

function buildWhatsAppProgressMessage(progress: WhatsAppProgressState, finalContent?: string): string {
  const elapsedMs = Date.now() - progress.startedAt;
  const { completed, active, total, pending, failed } = getProgressCounts(progress);

  const lines = [
    `*LiteClaw · ${whatsappProgressStatusLabel(progress.status).toUpperCase()}*`,
    '',
    `📊 *Overview*`,
    `${whatsappProgressStatusLabel(progress.status)}  •  ${formatDurationShort(elapsedMs)}`,
    `${whatsappTaskStatusIcon('completed')} ${completed}/${total || 0} done  •  ${whatsappTaskStatusIcon('in_progress')} ${active} active`,
    `${whatsappTaskStatusIcon('pending')} ${pending} pending${failed ? `  •  ${whatsappTaskStatusIcon('failed')} ${failed} issue${failed === 1 ? '' : 's'}` : ''}`,
  ];

  if (progress.planSummary) {
    lines.push('', `🗺 *Plan*`, progress.planSummary);
  }

  if (progress.tasks.length > 0) {
    lines.push('', `📋 *Tasks*`);
    progress.tasks.slice(0, 10).forEach((task, i) => {
      lines.push(`${i + 1}. ${whatsappTaskStatusIcon(task.status)} ${task.title}${task.summary ? ` - ${task.summary}` : ''}`);
    });
  }

  if (progress.recentTools.length > 0 || progress.currentTaskLabel || progress.error) {
    lines.push('', `⚙ *Activity*`);
    if (progress.currentTaskLabel && progress.status !== 'done') {
      lines.push(`Focus: ${progress.currentTaskLabel}`);
    }
    progress.recentTools.forEach(tool => lines.push(`- ${tool}`));
    if (progress.error) {
      lines.push(`⚠ *Error:* ${progress.error}`);
    }
  }

  if (finalContent || progress.status === 'done' || progress.status === 'error') {
    lines.push('', `🏁 *Outcome*`);
    if (progress.status === 'error') {
      lines.push('_Failed to complete task._');
    } else {
      const preview = formatProgressPreview(finalContent, 'whatsapp', 500);
      lines.push(preview || '_Response sent below._');
    }
  }

  return lines.join('\n');
}

function createWhatsAppProgressState(): WhatsAppProgressState {
  return createChannelProgressState();
}

function whatsappProgressStatusLabel(status: WhatsAppProgressState['status']): string {
  return formatProgressStatusLabel(status, 'whatsapp');
}

function whatsappTaskStatusIcon(status: string): string {
  return formatTaskStatusIcon(status, 'whatsapp');
}
