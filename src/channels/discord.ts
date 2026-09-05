/**
 * LiteClaw — Discord Channel
 * 
 * Features:
 * - Slash commands (/ask, /status, /clear, /help, /model)
 * - Emoji reaction progress (👀 → 🧠 → ⚙️ → ✅/❌)
 * - Dynamic bot status updates (reading files... → thinking... → idle)
 * - Button interactions for confirmations
 * - File attachment sending
 * - Formatted embed messages
 */

import {
  Client,
  GatewayIntentBits,
  Events,
  REST,
  Routes,
  SlashCommandBuilder,
  ActionRowBuilder,
  ButtonBuilder,
  ButtonStyle,
  EmbedBuilder,
  AttachmentBuilder,
  ActivityType,
  type Message,
  type ButtonInteraction,
  type ChatInputCommandInteraction,
  type InteractionEditReplyOptions,
  type AnySelectMenuInteraction,
  type ModalSubmitInteraction,
  type TextChannel,
  type ThreadChannel,
  type DMChannel,
  type User,
  type Guild,
  ChannelType,
  Partials,
} from 'discord.js';
import { existsSync } from 'fs';
import { lookup } from 'dns/promises';
import { basename, extname } from 'path';
import { AgentEngine, AgentRequest, AgentStreamEvent } from '../core/engine.js';
import { ConfirmationManager } from '../core/confirmation.js';
import { getConfig, getStateDir } from '../config.js';
import { resolveContextThresholds } from '../core/context.js';
import { createLogger } from '../logger.js';
import { preprocessImage } from '../tools/vision.js';
import { readMemoryFile } from '../core/personality_memory.js';
import type { InteractiveChoiceRequest } from '../core/tools.js';
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
import { DND_SLASH_COMMANDS, DndDiscordController } from '../dnd/discord.js';
import type { DndSessionDetails } from '../dnd/types.js';
import { VoicePipeline } from '../voice/voice-pipeline.js';

const log = createLogger('discord');

interface MentionTarget {
  id: string;
  label: string;
  aliases: string[];
}

type DiscordProgressState = ChannelProgressState;

// ─── Reaction Emojis (progress indicators) ──────────────────────────

const REACTIONS = {
  /** Message received, starting to process */
  RECEIVED: '👀',
  /** Agent is thinking / generating */
  THINKING: '🧠',
  /** Running a tool */
  TOOL: '⚙️',
  /** Reading a file */
  READING: '📖',
  /** Writing a file */
  WRITING: '✍️',
  /** Searching the web */
  SEARCHING: '🔍',
  /** Running a command */
  EXECUTING: '💻',
  /** Processing complete — success */
  DONE: '✅',
  /** Processing complete — error */
  ERROR: '❌',
  /** Waiting for confirmation */
  WAITING: '⏳',
} as const;

// Tool name → specific reaction emoji
const TOOL_REACTIONS: Record<string, string> = {
  read_file: REACTIONS.READING,
  write_file: REACTIONS.WRITING,
  delete_file: REACTIONS.WRITING,
  list_dir: REACTIONS.READING,
  send_file: REACTIONS.READING,
  exec: REACTIONS.EXECUTING,
  web_search: REACTIONS.SEARCHING,
  web_fetch: REACTIONS.SEARCHING,
};

// ─── Dynamic Status Messages ─────────────────────────────────────────

const STATUS_MESSAGES = {
  idle: [
    'Chilling 🦎',
    'Ready to help.',
    'Awaiting orders...',
    'Standing by.',
    'Idling...',
    'At your service.',
    'Listening...',
    'All systems nominal.',
    'Resting...',
    'On standby.',
    'Powered by {{MODEL}}.',
    'Let me know if you need anything.',
    'Watching the world go by...',
    'Zen mode 🧘',
    'Fingers on keyboard...',
    'Cogitating in the background...',
    'Nothing to do, nothing to worry about.',
    'Calm before the storm.',
    'Waiting patiently...',
    'Daydreaming about tokens...',
  ],
  thinking: [
    'Thinking...',
    'Processing your request...',
    'Working on it...',
    'Let me think about that...',
    'Crunching tokens...',
    'Pondering...',
    'Reasoning through this...',
    'Analyzing...',
    'Mulling it over...',
    'Connecting the dots...',
    'Deep in thought...',
    'Brainstorming...',
    'Generating response...',
    'Almost there...',
    'Weighing options...',
    'Cooking up a response...',
    'Assembling thoughts...',
    'Running inference...',
  ],
  reading: [
    'Reading files...',
    'Scanning contents...',
    'Inspecting a file...',
    'Looking through files...',
    'Opening a file...',
    'Parsing content...',
    'Reading source code...',
    'Browsing directories...',
    'Peeking at files...',
    'Digesting file contents...',
  ],
  writing: [
    'Writing files...',
    'Creating a file...',
    'Saving changes...',
    'Updating a file...',
    'Drafting content...',
    'Generating output...',
    'Committing to disk...',
    'Building something...',
    'Crafting code...',
    'Putting pen to paper...',
  ],
  searching: [
    'Searching the web...',
    'Googling that...',
    'Browsing the internet...',
    'Looking it up...',
    'Scouring the web...',
    'Fetching search results...',
    'Researching...',
    'Finding sources...',
    'Querying Google...',
    'Gathering intel...',
  ],
  executing: [
    'Running a command...',
    'Executing in terminal...',
    'Running a script...',
    'In the shell...',
    'Processing command...',
    'Terminal time...',
    'Launching process...',
    'Hacking away...',
    'Compiling...',
    'Running the thing...',
  ],
  confirming: [
    'Waiting for confirmation...',
    'Need your approval...',
    'Paused — awaiting OK...',
    'Hold on — confirmation needed...',
    'Permission required...',
    'Awaiting the green light...',
  ],
} as const;

function pickRandom(arr: readonly string[]): string {
  return arr[Math.floor(Math.random() * arr.length)];
}

function redactDiscordDebug(message: string): string {
  return String(message ?? '')
    .replace(/(Provided token:\s*)(\S+)/gi, (_whole, prefix, token) => `${prefix}${maskToken(token)}`)
    .replace(/([A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{20,})/g, token => maskToken(token));
}

function maskToken(token: string): string {
  if (!token) return token;
  if (token.length <= 10) return '*'.repeat(token.length);
  return `${token.slice(0, 6)}${'*'.repeat(Math.max(4, token.length - 10))}${token.slice(-4)}`;
}

function isDiscordDnsFailure(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? '');
  return /\bENOTFOUND\b/i.test(message) || /\bgetaddrinfo\b/i.test(message);
}

async function ensureDiscordGatewayReachable(): Promise<void> {
  try {
    await lookup('gateway.discord.gg');
  } catch (error: any) {
    throw new Error(`Could not resolve gateway.discord.gg (${error?.code ?? error?.message ?? 'lookup failed'})`);
  }
}

// ─── Slash Command Definitions ───────────────────────────────────────

const SLASH_COMMANDS = [
  new SlashCommandBuilder()
    .setName('ask')
    .setDescription('Ask LiteClaw a question or give it a task')
    .addStringOption(opt =>
      opt.setName('message')
        .setDescription('Your message to the agent')
        .setRequired(true)
    ),
  new SlashCommandBuilder()
    .setName('status')
    .setDescription('Show LiteClaw status and health'),
  new SlashCommandBuilder()
    .setName('clear')
    .setDescription('Clear the current session\'s conversation history'),
  new SlashCommandBuilder()
    .setName('help')
    .setDescription('Show LiteClaw commands and capabilities'),
  new SlashCommandBuilder()
    .setName('model')
    .setDescription('Show the current model and provider info'),
  new SlashCommandBuilder()
    .setName('tokens')
    .setDescription('Show current session token usage and compaction threshold'),
  new SlashCommandBuilder()
    .setName('question')
    .setDescription('Ask the GM an out-of-band question about the current DnD session')
    .addStringOption(opt =>
      opt.setName('message')
        .setDescription('Your question for the GM')
        .setRequired(true))
    .addStringOption(opt =>
      opt.setName('mode')
        .setDescription('Whether the GM answer should be private or visible to the table')
        .setRequired(false)
        .addChoices(
          { name: 'private', value: 'private' },
          { name: 'public', value: 'public' },
        )),
  new SlashCommandBuilder()
    .setName('voice')
    .setDescription('Manage real-time voice channel conversation')
    .addSubcommand(sub =>
      sub.setName('join')
        .setDescription('Join your current voice channel'))
    .addSubcommand(sub =>
      sub.setName('leave')
        .setDescription('Leave the voice channel')),
  new SlashCommandBuilder()
    .setName('retry')
    .setDescription('Re-run the last conversation turn with a fresh attempt'),
  new SlashCommandBuilder()
    .setName('undo')
    .setDescription('Revert the last user and assistant exchange'),
  new SlashCommandBuilder()
    .setName('stop')
    .setDescription('Immediately stop the currently running agent task in this channel'),
  new SlashCommandBuilder()
    .setName('memory')
    .setDescription('View persistent knowledge (MEMORY.md) and user profile (USER.md)'),
  new SlashCommandBuilder()
    .setName('search')
    .setDescription('Search past conversation history using SQLite FTS5')
    .addStringOption(opt =>
      opt.setName('query')
        .setDescription('Keywords to search for')
        .setRequired(true)),
  new SlashCommandBuilder()
    .setName('insights')
    .setDescription('View usage analytics, token consumption, and active sessions')
    .addIntegerOption(opt =>
      opt.setName('days')
        .setDescription('Number of days to analyze (default: 7)')
        .setRequired(false)),
  new SlashCommandBuilder()
    .setName('tasks')
    .setDescription('View active Kanban boards and cards'),
  ...DND_SLASH_COMMANDS,
];

// ─── Discord Channel Class ───────────────────────────────────────────

export class DiscordChannel {
  private client: Client;
  private engine: AgentEngine;
  private confirmations: ConfirmationManager;
  private dnd: DndDiscordController;
  private config: any;
  private voicePipelines = new Map<string, VoicePipeline>();
  private statusTimer: ReturnType<typeof setInterval> | null = null;
  private currentState: keyof typeof STATUS_MESSAGES = 'idle';
  private activeRequests = 0;
  private interactiveChoices = new Map<string, {
    prompt: string;
    options: string[];
    responses?: Record<string, string>;
    messageId: string;
    channelId: string;
    createdAt: number;
  }>();

  constructor(engine: AgentEngine, confirmations: ConfirmationManager) {
    this.engine = engine;
    this.confirmations = confirmations;
    this.config = getConfig().channels?.discord ?? {};

    this.client = new Client({
      intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.MessageContent,
        GatewayIntentBits.DirectMessages,
        GatewayIntentBits.GuildMessageReactions,
        GatewayIntentBits.GuildVoiceStates,
      ],
      partials: [
        Partials.Channel,
        Partials.Message,
        Partials.Reaction,
        Partials.User,
        Partials.GuildMember,
        Partials.ThreadMember
      ],
    });

    this.client.on('debug', m => console.log('[DISCORD_DEBUG]', redactDiscordDebug(m)));
    this.client.on('warn', m => console.warn('[DISCORD_WARN]', m));
    this.client.on('error', m => console.error('[DISCORD_ERROR]', m));
    this.client.on('shardError', (error, shardId) => {
      log.warn({ shardId, error: error.message }, 'Discord shard websocket error');
    });
    this.client.on('shardDisconnect', (event, shardId) => {
      log.warn({ shardId, code: event.code, reason: event.reason?.toString?.() ?? '' }, 'Discord shard disconnected');
    });
    this.client.on('shardReconnecting', shardId => {
      log.info({ shardId }, 'Discord shard reconnecting');
    });
    this.client.on('invalidated', () => {
      log.error('Discord session invalidated');
    });
    this.client.on('raw', (p: any) => {
      // Workaround for discord.js dropping uncached DMs
      if (p.t === 'MESSAGE_CREATE' && !p.d.guild_id && p.d.channel_id) {
        if (!this.client.channels.cache.has(p.d.channel_id)) {
          console.log(`[RAW/HYDRATE] Auto-hydrating missing DM channel: ${p.d.channel_id}`);
          try {
            // Force add a partial DM channel
            (this.client.channels as any)._add({
              id: p.d.channel_id,
              type: 1, // ChannelType.DM
              recipients: [p.d.author]
            }, null, { cache: true });
          } catch (e) {
            console.error('[RAW/HYDRATE] Failed to inject channel', e);
          }
        }
      }
    });

    this.dnd = new DndDiscordController(this.client, this.engine.getMemory());
    this.setupEventHandlers();
    this.setupConfirmationHandler();
  }

  // ─── Event Handlers ──────────────────────────────────────────────

  private setupEventHandlers(): void {
    this.client.once(Events.ClientReady, async (c) => {
      log.info({ user: c.user.tag }, 'Discord bot connected');
      console.log(`  ✓ Discord bot online as ${c.user.tag}`);

      // Register slash commands
      await this.registerSlashCommands(c.user.id);
      this.dnd.scheduleOpenVotes();

      // Set initial idle status
      this.setStatus('idle');

      // Rotate idle status every 60s when not busy
      this.statusTimer = setInterval(() => {
        if (this.currentState === 'idle') {
          this.setStatus('idle');
        }
      }, 60_000);
    });

    // Regular messages (mention or DM)
    this.client.on(Events.MessageCreate, async (message) => {
      console.log('=> RAW messageCreate event fired!', message.author?.tag, message.content?.length);
      await this.handleMessage(message);
    });

    // Slash command interactions
    this.client.on('interactionCreate', async (interaction) => {
      if (interaction.isChatInputCommand()) {
        await this.handleSlashCommand(interaction);
      } else if (interaction.isButton()) {
        await this.handleButtonInteraction(interaction as ButtonInteraction);
      } else if (interaction.isAnySelectMenu()) {
        await this.handleSelectMenuInteraction(interaction);
      } else if (interaction.isModalSubmit()) {
        await this.handleModalInteraction(interaction);
      }
    });
  }

  // ─── Dynamic Status ──────────────────────────────────────────────

  private setStatus(state: keyof typeof STATUS_MESSAGES): void {
    this.currentState = state;
    let statusText = pickRandom(STATUS_MESSAGES[state]);

    if (statusText.includes('{{MODEL}}')) {
      const modelId = this.engine.getLLMClient().getModelId();
      const modelName = modelId.split('/').pop() || modelId;
      // Format: deepseek-v4 -> Deepseek V4, gemma-4-e4b -> Gemma 4 E4b
      const formatted = modelName
        .split(/[-_]/)
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
      statusText = statusText.replace('{{MODEL}}', formatted);
    }

    const statusMap: Record<string, 'online' | 'idle' | 'dnd' | 'invisible'> = {
      idle: 'online',
      thinking: 'dnd',
      reading: 'dnd',
      writing: 'dnd',
      searching: 'dnd',
      executing: 'dnd',
      confirming: 'idle',
    };

    this.client.user?.setPresence({
      status: statusMap[state] ?? 'online',
      activities: [{
        name: statusText,
        type: ActivityType.Custom,
        state: statusText,
      }],
    });

    log.debug({ state, statusText }, 'Updated Discord status');
  }

  private beginRequest(initialState: keyof typeof STATUS_MESSAGES = 'thinking'): void {
    this.activeRequests++;
    this.setStatus(initialState);
  }

  private endRequest(): void {
    this.activeRequests = Math.max(0, this.activeRequests - 1);
    if (this.activeRequests === 0) {
      this.setStatus('idle');
    }
  }

  /**
   * Transition status based on agent event,
   * mapping tool names to specific activity states.
   */
  private updateStatusForEvent(eventType: string, toolName?: string): void {
    switch (eventType) {
      case 'thinking':
      case 'plan':
      case 'task_update':
        this.setStatus('thinking');
        break;
      case 'tool_start':
        if (toolName) {
          if (['read_file', 'list_dir', 'send_file'].includes(toolName)) {
            this.setStatus('reading');
          } else if (['write_file', 'delete_file'].includes(toolName)) {
            this.setStatus('writing');
          } else if (['web_search', 'web_fetch'].includes(toolName)) {
            this.setStatus('searching');
          } else if (toolName === 'exec') {
            this.setStatus('executing');
          } else {
            this.setStatus('thinking');
          }
        }
        break;
      case 'confirmation':
        this.setStatus('confirming');
        break;
    }
  }

  // ─── Reaction Progress ───────────────────────────────────────────

  /**
   * Add a reaction to the message. Silently fails on permission errors.
   */
  private async react(message: Message, emoji: string): Promise<void> {
    try {
      await message.react(emoji);
    } catch (err: any) {
      log.debug({ emoji, error: err.message }, 'Failed to react (missing permissions?)');
    }
  }

  /**
   * Remove a specific reaction the bot added.
   */
  private async unreact(message: Message, emoji: string): Promise<void> {
    try {
      const reaction = message.reactions.cache.find(r => r.emoji.name === emoji);
      if (reaction) {
        await reaction.users.remove(this.client.user!.id);
      } else {
        // Fallback: use raw REST API if cache is dead (happens in hydrated DMs)
        const emojiEncoded = encodeURIComponent(emoji);
        await (this.client as any).rest.delete(
          `/channels/${message.channelId}/messages/${message.id}/reactions/${emojiEncoded}/@me`
        );
      }
    } catch (err: any) {
      log.debug({ emoji, error: err.message }, 'Failed to remove reaction');
    }
  }

  /**
   * Get the appropriate reaction emoji for a tool.
   */
  private getToolReaction(toolName: string): string {
    return TOOL_REACTIONS[toolName] ?? REACTIONS.TOOL;
  }

  // ─── Slash Commands ──────────────────────────────────────────────

  private async registerSlashCommands(clientId: string): Promise<void> {
    try {
      const token = this.config.token ?? process.env.DISCORD_TOKEN;
      const rest = new REST({ version: '10' }).setToken(token);
      const commandData = SLASH_COMMANDS.map(cmd => cmd.toJSON());
      const configuredGuildId = this.config.guildId ?? process.env.DISCORD_GUILD_ID;
      if (configuredGuildId) {
        // Explicit guild mode: fast propagation in one guild, but no DM slash commands.
        await rest.put(Routes.applicationCommands(clientId), { body: [] });
        log.info('Cleared global slash commands because explicit guild-only registration is active');

        await rest.put(Routes.applicationGuildCommands(clientId, configuredGuildId), { body: commandData });
        log.info({ guildId: configuredGuildId, count: commandData.length }, 'Registered guild slash commands');
        console.log(`  ✓ Registered ${commandData.length} guild slash commands for ${configuredGuildId}`);
        console.log('  ℹ Guild-only slash commands are active; DM slash commands are disabled in this mode.');
      } else {
        // Default mode: global commands work in DMs and guilds without duplicate guild entries.
        await rest.put(Routes.applicationCommands(clientId), { body: commandData });
        log.info({ count: commandData.length }, 'Registered global slash commands');
        console.log(`  ✓ Registered ${commandData.length} global slash commands (may take up to 1h to show up)`);

        // Clear any previously configured guild commands to prevent duplicate slash entries.
        for (const guildId of this.client.guilds.cache.map(guild => guild.id)) {
          await rest.put(Routes.applicationGuildCommands(clientId, guildId), { body: [] });
          log.info({ guildId }, 'Cleared guild slash commands because global registration is active');
        }
      }
    } catch (err: any) {
      log.error({ error: err.message }, 'Failed to register slash commands');
      console.log(`  ⚠ Failed to register slash commands: ${err.message}`);
    }
  }

  private async handleSlashCommand(interaction: ChatInputCommandInteraction): Promise<void> {
    const dndResult = await this.dnd.handleCommand(interaction);
    if (dndResult === true) {
      return;
    }
    if (typeof dndResult === 'string') {
      // It's a roll result! Trigger the engine so the GM reacts.
      await this.processDndActionChoice(interaction, dndResult);
      return;
    }

    const dndSession = this.dnd.getSessionForThread(interaction.channelId);
    const inProtectedDndThread = Boolean(dndSession);
    const isDndPlayer = dndSession
      ? this.dnd.isPlayerInThread(interaction.channelId, interaction.user.id)
      : false;

    if (interaction.commandName === 'clear' && inProtectedDndThread) {
      await interaction.reply({
        content: 'This thread is protected as an active DnD session. `/clear` is disabled here.',
        ephemeral: true,
      });
      return;
    }

    if (inProtectedDndThread && !isDndPlayer) {
      await interaction.reply({
        content: 'Only enrolled DnD session players can use LiteClaw commands in this thread. Use `/dnd join` to join midway.',
        ephemeral: true,
      });
      return;
    }

    switch (interaction.commandName) {
      case 'ask':
        await this.handleAskCommand(interaction);
        break;
      case 'status':
        await this.handleStatusCommand(interaction);
        break;
      case 'clear':
        await this.handleClearCommand(interaction);
        break;
      case 'help':
        await this.handleHelpCommand(interaction);
        break;
      case 'model':
        await this.handleModelCommand(interaction);
        break;
      case 'tokens':
        await this.handleTokensCommand(interaction);
        break;
      case 'question':
        await this.handleQuestionCommand(interaction);
        break;
      case 'voice':
        await this.handleVoiceCommand(interaction);
        break;
      case 'retry':
        await this.handleRetryCommand(interaction);
        break;
      case 'undo':
        await this.handleUndoCommand(interaction);
        break;
      case 'stop':
        await this.handleStopCommand(interaction);
        break;
      case 'memory':
        await this.handleMemoryCommand(interaction);
        break;
      case 'search':
        await this.handleSearchCommand(interaction);
        break;
      case 'insights':
        await this.handleInsightsCommand(interaction);
        break;
      case 'tasks':
        await this.handleTasksCommand(interaction);
        break;
      default:
        await interaction.reply({ content: 'Unknown command.', ephemeral: true });
    }
  }

  private async handleAskCommand(interaction: ChatInputCommandInteraction): Promise<void> {
    const message = interaction.options.getString('message', true);
    const parsed = parseIncomingDiscordMentions(
      message,
      interaction.guild,
      this.client,
      this.client.user?.id
    );
    const mentionTargets = buildDiscordMentionTargetsFromInteraction(interaction);
    const dndSession = this.dnd.getSessionForThread(interaction.channelId);
    let dndRagContext = 'No relevant RAG context found for this session yet.';
    if (dndSession) {
      try {
        dndRagContext = await this.dnd.buildNarrativeRagContext(interaction.channelId, parsed.cleanContent);
      } catch (error: any) {
        log.warn({ error: error.message, channelId: interaction.channelId }, 'Failed to build DnD narrative RAG context');
      }
    }
    const rawMessage = dndSession
      ? this.dnd.buildTableTalkPrompt(interaction.channelId, interaction.user.id, parsed.cleanContent, dndRagContext)
      : parsed.cleanContent;
    const isGroup = Boolean(interaction.guildId);
    const sessionName = buildDiscordSessionName(interaction.channel, interaction.guild);

    const effectiveMessage = buildStructuredIncomingMessage(
      {
        platform: 'discord',
        conversationLabel: interaction.guild
          ? `Guild #${interaction.channel?.isTextBased() && 'name' in interaction.channel ? interaction.channel.name : interaction.channelId}`
          : 'Discord DM',
        sender: {
          id: interaction.user.id,
          label: interaction.user.tag,
          name: interaction.user.displayName ?? interaction.user.username,
          username: interaction.user.username,
        },
        isGroupChat: isGroup,
        wasMentioned: true,
        mentionTargets,
        taggedUsers: parsed.taggedUsers,
        taggedRoles: parsed.taggedRoles,
        taggedChannels: parsed.taggedChannels,
      },
      rawMessage
    );

    // Defer reply since processing may take a while
    await interaction.deferReply();

    const sessionKey = `discord:${interaction.channelId}`;
    const request: AgentRequest = {
      message: effectiveMessage,
      sessionKey,
      sessionName,
      isGroup,
      disablePlanner: Boolean(dndSession),
      channelType: 'discord',
      channelTarget: interaction.channelId,
      userIdentifier: interaction.user.tag,
      workingDir: this.config.workspace || getConfig().agent?.workspace || getStateDir(),
      sendInteractiveChoice: async (choiceRequest) => {
        return this.sendInteractiveChoice({
          channelId: interaction.channelId,
          replyTo: async (payload) => interaction.followUp(payload),
        }, choiceRequest);
      },
    };

    // For DnD sessions, use the dedicated GM system prompt and keep reasoning enabled
    if (dndSession) {
      const { readFileSync } = await import('fs');
      const { resolve } = await import('path');
      const gmPromptPath = resolve(process.cwd(), 'config/dnd_gm_prompt.md');
      if (existsSync(gmPromptPath)) {
        request.systemPromptOverride = readFileSync(gmPromptPath, 'utf-8');
      }
    }

    let fullContent = '';
    const toolUpdates: string[] = [];
      const progress = createDiscordProgressState();
    let lastProgressFlush = 0;

    const flushProgress = async (force = false, finalContent?: string): Promise<void> => {
      const now = Date.now();
      if (!force && now - lastProgressFlush < 1500) return;
      lastProgressFlush = now;

      const hasPlan = progress.tasks.length > 0;
      const payload: InteractionEditReplyOptions = {
        embeds: hasPlan ? [buildDiscordProgressEmbed(progress, finalContent)] : [],
      };

      if (finalContent !== undefined) {
        payload.content = finalContent || null;
      } else if (!hasPlan) {
        payload.content = `_${discordProgressStatusLabel(progress.status)}..._`;
      } else {
        payload.content = null;
      }

      await interaction.editReply(payload);
    };

    try {
      this.beginRequest('thinking');
      await flushProgress(true);
      for await (const event of this.engine.processRequest(request)) {
        this.updateStatusForEvent(event.type, event.toolName);
        applyEventToDiscordProgress(progress, event);

        switch (event.type) {
          case 'content':
            fullContent += event.content ?? '';
            break;
          case 'plan':
            toolUpdates.push(`🗺️ Planned ${event.plan?.tasks?.length ?? 0} task${(event.plan?.tasks?.length ?? 0) === 1 ? '' : 's'}`);
            break;
          case 'task_update': {
            const prefix = event.taskIndex && event.taskTotal
              ? `[${event.taskIndex}/${event.taskTotal}] `
              : '';
            if (event.taskStatus === 'in_progress') {
              toolUpdates.push(`→ ${prefix}${event.taskTitle}`);
            } else if (event.taskStatus) {
              const icon = event.taskStatus === 'completed' ? '✓' : event.taskStatus === 'blocked' ? '⚠' : '✗';
              toolUpdates.push(`${icon} ${prefix}${event.taskTitle}${event.taskSummary ? ` — ${event.taskSummary}` : ''}`);
            }
            break;
          }
          case 'tool_start':
            toolUpdates.push(`⚙ Running \`${event.toolName}\`...`);
            break;
          case 'tool_result':
            const icon = event.toolResult?.success ? '✓' : '✗';
            toolUpdates.push(`${icon} \`${event.toolName}\` ${event.toolResult?.success ? 'completed' : 'failed'}`);
            break;
          case 'error':
            fullContent += `\n⚠ Error: ${event.error}`;
            break;
        }

        await flushProgress();
      }

      const structuredNarrative = dndSession
        ? await this.dnd.processStructuredNarrativeResponse(interaction.channelId, fullContent)
        : { content: fullContent, shopEmbeds: [], combatEmbeds: [], combatComponents: null, actionComponents: null, rollComponents: null };

      progress.status = progress.error ? 'error' : 'done';

      if (dndSession) {
        // For DnD: send narrative as a rich embed, clear progress embed
        const narrativeEmbed = new EmbedBuilder()
          .setColor(0x8e44ad)
          .setDescription(structuredNarrative.content.slice(0, 4096));
        const trackerEmbed = this.dnd.buildTurnTrackerEmbed(interaction.channelId);

        await interaction.editReply({
          content: null,
          embeds: [
            narrativeEmbed,
            ...(trackerEmbed ? [trackerEmbed] : []),
            ...structuredNarrative.shopEmbeds,
            ...structuredNarrative.combatEmbeds,
          ],
          components: [
            ...(structuredNarrative.combatComponents || []),
            ...(structuredNarrative.actionComponents || []),
            ...(structuredNarrative.rollComponents || []),
          ].slice(0, 5),
        });
        const replyMessage = await interaction.fetchReply().catch(() => null);
        await this.dnd.recordCanonicalSceneState({
          threadId: interaction.channelId,
          sessionId: dndSession.session.id,
          source: 'narrative',
          title: narrativeEmbed.data.title ?? null,
          content: structuredNarrative.content,
          messageId: (replyMessage as any)?.id ?? null,
        });
      } else {
        // Non-DnD: standard text output
        const messages = buildOutgoingMessages(structuredNarrative.content, toolUpdates, {
          replyStyle: this.config.replyStyle ?? 'single',
          showToolProgress: this.config.showToolProgress ?? false,
          maxLen: 1900,
          format: 'discord',
        });

        const hasPlan = progress.tasks.length > 0;
        if (hasPlan) {
          await flushProgress(true);
          for (const msg of messages) {
            const resolved = resolveDiscordMentions(msg, mentionTargets, interaction.guild);
            await interaction.followUp({
              content: resolved.content,
              allowedMentions: {
                parse: ['users', 'roles'],
                users: resolved.userIds,
                roles: resolved.roleIds,
              },
            });
          }
        } else {
          const first = messages[0] ?? '(No response)';
          const resolvedFirst = resolveDiscordMentions(first, mentionTargets, interaction.guild);
          await flushProgress(true, resolvedFirst.content);
          for (let i = 1; i < messages.length; i++) {
            const resolved = resolveDiscordMentions(messages[i], mentionTargets, interaction.guild);
            await interaction.followUp({
              content: resolved.content,
              allowedMentions: {
                parse: ['users', 'roles'],
                users: resolved.userIds,
                roles: resolved.roleIds,
              },
            });
          }
        }
      }
    } catch (err: any) {
      log.error({ error: err.message }, 'Slash command /ask error');
      progress.status = 'error';
      progress.error = err.message;
      await flushProgress(true, `⚠ Error: ${err.message}`);
    } finally {
      this.endRequest();
    }
  }

  private async handleStatusCommand(interaction: ChatInputCommandInteraction): Promise<void> {
    const config = getConfig();
    const port = config.gateway?.port ?? 7860;

    let gatewayUptime: string | null = null;
    try {
      const res = await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) {
        const data = await res.json() as any;
        gatewayUptime = `${Math.floor(data.uptime / 60)}m`;
      }
    } catch { /* offline */ }

    const pending = this.confirmations.getPending().length;

    const embed = new EmbedBuilder()
      .setAuthor({ name: 'LiteClaw · Status' })
      .setColor(gatewayUptime ? 0x00897B : 0x757575)
      .setDescription([
        `**Gateway** · ${gatewayUptime ? `Online · uptime ${gatewayUptime}` : 'Offline'}`,
        `**State** · \`${this.currentState}\``,
        `**Requests** · ${this.activeRequests} active${pending > 0 ? ` · ${pending} pending confirmation${pending !== 1 ? 's' : ''}` : ''}`,
      ].join('\n'))
      .setTimestamp();

    await interaction.reply({ embeds: [embed] });
  }

  private async handleClearCommand(interaction: ChatInputCommandInteraction): Promise<void> {
    // Clear memory for this channel's session
    const { MemoryStore } = await import('../core/memory.js');
    const memory = new MemoryStore();
    const sessionKey = `discord:${interaction.channelId}`;
    memory.clearSession(sessionKey);
    memory.close();

    await interaction.reply({
      content: '🗑️ Session history cleared for this channel.',
      ephemeral: true,
    });
  }

  private async handleHelpCommand(interaction: ChatInputCommandInteraction): Promise<void> {
    const embed = new EmbedBuilder()
      .setAuthor({ name: 'LiteClaw · Help' })
      .setColor(0x5E35B1)
      .setDescription([
        '*Lightweight AI agent running locally. Mention me or use slash commands.*',
        '',
        '**Commands**',
        '`/ask` · Ask or task the agent',
        '`/status` · System status',
        '`/tokens` · Context window usage',
        '`/insights` · Usage analytics',
        '`/clear` · Clear session history',
        '`/memory` · View persistent memory',
        '`/search` · Search history',
        '`/model` · Current model info',
        '',
        '**Capabilities**',
        'File ops · Shell · Web search · Vision · Voice',
        '',
        '`@mention` me in any channel to chat.',
      ].join('\n'));

    await interaction.reply({ embeds: [embed] });
  }


  private async handleTokensCommand(interaction: ChatInputCommandInteraction): Promise<void> {
    const sessionKey = `discord:${interaction.channelId}`;
    const { MemoryStore } = await import('../core/memory.js');
    const memory = new MemoryStore();
    const metrics = memory.getSessionMetrics(sessionKey);
    const sessionInfo = memory.getSession(sessionKey);
    memory.close();

    const thresholds = resolveContextThresholds();
    const maxTokens = thresholds.maxContextTokens;
    const soft = thresholds.softThresholdTokens;
    const currentTokens = metrics.estimatedTokens;
    const percentage = Math.round((currentTokens / soft) * 100);

    let statusLabel = 'Healthy';
    let color = 0x00897B;
    if (percentage > 90) { statusLabel = 'Near limit'; color = 0xD50000; }
    else if (percentage > 75) { statusLabel = 'Moderate'; color = 0xFFAB00; }

    const barLength = 16;
    const filledCount = Math.round((percentage / 100) * barLength);
    const tokenBar = '█'.repeat(Math.min(filledCount, barLength)) + '░'.repeat(Math.max(0, barLength - filledCount));

    const channelName = interaction.channel && 'name' in interaction.channel && interaction.channel.name
      ? `#${interaction.channel.name}`
      : sessionInfo?.sessionName || `channel:${interaction.channelId}`;

    const embed = new EmbedBuilder()
      .setAuthor({ name: 'LiteClaw · Context Window' })
      .setColor(color)
      .setDescription([
        `\`${tokenBar}\` **${percentage}%** — ${statusLabel}`,
        `**${currentTokens.toLocaleString()}** / ${soft.toLocaleString()} tokens (max ${maxTokens.toLocaleString()})`,
        `${metrics.messageCount} messages · ${metrics.imageCount} images`,
        `Session: ${channelName}`,
      ].join('\n'));

    await interaction.reply({ embeds: [embed] });
  }

  private async handleRetryCommand(interaction: ChatInputCommandInteraction): Promise<void> {
    const sessionKey = `discord:${interaction.channelId}`;
    const channel = interaction.channel;
    const isGroup = Boolean(interaction.guild);
    const sessionName = buildDiscordSessionName(channel, interaction.guild);

    const lastUser = this.engine.getMemory().getLastUserMessage(sessionKey);
    if (!lastUser) {
      await interaction.reply({ content: '⚠️ No previous turn found to retry.', ephemeral: true });
      return;
    }
    this.engine.getMemory().undoLastExchange(sessionKey);
    await interaction.reply({ content: `🔄 Retrying turn: "${lastUser.content.slice(0, 100)}..."` });

    const req: AgentRequest = {
      sessionKey,
      sessionName,
      isGroup,
      message: lastUser.content,
      channelType: 'discord',
      channelTarget: interaction.channelId,
    };
    if (channel && 'send' in channel) {
      let output = '';
      for await (const event of this.engine.processRequest(req)) {
        if (event.type === 'content' && event.content) {
          output += event.content;
        }
      }
      if (output.trim()) {
        const resolved = resolveDiscordMentions(output.trim(), [], interaction.guild);
        await (channel as any).send({
          content: resolved.content,
          allowedMentions: {
            parse: ['users', 'roles'],
            users: resolved.userIds,
            roles: resolved.roleIds,
          },
        });
      }
    }
  }

  private async handleUndoCommand(interaction: ChatInputCommandInteraction): Promise<void> {
    const sessionKey = `discord:${interaction.channelId}`;
    const result = this.engine.getMemory().undoLastExchange(sessionKey);
    if (result.removedCount === 0) {
      await interaction.reply({ content: '⚠️ No previous exchange found to undo.', ephemeral: true });
    } else {
      const embed = new EmbedBuilder()
        .setAuthor({ name: 'LiteClaw · Undo' })
        .setColor(0x00897B)
        .setDescription(`↩️ Undid last exchange (${result.removedCount} messages removed).\n\nPrevious user prompt:\n> ${result.undoneUserMessage?.slice(0, 150) ?? ''}`);
      await interaction.reply({ embeds: [embed] });
    }
  }

  private async handleStopCommand(interaction: ChatInputCommandInteraction): Promise<void> {
    const sessionKey = `discord:${interaction.channelId}`;
    const stopped = this.engine.abortSession(sessionKey);
    await interaction.reply({
      content: stopped ? '⏹️ Successfully stopped the running agent task.' : 'ℹ️ No active task was running in this channel.',
      ephemeral: true,
    });
  }

  private async handleMemoryCommand(interaction: ChatInputCommandInteraction): Promise<void> {
    const mem = readMemoryFile('memory');
    const usr = readMemoryFile('user');
    const embed = new EmbedBuilder()
      .setAuthor({ name: 'LiteClaw · Persistent Memory' })
      .setColor(0x5E35B1)
      .addFields(
        { name: '👤 USER.md (User Profile)', value: usr.slice(0, 1000) || '(empty)' },
        { name: '📝 MEMORY.md (Facts & Knowledge)', value: mem.slice(0, 1000) || '(empty)' }
      );
    await interaction.reply({ embeds: [embed] });
  }

  private async handleSearchCommand(interaction: ChatInputCommandInteraction): Promise<void> {
    const query = interaction.options.getString('query', true);
    const matches = this.engine.getMemory().searchFTS(query, 5);
    if (matches.length === 0) {
      await interaction.reply({ content: `🔍 No past messages found matching "${query}".`, ephemeral: true });
      return;
    }
    const embed = new EmbedBuilder()
      .setAuthor({ name: `LiteClaw · History Search ("${query}")` })
      .setColor(0x00897B)
      .setDescription(
        matches.map(m => `• **${m.role.toUpperCase()}** (${new Date(m.timestamp).toLocaleDateString()}):\n${m.content.slice(0, 150)}...`).join('\n\n')
      );
    await interaction.reply({ embeds: [embed] });
  }

  private async handleInsightsCommand(interaction: ChatInputCommandInteraction): Promise<void> {
    const days = Math.max(1, Math.min(90, interaction.options.getInteger('days') ?? 7));
    const stats = this.engine.getMemory().getUsageStats(days);
    const topSess = stats.topSessions
      .slice(0, 5)
      .map(s => {
        const info = this.engine.getMemory().getSession(s.sessionKey);
        const name = info?.sessionName || s.sessionKey.replace(/^(whatsapp|discord|webui):/, '').slice(0, 32);
        return `\`${name}\` — ${s.messageCount} msgs · ~${(s.estimatedTokens ?? 0).toLocaleString()} tokens`;
      })
      .join('\n');
    const embed = new EmbedBuilder()
      .setAuthor({ name: `LiteClaw · Insights — Last ${days} days` })
      .setColor(0x3949AB)
      .addFields(
        { name: 'Messages', value: `${stats.totalMessages.toLocaleString()} total (${stats.userMessages} user · ${stats.assistantMessages} bot)`, inline: false },
        { name: 'Sessions', value: `${stats.totalSessions}`, inline: true },
        { name: 'Est. Tokens', value: `~${stats.estimatedTokens.toLocaleString()}`, inline: true },
        { name: 'Top Sessions', value: topSess || '(none)' }
      );
    await interaction.reply({ embeds: [embed] });
  }

  private async handleTasksCommand(interaction: ChatInputCommandInteraction): Promise<void> {
    const userKey = `discord:${interaction.user.id}`;
    const boards = this.engine.getMemory().listKanbanBoards(userKey);
    if (boards.length === 0) {
      await interaction.reply({ content: '📋 No Kanban boards found for you. Ask the bot to "create a task board for X".', ephemeral: true });
      return;
    }
    const board = boards[0];
    const cards = this.engine.getMemory().listKanbanCards(board.id);
    const formatted = cards.slice(0, 10).map(c => `• **[${c.columnName.toUpperCase()}]** ${c.title} ${c.priority ? `(${c.priority})` : ''}`).join('\n');
    const embed = new EmbedBuilder()
      .setAuthor({ name: `LiteClaw · Kanban: ${board.name}` })
      .setColor(0xFF9800)
      .setDescription(formatted || '(empty board)');
    await interaction.reply({ embeds: [embed] });
  }

  private async handleQuestionCommand(interaction: ChatInputCommandInteraction): Promise<void> {
    const dndDetails = this.dnd.getSessionForThread(interaction.channelId);
    if (!dndDetails) {
      await interaction.reply({
        content: '`/question` is only available inside a protected DnD session thread.',
        ephemeral: true,
      });
      return;
    }

    if (!this.dnd.isPlayerInThread(interaction.channelId, interaction.user.id)) {
      await interaction.reply({
        content: 'Only enrolled DnD session players can ask private GM questions in this thread.',
        ephemeral: true,
      });
      return;
    }

    const message = interaction.options.getString('message', true);
    const mode = interaction.options.getString('mode') === 'public' ? 'public' : 'private';
    let ragContext = 'RAG session context is not available yet.';
    try {
      ragContext = await this.dnd.buildQuestionContext(interaction.channelId, message);
    } catch (error: any) {
      log.warn({ error: error.message, channelId: interaction.channelId }, 'Failed to build DnD RAG question context');
    }

    const mentionTargets = buildDiscordMentionTargetsFromInteraction(interaction);
    const effectiveMessage = buildStructuredIncomingMessage(
      {
        platform: 'discord',
        conversationLabel: `DnD Thread #${interaction.channel?.isTextBased() && 'name' in interaction.channel ? interaction.channel.name : interaction.channelId}`,
        sender: {
          id: interaction.user.id,
          label: interaction.user.tag,
          name: interaction.user.displayName ?? interaction.user.username,
          username: interaction.user.username,
        },
        isGroupChat: true,
        wasMentioned: false,
        mentionTargets,
      },
      buildDndQuestionPrompt(dndDetails, interaction.user.id, message, mode, ragContext),
    );

    await interaction.deferReply({ ephemeral: mode === 'private' });

    const isGroup = Boolean(interaction.guild);
    const sessionName = `${buildDiscordSessionName(interaction.channel, interaction.guild)} [dnd:${mode}]`;

    const request: AgentRequest = {
      message: effectiveMessage,
      sessionKey: `discord:dnd-question:${dndDetails.session.id}:${interaction.user.id}:${mode}`,
      sessionName,
      isGroup,
      disablePlanner: true,
      disableReasoning: true,
      channelType: 'discord',
      channelTarget: interaction.channelId,
      userIdentifier: `${interaction.user.tag} [dnd-question:${mode}]`,
      workingDir: this.config.workspace || getConfig().agent?.workspace || getStateDir(),
    };

    let fullContent = '';
    const toolUpdates: string[] = [];
    const progress = createDiscordProgressState();
    let lastProgressFlush = 0;

    const flushProgress = async (force = false, finalContent?: string): Promise<void> => {
      const now = Date.now();
      if (!force && now - lastProgressFlush < 1500) return;
      lastProgressFlush = now;

      const hasPlan = progress.tasks.length > 0;
      const payload: InteractionEditReplyOptions = {
        embeds: hasPlan ? [buildDiscordProgressEmbed(progress, finalContent)] : [],
      };

      if (finalContent !== undefined) {
        payload.content = finalContent || null;
      } else if (!hasPlan) {
        payload.content = `_${discordProgressStatusLabel(progress.status)}..._`;
      } else {
        payload.content = null;
      }

      await interaction.editReply(payload);
    };

    try {
      this.beginRequest('thinking');
      await flushProgress(true);
      for await (const event of this.engine.processRequest(request)) {
        this.updateStatusForEvent(event.type, event.toolName);
        applyEventToDiscordProgress(progress, event);

        switch (event.type) {
          case 'content':
            fullContent += event.content ?? '';
            break;
          case 'plan':
            toolUpdates.push(`Planned ${event.plan?.tasks?.length ?? 0} tasks`);
            break;
          case 'task_update':
            if (event.taskStatus === 'in_progress') {
              toolUpdates.push(`Working on ${event.taskTitle}`);
            }
            break;
          case 'tool_start':
            toolUpdates.push(`Running \`${event.toolName}\`...`);
            break;
          case 'tool_result':
            toolUpdates.push(`${event.toolResult?.success ? 'Finished' : 'Failed'} \`${event.toolName}\``);
            break;
          case 'error':
            fullContent += `\nError: ${event.error}`;
            break;
        }

        await flushProgress();
      }

      const messages = buildOutgoingMessages(fullContent, toolUpdates, {
        replyStyle: 'single',
        showToolProgress: false,
        maxLen: 1900,
        format: 'plain',
      });

      progress.status = progress.error ? 'error' : 'done';
      await flushProgress(true, messages[0] ?? '(No response)');
      for (let i = 1; i < messages.length; i++) {
        await interaction.followUp({ content: messages[i], ephemeral: mode === 'private' });
      }
    } catch (err: any) {
      log.error({ error: err.message }, 'Slash command /question error');
      progress.status = 'error';
      progress.error = err.message;
      await flushProgress(true, `Error: ${err.message}`);
    } finally {
      this.endRequest();
    }
  }

  private async handleVoiceCommand(interaction: ChatInputCommandInteraction): Promise<void> {
    const subcommand = interaction.options.getSubcommand();
    const guild = interaction.guild;

    if (!guild) {
      await interaction.reply({
        content: 'This command can only be used in a server.',
        ephemeral: true,
      });
      return;
    }

    if (subcommand === 'join') {
      const member = await guild.members.fetch(interaction.user.id);
      const voiceChannel = member.voice.channel;

      if (!voiceChannel) {
        await interaction.reply({
          content: 'You must be in a voice channel to use this command.',
          ephemeral: true,
        });
        return;
      }

      await interaction.deferReply({ ephemeral: true });

      try {
        // Leave previous connection if exists in this guild
        const existingPipeline = this.voicePipelines.get(guild.id);
        if (existingPipeline) {
          existingPipeline.stop();
          this.voicePipelines.delete(guild.id);
        }

        // Join the channel
        const { joinVoiceChannel } = await import('@discordjs/voice');
        const connection = joinVoiceChannel({
          channelId: voiceChannel.id,
          guildId: guild.id,
          adapterCreator: guild.voiceAdapterCreator as any,
          selfDeaf: false, // Must be false to receive audio!
          selfMute: false,
        });

        // Initialize voice pipeline
        const pipeline = new VoicePipeline(connection, this.engine);
        pipeline.start();

        this.voicePipelines.set(guild.id, pipeline);

        log.info({ guildId: guild.id, channelId: voiceChannel.id }, 'Joined voice channel');
        await interaction.editReply({
          content: `Joined voice channel **${voiceChannel.name}** and started listening. Speak naturally!`,
        });
      } catch (err: any) {
        log.error({ error: err.message }, 'Failed to join voice channel');
        await interaction.editReply({
          content: `Failed to join voice channel: ${err.message}`,
        });
      }
    } else if (subcommand === 'leave') {
      const pipeline = this.voicePipelines.get(guild.id);
      if (!pipeline) {
        await interaction.reply({
          content: 'I am not currently in any voice channel in this server.',
          ephemeral: true,
        });
        return;
      }

      try {
        pipeline.stop();
        this.voicePipelines.delete(guild.id);
        
        // Find existing connection to destroy
        const { getVoiceConnection } = await import('@discordjs/voice');
        const connection = getVoiceConnection(guild.id);
        if (connection) {
          connection.destroy();
        }

        await interaction.reply({
          content: 'Left the voice channel.',
          ephemeral: true,
        });
      } catch (err: any) {
        log.error({ error: err.message }, 'Failed to leave voice channel');
        await interaction.reply({
          content: `Failed to leave voice channel: ${err.message}`,
          ephemeral: true,
        });
      }
    }
  }

  private async handleModelCommand(interaction: ChatInputCommandInteraction): Promise<void> {
    const providers = this.engine.getLLMClient().getAllProviders();
    const activeProvider = this.engine.getLLMClient().getProviders()[0];
    const primaryId = activeProvider?.id ?? 'unknown';

    const modelLines = providers.map(p => {
      const isPrimary = p.id === primaryId;
      const prefix = isPrimary ? '`▸`' : '`·`';
      const vision = p.supportsVision ? '👁️' : '';
      return `${prefix} **${p.id}** — ${p.contextWindow.toLocaleString()} ctx ${vision}`;
    });

    const embed = new EmbedBuilder()
      .setAuthor({ name: 'LiteClaw · Models' })
      .setColor(0x5E35B1)
      .setDescription(modelLines.length > 0
        ? modelLines.join('\n')
        : '*No models configured.*',
      );

    await interaction.reply({ embeds: [embed] });
  }

  // ─── Message Handler (with reactions + status) ───────────────────

  private async handleMessage(message: Message): Promise<void> {
    log.info({
      user: message.author?.tag,
      channel: message.channel.id,
      isPartial: message.partial,
      content: message.content ? '<has_content>' : '<empty>',
      attachments: message.attachments?.size ?? 0
    }, 'Discord message event triggered');

    if (message.partial) {
      try {
        await message.fetch();
      } catch (err) {
        log.error('Failed to fetch partial message');
        return;
      }
    }

    // Ignore own messages
    if (message.author.id === this.client.user?.id) return;
    // Ignore other bots unless configured
    if (message.author.bot && !this.config.allowBots) return;

    const dndSession = this.dnd.getSessionForThread(message.channel.id);
    const inProtectedDndThread = Boolean(dndSession);
    const isDndPlayer = dndSession
      ? this.dnd.isPlayerInThread(message.channel.id, message.author.id)
      : false;

    if (inProtectedDndThread && !isDndPlayer) {
      await message.reply({
        content: 'This DnD thread is reserved for enrolled players. Use `/dnd join` if you want to join the campaign midway.',
        allowedMentions: { repliedUser: false },
      });
      return;
    }

    const replyMeta = await this.getReplyMetadata(message);

    // Check if bot is mentioned, directly replied to, or it's a DM
    const isMentioned = message.mentions.has(this.client.user!);
    const isDM = !message.guild;
    const isReplyToBot = replyMeta.authorId === this.client.user?.id;
    const shouldTreatAsDndTableTalk = inProtectedDndThread && isDndPlayer && (isMentioned || isReplyToBot);

    if (shouldTreatAsDndTableTalk) {
      const gate = this.dnd.validateNarrativeTurn(message.channel.id, message.author.id);
      if (!gate.ok) {
        await message.reply({
          content: gate.reason,
          allowedMentions: { repliedUser: false },
        });
        return;
      }
    }

    if (!isMentioned && !isDM && !shouldTreatAsDndTableTalk) return;

    const parsed = parseIncomingDiscordMentions(
      message.content,
      message.guild,
      this.client,
      this.client.user?.id
    );
    const mentionTargets: MentionTarget[] = buildDiscordMentionTargetsFromMessage(message);
    const content = parsed.cleanContent;

    if (!content && message.attachments.size === 0) return;

    const replyContext = replyMeta.context;
    let dndRagContext = 'No relevant RAG context found for this session yet.';
    if (shouldTreatAsDndTableTalk) {
      try {
        dndRagContext = await this.dnd.buildNarrativeRagContext(message.channel.id, content || '(image attached)');
      } catch (error: any) {
        log.warn({ error: error.message, channelId: message.channel.id }, 'Failed to build DnD narrative RAG context');
      }
    }
    const effectivePrompt = shouldTreatAsDndTableTalk
      ? this.dnd.buildTableTalkPrompt(message.channel.id, message.author.id, content || '(image attached)', dndRagContext)
      : (content || '(image attached)');

    const isGroup = Boolean(message.guildId);
    const conversationLabel = message.guild
      ? `Guild #${message.channel.isTextBased() && 'name' in message.channel ? message.channel.name : message.channel.id}`
      : 'Discord DM';

    const effectiveMessage = buildStructuredIncomingMessage(
      {
        platform: 'discord',
        conversationLabel,
        sender: {
          id: message.author.id,
          label: message.author.tag,
          name: message.member?.displayName || message.author.globalName || message.author.username,
          username: message.author.username,
          tag: message.author.tag,
        },
        isGroupChat: isGroup,
        wasMentioned: isMentioned || parsed.wasBotMentioned || shouldTreatAsDndTableTalk,
        mentionTargets,
        taggedUsers: parsed.taggedUsers,
        taggedRoles: parsed.taggedRoles,
        taggedChannels: parsed.taggedChannels,
        replyContext,
      },
      effectivePrompt
    );

    const images = await this.collectMessageImages(message);

    await this.processAgentTurn({
      channel: message.channel as any,
      author: message.author,
      replyTo: message,
      effectiveMessage,
      mentionTargets,
      images: images.length > 0 ? images : undefined,
      shouldTreatAsDndTableTalk,
    });
  }

  /**
   * Core logic for processing a single agent turn and sending the response.
   * Shared by handleMessage and handleButtonInteraction.
   */
  private async processAgentTurn(params: {
    channel: TextChannel | ThreadChannel | DMChannel;
    author: User;
    replyTo: Message | null;
    effectiveMessage: string;
    mentionTargets: MentionTarget[];
    images?: string[];
    shouldTreatAsDndTableTalk: boolean;
  }): Promise<void> {
    const { channel, author, replyTo, effectiveMessage, mentionTargets, images, shouldTreatAsDndTableTalk } = params;
    const sessionKey = `discord:${channel.id}`;
    const guild = (channel as any).guild || (replyTo as any)?.guild || null;
    const isGroup = Boolean(guild || (channel as any).guildId || (channel.type !== ChannelType.DM));
    const sessionName = buildDiscordSessionName(channel, guild);

    log.info({
      user: author.tag,
      channel: channel.id,
      contentLength: effectiveMessage.length,
    }, 'Processing Discord agent turn');

    let fullContent = '';
    const toolUpdates: string[] = [];
    let hasThought = false;
    let interactiveChoiceSent = false;
    const addedReactions = new Set<string>();
    const progress = createDiscordProgressState();
    let progressMessage: Message | null = null;
    let lastProgressFlush = 0;

    const flushProgress = async (force = false, finalContent?: string): Promise<void> => {
      if (!progressMessage) return;
      const now = Date.now();
      if (!force && now - lastProgressFlush < 1500) return;
      lastProgressFlush = now;

      const hasPlan = progress.tasks.length > 0;
      const payload: any = {
        embeds: hasPlan ? [buildDiscordProgressEmbed(progress, finalContent)] : [],
      };

      if (finalContent !== undefined) {
        payload.content = finalContent || null;
      } else if (!hasPlan) {
        payload.content = `_${discordProgressStatusLabel(progress.status)}..._`;
      } else {
        payload.content = null;
      }

      await progressMessage.edit(payload);
    };

    const typingInterval = setInterval(async () => {
      try {
        await (channel as any).sendTyping();
      } catch { /* ignore */ }
    }, 7_000);

    try {
      // ── Step 1: React with 👀 (received) ──
      if (replyTo) await this.react(replyTo, REACTIONS.RECEIVED);
      this.beginRequest('thinking');

      const progressPayload = {
        content: `_${discordProgressStatusLabel(progress.status)}..._`,
        allowedMentions: { repliedUser: false },
      };

      progressMessage = replyTo
        ? await replyTo.reply(progressPayload)
        : await (channel as any).send(progressPayload);

      // ── Build Request ──
      const request: AgentRequest = {
        message: effectiveMessage,
        images,
        sessionKey,
        sessionName,
        isGroup,
        disablePlanner: shouldTreatAsDndTableTalk,
        channelType: 'discord',
        channelTarget: channel.id,
        userIdentifier: author.tag,
        workingDir: this.config.workspace || getConfig().agent?.workspace || getStateDir(),
        sendFile: async (filePath: string, fileName?: string) => {
          if (replyTo) {
            await this.sendFile(replyTo, filePath, fileName);
          } else {
            // Fallback for button interactions if we don't have a direct replyTo message
            const name = fileName ?? basename(filePath);
            const attachment = new AttachmentBuilder(filePath, { name });
            await (channel as any).send({
              content: `📎 Sending file: **${name}**`,
              files: [attachment],
            });
          }
        },
        sendInteractiveChoice: async (choiceRequest) => {
          interactiveChoiceSent = true;
          return this.sendInteractiveChoice({
            channelId: channel.id,
            replyTo: async (payload) => replyTo ? replyTo.reply(payload) : (channel as any).send(payload),
          }, choiceRequest);
        },
      };

      // For DnD sessions, use the dedicated GM system prompt and keep reasoning enabled
      if (shouldTreatAsDndTableTalk) {
        const { readFileSync } = await import('fs');
        const { resolve } = await import('path');
        const gmPromptPath = resolve(process.cwd(), 'config/dnd_gm_prompt.md');
        if (existsSync(gmPromptPath)) {
          request.systemPromptOverride = readFileSync(gmPromptPath, 'utf-8');
        }
      }

      for await (const event of this.engine.processRequest(request)) {
        this.updateStatusForEvent(event.type, event.toolName);
        applyEventToDiscordProgress(progress, event);
        await flushProgress();

        // Remove the looking emoji once the agent starts doing ANY work
        if (['thinking', 'content', 'tool_start', 'plan', 'task_update'].includes(event.type) && addedReactions.has(REACTIONS.RECEIVED) && replyTo) {
          await this.unreact(replyTo, REACTIONS.RECEIVED);
          addedReactions.delete(REACTIONS.RECEIVED);
        }

        switch (event.type) {
          case 'thinking':
            if (!hasThought && replyTo) {
              await this.react(replyTo, REACTIONS.THINKING);
              addedReactions.add(REACTIONS.THINKING);
              hasThought = true;
            }
            break;

          case 'content':
            fullContent += event.content ?? '';
            if (addedReactions.has(REACTIONS.THINKING) && replyTo) {
              await this.unreact(replyTo, REACTIONS.THINKING);
              addedReactions.delete(REACTIONS.THINKING);
            }
            break;

          case 'plan':
            toolUpdates.push(`🗺️ Planned ${event.plan?.tasks?.length ?? 0} task${(event.plan?.tasks?.length ?? 0) === 1 ? '' : 's'}`);
            break;

          case 'task_update': {
            const prefix = event.taskIndex && event.taskTotal ? `[${event.taskIndex}/${event.taskTotal}] ` : '';
            if (event.taskStatus === 'in_progress') {
              toolUpdates.push(`→ ${prefix}${event.taskTitle}`);
            } else if (event.taskStatus) {
              const icon = event.taskStatus === 'completed' ? '✓' : event.taskStatus === 'blocked' ? '⚠' : '✗';
              toolUpdates.push(`${icon} ${prefix}${event.taskTitle}${event.taskSummary ? ` — ${event.taskSummary}` : ''}`);
            }
            break;
          }

          case 'tool_start':
            const toolEmoji = this.getToolReaction(event.toolName ?? '');
            if (!addedReactions.has(toolEmoji) && replyTo) {
              await this.react(replyTo, toolEmoji);
              addedReactions.add(toolEmoji);
            }
            toolUpdates.push(`⚙ Running \`${event.toolName}\`...`);
            break;

          case 'tool_result': {
            const icon = event.toolResult?.success ? '✓' : '✗';
            toolUpdates.push(`${icon} \`${event.toolName}\` ${event.toolResult?.success ? 'completed' : 'failed'}`);
            const doneToolEmoji = this.getToolReaction(event.toolName ?? '');
            if (addedReactions.has(doneToolEmoji) && replyTo) {
              await this.unreact(replyTo, doneToolEmoji);
              addedReactions.delete(doneToolEmoji);
            }
            break;
          }

          case 'error':
            fullContent += `\n⚠ Error: ${event.error}`;
            break;
        }
      }

      await flushProgress(true);
      for (const emoji of addedReactions) {
        if (replyTo) await this.unreact(replyTo, emoji);
      }
      if (replyTo) await this.react(replyTo, REACTIONS.DONE);

      setTimeout(async () => {
        if (replyTo) await this.unreact(replyTo, REACTIONS.DONE);
      }, 10_000);

      // Format and send response
      const structuredNarrative = shouldTreatAsDndTableTalk
        ? await this.dnd.processStructuredNarrativeResponse(channel.id, fullContent)
        : { content: fullContent, shopEmbeds: [], combatEmbeds: [], combatComponents: null, actionComponents: null, rollComponents: null };

      progress.status = progress.error ? 'error' : 'done';

      if (shouldTreatAsDndTableTalk) {
        const dndDetails = this.dnd.getSessionForThread(channel.id);
        this.dnd.markNarrativeTurnSpent(channel.id, author.id);

        // For DnD narrative responses, send as a rich embed for best markdown rendering
        const spentNotice = this.dnd.buildNarrativeTurnSpentNotice(channel.id, author.id);
        const decoratedContent = spentNotice && !(structuredNarrative.combatEmbeds?.length > 0)
          ? `${structuredNarrative.content}\n\n*${spentNotice}*`
          : structuredNarrative.content;
        const resolvedContent = resolveDiscordMentions(decoratedContent, mentionTargets, guild);
        const narrativeEmbed = new EmbedBuilder()
          .setColor(0x8e44ad)
          .setDescription(resolvedContent.content.slice(0, 4096));
        const trackerEmbed = this.dnd.buildTurnTrackerEmbed(channel.id);
        const actionComponents = spentNotice && !(structuredNarrative.combatEmbeds?.length > 0)
          ? []
          : (structuredNarrative.actionComponents || []);

        if (progressMessage) {
          const components = [
            ...(structuredNarrative.combatComponents || []),
            ...actionComponents,
            ...(structuredNarrative.rollComponents || []),
          ].slice(0, 5); // Discord limit

          await progressMessage.edit({
            content: null,
            embeds: [
              ...(progress.tasks.length > 0 ? [buildDiscordProgressEmbed(progress, resolvedContent.content)] : []),
              narrativeEmbed,
              ...(trackerEmbed ? [trackerEmbed] : []),
              ...structuredNarrative.shopEmbeds,
              ...structuredNarrative.combatEmbeds,
            ],
            components,
            allowedMentions: {
              parse: ['users', 'roles'],
              users: resolvedContent.userIds,
              roles: resolvedContent.roleIds,
              repliedUser: false,
            },
          });
          if (dndDetails) {
            await this.dnd.recordCanonicalSceneState({
              threadId: channel.id,
              sessionId: dndDetails.session.id,
              source: 'narrative',
              title: narrativeEmbed.data.title ?? null,
              content: structuredNarrative.content,
              messageId: progressMessage.id,
            });
          }
        } else {
          const components = [
            ...(structuredNarrative.combatComponents || []),
            ...actionComponents,
            ...(structuredNarrative.rollComponents || []),
          ].slice(0, 5);

          const sentMessage = await (channel as any).send({
            embeds: [
              narrativeEmbed,
              ...(trackerEmbed ? [trackerEmbed] : []),
              ...structuredNarrative.shopEmbeds,
              ...structuredNarrative.combatEmbeds,
            ],
            components,
            allowedMentions: {
              parse: ['users', 'roles'],
              users: resolvedContent.userIds,
              roles: resolvedContent.roleIds,
            },
          });
          if (dndDetails) {
            await this.dnd.recordCanonicalSceneState({
              threadId: channel.id,
              sessionId: dndDetails.session.id,
              source: 'narrative',
              title: narrativeEmbed.data.title ?? null,
              content: structuredNarrative.content,
              messageId: sentMessage?.id ?? null,
            });
          }
        }
      } else {
        // Non-DnD responses: standard text output
        const messages = buildOutgoingMessages(structuredNarrative.content, toolUpdates, {
          replyStyle: this.config.replyStyle ?? 'single',
          showToolProgress: this.config.showToolProgress ?? false,
          maxLen: 1900,
          format: 'discord',
        });

        const hasPlan = progress.tasks.length > 0;

        if (hasPlan) {
          // Finalize monitoring card without body content so it stays as a clean progress embed
          if (progressMessage) {
            await progressMessage.edit({
              content: null,
              embeds: [buildDiscordProgressEmbed(progress)],
              allowedMentions: { repliedUser: false },
            });
          }

          // Send the full response as brand new message(s) after the monitoring card
          for (const msg of messages) {
            const resolved = resolveDiscordMentions(msg, mentionTargets, guild);
            await (channel as any).send({
              content: resolved.content,
              allowedMentions: {
                parse: ['users', 'roles'],
                users: resolved.userIds,
                roles: resolved.roleIds,
              },
            });
          }
        } else if (interactiveChoiceSent && isRedundantChoiceEcho(structuredNarrative.content)) {
          // Interactive choices were posted directly with buttons; delete the starting progress placeholder
          // to prevent duplicate listing of options
          if (progressMessage) {
            await progressMessage.delete().catch(() => null);
            progressMessage = null;
          }
        } else {
          const first = messages[0] ?? '(No response)';
          const resolvedFirst = resolveDiscordMentions(first, mentionTargets, guild);
          const hasTags = resolvedFirst.userIds.length > 0 || resolvedFirst.roleIds.length > 0;

          if (hasTags) {
            // Delete placeholder so sending a new message triggers an active mention notification/ping
            if (progressMessage) {
              await progressMessage.delete().catch(() => null);
              progressMessage = null;
            }
            if (replyTo) {
              await replyTo.reply({
                content: resolvedFirst.content,
                allowedMentions: {
                  parse: ['users', 'roles'],
                  users: resolvedFirst.userIds,
                  roles: resolvedFirst.roleIds,
                  repliedUser: true,
                },
              });
            } else {
              await (channel as any).send({
                content: resolvedFirst.content,
                allowedMentions: {
                  parse: ['users', 'roles'],
                  users: resolvedFirst.userIds,
                  roles: resolvedFirst.roleIds,
                },
              });
            }
          } else {
            if (progressMessage) {
              await progressMessage.edit({
                content: resolvedFirst.content,
                embeds: [],
                allowedMentions: {
                  parse: ['users', 'roles'],
                  users: resolvedFirst.userIds,
                  roles: resolvedFirst.roleIds,
                  repliedUser: false,
                },
              });
            }
          }

          for (let i = 1; i < messages.length; i++) {
            const resolved = resolveDiscordMentions(messages[i], mentionTargets, guild);
            await (channel as any).send({
              content: resolved.content,
              allowedMentions: {
                parse: ['users', 'roles'],
                users: resolved.userIds,
                roles: resolved.roleIds,
              },
            });
          }
        }
      }

      log.info({
        user: author.tag,
        responseLength: fullContent.length,
        tools: toolUpdates.length,
      }, 'Discord response sent');

    } catch (err: any) {
      log.error({ error: err.message }, 'Discord agent turn error');
      for (const emoji of addedReactions) {
        if (replyTo) await this.unreact(replyTo, emoji);
      }
      if (replyTo) {
        await this.react(replyTo, REACTIONS.ERROR);
        await replyTo.reply(`⚠ Error: ${err.message}`);
      } else {
        await (channel as any).send(`⚠ Error: ${err.message}`);
      }
    } finally {
      clearInterval(typingInterval);
      this.endRequest();
    }
  }

  private async getReplyMetadata(message: Message): Promise<{ context: string | null; authorId: string | null }> {
    if (!message.reference?.messageId) {
      return { context: null, authorId: null };
    }

    try {
      const referenced = typeof message.fetchReference === 'function'
        ? await message.fetchReference()
        : null;

      if (!referenced) return { context: null, authorId: null };

      const referencedContent = referenced.content?.trim() || summarizeAttachments(referenced.attachments);
      if (!referencedContent) {
        return {
          context: null,
          authorId: referenced.author?.id ?? null,
        };
      }

      return {
        context: formatReplyContext(
          referenced.author?.tag || referenced.author?.username || 'Unknown user',
          referencedContent
        ),
        authorId: referenced.author?.id ?? null,
      };
    } catch (err: any) {
      log.debug({ error: err.message, messageId: message.id }, 'Failed to fetch replied-to Discord message');
      return { context: null, authorId: null };
    }
  }

  // ─── Response Sending ────────────────────────────────────────────

  private async collectMessageImages(message: Message): Promise<string[]> {
    const images: string[] = [];

    for (const [, attachment] of message.attachments) {
      const image = await this.fetchAttachmentAsImageData(attachment, 'message');
      if (image) images.push(image);
    }

    if (!message.reference?.messageId) return images;

    try {
      const referenced = typeof message.fetchReference === 'function'
        ? await message.fetchReference()
        : null;

      if (!referenced) return images;

      for (const [, attachment] of referenced.attachments) {
        const image = await this.fetchAttachmentAsImageData(attachment, 'quoted_reply');
        if (image) images.push(image);
      }
    } catch (err: any) {
      log.debug({ error: err.message, messageId: message.id }, 'Failed to fetch replied-to Discord images');
    }

    return images;
  }

  private async fetchAttachmentAsImageData(attachment: any, reason: string): Promise<string | null> {
    const contentType = attachment.contentType ?? '';
    const imageByType = contentType.startsWith('image/');
    const imageByExtension = /\.(png|jpe?g|gif|webp|bmp|svg|heic|heif)$/i.test(
      attachment.name ?? extname(attachment.url)
    );

    if (!imageByType && !imageByExtension) return null;

    try {
      const response = await fetch(attachment.url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} when fetching attachment`);
      }
      const buffer = Buffer.from(await response.arrayBuffer());
      return await preprocessImage(buffer);
    } catch (err: any) {
      log.warn({
        error: err.message,
        reason,
        attachmentName: attachment.name,
        contentType: attachment.contentType,
        url: attachment.url,
      }, 'Failed to fetch Discord image attachment');
      return null;
    }
  }

  private async sendResponse(
    message: Message,
    content: string,
    toolUpdates: string[],
    mentionTargets: MentionTarget[]
  ): Promise<void> {
    // Convert markdown tables to bullets if configured
    if (this.config.markdown?.tables === 'bullets') {
      content = convertTablesToBullets(content);
    }

    const messages = buildOutgoingMessages(content, toolUpdates, {
      replyStyle: this.config.replyStyle ?? 'single',
      showToolProgress: this.config.showToolProgress ?? false,
      maxLen: 1900,
      format: 'discord',
    });

    for (let i = 0; i < messages.length; i++) {
      const resolved = resolveDiscordMentions(messages[i], mentionTargets);
      if (i === 0) {
        await message.reply({
          content: resolved.content,
          allowedMentions: {
            users: resolved.userIds,
            repliedUser: false,
          },
        });
      } else {
        await (message.channel as any).send({
          content: resolved.content,
          allowedMentions: {
            users: resolved.userIds,
          },
        });
      }
    }
  }

  // ─── File Sending ────────────────────────────────────────────────

  private async sendFile(message: Message, filePath: string, fileName?: string): Promise<void> {
    if (!existsSync(filePath)) {
      await (message.channel as any).send(`⚠ File not found: ${filePath}`);
      return;
    }

    const name = fileName ?? basename(filePath);
    const attachment = new AttachmentBuilder(filePath, { name });

    await (message.channel as any).send({
      content: `📎 Sending file: **${name}**`,
      files: [attachment],
    });

    log.info({ file: name, channel: message.channel.id }, 'Sent file to Discord');
  }

  // ─── Button Interactions (Confirmations) ─────────────────────────

  private async handleButtonInteraction(interaction: ButtonInteraction): Promise<void> {
    const customId = interaction.customId;

    const dndResult = await this.dnd.handleButton(interaction);
    if (dndResult === true) {
      return;
    }
    if (typeof dndResult === 'string') {
      // It's a DnD action choice OR a roll result!
      await this.processDndActionChoice(interaction, dndResult);
      return;
    }

    if (customId.startsWith('liteclaw_confirm_')) {
      const confirmId = customId.replace('liteclaw_confirm_', '');
      this.confirmations.resolveConfirmation(confirmId, true);
      await interaction.update({
        content: '✅ **Confirmed** — proceeding with the operation.',
        components: [],
      });
    } else if (customId.startsWith('liteclaw_reject_')) {
      const confirmId = customId.replace('liteclaw_reject_', '');
      this.confirmations.resolveConfirmation(confirmId, false);
      await interaction.update({
        content: '❌ **Cancelled** — operation was rejected.',
        components: [],
      });
    } else if (customId.startsWith('liteclaw_choice_')) {
      await this.handleInteractiveChoice(interaction);
    }
  }

  /**
   * Specifically handles action choices from DnD buttons by feeding them back into the engine.
   */
  private async processDndActionChoice(interaction: ChatInputCommandInteraction | ButtonInteraction, actionText: string): Promise<void> {
    const channel = interaction.channel;
    if (!channel || !channel.isTextBased()) return;

    if (this.dnd.shouldQueueNarrativeActions(channel.id)) {
      const queued = this.dnd.queueNarrativeAction(channel.id, interaction.user.id, actionText);
      await (channel as any).send({
        content: `${interaction.user} **->** ${actionText}`,
      });
      if (!queued.shouldResolve || !queued.combinedActionText) {
        await (channel as any).send({
          content: queued.waitingOn.length > 0
            ? `Waiting on: **${queued.waitingOn.join(', ')}** before resolving the shared scene.`
            : 'Queued action recorded.',
        });
        return;
      }
      actionText = queued.combinedActionText;
    }

    // Post the action as a visible message so the table sees what was chosen
    await (channel as any).send({
      content: `${interaction.user} **->** ${actionText}`,
    });

    // Build the engine request
    const mentionTargets: MentionTarget[] = [
      createDiscordMentionTarget(
        interaction.user.id,
        (interaction.member as any)?.displayName || interaction.user.globalName || interaction.user.username,
        interaction.user.username,
        interaction.user.tag
      )
    ];

    const dndRagContext = await this.dnd.buildNarrativeRagContext(channel.id, actionText);
    const effectivePrompt = this.dnd.buildTableTalkPrompt(channel.id, interaction.user.id, actionText, dndRagContext);

    const effectiveMessage = buildStructuredIncomingMessage(
      {
        platform: 'discord',
        conversationLabel: interaction.guild
          ? `Guild #${'name' in channel ? (channel as any).name : channel.id}`
          : 'Discord DM',
        sender: {
          id: interaction.user.id,
          label: interaction.user.tag,
          name: (interaction.member as any)?.displayName || interaction.user.globalName || interaction.user.username,
          username: interaction.user.username,
          tag: interaction.user.tag,
        },
        isGroupChat: Boolean(interaction.guildId),
        wasMentioned: true,
        mentionTargets,
      },
      effectivePrompt
    );

    await this.processAgentTurn({
      channel: channel as any,
      author: interaction.user,
      replyTo: null, // Buttons don't have a specific message to reply to for reactions
      effectiveMessage,
      mentionTargets,
      shouldTreatAsDndTableTalk: true,
    });
  }

  private async handleSelectMenuInteraction(interaction: AnySelectMenuInteraction): Promise<void> {
    if (await this.dnd.handleSelectMenu(interaction)) {
      return;
    }
  }

  private async handleModalInteraction(interaction: ModalSubmitInteraction): Promise<void> {
    if (await this.dnd.handleModalSubmit(interaction)) {
      return;
    }
  }

  // ─── Confirmation Handler ────────────────────────────────────────

  private setupConfirmationHandler(): void {
    this.confirmations.on('confirmation_request', async (conf) => {
      if (conf.channelType !== 'discord' || !conf.channelTarget) return;

      this.setStatus('confirming');

      try {
        const channel = await this.client.channels.fetch(conf.channelTarget);
        if (!channel?.isTextBased()) return;

        const row = new ActionRowBuilder<ButtonBuilder>().addComponents(
          new ButtonBuilder()
            .setCustomId(`liteclaw_confirm_${conf.id}`)
            .setLabel('✅ Confirm')
            .setStyle(ButtonStyle.Success),
          new ButtonBuilder()
            .setCustomId(`liteclaw_reject_${conf.id}`)
            .setLabel('❌ Cancel')
            .setStyle(ButtonStyle.Danger),
        );

        const embed = new EmbedBuilder()
          .setAuthor({ name: 'LiteClaw · Confirmation Required' })
          .setColor(0xFFAB00)
          .setDescription([
            conf.description,
            '',
            `⚙️ \`${conf.toolName}\` · ⏱️ ${conf.timeoutMs / 1000}s timeout`,
          ].join('\n'))
          .setFooter({ text: conf.id });

        await (channel as any).send({ embeds: [embed], components: [row] });
      } catch (err: any) {
        log.error({ error: err.message }, 'Failed to send Discord confirmation');
      }
    });
  }

  // ─── Lifecycle ───────────────────────────────────────────────────

  async start(): Promise<void> {
    const token = this.config.token ?? process.env.DISCORD_TOKEN;
    if (!token) {
      throw new Error('Discord token not configured. Set DISCORD_TOKEN in .env or config.yaml');
    }

    await ensureDiscordGatewayReachable();

    let retries = 0;
    const maxRetries = 5;
    while (retries < maxRetries) {
      try {
        await this.client.login(token);
        return;
      } catch (err: any) {
        retries++;
        const delay = Math.min(1000 * Math.pow(2, retries), 30000);
        log.warn({ error: err.message, retry: retries, nextRetryDelay: delay }, 'Discord login failed, retrying...');
        if (isDiscordDnsFailure(err)) {
          throw new Error(`Discord gateway DNS lookup failed: ${err.message}`);
        }
        if (retries >= maxRetries) {
          throw new Error(`Failed to login to Discord after ${maxRetries} attempts: ${err.message}`);
        }
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }

  stop(): void {
    if (this.statusTimer) clearInterval(this.statusTimer);
    void this.dnd.shutdown();
    this.client.destroy();
  }

  private async sendInteractiveChoice(
    target: {
      channelId: string;
      replyTo: (payload: {
        content: string;
        components: ActionRowBuilder<ButtonBuilder>[];
      }) => Promise<any>;
    },
    request: InteractiveChoiceRequest
  ): Promise<string> {
    this.pruneInteractiveChoices();

    const options = request.options
      .map(option => option.trim())
      .filter(Boolean)
      .slice(0, 5);

    if (options.length === 0) {
      throw new Error('Interactive choices require at least one option.');
    }

    const choiceId = `choice_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const row = new ActionRowBuilder<ButtonBuilder>().addComponents(
      ...options.map((option, index) =>
        new ButtonBuilder()
          .setCustomId(`liteclaw_choice_${choiceId}_${index}`)
          .setLabel(option.slice(0, 80))
          .setStyle(ButtonStyle.Secondary)
      )
    );

    const sent = await target.replyTo({
      content: request.prompt,
      components: [row],
    });

    this.interactiveChoices.set(choiceId, {
      prompt: request.prompt,
      options,
      responses: request.responses,
      messageId: sent?.id ?? '',
      channelId: target.channelId,
      createdAt: Date.now(),
    });

    return choiceId;
  }

  private async handleInteractiveChoice(interaction: ButtonInteraction): Promise<void> {
    const match = interaction.customId.match(/^liteclaw_choice_(choice_[^_]+_[^_]+)_(\d+)$/);
    if (!match) {
      await interaction.reply({ content: '⚠ That interactive choice is invalid.', ephemeral: true });
      return;
    }

    const [, choiceId, rawIndex] = match;
    const record = this.interactiveChoices.get(choiceId);
    if (!record) {
      await interaction.reply({ content: '⚠ That interactive choice has expired.', ephemeral: true });
      return;
    }

    const index = Number.parseInt(rawIndex, 10);
    const option = record.options[index];
    if (!option) {
      await interaction.reply({ content: '⚠ That choice option is no longer available.', ephemeral: true });
      return;
    }

    // Disable clicked buttons on original message so user cannot double-click
    try {
      if (interaction.message?.components) {
        const disabledRows = interaction.message.components.map(row => {
          const newRow = new ActionRowBuilder<ButtonBuilder>();
          for (const comp of (row as any).components) {
            const btn = ButtonBuilder.from(comp);
            btn.setDisabled(true);
            if (comp.customId === interaction.customId) {
              btn.setStyle(ButtonStyle.Primary);
            }
            newRow.addComponents(btn);
          }
          return newRow;
        });
        await interaction.message.edit({ components: disabledRows });
      }
    } catch { /* ignore if interaction message cannot be edited */ }

    const response = record.responses?.[option]?.trim()
      || `${interaction.user} picked **${option}**.`;
    const content = response.includes(`<@${interaction.user.id}>`) || response.includes(interaction.user.username)
      ? response
      : `${interaction.user} ${response}`;

    await interaction.reply({
      content,
      allowedMentions: { users: [interaction.user.id] },
    });

    // Invoke agent turn with the user's selected choice
    const channel = interaction.channel;
    if (!channel || !channel.isTextBased()) return;

    const mentionTargets: MentionTarget[] = [
      createDiscordMentionTarget(
        interaction.user.id,
        (interaction.member as any)?.displayName || interaction.user.globalName || interaction.user.username,
        interaction.user.username,
        interaction.user.tag
      ),
    ];

    const effectivePrompt = `I chose "${option}" in response to: "${record.prompt}". Please proceed with this selection.`;
    const effectiveMessage = buildStructuredIncomingMessage(
      {
        platform: 'discord',
        conversationLabel: interaction.guild
          ? `Guild #${'name' in channel ? (channel as any).name : channel.id}`
          : 'Discord DM',
        sender: {
          id: interaction.user.id,
          label: interaction.user.tag,
          name: (interaction.member as any)?.displayName || interaction.user.globalName || interaction.user.username,
          username: interaction.user.username,
          tag: interaction.user.tag,
        },
        isGroupChat: Boolean(interaction.guildId),
        wasMentioned: true,
        mentionTargets,
      },
      effectivePrompt
    );

    await this.processAgentTurn({
      channel: channel as any,
      author: interaction.user,
      replyTo: null,
      effectiveMessage,
      mentionTargets,
      shouldTreatAsDndTableTalk: false,
    });
  }

  private pruneInteractiveChoices(maxAgeMs: number = 24 * 60 * 60 * 1000): void {
    const cutoff = Date.now() - maxAgeMs;
    for (const [choiceId, record] of this.interactiveChoices) {
      if (record.createdAt < cutoff) {
        this.interactiveChoices.delete(choiceId);
      }
    }
  }
}

// ─── Utilities ───────────────────────────────────────────────────────



function buildEffectiveIncomingMessage(replyContext: string | null, content: string): string {
  return replyContext ? `${replyContext}\n\nUser reply: ${content}` : content;
}

function buildDndQuestionPrompt(
  details: DndSessionDetails,
  userId: string,
  question: string,
  mode: 'private' | 'public',
  ragContext: string,
): string {
  const asker = details.players.find(player => player.userId === userId);
  const activePlayer = details.players.find(player => player.userId === details.session.activePlayerUserId);
  const playerSummary = details.players
    .map(player => `${player.characterName} [${player.status}]${player.userId === details.session.hostUserId ? ' host' : ''}`)
    .join(', ');
  const voteSummary = details.vote
    ? `${details.vote.vote.question} | status=${details.vote.vote.status}`
    : 'none';

  return [
    `This is a ${mode} out-of-band player question about an ongoing DnD session.`,
    'Answer as the session GM, but do not advance turns, do not treat this as an in-world action, and do not change the party decision state.',
    mode === 'public'
      ? 'The answer will be visible to the table, so phrase it as GM clarification for the group.'
      : 'The answer will only be visible to the asking player, so you can be concise and direct.',
    'Keep the answer grounded in the current session state. If something is unknown, say so clearly.',
    '',
    `Session: ${details.session.title} (${details.session.id})`,
    `Phase: ${details.session.phase}`,
    `Tone: ${details.session.tone ?? 'not set'}`,
    `Round/Turn: ${details.session.roundNumber}/${details.session.turnNumber}`,
    `Active player: ${activePlayer?.characterName ?? 'none'}`,
    `Asking player: ${asker?.characterName ?? 'unknown player'}`,
    `Players: ${playerSummary}`,
    `Open vote: ${voteSummary}`,
    '',
    'Relevant retrieved context:',
    ragContext,
    '',
    `Question: ${question}`,
  ].join('\n');
}

function buildDiscordSessionName(channel: any, guild?: Guild | null): string {
  const gName = guild?.name || channel?.guild?.name;
  if (gName) {
    if (channel?.isThread?.() || channel?.isThread) {
      const parent = channel.parent?.name ? `#${channel.parent.name} > ` : '';
      return `${gName} > ${parent}${channel.name || 'thread'}`;
    }
    const cName = channel?.name ? `#${channel.name}` : `#${channel?.id ?? 'channel'}`;
    return `${gName} > ${cName}`;
  }
  if (channel?.isDMBased?.() || channel?.recipient) {
    const rec = channel.recipient;
    const name = rec ? (rec.displayName || rec.globalName || rec.username) : 'Direct Message';
    return `Discord DM > ${name}`;
  }
  return `Discord > #${channel?.name || channel?.id || 'channel'}`;
}

function parseIncomingDiscordMentions(
  rawContent: string,
  guild: Guild | null,
  client: Client,
  selfUserId?: string
): {
  cleanContent: string;
  taggedUsers: string[];
  taggedRoles: string[];
  taggedChannels: string[];
  wasBotMentioned: boolean;
} {
  let content = rawContent;
  const taggedUsers: string[] = [];
  const taggedRoles: string[] = [];
  const taggedChannels: string[] = [];
  let wasBotMentioned = false;

  // 1. Parse & resolve role mentions: <@&roleId>
  content = content.replace(/<@&(\d+)>/g, (match, roleId) => {
    const role = guild?.roles?.cache.get(roleId);
    const roleName = role ? role.name : roleId;
    const tag = `@${roleName}`;
    if (!taggedRoles.includes(tag)) taggedRoles.push(tag);
    return tag;
  });

  // 2. Parse & resolve channel mentions: <#channelId>
  content = content.replace(/<#(\d+)>/g, (match, channelId) => {
    const chan = guild?.channels?.cache.get(channelId) || client.channels?.cache.get(channelId);
    const chanName = (chan && 'name' in chan && chan.name) ? chan.name : channelId;
    const tag = `#${chanName}`;
    if (!taggedChannels.includes(tag)) taggedChannels.push(tag);
    return tag;
  });

  // 3. Parse & resolve user mentions: <@!?userId>
  content = content.replace(/<@!?(\d+)>/g, (match, userId) => {
    if (selfUserId && userId === selfUserId) {
      wasBotMentioned = true;
      return ''; // Strip bot's own mention from body
    }
    const member = guild?.members?.cache.get(userId);
    const user = client.users?.cache.get(userId);
    const name = member?.displayName || user?.globalName || user?.username || userId;
    const tag = `@${name}`;
    if (!taggedUsers.includes(tag)) taggedUsers.push(tag);
    return tag;
  });

  // Clean extra whitespace
  content = content.trim();

  return {
    cleanContent: content,
    taggedUsers,
    taggedRoles,
    taggedChannels,
    wasBotMentioned,
  };
}

function buildStructuredIncomingMessage(
  meta: {
    platform: 'discord' | 'whatsapp';
    conversationLabel: string;
    sender: Record<string, string | undefined>;
    isGroupChat: boolean;
    wasMentioned: boolean;
    mentionTargets: MentionTarget[];
    taggedUsers?: string[];
    taggedRoles?: string[];
    taggedChannels?: string[];
    replyContext?: string | null;
  },
  content: string
): string {
  // Compact context header — clear for LLM context
  const senderLabel = meta.sender.name || meta.sender.label || meta.sender.username || 'unknown';
  const senderHandle = meta.sender.username || meta.sender.tag || senderLabel;
  const chatType = meta.isGroupChat ? 'group' : 'DM';

  const parts: string[] = [
    `[context: ${meta.platform} | ${chatType} | ${meta.conversationLabel} | sender: ${senderLabel} (${senderHandle})]`,
    `[bot mentioned: ${meta.wasMentioned ? 'yes' : 'no'}]`,
  ];

  if (meta.taggedUsers && meta.taggedUsers.length > 0) {
    parts.push(`[tagged users: ${meta.taggedUsers.join(', ')}]`);
  }

  if (meta.taggedRoles && meta.taggedRoles.length > 0) {
    parts.push(`[tagged roles: ${meta.taggedRoles.join(', ')}]`);
  }

  if (meta.taggedChannels && meta.taggedChannels.length > 0) {
    parts.push(`[tagged channels: ${meta.taggedChannels.join(', ')}]`);
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

function buildDiscordMentionTargetsFromInteraction(interaction: ChatInputCommandInteraction): MentionTarget[] {
  return dedupeMentionTargets([
    createDiscordMentionTarget(
      interaction.user.id,
      interaction.user.displayName ?? interaction.user.globalName ?? interaction.user.username,
      interaction.user.username,
      interaction.user.tag
    ),
  ]);
}

function buildDiscordMentionTargetsFromMessage(message: Message): MentionTarget[] {
  const targets: MentionTarget[] = [];
  targets.push(
    createDiscordMentionTarget(
      message.author.id,
      message.member?.displayName || message.author.globalName || message.author.username,
      message.author.username,
      message.author.tag
    )
  );

  for (const [, user] of message.mentions.users) {
    if (user.id === message.client.user?.id) continue;
    const member = message.guild?.members.cache.get(user.id);
    targets.push(
      createDiscordMentionTarget(
        user.id,
        member?.displayName || user.globalName || user.username,
        user.username,
        user.tag
      )
    );
  }

  if (message.reference?.messageId && message.mentions.repliedUser) {
    const user = message.mentions.repliedUser;
    const member = message.guild?.members.cache.get(user.id);
    targets.push(
      createDiscordMentionTarget(
        user.id,
        member?.displayName || user.globalName || user.username,
        user.username,
        user.tag
      )
    );
  }

  return dedupeMentionTargets(targets);
}

function createDiscordMentionTarget(id: string, ...labels: Array<string | undefined>): MentionTarget {
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
  const clean = label
    .replace(/^@+/, '')
    .replace(/#\d{4}$/g, '')
    .trim();
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

function resolveDiscordMentions(
  text: string,
  targets: MentionTarget[],
  guild?: Guild | null
): { content: string; userIds: string[]; roleIds: string[] } {
  let content = text;
  const userIds = new Set<string>();
  const roleIds = new Set<string>();

  // 1. Resolve role mentions (@RoleName -> <@&roleId>)
  if (guild?.roles?.cache) {
    const roles = Array.from(guild.roles.cache.values())
      .filter(r => r.name && r.name !== '@everyone')
      .sort((a, b) => b.name.length - a.name.length);

    for (const role of roles) {
      const escaped = escapeRegex(role.name);
      const pattern = new RegExp(`(^|[^\\w<])@${escaped}(?=$|[^\\w>])`, 'giu');
      if (pattern.test(content)) {
        content = content.replace(pattern, (match, prefix) => {
          roleIds.add(role.id);
          return `${prefix}<@&${role.id}>`;
        });
      }
    }
  }

  // 2. Resolve channel mentions (#channel-name -> <#channelId>)
  if (guild?.channels?.cache) {
    const channels = Array.from(guild.channels.cache.values())
      .filter(c => c.name)
      .sort((a, b) => b.name.length - a.name.length);

    for (const chan of channels) {
      const escaped = escapeRegex(chan.name);
      const pattern = new RegExp(`(^|[^\\w<])#${escaped}(?=$|[^\\w>])`, 'giu');
      content = content.replace(pattern, (match, prefix) => {
        return `${prefix}<#${chan.id}>`;
      });
    }
  }

  // 3. Resolve user mentions (@Username / @DisplayName -> <@userId>)
  for (const target of targets) {
    const sortedAliases = [...target.aliases].sort((a, b) => b.length - a.length);
    for (const alias of sortedAliases) {
      const escaped = escapeRegex(alias.replace(/^@+/, ''));
      const pattern = new RegExp(`(^|[^\\w<])@${escaped}(?=$|[^\\w>])`, 'giu');
      if (pattern.test(content)) {
        content = content.replace(pattern, (match, prefix) => {
          userIds.add(target.id);
          return `${prefix}<@${target.id}>`;
        });
      }
    }
  }

  // Also check guild members cache for any other users mentioned by @Name
  if (guild?.members?.cache) {
    const members = Array.from(guild.members.cache.values())
      .filter(m => !m.user.bot)
      .sort((a, b) => {
        const nameA = a.displayName || a.user.username;
        const nameB = b.displayName || b.user.username;
        return nameB.length - nameA.length;
      });

    for (const member of members) {
      const names = [member.displayName, member.user.username, member.user.globalName].filter(Boolean) as string[];
      for (const name of names) {
        const escaped = escapeRegex(name.replace(/^@+/, ''));
        const pattern = new RegExp(`(^|[^\\w<])@${escaped}(?=$|[^\\w>])`, 'giu');
        if (pattern.test(content)) {
          content = content.replace(pattern, (match, prefix) => {
            userIds.add(member.id);
            return `${prefix}<@${member.id}>`;
          });
          break;
        }
      }
    }
  }

  // 4. Collect any raw user or role IDs already in the content
  const userMatches = content.matchAll(/<@!?(\d+)>/g);
  for (const match of userMatches) {
    userIds.add(match[1]);
  }
  const roleMatches = content.matchAll(/<@&(\d+)>/g);
  for (const match of roleMatches) {
    roleIds.add(match[1]);
  }

  return {
    content,
    userIds: Array.from(userIds),
    roleIds: Array.from(roleIds),
  };
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function summarizeAttachments(attachments: Message['attachments']): string {
  if (!attachments || attachments.size === 0) return '';
  return Array.from(attachments.values())
    .map(att => att.name ? `[attachment: ${att.name}]` : '[attachment]')
    .join(' ');
}

function createDiscordProgressState(): DiscordProgressState {
  return createChannelProgressState();
}

function applyEventToDiscordProgress(progress: DiscordProgressState, event: AgentStreamEvent): void {
  applyEventToChannelProgress(progress, event, 'discord');
}

function buildDiscordProgressEmbed(progress: DiscordProgressState, finalContent?: string): EmbedBuilder {
  const elapsedMs = Date.now() - progress.startedAt;
  const { completed, failed, active, total } = getProgressCounts(progress);

  const statusColor = progress.status === 'error' ? 0xD50000
    : progress.status === 'done' ? 0x00C853
    : 0x5E35B1;

  // Clean monospace progress bar
  const barLen = 16;
  const filledN = total > 0 ? Math.round((completed / total) * barLen) : 0;
  const activeN = total > 0 ? Math.min(Math.round((active / total) * barLen), barLen - filledN) : 0;
  const emptyN = Math.max(0, barLen - filledN - activeN);
  const progressBar = total > 0
    ? `${'█'.repeat(filledN + activeN)}${'░'.repeat(emptyN)}`
    : '';

  // Compact header
  const statusLine = `${discordProgressStatusLabel(progress.status)} · ${formatDurationShort(elapsedMs)}`;
  const countsText = total > 0
    ? ` · **${completed}**/${total} done${active ? ` · ${active} active` : ''}${failed ? ` · ${failed} failed` : ''}`
    : '';

  const descParts = [`${statusLine}${countsText}`];
  if (progressBar) {
    descParts.push(`\`${progressBar}\``);
  }
  if (progress.planSummary) {
    descParts.push(`> ${truncateDiscordEmbedField(progress.planSummary, 150)}`);
  }

  const embed = new EmbedBuilder()
    .setAuthor({ name: embedTitleForProgress(progress) })
    .setColor(statusColor)
    .setDescription(descParts.join('\n'))
    .setTimestamp(new Date(progress.startedAt));

  // Tasks — compact list without separate Legend
  if (progress.tasks.length > 0) {
    const taskLines = progress.tasks
      .slice(0, 10)
      .map((task) => {
        const icon = discordTaskStatusIcon(task.status);
        const summary = task.summary
          ? ` — ${truncateDiscordEmbedField(task.summary.replace(/\n/g, ' '), 80)}`
          : '';
        return `${icon} **${task.title}**${summary}`;
      });
    embed.addFields({
      name: 'Tasks',
      value: truncateDiscordEmbedField(taskLines.join('\n'), 1024),
      inline: false,
    });
  }

  // Activity — only when actively working
  if (progress.status !== 'done' && progress.status !== 'error') {
    const activityParts: string[] = [];
    if (progress.currentTaskLabel) {
      activityParts.push(`▸ ${progress.currentTaskLabel}`);
    }
    if (progress.recentTools.length > 0) {
      activityParts.push(...progress.recentTools.slice(-3).map(l => `\`·\` ${l}`));
    }
    if (activityParts.length > 0) {
      embed.addFields({
        name: 'Activity',
        value: truncateDiscordEmbedField(activityParts.join('\n'), 1024),
        inline: false,
      });
    }
  }

  // Error / outcome
  if (progress.status === 'error' && progress.error) {
    embed.addFields({
      name: '⚠ Error',
      value: truncateDiscordEmbedField(progress.error, 1024),
      inline: false,
    });
  }

  if (finalContent && progress.status === 'done') {
    const preview = formatProgressPreview(finalContent, 'discord', 280);
    if (preview) {
      embed.addFields({
        name: 'Outcome',
        value: truncateDiscordEmbedField(preview, 1024),
        inline: false,
      });
    }
  }

  return embed;
}

function discordProgressStatusLabel(status: DiscordProgressState['status']): string {
  return formatProgressStatusLabel(status, 'discord');
}

function discordTaskStatusIcon(status: string): string {
  return formatTaskStatusIcon(status, 'discord');
}

function truncateDiscordEmbedField(value: string, max: number): string {
  if (value.length <= max) return value;
  return `${value.slice(0, Math.max(0, max - 3)).trimEnd()}…`;
}

function embedTitleForProgress(progress: DiscordProgressState): string {
  switch (progress.status) {
    case 'planning': return 'LiteClaw · Planning';
    case 'working': return 'LiteClaw · Working';
    case 'done': return 'LiteClaw · Done';
    case 'error': return 'LiteClaw · Error';
    case 'thinking': return 'LiteClaw · Thinking';
    case 'starting':
    default: return 'LiteClaw · Starting';
  }
}


function convertTablesToBullets(text: string): string {
  const lines = text.split('\n');
  const result: string[] = [];
  let inTable = false;
  let headers: string[] = [];

  for (const line of lines) {
    if (line.match(/^\|.*\|$/)) {
      if (line.match(/^\|[\s-:|]+\|$/)) {
        continue;
      }
      const cells = line.split('|').filter(c => c.trim()).map(c => c.trim());
      if (!inTable) {
        headers = cells;
        inTable = true;
      } else {
        const bullet = cells.map((c, i) => `**${headers[i] ?? ''}**: ${c}`).join(' · ');
        result.push(`• ${bullet}`);
      }
    } else {
      inTable = false;
      headers = [];
      result.push(line);
    }
  }

  return result.join('\n');
}

function isRedundantChoiceEcho(text: string): boolean {
  if (!text) return true;
  const trimmed = text.trim();
  if (trimmed.length === 0) return true;

  const lines = trimmed.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  if (lines.length === 0) return true;

  const choiceEchoLines = lines.filter(l =>
    /^\d+[\.\)]\s+/.test(l) ||
    /^[-*•]\s+/.test(l) ||
    /^(once you (choose|pick|select)|please (choose|pick|select)|which (one|dimension|option)|make a choice|pick an option)/i.test(l)
  );

  return choiceEchoLines.length >= Math.ceil(lines.length * 0.7);
}

