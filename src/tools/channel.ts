/**
 * LiteClaw - Channel-native interaction tools
 *
 * These tools let the agent create richer chat UX in channels that support it:
 * - Native WhatsApp polls & Discord interactive buttons
 * - WhatsApp event cards
 * - WhatsApp emoji reactions
 */

import { toolRegistry, ToolContext, ToolResult, type InteractiveChoiceRequest } from '../core/tools.js';
import { parseScheduleTime } from '../core/scheduler.js';

// ─── Interactive Choices (Buttons / Single-choice poll) ───────────────

toolRegistry.register({
  name: 'send_interactive_choices',
  description: 'Send an interactive multi-choice message with clickable buttons (Discord) or single-choice poll (WhatsApp). Use when you want the user to pick an option before proceeding. IMPORTANT: The prompt and options are rendered directly on screen as interactive buttons. DO NOT repeat, list, or summarize the options in your response text. End your turn immediately and wait for the user to choose.',
  category: 'channel',
  parameters: [
    {
      name: 'prompt',
      type: 'string',
      description: 'The question or invitation shown above the choices.',
      required: true,
    },
    {
      name: 'options',
      type: 'array',
      description: 'Up to 5 short labels users can choose from.',
      required: true,
      items: { type: 'string' },
    },
    {
      name: 'responses',
      type: 'object',
      description: 'Optional map from option label to the follow-up message that should be sent when a user picks it.',
      properties: {},
      additionalProperties: { type: 'string' },
    },
  ],
  keywords: ['discord', 'interactive', 'button', 'buttons', 'choose', 'choice', 'choices', 'select', 'poll', 'vote'],
  handler: async (args, context): Promise<ToolResult> => {
    const request = normalizeInteractiveChoiceArgs(args);
    if (!request) {
      return {
        success: false,
        output: 'Invalid interactive choice arguments. Expected a prompt and 1-5 options.',
      };
    }

    if (context.sendInteractiveChoice) {
      const interactionId = await context.sendInteractiveChoice(request);
      return {
        success: true,
        output: `Interactive choice buttons posted with id ${interactionId}. The options are already displayed on the user's screen. DO NOT repeat the prompt or list the options in text. Conclude your turn immediately and wait for the user to click an option.`,
      };
    }

    // Fallback to WhatsApp native single-choice poll if available
    if (context.sendPoll) {
      const pollId = await context.sendPoll({
        name: request.prompt,
        options: request.options,
        selectableCount: 1,
      });
      return {
        success: true,
        output: `Interactive poll posted on WhatsApp with id ${pollId}. The options are already displayed in the poll. DO NOT repeat the prompt or list the options in text. Conclude your turn immediately and wait for the user to vote.`,
      };
    }

    return {
      success: false,
      output: 'Interactive choices and polls are not supported in the current channel.',
    };
  },
});

// ─── Native Polls ────────────────────────────────────────────────────

toolRegistry.register({
  name: 'send_poll',
  description: 'Create and send a poll in WhatsApp or current channel. Use when asked to create a poll, survey, or gather votes from group members.',
  category: 'channel',
  parameters: [
    {
      name: 'prompt',
      type: 'string',
      description: 'The question or topic of the poll.',
      required: true,
    },
    {
      name: 'options',
      type: 'array',
      description: 'List of 2 to 12 choices for users to vote on.',
      required: true,
      items: { type: 'string' },
    },
    {
      name: 'multiple_answers',
      type: 'boolean',
      description: 'If true, voters can select more than one option. Default is false (single choice).',
      required: false,
    },
  ],
  keywords: ['poll', 'vote', 'survey', 'voting', 'whatsapp poll', 'ballot', 'options', 'choose'],
  handler: async (args, context): Promise<ToolResult> => {
    const prompt = typeof args.prompt === 'string' ? args.prompt.trim() : '';
    const rawOptions = Array.isArray(args.options)
      ? args.options
      : typeof args.options === 'string'
        ? safeParseArray(args.options)
        : [];
    const options = rawOptions
      .map(o => typeof o === 'string' ? o.trim() : '')
      .filter(Boolean)
      .slice(0, 12);

    if (!prompt || options.length < 2) {
      return {
        success: false,
        output: 'A poll requires a question (prompt) and at least 2 options.',
      };
    }

    const selectableCount = args.multiple_answers ? 0 : 1;

    if (context.sendPoll) {
      const pollId = await context.sendPoll({
        name: prompt,
        options,
        selectableCount,
      });
      return {
        success: true,
        output: `Native WhatsApp poll created (id: ${pollId}): "${prompt}" with options: ${options.join(', ')}`,
      };
    }

    if (context.sendInteractiveChoice && options.length <= 5) {
      const choiceId = await context.sendInteractiveChoice({
        prompt,
        options,
      });
      return {
        success: true,
        output: `Interactive button poll created (id: ${choiceId}): "${prompt}" with options: ${options.join(', ')}`,
      };
    }

    return {
      success: false,
      output: 'Poll sending is not available in the current channel.',
    };
  },
});

// ─── WhatsApp Event Scheduling ───────────────────────────────────────

toolRegistry.register({
  name: 'schedule_whatsapp_event',
  description: 'Send a native WhatsApp event card into the chat with date, time, title, location, description, or call link.',
  category: 'channel',
  parameters: [
    {
      name: 'title',
      type: 'string',
      description: 'Title of the event.',
      required: true,
    },
    {
      name: 'start_time',
      type: 'string',
      description: 'When the event starts (e.g., "in 2 hours", "tomorrow at 15:00", or ISO string).',
      required: true,
    },
    {
      name: 'end_time',
      type: 'string',
      description: 'Optional end time (e.g. "in 3 hours", or ISO string).',
      required: false,
    },
    {
      name: 'description',
      type: 'string',
      description: 'Optional description or agenda.',
      required: false,
    },
    {
      name: 'location',
      type: 'string',
      description: 'Optional meeting location or address.',
      required: false,
    },
    {
      name: 'call_type',
      type: 'string',
      description: 'Optional call type: "none", "audio", or "video".',
      enum: ['none', 'audio', 'video'],
      required: false,
    },
  ],
  keywords: ['event', 'schedule event', 'whatsapp event', 'calendar', 'meeting', 'meet', 'appointment'],
  handler: async (args, context): Promise<ToolResult> => {
    if (!context.sendEvent) {
      return {
        success: false,
        output: 'Native event creation is only supported on WhatsApp.',
      };
    }

    const title = typeof args.title === 'string' ? args.title.trim() : '';
    if (!title) {
      return {
        success: false,
        output: 'Event title is required.',
      };
    }

    const startMs = parseScheduleTime(args.start_time);
    if (!startMs) {
      return {
        success: false,
        output: `Could not parse start_time "${args.start_time}". Use relative formats like "in 2 hours", "tomorrow at 14:00", or ISO timestamps.`,
      };
    }

    const startDate = new Date(startMs);
    let endDate: Date | undefined;
    if (args.end_time) {
      const endMs = parseScheduleTime(args.end_time);
      if (endMs && endMs > startMs) {
        endDate = new Date(endMs);
      }
    }

    const callType = args.call_type === 'audio' || args.call_type === 'video' ? args.call_type : undefined;

    const eventId = await context.sendEvent({
      name: title,
      description: args.description?.trim(),
      startDate,
      endDate,
      location: args.location?.trim(),
      call: callType,
    });

    return {
      success: true,
      output: `WhatsApp event card "${title}" created for ${startDate.toLocaleString()} (id: ${eventId}).`,
    };
  },
});

// ─── WhatsApp Reactions ──────────────────────────────────────────────

toolRegistry.register({
  name: 'whatsapp_react',
  description: 'React with an emoji to the incoming message on WhatsApp (e.g. 👍, ❤️, ⏳, ✅, 🔥).',
  category: 'channel',
  parameters: [
    {
      name: 'emoji',
      type: 'string',
      description: 'Single emoji character to react with (e.g. "👍", "✅", "🔥").',
      required: true,
    },
  ],
  keywords: ['react', 'reaction', 'emoji', 'whatsapp react'],
  handler: async (args, context): Promise<ToolResult> => {
    if (!context.react) {
      return {
        success: false,
        output: 'Reactions are not supported or available in the current channel.',
      };
    }

    const emoji = typeof args.emoji === 'string' ? args.emoji.trim() : '';
    if (!emoji) {
      return {
        success: false,
        output: 'Emoji is required for reaction.',
      };
    }

    await context.react(emoji);
    return {
      success: true,
      output: `Reacted with ${emoji} to the message.`,
    };
  },
});

// ─── Utilities ───────────────────────────────────────────────────────

function normalizeInteractiveChoiceArgs(args: Record<string, any>): InteractiveChoiceRequest | null {
  const prompt = typeof args.prompt === 'string' ? args.prompt.trim() : '';
  const rawOptions = Array.isArray(args.options)
    ? args.options
    : typeof args.options === 'string'
      ? safeParseArray(args.options)
      : [];
  const options = rawOptions
    .map(item => typeof item === 'string' ? item.trim() : '')
    .filter(Boolean)
    .slice(0, 5);

  if (!prompt || options.length === 0) {
    return null;
  }

  const responses = normalizeResponses(args.responses);
  return { prompt, options, responses };
}

function normalizeResponses(value: unknown): Record<string, string> | undefined {
  const parsed = typeof value === 'string' ? safeParseObject(value) : value;
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return undefined;

  const normalized = Object.fromEntries(
    Object.entries(parsed)
      .filter(([, response]) => typeof response === 'string' && response.trim().length > 0)
      .map(([option, response]) => [option.trim(), String(response).trim()])
  );

  return Object.keys(normalized).length > 0 ? normalized : undefined;
}

function safeParseArray(raw: string): unknown[] {
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return raw.split(',').map(part => part.trim()).filter(Boolean);
  }
}

function safeParseObject(raw: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : null;
  } catch {
    return null;
  }
}
