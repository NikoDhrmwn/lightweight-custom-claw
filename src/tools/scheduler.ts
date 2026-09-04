/**
 * LiteClaw — Autonomous Scheduler Tools
 *
 * Tools for scheduling delayed reminders, recurring alarms,
 * and autonomous background agent heartbeats with full context.
 */

import { toolRegistry, ToolResult } from '../core/tools.js';
import { getMemoryStore } from '../core/memory.js';
import { parseScheduleTime } from '../core/scheduler.js';

// ─── Schedule Task / Reminder / Heartbeat ─────────────────────────────

toolRegistry.register({
  name: 'schedule_task',
  description: 'Schedule a future reminder or an autonomous background agent heartbeat to run at a specific time or after a delay.',
  category: 'scheduler',
  parameters: [
    {
      name: 'time',
      type: 'string',
      description: 'When to trigger (e.g., "in 15 minutes", "in 2 hours", "tomorrow at 9:00", "15:30", "30s", or ISO string).',
      required: true,
    },
    {
      name: 'content',
      type: 'string',
      description: 'The reminder text or autonomous instruction to carry out when triggered.',
      required: true,
    },
    {
      name: 'action_type',
      type: 'string',
      description: '"reminder" to simply send the message back to chat, or "agent_prompt" to wake the agent autonomously with full context to run tools and report results.',
      enum: ['reminder', 'agent_prompt'],
      required: false,
    },
    {
      name: 'repeat_interval',
      type: 'string',
      description: 'Optional recurring interval (e.g. "1h", "24h", "1d"). Leave empty for a one-time schedule.',
      required: false,
    },
  ],
  keywords: ['remind', 'reminder', 'schedule', 'heartbeat', 'alarm', 'timer', 'in 10 minutes', 'later', 'autonomous task'],
  handler: async (args, context): Promise<ToolResult> => {
    const timeStr = typeof args.time === 'string' ? args.time.trim() : '';
    const content = typeof args.content === 'string' ? args.content.trim() : '';

    if (!timeStr || !content) {
      return {
        success: false,
        output: 'Both "time" and "content" are required to schedule a task.',
      };
    }

    const triggerAtMs = parseScheduleTime(timeStr);
    if (!triggerAtMs) {
      return {
        success: false,
        output: `Could not parse scheduled time "${timeStr}". Use formats like "in 15 minutes", "in 2 hours", "tomorrow at 9:00", or "15:30".`,
      };
    }

    const sessionKey = context.sessionKey || `${context.channelType}:${context.channelTarget || 'default'}`;
    const channelType = context.channelType || 'whatsapp';
    const channelTarget = context.channelTarget || sessionKey.split(':')[1] || '';
    const taskType = args.action_type === 'agent_prompt' ? 'agent_prompt' : 'reminder';

    let repeatIntervalMs = 0;
    if (args.repeat_interval) {
      const match = String(args.repeat_interval).match(/^(\d+(?:\.\d+)?)\s*(s|m|h|d)$/i);
      if (match) {
        const val = parseFloat(match[1]);
        const unit = match[2].toLowerCase();
        if (unit === 's') repeatIntervalMs = val * 1000;
        else if (unit === 'm') repeatIntervalMs = val * 60 * 1000;
        else if (unit === 'h') repeatIntervalMs = val * 3600 * 1000;
        else if (unit === 'd') repeatIntervalMs = val * 86400 * 1000;
      }
    }

    const memory = getMemoryStore();
    const created = memory.createScheduledTask({
      sessionKey,
      channelType,
      channelTarget,
      triggerAtMs,
      taskType,
      payload: content,
      repeatIntervalMs,
    });

    const triggerDate = new Date(triggerAtMs).toLocaleString();
    const typeLabel = taskType === 'agent_prompt' ? 'Autonomous Heartbeat' : 'Reminder';
    const repeatLabel = repeatIntervalMs > 0 ? ` (repeats every ${args.repeat_interval})` : '';

    return {
      success: true,
      output: `✅ ${typeLabel} scheduled for ${triggerDate}${repeatLabel}.\nID: ${created.id}\nAction: ${content}`,
    };
  },
});

// ─── List Scheduled Tasks ────────────────────────────────────────────

toolRegistry.register({
  name: 'list_scheduled_tasks',
  description: 'List upcoming scheduled tasks, reminders, and autonomous heartbeats.',
  category: 'scheduler',
  parameters: [
    {
      name: 'all',
      type: 'boolean',
      description: 'If true, also include completed and cancelled tasks. Default false.',
      required: false,
    },
  ],
  keywords: ['list tasks', 'my reminders', 'scheduled tasks', 'upcoming tasks', 'active timers'],
  handler: async (args, context): Promise<ToolResult> => {
    const memory = getMemoryStore();
    const filter = args.all ? {} : { status: 'pending' };

    const tasks = memory.listScheduledTasks(filter);

    if (tasks.length === 0) {
      return {
        success: true,
        output: args.all
          ? 'No scheduled tasks found in history.'
          : 'No pending scheduled tasks or reminders.',
      };
    }

    const lines = [`⏰ Scheduled Tasks (${tasks.length}):`];
    for (const t of tasks) {
      const time = new Date(t.triggerAtMs).toLocaleString();
      const type = t.taskType === 'agent_prompt' ? '🤖 Heartbeat' : '🔔 Reminder';
      const repeat = t.repeatIntervalMs ? ` [repeating]` : '';
      lines.push(`• [${t.id}] ${type} at ${time}${repeat} (${t.status}) — "${t.payload}"`);
    }

    return {
      success: true,
      output: lines.join('\n'),
    };
  },
});

// ─── Cancel Scheduled Task ───────────────────────────────────────────

toolRegistry.register({
  name: 'cancel_scheduled_task',
  description: 'Cancel a pending scheduled task or reminder by its ID.',
  category: 'scheduler',
  parameters: [
    {
      name: 'task_id',
      type: 'string',
      description: 'The unique ID of the scheduled task to cancel (e.g. "task_1725350...").',
      required: true,
    },
  ],
  keywords: ['cancel reminder', 'cancel task', 'remove reminder', 'delete timer', 'stop task'],
  handler: async (args): Promise<ToolResult> => {
    const taskId = args.task_id?.trim();
    if (!taskId) {
      return {
        success: false,
        output: 'task_id is required.',
      };
    }

    const memory = getMemoryStore();
    const cancelled = memory.cancelScheduledTask(taskId);

    if (!cancelled) {
      return {
        success: false,
        output: `Could not cancel task "${taskId}". It may not exist or is already completed/cancelled.`,
      };
    }

    return {
      success: true,
      output: `Task "${taskId}" has been cancelled.`,
    };
  },
});
