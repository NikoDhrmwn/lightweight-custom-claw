/**
 * LiteClaw — Autonomous Scheduler Service
 *
 * Persistently monitors and executes scheduled tasks and autonomous heartbeats
 * stored in SQLite. Dispatches directly to WhatsApp, Discord, or WebUI
 * via the ChannelRegistry and AgentEngine.
 */

import { MemoryStore, ScheduledTask } from './memory.js';
import { AgentEngine, AgentRequest } from './engine.js';
import { channelRegistry } from '../channels/registry.js';
import { createLogger } from '../logger.js';

const log = createLogger('scheduler');

export class SchedulerService {
  private memory: MemoryStore;
  private engine: AgentEngine | null = null;
  private intervalTimer: NodeJS.Timeout | null = null;
  private isTicking = false;
  private checkIntervalMs: number;

  constructor(memory: MemoryStore, checkIntervalMs: number = 3000) {
    this.memory = memory;
    this.checkIntervalMs = checkIntervalMs;
  }

  setEngine(engine: AgentEngine): void {
    this.engine = engine;
  }

  start(): void {
    if (this.intervalTimer) return;
    this.intervalTimer = setInterval(() => {
      void this.tick();
    }, this.checkIntervalMs);
    log.info({ intervalMs: this.checkIntervalMs }, 'Autonomous scheduler service started');
  }

  stop(): void {
    if (this.intervalTimer) {
      clearInterval(this.intervalTimer);
      this.intervalTimer = null;
    }
    log.info('Autonomous scheduler service stopped');
  }

  private async tick(): Promise<void> {
    if (this.isTicking) return;
    this.isTicking = true;

    try {
      const now = Date.now();
      const dueTasks = this.memory.getDueScheduledTasks(now);

      for (const task of dueTasks) {
        await this.executeTask(task);
      }
    } catch (err: any) {
      log.error({ error: err.message }, 'Error in scheduler tick');
    } finally {
      this.isTicking = false;
    }
  }

  private async executeTask(task: ScheduledTask): Promise<void> {
    log.info({ id: task.id, type: task.taskType, session: task.sessionKey }, 'Executing scheduled task');
    this.memory.updateScheduledTaskStatus(task.id, 'running');

    try {
      if (task.taskType === 'reminder') {
        const text = `⏰ *Reminder:*\n${task.payload}`;
        await channelRegistry.sendMessage(task.channelType, task.channelTarget, text);
      } else if (task.taskType === 'agent_prompt') {
        if (!this.engine) {
          throw new Error('Agent engine not attached to scheduler');
        }

        const request: AgentRequest = {
          message: `[SCHEDULED AUTONOMOUS HEARTBEAT]\nContext: Scheduled task "${task.id}" has triggered.\nObjective: ${task.payload}\n\nExecute any necessary checks or tools and provide your findings directly.`,
          sessionKey: task.sessionKey,
          channelType: (task.channelType as any) || 'whatsapp',
          channelTarget: task.channelTarget,
        };

        let responseText = '';
        for await (const event of this.engine.processRequest(request)) {
          if (event.type === 'content' && event.content) {
            responseText += event.content;
          }
        }

        if (responseText.trim()) {
          const header = `🤖 *Autonomous Update*\n\n`;
          await channelRegistry.sendMessage(task.channelType, task.channelTarget, header + responseText.trim());
        }
      }

      // Handle recurrence
      if (task.repeatIntervalMs && task.repeatIntervalMs > 0) {
        const nextTime = Date.now() + task.repeatIntervalMs;
        this.memory.rescheduleTask(task.id, nextTime);
        log.info({ id: task.id, nextTrigger: new Date(nextTime).toISOString() }, 'Rescheduled recurring task');
      } else {
        this.memory.updateScheduledTaskStatus(task.id, 'completed');
        log.info({ id: task.id }, 'Scheduled task completed successfully');
      }
    } catch (err: any) {
      log.error({ id: task.id, error: err.message }, 'Scheduled task execution failed');
      this.memory.updateScheduledTaskStatus(task.id, 'failed');
    }
  }
}

/**
 * Parses various natural language and relative time formats into epoch milliseconds.
 *
 * Supported formats:
 * - Relative seconds/minutes/hours/days: "30s", "10m", "2h", "1d", "in 15 minutes", "in 2 hours", "in 30 seconds"
 * - Daily times: "14:30", "2:30pm", "at 16:00"
 * - Tomorrow: "tomorrow at 9:00", "tomorrow 9am"
 * - ISO string: "2026-09-04T10:00:00Z"
 * - Raw epoch timestamp (number or numeric string)
 */
export function parseScheduleTime(input: string | number): number | null {
  if (typeof input === 'number') {
    return input > Date.now() ? input : null;
  }

  const str = input.trim();
  if (!str) return null;

  // Direct epoch timestamp
  if (/^\d{10,13}$/.test(str)) {
    const num = Number(str);
    const ms = str.length === 10 ? num * 1000 : num;
    return ms > Date.now() ? ms : null;
  }

  // Relative format: "in 10 minutes", "10m", "2h", "30s", "1d"
  const relativeMatch = str.match(/^(?:in\s+)?(\d+(?:\.\d+)?)\s*(s|sec|seconds?|m|min|minutes?|h|hr|hours?|d|days?)$/i);
  if (relativeMatch) {
    const value = parseFloat(relativeMatch[1]);
    const unit = relativeMatch[2].toLowerCase();
    let ms = 0;

    if (unit.startsWith('s')) ms = value * 1000;
    else if (unit.startsWith('m')) ms = value * 60 * 1000;
    else if (unit.startsWith('h')) ms = value * 60 * 60 * 1000;
    else if (unit.startsWith('d')) ms = value * 24 * 60 * 60 * 1000;

    return Date.now() + ms;
  }

  // Tomorrow format: "tomorrow at 9:00", "tomorrow 15:00", "tomorrow at 9am"
  const tomorrowMatch = str.match(/^tomorrow(?:\s+at)?\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$/i);
  if (tomorrowMatch) {
    let hours = parseInt(tomorrowMatch[1], 10);
    const minutes = tomorrowMatch[2] ? parseInt(tomorrowMatch[2], 10) : 0;
    const meridian = tomorrowMatch[3]?.toLowerCase();

    if (meridian === 'pm' && hours < 12) hours += 12;
    if (meridian === 'am' && hours === 12) hours = 0;

    const d = new Date();
    d.setDate(d.getDate() + 1);
    d.setHours(hours, minutes, 0, 0);
    return d.getTime();
  }

  // Today time format: "at 15:30", "15:30", "at 3:30pm", "3:30pm"
  const timeTodayMatch = str.match(/^(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$/i);
  if (timeTodayMatch && (timeTodayMatch[2] || timeTodayMatch[3])) {
    let hours = parseInt(timeTodayMatch[1], 10);
    const minutes = timeTodayMatch[2] ? parseInt(timeTodayMatch[2], 10) : 0;
    const meridian = timeTodayMatch[3]?.toLowerCase();

    if (meridian === 'pm' && hours < 12) hours += 12;
    if (meridian === 'am' && hours === 12) hours = 0;

    const d = new Date();
    d.setHours(hours, minutes, 0, 0);

    // If that time today has already passed, schedule for tomorrow
    if (d.getTime() <= Date.now()) {
      d.setDate(d.getDate() + 1);
    }
    return d.getTime();
  }

  // Standard Date parsing (ISO 8601 etc.)
  const parsedDate = Date.parse(str);
  if (!isNaN(parsedDate) && parsedDate > Date.now()) {
    return parsedDate;
  }

  return null;
}
