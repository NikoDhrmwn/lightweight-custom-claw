/**
 * LiteClaw — SQLite Memory Store
 * 
 * Per-channel, per-user conversation history with keyword search.
 * Compatible with import from OpenClaw's memory format.
 */

import Database from 'better-sqlite3';
import { existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { createLogger } from '../logger.js';
import { estimateTokens, resolveContextThresholds } from './context.js';
import type { TaskPlan } from './tasks.js';

const log = createLogger('memory');

// ─── Types ───────────────────────────────────────────────────────────

export interface MemoryEntry {
  id?: number;
  sessionKey: string;
  role: string;
  content: string;
  reasoningContent?: string;
  timestamp: number;
  metadata?: string;
}

export interface SessionInfo {
  sessionKey: string;
  sessionName?: string;
  channelType?: string;
  isGroup?: boolean;
  messageCount: number;
  lastActivity: number;
  firstActivity: number;
  userIdentifier?: string;
  estimatedTokens?: number;
}

export interface SessionMetrics {
  sessionKey: string;
  messageCount: number;
  estimatedTokens: number;
  imageCount: number;
  lastActivity: number | null;
  maxContextTokens?: number;
  budgetTokens?: number;
  softThresholdTokens?: number;
  usagePct?: number;
  compactionThresholdPct?: number;
  isNearCompaction?: boolean;
}

export interface ScheduledTask {
  id: string;
  sessionKey: string;
  channelType: string;
  channelTarget: string;
  triggerAtMs: number;
  taskType: 'reminder' | 'agent_prompt';
  payload: string;
  repeatIntervalMs?: number;
  status: 'pending' | 'running' | 'completed' | 'cancelled' | 'failed';
  createdAtMs: number;
}

export interface StoredTaskPlan {
  id: string;
  sessionKey: string;
  goal: string;
  status: string;
  plan: TaskPlan;
  createdAt: number;
  updatedAt: number;
}

export interface UsageStats {
  days: number;
  totalMessages: number;
  totalSessions: number;
  estimatedTokens: number;
  userMessages: number;
  assistantMessages: number;
  topSessions: Array<{ sessionKey: string; messageCount: number; estimatedTokens: number }>;
  dailyActivity: Array<{ date: string; messages: number; estimatedTokens: number }>;
}

export interface KanbanBoard {
  id: string;
  userKey: string;
  name: string;
  createdAt: number;
}

export interface KanbanCard {
  id: string;
  boardId: string;
  columnName: string;
  title: string;
  description?: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  dueDate?: string;
  createdAt: number;
  updatedAt: number;
}

export interface SkillUsageStat {
  skillName: string;
  count: number;
  lastUsed: number;
}

// ─── Memory Store ────────────────────────────────────────────────────

let defaultMemoryStore: MemoryStore | null = null;

export class MemoryStore {
  private db: Database.Database;

  constructor(dbPath?: string) {
    if (!defaultMemoryStore) {
      defaultMemoryStore = this;
    }
    const dataDir = process.env.LITECLAW_STATE_DIR ??
      join(process.env.USERPROFILE ?? process.env.HOME ?? '.', '.liteclaw');

    if (!existsSync(dataDir)) {
      mkdirSync(dataDir, { recursive: true });
    }

    const finalPath = dbPath ?? join(dataDir, 'memory.sqlite');
    this.db = new Database(finalPath);
    this.initialize();
    log.info({ path: finalPath }, 'Memory store initialized');
  }

  private initialize(): void {
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('synchronous = NORMAL');

    this.db.exec(`
      CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_key TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        metadata TEXT,
        created_at TEXT DEFAULT (datetime('now'))
      );

      CREATE INDEX IF NOT EXISTS idx_messages_session
        ON messages(session_key, timestamp);

      CREATE INDEX IF NOT EXISTS idx_messages_content
        ON messages(content);

      CREATE TABLE IF NOT EXISTS summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_key TEXT NOT NULL,
        summary TEXT NOT NULL,
        messages_summarized INTEGER NOT NULL,
        timestamp INTEGER NOT NULL,
        created_at TEXT DEFAULT (datetime('now'))
      );

      CREATE INDEX IF NOT EXISTS idx_summaries_session
        ON summaries(session_key, timestamp);

      CREATE TABLE IF NOT EXISTS task_plans (
        id TEXT PRIMARY KEY,
        session_key TEXT NOT NULL,
        goal TEXT NOT NULL,
        status TEXT NOT NULL,
        plan_json TEXT NOT NULL,
        created_at_ms INTEGER NOT NULL,
        updated_at_ms INTEGER NOT NULL,
        created_at TEXT DEFAULT (datetime('now'))
      );

      CREATE INDEX IF NOT EXISTS idx_task_plans_session
        ON task_plans(session_key, updated_at_ms DESC);

      CREATE TABLE IF NOT EXISTS scheduled_tasks (
        id TEXT PRIMARY KEY,
        session_key TEXT NOT NULL,
        channel_type TEXT NOT NULL,
        channel_target TEXT NOT NULL,
        trigger_at_ms INTEGER NOT NULL,
        task_type TEXT NOT NULL,
        payload TEXT NOT NULL,
        repeat_interval_ms INTEGER DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'pending',
        created_at_ms INTEGER NOT NULL,
        created_at TEXT DEFAULT (datetime('now'))
      );

      CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_trigger
        ON scheduled_tasks(status, trigger_at_ms);

      -- FTS5 full text search for cross-session recall
      CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        content,
        role,
        session_key UNINDEXED,
        content_rowid UNINDEXED
      );

      -- Kanban Board Schema
      CREATE TABLE IF NOT EXISTS kanban_boards (
        id TEXT PRIMARY KEY,
        user_key TEXT NOT NULL,
        name TEXT NOT NULL,
        created_at_ms INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS kanban_cards (
        id TEXT PRIMARY KEY,
        board_id TEXT NOT NULL,
        column_name TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT,
        priority TEXT DEFAULT 'medium',
        due_date TEXT,
        created_at_ms INTEGER NOT NULL,
        updated_at_ms INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_kanban_cards_board
        ON kanban_cards(board_id, column_name);

      -- Skill Usage Tracking
      CREATE TABLE IF NOT EXISTS skill_usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        skill_name TEXT NOT NULL,
        session_key TEXT NOT NULL,
        query TEXT,
        timestamp INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_skill_usage_name
        ON skill_usage(skill_name, timestamp DESC);

      -- Sessions Metadata Table
      CREATE TABLE IF NOT EXISTS sessions (
        session_key TEXT PRIMARY KEY,
        session_name TEXT NOT NULL,
        channel_type TEXT,
        channel_target TEXT,
        is_group INTEGER DEFAULT 0,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_sessions_updated
        ON sessions(updated_at DESC);
    `);

    // FTS sync triggers
    this.db.exec(`
      CREATE TRIGGER IF NOT EXISTS trg_messages_fts_insert AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(content, role, session_key, content_rowid)
        VALUES (new.content, new.role, new.session_key, new.id);
      END;

      CREATE TRIGGER IF NOT EXISTS trg_messages_fts_delete AFTER DELETE ON messages BEGIN
        DELETE FROM messages_fts WHERE content_rowid = old.id;
      END;
    `);

    // FTS backfill if needed
    try {
      const ftsCount = (this.db.prepare('SELECT count(*) as c FROM messages_fts').get() as any)?.c ?? 0;
      if (ftsCount === 0) {
        this.db.exec(`
          INSERT INTO messages_fts(content, role, session_key, content_rowid)
          SELECT content, role, session_key, id FROM messages;
        `);
      }
    } catch {}

    // ─── Migrations ──────────────────────────────────────────────
    try {
      this.db.exec('ALTER TABLE messages ADD COLUMN reasoning_content TEXT;');
    } catch (e) {
      // Column already exists or table doesn't exist yet
    }
  }

  /**
   * Save a message to memory.
   */
  saveMessage(entry: MemoryEntry): void {
    const stmt = this.db.prepare(`
      INSERT INTO messages (session_key, role, content, reasoning_content, timestamp, metadata)
      VALUES (?, ?, ?, ?, ?, ?)
    `);
    stmt.run(
      entry.sessionKey,
      entry.role,
      entry.content,
      entry.reasoningContent ?? null,
      entry.timestamp,
      entry.metadata ?? null
    );
  }

  /**
   * Get recent messages for a session.
   */
  getHistory(sessionKey: string, limit: number = 20): MemoryEntry[] {
    const stmt = this.db.prepare(`
      SELECT id, session_key as sessionKey, role, content, reasoning_content as reasoningContent, timestamp, metadata
      FROM messages
      WHERE session_key = ?
      ORDER BY timestamp DESC
      LIMIT ?
    `);
    const rows = stmt.all(sessionKey, limit) as MemoryEntry[];
    return rows.reverse(); // Oldest first
  }

  /**
   * Search messages by keyword across all sessions.
   */
  search(query: string, limit: number = 10): MemoryEntry[] {
    const stmt = this.db.prepare(`
      SELECT id, session_key as sessionKey, role, content, timestamp, metadata
      FROM messages
      WHERE content LIKE ?
      ORDER BY timestamp DESC
      LIMIT ?
    `);
    return stmt.all(`%${query}%`, limit) as MemoryEntry[];
  }

  /**
   * High-speed FTS5 full-text search with relevance ranking.
   */
  searchFTS(query: string, limit: number = 10, sessionKey?: string): MemoryEntry[] {
    const cleaned = query.replace(/["*]/g, '').trim();
    if (!cleaned) return [];
    const ftsQuery = cleaned
      .split(/\s+/)
      .filter(w => w.length > 0)
      .map(w => `"${w.replace(/"/g, '""')}"*`)
      .join(' ');

    if (!ftsQuery) return [];

    try {
      if (sessionKey) {
        const stmt = this.db.prepare(`
          SELECT m.id, m.session_key as sessionKey, m.role, m.content, m.reasoning_content as reasoningContent, m.timestamp, m.metadata
          FROM messages_fts f
          JOIN messages m ON m.id = f.content_rowid
          WHERE messages_fts MATCH ? AND f.session_key = ?
          ORDER BY rank
          LIMIT ?
        `);
        return stmt.all(ftsQuery, sessionKey, limit) as MemoryEntry[];
      } else {
        const stmt = this.db.prepare(`
          SELECT m.id, m.session_key as sessionKey, m.role, m.content, m.reasoning_content as reasoningContent, m.timestamp, m.metadata
          FROM messages_fts f
          JOIN messages m ON m.id = f.content_rowid
          WHERE messages_fts MATCH ?
          ORDER BY rank
          LIMIT ?
        `);
        return stmt.all(ftsQuery, limit) as MemoryEntry[];
      }
    } catch {
      return this.search(query, limit);
    }
  }

  /**
   * Save a compaction summary.
   */
  saveSummary(sessionKey: string, summary: string, messageCount: number): void {
    const stmt = this.db.prepare(`
      INSERT INTO summaries (session_key, summary, messages_summarized, timestamp)
      VALUES (?, ?, ?, ?)
    `);
    stmt.run(sessionKey, summary, messageCount, Date.now());
  }

  /**
   * Get the latest summary for a session.
   */
  getLatestSummary(sessionKey: string): string | null {
    const stmt = this.db.prepare(`
      SELECT summary FROM summaries
      WHERE session_key = ?
      ORDER BY timestamp DESC
      LIMIT 1
    `);
    const row = stmt.get(sessionKey) as { summary: string } | undefined;
    return row?.summary ?? null;
  }

  saveTaskPlan(sessionKey: string, plan: TaskPlan): void {
    const stmt = this.db.prepare(`
      INSERT INTO task_plans (id, session_key, goal, status, plan_json, created_at_ms, updated_at_ms)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        session_key = excluded.session_key,
        goal = excluded.goal,
        status = excluded.status,
        plan_json = excluded.plan_json,
        created_at_ms = excluded.created_at_ms,
        updated_at_ms = excluded.updated_at_ms
    `);

    stmt.run(
      plan.id,
      sessionKey,
      plan.goal,
      plan.status,
      JSON.stringify(plan),
      plan.createdAt,
      plan.updatedAt,
    );
  }

  getLatestTaskPlan(sessionKey: string): StoredTaskPlan | null {
    const stmt = this.db.prepare(`
      SELECT id, session_key as sessionKey, goal, status, plan_json as planJson, created_at_ms as createdAt, updated_at_ms as updatedAt
      FROM task_plans
      WHERE session_key = ?
      ORDER BY updated_at_ms DESC
      LIMIT 1
    `);

    const row = stmt.get(sessionKey) as {
      id: string;
      sessionKey: string;
      goal: string;
      status: string;
      planJson: string;
      createdAt: number;
      updatedAt: number;
    } | undefined;

    if (!row) return null;

    try {
      return {
        id: row.id,
        sessionKey: row.sessionKey,
        goal: row.goal,
        status: row.status,
        plan: JSON.parse(row.planJson) as TaskPlan,
        createdAt: row.createdAt,
        updatedAt: row.updatedAt,
      };
    } catch {
      return null;
    }
  }

  /**
   * Upsert a session record with a human-readable name (e.g. group name or server > channel).
   */
  upsertSession(sessionKey: string, data: {
    sessionName: string;
    channelType?: string;
    channelTarget?: string;
    isGroup?: boolean;
  }): void {
    const now = Date.now();
    const isGroupInt = data.isGroup ? 1 : 0;
    const stmt = this.db.prepare(`
      INSERT INTO sessions (session_key, session_name, channel_type, channel_target, is_group, created_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(session_key) DO UPDATE SET
        session_name = excluded.session_name,
        channel_type = COALESCE(excluded.channel_type, sessions.channel_type),
        channel_target = COALESCE(excluded.channel_target, sessions.channel_target),
        is_group = excluded.is_group,
        updated_at = excluded.updated_at
    `);
    stmt.run(
      sessionKey,
      data.sessionName,
      data.channelType ?? null,
      data.channelTarget ?? null,
      isGroupInt,
      now,
      now
    );
  }

  /**
   * Get a session record by its key.
   */
  getSession(sessionKey: string): { sessionKey: string; sessionName: string; channelType?: string; isGroup: boolean } | null {
    const stmt = this.db.prepare(`
      SELECT session_key as sessionKey, session_name as sessionName, channel_type as channelType, is_group as isGroup
      FROM sessions
      WHERE session_key = ?
    `);
    const row = stmt.get(sessionKey) as any;
    if (!row) return null;
    return {
      sessionKey: row.sessionKey,
      sessionName: row.sessionName,
      channelType: row.channelType,
      isGroup: Boolean(row.isGroup),
    };
  }

  /**
   * List all sessions with their human-readable sessionName.
   */
  listSessions(): SessionInfo[] {
    const stmt = this.db.prepare(`
      SELECT
        m.session_key as sessionKey,
        s.session_name as sessionName,
        s.channel_type as channelType,
        s.is_group as isGroup,
        COUNT(*) as messageCount,
        MAX(m.timestamp) as lastActivity,
        MIN(m.timestamp) as firstActivity,
        (SELECT metadata FROM messages m2 WHERE m2.session_key = m.session_key ORDER BY timestamp DESC LIMIT 1) as latestMetadata
      FROM messages m
      LEFT JOIN sessions s ON s.session_key = m.session_key
      GROUP BY m.session_key
      ORDER BY lastActivity DESC
    `);
    const rows = stmt.all() as any[];
    return rows.map(r => {
      let identifier: string | undefined = undefined;
      let metaSessionName: string | undefined = undefined;
      let metaIsGroup: boolean | undefined = undefined;
      let metaChannelType: string | undefined = undefined;
      if (r.latestMetadata) {
        try {
          const meta = JSON.parse(r.latestMetadata);
          identifier = meta.userIdentifier;
          metaSessionName = meta.sessionName;
          metaIsGroup = meta.isGroup;
          metaChannelType = meta.channelType;
        } catch(e) {}
      }

      // Priority: sessions table name > message metadata name > clean fallback
      const resolvedName = r.sessionName || metaSessionName || formatFallbackSessionName(r.sessionKey, identifier);

      return {
        sessionKey: r.sessionKey,
        sessionName: resolvedName,
        channelType: r.channelType || metaChannelType,
        isGroup: r.isGroup !== null && r.isGroup !== undefined ? Boolean(r.isGroup) : metaIsGroup,
        messageCount: r.messageCount,
        lastActivity: r.lastActivity,
        firstActivity: r.firstActivity,
        userIdentifier: identifier,
        estimatedTokens: this.getSessionMetrics(r.sessionKey).estimatedTokens
      };
    });
  }

  getSessionMetrics(sessionKey: string): SessionMetrics {
    const stmt = this.db.prepare(`
      SELECT
        session_key as sessionKey,
        COUNT(*) as messageCount,
        MAX(timestamp) as lastActivity
      FROM messages
      WHERE session_key = ?
      GROUP BY session_key
    `);
    const summary = stmt.get(sessionKey) as {
      sessionKey: string;
      messageCount: number;
      lastActivity: number | null;
    } | undefined;

    const messages = this.getHistory(sessionKey, 1000);
    let estimatedTokens = 0;
    let imageCount = 0;

    for (const message of messages) {
      estimatedTokens += 4;
      estimatedTokens += estimateTokens(message.content ?? '');

      if (!message.metadata) continue;
      try {
        const meta = JSON.parse(message.metadata);
        const count = Number(meta.imageCount || 0);
        if (count > 0) {
          imageCount += count;
          estimatedTokens += count * 300;
        }
      } catch {
        // Ignore invalid metadata.
      }
    }

    const thresholds = resolveContextThresholds();
    const usagePct = thresholds.budgetTokens > 0
      ? Math.round((estimatedTokens / thresholds.budgetTokens) * 100)
      : 0;
    const isNearCompaction = estimatedTokens >= thresholds.softThresholdTokens;

    return {
      sessionKey,
      messageCount: summary?.messageCount ?? 0,
      estimatedTokens,
      imageCount,
      lastActivity: summary?.lastActivity ?? null,
      maxContextTokens: thresholds.maxContextTokens,
      budgetTokens: thresholds.budgetTokens,
      softThresholdTokens: thresholds.softThresholdTokens,
      usagePct,
      compactionThresholdPct: thresholds.compactionPct,
      isNearCompaction,
    };
  }

  // ─── Scheduled Tasks ───────────────────────────────────────────────

  createScheduledTask(task: {
    id?: string;
    sessionKey: string;
    channelType: string;
    channelTarget: string;
    triggerAtMs: number;
    taskType: 'reminder' | 'agent_prompt';
    payload: string;
    repeatIntervalMs?: number;
  }): ScheduledTask {
    const id = task.id || `task_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    const createdAtMs = Date.now();
    const status = 'pending';
    const repeatIntervalMs = task.repeatIntervalMs || 0;

    const stmt = this.db.prepare(`
      INSERT INTO scheduled_tasks (
        id, session_key, channel_type, channel_target,
        trigger_at_ms, task_type, payload, repeat_interval_ms, status, created_at_ms
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    stmt.run(
      id,
      task.sessionKey,
      task.channelType,
      task.channelTarget,
      task.triggerAtMs,
      task.taskType,
      task.payload,
      repeatIntervalMs,
      status,
      createdAtMs
    );

    return {
      id,
      sessionKey: task.sessionKey,
      channelType: task.channelType,
      channelTarget: task.channelTarget,
      triggerAtMs: task.triggerAtMs,
      taskType: task.taskType,
      payload: task.payload,
      repeatIntervalMs,
      status,
      createdAtMs,
    };
  }

  getDueScheduledTasks(nowMs: number = Date.now()): ScheduledTask[] {
    const stmt = this.db.prepare(`
      SELECT
        id,
        session_key as sessionKey,
        channel_type as channelType,
        channel_target as channelTarget,
        trigger_at_ms as triggerAtMs,
        task_type as taskType,
        payload,
        repeat_interval_ms as repeatIntervalMs,
        status,
        created_at_ms as createdAtMs
      FROM scheduled_tasks
      WHERE status = 'pending' AND trigger_at_ms <= ?
      ORDER BY trigger_at_ms ASC
    `);

    return stmt.all(nowMs) as ScheduledTask[];
  }

  getScheduledTask(id: string): ScheduledTask | null {
    const stmt = this.db.prepare(`
      SELECT
        id,
        session_key as sessionKey,
        channel_type as channelType,
        channel_target as channelTarget,
        trigger_at_ms as triggerAtMs,
        task_type as taskType,
        payload,
        repeat_interval_ms as repeatIntervalMs,
        status,
        created_at_ms as createdAtMs
      FROM scheduled_tasks
      WHERE id = ?
    `);

    const row = stmt.get(id) as ScheduledTask | undefined;
    return row ?? null;
  }

  updateScheduledTaskStatus(id: string, status: ScheduledTask['status']): void {
    const stmt = this.db.prepare(`
      UPDATE scheduled_tasks
      SET status = ?
      WHERE id = ?
    `);
    stmt.run(status, id);
  }

  rescheduleTask(id: string, nextTriggerAtMs: number): void {
    const stmt = this.db.prepare(`
      UPDATE scheduled_tasks
      SET trigger_at_ms = ?, status = 'pending'
      WHERE id = ?
    `);
    stmt.run(nextTriggerAtMs, id);
  }

  listScheduledTasks(filter?: { sessionKey?: string; status?: string; limit?: number }): ScheduledTask[] {
    let sql = `
      SELECT
        id,
        session_key as sessionKey,
        channel_type as channelType,
        channel_target as channelTarget,
        trigger_at_ms as triggerAtMs,
        task_type as taskType,
        payload,
        repeat_interval_ms as repeatIntervalMs,
        status,
        created_at_ms as createdAtMs
      FROM scheduled_tasks
    `;

    const conditions: string[] = [];
    const params: any[] = [];

    if (filter?.sessionKey) {
      conditions.push('session_key = ?');
      params.push(filter.sessionKey);
    }

    if (filter?.status) {
      conditions.push('status = ?');
      params.push(filter.status);
    }

    if (conditions.length > 0) {
      sql += ` WHERE ${conditions.join(' AND ')}`;
    }

    sql += ` ORDER BY trigger_at_ms ASC`;

    if (filter?.limit) {
      sql += ` LIMIT ?`;
      params.push(filter.limit);
    }

    const stmt = this.db.prepare(sql);
    return stmt.all(...params) as ScheduledTask[];
  }

  cancelScheduledTask(id: string): boolean {
    const stmt = this.db.prepare(`
      UPDATE scheduled_tasks
      SET status = 'cancelled'
      WHERE id = ? AND status = 'pending'
    `);
    const result = stmt.run(id);
    return result.changes > 0;
  }

  /**
   * Prune old messages beyond retention limit.
   */
  prune(maxAgeMs: number = 30 * 24 * 60 * 60 * 1000): number {
    const cutoff = Date.now() - maxAgeMs;
    const stmt = this.db.prepare(`
      DELETE FROM messages WHERE timestamp < ?
    `);
    const result = stmt.run(cutoff);
    if (result.changes > 0) {
      log.info({ pruned: result.changes }, 'Pruned old messages');
    }
    return result.changes;
  }

  /**
   * Clear all messages for a session.
   */
  clearSession(sessionKey: string): void {
    this.db.prepare('DELETE FROM messages WHERE session_key = ?').run(sessionKey);
    this.db.prepare('DELETE FROM summaries WHERE session_key = ?').run(sessionKey);
    this.db.prepare('DELETE FROM task_plans WHERE session_key = ?').run(sessionKey);
    this.db.prepare('DELETE FROM sessions WHERE session_key = ?').run(sessionKey);
  }

  /**
   * Delete the last N messages for a session.
   */
  deleteLastMessages(sessionKey: string, count: number = 1): number {
    const stmt = this.db.prepare(`
      DELETE FROM messages 
      WHERE id IN (
        SELECT id FROM messages 
        WHERE session_key = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
      )
    `);
    const result = stmt.run(sessionKey, count);
    this.db.prepare('DELETE FROM task_plans WHERE session_key = ?').run(sessionKey);
    return result.changes;
  }

  /**
   * Get the most recent message sent by the user for this session.
   */
  getLastUserMessage(sessionKey: string): MemoryEntry | null {
    const row = this.db.prepare(`
      SELECT id, session_key as sessionKey, role, content, reasoning_content as reasoningContent, timestamp, metadata
      FROM messages
      WHERE session_key = ? AND role = 'user'
      ORDER BY timestamp DESC
      LIMIT 1
    `).get(sessionKey) as MemoryEntry | undefined;
    return row ?? null;
  }

  /**
   * Revert the last full exchange (last user turn and subsequent assistant messages).
   */
  undoLastExchange(sessionKey: string): { removedCount: number; undoneUserMessage?: string } {
    const lastUser = this.getLastUserMessage(sessionKey);
    if (!lastUser || !lastUser.timestamp) {
      return { removedCount: 0 };
    }
    const res = this.db.prepare(`
      DELETE FROM messages
      WHERE session_key = ? AND timestamp >= ?
    `).run(sessionKey, lastUser.timestamp);
    this.db.prepare('DELETE FROM task_plans WHERE session_key = ?').run(sessionKey);
    return {
      removedCount: res.changes,
      undoneUserMessage: lastUser.content,
    };
  }

  /**
   * Gather aggregated usage analytics over the past N days.
   */
  getUsageStats(days: number = 7): UsageStats {
    const sinceMs = Date.now() - (days * 24 * 60 * 60 * 1000);
    const totalRow = this.db.prepare(`
      SELECT 
        COUNT(*) as totalMessages,
        COUNT(DISTINCT session_key) as totalSessions,
        SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) as userMessages,
        SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) as assistantMessages
      FROM messages
      WHERE timestamp >= ?
    `).get(sinceMs) as any;

    const topSessionsRows = this.db.prepare(`
      SELECT 
        session_key as sessionKey,
        COUNT(*) as messageCount,
        SUM(LENGTH(content) / 4) as estimatedTokens
      FROM messages
      WHERE timestamp >= ?
      GROUP BY session_key
      ORDER BY messageCount DESC
      LIMIT 5
    `).all(sinceMs) as Array<{ sessionKey: string; messageCount: number; estimatedTokens: number }>;

    const dailyRows = this.db.prepare(`
      SELECT 
        DATE(created_at) as date,
        COUNT(*) as messages,
        SUM(LENGTH(content) / 4) as estimatedTokens
      FROM messages
      WHERE timestamp >= ?
      GROUP BY DATE(created_at)
      ORDER BY date ASC
    `).all(sinceMs) as Array<{ date: string; messages: number; estimatedTokens: number }>;

    const totalEstimatedTokens = this.db.prepare(`
      SELECT SUM(LENGTH(content) / 4) as tokens
      FROM messages
      WHERE timestamp >= ?
    `).get(sinceMs) as any;

    return {
      days,
      totalMessages: totalRow?.totalMessages ?? 0,
      totalSessions: totalRow?.totalSessions ?? 0,
      estimatedTokens: Math.round(totalEstimatedTokens?.tokens ?? 0),
      userMessages: totalRow?.userMessages ?? 0,
      assistantMessages: totalRow?.assistantMessages ?? 0,
      topSessions: topSessionsRows.map(s => ({ ...s, estimatedTokens: Math.round(s.estimatedTokens) })),
      dailyActivity: dailyRows.map(d => ({ ...d, estimatedTokens: Math.round(d.estimatedTokens) })),
    };
  }

  // ─── Kanban Board Operations ────────────────────────────────────────

  createKanbanBoard(userKey: string, name: string): KanbanBoard {
    const id = `board_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    const now = Date.now();
    this.db.prepare(`
      INSERT INTO kanban_boards (id, user_key, name, created_at_ms)
      VALUES (?, ?, ?, ?)
    `).run(id, userKey, name, now);
    return { id, userKey, name, createdAt: now };
  }

  listKanbanBoards(userKey: string): KanbanBoard[] {
    const rows = this.db.prepare(`
      SELECT id, user_key as userKey, name, created_at_ms as createdAt
      FROM kanban_boards
      WHERE user_key = ?
      ORDER BY created_at_ms DESC
    `).all(userKey) as KanbanBoard[];
    return rows;
  }

  addKanbanCard(
    boardId: string,
    title: string,
    description: string = '',
    columnName: string = 'todo',
    priority: KanbanCard['priority'] = 'medium',
    dueDate?: string
  ): KanbanCard {
    const id = `card_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    const now = Date.now();
    this.db.prepare(`
      INSERT INTO kanban_cards (id, board_id, column_name, title, description, priority, due_date, created_at_ms, updated_at_ms)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, boardId, columnName, title, description, priority, dueDate ?? null, now, now);
    return {
      id,
      boardId,
      columnName,
      title,
      description,
      priority,
      dueDate,
      createdAt: now,
      updatedAt: now,
    };
  }

  moveKanbanCard(cardId: string, targetColumn: string): boolean {
    const res = this.db.prepare(`
      UPDATE kanban_cards
      SET column_name = ?, updated_at_ms = ?
      WHERE id = ?
    `).run(targetColumn, Date.now(), cardId);
    return res.changes > 0;
  }

  listKanbanCards(boardId: string): KanbanCard[] {
    const rows = this.db.prepare(`
      SELECT 
        id, board_id as boardId, column_name as columnName,
        title, description, priority, due_date as dueDate,
        created_at_ms as createdAt, updated_at_ms as updatedAt
      FROM kanban_cards
      WHERE board_id = ?
      ORDER BY updated_at_ms DESC
    `).all(boardId) as KanbanCard[];
    return rows;
  }

  deleteKanbanCard(cardId: string): boolean {
    const res = this.db.prepare('DELETE FROM kanban_cards WHERE id = ?').run(cardId);
    return res.changes > 0;
  }

  // ─── Skill Usage Tracking ──────────────────────────────────────────

  recordSkillUsage(skillName: string, sessionKey: string, query?: string): void {
    this.db.prepare(`
      INSERT INTO skill_usage (skill_name, session_key, query, timestamp)
      VALUES (?, ?, ?, ?)
    `).run(skillName, sessionKey, query ?? null, Date.now());
  }

  getTopSkills(limit: number = 10): SkillUsageStat[] {
    const rows = this.db.prepare(`
      SELECT skill_name as skillName, COUNT(*) as count, MAX(timestamp) as lastUsed
      FROM skill_usage
      GROUP BY skill_name
      ORDER BY count DESC
      LIMIT ?
    `).all(limit) as SkillUsageStat[];
    return rows;
  }

  close(): void {
    this.db.close();
  }
}

export function getMemoryStore(dbPath?: string): MemoryStore {
  if (!defaultMemoryStore) {
    defaultMemoryStore = new MemoryStore(dbPath);
  }
  return defaultMemoryStore;
}

/**
 * Clean fallback formatter for session names when not explicitly set.
 */
export function formatFallbackSessionName(sessionKey: string, userIdentifier?: string): string {
  if (!sessionKey) return 'Session';
  const parts = sessionKey.split(':');
  const channel = parts[0];
  const target = parts.slice(1).join(':');

  if (channel === 'webui') {
    if (target === 'default' || !target) return 'WebUI Chat';
    return `WebUI (${target})`;
  }
  if (channel === 'cli') {
    return 'CLI Session';
  }
  if (channel === 'discord') {
    if (userIdentifier) return `Discord > ${userIdentifier}`;
    return target ? `Discord > #${target}` : 'Discord Session';
  }
  if (channel === 'whatsapp') {
    if (target.endsWith('@g.us')) {
      return `WhatsApp > Group (${target.split('@')[0]})`;
    }
    if (userIdentifier) return `WhatsApp > ${userIdentifier}`;
    return target ? `WhatsApp > ${target.replace('@s.whatsapp.net', '')}` : 'WhatsApp Session';
  }
  if (userIdentifier) {
    return `${channel} > ${userIdentifier}`;
  }
  return sessionKey;
}

