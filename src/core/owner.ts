/**
 * LiteClaw — Owner & Security Registry
 *
 * Persistently tracks registered instance owners/operators across channels.
 * Supports Absolute (Primary) Owner with approval delegation for secondary owners.
 * Used to ensure only authorized owners can approve confirmations,
 * access sensitive system paths, or execute administrative commands.
 */

import { getMemoryStore } from './memory.js';
import { getConfig } from '../config.js';
import { createLogger } from '../logger.js';

const log = createLogger('owner');

export interface OwnerInfo {
  channelType: string;
  ownerId: string;
  displayName?: string;
  isPrimary?: boolean;
  createdAtMs: number;
}

export class OwnerRegistry {
  private memory = getMemoryStore();

  constructor() {
    this.initialize();
  }

  private initialize(): void {
    const db = (this.memory as any).db;
    if (!db) return;

    db.exec(`
      CREATE TABLE IF NOT EXISTS owners (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_type TEXT NOT NULL,
        owner_id TEXT NOT NULL,
        display_name TEXT,
        is_primary INTEGER DEFAULT 0,
        created_at_ms INTEGER NOT NULL,
        UNIQUE(channel_type, owner_id)
      );
      CREATE INDEX IF NOT EXISTS idx_owners_channel ON owners(channel_type, owner_id);
    `);

    // Migration in case table already existed without is_primary
    try {
      db.exec(`ALTER TABLE owners ADD COLUMN is_primary INTEGER DEFAULT 0;`);
    } catch {
      // Column already exists
    }
  }

  /**
   * Normalize an owner/sender ID for consistent comparisons.
   * Handles WhatsApp JIDs (e.g., stripping device suffixes ":0" and domain aliases).
   */
  normalizeId(rawId: string): string {
    if (!rawId) return '';
    const clean = rawId.trim().toLowerCase();
    // Strip device ID (e.g., 628123:1@s.whatsapp.net -> 628123)
    const withoutDevice = clean.split(':')[0];
    // Strip domains like @s.whatsapp.net, @c.us, @lid
    return withoutDevice.split('@')[0];
  }

  /**
   * Check if a given sender is a registered owner (primary or secondary).
   */
  isOwner(channelType: string, senderId: string): boolean {
    if (!senderId) return false;

    const normalizedSender = this.normalizeId(senderId);
    const channel = channelType.toLowerCase();

    // 1. Check config.yaml overrides
    const config = getConfig();
    const channels = (config.channels ?? {}) as Record<string, any>;
    const channelConfig = channels[channel] ?? {};
    const agentConfig = (config.agent ?? {}) as Record<string, any>;
    const configOwner =
      channelConfig.ownerId ||
      channelConfig.ownerNumber ||
      channelConfig.ownerJid ||
      agentConfig.ownerId;

    if (configOwner && this.normalizeId(String(configOwner)) === normalizedSender) {
      return true;
    }

    // 2. Check allowFrom list if strict whitelist is configured
    const allowFrom = channelConfig.allowFrom;
    if (Array.isArray(allowFrom) && !allowFrom.includes('*')) {
      if (allowFrom.some((p: string) => normalizedSender.includes(this.normalizeId(p)))) {
        if (!this.hasAnyOwner(channel)) {
          return true;
        }
      }
    }

    // 3. Check SQLite database
    const db = (this.memory as any).db;
    if (!db) return false;

    const rows = db.prepare(`SELECT owner_id FROM owners WHERE channel_type = ?`).all(channel) as Array<{ owner_id: string }>;
    for (const row of rows) {
      if (this.normalizeId(row.owner_id) === normalizedSender) {
        return true;
      }
    }

    return false;
  }

  /**
   * Check if a given sender is the Absolute / Primary Owner.
   */
  isPrimaryOwner(channelType: string, senderId: string): boolean {
    if (!senderId) return false;

    const normalizedSender = this.normalizeId(senderId);
    const channel = channelType.toLowerCase();

    // 1. Check config.yaml overrides
    const config = getConfig();
    const channels = (config.channels ?? {}) as Record<string, any>;
    const channelConfig = channels[channel] ?? {};
    const agentConfig = (config.agent ?? {}) as Record<string, any>;
    const configOwner =
      channelConfig.ownerId ||
      channelConfig.ownerNumber ||
      channelConfig.ownerJid ||
      agentConfig.ownerId;

    if (configOwner && this.normalizeId(String(configOwner)) === normalizedSender) {
      return true;
    }

    // 2. Check SQLite database for is_primary = 1
    const db = (this.memory as any).db;
    if (!db) return false;

    const row = db.prepare(`SELECT owner_id FROM owners WHERE channel_type = ? AND is_primary = 1`).get(channel) as { owner_id: string } | undefined;
    if (row && this.normalizeId(row.owner_id) === normalizedSender) {
      return true;
    }

    // If no primary explicitly marked, the earliest registered owner is primary
    const firstRow = db.prepare(`SELECT owner_id FROM owners WHERE channel_type = ? ORDER BY created_at_ms ASC LIMIT 1`).get(channel) as { owner_id: string } | undefined;
    if (firstRow && this.normalizeId(firstRow.owner_id) === normalizedSender) {
      return true;
    }

    return false;
  }

  /**
   * Get the Absolute / Primary Owner for a channel.
   */
  getPrimaryOwner(channelType: string): OwnerInfo | null {
    const channel = channelType.toLowerCase();
    const config = getConfig();
    const channels = (config.channels ?? {}) as Record<string, any>;
    const channelConfig = channels[channel] ?? {};
    const agentConfig = (config.agent ?? {}) as Record<string, any>;
    const configOwner =
      channelConfig.ownerId ||
      channelConfig.ownerNumber ||
      channelConfig.ownerJid ||
      agentConfig.ownerId;

    if (configOwner) {
      return {
        channelType: channel,
        ownerId: String(configOwner),
        displayName: 'Absolute Owner (Config)',
        isPrimary: true,
        createdAtMs: 0,
      };
    }

    const db = (this.memory as any).db;
    if (!db) return null;

    type RawOwnerRow = {
      channelType: string;
      ownerId: string;
      displayName?: string;
      isPrimary: number;
      createdAtMs: number;
    };

    const row = db.prepare(`
      SELECT channel_type as channelType, owner_id as ownerId, display_name as displayName, is_primary as isPrimary, created_at_ms as createdAtMs
      FROM owners
      WHERE channel_type = ? AND is_primary = 1
      LIMIT 1
    `).get(channel) as RawOwnerRow | undefined;

    if (row) {
      return {
        channelType: row.channelType,
        ownerId: row.ownerId,
        displayName: row.displayName,
        isPrimary: true,
        createdAtMs: row.createdAtMs,
      };
    }

    const first = db.prepare(`
      SELECT channel_type as channelType, owner_id as ownerId, display_name as displayName, is_primary as isPrimary, created_at_ms as createdAtMs
      FROM owners
      WHERE channel_type = ?
      ORDER BY created_at_ms ASC
      LIMIT 1
    `).get(channel) as RawOwnerRow | undefined;

    if (first) {
      return {
        channelType: first.channelType,
        ownerId: first.ownerId,
        displayName: first.displayName,
        isPrimary: true,
        createdAtMs: first.createdAtMs,
      };
    }

    return null;
  }

  /**
   * Register a new owner for a channel.
   * If this is the first owner being registered, they are automatically designated Absolute Owner.
   */
  registerOwner(channelType: string, ownerId: string, displayName?: string, isPrimary?: boolean): boolean {
    const channel = channelType.toLowerCase();
    const cleanId = ownerId.trim();
    const db = (this.memory as any).db;
    if (!db) return false;

    const hasAny = this.hasAnyOwner(channel);
    const setPrimary = isPrimary !== undefined ? (isPrimary ? 1 : 0) : (hasAny ? 0 : 1);

    try {
      const stmt = db.prepare(`
        INSERT INTO owners (channel_type, owner_id, display_name, is_primary, created_at_ms)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(channel_type, owner_id) DO UPDATE SET
          display_name = coalesce(excluded.display_name, owners.display_name),
          is_primary = CASE WHEN excluded.is_primary = 1 THEN 1 ELSE owners.is_primary END
      `);
      stmt.run(channel, cleanId, displayName || null, setPrimary, Date.now());
      log.info({ channel, ownerId: cleanId, displayName, isPrimary: setPrimary === 1 }, 'Registered owner');
      return true;
    } catch (err: any) {
      log.error({ channel, ownerId: cleanId, error: err.message }, 'Failed to register owner');
      return false;
    }
  }

  /**
   * Remove an owner from a channel.
   */
  removeOwner(channelType: string, ownerId: string): boolean {
    const channel = channelType.toLowerCase();
    const cleanId = this.normalizeId(ownerId);
    const db = (this.memory as any).db;
    if (!db) return false;

    const rows = db.prepare(`SELECT owner_id FROM owners WHERE channel_type = ?`).all(channel) as Array<{ owner_id: string }>;
    let deleted = 0;
    const delStmt = db.prepare(`DELETE FROM owners WHERE channel_type = ? AND owner_id = ?`);
    for (const r of rows) {
      if (this.normalizeId(r.owner_id) === cleanId) {
        delStmt.run(channel, r.owner_id);
        deleted++;
      }
    }
    return deleted > 0;
  }

  /**
   * Check if any owner is registered for this channel.
   */
  hasAnyOwner(channelType: string): boolean {
    const channel = channelType.toLowerCase();
    const config = getConfig();
    const channels = (config.channels ?? {}) as Record<string, any>;
    const channelConfig = channels[channel] ?? {};
    const agentConfig = (config.agent ?? {}) as Record<string, any>;
    const configOwner =
      channelConfig.ownerId ||
      channelConfig.ownerNumber ||
      channelConfig.ownerJid ||
      agentConfig.ownerId;

    if (configOwner) return true;

    const db = (this.memory as any).db;
    if (!db) return false;

    const row = db.prepare(`SELECT COUNT(*) as count FROM owners WHERE channel_type = ?`).get(channel) as { count: number };
    return (row?.count ?? 0) > 0;
  }

  /**
   * Get all registered owners for a channel (Primary first, then by date).
   */
  getOwners(channelType: string): OwnerInfo[] {
    const channel = channelType.toLowerCase();
    const db = (this.memory as any).db;
    if (!db) return [];

    const rows = db.prepare(`
      SELECT channel_type as channelType, owner_id as ownerId, display_name as displayName, is_primary as isPrimary, created_at_ms as createdAtMs
      FROM owners
      WHERE channel_type = ?
      ORDER BY is_primary DESC, created_at_ms ASC
    `).all(channel) as Array<{ channelType: string; ownerId: string; displayName?: string; isPrimary: number; createdAtMs: number }>;

    return rows.map(r => ({
      channelType: r.channelType,
      ownerId: r.ownerId,
      displayName: r.displayName,
      isPrimary: r.isPrimary === 1,
      createdAtMs: r.createdAtMs,
    }));
  }
}

export const ownerRegistry = new OwnerRegistry();
