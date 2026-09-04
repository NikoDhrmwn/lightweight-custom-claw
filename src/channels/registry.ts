/**
 * LiteClaw — Channel Registry & Dispatcher
 *
 * Central registry allowing channels (WhatsApp, Discord, WebUI) to register
 * their send capabilities. Enables cross-session communication, tools,
 * and the autonomous scheduler to dispatch messages without circular dependencies.
 */

import { createLogger } from '../logger.js';

const log = createLogger('channel-registry');

export interface ChannelCapabilities {
  sendMessage: (target: string, content: string, options?: any) => Promise<any>;
  sendPoll?: (target: string, poll: { name: string; options: string[]; selectableCount?: number }) => Promise<string>;
  sendEvent?: (target: string, event: {
    name: string;
    description?: string;
    startDate: Date;
    endDate?: Date;
    location?: string;
    call?: 'audio' | 'video';
  }) => Promise<string>;
  sendFile?: (target: string, filePath: string, fileName?: string) => Promise<void>;
  react?: (target: string, messageKey: any, emoji: string) => Promise<void>;
}

class ChannelRegistry {
  private channels = new Map<string, ChannelCapabilities>();

  register(channelType: string, capabilities: ChannelCapabilities): void {
    this.channels.set(channelType.toLowerCase(), capabilities);
    log.info({ channelType }, 'Channel registered in registry');
  }

  unregister(channelType: string): void {
    this.channels.delete(channelType.toLowerCase());
    log.info({ channelType }, 'Channel unregistered from registry');
  }

  get(channelType: string): ChannelCapabilities | undefined {
    return this.channels.get(channelType.toLowerCase());
  }

  has(channelType: string): boolean {
    return this.channels.has(channelType.toLowerCase());
  }

  getAllChannelTypes(): string[] {
    return Array.from(this.channels.keys());
  }

  /**
   * Dispatch a message to any registered channel.
   */
  async sendMessage(channelType: string, target: string, content: string, options?: any): Promise<boolean> {
    const channel = this.get(channelType);
    if (!channel) {
      log.warn({ channelType, target }, 'Cannot dispatch message: channel not registered');
      return false;
    }

    try {
      await channel.sendMessage(target, content, options);
      return true;
    } catch (err: any) {
      log.error({ channelType, target, error: err.message }, 'Failed to dispatch message to channel');
      return false;
    }
  }

  /**
   * Dispatch a poll to a registered channel.
   */
  async sendPoll(
    channelType: string,
    target: string,
    poll: { name: string; options: string[]; selectableCount?: number }
  ): Promise<string | null> {
    const channel = this.get(channelType);
    if (!channel?.sendPoll) {
      log.warn({ channelType, target }, 'Channel does not support native polls');
      return null;
    }

    try {
      return await channel.sendPoll(target, poll);
    } catch (err: any) {
      log.error({ channelType, target, error: err.message }, 'Failed to dispatch poll to channel');
      return null;
    }
  }

  /**
   * Dispatch an event to a registered channel.
   */
  async sendEvent(
    channelType: string,
    target: string,
    event: {
      name: string;
      description?: string;
      startDate: Date;
      endDate?: Date;
      location?: string;
      call?: 'audio' | 'video';
    }
  ): Promise<string | null> {
    const channel = this.get(channelType);
    if (!channel?.sendEvent) {
      log.warn({ channelType, target }, 'Channel does not support native events');
      return null;
    }

    try {
      return await channel.sendEvent(target, event);
    } catch (err: any) {
      log.error({ channelType, target, error: err.message }, 'Failed to dispatch event to channel');
      return null;
    }
  }
}

export const channelRegistry = new ChannelRegistry();
