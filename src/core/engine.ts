/**
 * LiteClaw — ReAct Agent Engine
 * 
 * The core agent loop: Reason → Act → Observe.
 * Designed for small LLMs with lazy tool loading,
 * context-aware trimming, and streaming output.
 */

import { EventEmitter } from 'events';
import { LLMClient, LLMMessage, LLMStreamChunk, LLMToolCall } from './llm.js';
import { ContextManager, estimateTokens } from './context.js';
import { toolRegistry, ToolContext, ToolResult } from './tools.js';
import { MemoryStore } from './memory.js';
import { ConfirmationManager } from './confirmation.js';
import { getConfig, loadSystemPrompt } from '../config.js';
import { createLogger } from '../logger.js';
import { buildSkillPrompt, selectRelevantSkills } from './skills.js';

const log = createLogger('engine');

// ─── Types ───────────────────────────────────────────────────────────

export interface AgentRequest {
  /** The user's message text */
  message: string;
  /** Optional inline images (base64 data URIs) */
  images?: string[];
  /** Session key for conversation tracking */
  sessionKey: string;
  /** Channel info */
  channelType: 'webui' | 'discord' | 'whatsapp' | 'cli';
  channelTarget?: string;
  userIdentifier?: string;
  /** Working directory for file/exec tools */
  workingDir?: string;
  /** Callback to send files to the channel */
  sendFile?: (filePath: string, fileName?: string) => Promise<void>;
  /** Callback to send an interactive prompt to the channel */
  sendInteractiveChoice?: (request: import('./tools.js').InteractiveChoiceRequest) => Promise<string>;
}

export interface MessageMetrics {
  tokens: number;
  durationMs: number;
  tokPerSec: number;
}

export interface AgentStreamEvent {
  type: 'thinking' | 'content' | 'tool_start' | 'tool_result' | 'confirmation' | 'done' | 'error';
  content?: string;
  toolName?: string;
  toolArgs?: Record<string, any>;
  toolResult?: ToolResult;
  confirmationId?: string;
  error?: string;
  metrics?: MessageMetrics;
}

// ─── Agent Engine ────────────────────────────────────────────────────

export class AgentEngine extends EventEmitter {
  private llm: LLMClient;
  private context: ContextManager;
  private memory: MemoryStore;
  private confirmations: ConfirmationManager;
  private sessionLocks = new Map<string, Promise<void>>();

  constructor(
    llm: LLMClient,
    memory: MemoryStore,
    confirmations: ConfirmationManager
  ) {
    super();
    this.llm = llm;
    this.context = new ContextManager(llm);
    this.memory = memory;
    this.confirmations = confirmations;
  }

  /**
   * Get the underlying LLM client.
   */
  getLLMClient(): LLMClient {
    return this.llm;
  }

  private async acquireLock(sessionKey: string): Promise<() => void> {
    const currentLock = this.sessionLocks.get(sessionKey) || Promise.resolve();
    let release!: () => void;
    const nextLock = new Promise<void>(resolve => {
      release = resolve;
    });
    this.sessionLocks.set(sessionKey, currentLock.then(() => nextLock));
    await currentLock;
    return release;
  }

  /**
   * Process a user request through the ReAct loop.
   * Yields AgentStreamEvents for the channel to render.
   */
  async *processRequest(request: AgentRequest): AsyncGenerator<AgentStreamEvent> {
    const release = await this.acquireLock(request.sessionKey);
    try {
      yield* this._processRequest(request);
    } finally {
      release();
    }
  }

  /**
   * Save a user message to memory without triggering processing.
   * Useful for keeping group chat context when the bot isn't addressed.
   */
  saveMessageSilent(request: AgentRequest): void {
    this.memory.saveMessage({
      sessionKey: request.sessionKey,
      role: 'user',
      content: request.message,
      timestamp: Date.now(),
      metadata: buildRequestMetadata(request)
    });
  }

  private async *_processRequest(request: AgentRequest): AsyncGenerator<AgentStreamEvent> {
    const config = getConfig();
    const maxIterations = config.agent?.maxTurns ?? 20;

    // 1. Load system prompt
    const systemPrompt = loadSystemPrompt();
    const activeSkills = config.agent?.skills?.enabled === false
      ? []
      : selectRelevantSkills(request.message, config.agent?.skills?.maxInjected);
    const skillPrompt = buildSkillPrompt(activeSkills);

    // 2. Build user message (with optional images)
    const userMessage = this.buildUserMessage(request);

    // 3. Load conversation history from memory
    const historyLimit = config.agent?.historyMessageLimit ?? 30;
    const rawHistory = this.memory.getHistory(request.sessionKey, historyLimit);
    let history: LLMMessage[] = rawHistory.map(entry => ({
      role: entry.role as 'user' | 'assistant',
      content: entry.content,
    }));

    // 4. Check compaction
    if (this.context.shouldCompact(history)) {
      log.info({ session: request.sessionKey }, 'Triggering context compaction');
      const { summary, remaining } = await this.context.compactHistory(history);
      history = remaining;
      if (summary.content) {
        history.unshift(summary);
        // Save the summary to memory
        this.memory.saveSummary(
          request.sessionKey,
          typeof summary.content === 'string' ? summary.content : '',
          rawHistory.length - remaining.length
        );
      }
    }

    // 5. Append user message
    history.push(userMessage);

    // Save user message to memory
    this.memory.saveMessage({
      sessionKey: request.sessionKey,
      role: 'user',
      content: request.message,
      timestamp: Date.now(),
      metadata: buildRequestMetadata(request)
    });


     // 6. Select relevant tools (lazy loading)
     const toolLoading = config.agent?.toolLoading ?? 'lazy';
     const allTools = toolRegistry.getAll();
     const selectedTools = toolLoading === 'lazy'
       ? toolRegistry.selectRelevant(request.message, 12)
       : allTools;

    const toolDefs = toolRegistry.toLLMToolDefs(selectedTools);
    const toolGuidance = toolRegistry.buildToolGuidance(selectedTools, request.message);

    // 7. Trim history to fit context window
    const systemTokens = estimateTokens(systemPrompt);
    const toolTokens = estimateTokens(JSON.stringify(toolDefs));
    history = this.context.trimHistory(history, systemTokens, toolTokens);

    // 8. Format Tool Definitions into System Prompt (XML ReAct style)
    let finalSystemPrompt = systemPrompt;
    if (skillPrompt) {
      finalSystemPrompt += `\n\n---\n\n${skillPrompt}`;
    }
    if (toolDefs.length > 0) {
      finalSystemPrompt += `\n\n# Available Tools\nYou have access to the following tools:\n<tools>\n`;
      finalSystemPrompt += JSON.stringify(toolDefs, null, 2);
      finalSystemPrompt += `\n</tools>`;
      if (toolGuidance) {
        finalSystemPrompt += `\n\n${toolGuidance}`;
      }
       finalSystemPrompt += `\n\nTo use a tool, output exactly this xml format WITHOUT ANY NARRATION BEFORE OR AFTER:\n<tool_call>\n{"name": "tool_name", "arguments": {"param1": "value"}}\n</tool_call>\n\nCRITICAL: If a tool is needed, DO NOT SAY "I will verify", DO NOT SAY "I need to check", DO NOT NARRATE YOUR PLAN. Emit ONLY the <tool_call> block immediately. No other text. No explanation. No reasoning about tools. Just the tool call block.\n\nAfter emitting a tool call, you will receive a <tool_result> message. You must wait for this before continuing. Use only ONE tool at a time.`;
    }

    // 9. Build message array
    const messages: LLMMessage[] = [
      { role: 'system', content: finalSystemPrompt },
      ...history,
    ];

    // 9. ReAct loop
    let iterations = 0;
    let repairAttempts = 0;
    let consecutiveFailures = 0;
    let fullResponse = '';
    const toolCallHistory = new Set<string>(); // Track "toolName:argsHash" to prevent duplicates

    let totalTokens = 0;
    let allThinkingContent = '';
    const startTime = Date.now();

    while (iterations < maxIterations) {
      iterations++;
      log.debug({ iteration: iterations, messages: messages.length }, 'Agent iteration');

      let assistantContent = '';
      const toolCalls: LLMToolCall[] = [];
      let thinkingContent = '';

      // Stream LLM response
      for await (const chunk of this.llm.streamChat(messages, toolDefs)) {
        switch (chunk.type) {
          case 'thinking':
            thinkingContent += chunk.content ?? '';
            allThinkingContent += chunk.content ?? '';
            totalTokens += estimateTokens(chunk.content ?? '');
            yield { type: 'thinking', content: chunk.content };
            break;

          case 'content':
            assistantContent += chunk.content ?? '';
            totalTokens += estimateTokens(chunk.content ?? '');
            yield { type: 'content', content: chunk.content };
            break;

          case 'tool_call':
            if (chunk.toolCall) {
              toolCalls.push(chunk.toolCall);
            }
            break;

          case 'error':
            yield { type: 'error', error: chunk.error };
            return;

          case 'done':
            break;
        }
      }

      // Fallback: salvage embedded tool calls that slipped through as plain text
      // or reasoning text instead of native tool chunks.
      if (toolCalls.length === 0) {
        const recovered = extractEmbeddedToolCalls([assistantContent, thinkingContent], toolDefs);
        if (recovered.length > 0) {
          toolCalls.push(...recovered);
          assistantContent = stripEmbeddedToolCalls(assistantContent);
        }
      }

      if (toolCalls.length === 0) {
        const inferred = inferSafeToolCall(thinkingContent, request.message, toolDefs);
        if (inferred) {
          toolCalls.push(inferred);
        }
      }

      // If no tool calls, we're done — unless the model stalled after planning.
      if (toolCalls.length === 0) {
        if (!assistantContent.trim() && thinkingContent.trim()) {
          if (repairAttempts < 2) {
            repairAttempts++;
            log.warn('Model stopped after planning without action. Triggering safety net retry.');
            
            // Modify the last message instead of appending a new conversational turn
            // This prevents context blooming across multiple retries
            const lastMsg = messages.pop();
            if (lastMsg) {
              let newContent = typeof lastMsg.content === 'string' ? lastMsg.content : lastMsg.content[0].text;
              messages.push({
                ...lastMsg,
                content: `${newContent}\n\n[SYSTEM REPAIR: Your previous attempt stalled after planning. Do not narrate. Emit exactly one <tool_call> block now, or provide the final answer if no tool is needed.]`
              });
            }
            continue;
          } else {
            yield {
              type: 'error',
              error: 'Model reached stagnation after multiple planning attempts without action.',
            };
            return;
          }
        }

        fullResponse = assistantContent;
        break;
      }

      repairAttempts = 0;

      // ── Guard 1: Duplicate tool call rejection ──
      // Prevent the model from calling the same tool with identical args twice.
      const deduped: LLMToolCall[] = [];
      for (const tc of toolCalls) {
        const sig = `${tc.function.name}:${tc.function.arguments}`;
        if (toolCallHistory.has(sig)) {
          log.warn({ tool: tc.function.name }, 'Duplicate tool call detected, skipping');
          continue;
        }
        toolCallHistory.add(sig);
        deduped.push(tc);
      }

      // If ALL calls were duplicates, treat it as "no tool calls" and finalize.
      if (deduped.length === 0) {
        log.warn('All tool calls were duplicates. Finalizing response.');
        fullResponse = assistantContent;
        break;
      }

       // ── Guard 2: Substantial-answer early exit ──
       // If the model already produced a meaningful text answer AND is now
       // trying to do more tool calls, stop the loop. The answer is done.
       if (assistantContent.trim().length > 500 && iterations > 2) {
         log.info('Model produced substantial content alongside tool calls. Finalizing response.');
         fullResponse = assistantContent;
         break;
       }

      // Add assistant's tool call message to conversation
      messages.push({
        role: 'assistant',
        content: assistantContent || '',
        tool_calls: deduped,
      });

      // Execute tool calls
      let iterationFailures = 0;
      for (const tc of deduped) {
        yield {
          type: 'tool_start',
          toolName: tc.function.name,
          toolArgs: safeParseJSON(tc.function.arguments),
        };

        // Build tool context
        const toolContext: ToolContext = {
          channelType: request.channelType,
          channelTarget: request.channelTarget,
          workingDir: request.workingDir ?? process.cwd(),
          sendFile: request.sendFile,
          sendInteractiveChoice: request.sendInteractiveChoice,
          requestConfirmation: async (description: string) => {
            return this.confirmations.requestConfirmation(
              tc.function.name,
              description,
              request.channelType,
              request.channelTarget
            );
          },
        };

        const result = await toolRegistry.execute(tc, toolContext);

        yield { type: 'tool_result', toolName: tc.function.name, toolResult: result };

        if (!result.success) iterationFailures++;

        // Add tool result to conversation
        messages.push({
          role: 'tool',
          tool_call_id: tc.id,
          content: formatToolResultForModel(tc.function.name, result),
        });
      }

      // ── Guard 3: Consecutive failure circuit breaker ──
      // If every tool call in this iteration failed, increment the counter.
      // After 3 consecutive all-fail iterations, force the model to answer directly.
      if (iterationFailures === deduped.length) {
        consecutiveFailures++;
        log.warn({ consecutiveFailures }, 'All tool calls failed this iteration');
        if (consecutiveFailures >= 3) {
          log.warn('Circuit breaker tripped: 3 consecutive all-fail tool iterations. Forcing direct answer.');
          messages.push({
            role: 'user',
            content: '[SYSTEM: All recent tool attempts have failed. Do NOT call any more tools. Provide your best answer directly based on what you already know.]',
          });
          // Let the model try one more iteration with no tools
          toolDefs.length = 0;
          continue;
        }
      } else {
        consecutiveFailures = 0;
      }

      // Trim context again after tool results
      const currentSystemTokens = estimateTokens(systemPrompt);
      const currentToolTokens = estimateTokens(JSON.stringify(toolDefs));
      // Only trim history portion (skip system message at index 0)
      const historyPortion = messages.slice(1);
      const trimmed = this.context.trimHistory(historyPortion, currentSystemTokens, currentToolTokens);
      messages.length = 1; // Keep system prompt
      messages.push(...trimmed);
    }

    if (iterations >= maxIterations) {
      yield {
        type: 'error',
        error: `Agent reached maximum iterations (${maxIterations}). Stopping to prevent infinite loop.`,
      };
    }

    // Save assistant response to memory
    if (fullResponse || allThinkingContent) {
      const contentToSave = allThinkingContent 
        ? `<think>${allThinkingContent}</think>${fullResponse}`
        : fullResponse;
        
      this.memory.saveMessage({
        sessionKey: request.sessionKey,
        role: 'assistant',
        content: contentToSave,
        timestamp: Date.now(),
      });
    }

    const durationMs = Date.now() - startTime;
    const tokPerSec = durationMs > 0 ? (totalTokens / (durationMs / 1000)) : 0;

    yield { 
      type: 'done',
      metrics: {
        tokens: totalTokens,
        durationMs,
        tokPerSec
      }
    };
  }

  /**
   * Build a user message with optional images (native multimodal).
   */
  private buildUserMessage(request: AgentRequest): LLMMessage {
    if (request.images && request.images.length > 0) {
      // Multimodal message: text + images
      const content: any[] = [
        { type: 'text', text: request.message },
      ];

      for (const img of request.images) {
        content.push({
          type: 'image_url',
          image_url: {
            url: img.startsWith('data:') ? img : `data:image/jpeg;base64,${img}`,
            detail: 'auto',
          },
        });
      }

      return { role: 'user', content };
    }

    return { role: 'user', content: request.message };
  }
}

// ─── Utilities ───────────────────────────────────────────────────────

function buildRequestMetadata(request: AgentRequest): string | undefined {
  const metadata: Record<string, unknown> = {};
  if (request.userIdentifier) metadata.userIdentifier = request.userIdentifier;
  if (request.images?.length) metadata.imageCount = request.images.length;
  return Object.keys(metadata).length > 0 ? JSON.stringify(metadata) : undefined;
}

function safeParseJSON(str: string): Record<string, any> {
  try {
    return JSON.parse(str);
  } catch {
    return { raw: str };
  }
}

function extractEmbeddedToolCalls(
  sources: string[],
  toolDefs: Array<{ function: { name: string } }>
): LLMToolCall[] {
  const toolNames = new Set(toolDefs.map(td => td.function.name));
  const calls: LLMToolCall[] = [];

  for (const source of sources) {
    if (!source) continue;

    // 1. Standard XML Tags: <tool_call>{"name": "...", "arguments": {}}</tool_call>
    const xmlRegex = /<tool_call>([\s\S]*?)<\/tool_call>/g;
    // 2. Gemma Native Format: <|tool_call|>call: {"name": "...", "arguments": {}}
    const nativeRegex = /<\|tool_call\|>call:\s*({[\s\S]*?})(?:\n|$)/g;
    const nativeRegexAlternate = /<\|tool_call>call:\s*({[\s\S]*?})(?:\n|$)/g;
    // 3. Markdowns or raw JSON blocks: ```json ... ``` or just { ... }
    const jsonBlockRegex = /```(?:json)?\s*({[\s\S]*?})\s*```/g;

    const matchers = [
      { regex: xmlRegex, group: 1 },
      { regex: nativeRegex, group: 1 },
      { regex: nativeRegexAlternate, group: 1 },
      { regex: jsonBlockRegex, group: 1 }
    ];

    for (const { regex, group } of matchers) {
      const matches = source.matchAll(regex);
      for (const match of matches) {
        try {
          const rawJson = match[group].trim();
          const parsed = JSON.parse(rawJson);
          
          if (parsed?.name && toolNames.has(parsed.name)) {
            // Check for duplicates
            const argsStr = JSON.stringify(parsed.arguments ?? {});
            if (!calls.some(c => c.function.name === parsed.name && c.function.arguments === argsStr)) {
              calls.push({
                id: `embedded_${Date.now()}_${calls.length}`,
                type: 'function',
                function: { name: parsed.name, arguments: argsStr },
              });
            }
          }
        } catch {
          // Inner JSON might be partial or malformed, skip
        }
      }
    }
  }

  return calls;
}

function stripEmbeddedToolCalls(text: string): string {
  let stripped = text;
  stripped = stripped.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '');
  stripped = stripped.replace(/<\|tool_call\|?>call:[\s\S]*?(?:\n|$)/g, '');
  stripped = stripped.replace(/```(?:json)?\s*{[\s\S]*?}\s*```/g, '');
  return stripped.trim();
}



function inferSafeToolCall(
  thinkingContent: string,
  userMessage: string,
  toolDefs: Array<{ function: { name: string } }>
): LLMToolCall | null {
  const available = new Set(toolDefs.map(td => td.function.name));
  const combined = `${thinkingContent}\n${userMessage}`;

  const makeCall = (name: string, args: Record<string, unknown>): LLMToolCall => ({
    id: `inferred_${Date.now()}_${name}`,
    type: 'function',
    function: {
      name,
      arguments: JSON.stringify(args),
    },
  });

  if (available.has('read_file') && /read_file/i.test(combined)) {
    const pathMatch =
      combined.match(/`([^`\r\n]+?\.[a-z0-9]+)`/i) ||
      combined.match(/"([^"\r\n]+?\.[a-z0-9]+)"/i) ||
      combined.match(/\b([A-Za-z0-9._-]+\.[A-Za-z0-9]+)\b/);
    if (pathMatch?.[1]) {
      return makeCall('read_file', { path: pathMatch[1] });
    }
  }

  if (available.has('list_dir') && /list_dir/i.test(combined)) {
    const pathMatch =
      combined.match(/current directory/i) ? '.' :
      combined.match(/`([^`\r\n]+)`/)?.[1] ??
      combined.match(/"([^"\r\n]+)"/)?.[1];
    return makeCall('list_dir', { path: pathMatch || '.' });
  }

  if (available.has('web_search') && /web_search/i.test(combined)) {
    const queryMatch =
      combined.match(/query like "([^"]+)"/i) ||
      combined.match(/query "([^"]+)"/i) ||
      combined.match(/latest [^"\r\n]+/i);
    if (queryMatch?.[1] || queryMatch?.[0]) {
      return makeCall('web_search', { query: queryMatch[1] ?? queryMatch[0], maxResults: 5 });
    }
  }

  if (available.has('web_fetch') && /web_fetch/i.test(combined)) {
    const urlMatch = combined.match(/https?:\/\/\S+/i);
    if (urlMatch?.[0]) {
      return makeCall('web_fetch', { url: urlMatch[0] });
    }
  }

  return null;
}

function truncateToolOutput(output: string, maxChars: number): string {
  if (output.length <= maxChars) return output;
  const half = Math.floor(maxChars / 2);
  return `${output.slice(0, half)}\n\n... [truncated ${output.length - maxChars} chars] ...\n\n${output.slice(-half)}`;
}

function formatToolResultForModel(toolName: string, result: ToolResult): string {
  const output = truncateToolOutput(result.output, 2000);
  return `<tool_result>
{"tool":"${toolName}","success":${result.success ? 'true' : 'false'}}
</tool_result>

${output}`;
}
