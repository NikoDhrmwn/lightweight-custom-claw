/**
 * LiteClaw — Subagent Delegation Tool
 *
 * Allows the primary agent to spawn an isolated subagent for a parallel or exploratory subtask
 * (inspired by Hermes Agent's delegate_tool).
 */

import { toolRegistry, ToolResult } from '../core/tools.js';
import { getDefaultAgentEngine } from '../core/engine.js';

toolRegistry.register({
  name: 'delegate_task',
  description: 'Spawn an isolated subagent to execute a focused subtask, multi-step research, or exploratory investigation. The subagent runs in its own context, executes necessary tools, and returns a concise synthesized report back to you.',
  category: 'utility',
  parameters: [
    {
      name: 'task',
      type: 'string',
      description: 'The specific objective or question for the subagent to solve',
      required: true,
    },
    {
      name: 'context',
      type: 'string',
      description: 'Background facts, constraints, or previous findings relevant to the subtask',
      required: false,
    },
  ],
  usageNotes: [
    'Use this for deep multi-step research, analyzing multiple files, or testing alternatives without cluttering your main conversation context.',
    'The subagent has access to all tools (filesystem, web search, code execution) and runs autonomously.',
    'Do not delegate trivial single-step tasks that you can do directly with one tool call.',
  ],
  examples: [
    {
      userIntent: 'research different python libraries for audio extraction',
      arguments: {
        task: 'Evaluate pydub, ffmpeg-python, and soundfile for webm audio slicing. Compare ease of installation and performance.',
        context: 'We are running on Windows with ffmpeg installed on PATH.',
      },
    },
    {
      userIntent: 'investigate why tests in another repo are failing',
      arguments: {
        task: 'Inspect the test files in test/unit, check the latest failure log, and identify the root cause.',
      },
    },
  ],
  keywords: ['subagent', 'delegate', 'spawn', 'worker', 'parallel', 'research', 'investigate'],
  handler: async (args, context): Promise<ToolResult> => {
    const task = String(args.task ?? '').trim();
    if (!task) {
      return { success: false, output: 'Missing "task" description for delegation.' };
    }

    const engine = getDefaultAgentEngine();
    if (!engine) {
      return { success: false, output: 'AgentEngine is not initialized for delegation.' };
    }

    try {
      const report = await engine.executeSubagent(task, {
        parentSessionKey: context.sessionKey,
        context: args.context ? String(args.context).trim() : undefined,
        workingDir: context.workingDir,
      });

      return {
        success: true,
        output: `[SUBAGENT REPORT]\n\n${report}`,
      };
    } catch (err: any) {
      return {
        success: false,
        output: `Delegation error: ${err.message}`,
      };
    }
  },
});
