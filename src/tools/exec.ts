/**
 * LiteClaw — Command & Script Execution Tool
 * 
 * Executes shell commands, system binaries, or multiline scripts
 * (Python, PowerShell, Node.js, Bash) with safeBins allowlist checking,
 * safe script file spooling (preventing quote mangling and syntax errors),
 * and confirmation for destructive operations.
 */

import { spawnSync } from 'child_process';
import { writeFileSync, unlinkSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { toolRegistry, ToolContext, ToolResult } from '../core/tools.js';
import { getConfig, getStateDir } from '../config.js';
import { resolveFlexiblePath } from '../core/workspace.js';

// ─── Destructive command patterns ────────────────────────────────────

const DESTRUCTIVE_PATTERNS = [
  /\brm\s/i, /\bdel\s/i, /\brmdir\s/i, /\bformat\s/i,
  /\bdelete\s/i, /\bdrop\s/i, /\btruncate\s/i,
  /\bpurge\s/i, /\bwipe\s/i,
  /\bmkfs\b/i, /\bfdisk\b/i,
];

function isDestructive(command: string): boolean {
  return DESTRUCTIVE_PATTERNS.some(p => p.test(command));
}

// ─── SafeBins check ──────────────────────────────────────────────────

function isAllowed(executable: string): boolean {
  const config = getConfig();
  const safeBins = config.tools?.exec?.safeBins ?? [];

  if (safeBins.length === 0) return true; // No restriction

  const bin = executable.trim().split(/\s+/)[0].toLowerCase();

  return safeBins.some((safe: string) => {
    const safeLower = safe.toLowerCase();
    return bin === safeLower ||
      bin.endsWith(`\\${safeLower}`) ||
      bin.endsWith(`/${safeLower}`) ||
      bin === `${safeLower}.exe`;
  });
}

// ─── Python binary detection on Windows ──────────────────────────────

let cachedPythonBin: string | null = null;
function detectPython(): string {
  if (cachedPythonBin) return cachedPythonBin;

  const candidates = ['python', 'py -3', 'py', 'python3'];
  for (const cand of candidates) {
    try {
      const parts = cand.split(/\s+/);
      const res = spawnSync(parts[0], parts.slice(1).concat(['--version']), {
        encoding: 'utf-8',
        windowsHide: true,
      });
      if (res.status === 0) {
        cachedPythonBin = cand;
        return cand;
      }
    } catch {
      // Continue searching
    }
  }

  cachedPythonBin = 'python';
  return 'python';
}

// ─── exec tool ───────────────────────────────────────────────────────

toolRegistry.register({
  name: 'exec',
  description: 'Execute a shell command, terminal binary, or multiline script (Python, PowerShell, Node.js). Supports zero-escaping script execution to avoid command line syntax and quote errors.',
  category: 'exec',
  parameters: [
    { name: 'command', type: 'string', description: 'Shell command to execute (e.g. "git status", "pip list", "npm test").' },
    { name: 'script', type: 'string', description: 'Multiline script body to execute directly (Python, PowerShell, or Node.js). Spooled to a clean temp script file to avoid escaping and syntax errors.' },
    { name: 'interpreter', type: 'string', description: 'Interpreter for script: "python", "powershell", "node", "cmd", "bash", or "auto" (default: auto).', enum: ['python', 'powershell', 'node', 'cmd', 'bash', 'auto'] },
    { name: 'bin', type: 'string', description: 'Executable binary to run (e.g. "python", "git", "npm").' },
    { name: 'args', type: 'array', description: 'Array of arguments for binary', items: { type: 'string' } },
    { name: 'cwd', type: 'string', description: 'Working directory (supports "downloads", "documents", "~", "C:\\...", or relative path).' },
    { name: 'timeout', type: 'number', description: 'Timeout in seconds (default: 60).' },
  ],
  usageNotes: [
    'When running Python code (e.g. using openpyxl, pandas, requests), supply the code in `script` with `interpreter: "python"` to avoid all quoting and path escaping issues.',
    'When running PowerShell commands, supply them in `command` or `script` with `interpreter: "powershell"`.',
    'Working directory supports aliases like "downloads" and "documents".'
  ],
  examples: [
    { userIntent: 'run python script to read excel', arguments: { script: 'import openpyxl\nwb = openpyxl.load_workbook("data.xlsx")\nprint(wb.sheetnames)', interpreter: 'python' } },
    { userIntent: 'check git status', arguments: { command: 'git status' } },
    { userIntent: 'run npm test', arguments: { bin: 'npm', args: ['run', 'test'] } },
  ],
  keywords: ['run', 'execute', 'command', 'shell', 'terminal', 'cmd', 'powershell', 'script', 'python', 'node', 'pip', 'npm', 'git'],
  handler: async (args, context): Promise<ToolResult> => {
    const config = getConfig();

    let bin = args.bin as string | undefined;
    let cmdArgs = (args.args as string[] | undefined) ? [...args.args] : undefined;
    let command = args.command as string | undefined;
    const script = args.script as string | undefined;
    const interpreter = (args.interpreter as string | undefined)?.toLowerCase() || 'auto';

    if (!bin && !command && !script) {
      return { success: false, output: 'No command, script, or binary specified. Provide `command`, `script`, or `bin` and `args`.' };
    }

    // Resolve cwd safely through flexible path resolver
    let cwd: string;
    try {
      cwd = args.cwd
        ? resolveFlexiblePath(args.cwd, context.workingDir).absolute
        : context.workingDir;
    } catch {
      cwd = context.workingDir;
    }

    if (!existsSync(cwd)) {
      cwd = context.workingDir;
    }

    const tempDir = join(getStateDir(), 'temp');
    if (!existsSync(tempDir)) mkdirSync(tempDir, { recursive: true });

    let tempScriptFile: string | null = null;
    let spawnBin = bin;
    let spawnArgs = cmdArgs || [];

    try {
      // ─── Scenario A: Dedicated Script Execution ───
      if (script) {
        let chosenLang = interpreter;
        if (chosenLang === 'auto') {
          if (script.includes('import ') || script.includes('def ') || script.includes('print(')) {
            chosenLang = 'python';
          } else if (script.includes('require(') || script.includes('console.log') || script.includes('import *')) {
            chosenLang = 'node';
          } else if (script.includes('$') || script.includes('Get-') || script.includes('Write-Output')) {
            chosenLang = 'powershell';
          } else {
            chosenLang = process.platform === 'win32' ? 'powershell' : 'bash';
          }
        }

        const timestamp = `${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;

        if (chosenLang === 'python') {
          tempScriptFile = join(tempDir, `script_${timestamp}.py`);
          writeFileSync(tempScriptFile, script, 'utf-8');
          const pyBin = detectPython();
          const parts = pyBin.split(/\s+/);
          spawnBin = parts[0];
          spawnArgs = [...parts.slice(1), tempScriptFile];
        } else if (chosenLang === 'powershell') {
          tempScriptFile = join(tempDir, `script_${timestamp}.ps1`);
          writeFileSync(tempScriptFile, script, 'utf-8');
          spawnBin = 'powershell.exe';
          spawnArgs = ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', tempScriptFile];
        } else if (chosenLang === 'node') {
          tempScriptFile = join(tempDir, `script_${timestamp}.mjs`);
          writeFileSync(tempScriptFile, script, 'utf-8');
          spawnBin = 'node';
          spawnArgs = [tempScriptFile];
        } else if (chosenLang === 'cmd') {
          tempScriptFile = join(tempDir, `script_${timestamp}.bat`);
          writeFileSync(tempScriptFile, script, 'utf-8');
          spawnBin = 'cmd.exe';
          spawnArgs = ['/c', tempScriptFile];
        } else {
          tempScriptFile = join(tempDir, `script_${timestamp}.sh`);
          writeFileSync(tempScriptFile, script, 'utf-8');
          spawnBin = 'bash';
          spawnArgs = [tempScriptFile];
        }
      }

      // ─── Scenario B: Python -c Inline Script Auto-Diverter ───
      else if (bin && (bin.toLowerCase() === 'python' || bin.toLowerCase() === 'py') && cmdArgs && cmdArgs[0] === '-c' && cmdArgs[1]) {
        const code = cmdArgs[1];
        if (code.includes('\n') || code.length > 50 || code.includes('"') || code.includes('\\')) {
          const timestamp = `${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
          tempScriptFile = join(tempDir, `script_${timestamp}.py`);
          writeFileSync(tempScriptFile, code, 'utf-8');
          const pyBin = detectPython();
          const parts = pyBin.split(/\s+/);
          spawnBin = parts[0];
          spawnArgs = [...parts.slice(1), tempScriptFile, ...cmdArgs.slice(2)];
        }
      }

      // ─── Scenario C: Shell Command ───
      else if (command && !bin) {
        if (process.platform === 'win32') {
          spawnBin = 'powershell.exe';
          spawnArgs = ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', command];
        } else {
          spawnBin = '/bin/sh';
          spawnArgs = ['-c', command];
        }
      }

      // Safety checks
      const fullCommandString = command || `${spawnBin} ${spawnArgs.join(' ')}`;
      if (!isAllowed(spawnBin || '')) {
        return { success: false, output: `Executable not in safeBins allowlist: ${spawnBin}` };
      }

      if (config.tools?.exec?.confirmDestructive && isDestructive(fullCommandString)) {
        if (context.requestConfirmation) {
          const confirmed = await context.requestConfirmation(
            `⚠️ Destructive command detected:\n\`${fullCommandString}\`\n\nThis may delete data permanently.`
          );
          if (!confirmed) {
            return { success: false, output: 'Owner rejected the destructive command.' };
          }
        }
      }

      const timeoutMs = (args.timeout ?? 60) * 1000;

      const spawnResult = spawnSync(spawnBin!, spawnArgs, {
        cwd,
        timeout: timeoutMs,
        encoding: 'utf-8',
        maxBuffer: 2 * 1024 * 1024, // 2MB
        windowsHide: true,
        shell: false, // Prevents cmd.exe quote stripping
      });

      if (spawnResult.error) {
        throw spawnResult.error;
      }

      const stdout = (spawnResult.stdout || '').toString();
      const stderr = (spawnResult.stderr || '').toString();
      const combined = (stdout + (stderr ? `\n[stderr]\n${stderr}` : '')).trim();

      if (spawnResult.status !== 0 && spawnResult.status !== null) {
        return {
          success: false,
          output: `Command exited with code ${spawnResult.status}:\n${combined.slice(0, 4000) || '(no error output)'}`,
        };
      }

      const trimmed = combined.length > 5000
        ? combined.slice(0, 2500) + `\n\n... [truncated ${combined.length - 5000} chars] ...\n\n` + combined.slice(-2500)
        : combined;

      return {
        success: true,
        output: trimmed || '(command completed with no output)',
      };
    } catch (err: any) {
      return {
        success: false,
        output: `Execution error: ${err.message}`,
      };
    } finally {
      if (tempScriptFile && existsSync(tempScriptFile)) {
        try {
          unlinkSync(tempScriptFile);
        } catch {
          // ignore cleanup error
        }
      }
    }
  },
});

// ─── run_code tool ───────────────────────────────────────────────────

toolRegistry.register({
  name: 'run_code',
  description: 'Execute code directly in Python, JavaScript, TypeScript, or PowerShell. Safely spools to a temp file with zero-escaping issues, captures stdout/stderr, and returns execution status.',
  category: 'exec',
  parameters: [
    {
      name: 'language',
      type: 'string',
      description: 'Language runtime: "python", "javascript", "typescript", or "powershell"',
      required: true,
      enum: ['python', 'javascript', 'typescript', 'powershell'],
    },
    {
      name: 'code',
      type: 'string',
      description: 'The code to execute',
      required: true,
    },
    {
      name: 'timeout',
      type: 'number',
      description: 'Execution timeout in seconds (default: 30)',
      required: false,
    },
    {
      name: 'cwd',
      type: 'string',
      description: 'Working directory for the execution',
      required: false,
    },
  ],
  usageNotes: [
    'Use this instead of exec when you want to compute values, test scripts, parse data, or verify logic in Python or JS.',
    'No shell escaping needed — code is executed directly from an isolated script file.',
  ],
  examples: [
    { userIntent: 'compute large fibonacci in python', arguments: { language: 'python', code: 'def fib(n):\n  a, b = 0, 1\n  for _ in range(n): a, b = b, a + b\n  return a\nprint(fib(100))' } },
    { userIntent: 'test json parsing in node', arguments: { language: 'javascript', code: 'console.log(JSON.stringify({ test: 123 }, null, 2))' } },
  ],
  keywords: ['run', 'code', 'python', 'javascript', 'typescript', 'script', 'eval', 'calculate', 'compute'],
  handler: async (args, context): Promise<ToolResult> => {
    const code = String(args.code ?? '').trim();
    if (!code) {
      return { success: false, output: 'Missing "code" parameter.' };
    }

    const language = String(args.language ?? 'python').toLowerCase();
    const timeout = Math.max(5, Math.min(180, Number(args.timeout) || 30));
    const cwd = args.cwd ? resolveFlexiblePath(args.cwd, context.workingDir).absolute : context.workingDir;

    let extension = 'py';
    let interpreter = 'python';

    if (language === 'python') {
      extension = 'py';
      interpreter = 'python';
    } else if (language === 'javascript') {
      extension = 'mjs';
      interpreter = 'node';
    } else if (language === 'typescript') {
      extension = 'ts';
      interpreter = 'npx tsx';
    } else if (language === 'powershell') {
      extension = 'ps1';
      interpreter = 'powershell';
    } else {
      return { success: false, output: `Unsupported language: "${language}". Use python, javascript, typescript, or powershell.` };
    }

    // Reuse the exec tool handler with script & scriptType
    const execTool = toolRegistry.get('exec');
    if (!execTool) {
      return { success: false, output: 'Underlying exec tool not found.' };
    }

    return execTool.handler({
      script: code,
      scriptType: extension,
      timeout,
      cwd,
    }, context);
  },
});

