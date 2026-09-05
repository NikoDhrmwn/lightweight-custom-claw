# LiteClaw

Lightweight agent runtime for local LLMs with WebUI, Discord, and WhatsApp support.

Designed for small models like Gemma 4 E4B running on consumer GPUs. LiteClaw aims to be a lighter alternative to OpenClaw while keeping a familiar workflow and migration path.

## Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### [1.0.2] - 2026-09-05

#### Added

- **Centralized version singleton** (`src/version.ts`): Single source of truth reads from `package.json` at runtime with directory-traversal fallback. All source files now import `VERSION` instead of hardcoding it.
- **Version bump script** (`scripts/bump-version.js`): `npm run bump <version>` updates `package.json` and `package-lock.json` atomically.
- **WhatsApp group participant tagging**: Persistent contact cache (`~/.liteclaw/whatsapp-contacts.json`) with first-name alias support; group participants surface in agent context as `[group participants: @Name (phone)]`; outgoing mentions resolved to full JIDs via `buildNameAliases`.

#### Changed

- **Monitoring commands redesigned** (WhatsApp & Discord) — minimal, professional output with no raw session JIDs exposed:
  - `/tokens` — compact progress bar, human-readable session name, clear status label (Healthy / Moderate / Near limit).
  - `/sessions` — shows session name instead of raw `whatsapp:...@g.us` key; current session marked with `←`.
  - `/status` — condensed 5-line format; session name displayed.
  - `/insights` — top sessions show names, not truncated keys; field labels simplified.
  - `/help` — grouped into Conversation / Monitoring / Utilities sections.
  - Discord `/help`, `/status`, `/tokens`, `/insights` embeds similarly simplified.

### [1.0.1] - 2026-09-04

#### Added

- **Agent-Reach Integration**: Complete integration of Agent-Reach capabilities across 15 platforms (YouTube, Bilibili, Reddit, GitHub, XiaoHongShu, LinkedIn, V2EX, Xueqiu, Twitter/X, and web):
  - 5 core tools: `reach_doctor`, `web_extract` (Jina Reader), `reach_read` (YouTube transcripts, V2EX, Reddit, GitHub), `reach_search` (YouTube via flat-playlist yt-dlp, GitHub, dev communities), and `reach_transcribe` (Whisper STT).
  - Built-in `agent-reach` skill with platform routing and 7 reference playbooks (`search.md`, `social.md`, `career.md`, `dev.md`, `web.md`, `video.md`, `finance.md`).
  - CLI diagnostic command: `liteclaw reach doctor [--detailed]`.
  - Configurable API keys in `tools.reach` (`jinaApiKey`, `groqApiKey`, `exaApiKey`).

#### Fixed

- **Plan-mode web search** — tools selected at request-time (e.g. `web_search`) were silently dropped during per-task tool resolution because vague planner-generated task titles didn't re-trigger keyword scoring. `resolveTaskTools()` now merges the request-level selection with per-task scoring and force-includes any tool explicitly listed in `task.suggestedTools`.
- **Plan-mode final answer delivery** — `generateFinalResponse()` now synthesizes all findings, research data, and task notes into a comprehensive, detailed answer rather than outputting a generic confirmation ("The task is complete").
- **Plan-mode response separation on Discord** — The progress monitoring embed remains a dedicated status card, and the agent's final substantive answer is sent as brand new message(s) directly to the channel or via followUp.
- **Interactive choice button inactivity** — Clicking a choice button on Discord now disables the buttons to prevent duplicate clicks, announces the selection, and invokes an agent turn with the selected choice to continue execution seamlessly.
- **Duplicate choice text suppression** — Eliminated redundant text echoes where the agent repeated the choices after interactive buttons were already rendered.
- **Cross-channel table & markdown formatting**:
  - **Discord**: Automatically converts markdown tables into clean monospace code blocks with aligned box-drawing characters for compact tables, or structured cards for wide tables.
  - **WhatsApp**: Converts markdown tables into mobile-friendly bullet cards with bold headers and clear key-value lines.
  - **WebUI**: Fixed table parsing to isolate tables before paragraph splitting, restored empty cell handling, enabled inline formatting (bold, italic, code, links) within cells, and added modern styled CSS for `.table-wrap`.

### [1.0.0] - 2026-09-04

#### Added

- **Persistent Long-Term Memory**: `manage_memory` tool (remember / recall / view / update) backed by `MEMORY.md` injected into every system prompt. `/memory` slash command on WhatsApp and Discord.
- **Cross-Session FTS5 Search**: Full-text search across all messages via `search_history` tool and `/search <query>` slash command.
- **Session Control**: `/retry`, `/undo`, `/stop` slash commands for replaying, undoing, or aborting the current turn, backed by `AbortController` threading through the engine.
- **Subagent Delegation**: `delegate_task` tool spawns isolated child agent sessions for parallel or long-running sub-tasks.
- **Code Execution Sandbox**: `run_code` tool supporting Python, JavaScript, TypeScript, and PowerShell.
- **Self-Improving Skills Loop**: `manage_skills` tool (list / view / create / update) persists reusable skill templates with SQLite-tracked usage analytics.
- **Usage Analytics**: `getUsageStats()` aggregation and `/insights [days]` slash command.
- **Persistent Kanban Board**: `manage_kanban` tool with full CRUD and `/tasks` / `/kanban` slash commands.
- System prompt updated with guidance sections for all new v1.0 tools.

### [0.8.4] - 2026-05-08

#### Added

- **Extensions Tab**: New "Extensions" section in WebUI settings panel for managing opt-in features.
- **DnD as Extension**: The D&D system is now a fully configurable extension with:
  - Dedicated **Narrative Model** and **Loadout Model** selectors (falls back to primary model when unset).
  - **Default World** and **Tone** presets (Heroic, Dark, Comedic, Mystery, Horror, Sandbox).
  - **Max Players**, **Narrative Temperature**, and **Max Tokens** tuning.
  - **Auto-Provision** toggle for combat loadout generation on session start.

#### Changed

- **Config Schema**: Added `extensions.dnd` section to `LiteClawConfig` for persistent DnD settings.
- **Backward Compatibility**: `llm.defaults.loadoutModel` is still respected as a fallback when `extensions.dnd.loadoutModel` is unset.
- **Gateway API**: `GET /api/config` and `PATCH /api/config` now expose and accept `extensions.dnd` fields.

### [0.8.3] - 2026-05-06

#### Added

- **WebUI Onboarding**: Added a first-time initialization modal to allow users to name their agent.
- **Media Unfurling**: Native support for scraping and downloading media from URLs to ensure high-quality GIF and image previews in Discord.

#### Changed

- **Centralized Progress System**: Refactored Discord and WhatsApp channels to use a unified progress and status reporting system.
- **Percentage-Based Compaction**: Replaced absolute token thresholds with percentage-based `softThresholdPct` for more resilient context management.
- **D&D Session Polish**: Refined Discord embeds for a more compact and consistent visual style across all DnD commands.

## Features

- **Autonomous Task Planner**: Breaks down complex requests into multi-step executable plans.
- **Smart Context Management**: Lazy tool loading, compaction, and rolling history to stay within model context limits.
- **Multi-Channel**: Native support for WebUI, Discord, and WhatsApp.
- **Discord DnD Sessions**: New slash-command workflow for multiplayer session threads, persistent player rosters, join/resume flows, partial-party resumes, and vote-based turn skipping.
- **Core Tools**: Filesystem access, command execution, web search/fetch, and native vision.
- **MCP Integrations**: Connect external MCP servers and expose their tools, prompts, and resources to the agent.
- **Safety First**: Cross-channel confirmations for destructive or sensitive actions.
- **Skills System**: Project-level skills support with selective prompt injection.

## Discord DnD Commands

LiteClaw now includes a lightweight DnD session subsystem for Discord:

- `/dnd start` creates a dedicated session thread and opens the lobby.
- `/dnd join` joins the current or specified session with a character profile, including joining midway through an active session.
- `/dnd begin` starts play and establishes turn order.
- `/dnd save` pauses the session and stores a checkpoint in SQLite.
- `/dnd resume` restores a saved session into the current thread, including partial-party mode.
- `/dnd restore` restores a specific checkpoint by checkpoint ID.
- `/dnd list` shows resumable sessions in the current guild.
- `/dnd checkpoints` lists saved checkpoints for a session.
- `/dnd available` and `/dnd unavailable` toggle whether your turns should be skipped.
- `/stats` shows your persistent character sheet, level, and XP progress.
- `/quest complete` and `/quest log` track quest completions and XP rewards.
- `/combat enter`, `/combat status`, `/combat menu`, and `/combat end` manage initiative and active-turn combat controls.
- `/vote skip-turn` opens a party vote to skip a player who is unavailable.
- `/end-turn` advances to the next available player.
- `/question` asks the GM an out-of-band question tied to the current DnD session without consuming a turn or polluting the main session context.
- `/question mode:private|public` controls whether the answer stays private or is visible to the table.

### DnD RAG Notes

- LiteClaw stores DnD retrieval data in its state directory and uses a local embedding server for session-aware GM answers.
- Configure any local embedding bootstrap command in your LiteClaw state/config files instead of hardcoding machine-specific paths.
- Starting or refreshing a DnD session can ensure the configured embedding server is running before syncing session context.

## Quick Start

```bash
git clone https://github.com/NikoDhrmwn/liteclaw.git
cd liteclaw
npm install
npx tsx src/cli.ts setup
npx tsx src/cli.ts gateway run
```

Open `http://localhost:7860` for the Web UI.

For guided first-time setup with recommendations for local 4B-9B models:

```bash
npx tsx src/cli.ts setup --interactive
```

## Windows Setup

LiteClaw works well from Windows Terminal, PowerShell, or Command Prompt.

### PowerShell

```powershell
git clone https://github.com/NikoDhrmwn/liteclaw.git
Set-Location liteclaw
npm install
npx tsx src/cli.ts setup
npx tsx src/cli.ts gateway run
```

### Command Prompt

```bat
git clone https://github.com/NikoDhrmwn/liteclaw.git
cd liteclaw
npm install
npx tsx src/cli.ts setup
npx tsx src/cli.ts gateway run
```

### Batch launcher

If you prefer a double-clickable launcher on Windows:

```bat
start-liteclaw.bat
```

That script installs dependencies if needed and starts LiteClaw from the project folder.

## First-Time Configuration

Initialize the local state directory:

```bash
npx tsx src/cli.ts setup
```

Or run the guided onboarding wizard:

```bash
npx tsx src/cli.ts init
```

This creates your LiteClaw state under:

- Windows: `%USERPROFILE%\.liteclaw`

Then put your secrets in:

```text
%USERPROFILE%\.liteclaw\.env
```

Example values:

```env
DISCORD_TOKEN=
GOOGLE_API_KEY=
GATEWAY_TOKEN=
GITHUB_PERSONAL_ACCESS_TOKEN=
LLM_BASE_URL=http://localhost:8080/v1
LLM_API_KEY=sk-local
LLM_MODEL=gemma-4-e4b-heretic
```

You can use the project-root [.env.example](.env.example) as a reference.

## Prompt and Personality Customization

LiteClaw ships neutral, universal prompt templates by default. User-editable prompt files live in:

```text
%USERPROFILE%\.liteclaw\personality\
```

Recommended prompt commands:

```bash
npx tsx src/cli.ts prompts list
npx tsx src/cli.ts prompts doctor
npx tsx src/cli.ts prompts edit system
npx tsx src/cli.ts prompts edit behavior
npx tsx src/cli.ts prompts reset --profile neutral
```

Use `prompts doctor` after edits. It flags oversized prompts, personal machine paths, unsafe instructions, and reliability issues that commonly hurt smaller local models.

## Running LiteClaw

Start the gateway and WebUI:

```bash
npx tsx src/cli.ts gateway run
```

Or, after building:

```bash
npm run build
node dist/cli.js gateway run
```

Useful terminal commands:

```bash
npx tsx src/cli.ts doctor
npx tsx src/cli.ts mcp list
npx tsx src/cli.ts mcp add github
npx tsx src/cli.ts status
npx tsx src/cli.ts channels status
npx tsx src/cli.ts message "hello"
```

## MCP Setup

LiteClaw 0.8.4 adds native MCP client support and an extensible plugin system. MCP tools are discovered at startup and injected into the agent like built-in tools, while MCP prompts and resources are available through `mcp_*` utility tools.

### Quick GitHub setup

```bash
liteclaw mcp add github
liteclaw mcp login github
liteclaw mcp doctor
```

The GitHub preset uses the official remote GitHub MCP endpoint:

```text
https://api.githubcopilot.com/mcp/
```

Credentials are stored in your LiteClaw state `.env` as:

```env
GITHUB_PERSONAL_ACCESS_TOKEN=...
```

After setup, GitHub MCP tools will appear to the agent with a `github_` prefix, making tasks like pull requests, issue work, and repository review available through the normal tool-calling flow.

## Migrating From OpenClaw

If you already use OpenClaw, LiteClaw can import configuration and local state.

### Default migration

```bash
npx tsx src/cli.ts migrate
```

This attempts to import from the default OpenClaw directory:

- Windows: `%USERPROFILE%\.openclaw`

### Custom migration path

```powershell
npx tsx src/cli.ts migrate --openclaw-dir "/path/to/.openclaw"
```

Migration can bring over:

- model configuration
- Discord and WhatsApp channel config
- WhatsApp session files when present
- memory database
- personality files from the OpenClaw workspace

After migrating, review:

- `%USERPROFILE%\.liteclaw\config.yaml`
- `%USERPROFILE%\.liteclaw\.env`
- `%USERPROFILE%\.liteclaw\personality\`

## Common Commands

```bash
liteclaw gateway run
liteclaw init
liteclaw channels login --channel discord
liteclaw channels login --channel whatsapp
liteclaw channels status
liteclaw status
liteclaw doctor
liteclaw mcp list
liteclaw mcp add github
liteclaw mcp login github
liteclaw prompts list
liteclaw prompts doctor
liteclaw prompts edit system
liteclaw prompts edit behavior
liteclaw config get <key>
liteclaw models list
liteclaw message "hello"
liteclaw migrate
```

If you have not installed the global CLI yet, use the same commands through `npx tsx src/cli.ts ...`.

Examples:

```bash
npx tsx src/cli.ts channels login --channel discord
npx tsx src/cli.ts channels login --channel whatsapp
npx tsx src/cli.ts mcp doctor
npx tsx src/cli.ts config get gateway.port
npx tsx src/cli.ts models list
```

## Requirements

- Node.js >= 20
- A running LLM backend such as llama-server or Ollama, or a compatible hosted provider
- Discord bot token for Discord usage
- A linked phone for WhatsApp usage

## Security Notes

- Keep secrets in your LiteClaw state directory `.env`, not in the repository root.
- Do not commit runtime logs, SQLite databases, or WhatsApp session files.
- Review [SECURITY.md](SECURITY.md) before publishing a fork or deployment.
