(() => {
  let ws = null;
  let reconnectTimer = null;
  let isStreaming = false;
  let currentAssistantEl = null;
  let currentContent = "";
  let lastThinkingChunk = "";
  let currentSessionKey = "webui:default";
  let currentFilter = "";
  let currentConfig = {};
  let healthData = {};
  let sessions = [];
  let currentSessionMetrics = {
    estimatedTokens: 0,
    messageCount: 0,
    imageCount: 0,
    contextMaxTokens: 0,
    contextBudgetPct: 80,
    contextBudgetTokens: 0,
    compactionThresholdPct: 90,
    compactionThresholdTokens: 0,
    compactionProgressPct: 0,
  };
  let pendingConfirmationId = null;
  let pendingRegeneration = null;
  let isRegenerating = false;
  let workspacePath = ".";
  let selectedWorkspaceFile = "";
  const attachments = [];

  // Voice Lounge Variables
  let voiceAudioContext = null;
  let voiceMicStream = null;
  let voiceMicSource = null;
  let voiceScriptProcessor = null;
  let voiceMicAnalyser = null;
  let voiceSpeakerAnalyser = null;
  let isVoiceConnected = false;
  let isVoiceMuted = false;
  let voiceTriggerMode = 'vad';
  let voiceState = 'idle'; // 'idle', 'listening', 'thinking', 'speaking'
  let voicePlaybackQueue = [];
  let voicePlayingAudio = false;
  let voiceActiveAudioSource = null;
  let voiceAnimationFrameId = null;
  let spacebarPressed = false;

  const $ = (selector) => document.querySelector(selector);
  const messagesEl = $("#messages");
  const sessionListEl = $("#sessionList");
  const sessionSearch = $("#sessionSearch");
  const sendBtn = $("#sendBtn");
  const inputEl = $("#messageInput");
  const imageInput = $("#imageInput");
  const imagePreview = $("#imagePreview");
  const workspaceBtn = $("#workspaceBtn");
  const workspaceDrawer = $("#workspaceDrawer");
  const workspaceTree = $("#workspaceTree");
  const workspacePreviewMeta = $("#workspacePreviewMeta");
  const workspacePreviewContent = $("#workspacePreviewContent");
  const workspacePathLabel = $("#workspacePathLabel");
  const closeWorkspaceBtn = $("#closeWorkspaceBtn");
  const workspaceUpBtn = $("#workspaceUpBtn");
  const workspaceRefreshBtn = $("#workspaceRefreshBtn");
  const settingsBtn = $("#settingsBtn");
  const settingsOverlay = $("#settingsOverlay");
  const closeSettingsBtn = $("#closeSettingsBtn");
  const saveSettingsBtn = $("#saveSettingsBtn");
  const clearAllBtn = $("#clearAllBtn");
  const confirmModal = $("#confirmModal");
  const confirmBody = $("#confirmBody");
  const confirmAccept = $("#confirmAccept");
  const confirmReject = $("#confirmReject");
  const noticeStack = $("#noticeStack");
  const sidebar = $("#sidebar");
  const sidebarToggle = $("#sidebarToggle");
  const sidebarBackdrop = $("#sidebarBackdrop");
  const mobileMenuBtn = $("#mobileMenuBtn");

  const refs = {
    statusDot: $("#statusDot"),
    statusText: $("#statusText"),
    modelBadge: $("#modelBadge"),
    sessionCountBadge: $("#sessionCountBadge"),
    chatTitle: $("#chatTitle"),
    chatSubtitle: $("#chatSubtitle"),
    healthPill: $("#healthPill"),
    healthDot: $("#healthDot"),
    healthLabel: $("#healthLabel"),
    tokenLabel: $("#tokenLabel"),
    chWebui: $("#ch-webui"),
    chDiscord: $("#ch-discord"),
    chWhatsapp: $("#ch-whatsapp"),
    newChatBtn: $("#newChatBtn"),
    clearBtn: $("#clearBtn"),
    exportBtn: $("#exportBtn"),
    settingModel: $("#settingModel"),
    settingThinking: $("#settingThinking"),
    settingTemperature: $("#settingTemperature"),
    settingTopP: $("#settingTopP"),
    settingTopK: $("#settingTopK"),
    settingMaxOutputTokens: $("#settingMaxOutputTokens"),
    llmParamsGroup: $("#llmParamsGroup"),
    settingWorkspace: $("#settingWorkspace"),
    settingMaxTurns: $("#settingMaxTurns"),
    settingPlannerMode: $("#settingPlannerMode"),
    settingPlannerMaxReplans: $("#settingPlannerMaxReplans"),
    settingContextTokens: $("#settingContextTokens"),
    settingContextBudgetPct: $("#settingContextBudgetPct"),
    settingCompactionThresholdPct: $("#settingCompactionThresholdPct"),
    settingSkillsMaxInjected: $("#settingSkillsMaxInjected"),
    toggleSkillsEnabled: $("#toggleSkillsEnabled"),
    toggleDiscord: $("#toggleDiscord"),
    toggleWhatsapp: $("#toggleWhatsapp"),
    discordReplyStyle: $("#discordReplyStyle"),
    whatsappReplyStyle: $("#whatsappReplyStyle"),
    toggleDiscordToolProgress: $("#toggleDiscordToolProgress"),
    toggleWhatsappToolProgress: $("#toggleWhatsappToolProgress"),
    toggleExecEnabled: $("#toggleExecEnabled"),
    toggleExecConfirm: $("#toggleExecConfirm"),
    toggleWebFetchEnabled: $("#toggleWebFetchEnabled"),
    toggleWebFallback: $("#toggleWebFallback"),
    toggleFilesystemEnabled: $("#toggleFilesystemEnabled"),
    toggleConfirmDelete: $("#toggleConfirmDelete"),
    toggleVisionEnabled: $("#toggleVisionEnabled"),
    settingVisionMaxDimension: $("#settingVisionMaxDimension"),
    settingGatewayPort: $("#settingGatewayPort"),
    settingGatewayBind: $("#settingGatewayBind"),
    gatewayAuthNote: $("#gatewayAuthNote"),
    settingAgentName: $("#settingAgentName"),
    toggleGoogleVertex: $("#toggleGoogleVertex"),
    toggleGoogleExpress: $("#toggleGoogleExpress"),
    settingGoogleProject: $("#settingGoogleProject"),
    settingGoogleRegion: $("#settingGoogleRegion"),
    onboardingModal: $("#onboardingModal"),
    onboardingName: $("#onboardingName"),
    onboardingSubmit: $("#onboardingSubmit"),
    // Extensions: D&D
    toggleDndEnabled: $("#toggleDndEnabled"),
    dndExtensionBody: $("#dndExtensionBody"),
    settingDndNarrativeModel: $("#settingDndNarrativeModel"),
    settingDndLoadoutModel: $("#settingDndLoadoutModel"),
    settingDndWorld: $("#settingDndWorld"),
    settingDndTone: $("#settingDndTone"),
    settingDndMaxPlayers: $("#settingDndMaxPlayers"),
    settingDndNarrativeTemp: $("#settingDndNarrativeTemp"),
    settingDndNarrativeMaxTokens: $("#settingDndNarrativeMaxTokens"),
    toggleDndAutoProvision: $("#toggleDndAutoProvision"),
    toggleVoiceEnabled: $("#toggleVoiceEnabled"),
    settingVoiceTriggerMode: $("#settingVoiceTriggerMode"),
    settingVoiceWakeName: $("#settingVoiceWakeName"),
    settingVoiceMaxResponseTokens: $("#settingVoiceMaxResponseTokens"),
  };

  // ─── Sidebar Toggle ──────────────────────────────────────────────

  function setSidebarOpen(open) {
    sidebar.classList.toggle("open", open);
    if (sidebarBackdrop) sidebarBackdrop.classList.toggle("visible", open);
  }

  // ─── Fetch Helpers ────────────────────────────────────────────────

  async function fetchJson(url, options) {
    const res = await fetch(url, options);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  // ─── WebSocket ────────────────────────────────────────────────────

  function connect() {
    clearTimeout(reconnectTimer);
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${location.host}/ws`);

    ws.onopen = () => {
      refs.statusDot.classList.add("connected");
      refs.statusText.textContent = "Connected";
      fetchHealth();
      fetchConfig();
      loadSessions();
      sendSessionInit();
      showNotice("Realtime link restored.", "success", 2200);
    };

    ws.onclose = () => {
      refs.statusDot.classList.remove("connected");
      refs.statusText.textContent = "Reconnecting...";
      showNotice("Lost connection. Reconnecting...", "error", 2800);
      reconnectTimer = setTimeout(connect, 2500);
    };

    ws.onerror = () => {
      refs.statusText.textContent = "Connection error";
    };

    ws.onmessage = (event) => {
      try {
        handleServerMessage(JSON.parse(event.data));
      } catch (error) {
        console.error("Failed to parse ws payload", error);
      }
    };
  }

  // ─── Health & Config ──────────────────────────────────────────────

  async function fetchHealth() {
    try {
      healthData = await fetchJson("/health");
      renderHealth();
      checkOnboarding();
    } catch (error) {
      showNotice(`Failed to refresh health: ${error.message || error}`, "error", 2600);
    }
  }

  async function fetchConfig() {
    try {
      currentConfig = await fetchJson("/api/config");
      hydrateSettings(currentConfig);
    } catch (error) {
      showNotice(`Failed to load config: ${error.message || error}`, "error", 3200);
    }
  }

  // ─── Sessions ─────────────────────────────────────────────────────

  async function loadSessions() {
    try {
      const data = await fetchJson("/api/sessions");
      sessions = data.sessions || [];
      renderSessionList();
      fetchSessionMetrics(currentSessionKey);
    } catch {
      sessions = [];
      renderSessionList();
    }
  }

  function renderSessionList() {
    const filtered = sessions.filter((session) => {
      if (!currentFilter) return true;
      const haystack = `${session.sessionKey} ${session.userIdentifier || ""}`.toLowerCase();
      return haystack.includes(currentFilter.toLowerCase());
    });

    if (!filtered.find((session) => session.sessionKey === currentSessionKey)) {
      filtered.unshift({
        sessionKey: currentSessionKey,
        messageCount: 0,
        lastActivity: Date.now(),
      });
    }

    refs.sessionCountBadge.textContent = String(filtered.length);
    sessionListEl.innerHTML = "";

    for (const session of filtered) {
      const el = document.createElement("div");
      el.className = `session-item${session.sessionKey === currentSessionKey ? " active" : ""}`;
      const label = formatSessionName(session);
      el.innerHTML = `
        <button class="session-label" type="button" title="${escapeHtml(session.sessionKey)}">${escapeHtml(label)}</button>
        <span class="session-count" title="${Number(session.estimatedTokens || 0).toLocaleString()} tokens">${Number(session.estimatedTokens || 0).toLocaleString()}</span>
        <button class="session-delete" type="button" title="Delete session">×</button>
      `;

      el.querySelector(".session-label").addEventListener("click", () => switchSession(session.sessionKey));
      el.querySelector(".session-delete").addEventListener("click", async (event) => {
        event.stopPropagation();
        if (!confirm(`Delete session "${label}"?`)) return;
        await fetch(`/api/sessions/${encodeURIComponent(session.sessionKey)}`, { method: "DELETE" });
        if (session.sessionKey === currentSessionKey) {
          switchSession("webui:default");
        } else {
          loadSessions();
        }
      });

      sessionListEl.appendChild(el);
    }
  }

  function formatSessionName(sessionOrKey) {
    if (sessionOrKey && typeof sessionOrKey === "object" && sessionOrKey.sessionName) {
      return sessionOrKey.sessionName;
    }
    const key = typeof sessionOrKey === "string" ? sessionOrKey : (sessionOrKey?.sessionKey || "");
    const matching = sessions.find(s => s.sessionKey === key);
    if (matching?.sessionName) {
      return matching.sessionName;
    }
    const identifier = typeof sessionOrKey === "string" ? "" : sessionOrKey?.userIdentifier || "";
    if (key.startsWith("discord:")) return `Discord / ${identifier || key.slice(8)}`;
    if (key.startsWith("whatsapp:")) return `WhatsApp / ${identifier || key.slice(9)}`;
    if (key.startsWith("cli:")) return `CLI / ${key.slice(4)}`;
    if (key === "webui:default") return "WebUI / default";
    if (key.startsWith("webui:chat_")) return `WebUI / ${key.slice(11)}`;
    return key.replace(/^webui:/, "").replace(/_/g, " ");
  }

  function updateHeader() {
    refs.chatTitle.textContent = formatSessionName(currentSessionKey);
    refs.chatSubtitle.textContent = currentSessionKey;
    renderSessionMetrics();
  }

  function switchSession(sessionKey) {
    currentSessionKey = sessionKey;
    currentAssistantEl = null;
    currentContent = "";
    isStreaming = false;
    messagesEl.innerHTML = "";
    updateHeader();
    loadSessionHistory(sessionKey);
    loadSessionTaskPlan(sessionKey);
    loadSessions();
    fetchSessionMetrics(sessionKey);
    sendSessionInit();
    setSidebarOpen(false);
  }

  async function fetchSessionMetrics(sessionKey) {
    try {
      const data = await fetchJson(`/api/sessions/${encodeURIComponent(sessionKey)}/metrics`);
      if (sessionKey !== currentSessionKey) return;
      currentSessionMetrics = data || currentSessionMetrics;
      renderSessionMetrics();
    } catch {
      if (sessionKey !== currentSessionKey) return;
      currentSessionMetrics = {
        estimatedTokens: 0,
        messageCount: 0,
        imageCount: 0,
        contextMaxTokens: 0,
        contextBudgetPct: 80,
        contextBudgetTokens: 0,
        compactionThresholdPct: 90,
        compactionThresholdTokens: 0,
        compactionProgressPct: 0,
      };
      renderSessionMetrics();
    }
  }

  function renderSessionMetrics() {
    if (!refs.tokenLabel) return;
    const tokens = Number(currentSessionMetrics.estimatedTokens || 0).toLocaleString();
    const threshold = Number(currentSessionMetrics.compactionThresholdTokens || 0);
    const progress = Number(currentSessionMetrics.compactionProgressPct || 0);
    refs.tokenLabel.textContent = threshold > 0
      ? `Tokens ${tokens} / compaction ${progress}% of ${threshold.toLocaleString()}`
      : `Tokens ${tokens}`;
  }

  async function loadSessionHistory(sessionKey) {
    try {
      const data = await fetchJson(`/api/sessions/${encodeURIComponent(sessionKey)}/history`);
      const history = data.messages || [];
      if (history.length === 0) {
        showWelcome();
        return;
      }

      hideWelcome();
      history.forEach((msg) => {
        if (msg.role === "user") {
          const cleaned = cleanMessageContent(msg.content || "");
          let attachmentsList = [];
          if (msg.metadata) {
            try {
              const meta = JSON.parse(msg.metadata);
              attachmentsList = meta.attachments || [];
            } catch {}
          }
          addUserMessage(cleaned.text, attachmentsList, false, cleaned.sender);
        }
        if (msg.role === "assistant") {
          addRestoredAssistantMessage(msg.content || "");
        }
      });
      scrollToBottom();
    } catch {
      showWelcome();
    }
  }

  async function loadSessionTaskPlan(sessionKey) {
    try {
      const data = await fetchJson(`/api/sessions/${encodeURIComponent(sessionKey)}/task-plan`);
      if (sessionKey !== currentSessionKey) return;
      const taskPlan = data.taskPlan?.plan;
      if (!taskPlan || !Array.isArray(taskPlan.tasks) || taskPlan.tasks.length === 0) return;
      addRestoredTaskPlan(taskPlan);
    } catch {
      // Ignore missing task-plan state.
    }
  }

  function addRestoredTaskPlan(plan) {
    hideWelcome();
    const name = healthData.name || "LiteClaw";
    const initial = name.charAt(0).toUpperCase();
    const el = document.createElement("div");
    el.className = "message assistant restored-plan";
    el.dataset.messageRole = "plan";
    el.innerHTML = `
      <div class="message-avatar">${escapeHtml(initial)}</div>
      <div class="message-body">
        <div class="message-sender">${escapeHtml(name)}</div>
        <div class="message-content"></div>
      </div>
    `;
    messagesEl.appendChild(el);

    const previousAssistant = currentAssistantEl;
    currentAssistantEl = el;
    appendPlan(plan);
    currentAssistantEl = previousAssistant;
  }

  // ─── Health Rendering ─────────────────────────────────────────────

  function renderHealth() {
    const primary = healthData.model || "unknown";
    const name = healthData.name || "LiteClaw";
    
    // Update brand name in sidebar
    const brandName = $(".brand-name");
    if (brandName) brandName.textContent = name;
    
    // Update welcome screen if visible
    const welcomeTitle = $(".welcome-card h2");
    if (welcomeTitle) welcomeTitle.textContent = `I'm ${name}`;

    if (inputEl) inputEl.placeholder = `Message ${name}...`;

    refs.healthDot.className = "indicator-dot online";
    if (refs.healthLabel) refs.healthLabel.textContent = `Health (${healthData.sessionCount || sessions.length || 0})`;
    
    applyChannelState(refs.chWebui, healthData.channels?.webui?.status || "online");
    applyChannelState(refs.chDiscord, healthData.channels?.discord?.status || "unknown");
    applyChannelState(refs.chWhatsapp, healthData.channels?.whatsapp?.status || "unknown");
  }

  function applyChannelState(el, state) {
    if (!el) return;
    el.textContent = state;
    el.className = "channel-status-dot";
    if (state === "online" || state === "configured") el.classList.add("online");
    else if (state === "disabled") el.classList.add("offline");
    else el.classList.add("warning");
  }

  // ─── Settings ─────────────────────────────────────────────────────

  function hydrateSettings(config) {
    const models = config.llm?.availableModels || [];
    const currentPrimary = config.llm?.primary || "";
    const llmDefaults = config.llm?.defaults || config.llm || {};
    refs.settingModel.innerHTML = models.length
      ? models.map((model) => `<option value="${escapeHtml(model.id)}"${model.id === currentPrimary ? " selected" : ""}>${escapeHtml(model.label)}</option>`).join("")
      : `<option value="${escapeHtml(currentPrimary)}">${escapeHtml(currentPrimary || "unknown")}</option>`;

    refs.settingThinking.value = config.agent?.thinkingDefault || "medium";
    refs.settingTemperature.value = llmDefaults.temperature ?? 1.0;
    refs.settingTopP.value = llmDefaults.topP ?? 1.0;
    refs.settingTopK.value = llmDefaults.topK ?? 45;
    refs.settingMaxOutputTokens.value = llmDefaults.maxOutputTokens ?? 8192;
    updateLlmParamsVisibility();

    if (refs.settingAgentName) refs.settingAgentName.value = config.agent?.name || "";
    if (refs.settingWorkspace) refs.settingWorkspace.value = config.agent?.workspace || "";
    if (refs.settingMaxTurns) refs.settingMaxTurns.value = config.agent?.maxTurns || 20;
    refs.settingPlannerMode.value = config.agent?.planner?.mode || "auto";
    refs.settingPlannerMaxReplans.value = config.agent?.planner?.maxReplans ?? 2;
    refs.settingContextTokens.value = config.agent?.contextTokens || 64000;
    refs.settingContextBudgetPct.value = config.agent?.contextBudgetPct || 80;
    refs.settingCompactionThresholdPct.value = config.agent?.compaction?.softThresholdPct || 90;
    refs.settingSkillsMaxInjected.value = config.agent?.skills?.maxInjected || 2;
    refs.toggleSkillsEnabled.checked = !!config.agent?.skills?.enabled;

    document.querySelectorAll("#toolLoadingGroup .toggle-chip").forEach((button) => {
      button.classList.toggle("active", button.dataset.value === (config.agent?.toolLoading || "lazy"));
    });

    refs.toggleDiscord.checked = !!config.channels?.discord?.enabled;
    refs.discordReplyStyle.value = config.channels?.discord?.replyStyle || "single";
    refs.toggleDiscordToolProgress.checked = !!config.channels?.discord?.showToolProgress;
    refs.toggleWhatsapp.checked = !!config.channels?.whatsapp?.enabled;
    refs.whatsappReplyStyle.value = config.channels?.whatsapp?.replyStyle || "single";
    refs.toggleWhatsappToolProgress.checked = !!config.channels?.whatsapp?.showToolProgress;

    const voice = config.voice || {};
    refs.toggleVoiceEnabled.checked = !!voice.enabled;
    refs.settingVoiceTriggerMode.value = voice.triggerMode || "always";
    refs.settingVoiceWakeName.value = voice.wakeName || "";
    refs.settingVoiceMaxResponseTokens.value = voice.maxResponseTokens || 150;

    refs.toggleExecEnabled.checked = !!config.tools?.exec?.enabled;
    refs.toggleExecConfirm.checked = !!config.tools?.exec?.confirmDestructive;
    refs.toggleWebFetchEnabled.checked = !!config.tools?.web?.fetchEnabled;
    refs.toggleWebFallback.checked = !!config.tools?.web?.browserFallback;
    refs.toggleFilesystemEnabled.checked = !!config.tools?.filesystem?.enabled;
    refs.toggleConfirmDelete.checked = !!config.tools?.filesystem?.confirmDelete;
    refs.toggleVisionEnabled.checked = !!config.tools?.vision?.enabled;
    refs.settingVisionMaxDimension.value = config.tools?.vision?.maxDimensionPx || 1024;

    const google = config.llm?.providers?.google || {};
    if (refs.toggleGoogleVertex) refs.toggleGoogleVertex.checked = !!google.vertex;
    if (refs.toggleGoogleExpress) refs.toggleGoogleExpress.checked = !!google.express;
    if (refs.settingGoogleProject) refs.settingGoogleProject.value = google.project || "";
    if (refs.settingGoogleRegion) refs.settingGoogleRegion.value = google.region || "global";
    updateGoogleVisibility();

    refs.settingGatewayPort.value = config.gateway?.port || 7860;
    refs.settingGatewayBind.value = config.gateway?.bind || "loopback";
    refs.gatewayAuthNote.textContent = config.gateway?.authEnabled
      ? "Auth: gateway token configured"
      : "Auth: local WebUI endpoints open";

    // Extensions: D&D
    const dnd = config.extensions?.dnd || {};
    if (refs.toggleDndEnabled) refs.toggleDndEnabled.checked = dnd.enabled !== false;
    updateDndVisibility();

    // Populate DnD model selects from available models list
    const dndModels = config.llm?.availableModels || [];
    for (const sel of [refs.settingDndNarrativeModel, refs.settingDndLoadoutModel]) {
      if (!sel) continue;
      sel.innerHTML = `<option value="">(use primary model)</option>`
        + dndModels.map(m => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.label)}</option>`).join("");
    }
    if (refs.settingDndNarrativeModel) refs.settingDndNarrativeModel.value = dnd.narrativeModel || "";
    if (refs.settingDndLoadoutModel) refs.settingDndLoadoutModel.value = dnd.loadoutModel || "";
    if (refs.settingDndWorld) refs.settingDndWorld.value = dnd.defaultWorld || "elyndor";
    if (refs.settingDndTone) refs.settingDndTone.value = dnd.defaultTone || "heroic";
    if (refs.settingDndMaxPlayers) refs.settingDndMaxPlayers.value = dnd.maxPlayers || 6;
    if (refs.settingDndNarrativeTemp) refs.settingDndNarrativeTemp.value = dnd.narrativeTemperature ?? 0.9;
    if (refs.settingDndNarrativeMaxTokens) refs.settingDndNarrativeMaxTokens.value = dnd.narrativeMaxTokens || 4096;
    if (refs.toggleDndAutoProvision) refs.toggleDndAutoProvision.checked = dnd.autoProvision !== false;
  }

  function updateLlmParamsVisibility() {
    if (refs.llmParamsGroup) {
      refs.llmParamsGroup.style.display = "grid";
    }
  }

  function updateGoogleVisibility() {
    const isExpress = refs.toggleGoogleExpress?.checked;
    if (refs.settingGoogleProject) {
      refs.settingGoogleProject.disabled = isExpress;
      refs.settingGoogleProject.closest(".field").style.opacity = isExpress ? "0.5" : "1";
    }
    if (refs.settingGoogleRegion) {
      refs.settingGoogleRegion.disabled = isExpress;
      refs.settingGoogleRegion.closest(".field").style.opacity = isExpress ? "0.5" : "1";
    }
  }

  refs.settingModel.addEventListener("change", updateLlmParamsVisibility);
  if (refs.toggleGoogleExpress) {
    refs.toggleGoogleExpress.addEventListener("change", updateGoogleVisibility);
  }

  function updateDndVisibility() {
    if (!refs.dndExtensionBody) return;
    const enabled = refs.toggleDndEnabled?.checked ?? true;
    refs.dndExtensionBody.classList.toggle("disabled", !enabled);
  }

  if (refs.toggleDndEnabled) {
    refs.toggleDndEnabled.addEventListener("change", updateDndVisibility);
  }

  function gatherSettingsPayload() {
    const toolLoading = document.querySelector("#toolLoadingGroup .toggle-chip.active")?.dataset.value || "lazy";
    return {
      llm: {
        primary: refs.settingModel.value,
        temperature: Number(refs.settingTemperature.value || 1.0),
        topP: Number(refs.settingTopP.value || 1.0),
        topK: Number(refs.settingTopK.value || 45),
        maxOutputTokens: Number(refs.settingMaxOutputTokens.value || 8192),
      },
      agent: {
        name: refs.settingAgentName.value.trim(),
        workspace: refs.settingWorkspace.value.trim(),
        maxTurns: parseInt(refs.settingMaxTurns.value, 10),
        planner: {
          mode: refs.settingPlannerMode.value || "auto",
          maxReplans: Number(refs.settingPlannerMaxReplans.value || 2),
        },
        toolLoading,
        thinkingDefault: refs.settingThinking.value,
        contextTokens: Number(refs.settingContextTokens.value || 64000),
        contextBudgetPct: Number(refs.settingContextBudgetPct.value || 80),
        compaction: {
          softThresholdPct: Number(refs.settingCompactionThresholdPct.value || 90),
        },
        skills: {
          enabled: refs.toggleSkillsEnabled.checked,
          maxInjected: Number(refs.settingSkillsMaxInjected.value || 2),
        },
      },
      channels: {
        discord: {
          enabled: refs.toggleDiscord.checked,
          replyStyle: refs.discordReplyStyle.value,
          showToolProgress: refs.toggleDiscordToolProgress.checked,
        },
        whatsapp: {
          enabled: refs.toggleWhatsapp.checked,
          replyStyle: refs.whatsappReplyStyle.value,
          showToolProgress: refs.toggleWhatsappToolProgress.checked,
        },
      },
      voice: {
        enabled: refs.toggleVoiceEnabled.checked,
        triggerMode: refs.settingVoiceTriggerMode.value,
        wakeName: refs.settingVoiceWakeName.value.trim(),
        maxResponseTokens: parseInt(refs.settingVoiceMaxResponseTokens.value, 10) || 150,
      },
      tools: {
        exec: {
          enabled: refs.toggleExecEnabled.checked,
          confirmDestructive: refs.toggleExecConfirm.checked,
        },
        web: {
          fetchEnabled: refs.toggleWebFetchEnabled.checked,
          browserFallback: refs.toggleWebFallback.checked,
        },
        filesystem: {
          enabled: refs.toggleFilesystemEnabled.checked,
          confirmDelete: refs.toggleConfirmDelete.checked,
        },
        vision: {
          enabled: refs.toggleVisionEnabled.checked,
          maxDimensionPx: Number(refs.settingVisionMaxDimension.value || 1024),
        },
      },
      gateway: {
        port: Number(refs.settingGatewayPort.value || 7860),
        bind: refs.settingGatewayBind.value,
      },
      google: {
        vertex: refs.toggleGoogleVertex?.checked ?? false,
        express: refs.toggleGoogleExpress?.checked ?? false,
        project: refs.settingGoogleProject?.value.trim() ?? "",
        region: refs.settingGoogleRegion?.value.trim() ?? "global",
      },
      extensions: {
        dnd: {
          enabled: refs.toggleDndEnabled?.checked ?? true,
          narrativeModel: refs.settingDndNarrativeModel?.value ?? "",
          loadoutModel: refs.settingDndLoadoutModel?.value ?? "",
          defaultWorld: refs.settingDndWorld?.value ?? "elyndor",
          defaultTone: refs.settingDndTone?.value ?? "heroic",
          maxPlayers: Number(refs.settingDndMaxPlayers?.value || 6),
          autoProvision: refs.toggleDndAutoProvision?.checked ?? true,
          narrativeTemperature: Number(refs.settingDndNarrativeTemp?.value || 0.9),
          narrativeMaxTokens: Number(refs.settingDndNarrativeMaxTokens?.value || 4096),
        },
      },
    };
  }

  async function saveSettings() {
    try {
      saveSettingsBtn.disabled = true;
      saveSettingsBtn.textContent = "Saving...";
      const data = await fetchJson("/api/config", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(gatherSettingsPayload()),
      });
      currentConfig = data.config || currentConfig;
      hydrateSettings(currentConfig);
      await fetchHealth();
      showNotice("Settings saved. New turns will use the updated runtime.", "success", 2800);
    } catch (error) {
      showNotice(`Failed to save settings: ${error.message || error}`, "error", 3600);
    } finally {
      saveSettingsBtn.disabled = false;
      saveSettingsBtn.textContent = "Save settings";
    }
  }

  // ─── Workspace ────────────────────────────────────────────────────

  async function loadWorkspace(path = workspacePath) {
    try {
      const data = await fetchJson(`/api/workspace/tree?path=${encodeURIComponent(path)}`);
      workspacePath = data.currentPath || ".";
      workspacePathLabel.textContent = workspacePath;
      workspaceTree.innerHTML = "";

      for (const entry of data.entries || []) {
        const button = document.createElement("button");
        button.className = `workspace-entry${entry.path === selectedWorkspaceFile ? " active" : ""}`;
        button.type = "button";
        button.innerHTML = `
          <span class="workspace-entry-kind">${entry.kind === "directory" ? "dir" : "file"}</span>
          <span class="workspace-entry-name">${escapeHtml(entry.name)}</span>
          <span class="workspace-entry-meta">${formatSize(entry.size)}</span>
        `;
        button.addEventListener("click", () => {
          if (entry.kind === "directory") {
            selectedWorkspaceFile = "";
            loadWorkspace(entry.path);
            return;
          }
          selectedWorkspaceFile = entry.path;
          openWorkspaceFile(entry.path);
        });
        workspaceTree.appendChild(button);
      }
    } catch (error) {
      showNotice(`Workspace error: ${error.message || error}`, "error", 3000);
    }
  }

  async function openWorkspaceFile(path) {
    try {
      const data = await fetchJson(`/api/workspace/file?path=${encodeURIComponent(path)}`);
      workspacePreviewMeta.textContent = `${data.path} / ${formatSize(data.size)}${data.truncated ? " / truncated" : ""}`;
      workspacePreviewContent.textContent = data.isBinary
        ? "Binary preview is unavailable."
        : (data.content || "");
      Array.from(workspaceTree.querySelectorAll(".workspace-entry")).forEach((entry) => {
        entry.classList.toggle("active", entry.textContent.includes(path.split("/").pop()));
      });
    } catch (error) {
      workspacePreviewMeta.textContent = "Preview unavailable";
      workspacePreviewContent.textContent = String(error.message || error);
    }
  }

  // ─── Welcome Screen ──────────────────────────────────────────────

  function showWelcome() {
    if ($("#welcomeScreen")) return;
    const name = healthData.name || "LiteClaw";
    messagesEl.innerHTML = `
      <div class="welcome-panel" id="welcomeScreen">
        <div class="welcome-content">
          <div class="eyebrow">Launch pad</div>
          <h3>What can I help you with?</h3>
          <p>Chat with ${escapeHtml(name)} to inspect, edit, search, or plan across your workspace.</p>
          <div class="welcome-actions">
            <button class="welcome-chip" data-prompt="Inspect the current workspace and tell me what this project is.">Inspect workspace</button>
            <button class="welcome-chip" data-prompt="Read README.md and propose the next milestone.">Plan next milestone</button>
            <button class="welcome-chip" data-prompt="Search the web for today's AI news and summarize it.">Web research</button>
            <button class="welcome-chip" data-prompt="What tools and skills should you use for editing PDFs and DOCX files?">Tooling help</button>
          </div>
        </div>
      </div>
    `;
    bindWelcomeChips();
  }

  function hideWelcome() {
    const el = $("#welcomeScreen");
    if (el) el.remove();
  }

  // ─── Message Rendering ────────────────────────────────────────────

  /**
   * Strip Discord/WhatsApp metadata prefix from user messages
   * and extract the sender's identity.
   */
  function cleanMessageContent(text) {
    if (!text) return { text, sender: "You" };
    let cleaned = text;
    let sender = "You";

    // Extract sender from new compact format: [context: ... | sender: Alice (@alice)]
    const contextMatch = cleaned.match(/^\[context:[^\]]*sender:\s*([^\]|]+)(?:]|\|)/m);
    if (contextMatch && contextMatch[1]) {
      sender = contextMatch[1].trim();
    }

    // Strip compact format loops
    cleaned = cleaned.replace(/^\[context:[^\]]*\]\n?/gm, "");
    cleaned = cleaned.replace(/^\[participants:[^\]]*\]\n?/gm, "");

    // Old verbose format: strip everything before the actual user message
    if (cleaned.startsWith("Conversation info (untrusted metadata):")) {
      const lastCodeBlockEnd = cleaned.lastIndexOf("```");
      if (lastCodeBlockEnd !== -1) {
        cleaned = cleaned.slice(lastCodeBlockEnd + 3);
      }
      cleaned = cleaned.replace(/^Use only these handles[^\n]*\n?/gm, "");
    }

    // Strip backend-injected file contents
    cleaned = cleaned.replace(/\n\nAttached files content:[\s\S]*$/m, "");

    return { text: cleaned.trim(), sender };
  }

  function addUserMessage(text, attachmentsList, animate = true, sender = "You") {
    hideWelcome();
    const el = document.createElement("div");
    el.className = "message user" + (animate ? " animate-in" : "");
    el.dataset.messageRole = "user";
    el._rawText = text;
    const avatarInitial = sender.charAt(0).toUpperCase();

    let attachmentsHtml = "";
    if (attachmentsList && attachmentsList.length > 0) {
      attachmentsHtml = `<div class="message-attachments" style="margin-top: 8px; display: flex; flex-wrap: wrap; gap: 8px;">`;
      attachmentsList.forEach(att => {
        const dataUrl = typeof att === 'string' ? att : att.dataUrl;
        const name = typeof att === 'string' ? 'image' : att.name;
        const isImage = dataUrl.startsWith("data:image/");

        if (isImage) {
          attachmentsHtml += `<img src="${dataUrl}" alt="attachment" style="max-width:140px; border:1px solid var(--line); border-radius: 4px;">`;
        } else {
          const icon = name.endsWith(".pdf") ? "📄" : (name.endsWith(".xlsx") || name.endsWith(".csv")) ? "📊" : "📝";
          attachmentsHtml += `
            <div class="file-chip" style="opacity: 1; pointer-events: none;">
              <span class="icon">${icon}</span>
              <span>${escapeHtml(name)}</span>
            </div>`;
        }
      });
      attachmentsHtml += `</div>`;
    }

    el.innerHTML = `
      <div class="message-body">
        <div class="message-sender">${escapeHtml(sender)}</div>
        <div class="message-content">${escapeHtml(text)}</div>
        ${attachmentsHtml}
      </div>
      <div class="message-avatar">${escapeHtml(avatarInitial)}</div>
    `;
    messagesEl.appendChild(el);
    el._attachments = attachmentsList;
    scrollToBottom();
  }

  function addRestoredAssistantMessage(content) {
    const el = document.createElement("div");
    const name = healthData.name || "LiteClaw";
    const initial = name.charAt(0).toUpperCase();
    el.className = "message assistant";
    el.dataset.messageRole = "assistant";
    el.innerHTML = `
      <div class="message-avatar">${escapeHtml(initial)}</div>
      <div class="message-body">
        <div class="message-sender">${escapeHtml(name)}</div>
        <div class="message-content">${renderMarkdown(content)}</div>
      </div>
    `;
    messagesEl.appendChild(el);
    const contentEl = el.querySelector(".message-content");
    addCopyButtons(contentEl);
    addMessageActions(el.querySelector(".message-body"), content);
  }

  function ensureAssistantMessage() {
    if (currentAssistantEl) return;
    hideWelcome();
    const name = healthData.name || "LiteClaw";
    const initial = name.charAt(0).toUpperCase();
    currentAssistantEl = document.createElement("div");
    currentAssistantEl.className = "message assistant";
    currentAssistantEl.dataset.messageRole = "assistant";
    currentAssistantEl.innerHTML = `
      <div class="message-avatar">${escapeHtml(initial)}</div>
      <div class="message-body">
        <div class="message-sender">${escapeHtml(name)}</div>
        <div class="message-content"></div>
      </div>
    `;
    messagesEl.appendChild(currentAssistantEl);
    isStreaming = true;
  }

  function splitVisibleAndThinking(text) {
    let visible = String(text || "");
    const thinking = [];

    visible = visible.replace(/<(think|thought|thinking)>([\s\S]*?)(?:<\/\1>|$)/gi, (_, _tag, content) => {
      const entry = String(content || "");
      if (entry) thinking.push(entry);
      return "";
    });

    visible = visible
      .replace(/<\/?(think|thought|thinking)>/gi, "")
      .replace(/<[|｜]DSML[|｜]tool_calls[\s\S]*?<\/[|｜]DSML[|｜]tool_calls>/gi, "");

    const prefixMatch = visible.match(/^\s*(?:think|thoughts?|thinking)\s*\n([\s\S]*)$/i);
    if (prefixMatch && /\b(the user|i should|i need to|according to|first|let me|i'll|i will)\b/i.test(prefixMatch[1] || "")) {
      const trimmed = String(prefixMatch[1] || "").trim();
      if (trimmed) thinking.push(trimmed);
      visible = "";
    }

    return {
      visible: visible,
      thinking,
    };
  }

  function renderAssistantContent() {
    if (!currentAssistantEl) return;
    const contentEl = currentAssistantEl.querySelector(".message-content");
    const visibleContent = dedupeRepeatedParagraphs(splitVisibleAndThinking(currentContent).visible);
    contentEl.innerHTML = renderMarkdown(visibleContent) + (isStreaming ? '<span class="streaming-cursor"></span>' : "");
    addCopyButtons(contentEl);
    scrollToBottom();
    if (!isStreaming) {
      unfurlLinks(currentAssistantEl, visibleContent);
    }
  }

  function unfurlLinks(assistantEl, text) {
    const urls = text.match(/https?:\/\/[^\s<)\]]+/g);
    if (!urls) return;
    
    const uniqueUrls = [...new Set(urls)];
    let unfurlContainer = assistantEl.querySelector('.unfurl-container');
    if (!unfurlContainer) {
      unfurlContainer = document.createElement('div');
      unfurlContainer.className = 'unfurl-container';
      unfurlContainer.style.display = 'flex';
      unfurlContainer.style.flexWrap = 'wrap';
      unfurlContainer.style.gap = '8px';
      unfurlContainer.style.marginTop = '8px';
      assistantEl.querySelector(".message-body").appendChild(unfurlContainer);
    }

    uniqueUrls.forEach(async url => {
      if (unfurlContainer.querySelector(`[data-url="${CSS.escape(url)}"]`)) return;
      
      const placeholder = document.createElement('div');
      placeholder.dataset.url = url;
      placeholder.style.display = 'none';
      unfurlContainer.appendChild(placeholder);

      try {
        const data = await fetchJson(`/api/unfurl?url=${encodeURIComponent(url)}`);
        if (data.mediaUrl) {
          placeholder.style.display = 'block';
          if (data.mediaUrl.endsWith('.mp4') || data.mediaUrl.endsWith('.webm')) {
            placeholder.innerHTML = `<video src="${escapeHtml(data.mediaUrl)}" autoplay loop muted playsinline style="max-width:300px; max-height:300px; border-radius:8px; object-fit:contain;"></video>`;
          } else {
            placeholder.innerHTML = `<img src="${escapeHtml(data.mediaUrl)}" alt="media" style="max-width:300px; max-height:300px; border-radius:8px; object-fit:contain;" />`;
          }
          
          // Hide bare links from text nodes and empty links
          const contentEl = assistantEl.querySelector(".message-content");
          if (contentEl) {
            const walker = document.createTreeWalker(contentEl, NodeFilter.SHOW_TEXT, null, false);
            let node;
            while(node = walker.nextNode()) {
              if (node.nodeValue.includes(url)) {
                 node.nodeValue = node.nodeValue.replace(url, '').trim();
              }
            }
            const links = contentEl.querySelectorAll(`a[href="${CSS.escape(url)}"]`);
            links.forEach(link => {
               if (link.textContent === url || link.textContent === '') {
                 link.style.display = 'none';
               }
            });
          }
          
          scrollToBottom();
        }
      } catch (e) {
        console.error('Failed to unfurl', url, e);
      }
    });
  }

  function appendThinking(text) {
    const chunk = String(text || "");
    if (!chunk) return;
    if (chunk === lastThinkingChunk) return;

    ensureAssistantMessage();
    const body = currentAssistantEl.querySelector(".message-body");
    const wrappers = body.querySelectorAll(".thinking-wrapper");
    let wrapper = wrappers[wrappers.length - 1];

    if (!wrapper || wrapper.dataset.closed === "true") {
      wrapper = document.createElement("div");
      wrapper.className = "thinking-wrapper"; // Collapsed by default as requested

      const header = document.createElement("button");
      header.className = "thinking-header";
      header.type = "button";
      header.innerHTML = `
        <span class="thinking-toggle-icon"></span>
        <span class="thinking-label pulsing">Thinking...</span>
      `;

      const content = document.createElement("div");
      content.className = "thinking-content";

      wrapper.appendChild(header);
      wrapper.appendChild(content);
      body.insertBefore(wrapper, body.querySelector(".message-content"));
    }

    const contentEl = wrapper.querySelector(".thinking-content");
    const existing = contentEl._rawText || "";
    const merged = existing + chunk;
    contentEl._rawText = merged;

    // Render as Markdown for headers, lists, etc.
    contentEl.innerHTML = renderMarkdown(merged);

    lastThinkingChunk = chunk;
    scrollToBottom();
  }

  function appendToolBadge(toolName) {
    ensureAssistantMessage();
    const badge = document.createElement("div");
    badge.className = "tool-badge";
    badge.dataset.tool = toolName;
    badge.innerHTML = `<span class="tool-spinner"></span><span>${escapeHtml(toolName)}</span>`;
    currentAssistantEl.querySelector(".message-body").insertBefore(badge, currentAssistantEl.querySelector(".message-content"));
    scrollToBottom();
  }

  function appendToolResult(toolName, result) {
    const badges = currentAssistantEl?.querySelectorAll(".tool-badge") || [];
    badges.forEach((badge) => {
      if (badge.dataset.tool !== toolName || badge.dataset.resolved === "true") return;
      badge.dataset.resolved = "true";
      badge.classList.add(result?.success ? "success" : "error");
      badge.innerHTML = `<span>${result?.success ? "✓" : "✗"}</span><span>${escapeHtml(toolName)}</span>`;
    });
    scrollToBottom();
  }

  function appendPlan(plan) {
    if (!plan || !Array.isArray(plan.tasks)) return;
    ensureAssistantMessage();

    let block = currentAssistantEl.querySelector(".task-plan");
    if (!block) {
      block = document.createElement("div");
      block.className = "task-plan";
      currentAssistantEl.querySelector(".message-body").insertBefore(block, currentAssistantEl.querySelector(".message-content"));
    }

    const items = plan.tasks.map((task, index) => {
      const status = escapeHtml(task.status || "pending");
      const title = escapeHtml(task.title || `Task ${index + 1}`);
      return `<li data-task-id="${escapeHtml(task.id || `task_${index + 1}`)}"><span class="task-status ${status}">${status}</span><span class="task-title">${title}</span></li>`;
    }).join("");

    block.innerHTML = `
      <div class="task-plan-header">Task Plan</div>
      <div class="task-plan-summary">${escapeHtml(plan.summary || "Working through the request step by step.")}</div>
      <ol class="task-plan-list">${items}</ol>
    `;
    scrollToBottom();
  }

  function appendTaskUpdate(msg) {
    ensureAssistantMessage();
    if (msg.plan) appendPlan(msg.plan);

    const planEl = currentAssistantEl?.querySelector(".task-plan");
    if (!planEl) return;

    const taskId = msg.taskId || "";
    const taskStatus = msg.taskStatus || "pending";
    const taskTitle = msg.taskTitle || "Task";
    const taskIndex = msg.taskIndex || 0;
    const taskTotal = msg.taskTotal || 0;
    const taskSummary = msg.taskSummary || "";

    let item = taskId ? planEl.querySelector(`[data-task-id="${CSS.escape(taskId)}"]`) : null;
    if (!item) {
      const list = planEl.querySelector(".task-plan-list");
      item = document.createElement("li");
      item.dataset.taskId = taskId;
      list?.appendChild(item);
    }

    item.innerHTML = `
      <span class="task-status ${escapeHtml(taskStatus)}">${escapeHtml(taskStatus)}</span>
      <span class="task-title">[${escapeHtml(String(taskIndex))}/${escapeHtml(String(taskTotal))}] ${escapeHtml(taskTitle)}</span>
      ${taskSummary ? `<span class="task-summary">${escapeHtml(taskSummary)}</span>` : ""}
    `;
    scrollToBottom();
  }

  function appendError(text) {
    ensureAssistantMessage();
    const err = document.createElement("div");
    err.className = "error-block";
    err.textContent = text;
    currentAssistantEl.querySelector(".message-body").insertBefore(err, currentAssistantEl.querySelector(".message-content"));
  }

  function finishStreaming(metrics) {
    isStreaming = false;
    const finalized = splitVisibleAndThinking(currentContent);
    currentContent = dedupeRepeatedParagraphs(finalized.visible || "");
    renderAssistantContent();

    if (metrics && currentAssistantEl) {
      const body = currentAssistantEl.querySelector(".message-body");
      const metricsEl = document.createElement("div");
      metricsEl.className = "message-metrics";
      const totalSec = (metrics.durationMs / 1000).toFixed(1);
      const tps = metrics.tokPerSec.toFixed(1);
      metricsEl.innerHTML = `
        <span>${metrics.tokens} tokens</span>
        <span class="metric-sep"></span>
        <span>${totalSec}s</span>
        <span class="metric-sep"></span>
        <span>${tps} tok/s</span>
      `;
      body.appendChild(metricsEl);
    }

    addMessageActions(currentAssistantEl.querySelector(".message-body"), currentContent);

    currentAssistantEl?.querySelectorAll(".thinking-wrapper").forEach((el) => {
      el.dataset.closed = "true";
      el.classList.remove("expanded"); // Collapse after finishing
      const label = el.querySelector(".thinking-label");
      if (label) {
        label.classList.remove("pulsing");
        label.textContent = "Thoughts";
      }
    });
    currentAssistantEl = null;
    currentContent = "";
    lastThinkingChunk = "";
    inputEl.disabled = false;
    sendBtn.disabled = inputEl.value.trim().length === 0 && attachedImages.length === 0;
    inputEl.focus();
  }

  // ─── Server Message Handler ───────────────────────────────────────

  function handleServerMessage(msg) {
    switch (msg.type) {
      case "system":
        if (msg.health) {
          healthData = msg.health;
          renderHealth();
        }
        break;
      case "thinking":
        appendThinking(msg.content || "");
        break;
      case "content":
        ensureAssistantMessage();
        {
          const parsed = splitVisibleAndThinking(msg.content || "");
          parsed.thinking.forEach((entry) => appendThinking(entry));
          currentContent += parsed.visible || "";
        }
        renderAssistantContent();
        break;
      case "plan":
        appendPlan(msg.plan || null);
        break;
      case "task_update":
        appendTaskUpdate(msg);
        break;
      case "file_download": {
        const dataUrl = `data:application/octet-stream;base64,${msg.data}`;
        const a = document.createElement("a");
        a.href = dataUrl;
        a.download = msg.name || "downloaded_file";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        showNotice(`Downloading ${msg.name}...`, "success");
        break;
      }
      case "tool_start":
        appendToolBadge(msg.tool || "tool");
        break;
      case "tool_result":
        appendToolResult(msg.tool || "tool", msg.result || {});
        break;
      case "confirmation":
        showConfirmation(msg);
        break;
      case "done":
        finishStreaming(msg.metrics);
        loadSessions();
        fetchSessionMetrics(currentSessionKey);
        break;
      case "error":
        appendError(msg.content || "An unknown error occurred.");
        showNotice(msg.content || "An unknown error occurred.", "error", 3600);
        finishStreaming();
        break;
      case "config_reloaded":
        if (msg.config) {
          currentConfig = msg.config;
          hydrateSettings(currentConfig);
        }
        if (msg.health) {
          healthData = msg.health;
          renderHealth();
        }
        showNotice("Config reloaded from disk.", "info", 2400);
        break;
      case "session_metrics":
        if (msg.sessionKey === currentSessionKey && msg.metrics) {
          currentSessionMetrics = msg.metrics;
          renderSessionMetrics();
        }
        loadSessions();
        break;
      case "voice_state":
        setVoiceState(msg.state);
        break;
      case "voice_transcription":
        appendVoiceTranscript(msg.role, msg.content);
        break;
      case "voice_audio_chunk":
        handleVoiceAudioChunk(msg.audio, msg.text);
        break;
      case "voice_interrupt":
        interruptVoicePlayback();
        break;
      case "pong":
        break;
      default:
        break;
    }
  }

  // ─── Confirmation ─────────────────────────────────────────────────

  function showConfirmation(msg) {
    pendingConfirmationId = msg.confirmationId || msg.id || null;
    confirmBody.innerHTML = renderMarkdown(msg.body || msg.description || msg.content || "Confirmation required.");
    confirmModal.hidden = false;
  }

  function respondToConfirmation(confirmed) {
    if (!pendingConfirmationId || !ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
      type: "confirmation_response",
      confirmationId: pendingConfirmationId,
      confirmed,
    }));
    pendingConfirmationId = null;
    confirmModal.hidden = true;
  }

  function cancelPendingModalAction() {
    pendingRegeneration = null;
    pendingConfirmationId = null;
    confirmModal.hidden = true;
  }

  // ─── Send Message ─────────────────────────────────────────────────

  function sendMessage(manualText = null) {
    const text = manualText !== null ? manualText : inputEl.value.trim();
    if ((!text && attachments.length === 0) || !ws || ws.readyState !== WebSocket.OPEN) return;

    addUserMessage(text || "(attachments)", [...attachments]);
    ws.send(JSON.stringify({
      type: "message",
      content: text,
      sessionKey: currentSessionKey,
      workingDir: refs.settingWorkspace.value.trim() || currentConfig.agent?.workspace,
      attachments: attachments.length ? [...attachments] : undefined,
    }));

    if (manualText === null) {
      inputEl.value = "";
      inputEl.style.height = "auto";
      attachments.length = 0;
      imagePreview.hidden = true;
      imagePreview.innerHTML = "";
      sendBtn.disabled = true;
    }
  }

  // ─── Markdown Renderer ────────────────────────────────────────────

  function renderMarkdown(text) {
    const codeBlocks = [];
    const thinkBlocks = [];
    const tableBlocks = [];
    let working = normalizeMarkdownForDisplay(dedupeRepeatedParagraphs(String(text || "")));

    // Isolate thinking blocks so they don't get wrapped inside <p> tags.
    working = working.replace(/<(think|thought|thinking)>([\s\S]*?)(?:<\/\1>|$)/gi, (_, _tag, content) => {
      const token = `@@THINK${thinkBlocks.length}@@`;
      thinkBlocks.push(`
        <div class="thinking-wrapper">
          <button class="thinking-header" type="button">
            <span class="thinking-toggle-icon"></span>
            <span class="thinking-label">Thoughts</span>
          </button>
          <div class="thinking-content">${escapeHtml(content.trim())}</div>
        </div>
      `);
      return `\n\n${token}\n\n`;
    });

    working = working.replace(/```([\w-]*)\n([\s\S]*?)```/g, (_, lang, code) => {
      const token = `@@CODE${codeBlocks.length}@@`;
      codeBlocks.push(`<pre><code class="language-${escapeHtml(lang || "text")}">${escapeHtml(code.trim())}</code></pre>`);
      return token;
    });

    // Isolate markdown table blocks before paragraph splitting
    working = working.replace(/((?:^[ \t]*\|[^\n]+\|[ \t]*\n)[ \t]*\|(?:\s*:?-+:?\s*\|)+\n(?:[ \t]*\|[^\n]+\|[ \t]*(?:\n|$))+)/gm, (match) => {
      const token = `@@TABLE${tableBlocks.length}@@`;
      tableBlocks.push(renderMarkdownTable(match.trim()));
      return `\n\n${token}\n\n`;
    });

    let html = escapeHtml(working)
      .replace(/^### (.+)$/gm, "<h4>$1</h4>")
      .replace(/^## (.+)$/gm, "<h3>$1</h3>")
      .replace(/^# (.+)$/gm, "<h2>$1</h2>")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^\*])\*(.+?)\*/g, "$1<em>$2</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    html = html.split(/\n{2,}/).map((chunk) => {
      // Don't trim the end during streaming to avoid stripping the space the model just typed
      const trimmed = isStreaming ? chunk.trimStart() : chunk.trim();
      if (!trimmed) return "";
      if (/^@@CODE\d+@@$/.test(trimmed) || /^@@THINK\d+@@$/.test(trimmed) || /^@@TABLE\d+@@$/.test(trimmed) || /^<h[234]>/.test(trimmed)) return trimmed;
      if (looksLikeMarkdownTable(trimmed)) {
        return renderMarkdownTable(trimmed);
      }
      if (/^[-*]\s+/m.test(trimmed) || /^\d+\.\s+/m.test(trimmed)) {
        const lines = trimmed.split("\n");
        const result = [];
        let currentListType = null; // 'ul', 'ol', or null

        for (const line of lines) {
          const ulMatch = line.match(/^[-*]\s+(.+)$/);
          const olMatch = line.match(/^(\d+)\.\s+(.+)$/);
          const type = ulMatch ? "ul" : (olMatch ? "ol" : null);

          if (type) {
            if (currentListType !== type) {
              if (currentListType) result.push(`</${currentListType}>`);
              result.push(`<${type}>`);
              currentListType = type;
            }
            result.push(`<li>${ulMatch ? ulMatch[1] : olMatch[2]}</li>`);
          } else {
            if (currentListType) {
              result.push(`</${currentListType}>`);
              currentListType = null;
            }
            result.push(line);
          }
        }
        if (currentListType) result.push(`</${currentListType}>`);
        return result.join("\n");
      }
      return `<p>${trimmed.replace(/\n/g, "<br>")}</p>`;
    }).join("");

    // Restore tokens
    codeBlocks.forEach((block, i) => {
      html = html.replace(`@@CODE${i}@@`, block);
    });
    thinkBlocks.forEach((block, i) => {
      html = html.replace(`@@THINK${i}@@`, block);
    });
    tableBlocks.forEach((block, i) => {
      html = html.replace(`@@TABLE${i}@@`, block);
    });

    return html;
  }

  function normalizeMarkdownForDisplay(text) {
    let normalized = String(text || "").replace(/\r\n/g, "\n");

    normalized = normalized
      .replace(/([^\n])\s*(#{1,3}\s+)/g, "$1\n\n$2")
      .replace(/([^\n])\s*(\|[^\n]+\|\s*\n\|[-:| ]+\|)/g, "$1\n\n$2")
      .replace(/([^\n])\s*(?:-\s+)/g, (match, prefix) => `${prefix}\n- `)
      .replace(/([^\n])\s*(\d+\.\s+)/g, "$1\n$2")
      .replace(/(#{1,6}[^\n#|]+)(?=#{1,6}\s)/g, "$1\n\n")
      .replace(/(\|\|)(?=[A-Za-z#])/g, "||\n")
      .replace(/([a-z0-9])(?=#{1,6}[A-Z])/g, "$1\n\n")
      .replace(/([.!?])\s+(?=[A-Z][a-z].{0,40}:)/g, "$1\n")
      .replace(/([a-z0-9)])(\*\*[A-Z])/g, "$1\n\n$2")
      .replace(/([a-z0-9)])(#{1,6}\s)/g, "$1\n\n$2")
      .replace(/([a-z])([A-Z][a-z]+:)/g, "$1\n$2");

    return repairCompressedVisibleText(normalized);
  }

  function repairCompressedVisibleText(text) {
    return String(text || "")
      .split("\n")
      .map((line) => repairCompressedVisibleLine(line))
      .join("\n")
      .replace(/\n{3,}/g, "\n\n");
  }

  function repairCompressedVisibleLine(line) {
    return line; // Disabled aggressive repair that was stripping/mangling whitespace
  }

  function dedupeRepeatedParagraphs(text) {
    const parts = String(text || "")
      .replace(/\r\n/g, "\n")
      .split(/\n{2,}/)
      .filter(Boolean);

    const deduped = [];
    let previous = "";
    for (const part of parts) {
      if (part === previous) continue;
      if (deduped.includes(part) && part.length > 120) continue;
      deduped.push(part);
      previous = part;
    }

    return deduped.join("\n\n");
  }

  function normalizeThinkingForDisplay(text) {
    return String(text || "")
      .replace(/\r\n/g, "\n")
      .replace(/\t/g, "  ")
      .split("\n")
      .map((line) => repairCompressedThinkingLine(line))
      .join("\n")
      .replace(/([.!?])\s+(?=[A-Z][a-z])/g, "$1\n")
      .replace(/(:)\s+(?=(?:First|Second|Third|Next|Then|Finally|The user|I need|I should|I'll|I will|Let me)\b)/g, "$1\n")
      .replace(/\n{3,}/g, "\n\n");
  }

  function repairCompressedThinkingLine(line) {
    return line; // Disabled aggressive repair
  }

  function normalizeThinkingForComparison(text) {
    return String(text || "")
      .toLowerCase()
      .replace(/\s+/g, " ")
      .replace(/[^\p{L}\p{N} ]/gu, "")
      .trim();
  }

  function isNearDuplicateThinkingChunk(existing, chunk) {
    if (!chunk || chunk.trim().length < 40) return false;
    if (!existing) return false;

    const normalizedChunk = normalizeThinkingForComparison(chunk);
    const normalizedExisting = normalizeThinkingForComparison(existing);

    if (!normalizedChunk) return false;
    if (normalizedExisting.includes(normalizedChunk)) return true;
    if (normalizedChunk.length > 120 && normalizedExisting.includes(normalizedChunk.slice(0, 120))) {
      return true;
    }

    return false;
  }

  function mergeThinkingChunks(existing, chunk) {
    return existing + chunk;
  }

  function looksLikeMarkdownTable(text) {
    const lines = text.split("\n").map((line) => line.trim()).filter(Boolean);
    if (lines.length < 2) return false;
    if (!lines[0].includes("|")) return false;
    return /^\|?[\s:-]+\|[\s|:-]*$/.test(lines[1]);
  }

  function formatTableCell(cell) {
    return escapeHtml(cell)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^\*])\*(.+?)\*/g, "$1<em>$2</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  }

  function renderMarkdownTable(text) {
    const lines = text.split("\n").map((line) => line.trim()).filter(Boolean);
    if (lines.length < 2) return `<p>${escapeHtml(text)}</p>`;

    const parseRow = (line) => line
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => cell.trim());

    const headers = parseRow(lines[0]);
    const rows = lines.slice(2).map(parseRow);

    const headHtml = headers.map((cell) => `<th>${formatTableCell(cell)}</th>`).join("");
    const bodyHtml = rows.map((row) => {
      const cells = headers.map((_, index) => `<td>${formatTableCell(row[index] ?? "")}</td>`).join("");
      return `<tr>${cells}</tr>`;
    }).join("");

    return `<div class="table-wrap"><table><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table></div>`;
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function addCopyButtons(container) {
    if (!container) return;
    container.querySelectorAll("pre").forEach((pre) => {
      if (pre.querySelector(".copy-btn")) return;
      const btn = document.createElement("button");
      btn.className = "copy-btn";
      btn.innerHTML = "Copy";
      btn.addEventListener("click", () => {
        navigator.clipboard.writeText(pre.querySelector("code")?.textContent || "");
        btn.innerHTML = "✓ Copied";
        btn.classList.add("success");
        setTimeout(() => {
          btn.innerHTML = "Copy";
          btn.classList.remove("success");
        }, 2000);
      });
      pre.appendChild(btn);
    });
  }

  function addMessageActions(body, content) {
    if (!body || body.querySelector(".message-actions")) return;
    const actions = document.createElement("div");
    actions.className = "message-actions";
    const messageEl = body.closest(".message.assistant");

    const copyBtn = document.createElement("button");
    copyBtn.className = "action-btn";
    copyBtn.innerHTML = `<span>📋</span> Copy`;
    copyBtn.title = "Copy full message content";
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(content);
      copyBtn.innerHTML = `<span>✓</span> Copied`;
      copyBtn.classList.add("success");
      setTimeout(() => {
        copyBtn.innerHTML = `<span>📋</span> Copy`;
        copyBtn.classList.remove("success");
      }, 2000);
    });

    const regenBtn = document.createElement("button");
    regenBtn.className = "action-btn";
    regenBtn.innerHTML = `<span>⟳</span> Regenerate`;
    regenBtn.title = "Regenerate assistant response";
    regenBtn.addEventListener("click", () => {
      if (messageEl) requestRegeneration(messageEl);
    });

    actions.appendChild(copyBtn);
    actions.appendChild(regenBtn);
    body.appendChild(actions);
  }

  function requestRegeneration(assistantEl) {
    if (isStreaming || isRegenerating) return;
    const turn = findRegenerationTurn(assistantEl);
    if (!turn) {
      showNotice("Couldn’t locate the original prompt for that reply.", "error", 2800);
      return;
    }
    pendingRegeneration = turn;
    const turnsAffected = Math.max(1, Math.floor(turn.rollbackCount / 2));
    const label = turnsAffected > 1 ? `This will remove this reply and ${turnsAffected - 1} later turn(s), then replay the selected prompt.` : `This will replace this reply and replay the same prompt.`;
    confirmBody.innerHTML = `<p>${label}</p>`;
    confirmModal.hidden = false;
  }

  function findRegenerationTurn(assistantEl) {
    const conversation = Array.from(messagesEl.querySelectorAll('.message[data-message-role="user"], .message[data-message-role="assistant"]'));
    const assistantIndex = conversation.indexOf(assistantEl);
    if (assistantIndex === -1) return null;

    let userIndex = -1;
    for (let index = assistantIndex - 1; index >= 0; index -= 1) {
      if (conversation[index].dataset.messageRole === "user") {
        userIndex = index;
        break;
      }
    }

    if (userIndex === -1) return null;

    const userEl = conversation[userIndex];
    const text = userEl._rawText || userEl.querySelector(".message-content")?.textContent || "";
    const savedAttachments = Array.isArray(userEl._attachments) ? [...userEl._attachments] : [];
    const rollbackCount = conversation.length - userIndex;

    return {
      assistantEl,
      userEl,
      text,
      attachments: savedAttachments,
      rollbackCount,
    };
  }

  async function confirmRegeneration() {
    if (!pendingRegeneration || isRegenerating) return;
    const turn = pendingRegeneration;
    pendingRegeneration = null;
    pendingConfirmationId = null;
    confirmModal.hidden = true;
    isRegenerating = true;

    try {
      const response = await fetchJson(`/api/sessions/${encodeURIComponent(currentSessionKey)}/rollback?count=${encodeURIComponent(turn.rollbackCount)}`, { method: "POST" });
      if (!response.success) {
        throw new Error("Rollback request was rejected.");
      }

      removeConversationTail(turn.userEl);

      currentAssistantEl = null;
      currentContent = "";
      isStreaming = false;

      attachments.length = 0;
      turn.attachments.forEach((item) => attachments.push(item));
      sendMessage(turn.text);
      showNotice("Regenerating response…", "info", 1800);
      fetchSessionMetrics(currentSessionKey);
      loadSessions();
    } catch (error) {
      showNotice(`Regenerate failed: ${error.message || error}`, "error", 3200);
    } finally {
      isRegenerating = false;
    }
  }

  function removeConversationTail(startEl) {
    let node = startEl;
    while (node) {
      const next = node.nextElementSibling;
      node.remove();
      node = next;
    }
    if (!messagesEl.children.length) showWelcome();
  }

  // ─── Utilities ────────────────────────────────────────────────────

  function showNotice(text, level = "info", timeout = 2600) {
    const el = document.createElement("div");
    el.className = `notice ${level}`;
    el.textContent = text;
    noticeStack.appendChild(el);
    if (timeout > 0) setTimeout(() => el.remove(), timeout);
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    });
  }

  function sendSessionInit() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "session_init", sessionKey: currentSessionKey }));
    }
  }

  function bindWelcomeChips() {
    document.querySelectorAll(".welcome-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        inputEl.value = chip.dataset.prompt || "";
        sendBtn.disabled = !inputEl.value.trim();
        sendMessage();
      });
    });
  }

  function formatUptime(seconds) {
    if (seconds < 60) return `${Math.floor(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  }

  function formatSize(size) {
    if (!Number.isFinite(size)) return "--";
    if (size < 1024) return `${size} B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
    return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  }

  function exportCurrentSession() {
    const transcript = Array.from(messagesEl.querySelectorAll(".message")).map((message) => {
      const sender = message.querySelector(".message-sender")?.textContent || "Unknown";
      const content = message.querySelector(".message-content")?.innerText || "";
      return `${sender}\n${content}`.trim();
    }).join("\n\n---\n\n");
    const blob = new Blob([transcript], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${currentSessionKey.replace(/[:/]/g, "_")}.txt`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function createNewSession() {
    switchSession(`webui:chat_${Date.now().toString(36)}`);
  }

  // ─── Event Listeners ──────────────────────────────────────────────

  inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 200)}px`;
    sendBtn.disabled = inputEl.value.trim().length === 0 && attachments.length === 0;
  });

  inputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });

  window.addEventListener("paste", (event) => {
    const items = (event.clipboardData || event.originalEvent.clipboardData).items;
    const files = [];
    for (const item of items) {
      if (item.kind === "file") {
        files.push(item.getAsFile());
      }
    }
    if (files.length > 0) {
      handleFiles(files);
    }
  });

  function handleFiles(files) {
    const fileList = Array.from(files).slice(0, 8);
    // Don't clear attachedImages, just append
    fileList.forEach((file) => {
      const reader = new FileReader();
      const isImage = file.type.startsWith("image/");

      reader.onload = () => {
        const dataUrl = String(reader.result);
        const item = { name: file.name, dataUrl };
        attachments.push(item);

        const chip = document.createElement("div");
        if (isImage) {
          chip.className = "image-chip";
          chip.innerHTML = `<span>${escapeHtml(file.name)}</span><button type="button">×</button>`;
        } else {
          chip.className = "file-chip";
          const icon = file.name.endsWith(".pdf") ? "📄" : (file.name.endsWith(".xlsx") || file.name.endsWith(".csv")) ? "📊" : "📝";
          chip.innerHTML = `<span class="icon">${icon}</span><span>${escapeHtml(file.name)}</span><button type="button">×</button>`;
        }

        chip.querySelector("button").addEventListener("click", () => {
          const idx = attachments.indexOf(item);
          if (idx > -1) attachments.splice(idx, 1);
          chip.remove();
          if (attachments.length === 0) imagePreview.hidden = true;
          sendBtn.disabled = inputEl.value.trim().length === 0 && attachments.length === 0;
        });

        imagePreview.appendChild(chip);
        imagePreview.hidden = false;
        sendBtn.disabled = inputEl.value.trim().length === 0 && attachments.length === 0;
      };

      if (isImage || file.type === "application/pdf" || file.type.includes("word") || file.type.includes("sheet") || file.type.startsWith("text/")) {
        reader.readAsDataURL(file);
      } else {
        showNotice(`File type ${file.type} not supported for direct reading.`, "warning");
      }
    });
  }

  imageInput.addEventListener("change", () => {
    handleFiles(imageInput.files);
    imageInput.value = "";
  });

  sessionSearch.addEventListener("input", () => {
    currentFilter = sessionSearch.value.trim();
    renderSessionList();
  });

  refs.newChatBtn.addEventListener("click", createNewSession);
  refs.clearBtn.addEventListener("click", async () => {
    messagesEl.innerHTML = "";
    currentAssistantEl = null;
    currentContent = "";
    await fetch(`/api/sessions/${encodeURIComponent(currentSessionKey)}`, { method: "DELETE" }).catch(() => {});
    showWelcome();
    loadSessions();
  });
  refs.exportBtn.addEventListener("click", exportCurrentSession);

  sendBtn.addEventListener("click", sendMessage);
  workspaceBtn.addEventListener("click", async () => {
    workspaceDrawer.hidden = false;
    await loadWorkspace(".");
  });
  closeWorkspaceBtn.addEventListener("click", () => {
    workspaceDrawer.hidden = true;
  });
  workspaceRefreshBtn.addEventListener("click", () => loadWorkspace(workspacePath));
  workspaceUpBtn.addEventListener("click", () => {
    if (workspacePath === "." || !workspacePath) {
      loadWorkspace(".");
      return;
    }
    loadWorkspace(workspacePath.split("/").slice(0, -1).join("/") || ".");
  });

  settingsBtn.addEventListener("click", () => {
    settingsOverlay.hidden = false;
    fetchHealth();
    fetchConfig();
  });
  closeSettingsBtn.addEventListener("click", () => {
    settingsOverlay.hidden = true;
  });
  saveSettingsBtn.addEventListener("click", saveSettings);
  clearAllBtn.addEventListener("click", async () => {
    if (!confirm("Delete all sessions? This cannot be undone.")) return;
    try {
      const data = await fetchJson("/api/sessions");
      for (const session of data.sessions || []) {
        await fetch(`/api/sessions/${encodeURIComponent(session.sessionKey)}`, { method: "DELETE" });
      }
      switchSession("webui:default");
      showNotice("All sessions deleted.", "success", 2400);
    } catch (error) {
      showNotice(`Failed to clear sessions: ${error.message || error}`, "error", 3200);
    }
  });

  document.querySelectorAll("#toolLoadingGroup .toggle-chip").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("#toolLoadingGroup .toggle-chip").forEach((peer) => peer.classList.remove("active"));
      button.classList.add("active");
    });
  });

  confirmAccept.addEventListener("click", () => {
    if (pendingRegeneration) {
      confirmRegeneration();
      return;
    }
    respondToConfirmation(true);
  });
  confirmReject.addEventListener("click", () => {
    if (pendingRegeneration) {
      cancelPendingModalAction();
      return;
    }
    respondToConfirmation(false);
  });
  confirmModal.addEventListener("click", (event) => {
    if (event.target !== confirmModal) return;
    if (pendingRegeneration) {
      cancelPendingModalAction();
      return;
    }
    respondToConfirmation(false);
  });
  settingsOverlay.addEventListener("click", (event) => {
    if (event.target === settingsOverlay) settingsOverlay.hidden = true;
  });

  // Sidebar toggles (mobile)
  if (sidebarToggle) {
    sidebarToggle.addEventListener("click", () => setSidebarOpen(false));
  }
  if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener("click", () => setSidebarOpen(!sidebar.classList.contains("open")));
  }
  if (sidebarBackdrop) {
    sidebarBackdrop.addEventListener("click", () => setSidebarOpen(false));
  }

  // ─── Onboarding ───────────────────────────────────────────────────

  async function checkOnboarding() {
    // If agent name is default "LiteClaw", show onboarding
    if (healthData.name === "LiteClaw" || !healthData.name) {
      refs.onboardingModal.hidden = false;
    }
  }

  refs.onboardingSubmit?.addEventListener("click", async () => {
    const newName = refs.onboardingName.value.trim();
    if (!newName) return;
    
    try {
      refs.onboardingSubmit.disabled = true;
      refs.onboardingSubmit.textContent = "Setting up...";
      
      await fetchJson("/api/config", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          agent: { name: newName }
        })
      });
      
      refs.onboardingModal.hidden = true;
      showNotice(`Welcome, ${newName}!`, "success");
      fetchHealth(); // Refresh UI
    } catch (error) {
      showNotice(`Failed to set name: ${error.message}`, "error");
      refs.onboardingSubmit.disabled = false;
      refs.onboardingSubmit.textContent = "Get started";
    }
  });

  // ─── Initialization ───────────────────────────────────────────────

  // Event delegation for thinking accordions
  messagesEl.addEventListener("click", (e) => {
    const header = e.target.closest(".thinking-header");
    if (header) {
      const wrapper = header.closest(".thinking-wrapper");
      if (wrapper) {
        wrapper.classList.toggle("expanded");
      }
    }
  });

  // ─── Voice Lounge client logic ────────────────────────────────────────

  const voiceLoungeBtn = $("#voiceLoungeBtn");
  const voiceLoungeOverlay = $("#voiceLoungeOverlay");
  const closeVoiceLoungeBtn = $("#closeVoiceLoungeBtn");
  const voiceStatusLabel = $("#voiceStatusLabel");
  const voiceVisualizer = $("#voiceVisualizer");
  const toggleVoiceConnectBtn = $("#toggleVoiceConnectBtn");
  const voiceMuteCheckbox = $("#voiceMuteCheckbox");
  const voiceTriggerModeSelect = $("#voiceTriggerModeSelect");
  const voiceInputDeviceSelect = $("#voiceInputDeviceSelect");
  const voiceOutputDeviceSelect = $("#voiceOutputDeviceSelect");
  const pttBtn = $("#pttBtn");
  const voiceTranscript = $("#voiceTranscript");
  const voiceStateDot = $("#voiceStateDot");
  const voiceStateLabel = $("#voiceStateLabel");

  let selectedInputDeviceId = localStorage.getItem("liteclaw_voice_input_device") || "";
  let selectedOutputDeviceId = localStorage.getItem("liteclaw_voice_output_device") || "";

  async function populateVoiceDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      
      const currentInput = voiceInputDeviceSelect.value || selectedInputDeviceId;
      const currentOutput = voiceOutputDeviceSelect.value || selectedOutputDeviceId;
      
      voiceInputDeviceSelect.innerHTML = '<option value="">Default Microphone</option>';
      voiceOutputDeviceSelect.innerHTML = '<option value="">Default Speaker</option>';
      
      let inputCount = 0;
      let outputCount = 0;
      
      devices.forEach((device) => {
        if (device.kind === "audioinput") {
          inputCount++;
          const label = device.label || `Microphone ${inputCount}`;
          const option = document.createElement("option");
          option.value = device.deviceId;
          option.textContent = label;
          if (device.deviceId === currentInput) option.selected = true;
          voiceInputDeviceSelect.appendChild(option);
        } else if (device.kind === "audiooutput") {
          outputCount++;
          const label = device.label || `Speaker/Headphones ${outputCount}`;
          const option = document.createElement("option");
          option.value = device.deviceId;
          option.textContent = label;
          if (device.deviceId === currentOutput) option.selected = true;
          voiceOutputDeviceSelect.appendChild(option);
        }
      });
      
      selectedInputDeviceId = voiceInputDeviceSelect.value;
      selectedOutputDeviceId = voiceOutputDeviceSelect.value;
    } catch (err) {
      console.warn("Failed to enumerate audio devices:", err);
    }
  }

  voiceLoungeBtn.addEventListener("click", () => {
    voiceLoungeOverlay.hidden = false;
    initVisualizerAnimation();
    populateVoiceDevices();
  });

  closeVoiceLoungeBtn.addEventListener("click", () => {
    voiceLoungeOverlay.hidden = true;
    disconnectVoice();
    if (voiceAnimationFrameId) {
      cancelAnimationFrame(voiceAnimationFrameId);
      voiceAnimationFrameId = null;
    }
  });

  toggleVoiceConnectBtn.addEventListener("click", async () => {
    if (isVoiceConnected) {
      disconnectVoice();
    } else {
      await connectVoice();
    }
  });

  voiceMuteCheckbox.addEventListener("change", () => {
    isVoiceMuted = voiceMuteCheckbox.checked;
    voiceStatusLabel.textContent = isVoiceMuted ? "Microphone Muted" : "Microphone Active";
  });

  voiceTriggerModeSelect.addEventListener("change", () => {
    voiceTriggerMode = voiceTriggerModeSelect.value;
    if (voiceTriggerMode === 'push-to-talk') {
      pttBtn.style.display = "block";
    } else {
      pttBtn.style.display = "none";
    }
    // If connected, restart session with new mode
    if (isVoiceConnected) {
      sendVoiceStart();
    }
  });

  voiceInputDeviceSelect.addEventListener("change", async () => {
    selectedInputDeviceId = voiceInputDeviceSelect.value;
    localStorage.setItem("liteclaw_voice_input_device", selectedInputDeviceId);
    
    if (isVoiceConnected) {
      showNotice("Switching input device...", "info", 1500);
      
      if (voiceMicStream) {
        voiceMicStream.getTracks().forEach(track => track.stop());
      }
      
      try {
        const constraints = {
          audio: selectedInputDeviceId ? { deviceId: { exact: selectedInputDeviceId } } : true,
          video: false
        };
        voiceMicStream = await navigator.mediaDevices.getUserMedia(constraints);
        
        if (voiceAudioContext && voiceMicSource) {
          voiceMicSource.disconnect();
          voiceMicSource = voiceAudioContext.createMediaStreamSource(voiceMicStream);
          voiceMicSource.connect(voiceMicAnalyser);
        }
      } catch (err) {
        console.error("Failed to switch input device:", err);
        showNotice("Failed to switch input device: " + err.message, "error");
        disconnectVoice();
      }
    }
  });

  voiceOutputDeviceSelect.addEventListener("change", async () => {
    selectedOutputDeviceId = voiceOutputDeviceSelect.value;
    localStorage.setItem("liteclaw_voice_output_device", selectedOutputDeviceId);
    
    if (voiceAudioContext) {
      if (typeof voiceAudioContext.setSinkId === 'function') {
        try {
          await voiceAudioContext.setSinkId(selectedOutputDeviceId);
          showNotice("Output device switched.", "success", 1500);
        } catch (err) {
          console.error("Failed to switch output device:", err);
          showNotice("Failed to set audio output device: " + err.message, "error");
        }
      } else {
        showNotice("Custom output device selection not supported by your browser.", "warning", 2500);
      }
    }
  });

  async function connectVoice() {
    try {
      toggleVoiceConnectBtn.disabled = true;
      toggleVoiceConnectBtn.textContent = "Connecting...";

      // Get mic stream using selected device constraint if available
      const constraints = {
        audio: selectedInputDeviceId ? { deviceId: { exact: selectedInputDeviceId } } : true,
        video: false
      };
      voiceMicStream = await navigator.mediaDevices.getUserMedia(constraints);
      
      // Refresh the device list now that permission is granted and labels are visible
      await populateVoiceDevices();

      // Initialize AudioContext at 16000Hz (autoresampling!)
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      voiceAudioContext = new AudioCtx({ sampleRate: 16000 });

      // Apply selected output device if set
      if (selectedOutputDeviceId && typeof voiceAudioContext.setSinkId === 'function') {
        try {
          await voiceAudioContext.setSinkId(selectedOutputDeviceId);
        } catch (err) {
          console.warn("Failed to set output device sink on init:", err);
        }
      }

      if (voiceAudioContext.state === "suspended") {
        await voiceAudioContext.resume();
      }

      voiceMicSource = voiceAudioContext.createMediaStreamSource(voiceMicStream);
      voiceMicAnalyser = voiceAudioContext.createAnalyser();
      voiceMicAnalyser.fftSize = 256;

      // ScriptProcessorNode buffers mono audio
      voiceScriptProcessor = voiceAudioContext.createScriptProcessor(2048, 1, 1);
      
      voiceMicSource.connect(voiceMicAnalyser);
      voiceMicAnalyser.connect(voiceScriptProcessor);
      voiceScriptProcessor.connect(voiceAudioContext.destination);

      voiceScriptProcessor.onaudioprocess = (event) => {
        if (!isVoiceConnected || isVoiceMuted) return;
        if (voiceTriggerMode === 'push-to-talk' && !spacebarPressed) return;

        const inputBuffer = event.inputBuffer.getChannelData(0);
        // Convert Float32 to Int16
        const pcm = new Int16Array(inputBuffer.length);
        for (let i = 0; i < inputBuffer.length; i++) {
          const s = Math.max(-1, Math.min(1, inputBuffer[i]));
          pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        // Send binary frame over WebSocket
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(pcm.buffer);
        }
      };

      isVoiceConnected = true;
      voiceMuteCheckbox.disabled = false;
      voiceTriggerModeSelect.disabled = false;
      pttBtn.disabled = false;
      
      toggleVoiceConnectBtn.disabled = false;
      toggleVoiceConnectBtn.textContent = "Disconnect Mic";
      toggleVoiceConnectBtn.classList.remove("cta-solid");
      toggleVoiceConnectBtn.classList.add("cta-outline");
      voiceStatusLabel.textContent = "Microphone Active";

      // Send start message
      sendVoiceStart();
      setVoiceState("idle");

    } catch (err) {
      console.error("Failed to connect microphone:", err);
      showNotice("Failed to access microphone: " + err.message, "error");
      disconnectVoice();
    }
  }

  function disconnectVoice() {
    isVoiceConnected = false;
    
    // Stop recording tracks
    if (voiceMicStream) {
      voiceMicStream.getTracks().forEach(track => track.stop());
      voiceMicStream = null;
    }

    if (voiceScriptProcessor) {
      voiceScriptProcessor.disconnect();
      voiceScriptProcessor.onaudioprocess = null;
      voiceScriptProcessor = null;
    }

    if (voiceMicSource) {
      voiceMicSource.disconnect();
      voiceMicSource = null;
    }

    if (voiceAudioContext) {
      voiceAudioContext.close().catch(() => {});
      voiceAudioContext = null;
    }

    // Stop speaking
    interruptVoicePlayback();

    toggleVoiceConnectBtn.disabled = false;
    toggleVoiceConnectBtn.textContent = "Connect Microphone";
    toggleVoiceConnectBtn.classList.add("cta-solid");
    toggleVoiceConnectBtn.classList.remove("cta-outline");
    
    voiceMuteCheckbox.checked = false;
    voiceMuteCheckbox.disabled = true;
    isVoiceMuted = false;

    voiceStatusLabel.textContent = "Disconnected";
    setVoiceState("idle");

    // Send close message
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "voice_close" }));
    }
  }

  function sendVoiceStart() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: "voice_start",
        mode: voiceTriggerMode,
        sessionKey: currentSessionKey
      }));
    }
  }

  function setVoiceState(state) {
    voiceState = state;
    voiceStateDot.className = "voice-state-dot " + state;
    voiceStateLabel.textContent = state;
  }

  function appendVoiceTranscript(role, content) {
    // Remove placeholder
    const placeholder = voiceTranscript.querySelector(".transcript-placeholder");
    if (placeholder) placeholder.remove();

    const bubble = document.createElement("div");
    bubble.className = "transcript-bubble " + role;
    bubble.textContent = content;
    voiceTranscript.appendChild(bubble);
    voiceTranscript.scrollTop = voiceTranscript.scrollHeight;
  }

  // Playback queue logic
  async function handleVoiceAudioChunk(base64Wav, text) {
    if (!voiceAudioContext) return;

    try {
      // Decode base64 to binary ArrayBuffer
      const binary = atob(base64Wav);
      const len = binary.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binary.charCodeAt(i);
      }

      // decode audio
      const audioBuffer = await voiceAudioContext.decodeAudioData(bytes.buffer);
      voicePlaybackQueue.push({ buffer: audioBuffer, text });
      
      if (!voicePlayingAudio) {
        playNextVoiceChunk();
      }
    } catch (err) {
      console.error("Failed to decode voice audio:", err);
    }
  }

  function playNextVoiceChunk() {
    if (!isVoiceConnected || voicePlaybackQueue.length === 0) {
      voicePlayingAudio = false;
      voiceActiveAudioSource = null;
      if (voiceState === "speaking") {
        setVoiceState("idle");
      }
      return;
    }

    voicePlayingAudio = true;
    setVoiceState("speaking");

    const { buffer, text } = voicePlaybackQueue.shift();

    voiceActiveAudioSource = voiceAudioContext.createBufferSource();
    voiceActiveAudioSource.buffer = buffer;

    voiceSpeakerAnalyser = voiceAudioContext.createAnalyser();
    voiceSpeakerAnalyser.fftSize = 256;

    voiceActiveAudioSource.connect(voiceSpeakerAnalyser);
    voiceSpeakerAnalyser.connect(voiceAudioContext.destination);

    voiceActiveAudioSource.onended = () => {
      playNextVoiceChunk();
    };

    voiceActiveAudioSource.start(0);
  }

  function interruptVoicePlayback() {
    if (voiceActiveAudioSource) {
      try {
        voiceActiveAudioSource.stop();
      } catch (e) {}
      voiceActiveAudioSource = null;
    }
    voicePlaybackQueue = [];
    voicePlayingAudio = false;
  }

  // PTT Keyboard support
  window.addEventListener("keydown", (e) => {
    if (!isVoiceConnected || voiceTriggerMode !== 'push-to-talk') return;
    if (e.code === "Space") {
      // Don't trigger if user is typing in a text field
      const activeEl = document.activeElement;
      if (activeEl && (activeEl.tagName === "INPUT" || activeEl.tagName === "TEXTAREA" || activeEl.isContentEditable)) {
        return;
      }
      e.preventDefault();
      if (!spacebarPressed) {
        spacebarPressed = true;
        pttBtn.classList.add("active");
        pttBtn.textContent = "SPEAKING...";
        // Tell server voice started
        sendVoiceStart();
        setVoiceState("listening");
      }
    }
  });

  window.addEventListener("keyup", (e) => {
    if (!isVoiceConnected || voiceTriggerMode !== 'push-to-talk') return;
    if (e.code === "Space") {
      const activeEl = document.activeElement;
      if (activeEl && (activeEl.tagName === "INPUT" || activeEl.tagName === "TEXTAREA" || activeEl.isContentEditable)) {
        return;
      }
      e.preventDefault();
      if (spacebarPressed) {
        spacebarPressed = false;
        pttBtn.classList.remove("active");
        pttBtn.textContent = "HOLD TO TALK";
        // Tell server voice stopped
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "voice_stop" }));
        }
      }
    }
  });

  // PTT Mouse/Touch support
  pttBtn.addEventListener("mousedown", () => {
    if (!isVoiceConnected || voiceTriggerMode !== 'push-to-talk') return;
    spacebarPressed = true;
    pttBtn.textContent = "SPEAKING...";
    sendVoiceStart();
    setVoiceState("listening");
  });

  const stopPttMouse = () => {
    if (!isVoiceConnected || voiceTriggerMode !== 'push-to-talk' || !spacebarPressed) return;
    spacebarPressed = false;
    pttBtn.textContent = "HOLD TO TALK";
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "voice_stop" }));
    }
  };

  pttBtn.addEventListener("mouseup", stopPttMouse);
  pttBtn.addEventListener("mouseleave", stopPttMouse);
  pttBtn.addEventListener("touchstart", (e) => {
    e.preventDefault();
    if (!isVoiceConnected || voiceTriggerMode !== 'push-to-talk') return;
    spacebarPressed = true;
    pttBtn.textContent = "SPEAKING...";
    sendVoiceStart();
    setVoiceState("listening");
  });
  pttBtn.addEventListener("touchend", stopPttMouse);

  // Visualizer Animation
  function initVisualizerAnimation() {
    if (voiceAnimationFrameId) return;

    const ctx = voiceVisualizer.getContext("2d");
    const width = voiceVisualizer.width;
    const height = voiceVisualizer.height;

    function renderVisualizerFrame() {
      voiceAnimationFrameId = requestAnimationFrame(renderVisualizerFrame);
      ctx.clearRect(0, 0, width, height);

      if (voiceState === 'listening' && voiceMicAnalyser) {
        // Draw Microphone Oscilloscope Wave
        const bufferLength = voiceMicAnalyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        voiceMicAnalyser.getByteTimeDomainData(dataArray);

        ctx.lineWidth = 3;
        ctx.strokeStyle = "rgba(120, 209, 255, 0.85)"; // Cyan glow
        ctx.shadowBlur = 10;
        ctx.shadowColor = "rgba(120, 209, 255, 0.5)";

        ctx.beginPath();
        const sliceWidth = width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const v = dataArray[i] / 128.0;
          const y = (v * height) / 2;

          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
          x += sliceWidth;
        }

        ctx.lineTo(width, height / 2);
        ctx.stroke();
        ctx.shadowBlur = 0; // Reset

      } else if (voiceState === 'speaking' && voiceSpeakerAnalyser) {
        // Draw Speaker Oscilloscope Wave
        const bufferLength = voiceSpeakerAnalyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        voiceSpeakerAnalyser.getByteTimeDomainData(dataArray);

        ctx.lineWidth = 3;
        ctx.strokeStyle = "rgba(126, 255, 55, 0.85)"; // Green glow
        ctx.shadowBlur = 10;
        ctx.shadowColor = "rgba(126, 255, 55, 0.5)";

        ctx.beginPath();
        const sliceWidth = width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const v = dataArray[i] / 128.0;
          const y = (v * height) / 2;

          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
          x += sliceWidth;
        }

        ctx.lineTo(width, height / 2);
        ctx.stroke();
        ctx.shadowBlur = 0; // Reset

      } else if (voiceState === 'thinking') {
        // Draw swirl orb
        const centerX = width / 2;
        const centerY = height / 2;
        const time = Date.now() / 200;

        ctx.shadowBlur = 20;
        ctx.shadowColor = "rgba(181, 151, 255, 0.6)"; // Purple glow

        for (let j = 0; j < 3; j++) {
          ctx.beginPath();
          ctx.lineWidth = 2;
          ctx.strokeStyle = `rgba(181, 151, 255, ${0.4 + j * 0.2})`;

          const radius = 35 + j * 10 + Math.sin(time + j) * 4;
          
          for (let angle = 0; angle < Math.PI * 2; angle += 0.1) {
            const rOffset = Math.sin(angle * 5 + time + j) * 3;
            const x = centerX + (radius + rOffset) * Math.cos(angle);
            const y = centerY + (radius + rOffset) * Math.sin(angle);
            if (angle === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.closePath();
          ctx.stroke();
        }
        ctx.shadowBlur = 0;

      } else {
        // Idle
        ctx.lineWidth = 2;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();

        const centerX = width / 2;
        const centerY = height / 2;
        const breathing = 35 + Math.sin(Date.now() / 800) * 2;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        ctx.beginPath();
        ctx.arc(centerX, centerY, breathing, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    renderVisualizerFrame();
  }

  updateHeader();
  bindWelcomeChips();
  connect();

  setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "ping" }));
    }
  }, 30000);

  setInterval(fetchHealth, 30000);
})();
