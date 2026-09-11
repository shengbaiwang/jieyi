"use client";

import { DragEvent, useEffect, useId, useMemo, useRef, useState } from "react";
import { endpointPreview, modelSearch, selectedModels, uniqueModels, validBaseUrl } from "./model-config";

const API_BASE = process.env.NEXT_PUBLIC_JIEYI_API || "http://127.0.0.1:8000";

type ComputeMode = "economy" | "balanced" | "performance";

type ProviderPreset = {
  id: string;
  name: string;
  note: string;
  base_url: string;
  chat_path: string;
  models_path: string;
  protocol: string;
  auth_required: boolean;
  capabilities: string[];
  default_models: string[];
};

type ProviderProfileForm = {
  id: string;
  name: string;
  provider_type: string;
  base_url: string;
  chat_path: string;
  models_path: string;
  protocol: string;
  auth_required: boolean;
  capabilities: string[];
  api_key: string;
  api_key_configured: boolean;
  key_source: string;
  selected_models?: string[] | null;
};

type ProviderForm = {
  version: number;
  profiles: ProviderProfileForm[];
  presets: ProviderPreset[];
  draft_profile_id: string;
  draft_model: string;
  draft_compute_mode: ComputeMode;
  term_discovery_profile_id: string;
  term_discovery_model: string;
  term_discovery_compute_mode: ComputeMode;
  term_discovery_provider: string;
  warnings: string[];
  draft_provider: string;
  // Flattened compatibility fields consumed by the workbench shell.
  provider_type: string;
  base_url: string;
  api_key_configured: boolean;
  key_source: string;
};


type ModelProbeResult = {
  ok: boolean;
  model: string;
  baseline: { visible_output: boolean; latency_ms: number };
  reasoning: {
    kind: "effort" | "thinking" | "default";
    verification: "verified" | "unverified";
    supported_efforts: string[];
    accepted_efforts: string[];
    empty_efforts: string[];
    thinking_states: string[];
  };
  mode_mapping: Record<ComputeMode, string>;
  requests: number;
  total_tokens: number;
  duration_ms: number;
  notes: string[];
};

type ModelProbeState = {
  loading: boolean;
  result?: ModelProbeResult;
  error?: string;
};

type Project = {
  id: string;
  name: string;
  source_lang: string;
  target_lang: string;
  style_guide: string;
};

type ImportedBook = {
  projectId: string;
  documentId: string;
  title: string;
};

type ImportFile = {
  name: string;
  size: number;
  format: "txt" | "markdown" | "epub" | "pdf";
  text: string;
  bytes?: ArrayBuffer;
  blockCount: number;
  chapterCount?: number;
  pageCount?: number;
  warnings?: string[];
};

type EpubInspection = {
  title: string;
  block_count: number;
  chapter_count: number;
  page_count?: number;
  warnings?: string[];
  preview: { kind: string; text: string; heading_path: string }[];
};

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `请求失败（${response.status}）`);
  }
  return payload as T;
}

async function epubApi<T>(path: string, data: ArrayBuffer, format = "epub", signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": format === "pdf" ? "application/pdf" : "application/epub+zip" },
    body: data, signal,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `文件请求失败（${response.status}）`);
  }
  return payload as T;
}

function ModelSelect({ value, options, loading, disabled, onChange }: {
  value: string; options: string[]; loading?: boolean; disabled?: boolean; onChange: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [highlighted, setHighlighted] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const search = useRef<HTMLInputElement>(null);
  const listId = useId();
  const typed = query.trim();
  const matches = modelSearch(uniqueModels([value, ...options]), typed);
  const choices = [...matches, ...(typed && !matches.includes(typed) ? [typed] : [])];
  const index = Math.min(highlighted, Math.max(choices.length - 1, 0));

  useEffect(() => {
    if (!open) return;
    search.current?.focus();
    const closeOutside = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    const closeOnFocus = (event: FocusEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("focusin", closeOnFocus);
    return () => { document.removeEventListener("pointerdown", closeOutside); document.removeEventListener("focusin", closeOnFocus); };
  }, [open]);

  useEffect(() => {
    if (open) document.getElementById(`${listId}-${index}`)?.scrollIntoView({ block: "nearest" });
  }, [open, listId, index]);

  function commit(model: string) {
    onChange(model.trim()); setQuery(""); setOpen(false); trigger.current?.focus();
  }

  return <div className="model-select" ref={root}>
    <button type="button" ref={trigger} aria-label="选择模型" aria-haspopup="listbox" aria-expanded={open} aria-controls={listId}
      className={value ? "" : "empty"} disabled={disabled}
      onClick={() => { setQuery(""); setHighlighted(0); setOpen((current) => !current); }}
      onKeyDown={(event) => { if (event.key === "ArrowDown" || event.key === "ArrowUp") { event.preventDefault(); setHighlighted(0); setOpen(true); } }}
    ><span>{value || "选择或输入模型"}</span><i /></button>
    {open && <div className="model-select-pop">
      <input ref={search} aria-label="搜索或输入模型 ID" role="combobox" aria-autocomplete="list" aria-expanded="true" aria-controls={listId}
        aria-activedescendant={choices.length ? `${listId}-${index}` : undefined} value={query} placeholder="搜索已添加模型，或输入完整 ID"
        onChange={(event) => { setQuery(event.target.value); setHighlighted(0); }}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault(); setHighlighted(choices.length ? (index + (event.key === "ArrowDown" ? 1 : -1) + choices.length) % choices.length : 0);
          }
          if (event.key === "Enter" && choices[index]) { event.preventDefault(); commit(choices[index]); }
          if (event.key === "Escape") { event.stopPropagation(); setOpen(false); trigger.current?.focus(); }
        }} />
      <div className="model-select-list" id={listId} role="listbox" aria-label="模型">
        {choices.map((model, position) => <button type="button" tabIndex={-1} role="option" aria-selected={model === value} id={`${listId}-${position}`} key={model}
          className={`${model === value ? "active" : ""} ${position === index ? "highlighted" : ""} ${!matches.includes(model) ? "manual" : ""}`}
          onPointerDown={(event) => event.preventDefault()} onClick={() => commit(model)}>{matches.includes(model) ? model : `添加并使用“${model}”`}{model === value && <b>✓</b>}</button>)}
      </div>
      {!choices.length && <p className="field-hint">{loading ? "正在读取模型…" : "尚未添加模型，可到服务中管理，或输入完整模型 ID。"}</p>}
    </div>}
  </div>;
}

const COMPUTE_OPTIONS: { value: ComputeMode; label: string }[] = [
  { value: "economy", label: "节省" },
  { value: "balanced", label: "均衡" },
  { value: "performance", label: "性能" },
];

const STYLE_PRESETS = [
  { id: "academic", label: "学术严谨", note: "概念稳定，论证清楚", guide: "忠实原意，保持严谨、克制的学术表达；统一核心概念译法，保留引文、脚注、专名及论证层次。" },
  { id: "literary", label: "文学自然", note: "保留声调与节奏", guide: "准确传达原意与人物声调，译文自然流畅并保留文学节奏、意象与修辞；避免生硬直译。" },
  { id: "popular", label: "通俗易读", note: "清晰、顺畅、少术语", guide: "在不损失关键信息的前提下使用清晰、自然、易读的现代语言；必要术语首次出现时给出简短说明。" },
  { id: "faithful", label: "忠实直译", note: "贴近句法与措辞", guide: "尽量贴近原文句法、措辞和段落结构，不擅自增删或改写；歧义处保留原文的开放性。" },
] as const;

function ComputeModeSelect({ value, disabled, probe, onChange }: {
  value: ComputeMode;
  disabled?: boolean;
  probe?: ModelProbeResult;
  onChange: (value: ComputeMode) => void;
}) {
  return <div className="mode-segment" role="group" aria-label="选择模式">
    {COMPUTE_OPTIONS.map((option) => <button
      type="button"
      key={option.value}
      className={option.value === value ? "active" : ""}
      disabled={disabled}
      aria-pressed={option.value === value}
      onClick={() => onChange(option.value)}
    ><strong>{option.label}</strong>{probe && <small>{probe.mode_mapping[option.value]}</small>}</button>)}
  </div>;
}

function ModelCapabilityCard({ state }: { state: ModelProbeState }) {
  if (state.loading) {
    return <div className="model-capability-card loading"><i>⋯</i><span><strong>正在实测模型能力</strong><small>会发出少量极短请求，验证思考参数而不是只读取模型名称。</small></span></div>;
  }
  if (state.error) {
    return <div className="model-capability-card error"><i>!</i><span><strong>能力实测失败</strong><small>{state.error}</small></span></div>;
  }
  const result = state.result;
  if (!result) return null;
  const control = result.reasoning.kind === "effort"
    ? `可调强度：${result.reasoning.supported_efforts.join(" · ")}`
    : result.reasoning.kind === "thinking"
      ? "思考开关：可关闭 / 可开启"
      : "思考控制：无法验证，使用服务端默认";
  return <div className={`model-capability-card ${result.baseline.visible_output ? "verified" : "warning"}`}>
    <div className="capability-card-head"><span><i>{result.baseline.visible_output ? "✓" : "!"}</i><strong>{result.model}</strong></span><b>{result.reasoning.verification === "verified" ? "能力已实测" : "参数待确认"}</b></div>
    <div className="capability-facts">
      <span><small>短译输出</small><strong>{result.baseline.visible_output ? "正常" : "无可见文本"}</strong></span>
      <span><small>思考方式</small><strong>{control}</strong></span>
      <span><small>三档映射</small><strong>节省 {result.mode_mapping.economy} · 均衡 {result.mode_mapping.balanced} · 性能 {result.mode_mapping.performance}</strong></span>
      <span><small>测试开销</small><strong>{result.requests} 次请求 · {result.total_tokens || "未返回"} token · {(result.duration_ms / 1000).toFixed(1)} 秒</strong></span>
    </div>
    {result.notes.length > 0 && <p>{result.notes.join(" ")}</p>}
  </div>;
}

function probeSummary(result: ModelProbeResult): string {
  const control = result.reasoning.kind === "effort"
    ? `强度可调（${result.reasoning.supported_efforts.join("/")}）`
    : result.reasoning.kind === "thinking"
      ? "思考可开关"
      : "使用服务端默认思考";
  return `${result.baseline.visible_output ? "短译正常" : "短译无可见文本"} · ${control} · ${result.requests} 次请求 · ${(result.duration_ms / 1000).toFixed(1)} 秒`;
}

function bindingStatus(model: string, probe?: ModelProbeState): { tone: string; text: string } {
  if (!model.trim()) return { tone: "idle", text: "未选择模型" };
  if (probe?.loading) return { tone: "busy", text: "正在实测" };
  if (probe?.error) return { tone: "error", text: "实测失败" };
  if (probe?.result) {
    return probe.result.baseline.visible_output
      ? { tone: "ready", text: probe.result.reasoning.verification === "verified" ? "能力已实测" : "参数待确认" }
      : { tone: "warn", text: "无可见输出" };
  }
  return { tone: "idle", text: "未实测" };
}

function TaskBindingCard({ label, note, profiles, profileId, model, models, modelsLoading, mode, probe, mirror, onProfile, onModel, onMode, onTest, onRefresh }: {
  label: string;
  note: string;
  profiles: ProviderProfileForm[];
  profileId: string;
  model: string;
  models: string[];
  modelsLoading: boolean;
  mode: ComputeMode;
  probe?: ModelProbeState;
  mirror?: { label: string; disabled: boolean; onClick: () => void };
  onProfile: (value: string) => void;
  onModel: (value: string) => void;
  onMode: (value: ComputeMode) => void;
  onTest: () => void;
  onRefresh: () => void;
}) {
  const [showDetails, setShowDetails] = useState(false);
  const result = probe?.result;
  const status = bindingStatus(model, probe);

  return <section className={`binding-card ${status.tone}`}>
    <header className="binding-head">
      <div><strong>{label}</strong><small>{note}</small></div>
      <div className="binding-head-actions">
        {mirror && <button type="button" className="ghost-action" disabled={mirror.disabled} onClick={mirror.onClick}>{mirror.label}</button>}
        <span className={`binding-status ${status.tone}`}><i />{status.text}</span>
      </div>
    </header>
    <div className="binding-grid">
      <label><span>连接</span><select value={profileId} onChange={(event) => onProfile(event.target.value)}>{profiles.map((profile) => <option value={profile.id} key={profile.id}>{profile.name}</option>)}</select></label>
      <div className="binding-field"><span>模型</span><ModelSelect value={model} options={models} loading={modelsLoading} onChange={onModel} /></div>
      <div className="binding-field"><span>计算模式{result ? "（已按实测映射）" : ""}</span><ComputeModeSelect value={mode} probe={result} onChange={onMode} /></div>
    </div>
    <div className="binding-actions">
      {label === "术语发现" && model && <button type="button" className="ghost-action" onClick={() => onModel("")}>仅本地扫描</button>}
      <button type="button" className="probe-action" onClick={onTest} disabled={!model.trim() || probe?.loading}>{probe?.loading ? "实测中…" : result ? "重新实测" : "实测能力"}</button>
      <button type="button" className="ghost-action" onClick={onRefresh} disabled={modelsLoading}>管理此服务的模型</button>
      {result && <button type="button" className="ghost-action" onClick={() => setShowDetails((current) => !current)}>{showDetails ? "收起详情" : "查看详情"}</button>}
      {result && !showDetails && <span className="binding-summary">{probeSummary(result)}</span>}
    </div>
    {probe && (probe.loading || probe.error || (result && showDetails)) && <ModelCapabilityCard state={probe} />}
  </section>;
}

export function ProviderSettingsPanel({ onSaved, hidden = false }: { onSaved?: (value: ProviderForm) => void; hidden?: boolean }) {
  const [form, setForm] = useState<ProviderForm>({
    version: 4, profiles: [], presets: [], draft_profile_id: "", draft_model: "",
    draft_compute_mode: "economy", term_discovery_profile_id: "", term_discovery_model: "",
    term_discovery_compute_mode: "balanced", term_discovery_provider: "", warnings: [],
    draft_provider: "", provider_type: "custom", base_url: "", api_key_configured: false, key_source: "none",
  });
  const [savedForm, setSavedForm] = useState<ProviderForm | null>(null);
  const [activeProfileId, setActiveProfileId] = useState("");
  const [tab, setTab] = useState<"providers" | "tasks">("providers");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [adding, setAdding] = useState(false);
  const [providerQuery, setProviderQuery] = useState("");
  const [modelQuery, setModelQuery] = useState("");
  const [manualModel, setManualModel] = useState("");
  const [managing, setManaging] = useState(false);
  const [status, setStatus] = useState<{ kind: "success" | "warning" | "error"; text: string } | null>(null);
  const [availableModels, setAvailableModels] = useState<Record<string, string[]>>({});
  const [modelLoading, setModelLoading] = useState<Record<string, boolean>>({});
  const [connectionStatus, setConnectionStatus] = useState<Record<string, { kind: "success" | "warning" | "error"; text: string }>>({});
  const [modelProbes, setModelProbes] = useState<Record<string, ModelProbeState>>({});
  // Revisions keep responses from earlier credentials/endpoints out of the current UI.
  const revisions = useRef<Record<string, number>>({});
  const listingRequests = useRef(new Set<string>());

  useEffect(() => {
    let cancelled = false;
    api<ProviderForm>("/settings/provider").then((value) => {
      if (cancelled) return;
      const normalized = {
        ...value,
        profiles: value.profiles.map((item) => ({ ...item, api_key: "", selected_models: selectedModels(item.selected_models, [
          value.draft_profile_id === item.id ? value.draft_model : "",
          value.term_discovery_profile_id === item.id ? value.term_discovery_model : "",
        ]) })),
      };
      setForm(normalized);
      setSavedForm(normalized);
      setActiveProfileId(value.draft_profile_id || value.profiles[0]?.id || "");
      if (value.warnings?.length) setStatus({ kind: "warning", text: value.warnings.join("；") });
    }).catch(() => {
      if (!cancelled) setStatus({ kind: "error", text: "无法读取配置，请检查本地 API 后重新打开介译。" });
    }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  const dirty = savedForm !== null && JSON.stringify(form) !== JSON.stringify(savedForm);
  useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  const activeProfile = form.profiles.find((item) => item.id === activeProfileId) || form.profiles[0];
  const boundModels = (profileId: string) => uniqueModels([
    form.draft_profile_id === profileId ? form.draft_model : "",
    form.term_discovery_profile_id === profileId ? form.term_discovery_model : "",
  ]);
  const modelsForProfile = (profileId: string) => selectedModels(form.profiles.find((item) => item.id === profileId)?.selected_models, boundModels(profileId));

  function selectProfile(id: string) {
    setActiveProfileId(id); setShowKey(false); setModelQuery(""); setManualModel(""); setManaging(false); setAdding(false);
  }

  function invalidateProfile(id: string) {
    revisions.current[id] = (revisions.current[id] || 0) + 1;
    setAvailableModels((current) => { const next = { ...current }; delete next[id]; return next; });
    setConnectionStatus((current) => { const next = { ...current }; delete next[id]; return next; });
    setModelLoading((current) => ({ ...current, [id]: false }));
  }

  function updateActiveProfile(update: Partial<ProviderProfileForm>) {
    if (!activeProfile) return;
    if (Object.keys(update).some((key) => !["name", "selected_models"].includes(key))) invalidateProfile(activeProfile.id);
    setForm((current) => ({ ...current, profiles: current.profiles.map((item) => item.id === activeProfile.id ? { ...item, ...update } : item) }));
    setStatus(null);
  }

  function addProfile(preset: ProviderPreset) {
    const id = `connection-${crypto.randomUUID()}`;
    const number = form.profiles.filter((item) => item.provider_type === preset.id).length + 1;
    const profile: ProviderProfileForm = {
      id, name: `${preset.name}${number > 1 ? ` ${number}` : ""}`, provider_type: preset.id,
      base_url: preset.base_url, chat_path: preset.chat_path, models_path: preset.models_path,
      protocol: preset.protocol, auth_required: preset.auth_required, capabilities: preset.capabilities,
      api_key: "", api_key_configured: false, key_source: "none", selected_models: [],
    };
    setForm((current) => ({ ...current, profiles: [...current.profiles, profile],
      draft_profile_id: current.draft_profile_id || id, term_discovery_profile_id: current.term_discovery_profile_id || id }));
    selectProfile(id); setProviderQuery(""); setStatus(null);
  }

  function removeActiveProfile() {
    if (!activeProfile || form.profiles.length <= 1 || form.draft_profile_id === activeProfile.id || form.term_discovery_profile_id === activeProfile.id) return;
    const profiles = form.profiles.filter((item) => item.id !== activeProfile.id);
    invalidateProfile(activeProfile.id);
    setForm((current) => ({ ...current, profiles }));
    selectProfile(profiles[0].id);
    setStatus({ kind: "warning", text: "已从待保存配置中移除连接；保存前可以撤销更改。使用此连接的书籍需要重新选择服务。" });
  }

  function addModels(models: string[]) {
    if (!activeProfile) return;
    updateActiveProfile({ selected_models: uniqueModels([...modelsForProfile(activeProfile.id), ...models]) });
  }

  function changeBindingModel(role: "draft" | "term_discovery", model: string, profileId = form[`${role}_profile_id`]) {
    setForm((current) => ({ ...current, [`${role}_profile_id`]: profileId, [`${role}_model`]: model.trim(),
      profiles: current.profiles.map((profile) => profile.id === profileId
        ? { ...profile, selected_models: uniqueModels([...(profile.selected_models || []), model]) } : profile),
    }));
    setStatus(null);
  }

  async function loadModels(profileId: string) {
    const profile = form.profiles.find((item) => item.id === profileId);
    if (!profile || !validBaseUrl(profile.base_url)) return;
    const revision = revisions.current[profileId] || 0;
    const requestId = `${profileId}:${revision}`;
    if (listingRequests.current.has(requestId)) return;
    listingRequests.current.add(requestId);
    setModelLoading((current) => ({ ...current, [profileId]: true }));
    try {
      const value = await api<{ models: string[] }>("/settings/provider/test", {
        method: "POST", body: JSON.stringify({ profile_id: profile.id, provider_type: profile.provider_type,
          base_url: profile.base_url, models_path: profile.models_path, protocol: profile.protocol,
          api_key: profile.api_key, required_models: [] }),
      });
      if ((revisions.current[profileId] || 0) !== revision) return;
      setAvailableModels((current) => ({ ...current, [profileId]: uniqueModels(value.models) }));
      setConnectionStatus((current) => ({ ...current, [profileId]: value.models.length
        ? { kind: "success", text: `已读取 ${uniqueModels(value.models).length} 个模型。添加所需模型后，可实测验证输出。` }
        : { kind: "warning", text: "未获取到模型列表。可添加预设或手动输入模型 ID，再实测验证；当前尚未验证模型可用性。" } }));
    } catch (error) {
      if ((revisions.current[profileId] || 0) !== revision) return;
      setConnectionStatus((current) => ({ ...current, [profileId]: { kind: "error", text: error instanceof Error ? error.message : "获取模型失败，可重试或手动添加模型。" } }));
    } finally {
      listingRequests.current.delete(requestId);
      if ((revisions.current[profileId] || 0) === revision) setModelLoading((current) => ({ ...current, [profileId]: false }));
    }
  }

  function modelProbeKey(profileId: string, model: string): string {
    return model.trim() ? `${profileId}:${revisions.current[profileId] || 0}:${model.trim()}` : "";
  }

  async function testModel(profileId: string, model: string) {
    const profile = form.profiles.find((item) => item.id === profileId);
    const key = modelProbeKey(profileId, model);
    if (!profile || !key || modelProbes[key]?.loading) return;
    setModelProbes((current) => ({ ...current, [key]: { loading: true } }));
    try {
      const result = await api<ModelProbeResult>("/settings/provider/model-test", {
        method: "POST", body: JSON.stringify({ profile_id: profile.id, provider_type: profile.provider_type,
          base_url: profile.base_url, chat_path: profile.chat_path, protocol: profile.protocol, api_key: profile.api_key, model: model.trim() }),
      });
      setModelProbes((current) => ({ ...current, [key]: { loading: false, result } }));
    } catch (error) {
      setModelProbes((current) => ({ ...current, [key]: { loading: false, error: error instanceof Error ? error.message : "能力实测失败" } }));
    }
  }

  async function saveSettings() {
    if (saving || !dirty) return;
    setSaving(true); setStatus(null);
    try {
      const value = await api<ProviderForm>("/settings/provider", {
        method: "PATCH", body: JSON.stringify({ version: 4,
          profiles: form.profiles.map((profile) => ({ ...profile, selected_models: modelsForProfile(profile.id) })),
          draft_profile_id: form.draft_profile_id, draft_model: form.draft_model, draft_compute_mode: form.draft_compute_mode,
          term_discovery_profile_id: form.term_discovery_profile_id, term_discovery_model: form.term_discovery_model,
          term_discovery_compute_mode: form.term_discovery_compute_mode,
        }),
      });
      const normalized = { ...value, profiles: value.profiles.map((item) => ({ ...item, api_key: "" })) };
      setForm(normalized); setSavedForm(normalized); onSaved?.(value);
      setStatus(value.warnings?.length ? { kind: "warning", text: `已保存；${value.warnings.join("；")}` }
        : { kind: "success", text: "连接、已添加模型和默认任务设置已保存。" });
    } catch (error) {
      setStatus({ kind: "error", text: error instanceof Error ? error.message : "保存失败，请重试。" });
    } finally { setSaving(false); }
  }

  function discardChanges() {
    if (!savedForm) return;
    for (const profile of form.profiles) invalidateProfile(profile.id);
    setForm(savedForm); selectProfile(savedForm.profiles[0]?.id || ""); setStatus(null);
  }

  const activeModels = activeProfile ? modelsForProfile(activeProfile.id) : [];
  const protectedModels = activeProfile ? boundModels(activeProfile.id) : [];
  const preset = form.presets.find((item) => item.id === activeProfile?.provider_type);
  const discovered = activeProfile ? availableModels[activeProfile.id] : undefined;
  const candidates = modelSearch(discovered ?? preset?.default_models ?? [], modelQuery);
  const missingCandidates = candidates.filter((model) => !activeModels.includes(model));
  const visibleProfiles = form.profiles.filter((item) => `${item.name} ${item.provider_type}`.toLowerCase().includes(providerQuery.trim().toLowerCase()));
  const activeStatus = activeProfile ? connectionStatus[activeProfile.id] : undefined;
  const activeLoading = Boolean(activeProfile && modelLoading[activeProfile.id]);
  const assigned = activeProfile && (form.draft_profile_id === activeProfile.id || form.term_discovery_profile_id === activeProfile.id);
  const invalidProfile = form.profiles.find((item) => !item.name.trim() || !validBaseUrl(item.base_url));
  const sameAsDraft = form.term_discovery_profile_id === form.draft_profile_id && form.term_discovery_model === form.draft_model && form.term_discovery_compute_mode === form.draft_compute_mode;

  return <section className="setup-view settings-view" style={hidden ? { display: "none" } : undefined}>
    <header className="setup-header">
      <div><span className="page-kicker">偏好设置</span><h1>模型配置</h1><p>连接模型服务，添加常用模型，再为翻译任务分配模型。</p></div>
      <span className={`settings-save-state ${dirty ? "dirty" : ""}`}>{loading ? "正在读取…" : dirty ? "● 有未保存的更改" : savedForm ? "配置已保存" : "配置未加载"}</span>
    </header>
    <div className="settings-tabs" role="tablist" aria-label="模型配置分类">
      <button role="tab" id="providers-tab" aria-controls="providers-panel" aria-selected={tab === "providers"} onClick={() => setTab("providers")}>模型服务 <span>{form.profiles.length}</span></button>
      <button role="tab" id="tasks-tab" aria-controls="tasks-panel" aria-selected={tab === "tasks"} onClick={() => setTab("tasks")}>默认任务模型</button>
    </div>
    <fieldset className="settings-workspace" disabled={loading || saving}>
      {tab === "providers" ? <div className="provider-workspace" id="providers-panel" role="tabpanel" aria-labelledby="providers-tab">
        <aside className="provider-sidebar" aria-label="已配置的模型服务">
          <input aria-label="搜索服务" placeholder="搜索服务…" value={providerQuery} onChange={(event) => setProviderQuery(event.target.value)} />
          <div className="provider-list">
            {visibleProfiles.map((profile) => <button key={profile.id} className={activeProfile?.id === profile.id && !adding ? "active" : ""} aria-pressed={activeProfile?.id === profile.id && !adding} onClick={() => selectProfile(profile.id)}>
              <i>{profile.name.slice(0, 1) || "M"}</i><span><strong>{profile.name || "未命名服务"}</strong><small>{modelsForProfile(profile.id).length} 个模型 · {!validBaseUrl(profile.base_url) ? "待填地址" : profile.auth_required && !profile.api_key_configured && !profile.api_key ? "待填密钥" : "已填连接"}</small></span>
            </button>)}
            {!loading && !visibleProfiles.length && <p className="settings-empty">{form.profiles.length ? "没有匹配的服务" : "添加一个模型服务开始配置"}</p>}
          </div>
          <button className="secondary-action add-provider" onClick={() => { setAdding(true); setShowKey(false); }}>＋ 添加服务</button>
        </aside>
        <div className="provider-detail">
          {adding ? <section className="provider-add-panel">
            <header className="provider-detail-heading"><div><h2>添加模型服务</h2><p>选择服务商，自动填入地址与协议。每次添加都会创建独立连接。</p></div><button className="ghost-action" onClick={() => setAdding(false)}>取消</button></header>
            <div className="provider-grid">{form.presets.map((item) => <button key={item.id} className="provider-option" onClick={() => addProfile(item)}><i>{item.name.slice(0, 1)}</i><span><strong>{item.name}</strong><small>{item.note}</small></span></button>)}</div>
          </section> : activeProfile ? <>
            <header className="provider-detail-heading"><div><h2>{activeProfile.name || "未命名服务"}</h2><p>{preset?.name || activeProfile.provider_type} · 独立连接</p></div><button className="ghost-action" onClick={() => setTab("tasks")}>分配任务 →</button></header>
            <section className="provider-connection settings-fields">
              <label><span>服务名称</span><input value={activeProfile.name} onChange={(event) => updateActiveProfile({ name: event.target.value })} placeholder="例如：日常翻译" /></label>
              <label><span>API 密钥 <small>{activeProfile.api_key ? "待保存" : activeProfile.api_key_configured ? "已存储" : activeProfile.auth_required ? "必填" : "可选"}</small></span><div className="secret-input"><input type={showKey ? "text" : "password"} autoComplete="off" spellCheck={false} value={activeProfile.api_key} onChange={(event) => updateActiveProfile({ api_key: event.target.value })} placeholder={activeProfile.api_key_configured ? "已安全存储；留空保留现有密钥" : "输入此服务的 API Key"} /><button type="button" aria-label={showKey ? "隐藏 API 密钥" : "显示 API 密钥"} onClick={() => setShowKey((value) => !value)}>{showKey ? "隐藏" : "显示"}</button></div></label>
              <label><span>API 地址</span><input value={activeProfile.base_url} onChange={(event) => updateActiveProfile({ base_url: event.target.value })} placeholder="https://api.example.com/v1" aria-invalid={Boolean(activeProfile.base_url && !validBaseUrl(activeProfile.base_url))} /><small className="field-hint">{activeProfile.base_url && !validBaseUrl(activeProfile.base_url) ? "请输入完整的 http:// 或 https:// 地址。" : "填写基础地址；如果服务商给了完整请求地址，请在高级设置中填写完整请求路径。"}</small></label>
              <details className="provider-advanced" key={activeProfile.id}><summary>高级设置 <span>协议与请求路径</span></summary><div className="endpoint-fields">
                <label><span>协议</span><select value={activeProfile.protocol} onChange={(event) => { const protocol = event.target.value; updateActiveProfile({ protocol, chat_path: protocol === "responses" ? "responses" : protocol === "anthropic_messages" ? "messages" : protocol === "gemini_generate_content" ? "models/{model}:generateContent" : "chat/completions" }); }}><option value="chat_completions">OpenAI Chat Completions</option><option value="responses">OpenAI Responses</option><option value="anthropic_messages">Anthropic Messages</option><option value="gemini_generate_content">Gemini generateContent</option></select></label>
                <label><span>请求路径</span><input value={activeProfile.chat_path} onChange={(event) => updateActiveProfile({ chat_path: event.target.value })} /></label>
                <label><span>模型列表路径</span><input value={activeProfile.models_path} onChange={(event) => updateActiveProfile({ models_path: event.target.value })} /></label>
              </div><p className="endpoint-preview">请求地址 <code>{endpointPreview(activeProfile.base_url, activeProfile.chat_path)}</code></p></details>
            </section>
            <section className="provider-model-section">
              <header className="provider-model-heading"><div><h3>已添加模型 <span>{activeModels.length}</span></h3><p>只有已添加的模型会进入任务选择列表。</p></div><div><button className="secondary-action" disabled={activeLoading || !validBaseUrl(activeProfile.base_url)} onClick={() => { setManaging(true); void loadModels(activeProfile.id); }}>{activeLoading ? "正在获取…" : "获取模型"}</button><button className="ghost-action" aria-expanded={managing} onClick={() => setManaging((value) => !value)}>{managing ? "收起管理" : "管理模型"}</button></div></header>
              {activeStatus && <p className={`inline-status ${activeStatus.kind}`} role="status">{activeStatus.text}</p>}
              {managing && <div className="model-manager">
                <div className="model-manager-toolbar"><input aria-label="搜索可添加模型" placeholder="搜索模型名称…" value={modelQuery} onChange={(event) => setModelQuery(event.target.value)} /><button className="ghost-action" disabled={!missingCandidates.length} onClick={() => addModels(missingCandidates)}>添加筛选结果{missingCandidates.length ? `（${missingCandidates.length}）` : ""}</button></div>
                <p className="field-hint">{discovered ? "来自当前服务的模型列表；是否能生成译文，请以实测为准。" : "服务商预设，仅供参考；获取模型可查看此连接实际返回的列表。"}</p>
                <div className="model-candidates">{candidates.map((model) => <div key={model}><code>{model}</code><button className="ghost-action" aria-label={`添加模型 ${model}`} disabled={activeModels.includes(model)} onClick={() => addModels([model])}>{activeModels.includes(model) ? "已添加" : "＋ 添加"}</button></div>)}{!candidates.length && <p className="settings-empty">{activeLoading ? "正在获取模型…" : "没有匹配的模型，可在下方手动添加。"}</p>}</div>
              </div>}
              <form className="manual-model-form" onSubmit={(event) => { event.preventDefault(); if (manualModel.trim()) { addModels([manualModel]); setManualModel(""); } }}><input aria-label="手动添加模型 ID" placeholder="输入完整模型 ID，回车添加" value={manualModel} onChange={(event) => setManualModel(event.target.value)} /><button className="ghost-action" disabled={!manualModel.trim() || activeModels.includes(manualModel.trim())}>＋ 添加</button></form>
              <div className="added-model-list">{activeModels.map((model) => {
                const probe = modelProbes[modelProbeKey(activeProfile.id, model)];
                const state = bindingStatus(model, probe);
                return <div className="added-model" key={model}><div className="added-model-row"><div><code>{model}</code><span className={`binding-status ${state.tone}`}>{state.text}</span>{form.draft_profile_id === activeProfile.id && form.draft_model === model && <small className="model-role">草译</small>}{form.term_discovery_profile_id === activeProfile.id && form.term_discovery_model === model && <small className="model-role">术语</small>}</div><div><button className="ghost-action" disabled={probe?.loading} onClick={() => void testModel(activeProfile.id, model)}>实测</button><button className="ghost-action" title={protectedModels.includes(model) ? "请先在默认任务模型中更换此模型" : "从已添加列表中移除"} aria-label={`移除模型 ${model}`} disabled={protectedModels.includes(model)} onClick={() => updateActiveProfile({ selected_models: activeModels.filter((item) => item !== model) })}>移除</button></div></div>{probe && <ModelCapabilityCard state={probe} />}</div>;
              })}{!activeModels.length && <p className="settings-empty">还没有添加模型。获取模型列表，或输入服务商提供的模型 ID。</p>}</div>
              <p className="field-hint">实测会发送少量短请求并使用 API 额度。任务正在使用的模型需先更换，再移除。</p>
            </section>
            <div className="provider-detail-bottom"><span>密钥按连接存储，优先使用 macOS 钥匙串。</span><button className="remove-profile" disabled={form.profiles.length <= 1 || Boolean(assigned)} title={assigned ? "请先在默认任务模型中切换到其他连接" : "保存后移除此服务连接"} onClick={removeActiveProfile}>移除服务</button></div>
          </> : <p className="settings-empty">{loading ? "正在读取模型配置…" : "请添加模型服务。"}</p>}
        </div>
      </div> : <div className="settings-scroll task-settings-panel" id="tasks-panel" role="tabpanel" aria-labelledby="tasks-tab">
        <div className="provider-detail-heading"><div><h2>默认任务模型</h2><p>草译用于新书的默认设置；已单独配置的书籍继续使用各自的模型。</p></div></div>
        <div className="binding-cards">{(["draft", "term_discovery"] as const).map((role) => <TaskBindingCard key={role}
          label={role === "draft" ? "草译" : "术语发现"} note={role === "draft" ? "逐段生成译文草稿" : "留空时仅做本地术语扫描"}
          profiles={form.profiles} profileId={form[`${role}_profile_id`]} model={form[`${role}_model`]}
          models={modelsForProfile(form[`${role}_profile_id`])} modelsLoading={false} mode={form[`${role}_compute_mode`]}
          probe={modelProbes[modelProbeKey(form[`${role}_profile_id`], form[`${role}_model`])]}
          onProfile={(id) => changeBindingModel(role, modelsForProfile(id)[0] || "", id)}
          onModel={(model) => changeBindingModel(role, model)}
          onMode={(mode) => setForm((current) => ({ ...current, [`${role}_compute_mode`]: mode }))}
          onTest={() => void testModel(form[`${role}_profile_id`], form[`${role}_model`])}
          onRefresh={() => { selectProfile(form[`${role}_profile_id`]); setTab("providers"); setManaging(true); }}
          mirror={role === "term_discovery" ? { label: "沿用草译设置", disabled: sameAsDraft, onClick: () => setForm((current) => ({ ...current, term_discovery_profile_id: current.draft_profile_id, term_discovery_model: current.draft_model, term_discovery_compute_mode: current.draft_compute_mode })) } : undefined}
        />)}</div>
      </div>}
    </fieldset>
    <footer className="setup-footer"><div aria-live="polite">{status ? <span className={`inline-status ${status.kind}`}><i>{status.kind === "success" ? "✓" : "!"}</i>{status.text}</span> : <span className="field-hint">{invalidProfile ? `请补全“${invalidProfile.name || "未命名服务"}”的名称和有效 API 地址。` : dirty ? "更改会在保存后生效；切换页面会保留当前编辑。" : "模型服务与任务设置统一保存。"}</span>}</div><div><button className="secondary-action" disabled={!dirty || saving} onClick={discardChanges}>撤销更改</button><button className="blue-action" disabled={!dirty || loading || saving || Boolean(invalidProfile) || !form.profiles.length} onClick={() => void saveSettings()}>{saving ? "正在保存…" : "保存配置"}</button></div></footer>
  </section>;
}

export function ImportBookPanel({ onImported }: { onImported?: (result: ImportedBook) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectMode, setProjectMode] = useState<"new" | "existing">("new");
  const [projectId, setProjectId] = useState("");
  const [projectName, setProjectName] = useState("");
  const [title, setTitle] = useState("");
  const [sourceLang, setSourceLang] = useState("en");
  const [targetLang, setTargetLang] = useState("zh-CN");
  const [stylePreset, setStylePreset] = useState("academic");
  const [styleGuide, setStyleGuide] = useState<string>(STYLE_PRESETS[0].guide);
  const [file, setFile] = useState<ImportFile | null>(null);
  const [dragging, setDragging] = useState(false);
  const [inspecting, setInspecting] = useState(false);
  const [inspectionProgress, setInspectionProgress] = useState({ done: 0, total: 0 });
  const inspectionRequest = useRef<AbortController | null>(null);
  useEffect(() => () => inspectionRequest.current?.abort(), []);

  const [importing, setImporting] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<ImportedBook | null>(null);

  useEffect(() => {
    api<Project[]>("/projects").then((items) => {
      setProjects(items);
      if (items[0]) setProjectId(items[0].id);
    }).catch(() => undefined);
  }, []);

  const paragraphs = useMemo(() => file?.text.split(/\n\s*\n/).map((item) => item.trim()).filter(Boolean) || [], [file]);

  async function acceptFile(selected: File | undefined) {
    if (!selected || importing) return;
    inspectionRequest.current?.abort();
    const controller = new AbortController();
    inspectionRequest.current = controller;
    setFile(null);
    setError("");
    setResult(null);
    setInspecting(false);
    setInspectionProgress({ done: 0, total: 0 });
    const extension = selected.name.split(".").pop()?.toLowerCase() || "";
    if (!["txt", "md", "markdown", "epub", "pdf"].includes(extension)) {
      setError("当前支持 PDF、EPUB、TXT、MD 和 Markdown 文件。");
      return;
    }
    const sizeLimit = ["epub", "pdf"].includes(extension) ? 128 : 25;
    if (selected.size > sizeLimit * 1024 * 1024) {
      setError(`文件大于 ${sizeLimit} MB，请先拆分后再导入。`);
      return;
    }
    setInspecting(true);
    try {
      if (extension === "epub" || extension === "pdf") {
        const bytes = await selected.arrayBuffer();
        if (controller.signal.aborted) return;
        type InspectionJob = EpubInspection & { id: string; status: string; completed_pages: number; total_pages: number; detail?: string };
        let inspection = await epubApi<InspectionJob>(`/imports/${extension}/inspect`, bytes, extension, controller.signal);
        while (extension === "pdf" && inspection.status === "processing") {
          setInspectionProgress({ done: inspection.completed_pages, total: inspection.total_pages });
          await new Promise((resolve) => setTimeout(resolve, 500));
          if (controller.signal.aborted) return;
          inspection = await api<InspectionJob>(`/imports/pdf/inspect/${inspection.id}`, { signal: controller.signal });
        }
        if (controller.signal.aborted) return;
        if (inspection.status === "failed") throw new Error(inspection.detail || "PDF 解析失败");
        setFile({ name: selected.name, size: selected.size, format: extension,
          text: inspection.preview.map((item) => item.text).join("\n\n"), bytes,
          blockCount: inspection.block_count, chapterCount: inspection.chapter_count,
          pageCount: inspection.page_count, warnings: inspection.warnings });
        const bookTitle = /^Untitled (EPUB|PDF)$/.test(inspection.title)
          ? selected.name.replace(/\.(epub|pdf)$/i, "") : inspection.title;
        setTitle(bookTitle);
        setProjectName((current) => current || bookTitle);
      } else {
        const text = await selected.text();
        if (controller.signal.aborted) return;
        if (!text.trim()) throw new Error("文件内容为空，无法导入。");
        const cleanTitle = selected.name.replace(/\.(txt|md|markdown)$/i, "");
        const blocks = text.split(/\n\s*\n/).map((item) => item.trim()).filter(Boolean);
        setFile({ name: selected.name, size: selected.size, text,
          format: extension === "txt" ? "txt" : "markdown", blockCount: blocks.length });
        setTitle(cleanTitle);
        setProjectName((current) => current || cleanTitle);
      }
    } catch (caught) {
      if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : "文件解析失败");
    } finally {
      if (!controller.signal.aborted) setInspecting(false);
    }
  }

  function cancelInspection() {
    inspectionRequest.current?.abort();
    setInspecting(false);
    setFile(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    void acceptFile(event.dataTransfer.files[0]);
  }

  async function importBook() {
    if (!file || !title.trim()) return;
    if (projectMode === "new" && !projectName.trim()) return;
    if (projectMode === "existing" && !projectId) return;
    setImporting(true);
    setError("");
    try {
      let selectedProjectId = projectId;
      if (projectMode === "new") {
        const project = await api<Project>("/projects", {
          method: "POST",
          body: JSON.stringify({ name: projectName, source_lang: sourceLang, target_lang: targetLang, style_guide: styleGuide }),
        });
        selectedProjectId = project.id;
        setProjects((items) => [project, ...items]);
        setProjectId(project.id);
        setProjectMode("existing");
      } else {
        await api<Project>(`/projects/${selectedProjectId}/style`, {
          method: "PATCH",
          body: JSON.stringify({ style_guide: styleGuide }),
        });
        await api<Project>(`/projects/${selectedProjectId}/languages`, {
          method: "PATCH",
          body: JSON.stringify({ source_lang: sourceLang, target_lang: targetLang }),
        });
      }
      const document = (file.format === "epub" || file.format === "pdf") && file.bytes
        ? await epubApi<{ id: string }>(`/projects/${selectedProjectId}/documents/${file.format}?title=${encodeURIComponent(title)}`, file.bytes, file.format)
        : await api<{ id: string }>(`/projects/${selectedProjectId}/documents`, {
            method: "POST",
            body: JSON.stringify({ title, text: file.text, source_format: file.format }),
          });
      setResult({ projectId: selectedProjectId, documentId: document.id, title });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "导入失败");
    } finally {
      setImporting(false);
    }
  }

  return (
    <section className="setup-view import-view">
      <header className="setup-header">
        <div><span className="page-kicker">新建项目</span><h1>导入书籍</h1></div>
      </header>

      <div className="import-scroll">
        {!result ? <>
          <div role="button" tabIndex={0} aria-label="选择或拖入书稿文件" className={`drop-zone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") inputRef.current?.click(); }} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={handleDrop} onClick={() => inputRef.current?.click()}>
            <input ref={inputRef} type="file" disabled={importing} accept=".pdf,.epub,.txt,.md,.markdown,application/pdf,application/epub+zip,text/plain,text/markdown" hidden onChange={(event) => { void acceptFile(event.target.files?.[0]); event.target.value = ""; }} />
            {inspecting ? <div className="pdf-inspection" role="status" aria-live="polite"><strong>正在整理书稿…</strong><span>{inspectionProgress.total ? `已解析 ${inspectionProgress.done} / ${inspectionProgress.total} 页` : "正在读取文件与目录"}</span><progress max={inspectionProgress.total || 1} value={inspectionProgress.total ? inspectionProgress.done : undefined} /><button type="button" onClick={(event) => { event.stopPropagation(); cancelInspection(); }}>取消预览</button></div> : file ? <><div className={`file-icon ${file.format === "epub" ? "epub" : ""}`}>{file.format === "txt" ? "TXT" : file.format === "epub" ? "EPUB" : file.format === "pdf" ? "PDF" : "MD"}</div><div><strong>{file.name}</strong><span>{(file.size / 1024 / (file.size > 1024 * 1024 ? 1024 : 1)).toFixed(1)} {file.size > 1024 * 1024 ? "MB" : "KB"} · {file.blockCount} 个结构块{file.pageCount ? ` · ${file.pageCount} 页` : ""}{file.chapterCount ? ` · ${file.chapterCount} 个章节` : file.format === "pdf" ? " · 按页定位" : ""}</span></div><button type="button" onClick={(event) => { event.stopPropagation(); if (!importing) { cancelInspection(); inputRef.current?.click(); } }}>更换文件</button></> : <><div className="upload-mark">⇧</div><div><strong>拖入 PDF、EPUB 或文本书稿</strong><span>支持 PDF、EPUB、TXT、Markdown · PDF / EPUB 最大 128 MB</span></div></>}
          </div>

          {file?.warnings?.map((warning) => <div className="pdf-import-note" key={warning}>{warning}</div>)}
          {error && <div role="alert" className="import-error"><i>!</i>{error}</div>}

          <div className="import-form-grid">
            <section className="import-card">
              <div className="card-title"><span>项目与语言</span><div className="segmented"><button className={projectMode === "new" ? "active" : ""} onClick={() => setProjectMode("new")}>新项目</button><button className={projectMode === "existing" ? "active" : ""} onClick={() => { const project = projects.find((item) => item.id === projectId) || projects[0]; setProjectMode("existing"); if (project) { setProjectId(project.id); setSourceLang(project.source_lang); setTargetLang(project.target_lang); setStyleGuide(project.style_guide || STYLE_PRESETS[0].guide); setStylePreset(STYLE_PRESETS.find((item) => item.guide === project.style_guide)?.id || "custom"); } }} disabled={!projects.length}>已有项目</button></div></div>
              {projectMode === "new" ? <label><span>项目名称</span><input value={projectName} onChange={(event) => setProjectName(event.target.value)} placeholder="例如：社会理论选读" /></label> : <label><span>选择项目</span><select value={projectId} onChange={(event) => { const nextId = event.target.value; const project = projects.find((item) => item.id === nextId); setProjectId(nextId); if (project) { setSourceLang(project.source_lang); setTargetLang(project.target_lang); setStyleGuide(project.style_guide || STYLE_PRESETS[0].guide); setStylePreset(STYLE_PRESETS.find((item) => item.guide === project.style_guide)?.id || "custom"); } }}>{projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}</select></label>}
              <div className="two-fields"><label><span>原文语言</span><select value={sourceLang} onChange={(event) => setSourceLang(event.target.value)}><option value="en">英语</option><option value="fr">法语</option><option value="de">德语</option><option value="ja">日语</option><option value="zh-CN">简体中文</option></select></label><label><span>目标语言</span><select value={targetLang} onChange={(event) => setTargetLang(event.target.value)}><option value="zh-CN">简体中文</option><option value="zh-TW">繁体中文</option><option value="en">英语</option><option value="ja">日语</option></select></label></div>
              <label><span>书名 / 文档标题</span><input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="输入书名" /></label>
              <label><span>翻译风格</span><div className="style-preset-grid">{STYLE_PRESETS.map((preset) => <button type="button" key={preset.id} className={stylePreset === preset.id ? "selected" : ""} onClick={() => { setStylePreset(preset.id); setStyleGuide(preset.guide); }}><strong>{preset.label}</strong><small>{preset.note}</small></button>)}</div><textarea aria-label="翻译风格详细要求" value={styleGuide} onChange={(event) => { setStyleGuide(event.target.value); setStylePreset("custom"); }} placeholder="补充语气、术语、标点或读者对象等要求" /></label>
            </section>

            <section className="import-card preview-card">
              <div className="card-title"><span>分段预览</span>{file && <small>{file.format === "pdf" ? "按原页顺序 · 自动整理断行" : file.format === "epub" ? "按 EPUB 书脊顺序" : `显示前 ${Math.min(4, paragraphs.length)} 段`}</small>}</div>
              {file ? <div className="paragraph-preview">{paragraphs.slice(0, 4).map((paragraph, index) => <div key={`${index}-${paragraph.slice(0, 8)}`}><b>{String(index + 1).padStart(2, "0")}</b><p>{paragraph}</p></div>)}</div> : <div className="empty-preview"><i>¶</i><span>选择书稿后，这里会显示标题与段落识别结果。</span></div>}
            </section>
          </div>
        </> : <div className="import-success"><i>✓</i><span className="page-kicker">导入完成</span><h2>《{result.title}》已经准备好</h2><p>书稿已写入项目数据库并完成稳定分段，可以继续配置模型并创建翻译任务。</p><div><button className="secondary-action" onClick={() => { setResult(null); setFile(null); setTitle(""); }}>继续导入</button><button className="blue-action" onClick={() => onImported?.(result)}>打开这本书</button></div></div>}
      </div>

      {!result && <footer className="setup-footer"><div><span className="privacy-note">书稿只保存在本机项目数据库中</span></div><button className="blue-action import-action" disabled={!file || !title.trim() || importing || inspecting || (projectMode === "new" ? !projectName.trim() : !projectId)} onClick={importBook}>{importing ? "正在解析并导入…" : projectMode === "new" ? "导入并创建项目" : "导入到项目"}</button></footer>}
    </section>
  );
}
