"use client";

import { DragEvent, useEffect, useMemo, useRef, useState } from "react";

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
};

type ProviderForm = {
  version: number;
  profiles: ProviderProfileForm[];
  presets: ProviderPreset[];
  draft_profile_id: string;
  draft_model: string;
  draft_compute_mode: ComputeMode;
  reviewer_profile_id: string;
  reviewer_model: string;
  reviewer_compute_mode: ComputeMode;
  review_enabled: boolean;
  warnings: string[];
  draft_provider: string;
  reviewer_provider: string;
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
  format: "txt" | "markdown" | "epub";
  text: string;
  bytes?: ArrayBuffer;
  blockCount: number;
  chapterCount?: number;
};

type EpubInspection = {
  title: string;
  block_count: number;
  chapter_count: number;
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

async function epubApi<T>(path: string, data: ArrayBuffer): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/epub+zip" },
    body: data,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `EPUB 请求失败（${response.status}）`);
  }
  return payload as T;
}

function uniqueModels(items: string[]): string[] {
  return [...new Set(items.filter(Boolean))];
}

function ModelPicker({ value, options, disabled, onChange }: {
  value: string;
  options: string[];
  disabled?: boolean;
  onChange: (value: string) => void;
}) {
  const known = options.includes(value);
  const [manual, setManual] = useState(Boolean(value && !known));
  const showManual = manual || Boolean(value && !known);

  return <div className="model-picker">
    <select
      aria-label="选择模型"
      value={showManual ? "__manual__" : value}
      disabled={disabled}
      onChange={(event) => {
        if (event.target.value === "__manual__") {
          setManual(true);
          if (known) onChange("");
        } else {
          setManual(false);
          onChange(event.target.value);
        }
      }}
    >
      <option value="">选择模型</option>
      {options.map((model) => <option value={model} key={model}>{model}</option>)}
      <option value="__manual__">手动输入其他模型…</option>
    </select>
    {showManual && <input aria-label="自定义模型 ID" disabled={disabled} value={value} onChange={(event) => onChange(event.target.value)} placeholder="输入服务端模型 ID" />}
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

function ComputeModePicker({ value, disabled, probe, onChange }: {
  value: ComputeMode;
  disabled?: boolean;
  probe?: ModelProbeResult;
  onChange: (value: ComputeMode) => void;
}) {
  return <div className="model-picker">
    <select aria-label="选择模式" value={value} disabled={disabled} onChange={(event) => onChange(event.target.value as ComputeMode)}>
      {COMPUTE_OPTIONS.map((option) => <option value={option.value} key={option.value}>{option.label}{probe ? ` · ${probe.mode_mapping[option.value]}` : ""}</option>)}
    </select>
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

export function ProviderSettingsPanel({ onSaved }: { onSaved?: (value: ProviderForm) => void }) {
  const [form, setForm] = useState<ProviderForm>({
    version: 3,
    profiles: [],
    presets: [],
    draft_profile_id: "",
    draft_model: "",
    draft_compute_mode: "economy",
    reviewer_profile_id: "",
    reviewer_model: "",
    reviewer_compute_mode: "performance",
    review_enabled: false,
    warnings: [],
    draft_provider: "",
    reviewer_provider: "",
    provider_type: "custom",
    base_url: "",
    api_key_configured: false,
    key_source: "none",
  });
  const [activeProfileId, setActiveProfileId] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [status, setStatus] = useState<{ kind: "success" | "warning" | "error"; text: string } | null>(null);
  const [availableModels, setAvailableModels] = useState<Record<string, string[]>>({});
  const [modelProbes, setModelProbes] = useState<Record<string, ModelProbeState>>({});

  useEffect(() => {
    api<ProviderForm>("/settings/provider")
      .then((value) => {
        const profiles = value.profiles.map((item) => ({ ...item, api_key: "" }));
        const draftPreset = value.presets.find((item) => item.id === profiles.find((profile) => profile.id === value.draft_profile_id)?.provider_type);
        const reviewerPreset = value.presets.find((item) => item.id === profiles.find((profile) => profile.id === value.reviewer_profile_id)?.provider_type);
        setForm({
          ...value,
          profiles,
          draft_model: value.draft_model || draftPreset?.default_models[0] || "",
          draft_compute_mode: value.draft_compute_mode || "economy",
          reviewer_model: value.reviewer_model || reviewerPreset?.default_models[0] || "",
          reviewer_compute_mode: value.reviewer_compute_mode || "performance",
        });
        setActiveProfileId(value.draft_profile_id || profiles[0]?.id || "");
        if (value.warnings?.length) setStatus({ kind: "warning", text: value.warnings.join("；") });
      })
      .catch(() => setStatus({ kind: "error", text: "本地 API 尚未启动，请用启动器重新打开介译。" }))
      .finally(() => setLoading(false));
  }, []);

  const activeProfile = form.profiles.find((item) => item.id === activeProfileId) || form.profiles[0];

  function updateActiveProfile(update: Partial<ProviderProfileForm>) {
    if (!activeProfile) return;
    setForm((current) => ({
      ...current,
      profiles: current.profiles.map((item) => item.id === activeProfile.id ? { ...item, ...update } : item),
    }));
    setStatus(null);
  }

  function chooseProvider(preset: ProviderPreset) {
    if (!activeProfile) return;
    const recommended = preset.default_models[0] || "";
    setForm((current) => ({
      ...current,
      profiles: current.profiles.map((item) => item.id === activeProfile.id ? {
        ...item,
        name: preset.name,
        provider_type: preset.id,
        base_url: preset.base_url,
        chat_path: preset.chat_path,
        models_path: preset.models_path,
        protocol: preset.protocol,
        auth_required: preset.auth_required,
        capabilities: preset.capabilities,
      } : item),
      draft_model: current.draft_profile_id === activeProfile.id && recommended ? recommended : current.draft_model,
      reviewer_model: current.reviewer_profile_id === activeProfile.id && recommended ? recommended : current.reviewer_model,
    }));
    setStatus(recommended ? { kind: "success", text: `已选择 ${preset.name}，推荐模型为 ${recommended}。` } : null);
    if (preset.default_models[0]) {
      setAvailableModels((current) => ({ ...current, [activeProfile?.id || ""]: preset.default_models }));
    }
  }

  function addProfile() {
    const preset = form.presets.find((item) => item.id === "custom") || form.presets[0];
    if (!preset) return;
    const id = `connection-${Date.now().toString(36)}`;
    const profile: ProviderProfileForm = {
      id,
      name: "新连接",
      provider_type: preset.id,
      base_url: preset.base_url,
      chat_path: preset.chat_path,
      models_path: preset.models_path,
      protocol: preset.protocol,
      auth_required: preset.auth_required,
      capabilities: preset.capabilities,
      api_key: "",
      api_key_configured: false,
      key_source: "none",
    };
    setForm((current) => ({ ...current, profiles: [...current.profiles, profile] }));
    setActiveProfileId(id);
    setStatus(null);
  }

  function removeActiveProfile() {
    if (!activeProfile || form.profiles.length <= 1) return;
    const profiles = form.profiles.filter((item) => item.id !== activeProfile.id);
    const fallbackId = profiles[0].id;
    setForm((current) => ({
      ...current,
      profiles,
      draft_profile_id: current.draft_profile_id === activeProfile.id ? fallbackId : current.draft_profile_id,
      reviewer_profile_id: current.reviewer_profile_id === activeProfile.id ? fallbackId : current.reviewer_profile_id,
    }));
    setActiveProfileId(fallbackId);
    setStatus(null);
  }

  function modelsForProfile(profileId: string): string[] {
    const profile = form.profiles.find((item) => item.id === profileId);
    const defaults = form.presets.find((item) => item.id === profile?.provider_type)?.default_models || [];
    return uniqueModels([...(availableModels[profileId] || []), ...defaults]);
  }

  function changeBindingProfile(role: "draft" | "reviewer", profileId: string) {
    const recommended = modelsForProfile(profileId)[0] || "";
    setForm((current) => role === "draft"
      ? { ...current, draft_profile_id: profileId, draft_model: recommended }
      : { ...current, reviewer_profile_id: profileId, reviewer_model: recommended });
  }


  function modelProbeKey(profileId: string, model: string): string {
    const profile = form.profiles.find((item) => item.id === profileId);
    if (!profile || !model.trim()) return "";
    return [profile.id, profile.provider_type, profile.base_url, profile.chat_path, profile.protocol, model.trim()].join("|");
  }

  async function testModel(profileId: string, model: string) {
    const profile = form.profiles.find((item) => item.id === profileId);
    const key = modelProbeKey(profileId, model);
    if (!profile || !key) return;
    const normalizedModel = model.trim().toLowerCase();
    if (normalizedModel.startsWith("claude-") && profile.protocol !== "anthropic_messages") {
      setModelProbes((current) => ({ ...current, [key]: { loading: false, error: "Claude 模型需要选择 Anthropic Messages 连接；当前连接使用的是其他协议。" } }));
      return;
    }
    if (normalizedModel.startsWith("gemini-") && profile.protocol !== "gemini_generate_content" && profile.protocol !== "responses") {
      setModelProbes((current) => ({ ...current, [key]: { loading: false, error: "Gemini 模型需要选择 Gemini generateContent 或 Responses 连接。" } }));
      return;
    }
    setModelProbes((current) => ({ ...current, [key]: { loading: true } }));
    try {
      const result = await api<ModelProbeResult>("/settings/provider/model-test", {
        method: "POST",
        body: JSON.stringify({
          profile_id: profile.id,
          provider_type: profile.provider_type,
          base_url: profile.base_url,
          chat_path: profile.chat_path,
          protocol: profile.protocol,
          api_key: profile.api_key,
          model,
        }),
      });
      setModelProbes((current) => ({ ...current, [key]: { loading: false, result } }));
    } catch (error) {
      setModelProbes((current) => ({
        ...current,
        [key]: {
          loading: false,
          error: error instanceof Error ? error.message : "能力实测失败",
        },
      }));
    }
  }

  async function saveSettings() {
    setSaving(true);
    setStatus(null);
    try {
      const value = await api<ProviderForm>("/settings/provider", {
        method: "PATCH",
        body: JSON.stringify({
          version: 3,
          profiles: form.profiles.map((profile) => ({
            id: profile.id,
            name: profile.name,
            provider_type: profile.provider_type,
            base_url: profile.base_url,
            chat_path: profile.chat_path,
            models_path: profile.models_path,
            protocol: profile.protocol,
            auth_required: profile.auth_required,
            capabilities: profile.capabilities,
            api_key: profile.api_key,
          })),
          draft_profile_id: form.draft_profile_id,
          draft_model: form.draft_model,
          draft_compute_mode: form.draft_compute_mode,
          reviewer_profile_id: form.reviewer_profile_id,
          reviewer_model: form.reviewer_model,
          reviewer_compute_mode: form.reviewer_compute_mode,
          review_enabled: form.review_enabled,
        }),
      });
      const profiles = value.profiles.map((item) => ({ ...item, api_key: "" }));
      const normalized = { ...value, profiles };
      setForm(normalized);
      onSaved?.(value);
      setStatus(value.warnings?.length
        ? { kind: "warning", text: `配置已保存；${value.warnings.join("；")}` }
        : { kind: "success", text: "配置已保存，草译与审校将使用各自绑定的连接。" });
    } catch (error) {
      setStatus({ kind: "error", text: error instanceof Error ? error.message : "保存失败" });
    } finally {
      setSaving(false);
    }
  }

  async function testConnection() {
    if (!activeProfile) return;
    setTesting(true);
    setStatus(null);
    try {
      const boundModels = [
        form.draft_profile_id === activeProfile.id ? form.draft_model : "",
        form.review_enabled && form.reviewer_profile_id === activeProfile.id ? form.reviewer_model : "",
      ].filter(Boolean);
      const value = await api<{ models: string[]; stages: { message: string }[] }>("/settings/provider/test", {
        method: "POST",
        body: JSON.stringify({
          profile_id: activeProfile.id,
          provider_type: activeProfile.provider_type,
          base_url: activeProfile.base_url,
          models_path: activeProfile.models_path,
          protocol: activeProfile.protocol,
          api_key: activeProfile.api_key,
          required_models: boundModels,
        }),
      });
      const suffix = value.models.length ? `，发现 ${value.models.length} 个可用模型` : "";
      setAvailableModels((current) => ({ ...current, [activeProfile.id]: value.models }));
      if (value.models[0]) {
        setForm((current) => ({
          ...current,
          draft_model: current.draft_profile_id === activeProfile.id && !current.draft_model ? value.models[0] : current.draft_model,
          reviewer_model: current.reviewer_profile_id === activeProfile.id && !current.reviewer_model ? value.models[0] : current.reviewer_model,
        }));
      }
      const message = activeProfile.protocol === "anthropic_messages" && !value.models.length
        ? "Claude 地址已保存；该接口不提供模型列表，请手动输入模型 ID 后点击“实测能力”。"
        : `地址、认证和已绑定模型均正常${suffix}。`;
      setStatus({ kind: "success", text: message });
    } catch (error) {
      setStatus({ kind: "error", text: error instanceof Error ? error.message : "连接失败" });
    } finally {
      setTesting(false);
    }
  }

  const configuredCount = form.profiles.filter((item) => !item.auth_required || item.api_key_configured || item.api_key).length;
  const draftModels = modelsForProfile(form.draft_profile_id);
  const reviewerModels = modelsForProfile(form.reviewer_profile_id);

  const draftProbeState = modelProbes[modelProbeKey(form.draft_profile_id, form.draft_model)];
  const reviewerProbeState = modelProbes[modelProbeKey(form.reviewer_profile_id, form.reviewer_model)];
  const draftProbe = draftProbeState?.result;
  const reviewerProbe = reviewerProbeState?.result;

  return (
    <section className="setup-view settings-view">
      <header className="setup-header">
        <div><span className="page-kicker">偏好设置</span><h1>模型配置</h1><p>管理多个模型连接，并分别绑定草译与审校任务。</p></div>
        <div className={`connection-pill ${configuredCount > 0 ? "online" : ""}`}><i />{loading ? "正在读取" : `${configuredCount} / ${form.profiles.length} 个连接就绪`}</div>
      </header>

      <div className="settings-scroll">
        <section className="form-section">
          <div className="form-section-title"><span>1</span><div><strong>服务连接</strong><small>每个连接拥有独立地址、端点和密钥</small></div></div>
          <div className="profile-tabs">
            {form.profiles.map((profile) => <button key={profile.id} className={profile.id === activeProfile?.id ? "active" : ""} onClick={() => setActiveProfileId(profile.id)}><i>{profile.name.slice(0, 1)}</i><span><strong>{profile.name}</strong><small>{profile.api_key_configured ? "密钥已保存" : profile.auth_required ? "等待密钥" : "无需密钥"}</small></span></button>)}
            <button className="add-profile" onClick={addProfile}>＋ 添加连接</button>
          </div>
          <div className="provider-grid">
            {form.presets.map((provider) => (
              <button key={provider.id} className={`provider-option ${activeProfile?.provider_type === provider.id ? "selected" : ""}`} onClick={() => chooseProvider(provider)}>
                <i>{provider.name.slice(0, 1)}</i><span><strong>{provider.name}</strong><small>{provider.note}</small></span><b>✓</b>
              </button>
            ))}
          </div>
        </section>

        <section className="form-section">
          <div className="form-section-title"><span>2</span><div><strong>连接详情</strong><small>端点显式配置，不再猜测 API 版本</small></div></div>
          <div className="settings-fields">
            <label className="wide-field"><span>连接名称</span><input value={activeProfile?.name || ""} onChange={(event) => updateActiveProfile({ name: event.target.value })} placeholder="例如 Kimi 草译" /></label>
            <label className="wide-field"><span>API Base URL</span><input value={activeProfile?.base_url || ""} onChange={(event) => updateActiveProfile({ base_url: event.target.value })} placeholder="https://api.example.com/v1" /></label>
            <div className="endpoint-fields"><label><span>协议</span><select value={activeProfile?.protocol || "chat_completions"} onChange={(event) => updateActiveProfile({ protocol: event.target.value })}><option value="chat_completions">OpenAI Chat Completions</option><option value="responses">OpenAI Responses</option><option value="anthropic_messages">Anthropic Messages</option><option value="gemini_generate_content">Gemini generateContent</option></select></label><label><span>Chat / Messages 路径</span><input value={activeProfile?.chat_path || ""} onChange={(event) => updateActiveProfile({ chat_path: event.target.value })} /></label><label><span>Models 路径</span><input value={activeProfile?.models_path || ""} onChange={(event) => updateActiveProfile({ models_path: event.target.value })} /></label></div>
            <label className="wide-field"><span>API 密钥</span><div className="secret-input"><input type={showKey ? "text" : "password"} value={activeProfile?.api_key || ""} onChange={(event) => updateActiveProfile({ api_key: event.target.value })} placeholder={activeProfile?.api_key_configured ? "已安全存储；留空表示不修改" : activeProfile?.auth_required ? "输入该连接的 API Key" : "本地服务无需填写"} /><button type="button" onClick={() => setShowKey((value) => !value)}>{showKey ? "隐藏" : "显示"}</button></div></label>
          </div>
          <div className="keychain-note"><i>⌾</i><span><strong>分连接安全存储</strong>优先写入 macOS 钥匙串；授权不可用时自动降级为本次运行会话。</span></div>
          {form.profiles.length > 1 && <button className="remove-profile" onClick={removeActiveProfile}>移除此连接</button>}
        </section>

        <section className="form-section">
          <div className="form-section-title"><span>3</span><div><strong>任务模型</strong><small>草译和审校可以来自不同服务</small></div></div>
          <div className="settings-fields three-columns">
            <label><span>草译连接</span><select value={form.draft_profile_id} onChange={(event) => changeBindingProfile("draft", event.target.value)}>{form.profiles.map((profile) => <option value={profile.id} key={profile.id}>{profile.name}</option>)}</select></label>
            <div className="model-field"><span>草译模型</span><div className="model-control-row"><ModelPicker key={"draft-" + form.draft_profile_id} value={form.draft_model} options={draftModels} onChange={(model) => setForm((current) => ({ ...current, draft_model: model }))} /><button type="button" onClick={() => void testModel(form.draft_profile_id, form.draft_model)} disabled={!form.draft_model || draftProbeState?.loading}>{draftProbeState?.loading ? "实测中…" : draftProbeState?.result ? "重新实测" : "实测能力"}</button></div></div>
            <div className="model-field"><span>草译模式</span><ComputeModePicker value={form.draft_compute_mode} probe={draftProbe} onChange={(draft_compute_mode) => setForm((current) => ({ ...current, draft_compute_mode }))} /></div>
            {draftProbeState && <ModelCapabilityCard state={draftProbeState} />}
            <label><span>审校连接</span><select value={form.reviewer_profile_id} disabled={!form.review_enabled} onChange={(event) => changeBindingProfile("reviewer", event.target.value)}>{form.profiles.map((profile) => <option value={profile.id} key={profile.id}>{profile.name}</option>)}</select></label>
            <div className="model-field"><span>审校模型</span><div className="model-control-row"><ModelPicker key={"reviewer-" + form.reviewer_profile_id} value={form.reviewer_model} options={reviewerModels} disabled={!form.review_enabled} onChange={(model) => setForm((current) => ({ ...current, reviewer_model: model }))} /><button type="button" onClick={() => void testModel(form.reviewer_profile_id, form.reviewer_model)} disabled={!form.review_enabled || !form.reviewer_model || reviewerProbeState?.loading}>{reviewerProbeState?.loading ? "实测中…" : reviewerProbeState?.result ? "重新实测" : "实测能力"}</button></div></div>
            <div className="model-field"><span>审校模式</span><ComputeModePicker value={form.reviewer_compute_mode} probe={reviewerProbe} disabled={!form.review_enabled} onChange={(reviewer_compute_mode) => setForm((current) => ({ ...current, reviewer_compute_mode }))} /></div>
            {form.review_enabled && reviewerProbeState && <ModelCapabilityCard state={reviewerProbeState} />}
          </div>
          <label className="switch-row" htmlFor="review-enabled"><span><strong>启用第二模型审校</strong><small>草译后自动调用审校模型检查问题段落</small></span><input id="review-enabled" aria-label="启用第二模型审校" type="checkbox" checked={form.review_enabled} onChange={(event) => setForm((current) => ({ ...current, review_enabled: event.target.checked }))} /><i /></label>
        </section>
      </div>

      <footer className="setup-footer">
        <div>{status && <span className={`inline-status ${status.kind}`}><i>{status.kind === "success" ? "✓" : "!"}</i>{status.text}</span>}</div>
        <div><button className="secondary-action" onClick={testConnection} disabled={testing || !activeProfile?.base_url}>{testing ? "正在测试…" : "测试当前连接"}</button><button className="blue-action" onClick={saveSettings} disabled={saving || form.profiles.some((item) => !item.base_url)}>{saving ? "正在保存…" : "保存全部配置"}</button></div>
      </footer>
    </section>
  );
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
  const [styleGuide, setStyleGuide] = useState(STYLE_PRESETS[0].guide);
  const [file, setFile] = useState<ImportFile | null>(null);
  const [dragging, setDragging] = useState(false);
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
    if (!selected) return;
    setError("");
    setResult(null);
    const extension = selected.name.split(".").pop()?.toLowerCase() || "";
    if (!["txt", "md", "markdown", "epub"].includes(extension)) {
      setError("当前支持 EPUB、TXT、MD 和 Markdown 文件。");
      return;
    }
    const sizeLimit = extension === "epub" ? 128 : 25;
    if (selected.size > sizeLimit * 1024 * 1024) {
      setError(`文件大于 ${sizeLimit} MB，请先压缩或拆分后再导入。`);
      return;
    }
    if (extension === "epub") {
      try {
        const bytes = await selected.arrayBuffer();
        const inspection = await epubApi<EpubInspection>("/imports/epub/inspect", bytes);
        const previewText = inspection.preview.map((item) => item.text).join("\n\n");
        setFile({
          name: selected.name,
          size: selected.size,
          format: "epub",
          text: previewText,
          bytes,
          blockCount: inspection.block_count,
          chapterCount: inspection.chapter_count,
        });
        const bookTitle = inspection.title === "Untitled EPUB" ? selected.name.replace(/\.epub$/i, "") : inspection.title;
        setTitle(bookTitle);
        setProjectName((current) => current || bookTitle);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "EPUB 解析失败");
      }
      return;
    }
    const text = await selected.text();
    if (!text.trim()) {
      setError("文件内容为空，无法导入。");
      return;
    }
    const cleanTitle = selected.name.replace(/\.(txt|md|markdown)$/i, "");
    const textBlocks = text.split(/\n\s*\n/).map((item) => item.trim()).filter(Boolean);
    setFile({ name: selected.name, size: selected.size, text, format: extension === "txt" ? "txt" : "markdown", blockCount: textBlocks.length });
    setTitle((current) => current || cleanTitle);
    setProjectName((current) => current || cleanTitle);
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
      } else {
        await api<Project>(`/projects/${selectedProjectId}/style`, {
          method: "PATCH",
          body: JSON.stringify({ style_guide: styleGuide }),
        });
      }
      const document = file.format === "epub" && file.bytes
        ? await epubApi<{ id: string }>(`/projects/${selectedProjectId}/documents/epub?title=${encodeURIComponent(title)}`, file.bytes)
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
        <div><span className="page-kicker">新建项目</span><h1>导入书籍</h1><p>导入原文，介译会保留标题结构并生成稳定段落。</p></div>
        <div className="import-steps"><span className={file ? "done" : "active"}><i>{file ? "✓" : "1"}</i>选择文件</span><b /><span className={file ? "active" : ""}><i>2</i>确认信息</span><b /><span className={result ? "done" : ""}><i>{result ? "✓" : "3"}</i>完成</span></div>
      </header>

      <div className="import-scroll">
        {!result ? <>
          <div role="button" tabIndex={0} aria-label="选择或拖入书稿文件" className={`drop-zone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") inputRef.current?.click(); }} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={handleDrop} onClick={() => inputRef.current?.click()}>
            <input ref={inputRef} type="file" accept=".epub,.txt,.md,.markdown,application/epub+zip,text/plain,text/markdown" hidden onChange={(event) => void acceptFile(event.target.files?.[0])} />
            {file ? <><div className={`file-icon ${file.format === "epub" ? "epub" : ""}`}>{file.format === "txt" ? "TXT" : file.format === "epub" ? "EPUB" : "MD"}</div><div><strong>{file.name}</strong><span>{(file.size / 1024 / (file.size > 1024 * 1024 ? 1024 : 1)).toFixed(1)} {file.size > 1024 * 1024 ? "MB" : "KB"} · {file.blockCount} 个结构块{file.chapterCount ? ` · ${file.chapterCount} 个章节文件` : " · UTF-8"}</span></div><button type="button" onClick={(event) => { event.stopPropagation(); setFile(null); }}>更换文件</button></> : <><div className="upload-mark">⇧</div><div><strong>拖入 EPUB 或文本书稿</strong><span>支持 EPUB、TXT、Markdown · EPUB 最大 128 MB</span></div></>}
          </div>

          {error && <div className="import-error"><i>!</i>{error}</div>}

          <div className="import-form-grid">
            <section className="import-card">
              <div className="card-title"><span>项目与语言</span><div className="segmented"><button className={projectMode === "new" ? "active" : ""} onClick={() => setProjectMode("new")}>新项目</button><button className={projectMode === "existing" ? "active" : ""} onClick={() => { const project = projects.find((item) => item.id === projectId) || projects[0]; setProjectMode("existing"); if (project) { setProjectId(project.id); setSourceLang(project.source_lang); setTargetLang(project.target_lang); setStyleGuide(project.style_guide || STYLE_PRESETS[0].guide); setStylePreset(STYLE_PRESETS.find((item) => item.guide === project.style_guide)?.id || "custom"); } }} disabled={!projects.length}>已有项目</button></div></div>
              {projectMode === "new" ? <label><span>项目名称</span><input value={projectName} onChange={(event) => setProjectName(event.target.value)} placeholder="例如：社会理论选读" /></label> : <label><span>选择项目</span><select value={projectId} onChange={(event) => { const nextId = event.target.value; const project = projects.find((item) => item.id === nextId); setProjectId(nextId); if (project) { setSourceLang(project.source_lang); setTargetLang(project.target_lang); setStyleGuide(project.style_guide || STYLE_PRESETS[0].guide); setStylePreset(STYLE_PRESETS.find((item) => item.guide === project.style_guide)?.id || "custom"); } }}>{projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}</select></label>}
              <div className="two-fields"><label><span>原文语言</span><select value={sourceLang} onChange={(event) => setSourceLang(event.target.value)}><option value="en">英语</option><option value="fr">法语</option><option value="de">德语</option><option value="ja">日语</option><option value="zh-CN">简体中文</option></select></label><label><span>目标语言</span><select value={targetLang} onChange={(event) => setTargetLang(event.target.value)}><option value="zh-CN">简体中文</option><option value="zh-TW">繁体中文</option><option value="en">英语</option><option value="ja">日语</option></select></label></div>
              <label><span>书名 / 文档标题</span><input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="输入书名" /></label>
              <label><span>翻译风格</span><div className="style-preset-grid">{STYLE_PRESETS.map((preset) => <button type="button" key={preset.id} className={stylePreset === preset.id ? "selected" : ""} onClick={() => { setStylePreset(preset.id); setStyleGuide(preset.guide); }}><strong>{preset.label}</strong><small>{preset.note}</small></button>)}</div><textarea aria-label="翻译风格详细要求" value={styleGuide} onChange={(event) => { setStyleGuide(event.target.value); setStylePreset("custom"); }} placeholder="补充语气、术语、标点或读者对象等要求" /></label>
            </section>

            <section className="import-card preview-card">
              <div className="card-title"><span>分段预览</span>{file && <small>{file.format === "epub" ? "按 EPUB 书脊顺序" : `显示前 ${Math.min(4, paragraphs.length)} 段`}</small>}</div>
              {file ? <div className="paragraph-preview">{paragraphs.slice(0, 4).map((paragraph, index) => <div key={`${index}-${paragraph.slice(0, 8)}`}><b>{String(index + 1).padStart(2, "0")}</b><p>{paragraph}</p></div>)}</div> : <div className="empty-preview"><i>¶</i><span>选择书稿后，这里会显示标题与段落识别结果。</span></div>}
            </section>
          </div>
        </> : <div className="import-success"><i>✓</i><span className="page-kicker">导入完成</span><h2>《{result.title}》已经准备好</h2><p>书稿已写入项目数据库并完成稳定分段，可以继续配置模型并创建翻译任务。</p><div><button className="secondary-action" onClick={() => { setResult(null); setFile(null); setTitle(""); }}>继续导入</button><button className="blue-action" onClick={() => onImported?.(result)}>打开这本书</button></div></div>}
      </div>

      {!result && <footer className="setup-footer"><div><span className="privacy-note">书稿只保存在本机项目数据库中</span></div><button className="blue-action import-action" disabled={!file || !title.trim() || importing || (projectMode === "new" ? !projectName.trim() : !projectId)} onClick={importBook}>{importing ? "正在解析并导入…" : "导入并创建项目"}</button></footer>}
    </section>
  );
}
