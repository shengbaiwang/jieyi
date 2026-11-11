"use client";

import { useId, useRef, useState, type KeyboardEvent } from "react";
import { STYLE_PRESETS } from "./translation-styles";
import styles from "./text-translation.module.css";

type ComputeMode = "economy" | "balanced" | "performance";
type Settings = {
  draft_profile_id: string; draft_model: string; draft_compute_mode: ComputeMode;
  profiles: Array<{ id: string; name: string; provider_type: string; selected_models?: string[] | null }>;
  presets: Array<{ id: string; default_models: string[] }>;
};
type Result = { text: string; model: string; prompt_tokens: number; completion_tokens: number; reasoning_tokens: number };
const LANGUAGES = [
  ["zh-CN", "简体中文"], ["zh-TW", "繁体中文"], ["en", "英语"], ["ja", "日语"],
  ["fr", "法语"], ["de", "德语"], ["es", "西班牙语"], ["ko", "韩语"],
  ["ru", "俄语"], ["it", "意大利语"], ["pt", "葡萄牙语"], ["ar", "阿拉伯语"],
];
const MAX_CHARS = 12_000;

export function TextTranslationPanel({ active, apiBase, settings, onConfigure }: {
  active: boolean; apiBase: string; settings: Settings | null; onConfigure: () => void;
}) {
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [sourceLang, setSourceLang] = useState("auto");
  const [targetLang, setTargetLang] = useState("zh-CN");
  const [styleGuide, setStyleGuide] = useState<string>(STYLE_PRESETS[0].guide);
  const [selection, setSelection] = useState<{ profileId: string; model: string } | null>(null);
  const [computeMode, setComputeMode] = useState<ComputeMode | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [translatedInput, setTranslatedInput] = useState("");
  const [cleared, setCleared] = useState<{ source: string; target: string; result: Result | null; input: string } | null>(null);
  const controller = useRef<AbortController | null>(null);
  const sourceInput = useRef<HTMLTextAreaElement>(null);
  const modelListId = useId();
  const customStyleId = useId();
  const profileId = selection?.profileId ?? settings?.draft_profile_id ?? "";
  const model = selection?.model ?? settings?.draft_model ?? "";
  const mode = computeMode ?? settings?.draft_compute_mode ?? "economy";
  const profile = settings?.profiles.find((item) => item.id === profileId);
  const configured = Boolean(profile && model.trim());
  const models = [...new Set([
    ...(profile?.selected_models ?? settings?.presets.find((item) => item.id === profile?.provider_type)?.default_models ?? []),
    ...(settings?.draft_profile_id === profileId ? [settings.draft_model] : []),
  ].filter(Boolean))];
  const fingerprint = JSON.stringify([source, sourceLang, targetLang, styleGuide, profileId, model, mode]);
  const stale = Boolean(target && translatedInput && translatedInput !== fingerprint);
  const canTranslate = configured && source.trim().length > 0 && source.length <= MAX_CHARS;

  async function translate() {
    if (!canTranslate || controller.current) return;
    const requestController = new AbortController();
    controller.current = requestController;
    setBusy(true); setError(""); setMessage("");
    try {
      const response = await fetch(`${apiBase}/translate/text`, {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: requestController.signal,
        body: JSON.stringify({ text: source, source_lang: sourceLang, target_lang: targetLang, style_guide: styleGuide, profile_id: profileId, model: model.trim(), compute_mode: mode }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : `翻译请求失败（${response.status}）`);
      if (typeof payload.text !== "string" || !payload.text.trim()) throw new Error("模型返回了空译文，请重试。");
      if (requestController.signal.aborted) return;
      setTarget(payload.text); setResult(payload); setTranslatedInput(fingerprint); setCleared(null);
      setMessage("翻译完成，可直接修改译文。");
    } catch (failure) {
      if (requestController.signal.aborted) setMessage("已停止等待，可重新翻译。");
      else setError(failure instanceof Error ? failure.message : "无法连接翻译服务，请检查本地 API。");
    } finally {
      controller.current = null; setBusy(false);
    }
  }

  function shortcut(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (!event.nativeEvent.isComposing && (event.metaKey || event.ctrlKey) && event.key === "Enter") { event.preventDefault(); void translate(); }
  }

  function clear() {
    setCleared({ source, target, result, input: translatedInput });
    setSource(""); setTarget(""); setResult(null); setTranslatedInput(""); setError(""); setMessage("");
    sourceInput.current?.focus();
  }

  async function copy() {
    try { await navigator.clipboard.writeText(target); setMessage("译文已复制。"); }
    catch { setError("复制失败，请选中译文后手动复制。"); }
  }

  return <section className={styles.page} hidden={!active} aria-label="文本翻译">
    <form className={styles.form} onSubmit={(event) => { event.preventDefault(); void translate(); }}>
      <header className={styles.heading}><div><span className="page-kicker">随写随译</span><h1>文本翻译</h1><p>一句话，一段文字。让表达跨越语言。</p></div><span className={styles.headingMark} aria-hidden="true">文<span>↗</span>A</span></header>
      <fieldset className={styles.options} disabled={busy}>
        <legend className={styles.srOnly}>翻译选项</legend>
        <div className={styles.modelRow}>
          <label><span>连接</span><select value={profileId} onChange={(event) => {
            const id = event.target.value;
            const next = settings?.profiles.find((item) => item.id === id);
            const defaults = next?.selected_models ?? settings?.presets.find((item) => item.id === next?.provider_type)?.default_models ?? [];
            setSelection({ profileId: id, model: id === settings?.draft_profile_id ? settings.draft_model : defaults[0] || "" });
          }}><option value="" disabled>选择连接</option>{settings?.profiles.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
          <label className={styles.modelField}><span>模型</span><input list={modelListId} value={model} onChange={(event) => setSelection({ profileId, model: event.target.value })} placeholder="选择或输入模型 ID" autoComplete="off" /><datalist id={modelListId}>{models.map((item) => <option key={item} value={item} />)}</datalist></label>
          <label><span>计算模式</span><select value={mode} onChange={(event) => setComputeMode(event.target.value as ComputeMode)}><option value="economy">节省</option><option value="balanced">均衡</option><option value="performance">性能</option></select></label>
        </div>
        <div className={styles.styleRow}><span>风格</span><div role="group" aria-label="翻译风格">{STYLE_PRESETS.map((preset) => <button type="button" key={preset.id} aria-pressed={styleGuide === preset.guide} title={preset.note} className={styleGuide === preset.guide ? styles.selected : ""} onClick={() => setStyleGuide(preset.guide)}>{preset.label}</button>)}</div><details className={styles.customStyle}><summary>自定义</summary><label htmlFor={customStyleId}>翻译要求</label><textarea id={customStyleId} value={styleGuide} maxLength={12_000} onChange={(event) => setStyleGuide(event.target.value)} placeholder="补充语气、读者或用词要求…" /></details></div>
      </fieldset>
      {!configured && <div className={styles.configHint}>选择一个已配置的模型即可开始。<button type="button" onClick={onConfigure}>模型配置 ↗</button></div>}
      <div className={styles.workspace}>
        <div className={styles.column}>
          <div className={styles.columnHeader}><label><span>原文</span><select aria-label="原文语言" disabled={busy} value={sourceLang} onChange={(event) => setSourceLang(event.target.value)}><option value="auto">自动识别</option>{LANGUAGES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><button type="button" disabled={busy || (!source && !target)} onClick={clear}>清空</button></div>
          <textarea ref={sourceInput} className={styles.editor} onKeyDown={shortcut} aria-label="输入要翻译的文本" value={source} disabled={busy} maxLength={MAX_CHARS} onChange={(event) => { setSource(event.target.value); setCleared(null); setMessage(""); }} placeholder={"在这里输入或粘贴文字…\n\n保留段落，让思想自然展开。"} spellCheck={false} />
          <div className={styles.columnFooter}><span>{source.length.toLocaleString()} / {MAX_CHARS.toLocaleString()} 字符</span>{cleared && <button type="button" onClick={() => { setSource(cleared.source); setTarget(cleared.target); setResult(cleared.result); setTranslatedInput(cleared.input); setCleared(null); setMessage("已恢复清空前的内容。"); }}>撤销清空</button>}</div>
        </div>
        <button type="button" className={styles.swap} title={sourceLang === "auto" ? "选择原文语言后可交换" : "交换语言与文本"} aria-label="交换语言与文本" disabled={busy || sourceLang === "auto" || !target} onClick={() => { setSource(target); setTarget(source); setSourceLang(targetLang); setTargetLang(sourceLang); setResult(null); setTranslatedInput(""); setCleared(null); setError(""); setMessage("已交换，点击翻译生成新译文。"); }}>⇄</button>
        <div className={`${styles.column} ${styles.target}`} aria-busy={busy}>
          <div className={styles.columnHeader}><label><span>译文</span><select aria-label="译文语言" disabled={busy} value={targetLang} onChange={(event) => setTargetLang(event.target.value)}>{LANGUAGES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><button type="button" disabled={busy || !target} onClick={() => void copy()}>复制译文</button></div>
          <textarea className={styles.editor} onKeyDown={shortcut} aria-label="译文，可直接编辑" value={target} disabled={busy} onChange={(event) => { setTarget(event.target.value); setMessage("译文已修改。"); }} placeholder={busy ? "正在翻译，请稍候…" : "译文将在这里呈现"} spellCheck={false} />
          <div className={styles.columnFooter}><span>{busy ? "模型正在组织语言…" : stale ? "原文或设置已更改，可重新翻译" : result ? `${result.model} · ${result.prompt_tokens + result.completion_tokens} tokens` : "可原位编辑译文"}</span></div>
        </div>
      </div>
      <footer className={styles.actions}><div className={styles.feedback} aria-live="polite">{error ? <span className={styles.error} role="alert">{error}</span> : <span>{message || "⌘ / Ctrl + Enter 翻译"}</span>}</div>{busy ? <button type="button" className={styles.stop} onClick={(event) => { event.preventDefault(); controller.current?.abort(); setMessage("已停止等待，可重新翻译。"); }}>停止等待</button> : <button className={styles.translate} type="submit" disabled={!canTranslate}>{target ? "重新翻译" : "开始翻译"}<span aria-hidden="true">↗</span></button>}</footer>
    </form>
  </section>;
}
