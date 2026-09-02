"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import styles from "./term-discovery.module.css";

const API_BASE = process.env.NEXT_PUBLIC_JIEYI_API || "http://127.0.0.1:8000";

type Evidence = {
  id: string; ordinal: number; source_form: string; quote: string; heading_path: string;
};
type Sense = {
  id: string; sense: string; concept_definition: string; proposed_target: string;
  rationale: string; disambiguation: string; confidence: number;
  ai_recommended: boolean | null; proposer: string; status: "pending" | "approved" | "rejected";
};
type Candidate = {
  id: string; canonical_form: string; forms: string[]; frequency: number;
  segment_frequency: number; risk_score: number; rank: number;
  candidate_type: "concept" | "named_entity" | "lexical_risk" | "unclassified";
  boundary_confidence: number;
  extraction_methods: string[]; evidence: Evidence[]; senses: Sense[];
};
type Run = {
  id: string; status: string; provider: string; model: string; error: string;
  prompt_tokens: number; completion_tokens: number; cost_usd: number;
  coverage: {
    segments_total?: number; segments_scanned?: number; characters_scanned?: number;
    retained_candidates?: number; model_calls?: number; invalid_model_proposals?: number;
    model_decisions?: number; missing_model_decisions?: number;
    model_kept?: number; model_omitted?: number; language_profile?: string;
    model_error?: string; algorithm?: string;
  };
};
export type ApprovedTerm = {
  id: string; source: string; target: string; status: string; scope: string; rationale: string;
  aliases: string[]; context_keywords: string[]; sense: string; disambiguation: string;
};
type Props = {
  documentId: string;
  provider: string;
  model: string;
  computeMode: string;
  onApproved: (term: ApprovedTerm) => void;
  notify: (message: string) => void;
};

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(API_BASE + path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `请求失败（${response.status}）`);
  return payload as T;
}

function formatCount(value = 0) {
  return value >= 1_000_000 ? `${(value / 1_000_000).toFixed(1)}M` : value >= 1_000 ? `${(value / 1_000).toFixed(1)}K` : String(value);
}

const candidateTypeLabel: Record<Candidate["candidate_type"], string> = {
  concept: "概念",
  named_entity: "专名",
  lexical_risk: "词形风险",
  unclassified: "未分类",
};

export function TermDiscoveryPanel({ documentId, provider, model, computeMode, onApproved, notify }: Props) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [discovering, setDiscovering] = useState(false);
  const [busySense, setBusySense] = useState("");
  const [targets, setTargets] = useState<Record<string, string>>({});
  const [senses, setSenses] = useState<Record<string, string>>({});
  const [contexts, setContexts] = useState<Record<string, string>>({});
  const [showOmitted, setShowOmitted] = useState(false);

  const load = useCallback(async () => {
    const [runItems, candidateItems] = await Promise.all([
      request<Run[]>(`/documents/${documentId}/term-discovery-runs`).catch(() => []),
      request<Candidate[]>(`/documents/${documentId}/term-candidates`).catch(() => []),
    ]);
    setRuns(runItems);
    setCandidates(candidateItems);
  }, [documentId]);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      request<Run[]>(`/documents/${documentId}/term-discovery-runs`).catch(() => []),
      request<Candidate[]>(`/documents/${documentId}/term-candidates`).catch(() => []),
    ]).then(([runItems, candidateItems]) => {
      if (cancelled) return;
      setRuns(runItems);
      setCandidates(candidateItems);
    });
    return () => { cancelled = true; };
  }, [documentId]);

  const hasRunningRun = runs.some((run) => run.status === "running");
  useEffect(() => {
    if (!hasRunningRun) return undefined;
    const timer = window.setInterval(() => { void load(); }, 3_000);
    return () => window.clearInterval(timer);
  }, [hasRunningRun, load]);

  const activeRun = runs.find((run) => run.status === "running");
  const latestCompleted = runs.find((run) => run.status === "completed");
  const displayRun = latestCompleted ?? runs.find((run) => run.status !== "failed");
  const running = discovering || Boolean(activeRun);
  const omittedCandidates = useMemo(
    () => candidates.filter((candidate) =>
      candidate.senses.length > 0
      && candidate.senses.every((sense) => sense.ai_recommended === false && sense.status === "pending")),
    [candidates],
  );
  const visibleCandidates = useMemo(
    () => showOmitted ? candidates : candidates.filter((candidate) => !omittedCandidates.includes(candidate)),
    [candidates, omittedCandidates, showOmitted],
  );
  const pending = useMemo(
    () => candidates.reduce(
      (total, item) => total + item.senses.filter(
        (sense) => sense.status === "pending" && sense.ai_recommended !== false).length,
      0,
    ),
    [candidates],
  );

  async function discover() {
    setDiscovering(true);
    try {
      const useModel = Boolean(provider && model);
      const run = await request<Run>(`/documents/${documentId}/term-discovery-runs`, {
        method: "POST",
        body: JSON.stringify({
          provider: useModel ? provider : "",
          model: useModel ? model : "",
          compute_mode: computeMode || "balanced",
          max_candidates: 40,
          max_model_candidates: useModel ? 40 : 0,
        }),
      });
      await load();
      const note = run.error ? `全文扫描完成；模型建议未完成：${run.error}` : "全文扫描和候选生成完成，等待人工审核。";
      notify(note);
    } catch (error) {
      notify(error instanceof Error ? error.message : "术语发现失败");
    } finally {
      setDiscovering(false);
    }
  }

  async function reject(sense: Sense) {
    setBusySense(sense.id);
    try {
      await request(`/term-candidate-senses/${sense.id}`, {
        method: "PATCH",
        body: JSON.stringify({
          status: "rejected",
          proposed_target: targets[sense.id] ?? sense.proposed_target,
          sense: senses[sense.id] ?? sense.sense,
          rationale: sense.rationale || "人工判定为无需约束的候选",
          disambiguation: sense.disambiguation,
        }),
      });
      await load();
      notify("候选已驳回；记录仍保留在本次发现运行中。 ");
    } catch (error) {
      notify(error instanceof Error ? error.message : "驳回失败");
    } finally {
      setBusySense("");
    }
  }

  async function approve(sense: Sense) {
    const target = (targets[sense.id] ?? sense.proposed_target).trim();
    if (!target) { notify("请先填写拟定译法，再批准这个义项。 "); return; }
    setBusySense(sense.id);
    try {
      const payload = await request<{ term: ApprovedTerm; impact: { translated_occurrences_checked: number; segments_needing_revision: number } }>(
        `/term-candidate-senses/${sense.id}/approve`,
        {
          method: "POST",
          body: JSON.stringify({
            target,
            sense: senses[sense.id] ?? sense.sense,
            rationale: sense.rationale || "人工依据候选证据批准",
            context_keywords: (contexts[sense.id] || "").split(/[,，;；]/).map((item) => item.trim()).filter(Boolean),
            disambiguation: sense.disambiguation,
          }),
        },
      );
      onApproved(payload.term);
      await load();
      notify(`术语已批准并检查 ${payload.impact.translated_occurrences_checked} 处既有译文；${payload.impact.segments_needing_revision} 处需复核。`);
    } catch (error) {
      notify(error instanceof Error ? error.message : "批准失败");
    } finally {
      setBusySense("");
    }
  }

  return <section className={styles.panel} aria-label="候选术语发现与审核">
    <div className={styles.heading}>
      <div><span>全书概念发现</span><h2>候选术语</h2><p>扫描覆盖全文；模型只依据原文证据提出候选和译法，人工批准前不会进入翻译约束。</p></div>
      <button type="button" onClick={() => void discover()} disabled={running}>{running ? "扫描与模型复核中…" : displayRun ? "重新扫描" : "扫描全书并生成候选"}</button>
    </div>
    {activeRun && <div className={styles.running}>扫描已提交，正在完成全文分析与模型复核；页面会自动刷新，请勿重复提交。</div>}
    {displayRun && <div className={styles.metrics}>
      <div><strong>{displayRun.coverage.segments_scanned || 0}/{displayRun.coverage.segments_total || 0}</strong><span>段落已扫描</span></div>
      <div><strong>{formatCount(displayRun.coverage.characters_scanned)}</strong><span>原文字符</span></div>
      <div><strong>{displayRun.coverage.model_calls ? displayRun.coverage.model_kept || 0 : displayRun.coverage.retained_candidates || 0}</strong><span>{displayRun.coverage.model_calls ? "模型建议候选" : "本地复核池"}</span></div>
      <div><strong>{pending}</strong><span>待人工审核</span></div>
      <div><strong>{formatCount(displayRun.prompt_tokens + displayRun.completion_tokens)}</strong><span>模型 token</span></div>
    </div>}
    {displayRun?.error && <div className={styles.modelError}>确定性全文候选已保留；模型复核未完成：{displayRun.error}</div>}
    {omittedCandidates.length > 0 && <div className={styles.omittedBar}>
      <span>模型已逐项略过 {omittedCandidates.length} 项普通词、片段或无需统一译法的内容；记录仍可审核。</span>
      <button type="button" onClick={() => setShowOmitted((value) => !value)}>{showOmitted ? "收起略过项" : "查看略过项"}</button>
    </div>}
    <div className={styles.list}>
      {visibleCandidates.map((candidate) => <article className={styles.card} key={candidate.id}>
        <header><div><span>{candidateTypeLabel[candidate.candidate_type] || "未分类"} · 风险 {Math.round(candidate.risk_score * 100)} · 排名 {candidate.rank}</span><h3>{candidate.canonical_form}</h3>{candidate.forms.length > 1 && <small>原文词形：{candidate.forms.join(" · ")}</small>}</div><div className={styles.frequency}><strong>{candidate.frequency}</strong><span>次 / {candidate.segment_frequency} 段</span></div></header>
        {candidate.senses.map((sense) => <div className={styles.sense} key={sense.id}>
          <div className={[styles.senseMeta].join(" ")}><span className={[styles.status, styles[sense.status]].join(" ")}>{sense.status === "pending" ? "待审核" : sense.status === "approved" ? "已批准" : "已驳回"}</span><small>{sense.ai_recommended === false ? "模型建议略过 · 置信度 " + Math.round(sense.confidence * 100) + "%" : sense.proposer.startsWith("deterministic") ? "本地统计召回 · 尚未模型复核" : "模型建议保留 · 置信度 " + Math.round(sense.confidence * 100) + "%"}</small></div>
          {sense.concept_definition && <p>{sense.concept_definition}</p>}
          <div className={styles.editors}><label><span>拟定译法</span><input value={targets[sense.id] ?? sense.proposed_target} onChange={(event) => setTargets((items) => ({ ...items, [sense.id]: event.target.value }))} disabled={sense.status !== "pending"} placeholder="人工确认或修改译法" /></label><label><span>义项</span><input value={senses[sense.id] ?? sense.sense} onChange={(event) => setSenses((items) => ({ ...items, [sense.id]: event.target.value }))} disabled={sense.status !== "pending"} placeholder="说明该概念在本书中的义项" /></label><label><span>语境关键词</span><input value={contexts[sense.id] || ""} onChange={(event) => setContexts((items) => ({ ...items, [sense.id]: event.target.value }))} disabled={sense.status !== "pending"} placeholder="逗号分隔，用于同形词消歧" /></label></div>
          {sense.rationale && <p className={styles.rationale}>{sense.rationale}</p>}
          <details><summary>查看 {candidate.evidence.length} 条原文证据</summary>{candidate.evidence.map((evidence) => <blockquote key={evidence.id}><small>第 {evidence.ordinal + 1} 段{evidence.heading_path ? ` · ${evidence.heading_path}` : ""}</small><p>{evidence.quote}</p></blockquote>)}</details>
          {sense.status === "pending" && <div className={styles.actions}><button type="button" className={styles.reject} disabled={busySense === sense.id} onClick={() => void reject(sense)}>驳回</button><button type="button" className={styles.approve} disabled={busySense === sense.id || !(targets[sense.id] ?? sense.proposed_target).trim()} onClick={() => void approve(sense)}>人工批准并回查译文</button></div>}
        </div>)}
      </article>)}
    </div>
    {!displayRun && !activeRun && <div className={styles.empty}>尚未扫描。系统会读取每一个原文段落，先在本地进行高召回筛选，再把少量高风险候选交给模型复核。</div>}
    {displayRun?.status === "completed" && candidates.length === 0 && <div className={styles.empty}>本次扫描没有发现需要进入人工审核的关键术语候选。</div>}
    {displayRun?.status === "completed" && candidates.length > 0 && visibleCandidates.length === 0 && !showOmitted && <div className={styles.empty}>模型已逐项复核本地召回项，本次没有建议进入人工审核的术语；略过记录仍可展开检查。</div>}
  </section>;
}

