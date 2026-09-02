"use client";

import { useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_JIEYI_API || "http://127.0.0.1:8000";
type Summary = {
  configured_model: string; service_host: string;
  occurrences: number; pending: number; pending_segments: number;
  counts: Record<string, number>;
  terms: { term_id: string; source: string; target: string; pending: number; checked: number }[];
  latest_run: null | {
    id: string; status: string; error: string; model: string;
    usage: { verified?: number; cost_usd?: number };
  };
};

export function TerminologyReviewPanel({ documentId, onChange }: {
  documentId: string; onChange: () => void;
}) {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);
  const revision = useRef("");
  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    const refresh = async () => {
      try {
        const response = await fetch(`${API_BASE}/documents/${documentId}/terminology-review`);
        if (!response.ok) throw new Error("无法读取术语核验状态，请检查本地服务。");
        const value: Summary = await response.json();
        if (disposed) return;
        setSummary(value);
        setError("");
        const next = JSON.stringify(value);
        if (revision.current !== next) { revision.current = next; onChange(); }
      } catch (failure) {
        if (!disposed) setError(failure instanceof Error ? failure.message : "核验状态读取失败");
      } finally {
        if (!disposed) timer = setTimeout(refresh, 3000);
      }
    };
    void refresh();
    return () => { disposed = true; clearTimeout(timer); };
  }, [documentId, onChange]);

  const running = starting || ["pending", "running"].includes(summary?.latest_run?.status || "");
  async function start() {
    setStarting(true); setError("");
    try {
      const response = await fetch(`${API_BASE}/documents/${documentId}/terminology-review`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
      });
      const value = await response.json();
      if (!response.ok) throw new Error(value.detail || "无法启动术语核验");
      setSummary((current) => current ? { ...current, latest_run: value } : current);
    } catch (failure) { setError(failure instanceof Error ? failure.message : "核验启动失败"); }
    finally { setStarting(false); }
  }
  if (!summary && !error) return <div className="empty-inline">正在读取术语核验状态…</div>;
  if (!summary?.occurrences && !error) return null;
  return <section className="terminology-review" aria-label="术语语境核验">
    <div className="terminology-review-heading"><div><strong>术语语境核验</strong>
      <p aria-live="polite">{summary ? `共 ${summary.occurrences} 处用法；${summary.pending_segments} 段、${summary.pending} 处待机器核验。` : "状态暂不可用。"}</p>
      <p>先判断每处用法是否属于批准义项，再核对对应译文。待核验项不计入人工必检，也不视为已通过。</p>
    </div><button className="primary-button" disabled={running || !summary?.pending} onClick={() => void start()}>
      {running ? "核验中…" : summary?.latest_run ? "继续核验剩余项" : "开始语境核验"}
    </button></div>
    {summary?.configured_model && <p>启动后会将待核验的原译文发送至 {summary.service_host}，使用 {summary.configured_model} 核验。</p>}
    {summary && <div className="terminology-review-counts">
      <span>译法一致 {summary.counts.consistent || 0}</span>
      <span>义项不适用 {summary.counts.not_applicable || 0}</span>
      <span>疑似不一致 {summary.counts.inconsistent || 0}</span>
      <span>需人工判断 {summary.counts.uncertain || 0}</span>
    </div>}
    {summary?.terms.map((term) => <p key={term.term_id}>{term.source} → {term.target}：{term.checked} 处已核验，{term.pending} 处待核验</p>)}
    {summary?.latest_run && <p>最近核验：{summary.latest_run.model} · 本轮费用 ${(summary.latest_run.usage.cost_usd || 0).toFixed(4)}{summary.latest_run.status === "completed" ? " · 本轮已完成" : ""}</p>}
    {(error || summary?.latest_run?.error) && <p className="job-error" role="status">{error || summary?.latest_run?.error}</p>}
  </section>;
}
