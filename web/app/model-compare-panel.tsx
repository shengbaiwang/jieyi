"use client";

import { useMemo, useState } from "react";
import {
  availableCompareModels,
  blindCandidateLabel,
  formatCompareUsage,
  groupCandidatesByModel,
  modelOptionKey,
  shuffleCompareItems,
  toggleModelKey,
  type CompareCandidate,
  type CompareResultItem,
  type ProviderSettingsLike,
} from "./model-compare";

type ProviderSettings = ProviderSettingsLike;

type Props = {
  open: boolean;
  ordinal: number;
  sourceText: string;
  candidates: CompareCandidate[];
  settings: ProviderSettings | null;
  busy: boolean;
  results: CompareResultItem[];
  onRun: (models: Array<{ provider: string; model: string }>, computeMode: "economy" | "balanced" | "performance") => Promise<void> | void;
  onApply: (text: string) => void;
  onClose: () => void;
};

function ResultCard({
  title,
  subtitle,
  text,
  meta,
  issues,
  error,
  onApply,
  busy,
}: {
  title: string;
  subtitle?: string;
  text: string;
  meta?: string;
  issues?: Array<{ code: string; message: string; severity: string }>;
  error?: string | null;
  onApply?: () => void;
  busy: boolean;
}) {
  return (
    <article className="compare-card">
      <header>
        <div>
          <strong title={subtitle || title}>{title}</strong>
          {subtitle && <small>{subtitle}</small>}
        </div>
      </header>
      {error ? (
        <div className="compare-error">{error}</div>
      ) : (
        <pre className="compare-text">{text || "（空译文）"}</pre>
      )}
      {Boolean(issues?.length) && (
        <ul className="compare-issues">
          {issues?.map((issue) => (
            <li key={issue.code + issue.message} data-severity={issue.severity}>
              {issue.message}
            </li>
          ))}
        </ul>
      )}
      <footer>
        <small>{meta || ""}</small>
        <button
          type="button"
          className="compare-apply"
          disabled={busy || Boolean(error) || !text.trim()}
          onClick={onApply}
        >
          采用此译文
        </button>
      </footer>
    </article>
  );
}

export function ModelComparePanel(props: Props) {
  const {
    open,
    ordinal,
    sourceText,
    candidates,
    settings,
    busy,
    results,
    onRun,
    onApply,
    onClose,
  } = props;
  const options = useMemo(() => availableCompareModels(settings), [settings]);
  const groups = useMemo(() => shuffleCompareItems(groupCandidatesByModel(candidates)), [candidates]);
  const defaultSelected = useMemo(
    () => options.slice(0, Math.min(2, options.length)).map((item) => modelOptionKey(item.provider, item.model)),
    [options],
  );
  const [selected, setSelected] = useState<string[] | null>(null);
  const [computeMode, setComputeMode] = useState<"economy" | "balanced" | "performance">("economy");
  const [revealModels, setRevealModels] = useState(false);
  const activeSelected = selected ?? defaultSelected;
  const resultCards = useMemo(() => {
    const cards: CompareResultItem[] = results.length
      ? results
      : groups.map((group) => {
          const latest = group.latest!;
          return {
            provider: group.provider,
            model: group.model,
            text: latest.text,
            ok: true,
            prompt_tokens: latest.prompt_tokens,
            completion_tokens: latest.completion_tokens,
            reasoning_tokens: latest.reasoning_tokens,
            cost_usd: latest.cost_usd,
            issues: [],
          };
        });
    // Re-shuffle only when the result set identity changes, not on reveal toggle.
    return shuffleCompareItems(cards);
  }, [results, groups]);

  if (!open) return null;

  const selectedModels = options.filter((item) => activeSelected.includes(modelOptionKey(item.provider, item.model)));

  return (
    <div
      className="command-overlay model-compare-overlay"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !busy) onClose();
      }}
    >
      <div className="model-compare-dialog" role="dialog" aria-modal="true" aria-labelledby="model-compare-title">
        <header>
          <div>
            <span>多模型对比</span>
            <h2 id="model-compare-title">第 {ordinal + 1} 段 · 并排比较译文</h2>
            <p>同一段原文交给多个模型；默认隐匿模型名便于盲评，对比结果只作候选，采用后才会写入当前译文编辑框。</p>
          </div>
          <button aria-label="关闭模型对比" onClick={onClose} disabled={busy}>×</button>
        </header>

        <div className="model-compare-body">
          <section className="compare-source">
            <h3>原文</h3>
            <p>{sourceText}</p>
          </section>

          <section className="compare-controls">
            <div className="compare-model-picks">
              <h3>选择模型 <small>建议 2–4 个</small></h3>
              {options.length ? (
                <div className="compare-model-list" role="group" aria-label="选择对比模型">
                  {options.map((item) => {
                    const key = modelOptionKey(item.provider, item.model);
                    const checked = activeSelected.includes(key);
                    return (
                      <label key={key} className={checked ? "selected" : ""}>
                        <input
                          type="checkbox"
                          checked={checked}
                          disabled={busy || (!checked && activeSelected.length >= 6)}
                          onChange={() => setSelected(toggleModelKey(activeSelected, key))}
                        />
                        <span className="chapter-check" aria-hidden="true">✓</span>
                        <span>{item.label || item.model}</span>
                      </label>
                    );
                  })}
                </div>
              ) : (
                <p className="compare-empty">请先在「模型配置」中添加连接与模型。</p>
              )}
            </div>
            <div className="compare-actions">
              <label>
                <span>计算档位</span>
                <select
                  value={computeMode}
                  disabled={busy}
                  onChange={(event) => setComputeMode(event.target.value as "economy" | "balanced" | "performance")}
                >
                  <option value="economy">节省</option>
                  <option value="balanced">均衡</option>
                  <option value="performance">性能</option>
                </select>
              </label>
              <button
                type="button"
                className="accent"
                disabled={busy || !selectedModels.length}
                onClick={() => void onRun(selectedModels.map(({ provider, model }) => ({ provider, model })), computeMode)}
              >
                {busy ? "对比中…" : selectedModels.length < 2 ? "至少选 2 个模型" : "开始对比"}
              </button>
            </div>
          </section>

          <section className="compare-results" aria-live="polite">
            <div className="compare-results-head">
              <h3>
                译文对比
                <small>
                  {busy
                    ? "正在调用所选模型…"
                    : resultCards.length
                      ? revealModels
                        ? `${resultCards.length} 列 · 模型名已显示`
                        : `${resultCards.length} 列 · 模型名已隐匿`
                      : "运行后并排显示各模型译文"}
                </small>
              </h3>
              {resultCards.length > 0 && (
                <button
                  type="button"
                  className="compare-reveal"
                  aria-pressed={revealModels}
                  disabled={busy}
                  onClick={() => setRevealModels((value) => !value)}
                >
                  {revealModels ? "隐匿模型名" : "显示模型名"}
                </button>
              )}
            </div>
            {resultCards.length ? (
              <div className="compare-grid" style={{ gridTemplateColumns: `repeat(${Math.max(2, resultCards.length)}, minmax(220px, 1fr))` }}>
                {resultCards.map((item, index) => (
                  <ResultCard
                    key={`${item.provider}::${item.model}::${item.candidate_id || item.text.slice(0, 24)}`}
                    title={revealModels ? item.model : blindCandidateLabel(index)}
                    subtitle={revealModels ? item.provider : "盲评 · 模型名已隐匿"}
                    text={item.text || item.raw_text || ""}
                    meta={formatCompareUsage(item)}
                    issues={item.issues}
                    error={item.ok === false ? item.error || "本模型生成失败" : item.error}
                    busy={busy}
                    onApply={() => onApply(item.text || item.raw_text || "")}
                  />
                ))}
              </div>
            ) : (
              <div className="compare-empty">还没有可对比的译文。选择模型后点击「开始对比」。</div>
            )}
          </section>

          {groups.length > 0 && !results.length && (
            <section className="compare-history">
              <h3>
                历史候选
                <small>{revealModels ? "本段已有记录，按模型归类" : "本段已有记录 · 模型名已隐匿"}</small>
              </h3>
              <div className="compare-history-list">
                {groups.map((group, index) => (
                  <div key={group.key}>
                    <strong>{revealModels ? group.model : blindCandidateLabel(index)}</strong>
                    <small>
                      {revealModels
                        ? `${group.provider} · ${group.candidates.length} 条 · 最新 ${group.latest?.created_at?.slice(0, 19).replace("T", " ") || "—"}`
                        : `${group.candidates.length} 条 · 最新 ${group.latest?.created_at?.slice(0, 19).replace("T", " ") || "—"}`}
                    </small>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>

        <footer>
          <span>对比不会覆盖人工确认译文 · 默认隐匿模型名，便于盲评</span>
          <div>
            <button type="button" onClick={onClose} disabled={busy}>关闭</button>
          </div>
        </footer>
      </div>
    </div>
  );
}
