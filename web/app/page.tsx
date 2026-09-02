"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ImportBookPanel, ProviderSettingsPanel } from "./setup-panels";
import { TermDiscoveryPanel } from "./term-discovery-panel";

const API_BASE = process.env.NEXT_PUBLIC_JIEYI_API || "http://127.0.0.1:8000";
const PAGE_SIZE = 120;

type ReasoningEffort = "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";
type ComputeMode = "economy" | "balanced" | "performance";
type Panel = "library" | "translate" | "reader" | "terms" | "quality" | "import" | "settings";
type ViewMode = "split" | "target";
type Project = { id: string; name: string; source_lang: string; target_lang: string; style_guide: string };
type Document = { id: string; project_id: string; title: string; source_format: string; created_at: string; cover_url: string | null };
type Segment = {
  id: string; document_id: string; ordinal: number; kind: "heading" | "paragraph" | "blockquote" | "footnote";
  source_text: string; heading_path: string; machine_translation: string | null; edited_translation: string | null; reviewed_translation: string | null;
  accepted_translation: string | null; status: "source" | "machine_translated" | "human_confirmed";
};
type Chapter = { title: string; level: number; start_ordinal: number; end_ordinal: number; segment_count: number; translated_count: number; confirmed_count: number };
type Overview = { document: Document; project: Project; segment_count: number; translated_count: number; reviewed_count: number; confirmed_count: number; chapters: Chapter[] };
type SegmentPage = { document: Document; project: Project; total: number; offset: number; limit: number; items: Segment[] };
type Term = {
  id: string; source: string; target: string; status: string; scope: string; rationale: string;
  aliases: string[]; context_keywords: string[]; sense: string; disambiguation: string;
};
type Issue = { id: string; segment_id: string; ordinal: number; code: string; message: string; severity: string; resolved: number };
type HumanReviewItem = { segment_id: string; ordinal: number; reason: "error" | "warning" | "sample"; issue_count: number; source_text: string; translation: string };
type Candidate = { id: string; stage: string; provider: string; model: string; prompt_tokens: number; completion_tokens: number; cost_usd: number; created_at: string };
type Job = {
  id: string; status: "pending" | "running" | "paused" | "completed" | "failed" | "cancelled";
  next_ordinal: number; total_cost_usd: number; last_error: string | null;
  total_segments?: number; processed_segments?: number; batch_count?: number;
  prompt_tokens?: number; completion_tokens?: number; reasoning_tokens?: number; total_tokens?: number;
  elapsed_seconds?: number; eta_seconds?: number | null;
  deferred_segments?: number;
  recipe: {
    task_mode?: "draft" | "review"; review_policy?: "never" | "on_issue" | "all"; batch_size?: number; concurrency?: number; max_concurrency?: number;
    draft_thinking?: boolean; review_thinking?: boolean; draft_compute_mode?: ComputeMode; review_compute_mode?: ComputeMode; draft_reasoning_effort?: ReasoningEffort; review_reasoning_effort?: ReasoningEffort; review_sample_rate?: number; segment_ranges?: [number, number][];
    draft: { model: string; provider: string };
    reviewer?: { model: string; provider: string } | null;
  };
};
type ProviderSettings = {
  base_url: string; draft_model: string; draft_compute_mode: ComputeMode; reviewer_model: string; reviewer_compute_mode: ComputeMode; review_enabled: boolean;
  api_key_configured: boolean; provider_type: string;
  draft_provider: string; reviewer_provider: string;
  draft_profile_id: string; reviewer_profile_id: string;
  profiles: Array<{ id: string; name: string; provider_type: string; base_url: string }>;
  presets: Array<{ id: string; default_models: string[] }>;
};
type BookTranslationSettings = {
  draft_profile_id: string; draft_model: string; draft_compute_mode: ComputeMode;
  reviewer_profile_id: string; reviewer_model: string; reviewer_compute_mode: ComputeMode;
};
type BookSettingsEditor = { document: Document; project: Project; value: BookTranslationSettings; style_guide: string };
type ReaderMode = "original" | "translated" | "bilingual";
type ReaderLayout = "faithful" | "comfort";
type EpubReaderManifest = {
  document_id: string; cover_url: string | null; rendition_layout: string;
  spine: Array<{ spine_index: number; path: string; linear: boolean; fixed_layout: boolean }>;
  segment_locations: Array<{ segment_ordinal: number; spine_index: number }>;
};
type ReaderPageTarget = { spineIndex: number; pageNumber: number };

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers: { "Content-Type": "application/json", ...(options?.headers || {}) } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `请求失败（${response.status}）`);
  return payload as T;
}

function Mark({ children }: { children: React.ReactNode }) { return <span className="symbol" aria-hidden="true">{children}</span>; }
function TranslationIcon() {
  return <svg viewBox="0 0 16 16" focusable="false"><path d="M3 2.5h7.5A1.5 1.5 0 0 1 12 4v3.1M4.5 5h5M4.5 7.5h3" /><path d="m8.2 12.7.4-1.8 3.6-3.6 1.5 1.5-3.6 3.6-1.9.3Z" /></svg>;
}
function LibraryIcon() {
  return <svg viewBox="0 0 16 16" focusable="false"><rect x="1.75" y="2.5" width="3.35" height="10.75" rx=".65" /><rect x="6.15" y="3.25" width="3.35" height="10" rx=".65" /><path d="m10.7 3.1 2.55-.55 2.05 9.7-2.55.55-2.05-9.7Z" /></svg>;
}
function ThemeIcon({ dark }: { dark: boolean }) {
  return dark
    ? <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3.5" /><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M18.7 5.3l-1.4 1.4M6.7 17.3l-1.4 1.4" /></svg>
    : <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 15.2A8.2 8.2 0 0 1 8.8 4a8.2 8.2 0 1 0 11.2 11.2Z" /></svg>;
}
function segmentTranslation(segment: Segment | null) { return segment?.accepted_translation || segment?.reviewed_translation || segment?.edited_translation || segment?.machine_translation || ""; }
function statusLabel(segment: Segment | null) { return segment?.status === "human_confirmed" ? "人工已确认" : segment?.reviewed_translation ? "AI 已审校" : segment?.status === "machine_translated" ? "机器草译" : "待翻译"; }
function jobStatusLabel(status: Job["status"]) { return ({ pending: "等待中", running: "运行中", paused: "已暂停", completed: "已完成", failed: "失败", cancelled: "已取消" } as const)[status]; }
function formatTokens(value = 0) { return value >= 1_000_000 ? (value / 1_000_000).toFixed(2) + "M" : value >= 1_000 ? (value / 1_000).toFixed(1) + "K" : String(value); }
function formatDuration(value?: number | null) { if (value == null) return "测算中"; if (value < 60) return Math.max(1, Math.round(value)) + " 秒"; const minutes = Math.round(value / 60); return minutes < 60 ? minutes + " 分钟" : Math.floor(minutes / 60) + " 小时 " + (minutes % 60) + " 分"; }
const STYLE_PRESETS = [
  { id: "academic", label: "学术严谨", guide: "忠实原意，保持严谨、克制的学术表达；统一核心概念译法，保留引文、脚注、专名及论证层次。" },
  { id: "literary", label: "文学自然", guide: "准确传达原意与人物声调，译文自然流畅并保留文学节奏、意象与修辞；避免生硬直译。" },
  { id: "popular", label: "通俗易读", guide: "在不损失关键信息的前提下使用清晰、自然、易读的现代语言；必要术语首次出现时给出简短说明。" },
  { id: "faithful", label: "忠实直译", guide: "尽量贴近原文句法、措辞和段落结构，不擅自增删或改写；歧义处保留原文的开放性。" },
];
function defaultBookSettings(settings: ProviderSettings): BookTranslationSettings {
  return {
    draft_profile_id: settings.draft_profile_id,
    draft_model: settings.draft_model,
    draft_compute_mode: settings.draft_compute_mode || "economy",
    reviewer_profile_id: settings.reviewer_profile_id || settings.draft_profile_id,
    reviewer_model: settings.reviewer_model,
    reviewer_compute_mode: settings.reviewer_compute_mode || "performance",
  };
}
function modelsForProfile(settings: ProviderSettings | null, profileId: string): string[] {
  if (!settings) return [];
  const profile = settings.profiles.find((item) => item.id === profileId);
  const preset = settings.presets.find((item) => item.id === profile?.provider_type);
  const bound = [
    settings.draft_profile_id === profileId ? settings.draft_model : "",
    settings.reviewer_profile_id === profileId ? settings.reviewer_model : "",
  ];
  return [...new Set([...(preset?.default_models || []), ...bound].filter(Boolean))];
}
function exportName(document: Document) {
  const extension = document.source_format === "epub" ? "epub" : document.source_format === "markdown" ? "md" : "txt";
  return `${document.title}-译文.${extension}`;
}
function computeModeFromLegacy(mode: ComputeMode | undefined, effort: ReasoningEffort | undefined, thinking: boolean | undefined, fallback: ComputeMode): ComputeMode {
  if (mode) return mode;
  if (effort === "medium") return "balanced";
  if (effort && ["high", "xhigh", "max"].includes(effort)) return "performance";
  if (effort) return "economy";
  if (thinking !== undefined) return thinking ? "performance" : "economy";
  return fallback;
}
function normalizeSegmentRanges(ranges: [number, number][]) {
  const sorted = ranges
    .map(([start, end]) => [Math.min(start, end), Math.max(start, end)] as [number, number])
    .sort((left, right) => left[0] - right[0] || left[1] - right[1]);
  const merged: [number, number][] = [];
  for (const [start, end] of sorted) {
    const previous = merged.at(-1);
    if (!previous || start > previous[1] + 1) merged.push([start, end]);
    else previous[1] = Math.max(previous[1], end);
  }
  return merged;
}
function jobMatchesSettings(job: Job, settings: ProviderSettings, mode: "draft" | "review", segmentRanges: [number, number][] = []) {
  if (JSON.stringify(normalizeSegmentRanges(job.recipe.segment_ranges || [])) !== JSON.stringify(normalizeSegmentRanges(segmentRanges))) return false;
  if ((job.recipe.task_mode || "draft") !== mode) return false;
  if (job.recipe.draft.provider !== settings.draft_provider || job.recipe.draft.model !== settings.draft_model) return false;
  const draftMode = computeModeFromLegacy(job.recipe.draft_compute_mode, job.recipe.draft_reasoning_effort, job.recipe.draft_thinking, "economy");
  if (draftMode !== settings.draft_compute_mode) return false;
  if (mode === "review") {
    if (job.recipe.review_policy !== "all") return false;
    const reviewMode = computeModeFromLegacy(job.recipe.review_compute_mode, job.recipe.review_reasoning_effort, job.recipe.review_thinking, "performance");
    return job.recipe.reviewer?.provider === settings.reviewer_provider && job.recipe.reviewer?.model === settings.reviewer_model && reviewMode === settings.reviewer_compute_mode;
  }
  return true;
}
function overviewFromSegments(document: Document, project: Project, segments: Segment[]): Overview {
  const chapters: Chapter[] = [];
  for (const segment of segments) {
    const startsChapter = segment.kind === "heading";
    const title = startsChapter ? segment.source_text : segment.heading_path.split(" / ", 1)[0].trim() || "正文";
    if (!chapters.length || startsChapter) chapters.push({ title, level: startsChapter ? Math.max(0, segment.heading_path.split(" / ").length - 1) : 0, start_ordinal: segment.ordinal, end_ordinal: segment.ordinal, segment_count: 0, translated_count: 0, confirmed_count: 0 });
    const chapter = chapters.at(-1)!;
    chapter.end_ordinal = segment.ordinal; chapter.segment_count += 1;
    if (segmentTranslation(segment)) chapter.translated_count += 1;
    if (segment.status === "human_confirmed") chapter.confirmed_count += 1;
  }
  return {
    document, project, chapters, segment_count: segments.length,
    translated_count: segments.filter((item) => Boolean(segmentTranslation(item))).length,
    reviewed_count: segments.filter((item) => Boolean(item.reviewed_translation)).length,
    confirmed_count: segments.filter((item) => item.status === "human_confirmed").length,
  };
}

export default function Home() {
  const [panel, setPanel] = useState<Panel>("library");
  const [view, setView] = useState<ViewMode>("split");
  const [projects, setProjects] = useState<Project[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState("");
  const [overview, setOverview] = useState<Overview | null>(null);
  const [page, setPage] = useState<SegmentPage | null>(null);
  const [ordinal, setOrdinal] = useState(0);
  const [terms, setTerms] = useState<Term[]>([]);
  const [issues, setIssues] = useState<Issue[]>([]);
  const [humanReviewQueue, setHumanReviewQueue] = useState<HumanReviewItem[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [jobsDocumentId, setJobsDocumentId] = useState("");
  const [settings, setSettings] = useState<ProviderSettings | null>(null);
  const [bookSettings, setBookSettings] = useState<Record<string, BookTranslationSettings>>({});
  const [bookSettingsEditor, setBookSettingsEditor] = useState<BookSettingsEditor | null>(null);
  const [savingBookSettings, setSavingBookSettings] = useState(false);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [translation, setTranslation] = useState("");
  const [draftDirty, setDraftDirty] = useState(false);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [loading, setLoading] = useState(true);
  const [wholeBookAction, setWholeBookAction] = useState<"draft" | "review" | null>(null);
  const [draftPicker, setDraftPicker] = useState<{ document: Document; overview: Overview } | null>(null);
  const [selectedChapterStarts, setSelectedChapterStarts] = useState<number[]>([]);
  const [draftPickerLoading, setDraftPickerLoading] = useState(false);
  const [readerSegments, setReaderSegments] = useState<Segment[]>([]);
  const [readerLoading, setReaderLoading] = useState(false);
  const [readerMode, setReaderMode] = useState<ReaderMode>("bilingual");
  const [readerLayout, setReaderLayout] = useState<ReaderLayout>("faithful");
  const [epubReader, setEpubReader] = useState<EpubReaderManifest | null>(null);
  const [epubSpineHeights, setEpubSpineHeights] = useState<Record<number, number>>({});
  const [readerPageTargets, setReaderPageTargets] = useState<Record<number, ReaderPageTarget>>({});
  const [readerOrdinal, setReaderOrdinal] = useState(0);
  const [readerCurrentPage, setReaderCurrentPage] = useState(1);
  const [pendingDelete, setPendingDelete] = useState<Document | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [dark, setDark] = useState(false);
  const [toast, setToast] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<Segment[]>([]);
  const [termSource, setTermSource] = useState("");
  const [termTarget, setTermTarget] = useState("");
  const [termAliases, setTermAliases] = useState("");
  const [termContext, setTermContext] = useState("");
  const [termSense, setTermSense] = useState("");
  const [termDisambiguation, setTermDisambiguation] = useState("");
  const [libraryError, setLibraryError] = useState("");
  const searchInput = useRef<HTMLInputElement>(null);
  const readerView = useRef<HTMLElement>(null);
  const readerPendingOrdinal = useRef<number | null>(null);
  const readerScrollLockOrdinal = useRef<number | null>(null);
  const legacySegments = useRef(new Map<string, Segment[]>());

  const currentSegment = useMemo(() => page?.items.find((item) => item.ordinal === ordinal) || null, [ordinal, page]);
  const currentChapter = useMemo(() => {
    const activeOrdinal = panel === "reader" ? readerOrdinal : ordinal;
    return overview?.chapters.find((item) => activeOrdinal >= item.start_ordinal && activeOrdinal <= item.end_ordinal) || null;
  }, [ordinal, overview, panel, readerOrdinal]);
  const currentIssues = useMemo(() => issues.filter((item) => item.segment_id === currentSegment?.id), [currentSegment, issues]);
  const errorIssues = useMemo(() => issues.filter((item) => item.severity === "error"), [issues]);
  const displayIssues = useMemo(() => [...issues].sort((a, b) => Number(a.severity !== "error") - Number(b.severity !== "error") || a.ordinal - b.ordinal), [issues]);
  const requiredHumanChecks = useMemo(() => humanReviewQueue.filter((item) => item.reason !== "sample").length, [humanReviewQueue]);
  const currentErrorIssues = useMemo(() => currentIssues.filter((item) => item.severity === "error"), [currentIssues]);
  const currentWarningIssues = useMemo(() => currentIssues.filter((item) => item.severity !== "error"), [currentIssues]);
  const relevantTerms = useMemo(() => {
    const source = currentSegment?.source_text.toLocaleLowerCase() || "";
    return terms.filter((term) => [term.source, ...(term.aliases || [])].some((form) => source.includes(form.toLocaleLowerCase()))).slice(0, 12);
  }, [currentSegment, terms]);
  const selectedDocument = useMemo(() => documents.find((item) => item.id === selectedDocumentId) || null, [documents, selectedDocumentId]);
  const latestJob = jobsDocumentId === overview?.document.id ? jobs[0] || null : null;
  const selectedJob = jobsDocumentId === selectedDocumentId ? jobs[0] || null : null;
  const selectedDraftJob = jobsDocumentId === selectedDocumentId ? jobs.find((item) => (item.recipe.task_mode || "draft") === "draft" && item.status !== "cancelled") || null : null;
  const selectedReviewJob = jobsDocumentId === selectedDocumentId ? jobs.find((item) => item.recipe.task_mode === "review" && item.status !== "cancelled") || null : null;
  const activeJobId = jobs.find((item) => item.status === "running" || item.status === "pending")?.id || "";
  const completion = overview?.segment_count ? Math.round((overview.confirmed_count / overview.segment_count) * 100) : 0;
  const translationProgress = overview?.segment_count ? Math.round((overview.translated_count / overview.segment_count) * 100) : 0;
  const selectedDraftChapters = draftPicker?.overview.chapters.filter((chapter) => selectedChapterStarts.includes(chapter.start_ordinal)) || [];
  const selectedDraftRanges = normalizeSegmentRanges(selectedDraftChapters.map((chapter) => [chapter.start_ordinal, chapter.end_ordinal]));
  const selectedDraftSegmentCount = selectedDraftRanges.reduce((total, [start, end]) => total + end - start + 1, 0);

  function notify(message: string) { setToast(message); window.setTimeout(() => setToast(""), 2800); }

  function settingsForBook(documentId: string, providerSettings: ProviderSettings): BookTranslationSettings {
    return bookSettings[documentId] || defaultBookSettings(providerSettings);
  }

  function openBookSettings(document: Document) {
    if (!settings) { setPanel("settings"); notify("请先添加至少一个模型连接。"); return; }
    const project = projects.find((item) => item.id === document.project_id);
    if (!project) return;
    setBookSettingsEditor({ document, project, value: settingsForBook(document.id, settings), style_guide: project.style_guide || STYLE_PRESETS[0].guide });
  }

  async function saveBookSettings() {
    if (!bookSettingsEditor) return;
    const value = bookSettingsEditor.value;
    if (!value.draft_profile_id || !value.draft_model.trim()) { notify("请选择草译连接并填写模型 ID。"); return; }
    setSavingBookSettings(true);
    try {
      const updatedProject = await api<Project>(`/projects/${bookSettingsEditor.project.id}/style`, { method: "PATCH", body: JSON.stringify({ style_guide: bookSettingsEditor.style_guide }) });
      const next = { ...value, draft_model: value.draft_model.trim(), reviewer_model: value.reviewer_model.trim() };
      setBookSettings((current) => ({ ...current, [bookSettingsEditor.document.id]: next }));
      localStorage.setItem(`jieyi.book-settings.${bookSettingsEditor.document.id}`, JSON.stringify(next));
      setProjects((items) => items.map((item) => item.id === updatedProject.id ? updatedProject : item));
      setOverview((current) => current?.project.id === updatedProject.id ? { ...current, project: updatedProject } : current);
      setBookSettingsEditor(null);
      notify("已保存这本书的翻译风格与模型设置。");
    } catch (error) { notify(error instanceof Error ? error.message : "书籍设置保存失败"); }
    finally { setSavingBookSettings(false); }
  }

  const loadAllSegments = useCallback(async (documentId: string) => {
    const cached = legacySegments.current.get(documentId);
    if (cached) return cached;
    const items = await api<Segment[]>(`/documents/${documentId}/segments`);
    legacySegments.current.set(documentId, items);
    return items;
  }, []);

  const loadPageAt = useCallback(async (documentId: string, targetOrdinal: number) => {
    const offset = Math.floor(Math.max(0, targetOrdinal) / PAGE_SIZE) * PAGE_SIZE;
    let value: SegmentPage;
    try {
      value = await api<SegmentPage>(`/documents/${documentId}/segments/page?offset=${offset}&limit=${PAGE_SIZE}`);
    } catch {
      const items = await loadAllSegments(documentId);
      let document = documents.find((item) => item.id === documentId);
      let project = projects.find((item) => item.id === document?.project_id);
      if (!document || !project) {
        const projectItems = await api<Project[]>("/projects");
        const grouped = await Promise.all(projectItems.map((item) => api<Document[]>(`/projects/${item.id}/documents`)));
        document = grouped.flat().find((item) => item.id === documentId);
        project = projectItems.find((item) => item.id === document?.project_id);
      }
      if (!document || !project) throw new Error("书籍数据不完整，请重新启动介译后再试。");
      value = { document, project, total: items.length, offset, limit: PAGE_SIZE, items: items.slice(offset, offset + PAGE_SIZE) };
    }
    setPage(value);
    const safeOrdinal = Math.min(Math.max(0, targetOrdinal), Math.max(0, value.total - 1));
    setOrdinal(safeOrdinal);
    localStorage.setItem("jieyi.document", documentId);
    localStorage.setItem(`jieyi.ordinal.${documentId}`, String(safeOrdinal));
  }, [documents, loadAllSegments, projects]);

  const openDocument = useCallback(async (document: Document, targetOrdinal?: number) => {
    setLoading(true);
    setSelectedDocumentId(document.id);
    try {
      setLibraryError("");
      let summary: Overview;
      try {
        summary = await api<Overview>(`/documents/${document.id}/overview`);
      } catch {
        const allSegments = await loadAllSegments(document.id);
        const project = projects.find((item) => item.id === document.project_id) || (await api<Project[]>("/projects")).find((item) => item.id === document.project_id);
        if (!project) throw new Error("找不到这本书所属的项目。");
        summary = overviewFromSegments(document, project, allSegments);
      }
      const [termItems, issueItems, reviewItems, jobItems] = await Promise.all([
        api<Term[]>(`/projects/${summary.project.id}/terms`).catch(() => []),
        api<Issue[]>(`/documents/${document.id}/issues`).catch(() => []),
        api<HumanReviewItem[]>(`/documents/${document.id}/human-review-queue`).catch(() => []),
        api<Job[]>(`/documents/${document.id}/jobs`).catch(() => []),
      ]);
      setOverview(summary); setTerms(termItems); setIssues(issueItems); setHumanReviewQueue(reviewItems); setJobs(jobItems); setJobsDocumentId(document.id);
      const remembered = Number(localStorage.getItem(`jieyi.ordinal.${document.id}`) || 0);
      await loadPageAt(document.id, targetOrdinal ?? remembered);
      setPanel("translate");
    } catch (error) { const message = error instanceof Error ? error.message : "无法打开书籍"; setLibraryError(message); notify(message); }
    finally { setLoading(false); }
  }, [loadAllSegments, loadPageAt, projects]);

  const loadLibrary = useCallback(async () => {
    setLoading(true);
    try {
      const projectItems = await api<Project[]>("/projects");
      const grouped = await Promise.all(projectItems.map((project) => api<Document[]>(`/projects/${project.id}/documents`)));
      const documentItems = grouped.flat();
      setProjects(projectItems); setDocuments(documentItems);
      const providerSettings = await api<ProviderSettings>("/settings/provider").catch(() => null);
      setSettings(providerSettings);
      const savedBookSettings: Record<string, BookTranslationSettings> = {};
      if (providerSettings) {
        for (const document of documentItems) {
          try {
            const saved = JSON.parse(localStorage.getItem(`jieyi.book-settings.${document.id}`) || "null") as Partial<BookTranslationSettings> | null;
            if (saved?.draft_profile_id && saved.draft_model) savedBookSettings[document.id] = { ...defaultBookSettings(providerSettings), ...saved };
          } catch { /* Ignore malformed local preferences and use the global defaults. */ }
        }
      }
      setBookSettings(savedBookSettings);
      const rememberedId = localStorage.getItem("jieyi.document");
      const initial = documentItems.find((item) => item.id === rememberedId) || documentItems[0];
      setSelectedDocumentId(initial?.id || "");
      if (initial) await openDocument(initial); else setPanel("library");
    } catch { notify("无法连接本地 API，请使用“启动介译.command”重新启动。"); }
    finally { setLoading(false); }
  }, [openDocument]);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadLibrary(), 0);
    return () => window.clearTimeout(timer);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const savedMode = localStorage.getItem("jieyi.reader.mode") as ReaderMode | null;
      if (savedMode && ["original", "translated", "bilingual"].includes(savedMode)) {
        setReaderMode(savedMode);
      } else {
        const legacy = localStorage.getItem("jieyi.reader.bilingual");
        if (legacy !== null) setReaderMode(legacy === "true" ? "bilingual" : "translated");
      }
      const savedLayout = localStorage.getItem("jieyi.reader.layout") as ReaderLayout | null;
      if (savedLayout && ["faithful", "comfort"].includes(savedLayout)) setReaderLayout(savedLayout);
      const savedTheme = localStorage.getItem("jieyi.theme");
      setDark(savedTheme ? savedTheme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches);
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!epubReader) return;
    function handleEpubMessage(event: MessageEvent) {
      const data = event.data as { type?: string; documentId?: string; spineIndex?: number; segmentOrdinal?: number; height?: number; top?: number };
      if (data?.documentId !== epubReader?.document_id) return;
      const spineIndex = Number(data.spineIndex);
      if (!Number.isInteger(spineIndex)) return;
      const frame = document.querySelector<HTMLIFrameElement>(`iframe[data-spine-index="${spineIndex}"]`);
      if (!frame || event.source !== frame.contentWindow) return;
      if (data.type === "jy-epub-location") {
        const segmentOrdinal = Number(data.segmentOrdinal);
        const top = Number(data.top);
        if (!Number.isInteger(segmentOrdinal) || !Number.isFinite(top) || readerPendingOrdinal.current !== segmentOrdinal) return;
        const root = readerView.current;
        const section = frame.closest<HTMLElement>(".epub-spine-sheet");
        if (!root || !section) return;
        const rootRect = root.getBoundingClientRect();
        const sectionRect = section.getBoundingClientRect();
        readerPendingOrdinal.current = null;
        readerScrollLockOrdinal.current = segmentOrdinal;
        setReaderOrdinal(segmentOrdinal);
        root.scrollTo({ top: Math.max(0, root.scrollTop + sectionRect.top - rootRect.top + top - 64), behavior: "smooth" });
        window.setTimeout(() => {
          if (readerScrollLockOrdinal.current === segmentOrdinal) readerScrollLockOrdinal.current = null;
        }, 900);
        return;
      }
      if (data.type !== "jy-epub-resize") return;
      const reportedHeight = Number(data.height);
      if (!Number.isFinite(reportedHeight)) return;
      const height = Math.min(50000, Math.max(96, Math.ceil(reportedHeight)));
      setEpubSpineHeights((current) =>
        Math.abs((current[spineIndex] || 0) - height) < 4
          ? current
          : { ...current, [spineIndex]: height }
      );
    }
    window.addEventListener("message", handleEpubMessage);
    return () => window.removeEventListener("message", handleEpubMessage);
  }, [epubReader]);

  useEffect(() => {
    if (panel !== "reader" || !readerView.current) return;
    const root = readerView.current;
    const targets = Array.from(root.querySelectorAll<HTMLElement>("[data-reader-ordinal]"));
    if (!targets.length) return;
    const observer = new IntersectionObserver((entries) => {
      if (readerScrollLockOrdinal.current !== null) return;
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((left, right) => left.boundingClientRect.top - right.boundingClientRect.top)[0];
      const target = visible?.target as HTMLElement | undefined;
      const activeOrdinal = Number(target?.dataset.readerOrdinal);
      if (Number.isInteger(activeOrdinal)) setReaderOrdinal(activeOrdinal);
      const pageNumber = Number(target?.dataset.readerPage);
      if (Number.isInteger(pageNumber) && pageNumber > 0) setReaderCurrentPage(pageNumber);
    }, { root, rootMargin: "-64px 0px -70% 0px", threshold: 0 });
    targets.forEach((target) => observer.observe(target));
    return () => observer.disconnect();
  }, [epubReader, panel, readerPageTargets, readerSegments]);

  useEffect(() => {
    if (!activeJobId || !jobsDocumentId) return;
    let disposed = false;
    async function pollJob() {
      try {
        const jobItems = await api<Job[]>("/documents/" + jobsDocumentId + "/jobs");
        if (disposed) return;
        setJobs(jobItems);
        if (overview?.document.id === jobsDocumentId) {
          const [summary, issueItems, reviewItems] = await Promise.all([
            api<Overview>("/documents/" + jobsDocumentId + "/overview"),
            api<Issue[]>("/documents/" + jobsDocumentId + "/issues"),
            api<HumanReviewItem[]>("/documents/" + jobsDocumentId + "/human-review-queue"),
          ]);
          if (!disposed) { setOverview(summary); setIssues(issueItems); setHumanReviewQueue(reviewItems); }
        }
      } catch { return; }
    }
    const timer = window.setInterval(() => void pollJob(), 2000);
    return () => { disposed = true; window.clearInterval(timer); };
  }, [activeJobId, jobsDocumentId, overview?.document.id]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setTranslation(segmentTranslation(currentSegment)); setDraftDirty(false); setSaveState("idle");
      if (currentSegment) {
        void api<Candidate[]>(`/segments/${currentSegment.id}/candidates`).then(setCandidates).catch(() => setCandidates([]));
        if (overview) localStorage.setItem(`jieyi.ordinal.${overview.document.id}`, String(currentSegment.ordinal));
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, [currentSegment?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!draftDirty || !currentSegment) return;
    const timer = window.setTimeout(async () => {
      setSaveState("saving");
      try {
        const saved = await api<Segment>(`/segments/${currentSegment.id}/draft`, { method: "PATCH", body: JSON.stringify({ translation }) });
        setPage((current) => current ? { ...current, items: current.items.map((item) => item.id === saved.id ? saved : item) } : current);
        setDraftDirty(false); setSaveState("saved");
      } catch { setSaveState("error"); }
    }, 700);
    return () => window.clearTimeout(timer);
  }, [currentSegment, draftDirty, translation]);

  async function saveDraftImmediately() {
    if (!draftDirty || !currentSegment) return;
    setDraftDirty(false);
    setSaveState("saving");
    try {
      const saved = await api<Segment>(`/segments/${currentSegment.id}/draft`, {
        method: "PATCH",
        body: JSON.stringify({ translation }),
      });
      setPage((current) => current ? { ...current, items: current.items.map((item) => item.id === saved.id ? saved : item) } : current);
      setHumanReviewQueue((items) => items.filter((item) => item.segment_id !== saved.id));
      setSaveState("saved");
    } catch (error) {
      setDraftDirty(true);
      setSaveState("error");
      throw error;
    }
  }

  async function navigateTo(target: number) {
    if (!overview || target < 0 || target >= overview.segment_count) return;
    try { await saveDraftImmediately(); }
    catch { notify("当前译文保存失败，已停留在本段。"); return; }
    if (!page || target < page.offset || target >= page.offset + page.items.length) await loadPageAt(overview.document.id, target);
    else setOrdinal(target);
  }

  async function confirmCurrent() {
    if (!currentSegment || !translation.trim()) return;
    setDraftDirty(false);
    setSaveState("saving");
    try {
      const saved = await api<Segment>(`/segments/${currentSegment.id}/confirm`, { method: "PATCH", body: JSON.stringify({ translation, rationale: "在介译工作台人工确认" }) });
      setPage((current) => current ? { ...current, items: current.items.map((item) => item.id === saved.id ? saved : item) } : current);
      setHumanReviewQueue((items) => items.filter((item) => item.segment_id !== saved.id));
      setOverview((current) => current ? { ...current, confirmed_count: current.confirmed_count + (currentSegment.status === "human_confirmed" ? 0 : 1), translated_count: current.translated_count + (segmentTranslation(currentSegment) ? 0 : 1) } : current);
      setDraftDirty(false); setSaveState("saved"); notify(`第 ${currentSegment.ordinal + 1} 段已确认并写入翻译记忆`);
    } catch (error) { setSaveState("error"); notify(error instanceof Error ? error.message : "确认失败"); }
  }

  async function performSearch(event: FormEvent) {
    event.preventDefault(); if (!overview || !searchQuery.trim()) return; setSearching(true);
    try {
      try { setSearchResults(await api<Segment[]>(`/documents/${overview.document.id}/search?q=${encodeURIComponent(searchQuery.trim())}`)); }
      catch {
        const query = searchQuery.trim().toLocaleLowerCase();
        const items = await loadAllSegments(overview.document.id);
        setSearchResults(items.filter((item) => `${item.source_text}\n${segmentTranslation(item)}`.toLocaleLowerCase().includes(query)).slice(0, 30));
      }
    }
    finally { setSearching(false); }
  }

  async function addTerm(event: FormEvent) {
    event.preventDefault(); if (!overview || !termSource.trim() || !termTarget.trim()) return;
    try {
      const splitList = (value: string) => value.split(/[,，;；\n]/).map((item) => item.trim()).filter(Boolean);
      const term = await api<Term>(`/projects/${overview.project.id}/terms`, { method: "POST", body: JSON.stringify({
        source: termSource, target: termTarget, status: "approved",
        aliases: splitList(termAliases), context_keywords: splitList(termContext),
        sense: termSense, disambiguation: termDisambiguation,
      }) });
      setTerms((items) => [...items, term].sort((a, b) => a.source.localeCompare(b.source)));
      setTermSource(""); setTermTarget(""); setTermAliases(""); setTermContext(""); setTermSense(""); setTermDisambiguation("");
      notify("术语约束已加入当前项目，并会在翻译与校对中自动检查。");
    } catch (error) { notify(error instanceof Error ? error.message : "添加术语失败"); }
  }

  async function handleImported(result: { projectId: string; documentId: string; title: string }) {
    const projectItems = await api<Project[]>("/projects");
    const grouped = await Promise.all(projectItems.map((project) => api<Document[]>(`/projects/${project.id}/documents`)));
    const documentItems = grouped.flat(); setProjects(projectItems); setDocuments(documentItems);
    const document = documentItems.find((item) => item.id === result.documentId); if (document) await openDocument(document, 0);
  }

  async function selectLibraryDocument(document: Document) {
    setSelectedDocumentId(document.id);
    try {
      setJobs(await api<Job[]>("/documents/" + document.id + "/jobs"));
      setJobsDocumentId(document.id);
    } catch { setJobs([]); setJobsDocumentId(document.id); }
  }

  async function openReader(document: Document) {
    setReaderLoading(true);
    setPanel("reader");
    setSelectedDocumentId(document.id);
    setEpubSpineHeights({});
    setReaderPageTargets({});
    setReaderCurrentPage(1);
    setEpubReader(null);
    setReaderSegments([]);
    readerPendingOrdinal.current = null;
    readerScrollLockOrdinal.current = null;
    localStorage.setItem("jieyi.document", document.id);
    try {
      const summaryPromise = overview?.document.id === document.id
        ? Promise.resolve(overview)
        : api<Overview>("/documents/" + document.id + "/overview");
      if (document.source_format === "epub") {
        const [summary, manifest] = await Promise.all([
          summaryPromise,
          api<EpubReaderManifest>("/documents/" + document.id + "/epub"),
        ]);
        setOverview(summary);
        const startOrdinal = summary.chapters[0]?.start_ordinal || 0;
        setReaderOrdinal(startOrdinal);
        const pageBySpine = new Map(
          manifest.spine.filter((item) => item.linear).map((item, index) => [item.spine_index, index + 1])
        );
        const targets: Record<number, ReaderPageTarget> = {};
        for (const location of manifest.segment_locations) {
          const pageNumber = pageBySpine.get(location.spine_index);
          if (pageNumber === undefined || targets[location.segment_ordinal]) continue;
          targets[location.segment_ordinal] = { spineIndex: location.spine_index, pageNumber };
        }
        setReaderPageTargets(targets);
        setEpubReader(manifest);
      } else {
        const [summary, items] = await Promise.all([
          summaryPromise,
          api<Segment[]>("/documents/" + document.id + "/segments"),
        ]);
        legacySegments.current.set(document.id, items);
        setOverview(summary);
        setReaderSegments(items);
        setReaderOrdinal(items[0]?.ordinal || 0);
      }
    } catch (error) {
      notify(error instanceof Error ? error.message : "无法打开阅读模式");
    } finally { setReaderLoading(false); }
  }

  function jumpToReaderChapter(chapter: Chapter) {
    setReaderOrdinal(chapter.start_ordinal);
    const pageTarget = readerPageTargets[chapter.start_ordinal];
    const targetId = pageTarget
      ? `reader-spine-${pageTarget.spineIndex}`
      : `reader-segment-${chapter.start_ordinal}`;
    const target = document.getElementById(targetId);
    if (!target) { notify("暂时无法定位这一节在原书中的页码。"); return; }
    if (!pageTarget) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
    setReaderCurrentPage(pageTarget.pageNumber);
    const frame = target.querySelector<HTMLIFrameElement>("iframe");
    const fixedLayout = epubReader?.spine.find((item) => item.spine_index === pageTarget.spineIndex)?.fixed_layout;
    if (!frame || fixedLayout) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
    readerPendingOrdinal.current = chapter.start_ordinal;
    readerScrollLockOrdinal.current = chapter.start_ordinal;
    if (frame.dataset.loaded === "true") {
      requestReaderLocation(frame, chapter.start_ordinal);
    } else {
      target.scrollIntoView({ behavior: "auto", block: "start" });
    }
    window.setTimeout(() => {
      if (readerPendingOrdinal.current === chapter.start_ordinal) {
        readerPendingOrdinal.current = null;
        if (readerScrollLockOrdinal.current === chapter.start_ordinal) readerScrollLockOrdinal.current = null;
      }
    }, 3000);
  }

  function requestReaderLocation(frame: HTMLIFrameElement, segmentOrdinal: number) {
    const pageTarget = readerPageTargets[segmentOrdinal];
    if (!epubReader || !pageTarget) return;
    frame.contentWindow?.postMessage({
      type: "jy-epub-locate",
      documentId: epubReader.document_id,
      spineIndex: pageTarget.spineIndex,
      segmentOrdinal,
    }, "*");
  }

  function readerOrdinalForSpine(spineIndex: number) {
    const entry = Object.entries(readerPageTargets).find(([, target]) => target.spineIndex === spineIndex);
    return Number(entry?.[0] || 0);
  }

  function changeReaderMode(value: ReaderMode) {
    setEpubSpineHeights({});
    setReaderMode(value);
    localStorage.setItem("jieyi.reader.mode", value);
  }

  function changeReaderLayout(value: ReaderLayout) {
    setEpubSpineHeights({});
    setReaderLayout(value);
    localStorage.setItem("jieyi.reader.layout", value);
  }

  function changeDark() {
    setDark((value) => {
      const next = !value;
      localStorage.setItem("jieyi.theme", next ? "dark" : "light");
      return next;
    });
  }

  async function openDraftPicker(document: Document) {
    setDraftPickerLoading(true);
    try {
      const summary = overview?.document.id === document.id
        ? overview
        : await api<Overview>("/documents/" + document.id + "/overview");
      const remaining = summary.chapters.filter((chapter) => chapter.translated_count < chapter.segment_count);
      setDraftPicker({ document, overview: summary });
      setSelectedChapterStarts([...new Set((remaining.length ? remaining : summary.chapters).map((chapter) => chapter.start_ordinal))]);
    } catch (error) {
      notify(error instanceof Error ? error.message : "无法读取目录");
    } finally { setDraftPickerLoading(false); }
  }

  function toggleDraftChapter(startOrdinal: number) {
    setSelectedChapterStarts((current) => current.includes(startOrdinal)
      ? current.filter((item) => item !== startOrdinal)
      : [...current, startOrdinal]);
  }

  async function runWholeBook(document: Document, mode: "draft" | "review", segmentRanges: [number, number][] = []) {
    if (wholeBookAction) return;
    const normalizedRanges = normalizeSegmentRanges(segmentRanges);
    const providerSettings = settings || await api<ProviderSettings>("/settings/provider").catch(() => null);
    if (!providerSettings) { setPanel("settings"); notify("请先配置并测试草译模型。"); return; }
    const bookConfig = settingsForBook(document.id, providerSettings);
    const draftProfile = providerSettings.profiles.find((item) => item.id === bookConfig.draft_profile_id);
    const reviewerProfile = providerSettings.profiles.find((item) => item.id === bookConfig.reviewer_profile_id);
    if (!draftProfile?.base_url || !bookConfig.draft_model) { openBookSettings(document); notify("请先为这本书选择草译连接与模型。"); return; }
    if (mode === "review" && (!reviewerProfile?.base_url || !bookConfig.reviewer_model)) { openBookSettings(document); notify("审校前，请先为这本书选择审校连接与模型。"); return; }
    const currentSettings: ProviderSettings = {
      ...providerSettings,
      base_url: draftProfile.base_url,
      draft_profile_id: bookConfig.draft_profile_id,
      draft_provider: `profile:${bookConfig.draft_profile_id}`,
      draft_model: bookConfig.draft_model,
      draft_compute_mode: bookConfig.draft_compute_mode,
      reviewer_profile_id: bookConfig.reviewer_profile_id,
      reviewer_provider: `profile:${bookConfig.reviewer_profile_id}`,
      reviewer_model: bookConfig.reviewer_model,
      reviewer_compute_mode: bookConfig.reviewer_compute_mode,
      review_enabled: Boolean(bookConfig.reviewer_model),
    };
    setWholeBookAction(mode);
    try {
      const existingJobs = await api<Job[]>("/documents/" + document.id + "/jobs");
      let job = existingJobs.find((item) => ["pending", "running", "paused", "failed"].includes(item.status) && jobMatchesSettings(item, currentSettings, mode, normalizedRanges));
      if (!job) {
        job = await api<Job>("/documents/" + document.id + "/jobs", { method: "POST", body: JSON.stringify({
          draft_provider: currentSettings.draft_provider || "openai-compatible", draft_model: currentSettings.draft_model, task_mode: mode,
          reviewer_provider: mode === "review" ? (currentSettings.reviewer_provider || currentSettings.draft_provider || "openai-compatible") : null,
          reviewer_model: mode === "review" ? currentSettings.reviewer_model : null,
          review_policy: mode === "review" ? "all" : "never",
          batch_size: 10, concurrency: 3, max_concurrency: 5, max_batch_chars: 4000,
          draft_thinking: currentSettings.draft_compute_mode !== "economy", review_thinking: currentSettings.reviewer_compute_mode !== "economy",
          draft_compute_mode: currentSettings.draft_compute_mode, review_compute_mode: currentSettings.reviewer_compute_mode, review_sample_rate: 0,
          max_output_tokens: 6000, token_budget: 2000000, segment_ranges: normalizedRanges,
        }) });
      }
      const started = await api<Job>("/jobs/" + job.id + "/start", { method: "POST" });
      setJobs([started, ...existingJobs.filter((item) => item.id !== started.id)]);
      setJobsDocumentId(document.id);
      if (mode === "draft") setDraftPicker(null);
      notify(mode === "draft"
        ? normalizedRanges.length ? "已启动所选范围的草译，共 " + normalizedRanges.reduce((total, range) => total + range[1] - range[0] + 1, 0) + " 段；已有译文的段落会自动跳过。" : "全书草译已在后台启动；已有译文的段落会自动跳过。"
        : "全量独立审校已在后台启动；审校 AI 会逐段对照原文，人工抽检队列将在完成后生成。");
    } catch (error) { notify(error instanceof Error ? error.message : mode === "draft" ? "全书草译启动失败" : "全书审校启动失败"); }
    finally { setWholeBookAction(null); }
  }

  async function controlJob(job: Job, action: "pause" | "cancel") {
    try {
      const updated = await api<Job>("/jobs/" + job.id + "/" + action, { method: "POST" });
      setJobs((items) => items.map((item) => item.id === updated.id ? updated : item));
      notify(action === "pause" ? "任务已暂停，进度与译文均已保存。" : "任务已取消，已有译文仍会保留。");
    } catch (error) { notify(error instanceof Error ? error.message : "任务控制失败"); }
  }

  async function deleteSelectedDocument() {
    if (!pendingDelete || deleting) return;
    setDeleting(true);
    try {
      await api<void>("/documents/" + pendingDelete.id, { method: "DELETE" });
      const remaining = documents.filter((item) => item.id !== pendingDelete.id);
      setDocuments(remaining);
      setSelectedDocumentId(remaining[0]?.id || "");
      legacySegments.current.delete(pendingDelete.id);
      localStorage.removeItem("jieyi.ordinal." + pendingDelete.id);
      localStorage.removeItem("jieyi.book-settings." + pendingDelete.id);
      setBookSettings((current) => { const next = { ...current }; delete next[pendingDelete.id]; return next; });
      if (overview?.document.id === pendingDelete.id) {
        setOverview(null); setPage(null); setTerms([]); setIssues([]); setJobs([]); setReaderSegments([]);
        localStorage.removeItem("jieyi.document");
      }
      setPendingDelete(null); setPanel("library"); notify("《" + pendingDelete.title + "》已从书库删除。");
    } catch (error) { notify(error instanceof Error ? error.message : "删除失败"); }
    finally { setDeleting(false); }
  }

  useEffect(() => {
    function keydown(event: KeyboardEvent) {
      const command = event.metaKey || event.ctrlKey;
      if (command && event.key.toLowerCase() === "k") { event.preventDefault(); setSearchOpen(true); window.setTimeout(() => searchInput.current?.focus(), 30); }
      if (command && event.key === "Enter" && panel === "translate") { event.preventDefault(); void confirmCurrent(); }
      if (event.key === "Escape") setSearchOpen(false);
    }
    window.addEventListener("keydown", keydown); return () => window.removeEventListener("keydown", keydown);
  });

  const currentStatus = currentSegment?.status || "source";
  const activeDocument = overview?.document;

  return <main className={`desktop-shell ${dark ? "dark" : ""}`}>
    <section className={`workspace-window ${!sidebarOpen ? "sidebar-closed" : ""} ${!inspectorOpen || panel !== "translate" ? "inspector-closed" : ""}`} aria-label="介译翻译工作台">
      <header className="titlebar">
        <div className="titlebar-start"><div className="traffic-lights" aria-hidden="true"><span className="traffic red" /><span className="traffic yellow" /><span className="traffic green" /></div><button className="icon-button sidebar-toggle" aria-label="切换侧栏" onClick={() => setSidebarOpen((value) => !value)}><Mark>▥</Mark></button></div>
        <div className="document-title"><strong>{panel === "library" ? "我的书库" : panel === "import" ? "导入书籍" : panel === "settings" ? "模型配置" : activeDocument?.title || "介译"}</strong><span>{panel === "translate" ? `${currentChapter?.title || "正文"} · ${saveState === "saving" ? "正在保存" : saveState === "error" ? "保存失败" : "已保存"}` : panel === "reader" ? `原书排版阅读 · ${readerMode === "original" ? "原版" : readerMode === "translated" ? "仅译文" : "双语对照"}` : overview?.project.name || "本地翻译工作台"}</span></div>
        <div className="titlebar-actions"><button className="icon-button" aria-label="全文搜索" onClick={() => setSearchOpen(true)}><Mark>⌕</Mark><kbd>⌘K</kbd></button><button className="icon-button theme-toggle" title={dark ? "切换到日间模式" : "切换到夜览模式"} aria-label={dark ? "切换到日间模式" : "切换到夜览模式"} aria-pressed={dark} onClick={changeDark}><ThemeIcon dark={dark} /></button>{panel === "translate" && <button className={"primary-button " + (latestJob?.status === "running" ? "running" : "")} onClick={() => activeDocument && void openDraftPicker(activeDocument)} disabled={!activeDocument || wholeBookAction !== null || draftPickerLoading || latestJob?.status === "running"}>{draftPickerLoading ? "正在读取目录…" : latestJob?.status === "running" ? "篇章草译运行中" : "选择篇章草译"}</button>}</div>
        {latestJob?.status === "running" && <div className="title-progress indeterminate" />}
      </header>

      <div className="workspace-grid">
        <aside className="sidebar"><div className="sidebar-scroll"><div className="eyebrow">工作空间</div><nav aria-label="工作空间">
          <button className={`nav-item ${panel === "library" ? "active" : ""}`} onClick={() => setPanel("library")}><Mark><LibraryIcon /></Mark>书库 <b>{documents.length}</b></button>
          <button className={`nav-item ${panel === "translate" ? "active" : ""}`} disabled={!overview} onClick={() => { if (panel === "reader" && overview?.document) void openDocument(overview.document, readerOrdinal); else setPanel("translate"); }}><Mark><TranslationIcon /></Mark>翻译</button>
          <button className={"nav-item " + (panel === "reader" ? "active" : "")} disabled={!overview} onClick={() => overview?.document && void openReader(overview.document)}><Mark>☷</Mark>阅读</button>
          <button className={`nav-item ${panel === "quality" ? "active" : ""}`} disabled={!overview} onClick={() => setPanel("quality")}><Mark>◉</Mark>审校 {errorIssues.length > 0 && <b className="issue-count">{errorIssues.length}</b>}</button>
          <button className={`nav-item ${panel === "terms" ? "active" : ""}`} disabled={!overview} onClick={() => setPanel("terms")}><Mark>⌁</Mark>术语库 <b>{terms.length}</b></button>
          <div className="nav-separator" /><button className={`nav-item ${panel === "import" ? "active" : ""}`} onClick={() => setPanel("import")}><Mark>⇧</Mark>导入书籍</button><button className={`nav-item ${panel === "settings" ? "active" : ""}`} onClick={() => setPanel("settings")}><Mark>⚙</Mark>模型配置</button>
        </nav>
        {overview && <><div className="eyebrow section-label">
          <span>目录</span><small>{overview.chapters.length} 节</small></div><div className="chapter-list real-chapters">{overview.chapters.map((chapter, index) => <button key={`${chapter.start_ordinal}-${chapter.level}-${chapter.title}`} data-level={chapter.level} style={{ paddingLeft: `${9 + Math.min(chapter.level, 3) * 12}px` }} className={`chapter ${currentChapter?.start_ordinal === chapter.start_ordinal ? "active" : ""}`}
          onClick={() => { if (panel === "reader") jumpToReaderChapter(chapter); else void navigateTo(chapter.start_ordinal); }}
          title={panel === "reader" && readerPageTargets[chapter.start_ordinal] ? `${chapter.title} · 第 ${readerPageTargets[chapter.start_ordinal].pageNumber} 页` : chapter.title}><span>{String(index + 1).padStart(2, "0")}</span>
          <div><strong>{chapter.title}</strong>
            <small>{panel === "reader" && readerPageTargets[chapter.start_ordinal] ? `第 ${readerPageTargets[chapter.start_ordinal].pageNumber} 页 · ${chapter.segment_count} 段` : `${chapter.confirmed_count} 已确认 · ${chapter.segment_count} 段`}</small>
          </div>
        </button>)}</div></>}
        </div>{overview && <div className="project-progress"><div><span>人工确认</span><strong>{completion}%</strong></div><div className="progress-track"><span style={{ width: `${completion}%` }} /></div><small>草译 {overview.translated_count} / {overview.segment_count} · {translationProgress}%</small></div>}</aside>

        {panel === "library" && <section className="library-view book-library">
          {selectedDocument && <div className="library-quick-actions"><span>{settings ? <>本书草译模型：<strong>{settingsForBook(selectedDocument.id, settings).draft_model || "未选择"}</strong> · 可与其他书使用不同模型同时运行</> : "当前书籍可单独配置翻译风格与模型。"}</span><button onClick={() => openBookSettings(selectedDocument)}>翻译设置</button><a href={`${API_BASE}/documents/${selectedDocument.id}/export?format=book`} download={exportName(selectedDocument)}>一键导出译后书籍</a></div>}
          <div className="library-header"><div><span className="page-kicker">本机项目</span><h1>我的书库</h1><p>{documents.length ? "共 " + projects.length + " 个项目、" + documents.length + " 本书或文档。" : "导入第一本书，开始可追溯的长文本翻译。"}</p></div><button className="primary-button" onClick={() => setPanel("import")}>＋ 导入书籍</button></div>
          {libraryError && <div className="library-error"><i>!</i><span><strong>书籍打开失败</strong><small>{libraryError}</small></span><button onClick={() => void loadLibrary()}>重新连接</button></div>}
          {selectedDocument && <><div className="book-actions" aria-label="所选书目操作"><div><span>已选中</span><strong>《{selectedDocument.title}》</strong></div><button onClick={() => void openDocument(selectedDocument)}>打开工作台</button><button onClick={() => void openReader(selectedDocument)} disabled={readerLoading}>{readerLoading ? "正在载入…" : "阅读整本"}</button><button className="accent" onClick={() => void openDraftPicker(selectedDocument)} disabled={wholeBookAction !== null || draftPickerLoading || selectedDraftJob?.status === "running"}>{draftPickerLoading ? "正在读取目录…" : selectedDraftJob?.status === "running" ? "草译运行中" : "选择篇章草译"}</button><button onClick={() => void runWholeBook(selectedDocument, "review")} disabled={wholeBookAction !== null || selectedReviewJob?.status === "running"}>{wholeBookAction === "review" ? "正在启动…" : selectedReviewJob?.status === "paused" || selectedReviewJob?.status === "failed" ? "继续全量审校" : selectedReviewJob?.status === "running" ? "审校运行中" : "全量独立审校"}</button><button className="danger" onClick={() => setPendingDelete(selectedDocument)}>删除</button></div>{selectedJob && <div className={"job-progress-card status-" + selectedJob.status}><div className="job-progress-head"><div><span>{selectedJob.recipe.task_mode === "review" ? "全量独立审校" : selectedJob.recipe.segment_ranges?.length ? "篇章草译" : "全书草译"}</span><strong>{jobStatusLabel(selectedJob.status)}</strong></div><small>{selectedJob.processed_segments ?? selectedJob.next_ordinal} / {selectedJob.total_segments ?? "—"} 段</small></div><div className="job-progress-track"><span style={{ width: String(Math.min(100, Math.round(((selectedJob.processed_segments ?? selectedJob.next_ordinal) / Math.max(1, selectedJob.total_segments || 1)) * 100))) + "%" }} /></div><div className="job-metrics"><span><b>{selectedJob.batch_count || 0}</b> 批</span><span><b>{formatTokens(selectedJob.total_tokens)}</b> token</span><span><b>{formatTokens(selectedJob.reasoning_tokens)}</b> 推理</span>{Boolean(selectedJob.deferred_segments) && <span><b>{selectedJob.deferred_segments}</b> 隔离待复核</span>}<span><b>{formatDuration(selectedJob.eta_seconds)}</b> 预计剩余</span></div><p>{selectedJob.recipe.task_mode === "review" ? "审校 AI 逐段独立对照原文；清晰问题直接修订，未解决的确定性风险进入人工必检队列。" : `单段隔离、${selectedJob.recipe.concurrency || 3}→${selectedJob.recipe.max_concurrency || selectedJob.recipe.concurrency || 3} 路自适应并发；异常段不会覆盖正文。`}</p>{selectedJob.last_error && <div className="job-progress-error">{selectedJob.last_error}</div>}<div className="job-controls">{selectedJob.status === "running" && <button onClick={() => void controlJob(selectedJob, "pause")}>暂停</button>}{(selectedJob.status === "paused" || selectedJob.status === "failed") && selectedDocument && <button className="accent" onClick={() => void runWholeBook(selectedDocument, selectedJob.recipe.task_mode === "review" ? "review" : "draft", selectedJob.recipe.segment_ranges || [])}>继续</button>}{!["completed", "cancelled"].includes(selectedJob.status) && <button className="danger" onClick={() => void controlJob(selectedJob, "cancel")}>取消任务</button>}</div></div>}</>}
          {loading ? <div className="empty-workspace">正在读取书库…</div> : documents.length === 0 ? <div className="empty-workspace"><Mark>▦</Mark><h2>书库还是空的</h2><p>支持 EPUB、TXT 与 Markdown，原文保存在本机数据库。</p><button className="blue-action" onClick={() => setPanel("import")}>导入第一本书</button></div> : <div className="book-grid">{documents.map((document) => { const project = projects.find((item) => item.id === document.project_id); return <button key={document.id} className={"book-card " + (selectedDocumentId === document.id ? "active" : "")} aria-pressed={selectedDocumentId === document.id} onClick={() => void selectLibraryDocument(document)}><i className={document.cover_url ? "has-cover" : ""}>{document.cover_url ? <img src={API_BASE + document.cover_url} alt={`《${document.title}》封面`} loading="lazy" referrerPolicy="no-referrer" /> : document.source_format.toUpperCase()}</i><div><span>{project?.name}</span><strong>{document.title}</strong><small>{project?.source_lang} → {project?.target_lang} · 导入于 {new Date(document.created_at).toLocaleDateString()}</small></div><Mark>{selectedDocumentId === document.id ? "✓" : "›"}</Mark></button>; })}</div>}
        </section>}

        {panel === "translate" && <section className="editor">{!currentSegment ? <div className="empty-workspace">{loading ? "正在加载书稿…" : "请选择一本书。"}</div> : <><div className="editor-toolbar"><div className="segment-nav"><button aria-label="上一段" disabled={ordinal === 0} onClick={() => void navigateTo(ordinal - 1)}>‹</button><span><i className={`status-dot ${currentStatus === "human_confirmed" ? "confirmed" : currentStatus === "machine_translated" ? "draft" : "source"}`} /> 第 {ordinal + 1} 段，共 {overview?.segment_count || page?.total} 段</span><button aria-label="下一段" disabled={ordinal + 1 >= (overview?.segment_count || 0)} onClick={() => void navigateTo(ordinal + 1)}>›</button><label className="ordinal-jump">跳至<input aria-label="段落编号" type="number" min={1} max={overview?.segment_count} value={ordinal + 1} onChange={(event) => void navigateTo(Number(event.target.value) - 1)} /></label></div><div className="editor-tools"><a className="export-link" href={`${API_BASE}/documents/${overview?.document.id}/export?bilingual=true`} download={`${overview?.document.title || "jieyi"}-双语.txt`}>导出双语</a><div className="segmented" aria-label="编辑器视图"><button className={view === "split" ? "active" : ""} onClick={() => setView("split")}>对照</button><button className={view === "target" ? "active" : ""} onClick={() => setView("target")}>译文</button></div><button className={`icon-button inspector-toggle ${inspectorOpen ? "active" : ""}`} aria-label="切换检查器" onClick={() => setInspectorOpen((value) => !value)}><Mark>▧</Mark></button></div></div>
          <div className={`editor-columns ${view === "target" ? "target-only" : ""}`}>{view === "split" && <article className={`source-pane content-${currentSegment.kind}`}><div className="column-label"><span>原文 · {overview?.project.source_lang.toUpperCase()}</span><button onClick={() => navigator.clipboard?.writeText(currentSegment.source_text).then(() => notify("原文已复制"))}>复制</button></div>{currentSegment.kind === "heading" ? <h2>{currentSegment.source_text}</h2> : <p>{currentSegment.source_text}</p>}<div className="context-note"><Mark>¶</Mark><div><strong>结构位置</strong><span>{currentSegment.heading_path || "正文"}</span></div></div></article>}<article className="translation-pane"><div className="column-label"><span>译文 · {overview?.project.target_lang}</span><div className={`draft-state ${currentStatus === "human_confirmed" ? "confirmed" : ""}`}><i />{statusLabel(currentSegment)}</div></div><textarea aria-label="译文编辑器" value={translation} onChange={(event) => { setTranslation(event.target.value); setDraftDirty(true); setSaveState("idle"); }} placeholder={currentStatus === "source" ? "等待模型草译，或直接输入人工译文…" : "编辑译文…"} spellCheck={false} /><div className="editor-footer"><span>{translation.length} 字 · {saveState === "saving" ? "正在保存" : saveState === "error" ? "保存失败" : draftDirty ? "待保存" : "已保存"}</span><button className="confirm-button" onClick={() => void confirmCurrent()} disabled={!translation.trim() || saveState === "saving"}>{currentStatus === "human_confirmed" ? "更新确认" : "确认译文"}<kbd>⌘↵</kbd></button></div></article></div></>}</section>}

        {panel === "reader" && <section ref={readerView} className="reader-view" aria-label="原书排版阅读模式">
          <div className="reader-toolbar"><div>
            <span>{epubReader ? `第 ${readerCurrentPage} / ${epubReader.spine.filter((item) => item.linear).length} 页` : "全文阅读"}</span>
            <strong>{epubReader ? `${epubReader.spine.filter((item) => item.linear).length} 个书脊文档` : `${readerSegments.length} 段`} · {overview?.translated_count || 0} 段已有译文</strong></div>
            <div className="reader-toolbar-actions">
              <button onClick={() => overview?.document && void openDocument(overview.document, readerOrdinal)}>返回工作台</button>{epubReader?.spine.some((item) => item.fixed_layout) && <div className="segmented" aria-label="定版 EPUB 排版策略"><button className={readerLayout === "faithful" ? "active" : ""} onClick={() => changeReaderLayout("faithful")}>忠实排版</button><button className={readerLayout === "comfort" ? "active" : ""} onClick={() => changeReaderLayout("comfort")}>舒适阅读</button></div>}<div className="segmented" aria-label="阅读语言显示"><button className={readerMode === "original" ? "active" : ""} onClick={() => changeReaderMode("original")}>原版</button><button className={readerMode === "translated" ? "active" : ""} onClick={() => changeReaderMode("translated")}>仅译文</button><button className={readerMode === "bilingual" ? "active" : ""} onClick={() => changeReaderMode("bilingual")}>双语对照</button></div></div></div>
          {readerLoading ? <div className="empty-workspace">正在准备原书资源…</div> : epubReader ? <div className={`epub-reader-stack layout-${readerLayout}`}>{epubReader.cover_url && <section className="epub-cover-sheet"><img src={API_BASE + epubReader.cover_url} alt={`《${overview?.document.title || "EPUB"}》封面`} /></section>}
            {epubReader.spine.filter((item) => item.linear).map((item, index) =>
              <section id={`reader-spine-${item.spine_index}`} data-reader-ordinal={readerOrdinalForSpine(item.spine_index)} data-reader-page={index + 1} className={`epub-spine-sheet ${item.fixed_layout ? "fixed-layout" : "reflowable"}`} key={item.spine_index}>
                <iframe title={`${overview?.document.title || "EPUB"} · 第 ${index + 1} 页 · ${item.path}`}
                  key={`${item.spine_index}-${readerMode}-${readerLayout}`}
                  data-spine-index={item.spine_index}
                  loading={index < 2 ? "eager" : "lazy"}
                  onLoad={(event) => {
                    const frame = event.currentTarget;
                    frame.dataset.loaded = "true";
                    const pendingOrdinal = readerPendingOrdinal.current;
                    if (pendingOrdinal !== null && readerPageTargets[pendingOrdinal]?.spineIndex === item.spine_index) requestReaderLocation(frame, pendingOrdinal);
                  }}
                  sandbox="allow-scripts"
                  referrerPolicy="no-referrer"
                  style={!item.fixed_layout ? { height: `${epubSpineHeights[item.spine_index] || 96}px` } : undefined}
                  src={`${API_BASE}/documents/${overview?.document.id}/epub/spine/${item.spine_index}?mode=${readerMode}&layout=${readerLayout}`} />
              </section>)}
            </div> : <article className={"reader-paper " + (readerMode === "bilingual" ? "bilingual" : "target-only")}><header><span>{overview?.project.name}</span><h1>{overview?.document.title}</h1><p>{overview?.project.source_lang.toUpperCase()} → {overview?.project.target_lang} · 连贯阅读</p></header>{readerSegments.map((segment) => { const target = segmentTranslation(segment); return <section
              id={"reader-segment-" + segment.ordinal} data-reader-ordinal={segment.ordinal} className={"reader-block reader-" + segment.kind} key={segment.id}>{readerMode !== "translated" && <div className="reader-source"><small>原文</small>{segment.kind === "heading" ? <h2>{segment.source_text}</h2> : <p>{segment.source_text}</p>}</div>}{readerMode !== "original" && <div className={"reader-target " + (!target ? "missing" : "")}><small>{segment.status === "human_confirmed" ? "译文 · 已确认" : "译文"}</small>{segment.kind === "heading" ? <h2>{target || "本节尚未草译"}</h2> : <p>{target || "本段尚未草译"}</p>}</div>}</section>; })}</article>}
        </section>}

        {panel === "terms" && <section className="library-view"><div className="library-header"><div><span className="page-kicker">翻译约束</span><h1>术语库</h1><p>当前项目共有 {terms.length} 条术语约束；系统会主动识别源词和同义词，在生成与校对阶段统一译法。</p></div></div><TermDiscoveryPanel documentId={overview.document.id} provider={settings?.draft_provider || ""} model={settings?.draft_model || ""} computeMode={settings?.draft_compute_mode || "balanced"} onApproved={(term) => setTerms((items) => [...items, term].sort((left, right) => left.source.localeCompare(right.source)))} notify={notify} /><form className="term-create" onSubmit={(event) => void addTerm(event)}><label><span>规范源词</span><input value={termSource} onChange={(event) => setTermSource(event.target.value)} placeholder="例如 bank" /></label><label><span>强制译法</span><input value={termTarget} onChange={(event) => setTermTarget(event.target.value)} placeholder="例如 银行" /></label><label><span>同义词 / 变体</span><input value={termAliases} onChange={(event) => setTermAliases(event.target.value)} placeholder="逗号分隔，例如 banking institution" /></label><label><span>语境关键词</span><input value={termContext} onChange={(event) => setTermContext(event.target.value)} placeholder="用于消歧，例如 loan, credit" /></label><label><span>义项</span><input value={termSense} onChange={(event) => setTermSense(event.target.value)} placeholder="例如 金融机构" /></label><label><span>消歧说明</span><input value={termDisambiguation} onChange={(event) => setTermDisambiguation(event.target.value)} placeholder="说明何时采用此译法" /></label><button className="primary-button" disabled={!termSource.trim() || !termTarget.trim()}>添加术语约束</button></form><div className="term-table" role="table" aria-label="术语列表"><div className="term-table-head" role="row"><span>源词与同义词</span><span>强制译法 / 义项</span><span>语境触发</span><span>状态</span></div>{terms.map((term) => <div className="term-table-row" role="row" key={term.id}><div className="term-cell"><strong>{term.source}</strong>{term.aliases?.length > 0 && <small>{term.aliases.join(" · ")}</small>}</div><div className="term-cell"><span>{term.target}</span>{term.sense && <small>{term.sense}</small>}</div><small>{term.context_keywords?.length ? term.context_keywords.join(" · ") : "默认义项"}</small><span className={`term-status ${term.status !== "approved" ? "pending" : ""}`}>{term.status === "approved" ? "强制执行" : term.status}</span></div>)}</div>{terms.length === 0 && <div className="empty-inline">还没有术语约束。添加后，命中的源词会直接参与译文生成和质量检查。</div>}</section>}

        {panel === "quality" && <section className="library-view quality-view">
          <div className="library-header"><div><span className="page-kicker">第三阶段 · 人工把关</span><h1>审校</h1><p>{requiredHumanChecks ? `${requiredHumanChecks} 段因未解决问题必须人工检查；另有 ${humanReviewQueue.length - requiredHumanChecks} 段来自约 8% 的全书分布式抽样。` : humanReviewQueue.length ? `没有遗留确定问题；请抽检 ${humanReviewQueue.length} 个已审校段落。` : overview?.reviewed_count ? "当前人工抽检队列已处理完成。" : "请先完成草译和全量独立审校。"}</p></div><div className="large-score"><span>{humanReviewQueue.length}</span><small>待人工检查</small></div></div>
          <div className="quality-summary"><div><strong>{overview?.reviewed_count || 0}</strong><span>AI 已审校</span></div><div><strong>{requiredHumanChecks}</strong><span>问题必检</span></div><div><strong>{humanReviewQueue.length - requiredHumanChecks}</strong><span>分布抽检</span></div><div><strong>{overview?.confirmed_count || 0}</strong><span>人工确认</span></div></div>
          <div className="issue-list">{humanReviewQueue.map((item) => <button key={item.segment_id} onClick={() => { setPanel("translate"); void navigateTo(item.ordinal); }}><i className={item.reason === "error" ? "warning-icon" : "format-icon"}>{item.reason === "error" ? "!" : item.reason === "warning" ? "?" : "抽"}</i><div><strong>{item.reason === "error" ? "确定问题 · 必检" : item.reason === "warning" ? "建议复核 · 必检" : "全书分布抽检"}</strong><span>第 {item.ordinal + 1} 段 · {item.source_text.slice(0, 90)}</span></div><Mark>›</Mark></button>)}</div>
          {humanReviewQueue.length === 0 && <div className="empty-inline">{overview?.reviewed_count ? "抽检项已全部人工确认；仍可从工作台逐段终审。" : "完成全量独立审校后，这里会自动生成问题必检与约 8% 的抽检队列。"}</div>}
          {displayIssues.length > 0 && <div className="issue-list">{displayIssues.map((issue) => <button key={issue.id} onClick={() => { setPanel("translate"); void navigateTo(issue.ordinal); }}><i className={issue.severity === "error" ? "warning-icon" : "format-icon"}>{issue.severity === "error" ? "!" : "?"}</i><div><strong>自动检查 · {issue.code}</strong><span>第 {issue.ordinal + 1} 段 · {issue.message}</span></div><Mark>›</Mark></button>)}</div>}
        </section>}

        {panel === "import" && <ImportBookPanel onImported={(result) => void handleImported(result)} />}
        {panel === "settings" && <ProviderSettingsPanel onSaved={(value) => { setSettings(value); notify("模型配置已更新。"); }} />}

        <aside className="inspector"><div className="inspector-heading"><strong>检查器</strong><button aria-label="关闭检查器" onClick={() => setInspectorOpen(false)}>×</button></div><div className="inspector-scroll"><div className="quality-card"><div className={`segment-state-mark ${currentStatus}`}><Mark>{currentStatus === "human_confirmed" ? "✓" : currentSegment?.reviewed_translation ? "校" : currentStatus === "machine_translated" ? "AI" : "—"}</Mark></div><div><strong>{statusLabel(currentSegment)}</strong><small>{currentErrorIssues.length ? `${currentErrorIssues.length} 项确定问题` : currentWarningIssues.length ? `${currentWarningIssues.length} 项建议复核` : "本段没有质量问题"}</small></div></div>{currentIssues.length > 0 && <div className="inspector-section"><div className="eyebrow">问题</div>{currentIssues.map((issue) => <div className="issue-chip" key={issue.id}><i>{issue.severity === "error" ? "!" : "?"}</i><span>{issue.message}</span></div>)}</div>}<div className="inspector-section"><div className="eyebrow">本段术语 <button onClick={() => setPanel("terms")}>查看全部</button></div>{relevantTerms.length ? relevantTerms.map((term) => <div className="term-row" key={term.id}><span>{term.source}</span><strong>{term.target}</strong></div>) : <p className="inspector-empty">未命中已批准术语</p>}</div><div className="inspector-section"><div className="eyebrow">生成信息</div>{candidates.length ? <dl><div><dt>模型</dt><dd>{candidates.at(-1)?.model}</dd></div><div><dt>输入 / 输出</dt><dd>{candidates.at(-1)?.prompt_tokens} / {candidates.at(-1)?.completion_tokens}</dd></div><div><dt>费用</dt><dd>${(candidates.at(-1)?.cost_usd || 0).toFixed(6)}</dd></div></dl> : <p className="inspector-empty">本段还没有模型候选</p>}</div>{latestJob && <div className="inspector-section"><div className="eyebrow">当前任务</div><dl><div><dt>状态</dt><dd>{latestJob.status}</dd></div><div><dt>进度</dt><dd>{latestJob.next_ordinal} / {overview?.segment_count}</dd></div><div><dt>累计费用</dt><dd>${latestJob.total_cost_usd.toFixed(4)}</dd></div></dl>{latestJob.last_error && <p className="job-error">{latestJob.last_error}</p>}</div>}</div></aside>
      </div>
    </section>

    {searchOpen && <div className="command-overlay"><div className="command-palette search-palette"><form onSubmit={(event) => void performSearch(event)}><Mark>⌕</Mark><input ref={searchInput} value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} placeholder="搜索当前书籍的原文或译文…" /><button type="submit" className="search-submit" disabled={!searchQuery.trim() || searching}>{searching ? "搜索中" : "搜索"}</button><button type="button" className="search-close" aria-label="关闭搜索" onClick={() => setSearchOpen(false)}>×</button></form><div className="command-results">{searching ? <div className="search-message">正在搜索…</div> : searchResults.length ? searchResults.map((item) => <button key={item.id} onClick={() => { setSearchOpen(false); setPanel("translate"); void navigateTo(item.ordinal); }}><span><strong>第 {item.ordinal + 1} 段</strong><small>{item.source_text.slice(0, 120)}</small></span><Mark>↵</Mark></button>) : <div className="search-message">输入关键词，搜索整本书的原文和译文。</div>}</div></div></div>}
    {draftPicker && <div className="command-overlay chapter-picker-overlay" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && wholeBookAction === null) setDraftPicker(null); }}><div className="chapter-picker-dialog" role="dialog" aria-modal="true" aria-labelledby="chapter-picker-title"><header><div><span>草译范围</span><h2 id="chapter-picker-title">选择要翻译的篇章</h2><p>《{draftPicker.document.title}》· 可多选，任务只处理勾选范围。</p></div><button aria-label="关闭篇章选择" onClick={() => setDraftPicker(null)} disabled={wholeBookAction !== null}>×</button></header><div className="chapter-picker-tools"><strong>目录 · {draftPicker.overview.chapters.length} 节</strong><div><button onClick={() => setSelectedChapterStarts([...new Set(draftPicker.overview.chapters.map((chapter) => chapter.start_ordinal))])}>全选</button><button onClick={() => setSelectedChapterStarts([])}>清空</button></div></div><div className="chapter-picker-list" role="group" aria-label="选择草译篇章">{draftPicker.overview.chapters.map((chapter, index) => { const checked = selectedChapterStarts.includes(chapter.start_ordinal); return <label className={checked ? "selected" : ""} key={chapter.start_ordinal + "-" + chapter.title}><input type="checkbox" checked={checked} onChange={() => toggleDraftChapter(chapter.start_ordinal)} /><span className="chapter-check" aria-hidden="true">✓</span><span className="chapter-number">{String(index + 1).padStart(2, "0")}</span><div><strong>{chapter.title}</strong><small>{chapter.segment_count} 段 · 已草译 {chapter.translated_count} · 已确认 {chapter.confirmed_count}</small></div></label>; })}</div><footer><span>已选 <strong>{selectedDraftChapters.length}</strong> 篇 · 去重后共 <strong>{selectedDraftSegmentCount}</strong> 段</span><div><button onClick={() => setDraftPicker(null)} disabled={wholeBookAction !== null}>取消</button><button className="accent" disabled={!selectedDraftRanges.length || wholeBookAction !== null} onClick={() => void runWholeBook(draftPicker.document, "draft", selectedDraftRanges)}>{wholeBookAction === "draft" ? "正在启动…" : "开始草译所选篇章"}</button></div></footer></div></div>}
    {pendingDelete && <div className="command-overlay delete-overlay" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !deleting) setPendingDelete(null); }}><div className="delete-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-title"><div className="delete-mark">!</div><h2 id="delete-title">从书库删除《{pendingDelete.title}》？</h2><p>这会同时删除该书的原文、译文、任务记录与质量检查结果。此操作无法撤销。</p><div><button onClick={() => setPendingDelete(null)} disabled={deleting}>取消</button><button className="danger" onClick={() => void deleteSelectedDocument()} disabled={deleting}>{deleting ? "正在删除…" : "确认删除"}</button></div></div></div>}
    {bookSettingsEditor && settings && <div className="command-overlay book-settings-overlay" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !savingBookSettings) setBookSettingsEditor(null); }}>
      <div className="book-settings-dialog" role="dialog" aria-modal="true" aria-labelledby="book-settings-title">
        <header><div><span>逐书配置</span><h2 id="book-settings-title">《{bookSettingsEditor.document.title}》翻译设置</h2><p>此书的任务会固定使用下列模型；其他书可以同时使用不同模型运行。</p></div><button aria-label="关闭书籍设置" onClick={() => setBookSettingsEditor(null)} disabled={savingBookSettings}>×</button></header>
        <div className="book-settings-body">
          <section><h3>草译模型</h3><div className="book-model-grid">
            <label><span>连接</span><select value={bookSettingsEditor.value.draft_profile_id} onChange={(event) => { const profileId = event.target.value; setBookSettingsEditor((current) => current ? { ...current, value: { ...current.value, draft_profile_id: profileId, draft_model: modelsForProfile(settings, profileId)[0] || "" } } : current); }}>{settings.profiles.map((profile) => <option value={profile.id} key={profile.id}>{profile.name}</option>)}</select></label>
            <label><span>模型 ID</span><input list={"draft-models-" + bookSettingsEditor.document.id} value={bookSettingsEditor.value.draft_model} onChange={(event) => setBookSettingsEditor((current) => current ? { ...current, value: { ...current.value, draft_model: event.target.value } } : current)} placeholder="选择或输入模型 ID" /><datalist id={"draft-models-" + bookSettingsEditor.document.id}>{modelsForProfile(settings, bookSettingsEditor.value.draft_profile_id).map((model) => <option value={model} key={model} />)}</datalist></label>
            <label><span>计算档位</span><select value={bookSettingsEditor.value.draft_compute_mode} onChange={(event) => setBookSettingsEditor((current) => current ? { ...current, value: { ...current.value, draft_compute_mode: event.target.value as ComputeMode } } : current)}><option value="economy">节省</option><option value="balanced">均衡</option><option value="performance">性能</option></select></label>
          </div></section>
          <section><h3>审校模型 <small>可选</small></h3><div className="book-model-grid">
            <label><span>连接</span><select value={bookSettingsEditor.value.reviewer_profile_id} onChange={(event) => { const profileId = event.target.value; setBookSettingsEditor((current) => current ? { ...current, value: { ...current.value, reviewer_profile_id: profileId, reviewer_model: modelsForProfile(settings, profileId)[0] || "" } } : current); }}>{settings.profiles.map((profile) => <option value={profile.id} key={profile.id}>{profile.name}</option>)}</select></label>
            <label><span>模型 ID</span><input list={"review-models-" + bookSettingsEditor.document.id} value={bookSettingsEditor.value.reviewer_model} onChange={(event) => setBookSettingsEditor((current) => current ? { ...current, value: { ...current.value, reviewer_model: event.target.value } } : current)} placeholder="留空则不启用审校" /><datalist id={"review-models-" + bookSettingsEditor.document.id}>{modelsForProfile(settings, bookSettingsEditor.value.reviewer_profile_id).map((model) => <option value={model} key={model} />)}</datalist></label>
            <label><span>计算档位</span><select value={bookSettingsEditor.value.reviewer_compute_mode} onChange={(event) => setBookSettingsEditor((current) => current ? { ...current, value: { ...current.value, reviewer_compute_mode: event.target.value as ComputeMode } } : current)}><option value="economy">节省</option><option value="balanced">均衡</option><option value="performance">性能</option></select></label>
          </div></section>
          <section className="book-style-section"><h3>翻译风格</h3><div className="book-style-presets">{STYLE_PRESETS.map((preset) => <button type="button" className={bookSettingsEditor.style_guide === preset.guide ? "selected" : ""} key={preset.id} onClick={() => setBookSettingsEditor((current) => current ? { ...current, style_guide: preset.guide } : current)}>{preset.label}</button>)}</div><textarea aria-label="这本书的翻译风格" value={bookSettingsEditor.style_guide} onChange={(event) => setBookSettingsEditor((current) => current ? { ...current, style_guide: event.target.value } : current)} placeholder="描述语气、读者、术语和行文要求" /></section>
        </div>
        <footer><span>模型选择只作用于《{bookSettingsEditor.document.title}》</span><div><button onClick={() => setBookSettingsEditor(null)} disabled={savingBookSettings}>取消</button><button className="accent" onClick={() => void saveBookSettings()} disabled={savingBookSettings}>{savingBookSettings ? "正在保存…" : "保存设置"}</button></div></footer>
      </div>
    </div>}
    {toast && <div className="toast" role="status"><span>✓</span>{toast}</div>}
  </main>;
}
