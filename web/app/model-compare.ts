export type CompareCandidate = {
  id: string;
  job_id: string;
  stage: string;
  provider: string;
  model: string;
  text: string;
  prompt_tokens: number;
  completion_tokens: number;
  reasoning_tokens?: number;
  cost_usd: number;
  created_at: string;
};

export type CompareModelOption = {
  provider: string;
  model: string;
  label?: string;
};

export type ProviderSettingsLike = {
  profiles: Array<{ id: string; name: string; provider_type: string; base_url: string; selected_models?: string[] | null }>;
  presets: Array<{ id: string; default_models: string[] }>;
  draft_profile_id: string;
  draft_model: string;
};

export type CompareResultItem = {
  candidate_id?: string | null;
  provider: string;
  model: string;
  text: string;
  raw_text?: string;
  ok: boolean;
  error?: string | null;
  prompt_tokens?: number;
  completion_tokens?: number;
  reasoning_tokens?: number;
  cost_usd?: number;
  issues?: Array<{ code: string; message: string; severity: string }>;
  compute_mode?: string;
};

export type CompareModelGroup = {
  key: string;
  provider: string;
  model: string;
  candidates: CompareCandidate[];
  latest: CompareCandidate | null;
};

export function modelOptionKey(provider: string, model: string): string {
  return `${provider}::${model}`;
}

export function groupCandidatesByModel(items: CompareCandidate[]): CompareModelGroup[] {
  const groups = new Map<string, CompareModelGroup>();
  for (const item of items) {
    if (!item?.text?.trim() && item?.stage !== "compare") continue;
    const key = modelOptionKey(item.provider, item.model);
    const existing = groups.get(key) || {
      key,
      provider: item.provider,
      model: item.model,
      candidates: [],
      latest: null,
    };
    existing.candidates.push(item);
    groups.set(key, existing);
  }
  const ordered = [...groups.values()].map((group) => {
    const candidates = [...group.candidates].sort((left, right) =>
      (right.created_at || "").localeCompare(left.created_at || ""));
    return { ...group, candidates, latest: candidates[0] || null };
  });
  return ordered.sort((left, right) =>
    (right.latest?.created_at || "").localeCompare(left.latest?.created_at || "")
    || left.model.localeCompare(right.model));
}

export function selectedModelKeys(selected: string[], key: string): boolean {
  return selected.includes(key);
}

export function toggleModelKey(selected: string[], key: string): string[] {
  return selected.includes(key)
    ? selected.filter((item) => item !== key)
    : [...selected, key];
}

const BLIND_LABELS = "ABCDEFGH";

/** Blind-evaluation column label: never contains provider or model IDs. */
export function blindCandidateLabel(index: number): string {
  const letter = BLIND_LABELS[index] || String(index + 1);
  return `译文 ${letter}`;
}

export function blindCandidateLabels(count: number): string[] {
  return Array.from({ length: count }, (_, index) => blindCandidateLabel(index));
}

/**
 * Fisher–Yates shuffle for blind A/B columns.
 * Inject `random` in tests; production uses Math.random so model→slot mapping
 * is not fixed across runs (a fixed column would still leak identity).
 */
export function shuffleCompareItems<T>(items: T[], random: () => number = Math.random): T[] {
  const next = [...items];
  for (let index = next.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1));
    const held = next[index];
    next[index] = next[swap];
    next[swap] = held;
  }
  return next;
}

export function formatCompareUsage(item: {
  prompt_tokens?: number;
  completion_tokens?: number;
  reasoning_tokens?: number;
  cost_usd?: number;
}): string {
  const prompt = item.prompt_tokens || 0;
  const completion = item.completion_tokens || 0;
  const reasoning = item.reasoning_tokens || 0;
  const cost = item.cost_usd || 0;
  const tokens = `输入 ${prompt} / 输出 ${completion}${reasoning ? ` / 推理 ${reasoning}` : ""}`;
  return `${tokens} · $${cost.toFixed(6)}`;
}

export function availableCompareModels(settings: ProviderSettingsLike | null): CompareModelOption[] {
  if (!settings) return [];
  const options: CompareModelOption[] = [];
  const seen = new Set<string>();
  for (const profile of settings.profiles) {
    const preset = settings.presets.find((item) => item.id === profile.provider_type);
    const bound = settings.draft_profile_id === profile.id ? [settings.draft_model] : [];
    const models = [...new Set([...(profile.selected_models ?? preset?.default_models ?? []), ...bound].filter(Boolean))];
    for (const model of models) {
      const key = modelOptionKey(`profile:${profile.id}`, model);
      if (seen.has(key)) continue;
      seen.add(key);
      options.push({ provider: `profile:${profile.id}`, model, label: `${profile.name} · ${model}` });
    }
  }
  if (!options.length && settings.draft_model) {
    options.push({
      provider: `profile:${settings.draft_profile_id}`,
      model: settings.draft_model,
      label: settings.draft_model,
    });
  }
  return options;
}
