export function uniqueModels(items: string[]): string[] {
  return [...new Set(items.map((item) => item.trim()).filter(Boolean))];
}

// Bindings from older settings must remain selectable after migration.
export function selectedModels(selected: string[] | null | undefined, bound: string[]): string[] {
  return uniqueModels([...(selected || []), ...bound]);
}

export function modelSearch(items: string[], query: string): string[] {
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  return uniqueModels(items).filter((model) => terms.every((term) => model.toLowerCase().includes(term)));
}

export function endpointPreview(base: string, path: string): string {
  const normalized = path.trim();
  return /^https?:\/\//.test(normalized) ? normalized.replace(/\/+$/, "")
    : `${base.trim().replace(/\/+$/, "")}/${normalized.replace(/^\/+/, "")}`;
}

export function validBaseUrl(value: string): boolean {
  try {
    const url = new URL(value.trim());
    return ["https:", "http:"].includes(url.protocol) && Boolean(url.hostname);
  } catch { return false; }
}
