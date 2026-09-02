export type ReaderPageTarget = { spineIndex: number; pageNumber: number };
type Location = { ordinal: number; top: number };
type Options = {
  documentId?: string;
  targets: Record<number, ReaderPageTarget>;
  onPosition: (ordinal: number, pageNumber?: number) => void;
  onHeight: (spineIndex: number, height: number) => void;
  onPending: (pending: boolean) => void;
  onUnavailable: () => void;
};

export function ordinalAtOffset(locations: Location[], top: number): number | undefined {
  let active = locations[0]?.ordinal;
  for (const location of locations) {
    if (location.top > top + 2) break;
    active = location.ordinal;
  }
  return active;
}

// Keep navigation anchored through asynchronous iframe, font and image resizing.
// Only deliberate reader input releases the anchor; elapsed time is not readiness.
export function createReaderNavigation(root: HTMLElement, options: Options) {
  const win = root.ownerDocument.defaultView!;
  const locations = new Map<number, Location[]>();
  let anchor: { ordinal: number; section: HTMLElement; offset: number | null } | null = null;
  let animation = 0;
  let disposed = false;

  const inset = () => (root.querySelector<HTMLElement>(".reader-toolbar")?.getBoundingClientRect().height || 0) + 12;
  const frameFor = (spineIndex: number) => root.querySelector<HTMLIFrameElement>(`iframe[data-spine-index="${spineIndex}"]`);

  function align() {
    if (!anchor) return;
    const top = root.scrollTop + anchor.section.getBoundingClientRect().top
      - root.getBoundingClientRect().top + (anchor.offset ?? 0) - inset();
    // Instant navigation also cancels any previous smooth scroll on rapid clicks.
    root.scrollTo({ top: Math.max(0, top), behavior: "instant" });
  }

  function trackPosition() {
    if (anchor) { align(); return; }
    const line = root.getBoundingClientRect().top + inset();
    const sections = Array.from(root.querySelectorAll<HTMLElement>("[data-reader-ordinal]"));
    const section = sections.find((item) => item.getBoundingClientRect().bottom > line) || sections.at(-1);
    if (!section) return;
    const frame = section.querySelector<HTMLIFrameElement>("iframe");
    const entries = frame ? locations.get(Number(frame.dataset.spineIndex)) : undefined;
    const ordinal = entries?.length
      ? ordinalAtOffset(entries, line - section.getBoundingClientRect().top)
      : Number(section.dataset.readerOrdinal);
    if (ordinal !== undefined && Number.isInteger(ordinal)) {
      const page = Number(section.dataset.readerPage);
      options.onPosition(ordinal, page > 0 ? page : undefined);
    }
  }

  function schedule() {
    if (disposed || animation) return;
    animation = win.requestAnimationFrame(() => { animation = 0; trackPosition(); });
  }

  function release() {
    if (!anchor) return;
    anchor = null;
    options.onPending(false);
    schedule();
  }

  function keydown(event: KeyboardEvent) {
    if (["ArrowDown", "ArrowUp", "PageDown", "PageUp", "Home", "End", " "].includes(event.key)) release();
  }

  function requestLocation(frame: HTMLIFrameElement, ordinal: number) {
    frame.contentWindow?.postMessage({ type: "jy-epub-locate", documentId: options.documentId,
      spineIndex: Number(frame.dataset.spineIndex), segmentOrdinal: ordinal }, "*");
  }

  function frameLoaded(frame: HTMLIFrameElement) {
    frame.dataset.loaded = "true";
    if (anchor && options.targets[anchor.ordinal]?.spineIndex === Number(frame.dataset.spineIndex)) {
      requestLocation(frame, anchor.ordinal);
    }
  }

  function goTo(ordinal: number) {
    const page = options.targets[ordinal];
    const section = root.querySelector<HTMLElement>(page ? `#reader-spine-${page.spineIndex}` : `#reader-segment-${ordinal}`);
    if (!section) { options.onUnavailable(); return; }
    const frame = section.querySelector<HTMLIFrameElement>("iframe");
    const needsLocation = Boolean(frame && !section.classList.contains("fixed-layout"));
    const known = page ? locations.get(page.spineIndex)?.find((item) => item.ordinal === ordinal) : undefined;
    anchor = { ordinal, section, offset: needsLocation ? known?.top ?? null : 0 };
    options.onPosition(ordinal, page?.pageNumber);
    options.onPending(needsLocation && !known);
    // Explicitly load the destination even if it is far outside the lazy-load range.
    if (frame) {
      frame.loading = "eager";
      requestLocation(frame, ordinal);
    }
    align();
    schedule();
  }

  function message(event: MessageEvent) {
    const data = event.data;
    if (!data || data.documentId !== options.documentId || !Number.isInteger(data.spineIndex)) return;
    const frame = frameFor(data.spineIndex);
    if (!frame || event.source !== frame.contentWindow) return;
    if (data.type === "jy-epub-interact") { release(); return; }
    if (data.type === "jy-epub-location") {
      if (!anchor || anchor.ordinal !== data.segmentOrdinal || options.targets[anchor.ordinal]?.spineIndex !== data.spineIndex) return;
      if (Number.isFinite(data.top)) {
        anchor.offset = Math.max(0, data.top);
        options.onPending(false);
        schedule();
      } else if (data.found === false) {
        anchor.offset = 0;
        options.onPending(false);
        options.onUnavailable();
        schedule();
      }
      return;
    }
    if (data.type !== "jy-epub-resize" || !Number.isFinite(data.height)) return;
    const height = Math.max(96, Math.ceil(data.height));
    // Apply before aligning so the destination cannot be clamped to stale height.
    frame.style.height = `${height}px`;
    options.onHeight(data.spineIndex, height);
    if (Array.isArray(data.locations)) {
      const entries: Location[] = data.locations.filter((item: Location) =>
        Number.isInteger(item?.ordinal) && Number.isFinite(item?.top) && item.top >= 0)
        .sort((a: Location, b: Location) => a.top - b.top || a.ordinal - b.ordinal);
      locations.set(data.spineIndex, entries);
      if (anchor && options.targets[anchor.ordinal]?.spineIndex === data.spineIndex) {
        const match = entries.find((item) => item.ordinal === anchor!.ordinal);
        if (match) { anchor.offset = match.top; options.onPending(false); }
      }
    }
    schedule();
  }

  const observer = new ResizeObserver(schedule);
  observer.observe(root);
  root.querySelectorAll(".reader-toolbar, .epub-spine-sheet, .epub-cover-sheet, .reader-paper").forEach((element) => observer.observe(element));
  root.addEventListener("scroll", schedule, { passive: true });
  root.addEventListener("wheel", release, { passive: true });
  root.addEventListener("touchstart", release, { passive: true });
  root.addEventListener("pointerdown", release);
  root.addEventListener("keydown", keydown);
  win.addEventListener("message", message);
  // A frame may finish before the React effect attaches the message listener.
  root.querySelectorAll<HTMLIFrameElement>("iframe[data-spine-index]").forEach((frame) => {
    const ordinal = Number(frame.closest<HTMLElement>("[data-reader-ordinal]")?.dataset.readerOrdinal);
    requestLocation(frame, ordinal);
  });
  schedule();

  return {
    goTo, frameLoaded,
    destroy() {
      disposed = true;
      win.cancelAnimationFrame(animation);
      observer.disconnect();
      root.removeEventListener("scroll", schedule);
      root.removeEventListener("wheel", release);
      root.removeEventListener("touchstart", release);
      root.removeEventListener("pointerdown", release);
      root.removeEventListener("keydown", keydown);
      win.removeEventListener("message", message);
    },
  };
}
