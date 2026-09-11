"use client";

import { useEffect, useRef, useState, type ReactNode, type KeyboardEvent } from "react";
import { createTextHistory, joinSoftLines, recordText, travelText } from "./source-editing";

type Props = {
  value: string; kind: string; editing: boolean; dirty: boolean;
  initialSelection: { start: number; end: number };
  saveState: "idle" | "saving" | "saved" | "error"; busy: boolean; structureLocked: boolean;
  canMergePrevious: boolean; canMergeNext: boolean; headingPath: string;
  label: ReactNode; onChange: (text: string) => void; onEditing: () => void;
  onCancel: () => void; onSave: () => void; onHeading: () => void; onDelete: () => void;
  onMerge: (direction: "previous" | "next") => void; onSplit: (heading: boolean) => void;
  onSelection: (selection: { start: number; end: number }) => void;
  notify: (message: string) => void;
};

export function SourceEditor(props: Props) {
  const { value, kind, editing, dirty, busy, structureLocked, saveState } = props;
  const editor = useRef<HTMLTextAreaElement>(null);
  const menu = useRef<HTMLDetailsElement>(null);
  const beforeSelection = useRef(props.initialSelection);
  const lastEdit = useRef(0);
  const composing = useRef(false);
  const [selection, setSelection] = useState(props.initialSelection);
  const [history, setHistory] = useState(() => createTextHistory(value));
  const selected = value.slice(selection.start, selection.end).trim();
  const canSplit = kind !== "heading" && Boolean(selected) && selected !== value.trim();
  const saving = busy || saveState === "saving";

  useEffect(() => {
    if (!editing) return;
    editor.current?.focus();
    editor.current?.setSelectionRange(beforeSelection.current.start, beforeSelection.current.end);
  }, [editing]);
  useEffect(() => {
    function close(event: PointerEvent) {
      if (event.target instanceof Node && !menu.current?.contains(event.target)) menu.current?.removeAttribute("open");
    }
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, []);

  function rememberSelection(start: number, end: number) {
    const next = { start, end };
    beforeSelection.current = next;
    setSelection(next);
    props.onSelection(next);
  }

  function change(text: string, start: number, end: number, group = false) {
    const now = Date.now();
    const after = { text, start, end };
    const before = { text: value, ...beforeSelection.current };
    const previousEditTime = lastEdit.current;
    const isComposing = composing.current;
    setHistory((current) => recordText(current, before, after, group && previousEditTime > 0 && (isComposing || (now - previousEditTime < 700 && before.start === current.present.start && before.end === current.present.end))));
    lastEdit.current = group ? now : 0;
    props.onChange(text);
    rememberSelection(start, end);
  }

  function restore(direction: "undo" | "redo") {
    const next = travelText(history, direction);
    setHistory(next);
    lastEdit.current = 0;
    props.onChange(next.present.text);
    rememberSelection(next.present.start, next.present.end);
    requestAnimationFrame(() => { editor.current?.focus(); editor.current?.setSelectionRange(next.present.start, next.present.end); });
  }

  function tidyLines() {
    const start = selected ? selection.start : 0;
    const end = selected ? selection.end : value.length;
    const cleaned = joinSoftLines(value.slice(start, end));
    const next = value.slice(0, start) + cleaned + value.slice(end);
    if (next === value) { props.notify("没有需要整理的换行。"); return; }
    change(next, start, start + cleaned.length);
    requestAnimationFrame(() => { editor.current?.focus(); editor.current?.setSelectionRange(start, start + cleaned.length); });
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(selected ? value.slice(selection.start, selection.end) : value);
      props.notify(selected ? "已复制所选文字" : "原文已复制");
    } catch { props.notify("复制失败，请选中文字后使用系统复制快捷键。"); }
  }

  function onKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (event.nativeEvent.isComposing) return;
    const command = event.metaKey || event.ctrlKey;
    if (command && event.key.toLowerCase() === "s") {
      event.preventDefault(); event.stopPropagation();
      if (editing && dirty && value.trim() && !saving) props.onSave();
    }
    if (editing && command && ["z", "y"].includes(event.key.toLowerCase())) {
      event.preventDefault(); event.stopPropagation();
      if (!saving) restore(event.shiftKey || event.key.toLowerCase() === "y" ? "redo" : "undo");
    }
    if (command && event.key === "Enter") {
      event.preventDefault(); event.stopPropagation();
      if (editing && dirty && value.trim() && !saving) props.onSave();
    }
    if (event.key === "Escape") menu.current?.removeAttribute("open");
  }

  function menuKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const buttons = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>("button:not(:disabled)"));
    const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
    if (event.key === "Escape") {
      event.preventDefault(); event.stopPropagation();
      menu.current?.removeAttribute("open");
      menu.current?.querySelector("summary")?.focus();
    } else if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key) && buttons.length) {
      event.preventDefault();
      const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1
        : (index + (event.key === "ArrowDown" ? 1 : -1) + buttons.length) % buttons.length;
      buttons[next].focus();
    }
  }

  function action(callback: () => void) { menu.current?.removeAttribute("open"); callback(); }

  return <article className={`source-pane content-${kind} ${editing ? "is-editing" : "is-readonly"}`}>
    <div className="column-label">{props.label}<button className="source-copy" onClick={() => void copy()}>{selected ? "复制所选" : "复制"}</button></div>
    <div className="source-toolbar" role="group" aria-label="原文编辑工具">
      <div className="source-mode segmented" aria-label="原文模式">
        <button className={!editing ? "active" : ""} aria-pressed={!editing} disabled={saving} onClick={() => { if (editing) props.onCancel(); }}>阅读</button>
        <button className={editing ? "active" : ""} aria-pressed={editing} disabled={saving} onClick={props.onEditing}>编辑</button>
      </div>
      {editing && <div className="source-history">
        <button aria-label="撤销" title="撤销（⌘/Ctrl Z）" disabled={saving || !history.past.length} onClick={() => restore("undo")}>↶</button>
        <button aria-label="重做" title="重做（⌘/Ctrl Shift Z）" disabled={saving || !history.future.length} onClick={() => restore("redo")}>↷</button>
      </div>}
      <details className="source-menu" ref={menu}>
        <summary aria-label="段落操作" aria-haspopup="menu" onKeyDown={(event) => { if (event.key === "ArrowDown") { event.preventDefault(); menu.current?.setAttribute("open", ""); requestAnimationFrame(() => menu.current?.querySelector<HTMLButtonElement>("button:not(:disabled)")?.focus()); } if (event.key === "Escape") { menu.current?.removeAttribute("open"); event.stopPropagation(); } }}>段落 <span aria-hidden="true">⌄</span></summary>
        <div className="source-menu-items" role="menu" tabIndex={-1} aria-label="段落操作" onKeyDown={menuKeyDown}>
          <span className="source-menu-caption">{kind === "heading" ? "当前为目录标题" : "当前为正文段落"}</span>
          <button role="menuitem" disabled={structureLocked} onClick={() => action(props.onHeading)}>{kind === "heading" ? "改为正文" : "设为目录标题"}</button>
          <>
            <button role="menuitem" disabled={structureLocked || !props.canMergePrevious} onClick={() => action(() => props.onMerge("previous"))}>与上一段合并</button>
            <button role="menuitem" disabled={structureLocked || !props.canMergeNext} onClick={() => action(() => props.onMerge("next"))}>与下一段合并</button>
            {editing && kind !== "heading" && <button role="menuitem" disabled={saving} onClick={() => action(tidyLines)}>整理{selected ? "所选" : "本段"}换行</button>}
          </>
          <div className="source-menu-divider" />
          <button role="menuitem" className="source-delete-action" title="删除整段原文及其译文，可撤销" disabled={structureLocked} onClick={() => action(props.onDelete)}>删除整段<small>可撤销</small></button>
          {dirty && <small>保存修改后可调整段落结构</small>}
        </div>
      </details>
      <span className={`source-save-status ${dirty ? "unsaved" : ""}`} role="status">{busy ? "处理中…" : saving ? "保存中…" : saveState === "error" ? "保存失败，请重试" : dirty ? "未保存" : saveState === "saved" ? "已保存" : editing ? "可编辑" : ""}</span>
    </div>
    <div className={`source-selection-tools ${selected ? "has-selection" : ""}`} role="group" aria-label="所选原文操作">
      <span>{selected ? `已选 ${Array.from(selected).length} 字` : editing ? "直接修改文字，空行分隔段落" : "选中文字拆段，或切换编辑修改"}</span>
      {selected && !editing && <button disabled={saving} onClick={props.onEditing}>修改所选</button>}
      {canSplit && <>
        <button disabled={structureLocked} title={dirty ? "请先保存修改" : "将所选文字拆为独立段落"} onClick={() => props.onSplit(false)}>拆为独立段</button>
        <button disabled={structureLocked} title={dirty ? "请先保存修改" : "将所选文字拆为目录标题"} onClick={() => props.onSplit(true)}>拆为标题</button>
      </>}
    </div>
    <textarea ref={editor} className={kind === "heading" ? "source-editor source-heading-editor" : "source-editor"} aria-label={editing ? "原文编辑器" : "原文，只读"} readOnly={!editing || saving} aria-busy={saving} value={value}
      onSelect={(event) => rememberSelection(event.currentTarget.selectionStart, event.currentTarget.selectionEnd)}
      onBeforeInput={() => { if (editor.current) beforeSelection.current = { start: editor.current.selectionStart, end: editor.current.selectionEnd }; }}
      onChange={(event) => { const input = event.currentTarget; change(input.value, input.selectionStart, input.selectionEnd, true); }}
      onKeyDown={onKeyDown}
      onCompositionStart={() => { composing.current = true; lastEdit.current = 0; }}
      onCompositionEnd={() => { composing.current = false; lastEdit.current = 0; }}
      spellCheck={false} />
    <div className="editor-footer source-footer">
      <span>{Array.from(value).length} 字 · {value.trim() ? value.trim().split(/\n\s*\n/).length : 0} 段</span>
      {editing && <div className="editor-footer-actions">
        <button className="source-cancel" disabled={saving} onClick={props.onCancel}>取消</button>
        <button className="confirm-button" onClick={props.onSave} disabled={!dirty || !value.trim() || saving}>{saving ? "保存中…" : "保存"}<kbd>⌘ S</kbd></button>
      </div>}
    </div>
    {props.headingPath && <div className="source-location" title={props.headingPath}><span aria-hidden="true">¶</span> {props.headingPath}</div>}
  </article>;
}
