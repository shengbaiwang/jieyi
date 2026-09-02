from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

from jieyi.domain.models import TermEntry, TermStatus, new_id
from jieyi.ingestion import take_distributed_sample
from jieyi.persistence.sqlite import SQLiteStore
from jieyi.providers import EchoProvider, OpenAICompatibleProvider, ProviderRegistry
from jieyi.quality import refresh_segment_quality, reindex_all_quality, reindex_project_quality
from jieyi.workflow import (
    TranslationEngine,
    create_document,
    create_epub_document,
    create_job,
    create_project,
)


def _store(path: str) -> SQLiteStore:
    store = SQLiteStore(path)
    store.migrate()
    reindex_all_quality(store)
    return store


def _providers(args: argparse.Namespace) -> ProviderRegistry:
    providers = ProviderRegistry()
    providers.register("echo", EchoProvider())
    base_url = getattr(args, "base_url", None) or os.getenv("JIEYI_OPENAI_BASE_URL")
    if base_url:
        key = os.getenv(getattr(args, "api_key_env", "JIEYI_OPENAI_API_KEY"), "")
        providers.register("openai-compatible", OpenAICompatibleProvider(base_url, key))
    return providers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jieyi", description="Traceable scholarly translation")
    parser.add_argument("--db", default=os.getenv("JIEYI_DB", "jieyi.db"))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db")

    project = sub.add_parser("project-create")
    project.add_argument("--name", required=True)
    project.add_argument("--source-lang", required=True)
    project.add_argument("--target-lang", required=True)
    project.add_argument("--domain", default="humanities_and_social_sciences")
    project.add_argument("--style-guide", default="")

    document = sub.add_parser("document-import")
    document.add_argument("--project", required=True)
    document.add_argument("--file", type=Path, required=True)
    document.add_argument("--title")
    document.add_argument("--format", choices=["txt", "markdown", "epub"])

    term = sub.add_parser("term-add")
    term.add_argument("--project", required=True)
    term.add_argument("--source", required=True)
    term.add_argument("--target", required=True)
    term.add_argument("--status", choices=[item.value for item in TermStatus], default="approved")
    term.add_argument("--rationale", default="")
    term.add_argument("--forbid", action="append", default=[])
    term.add_argument("--alias", action="append", default=[])
    term.add_argument("--context", action="append", default=[])
    term.add_argument("--sense", default="")
    term.add_argument("--disambiguation", default="")

    job = sub.add_parser("job-create")
    job.add_argument("--document", required=True)
    job.add_argument("--draft-provider", default="echo")
    job.add_argument("--draft-model", default="dry-run")
    job.add_argument("--no-tm", action="store_true")
    job.add_argument("--tm-threshold", type=float, default=0.78)
    job.add_argument("--tm-max-results", type=int, default=3)

    run = sub.add_parser("job-run")
    run.add_argument("--job", required=True)
    run.add_argument("--base-url")
    run.add_argument("--api-key-env", default="JIEYI_OPENAI_API_KEY")
    run.add_argument("--max-segments", type=int)

    confirm = sub.add_parser("segment-confirm")
    confirm.add_argument("--segment", required=True)
    confirm.add_argument("--translation", required=True)
    confirm.add_argument("--rationale", default="")

    export = sub.add_parser("document-export")
    export.add_argument("--document", required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--bilingual", action="store_true")

    inspect = sub.add_parser("document-show")
    inspect.add_argument("--document", required=True)

    preview = sub.add_parser("prompt-preview")
    preview.add_argument("--job", required=True)
    preview.add_argument("--segment", required=True)

    tm_show = sub.add_parser("tm-show")
    tm_show.add_argument("--project", required=True)

    sample = sub.add_parser("document-sample")
    sample.add_argument("--document", required=True)
    sample.add_argument("--budget", type=int, default=6_000)
    sample.add_argument("--count", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = _store(args.db)

    if args.command == "init-db":
        print(json.dumps({"database": args.db, "status": "ready"}, ensure_ascii=False))
    elif args.command == "project-create":
        item = create_project(
            store,
            name=args.name,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            domain=args.domain,
            style_guide=args.style_guide,
        )
        print(json.dumps(asdict(item), ensure_ascii=False))
    elif args.command == "document-import":
        suffix = args.file.suffix.lower()
        source_format = args.format or (
            "epub" if suffix == ".epub" else "markdown" if suffix == ".md" else "txt"
        )
        if source_format == "epub":
            item = create_epub_document(
                store,
                project_id=args.project,
                title=args.title,
                file_data=args.file.read_bytes(),
            )
        else:
            item = create_document(
                store,
                project_id=args.project,
                title=args.title or args.file.stem,
                text=args.file.read_text(encoding="utf-8"),
                source_format=source_format,
            )
        print(json.dumps(asdict(item), ensure_ascii=False))
    elif args.command == "term-add":
        item = store.add_term(
            TermEntry(
                id=new_id("term"),
                project_id=args.project,
                source=args.source,
                target=args.target,
                status=TermStatus(args.status),
                rationale=args.rationale,
                forbidden_targets=tuple(args.forbid),
                aliases=tuple(args.alias),
                context_keywords=tuple(args.context),
                sense=args.sense,
                disambiguation=args.disambiguation,
            )
        )
        reindex_project_quality(store, args.project)
        print(json.dumps(asdict(item), ensure_ascii=False))
    elif args.command == "job-create":
        item = create_job(
            store,
            document_id=args.document,
            draft_provider=args.draft_provider,
            draft_model=args.draft_model,
            tm_enabled=not args.no_tm,
            tm_threshold=args.tm_threshold,
            tm_max_results=args.tm_max_results,
        )
        print(json.dumps(asdict(item), ensure_ascii=False, default=str))
    elif args.command == "job-run":
        engine = TranslationEngine(store, _providers(args))
        item = asyncio.run(engine.run(args.job, max_segments=args.max_segments))
        print(json.dumps(asdict(item), ensure_ascii=False, default=str))
    elif args.command == "segment-confirm":
        store.confirm_segment(args.segment, args.translation, args.rationale)
        refresh_segment_quality(store, args.segment)
        print(json.dumps({"segment": args.segment, "status": "human_confirmed"}, ensure_ascii=False))
    elif args.command == "document-export":
        args.output.write_text(
            store.export_document(args.document, bilingual=args.bilingual), encoding="utf-8"
        )
        print(json.dumps({"output": str(args.output)}, ensure_ascii=False))
    elif args.command == "document-show":
        payload = [asdict(item) for item in store.list_segments(args.document)]
        print(json.dumps(payload, ensure_ascii=False, default=str, indent=2))
    elif args.command == "prompt-preview":
        engine = TranslationEngine(store, _providers(args))
        print(json.dumps(engine.preview(args.job, args.segment), ensure_ascii=False, default=str, indent=2))
    elif args.command == "tm-show":
        store.get_project(args.project)
        payload = [asdict(item) for item in store.list_translation_memory(args.project)]
        print(json.dumps(payload, ensure_ascii=False, default=str, indent=2))
    elif args.command == "document-sample":
        store.get_document(args.document)
        source = "\n\n".join(item.source_text for item in store.list_segments(args.document))
        payload = take_distributed_sample(
            source, total_budget=args.budget, excerpt_count=args.count
        )
        print(json.dumps(asdict(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
