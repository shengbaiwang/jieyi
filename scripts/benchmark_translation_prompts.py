"""Read-only comparison of optimized prompts against a pre-change git revision.

Usage: python scripts/benchmark_translation_prompts.py --database jieyi.db --baseline <ref>
No provider is initialized. Reports characters, NOT billable/model-specific tokens.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from jieyi.persistence.sqlite import SQLiteStore
from jieyi.prompting import build_messages
from jieyi.protection import ProtectedTextCodec
from jieyi.workflow.requests import prepare_translation


class ReadOnlyStore(SQLiteStore):
    def __init__(self, path: Path):
        self.path = path.resolve().as_uri() + '?mode=ro'

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA query_only = ON')
        try:
            yield connection
        finally:
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--jobs', type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ['git', 'rev-parse', '--verify', args.baseline + '^{commit}'], cwd=root, text=True,
    ).strip()
    # Only trusted local repository code is loaded. No user book text is executed.
    old_prompting = subprocess.check_output(
        ['git', 'show', revision + ':src/jieyi/prompting.py'], cwd=root, text=True,
    )
    namespace = {'__name__': 'jieyi_baseline_prompting'}
    exec(compile(old_prompting, '<baseline prompting>', 'exec'), namespace)  # noqa: S102 -- trusted local git baseline
    baseline_messages = namespace['build_messages']
    store = ReadOnlyStore(args.database)
    codec = ProtectedTextCodec()
    reports = []
    with store._connect() as connection:
        job_ids = [row[0] for row in connection.execute(
            'SELECT id FROM jobs ORDER BY created_at DESC LIMIT ?', (args.jobs,),
        )]
    for job_id in job_ids:
        job = store.get_job(job_id)
        document = store.get_document(job.document_id)
        project = store.get_project_for_document(job.document_id)
        segments = store.list_segments(job.document_id)
        by_ordinal = {segment.ordinal: segment for segment in segments}
        approved = [term for term in store.list_terms(project.id) if term.status.value == 'approved']
        before = after = local_envelopes = count = context_chars = 0
        for segment in segments:
            if job.recipe.segment_ranges and not any(
                start <= segment.ordinal <= end for start, end in job.recipe.segment_ranges
            ):
                continue
            prepared = prepare_translation(
                store, codec, project, document, segment, job.recipe, approved, by_ordinal,
            )
            original_source = store.epub_translation_source(segment.id)
            original = codec.encode(original_source or segment.source_text)
            assert prepared.protected.restore(prepared.protected.masked) == (
                original_source or segment.source_text
            ), 'Source compaction was not lossless'
            old_request = replace(
                prepared.request,
                segment=replace(segment, source_text=original.masked),
                atom_boundaries=original.atom_boundaries if original_source else (),
            )
            before += sum(len(message['content']) for message in baseline_messages(old_request))
            after += sum(len(message['content']) for message in build_messages(prepared.request))
            context_chars += len(prepared.request.context) + len(prepared.request.segment_context)
            local_envelopes += int(prepared.protected.local_wrapper is not None)
            count += 1
        reports.append({
            'job_id': job_id, 'title': document.title, 'segments': count,
            'local_atom_envelopes': local_envelopes,
            'before_prompt_characters': before, 'after_prompt_characters': after,
            'saved_prompt_characters': before - after,
            'input_character_reduction_percent': round(100 * (before - after) / before, 2)
            if before else 0,
            'unchanged_context_characters': context_chars,
        })
    print(json.dumps({
        'baseline': revision,
        'measurement': 'prompt characters; not token counts or a translation-quality evaluation',
        'scope': 'all scoped segments, including already translated ones; no model calls',
        'reports': reports,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
