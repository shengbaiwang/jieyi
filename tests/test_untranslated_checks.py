import pytest

from jieyi.domain.models import IssueSeverity, SegmentKind
from jieyi.quality import run_deterministic_checks
from jieyi.quality.checks import DETECTOR_VERSION

SOURCE = "How bow'd the woods beneath their sturdy stroke!"


@pytest.mark.parametrize("kind", [SegmentKind.PARAGRAPH, SegmentKind.VERSE])
@pytest.mark.parametrize("target", [
    SOURCE,
    "How bow'd the woods\nbeneath their sturdy stroke!",
    "  How   bow'd the woods beneath their sturdy stroke!  ",
    "HOW BOW'D THE WOODS BENEATH THEIR STURDY STROKE!",
    "How bow’d the woods beneath their sturdy stroke！",
    "How bowd the woods beneath their sturdy stroke",
])
def test_screenshot_copy_is_a_required_review_issue(kind, target):
    issues = run_deterministic_checks(
        SOURCE, target, segment_kind=kind, source_lang="en", target_lang="zh-CN",
    )
    issue = next(item for item in issues if item.code == "untranslated_text")
    assert issue.severity is IssueSeverity.ERROR
    assert issue.details["detector_version"] == DETECTOR_VERSION
    assert issue.details["confidence"] == "high"
    assert issue.details["source_lang"] == "en"
    assert issue.details["target_lang"] == "zh-CN"


@pytest.mark.parametrize("source,source_lang,target_lang", [
    ("古老的树林在有力的砍伐下低垂。", "zh-CN", "en"),
    ("Les arbres plient sous leurs coups puissants !", "fr", "zh-CN"),
    ("The woods bowed.", "en", "zh-CN"),
    (SOURCE.upper(), "en", "zh-CN"),
])
def test_copied_prose_is_detected_for_other_language_pairs(source, source_lang, target_lang):
    issues = run_deterministic_checks(
        source, source, source_lang=source_lang, target_lang=target_lang,
    )
    assert "untranslated_text" in {item.code for item in issues}


@pytest.mark.parametrize("source,target", [
    (SOURCE, "林木在他们有力的砍伐下纷纷低垂！"),
    (
        "Eternal Providence guided the poet through this difficult passage.",
        "永恒的天意（Eternal Providence）引领诗人度过这段艰难的旅程。",
    ),
    (
        "The book mentions Thomas Gray and his best-known poem.",
        "书中提到了 Thomas Gray 和他最著名的诗。",
    ),
])
def test_actual_translation_and_preserved_names_are_not_untranslated(source, target):
    issues = run_deterministic_checks(source, target, source_lang="en", target_lang="zh-CN")
    assert "untranslated_text" not in {item.code for item in issues}


@pytest.mark.parametrize("source", [
    "Thomas Gray",
    "Ludwig van Beethoven",
    "UNITED NATIONS EDUCATIONAL SCIENTIFIC AND CULTURAL ORGANIZATION",
    "2026-10-30 · 12.5% · 237",
    "https://example.org/how-the-woods-bowed-beneath-their-sturdy-stroke",
    "10.1234/how.the.woods.bowed.beneath.their.sturdy.stroke",
    "`how the woods bowed beneath their sturdy stroke`",
    "```python\nprint('how the woods bowed beneath their sturdy stroke')\n```",
    "(Smith and Jones 2020)",
    "Jones, David. 2004. Reforming the morality of usury: a study of the social space of money.",
    "49. Nelida Fuccaro, Histories of City and State in the Persian Gulf.",
    "Anscombe, Frederick, 228n13, 228n15",
    "[12, 13] [^3]",
    "<span class='how the woods bowed beneath their sturdy stroke'></span>",
    "林木在他们有力的砍伐下纷纷低垂！",
])
def test_unchanged_nonprose_and_already_translated_content_is_not_flagged(source):
    issues = run_deterministic_checks(source, source, source_lang="en", target_lang="zh-CN")
    assert "untranslated_text" not in {item.code for item in issues}


@pytest.mark.parametrize("source_lang,target_lang", [
    ("en", "en"),
    ("en-US", "en-GB"),
    ("EN_us", "en-GB"),
    (None, "zh-CN"),
    ("en", None),
])
def test_same_language_and_missing_language_context_do_not_guess(source_lang, target_lang):
    issues = run_deterministic_checks(
        SOURCE, SOURCE, source_lang=source_lang, target_lang=target_lang,
    )
    assert "untranslated_text" not in {item.code for item in issues}


@pytest.mark.parametrize("suffix", [
    " (Smith and Jones 2020)",
    " [^3]",
    " https://example.org/poetry",
    " `sturdy_stroke`",
])
def test_protected_spans_do_not_hide_copied_prose(suffix):
    source = SOURCE + suffix
    issues = run_deterministic_checks(source, source, source_lang="en", target_lang="zh-CN")
    assert "untranslated_text" in {item.code for item in issues}
