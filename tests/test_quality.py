import unittest

from jieyi.domain.models import IssueSeverity, SegmentKind, TermEntry, TermStatus
from jieyi.quality import run_deterministic_checks


class QualityTests(unittest.TestCase):
    def test_detects_numbers_citations_and_terms(self):
        term = TermEntry(
            id="term_1",
            project_id="proj_1",
            source="agency",
            target="能动性",
            status=TermStatus.APPROVED,
            forbidden_targets=("代理性",),
        )
        issues = run_deterministic_checks(
            "Agency rose by 12% (Smith 2020) [^3].",
            "代理性有所上升（Smith 2021）。",
            [term],
        )
        codes = {item.code for item in issues}
        self.assertIn("number_mismatch", codes)
        self.assertIn("citation_mismatch", codes)
        self.assertIn("footnote_mismatch", codes)
        self.assertIn("approved_term_missing", codes)
        self.assertIn("forbidden_term_used", codes)

    def test_clean_translation_has_no_deterministic_issue(self):
        issues = run_deterministic_checks(
            "Agency rose by 12% (Smith 2020) [^3].",
            "能动性提高了 12% (Smith 2020) [^3]。",
            [
                TermEntry(
                    id="term_1",
                    project_id="proj_1",
                    source="agency",
                    target="能动性",
                    status=TermStatus.APPROVED,
                )
            ],
        )
        self.assertEqual(issues, [])

    def test_cjk_adjacent_numbers_grouping_and_unicode_digits_are_equivalent(self):
        issues = run_deterministic_checks(
            "Winner in 2012; first 5,000 years; note 8.",
            "荣获2012年；最初5000年；注8。",
        )
        self.assertNotIn("number_mismatch", {item.code for item in issues})

    def test_localised_numeric_facts_are_semantically_equivalent(self):
        pairs = (
            ("First printing: October 2014", "首次印刷：2014年10月"),
            ("In May the army advanced", "军队于五月推进"),
            ("Starting in the 1980s", "自20世纪80年代起"),
            ("the nineteenth century", "19世纪"),
            ("150 million francs", "1.5亿法郎"),
            ("60 percent", "60%"),
            ("30 percent of the harvest", "收成的三成"),
            ("Amos 2.6", "《阿摩司书》2:6"),
            ("800 to 1000 pesos", "八百到一千比索"),
            ("Workers arrived at 8:00", "工人于8点到达"),
            ("Release v3.1_r2", "版本 v3.1_r2"),
        )
        for source, target in pairs:
            with self.subTest(source=source, target=target):
                issues = run_deterministic_checks(source, target)
                self.assertNotIn("number_mismatch", {item.code for item in issues})

    def test_note_styles_and_inline_enumerations_do_not_create_numeric_mismatches(self):
        pairs = (
            ("An example.10 Next sentence.", "一个例子。¹⁰ 下一句。"),
            ("Claim.)5 Next sentence.", "主张。）⁵ 下一句。"),
            (
                "noun 1 first meaning. 2 second meaning. 3 third meaning.",
                "名词 1 第一义。2 第二义。3 第三义。",
            ),
        )
        for source, target in pairs:
            with self.subTest(source=source, target=target):
                issues = run_deterministic_checks(source, target)
                self.assertNotIn("number_mismatch", {item.code for item in issues})

    def test_missing_numeric_fact_is_a_high_confidence_warning(self):
        issues = run_deterministic_checks(
            "Revenue fell from 12% to 9%.",
            "收入从12%下降到8%。",
        )
        issue = next(item for item in issues if item.code == "number_mismatch")
        self.assertEqual(issue.severity, IssueSeverity.WARNING)
        self.assertEqual(issue.details["detector_version"], "4")
        self.assertEqual(issue.details["confidence"], "high")
        self.assertEqual(issue.details["missing_typed"], {"percent:9": 1})

    def test_footnote_segments_skip_ambiguous_generic_number_checks(self):
        issues = run_deterministic_checks(
            "24. See Amos 2.6 and page 193.",
            "参见《阿摩司书》2:6及第193页。",
            segment_kind=SegmentKind.FOOTNOTE,
        )
        self.assertNotIn("number_mismatch", {item.code for item in issues})

    def test_list_markers_do_not_look_like_unbalanced_parentheses(self):
        clean = run_deterministic_checks("1) First", "1）第一")
        broken = run_deterministic_checks("Text", "文字（缺少右括号")
        self.assertNotIn("unbalanced_parentheses", {item.code for item in clean})
        self.assertIn("unbalanced_parentheses", {item.code for item in broken})

    def test_western_terms_use_token_boundaries(self):
        term = TermEntry(
            id="term_1",
            project_id="proj_1",
            source="agency",
            target="能动性",
            status=TermStatus.APPROVED,
        )
        issues = run_deterministic_checks("Interagency work matters.", "跨部门工作很重要。", [term])
        self.assertNotIn("approved_term_missing", {item.code for item in issues})

    def test_aliases_are_active_source_forms_not_passive_metadata(self):
        term = TermEntry(
            id="term_1",
            project_id="proj_1",
            source="artificial intelligence",
            target="人工智能",
            aliases=("AI", "machine intelligence"),
            status=TermStatus.APPROVED,
        )
        issues = run_deterministic_checks(
            "AI changes the workflow.",
            "智能技术改变了工作流程。",
            [term],
        )
        issue = next(item for item in issues if item.code == "approved_term_missing")
        self.assertEqual(issue.details["matched_forms"], ["AI"])
        self.assertEqual(issue.severity, IssueSeverity.ERROR)

    def test_context_keywords_select_one_sense_for_an_ambiguous_form(self):
        terms = [
            TermEntry(
                id="financial",
                project_id="proj_1",
                source="bank",
                target="银行",
                sense="金融机构",
                context_keywords=("loan", "credit"),
                status=TermStatus.APPROVED,
            ),
            TermEntry(
                id="river",
                project_id="proj_1",
                source="bank",
                target="河岸",
                sense="河流边缘",
                context_keywords=("river", "flood"),
                status=TermStatus.APPROVED,
            ),
        ]
        issues = run_deterministic_checks(
            "The bank approved the loan.",
            "河岸批准了贷款。",
            terms,
        )
        codes = {item.code for item in issues}
        self.assertIn("approved_term_missing", codes)
        self.assertNotIn("ambiguous_term_unresolved", codes)
        issue = next(item for item in issues if item.code == "approved_term_missing")
        self.assertEqual(issue.details["target"], "银行")

    def test_unresolved_sense_is_warned_without_forcing_conflicting_targets(self):
        terms = [
            TermEntry(
                id="financial",
                project_id="proj_1",
                source="bank",
                target="银行",
                sense="金融机构",
                context_keywords=("loan",),
                status=TermStatus.APPROVED,
            ),
            TermEntry(
                id="river",
                project_id="proj_1",
                source="bank",
                target="河岸",
                sense="河流边缘",
                context_keywords=("river",),
                status=TermStatus.APPROVED,
            ),
        ]
        issues = run_deterministic_checks("The bank was old.", "它很古老。", terms)
        codes = {item.code for item in issues}
        self.assertIn("ambiguous_term_unresolved", codes)
        self.assertNotIn("approved_term_missing", codes)


if __name__ == "__main__":
    unittest.main()
