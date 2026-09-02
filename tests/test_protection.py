import unittest

from jieyi.protection import PlaceholderIntegrityError, ProtectedTextCodec


class ProtectionTests(unittest.TestCase):
    def test_protected_spans_round_trip_exactly(self):
        source = (
            "See `agency()` at https://example.org/a and (Smith 2020) [^4]. "
            "The DOI is 10.1234/ABC.7."
        )
        protected = ProtectedTextCodec().encode(source)
        self.assertGreaterEqual(len(protected.spans), 5)
        self.assertNotIn("https://example.org/a", protected.masked)
        self.assertEqual(protected.restore(protected.masked), source)

    def test_missing_duplicate_and_reordered_tokens_are_rejected(self):
        protected = ProtectedTextCodec().encode("(Smith 2020) then [^2]")
        first, second = protected.tokens
        with self.assertRaises(PlaceholderIntegrityError):
            protected.restore(first)
        with self.assertRaises(PlaceholderIntegrityError):
            protected.restore(first + first + second)
        with self.assertRaises(PlaceholderIntegrityError):
            protected.restore(second + first)

    def test_surplus_duplicate_token_can_be_repaired_without_moving_text(self):
        protected = ProtectedTextCodec().encode("(Smith 2020) then [^2]")
        first, second = protected.tokens
        candidate = f"{first}译文甲{first}译文乙{second}"

        repaired = protected.repair_surplus_placeholders(candidate)

        self.assertEqual(repaired, f"{first}译文甲译文乙{second}")
        self.assertIsNone(protected.repair_surplus_placeholders(first))

    def test_hallucinated_token_is_rejected_when_source_has_no_protected_span(self):
        protected = ProtectedTextCodec().encode("ordinary source")
        with self.assertRaises(PlaceholderIntegrityError):
            protected.restore("译文 [[JY_PH_9999]]")

    def test_literal_protocol_tokens_in_source_cannot_collide_with_generated_masks(self):
        source = "Literal [[JY_PH_0000]], `code`, and [[JY_PH_0001]]."
        protected = ProtectedTextCodec().encode(source)

        self.assertNotIn("[[JY_PH_0000]]", protected.masked)
        self.assertNotIn("[[JY_PH_0001]]", protected.masked)
        self.assertEqual(protected.restore(protected.masked), source)



class AtomRepairTests(unittest.TestCase):
    def setUp(self):
        self.protected = ProtectedTextCodec().encode(
            '<jy-atom data-jy-id="a">See <em>the idea</em>.</jy-atom>'
            '<jy-atom data-jy-id="b">The Book</jy-atom>'
        )

    def test_rebuilds_boundaries_and_preserves_inline_formatting_and_literal_text(self):
        import json

        from jieyi.ingestion.epub_roundtrip import parse_structured_translation
        first, second = self.protected.atom_boundaries
        tokens = self.protected.tokens
        response = json.dumps({
            second[0]: "《书名》",
            first[0]: f"参见 {tokens[1]}A & B{tokens[2]}。",
        }, ensure_ascii=False)
        restored = self.protected.restore(self.protected.assemble_atom_repair(response))
        plain, atoms = parse_structured_translation(restored, ("a", "b"))
        self.assertEqual(plain, "参见 A & B。 《书名》")
        self.assertIn("<em>A &amp; B</em>", atoms["a"][1])
        self.assertEqual(atoms["b"][0], "《书名》")

    def test_rejects_missing_extra_duplicate_empty_or_misplaced_fragment_content(self):
        import json
        first, second = self.protected.atom_boundaries
        tokens = self.protected.tokens
        valid = {first[0]: f"译文{tokens[1]}强调{tokens[2]}", second[0]: "书名"}
        bad = [
            json.dumps({first[0]: valid[first[0]]}),
            json.dumps(valid | {"extra": "不能丢弃的内容"}),
            json.dumps(valid | {second[0]: ""}),
            json.dumps(valid | {second[0]: None}),
            json.dumps(valid | {second[0]: first[1] + "书名"}),
            json.dumps(valid | {first[0]: "丢失斜体标记"}),
            json.dumps({first[0]: "译文", second[0]: valid[first[0]]}),
            '{"' + first[0] + '":"甲","' + first[0] + '":"乙"}',
            json.dumps(valid) + "越界句尾",
        ]
        for response in bad:
            with self.subTest(response=response), self.assertRaises(PlaceholderIntegrityError):
                self.protected.assemble_atom_repair(response)

    def test_strict_parser_rejects_outer_text_unknown_elements_and_nested_atoms(self):
        from jieyi.ingestion.epub_roundtrip import parse_structured_translation
        valid = '<jy-atom data-jy-id="a">甲</jy-atom><jy-atom data-jy-id="b">乙</jy-atom>'
        for value in (valid + "越界", "越界" + valid, valid + "<unexpected/>",
                      valid.replace("甲", '<jy-atom data-jy-id="c">丙</jy-atom>')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_structured_translation(value, ("a", "b"))

if __name__ == "__main__":
    unittest.main()
