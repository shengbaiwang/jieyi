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


if __name__ == "__main__":
    unittest.main()
