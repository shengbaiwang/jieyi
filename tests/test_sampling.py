import unittest

from jieyi.ingestion import take_distributed_sample


class SamplingTests(unittest.TestCase):
    def test_long_text_is_sampled_from_beginning_middle_and_end(self):
        text = "BEGIN " + ("alpha " * 500) + "MIDDLE " + ("omega " * 500) + "END"
        sample = take_distributed_sample(
            text, total_budget=900, excerpt_count=3, minimum_excerpt_chars=200
        )
        self.assertLessEqual(len(sample.text), 900)
        self.assertEqual(sample.excerpt_count, 3)
        self.assertIn("BEGIN", sample.text)
        self.assertIn("END", sample.text)
        self.assertIn("omitted", sample.text)

    def test_short_text_is_returned_untouched(self):
        sample = take_distributed_sample("short", total_budget=100)
        self.assertEqual(sample.text, "short")
        self.assertEqual(sample.excerpt_count, 1)


if __name__ == "__main__":
    unittest.main()

