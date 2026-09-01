import unittest

from jieyi.domain.reasoning import normalize_compute_mode, resolve_reasoning_control


class ReasoningPolicyTests(unittest.TestCase):
    def test_migrates_vendor_levels_to_user_intent(self):
        self.assertEqual(normalize_compute_mode("none"), "economy")
        self.assertEqual(normalize_compute_mode("medium"), "balanced")
        self.assertEqual(normalize_compute_mode("xhigh"), "performance")

    def test_known_models_use_their_own_supported_scale(self):
        features = {"reasoning_effort"}
        self.assertEqual(
            [
                resolve_reasoning_control("glm-5.3-flash", features, mode).effort_candidates[0]
                for mode in ("economy", "balanced", "performance")
            ],
            ["low", "high", "max"],
        )
        self.assertEqual(
            resolve_reasoning_control("gpt-5.6-sol", features, "performance").effort_candidates[0],
            "max",
        )
        self.assertEqual(
            resolve_reasoning_control("gpt-5.5-pro", features, "economy").effort_candidates[0],
            "medium",
        )
        self.assertEqual(
            resolve_reasoning_control("gpt-oss-120b", features, "performance").effort_candidates[0],
            "high",
        )

    def test_unknown_models_use_portable_fallback_sequences(self):
        control = resolve_reasoning_control("vendor-new-reasoner", {"reasoning_effort"}, "economy")
        self.assertEqual(control.effort_candidates, ("none", "minimal", "low", None))
        self.assertEqual(control.source, "adaptive")

    def test_thinking_only_models_map_mode_to_boolean(self):
        economy = resolve_reasoning_control("local-model", {"thinking"}, "economy")
        performance = resolve_reasoning_control("local-model", {"thinking"}, "performance")
        self.assertFalse(economy.thinking)
        self.assertTrue(performance.thinking)
        self.assertEqual(performance.effort_candidates, (None,))


if __name__ == "__main__":
    unittest.main()
