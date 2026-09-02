import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from jieyi.settings import (
    LocalSecretStore,
    LocalSettingsStore,
    ModelBinding,
    ProviderSettings,
    profile_from_preset,
)
from jieyi.settings import (
    test_openai_compatible_connection as check_openai_compatible_connection,
)
from jieyi.settings import (
    test_openai_compatible_model as check_openai_compatible_model,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class SettingsTests(unittest.TestCase):
    def test_connection_check_rejects_missing_bound_model(self):
        payload = {"data": [{"id": "deepseek-v4-flash"}]}

        with (
            patch("urllib.request.urlopen", return_value=FakeResponse(payload)),
            self.assertRaisesRegex(RuntimeError, "模型列表中找不到.*old-model"),
        ):
            check_openai_compatible_connection(
                "https://api.example.com/v1",
                "secret",
                required_models=("old-model",),
            )

    def test_connection_check_accepts_bound_model(self):
        payload = {"data": [{"id": "deepseek-v4-flash"}]}

        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            result = check_openai_compatible_connection(
                "https://api.example.com/v1",
                "secret",
                required_models=("deepseek-v4-flash",),
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["stages"][-1]["id"], "binding")

    def test_saved_provider_profile_gains_new_preset_capabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "profiles": [
                            {
                                "id": "deepseek",
                                "name": "DeepSeek",
                                "provider_type": "deepseek",
                                "base_url": "https://api.deepseek.com/v1",
                                "capabilities": ["tools"],
                            }
                        ],
                        "draft": {"profile_id": "deepseek", "model": "deepseek-v4-flash"},
                    }
                ),
                encoding="utf-8",
            )

            settings = LocalSettingsStore(path).load()
            profile = settings.profiles[0]

        self.assertEqual(settings.draft.compute_mode, "economy")
        self.assertIn("thinking", profile.capabilities)
        self.assertIn("reasoning_effort", profile.capabilities)
        self.assertIn("tools", profile.capabilities)

    def test_migrates_legacy_glm_url_without_inserting_v1(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "provider_type": "custom",
                        "base_url": "https://open.bigmodel.cn/api/paas/v4",
                        "draft_model": "glm-5.3-flash",
                    }
                ),
                encoding="utf-8",
            )

            settings = LocalSettingsStore(path).load()

            self.assertEqual(settings.profiles[0].provider_type, "glm-cn")
            self.assertEqual(
                settings.profiles[0].chat_endpoint,
                "https://open.bigmodel.cn/api/paas/v4/chat/completions",
            )
            self.assertEqual(
                settings.profiles[0].models_endpoint,
                "https://open.bigmodel.cn/api/paas/v4/models",
            )

    def test_round_trips_independent_draft_and_discovery_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            store = LocalSettingsStore(path)
            settings = ProviderSettings(
                profiles=(
                    profile_from_preset("kimi", "kimi-coding"),
                    profile_from_preset("glm", "glm-cn"),
                ),
                draft=ModelBinding("kimi", "k3", "economy"),
                term_discovery=ModelBinding("glm", "glm-5.3-flash", "balanced"),
            )

            store.save(settings)
            loaded = store.load()

            self.assertEqual(loaded.draft, ModelBinding("kimi", "k3", "economy"))
            self.assertEqual(loaded.term_discovery, ModelBinding("glm", "glm-5.3-flash", "balanced"))
            self.assertEqual(len(loaded.profiles), 2)
            self.assertNotIn("api_key", path.read_text(encoding="utf-8"))

    def test_model_probe_verifies_supported_efforts_and_mode_mapping(self):
        success = {
            "choices": [{"message": {"content": "书是打开的。"}}],
            "usage": {"total_tokens": 7},
        }

        def respond(request, timeout):
            del timeout
            body = json.loads(request.data)
            effort = body.get("reasoning_effort")
            thinking = body.get("thinking")
            accepted_efforts = {"low", "medium", "high"}
            if (effort is not None and effort not in accepted_efforts) or thinking is not None:
                detail = json.dumps({"error": {"message": "unsupported reasoning control"}}).encode(
                    "utf-8"
                )
                raise urllib.error.HTTPError(
                    request.full_url, 400, "Bad Request", {}, io.BytesIO(detail)
                )
            if effort == "high":
                return FakeResponse(
                    {"choices": [{"message": {"content": None}}], "usage": {"total_tokens": 7}}
                )
            return FakeResponse(success)

        with patch("urllib.request.urlopen", side_effect=respond):
            result = check_openai_compatible_model(
                "https://api.example.com/v1/chat/completions",
                "secret",
                "verified-reasoner",
            )

        self.assertEqual(result["reasoning"]["supported_efforts"], ["low", "medium", "high"])
        self.assertEqual(result["reasoning"]["empty_efforts"], ["high"])
        self.assertTrue(any("提高输出 token" in note for note in result["notes"]))
        self.assertEqual(
            result["mode_mapping"],
            {
                "economy": "强度 low",
                "balanced": "强度 medium",
                "performance": "强度 high",
            },
        )
        self.assertTrue(result["baseline"]["visible_output"])
        self.assertEqual(result["requests"], 12)
        self.assertEqual(result["total_tokens"], 28)

    def test_model_probe_does_not_claim_silently_ignored_controls(self):
        success = {
            "choices": [{"message": {"content": "书是打开的。"}}],
            "usage": {"total_tokens": 2},
        }

        with patch("urllib.request.urlopen", return_value=FakeResponse(success)):
            result = check_openai_compatible_model(
                "https://api.example.com/v1/chat/completions",
                "secret",
                "permissive-proxy",
            )

        self.assertEqual(result["reasoning"]["kind"], "default")
        self.assertEqual(result["reasoning"]["supported_efforts"], [])
        self.assertEqual(result["mode_mapping"]["performance"], "服务端默认")
        self.assertTrue(any("静默忽略" in note for note in result["notes"]))

    def test_keychain_authorization_failure_falls_back_to_session(self):
        secrets = LocalSecretStore()
        failed = Mock(returncode=1, stderr="Unable to obtain authorization", stdout="")

        with (
            patch.object(LocalSecretStore, "_keychain_available", return_value=True),
            patch("subprocess.run", return_value=failed),
        ):
            result = secrets.set("kimi", "secret-value")
            value, source = secrets.get("kimi")

        self.assertEqual(result.source, "session")
        self.assertIn("本次运行有效", result.warning)
        self.assertEqual((value, source), ("secret-value", "session"))


if __name__ == "__main__":
    unittest.main()
