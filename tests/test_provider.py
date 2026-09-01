import http.client
import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from jieyi.domain.models import ModelSpec
from jieyi.providers.openai_compatible import OpenAICompatibleProvider, ProviderError
from jieyi.workflow.provider_responses import is_content_filtered_error


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class ProviderUsageTests(unittest.TestCase):
    def test_recognizes_glm_http_1301_safety_rejection_for_workflow_deferral(self):
        error = ProviderError(
            "Provider request failed: HTTP 400: "
            "系统检测到输入或生成内容可能包含不安全或敏感内容（代码 1301）"
        )

        self.assertTrue(is_content_filtered_error(error))

    def test_reads_standard_cached_prompt_token_details(self):
        payload = {
            "choices": [{"message": {"content": "译文"}}],
            "usage": {
                "prompt_tokens": 1_200,
                "completion_tokens": 200,
                "prompt_tokens_details": {"cached_tokens": 800},
                "completion_tokens_details": {"reasoning_tokens": 50},
            },
        }
        provider = OpenAICompatibleProvider("https://example.com/v1")
        model = ModelSpec(provider="openai-compatible", model="test")

        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            result = provider._complete_sync(
                [{"role": "user", "content": "source"}],
                model,
                thinking=False,
                reasoning_effort=None,
                max_tokens=512,
            )

        self.assertEqual(result.prompt_tokens, 1_200)
        self.assertEqual(result.completion_tokens, 200)
        self.assertEqual(result.reasoning_tokens, 50)
        self.assertEqual(result.prompt_cache_hit_tokens, 800)
        self.assertEqual(result.prompt_cache_miss_tokens, 400)

    def test_sends_explicit_reasoning_effort_including_none(self):
        payload = {"choices": [{"message": {"content": "译文"}}]}
        provider = OpenAICompatibleProvider(
            "https://example.com/v1", capabilities=("reasoning_effort",)
        )
        model = ModelSpec(provider="openai-compatible", model="reasoning-model")
        request_bodies = []

        def respond(request, timeout):
            del timeout
            request_bodies.append(json.loads(request.data))
            return FakeResponse(payload)

        with patch("urllib.request.urlopen", side_effect=respond):
            provider._complete_sync(
                [{"role": "user", "content": "source"}],
                model,
                thinking=False,
                reasoning_effort="none",
                max_tokens=512,
            )
            provider._complete_sync(
                [{"role": "user", "content": "source"}],
                model,
                thinking=True,
                reasoning_effort="high",
                max_tokens=512,
            )

        self.assertEqual([item["reasoning_effort"] for item in request_bodies], ["none", "high"])

    def test_compute_mode_falls_back_and_caches_supported_effort(self):
        payload = {"choices": [{"message": {"content": "译文"}}]}
        provider = OpenAICompatibleProvider(
            "https://example.com/v1", capabilities=("reasoning_effort",)
        )
        model = ModelSpec(provider="openai-compatible", model="vendor-new-reasoner")
        request_bodies = []

        def respond(request, timeout):
            del timeout
            body = json.loads(request.data)
            request_bodies.append(body)
            if body.get("reasoning_effort") == "none":
                raise urllib.error.HTTPError(
                    request.full_url,
                    400,
                    "Bad Request",
                    {},
                    io.BytesIO(
                        b'{"error":{"message":"reasoning_effort invalid; allowed values: minimal, low"}}'
                    ),
                )
            return FakeResponse(payload)

        with patch("urllib.request.urlopen", side_effect=respond):
            for _ in range(2):
                result = provider._complete_sync(
                    [{"role": "user", "content": "source"}],
                    model,
                    thinking=False,
                    reasoning_effort=None,
                    max_tokens=512,
                    compute_mode="economy",
                )
                self.assertEqual(result.text, "译文")

        self.assertEqual(
            [item.get("reasoning_effort") for item in request_bodies],
            ["none", "minimal", "minimal"],
        )

    def test_chinese_always_thinking_error_falls_back_and_caches_effort(self):
        payload = {"choices": [{"message": {"content": "译文"}}]}
        provider = OpenAICompatibleProvider(
            "https://example.com/v1",
            capabilities=("thinking", "reasoning_effort"),
        )
        model = ModelSpec(provider="openai-compatible", model="vendor-future-reasoner")
        request_bodies = []

        def respond(request, timeout):
            del timeout
            body = json.loads(request.data)
            request_bodies.append(body)
            if body.get("reasoning_effort") == "medium":
                detail = json.dumps(
                    {
                        "error": {
                            "message": "该模型始终思考，不支持关闭思考；请使用 low、high 或 max。",
                            "code": 1210,
                        }
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                raise urllib.error.HTTPError(
                    request.full_url, 400, "Bad Request", {}, io.BytesIO(detail)
                )
            return FakeResponse(payload)

        with patch("urllib.request.urlopen", side_effect=respond):
            for _ in range(2):
                result = provider._complete_sync(
                    [{"role": "user", "content": "source"}],
                    model,
                    thinking=None,
                    reasoning_effort=None,
                    max_tokens=512,
                    compute_mode="balanced",
                )
                self.assertEqual(result.text, "译文")

        self.assertEqual(
            [item.get("reasoning_effort") for item in request_bodies],
            ["medium", "low", "low"],
        )

    def test_preserves_empty_null_content_for_workflow_retry(self):
        payload = {
            "choices": [
                {
                    "message": {"content": None, "reasoning_content": "hidden work"},
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 512,
                "completion_tokens_details": {"reasoning_tokens": 512},
            },
        }
        provider = OpenAICompatibleProvider("https://example.com/v1")
        model = ModelSpec(provider="openai-compatible", model="reasoning-model")

        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            result = provider._complete_sync(
                [{"role": "user", "content": "source"}],
                model,
                thinking=False,
                reasoning_effort=None,
                max_tokens=512,
            )

        self.assertEqual(result.text, "")
        self.assertEqual(result.completion_tokens, 512)
        self.assertEqual(result.reasoning_tokens, 512)

    def test_reads_text_from_content_blocks(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "第一段"},
                            {"type": "text", "text": "第二段"},
                        ]
                    }
                }
            ]
        }
        provider = OpenAICompatibleProvider("https://example.com/v1")
        model = ModelSpec(provider="openai-compatible", model="block-content-model")

        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            result = provider._complete_sync(
                [{"role": "user", "content": "source"}],
                model,
                thinking=False,
                reasoning_effort=None,
                max_tokens=512,
            )

        self.assertEqual(result.text, "第一段\n第二段")

    def test_preserves_provider_http_error_detail(self):
        error = urllib.error.HTTPError(
            "https://example.com/v1/chat/completions",
            400,
            "Bad Request",
            {},
            io.BytesIO(
                json.dumps(
                    {"error": {"message": "Unknown model", "code": "model_not_found"}}
                ).encode()
            ),
        )
        provider = OpenAICompatibleProvider("https://example.com/v1")
        model = ModelSpec(provider="openai-compatible", model="missing")

        with (
            patch("urllib.request.urlopen", side_effect=error),
            self.assertRaisesRegex(ProviderError, "Unknown model.*model_not_found"),
        ):
            provider._complete_sync(
                [{"role": "user", "content": "source"}],
                model,
                thinking=False,
                reasoning_effort=None,
                max_tokens=512,
            )

    def test_retries_incomplete_response_body(self):
        payload = {"choices": [{"message": {"content": "译文"}}]}
        provider = OpenAICompatibleProvider("https://example.com/v1")
        model = ModelSpec(provider="openai-compatible", model="test")

        with (
            patch(
                "urllib.request.urlopen",
                side_effect=[http.client.IncompleteRead(b""), FakeResponse(payload)],
            ) as urlopen,
            patch("time.sleep"),
        ):
            result = provider._complete_sync(
                [{"role": "user", "content": "source"}],
                model,
                thinking=False,
                reasoning_effort=None,
                max_tokens=512,
            )

        self.assertEqual(result.text, "译文")
        self.assertEqual(urlopen.call_count, 2)

    def test_retries_with_temperature_one_when_model_requires_it(self):
        error = urllib.error.HTTPError(
            "https://example.com/v1/chat/completions",
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"error":{"message":"invalid temperature: only 1 is allowed"}}'),
        )
        provider = OpenAICompatibleProvider("https://example.com/v1")
        model = ModelSpec(provider="openai-compatible", model="fixed-temperature")
        temperatures = []
        payload = {"choices": [{"message": {"content": "译文"}}]}

        def respond(request, timeout):
            temperatures.append(json.loads(request.data)["temperature"])
            if len(temperatures) == 1:
                # Another concurrent request may cache the override before this one fails.
                provider._temperature_overrides[model.model] = 1.0
                raise error
            return FakeResponse(payload)

        with patch("urllib.request.urlopen", side_effect=respond) as urlopen:
            result = provider._complete_sync(
                [{"role": "user", "content": "source"}],
                model,
                thinking=False,
                reasoning_effort=None,
                max_tokens=512,
            )
            provider._complete_sync(
                [{"role": "user", "content": "next"}],
                model,
                thinking=False,
                reasoning_effort=None,
                max_tokens=512,
            )

        self.assertEqual(result.text, "译文")
        self.assertEqual(temperatures, [0.1, 1.0, 1.0])
        self.assertEqual(urlopen.call_count, 3)

    def test_negotiates_fractional_fixed_temperature_and_caches_per_model(self):
        error = urllib.error.HTTPError(
            "https://example.com/v1/chat/completions",
            400,
            "Bad Request",
            {},
            io.BytesIO(
                b'{"error":{"message":"invalid temperature: only 0.6 is allowed for this model"}}'
            ),
        )
        provider = OpenAICompatibleProvider("https://example.com/v1")
        model = ModelSpec(provider="openai-compatible", model="fractional-temperature")
        temperatures = []
        payload = {"choices": [{"message": {"content": "译文"}}]}

        def respond(request, timeout):
            del timeout
            temperatures.append(json.loads(request.data)["temperature"])
            if len(temperatures) == 1:
                raise error
            return FakeResponse(payload)

        with patch("urllib.request.urlopen", side_effect=respond) as urlopen:
            result = provider._complete_sync(
                [{"role": "user", "content": "source"}],
                model,
                thinking=False,
                reasoning_effort=None,
                max_tokens=512,
            )
            provider._complete_sync(
                [{"role": "user", "content": "next"}],
                model,
                thinking=False,
                reasoning_effort=None,
                max_tokens=512,
            )

        self.assertEqual(result.text, "译文")
        self.assertEqual(temperatures, [0.1, 0.6, 0.6])
        self.assertEqual(urlopen.call_count, 3)


if __name__ == "__main__":
    unittest.main()
