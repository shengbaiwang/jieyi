from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from jieyi.domain.models import (
    Document,
    ModelSpec,
    Project,
    Segment,
    SegmentKind,
    TranslationRequest,
)
from jieyi.protection import PlaceholderIntegrityError, ProtectedTextCodec
from jieyi.workflow.optimized import _complete_one
from jieyi.workflow.requests import initial_output_budget, project_context


class TextTranslationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12_000)
    source_lang: str = Field(default="auto", min_length=1, max_length=32)
    target_lang: str = Field(default="zh-CN", min_length=1, max_length=32)
    style_guide: str = Field(default="", max_length=12_000)
    profile_id: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=256)
    compute_mode: Literal["economy", "balanced", "performance"] = "economy"


def install_text_routes(app: FastAPI, providers, settings_store) -> None:
    @app.post("/translate/text")
    async def translate_text(body: TextTranslationRequest):
        if not body.text.strip():
            raise HTTPException(status_code=422, detail="请输入要翻译的文本。")
        if not body.model.strip() or not body.source_lang.strip() or not body.target_lang.strip():
            raise HTTPException(status_code=422, detail="请选择语言和模型。")
        settings = settings_store.load()
        profile = settings.profile(body.profile_id)
        if profile is None:
            raise HTTPException(status_code=422, detail="模型连接不存在，请重新选择。")
        if profile.auth_required and not settings_store.get_api_key(profile.id)[0]:
            raise HTTPException(status_code=422, detail="请先在模型配置中为此连接设置 API 密钥。")
        try:
            provider = providers.get(profile.registry_name)
        except KeyError as exc:
            raise HTTPException(status_code=422, detail="模型连接尚未就绪，请检查模型配置。") from exc

        # These request objects stay in memory. No project, book, segment or job is persisted.
        project = Project(
            id="text", name="文本翻译",
            source_lang=("auto (detect the source language)" if body.source_lang == "auto"
                         else body.source_lang.strip()),
            target_lang=body.target_lang.strip(), domain="general",
            style_guide=body.style_guide.strip(),
        )
        document = Document("text", project.id, "文本翻译", "txt", "")
        protected = ProtectedTextCodec().encode(body.text)
        segment = Segment("text", document.id, "text", 0, SegmentKind.PARAGRAPH, protected.masked)
        request = TranslationRequest(
            project, document, segment,
            project_context(project, 16_000) + "\nPreserve the source paragraph and line breaks.",
        )
        spec = ModelSpec(profile.registry_name, body.model.strip())
        try:
            result = await _complete_one(
                provider, request, spec,
                thinking=body.compute_mode != "economy",
                reasoning_effort={"economy": "none", "balanced": "medium", "performance": "high"}[
                    body.compute_mode
                ],
                compute_mode=body.compute_mode,
                max_tokens=initial_output_budget(request, body.compute_mode, 32_000),
                max_output_tokens=32_000,
            )
            text = protected.restore(result.text)
        except PlaceholderIntegrityError as exc:
            raise HTTPException(
                status_code=502, detail="译文中的链接、引文或格式不完整，请重试或更换模型。"
            ) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="模型响应超时，请重试。") from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail="模型未返回完整可用的译文，请重试或检查模型连接。"
            ) from exc
        usage = asdict(result)
        usage.pop("raw_response", None)
        return {**usage, "text": text, "model": spec.model, "profile_id": profile.id}
