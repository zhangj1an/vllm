# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression: ASR models should post-process chat completion text like STT."""

from unittest.mock import MagicMock, patch

import pytest

from vllm.entrypoints.openai.chat_completion.asr_chat_output import (
    postprocess_chat_completion_text_for_model,
)
from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration


def test_postprocess_strips_qwen3_asr_wrapper() -> None:
    cfg = MagicMock()
    raw = "language English<asr_text>hello world"
    with patch(
        "vllm.model_executor.model_loader.get_model_cls",
        return_value=Qwen3ASRForConditionalGeneration,
    ):
        out = postprocess_chat_completion_text_for_model(cfg, raw)
    assert out == "hello world"


def test_postprocess_passes_through_when_no_asr_tag() -> None:
    cfg = MagicMock()
    raw = "plain assistant reply"
    with patch(
        "vllm.model_executor.model_loader.get_model_cls",
        return_value=Qwen3ASRForConditionalGeneration,
    ):
        out = postprocess_chat_completion_text_for_model(cfg, raw)
    assert out == raw


@pytest.mark.parametrize("text", [None, ""])
def test_postprocess_handles_empty(text: str | None) -> None:
    cfg = MagicMock()
    with patch(
        "vllm.model_executor.model_loader.get_model_cls",
        return_value=Qwen3ASRForConditionalGeneration,
    ):
        out = postprocess_chat_completion_text_for_model(cfg, text)
    assert out == text


def test_postprocess_identity_for_non_transcription_model() -> None:
    class _PlainLM:
        pass

    cfg = MagicMock()
    raw = "language English<asr_text>should not strip"
    with patch(
        "vllm.model_executor.model_loader.get_model_cls",
        return_value=_PlainLM,
    ):
        out = postprocess_chat_completion_text_for_model(cfg, raw)
    assert out == raw
