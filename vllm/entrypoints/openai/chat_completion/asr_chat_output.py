# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Post-process decoded assistant text for speech-to-text model classes."""

from __future__ import annotations


def postprocess_chat_completion_text_for_model(
    model_config: object,
    text: str | None,
) -> str | None:
    """Normalize decoded assistant text for speech-to-text model classes.

    The OpenAI transcription API applies
    :meth:`~vllm.model_executor.models.interfaces.SupportsTranscription.post_process_output`
    to engine output; non-streaming chat completions do the same so multimodal
    ASR chat returns the same cleaned transcript string.

    Streaming chat does not run this: clients concatenate deltas, while ASR
    cleanup can remove prefixes that were already streamed (e.g.
    ``language …<asr_text>`` before the transcript).
    """
    if text is None:
        return None
    from vllm.model_executor.model_loader import get_model_cls
    from vllm.model_executor.models.interfaces import SupportsTranscription

    model_cls = get_model_cls(model_config)
    if issubclass(model_cls, SupportsTranscription):
        return model_cls.post_process_output(text)
    return text
