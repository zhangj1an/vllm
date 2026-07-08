# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Demo: EarlyTom video token compression on Qwen2.5-VL.

Usage:
    python earlytom_demo.py --mode baseline
    python earlytom_demo.py --mode evs --pruning-rate 0.75
    VLLM_EARLYTOM=1 python earlytom_demo.py --mode earlytom --pruning-rate 0.75
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

VIDEO_TOKEN_ID = 151656  # <|video_pad|>


def load_video(path: str, num_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = set(np.linspace(0, total - 1, num_frames).astype(int).tolist())
    frames = []
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        if i in idxs:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "evs", "earlytom"])
    parser.add_argument("--pruning-rate", type=float, default=0.75)
    parser.add_argument(
        "--video", default="/workspace/EarlyTom/LLaVA-NeXT/docs/jobs.mp4"
    )
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.mode == "earlytom":
        os.environ["VLLM_EARLYTOM"] = "1"

    from vllm import LLM, SamplingParams

    frames = load_video(args.video, args.num_frames)
    print(f"video frames: {frames.shape}")

    llm = LLM(
        model=args.model,
        max_model_len=16384,
        limit_mm_per_prompt={"video": 1},
        video_pruning_rate=(None if args.mode == "baseline" else args.pruning_rate),
        enforce_eager=False,
        gpu_memory_utilization=0.85,
        # A fully-cached repeat of the same video prompt skips the encoder
        # and the mrope recomputation; keep runs comparable and correct.
        enable_prefix_caching=False,
    )

    prompt = (
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
        "Describe what's happening in this video.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {"video": frames},
    }
    sampling = SamplingParams(temperature=0.0, max_tokens=128)

    # Warmup (compile/caches), then timed run.
    warmup_out = llm.generate(inputs, sampling)
    warmup_text = warmup_out[0].outputs[0].text
    t0 = time.perf_counter()
    outputs = llm.generate(inputs, sampling)
    elapsed = time.perf_counter() - t0

    out = outputs[0]
    n_video_tokens = sum(1 for t in out.prompt_token_ids if t == VIDEO_TOKEN_ID)
    result = {
        "mode": args.mode,
        "pruning_rate": None if args.mode == "baseline" else args.pruning_rate,
        "prompt_tokens": len(out.prompt_token_ids),
        "video_tokens": n_video_tokens,
        "gen_tokens": len(out.outputs[0].token_ids),
        "wall_time_s": round(elapsed, 3),
        "text": out.outputs[0].text,
        "warmup_text": warmup_text,
    }
    print("=" * 70)
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
