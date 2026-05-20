"""Minimal offline request for stepping through the vLLM request lifecycle.

Companion to /root/request(2).md and /root/vllm_debug_guide.md.

Suggested first breakpoints:
  - vllm/v1/core/sched/scheduler.py    : Scheduler.schedule()
  - vllm/v1/worker/gpu_model_runner.py : execute_model()
"""

from vllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(
        model="/root/models/Qwen3-0.6B",
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        enforce_eager=True,
    )

    prompts = ["Explain what a transformer is in one sentence."]
    sampling = SamplingParams(temperature=0.7, max_tokens=32)

    outputs = llm.generate(prompts, sampling)
    for out in outputs:
        print("PROMPT :", out.prompt)
        print("OUTPUT :", out.outputs[0].text)


if __name__ == "__main__":
    main()
