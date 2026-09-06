import argparse
import time

from dyllm import SamplingParams, dLLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/data/model/Dream-v0-Instruct-7B")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.995)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--num-full-steps", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Keep every denoising step on the full path for TP comparison.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    num_full_steps = args.steps if args.full_only else args.num_full_steps

    dllm = dLLM(
        args.model_path,
        threshold=args.threshold,
        enforce_eager=True,
        tensor_parallel_size=args.tp_size,
        expert_parallel_size=args.ep_size,
    )

    sampling_params = SamplingParams(
        temperature=None,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        num_full_steps=num_full_steps,
        block_size=args.block_size,
        ignore_eos=True,
        algorithm="confidence",
    )

    prompts = [
        "Describe the water cycle in detail.",
    ]
    prompts = prompts * args.batch_size

    templated = [
        dllm.tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]

    try:
        started = time.perf_counter()
        for _ in range(args.repeats):
            outputs = dllm.generate(templated, sampling_params)
        elapsed = time.perf_counter() - started

        for p, out in zip(prompts, outputs):
            print("\nPrompt:", repr(p))
            print("Completion:", repr(out["text"]))
            print("Token IDs:", out["token_ids"])
        print("\nExecution stats:", dllm.execution_stats)
        print(f"Elapsed: {elapsed:.3f}s")
    finally:
        dllm.exit()


if __name__ == "__main__":
    main()
