import argparse
import os
from typing import Optional, Tuple

from tiny_inference import GenerationConfig, TinyQwenEngine, parse_messages


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tiny Qwen inference engine (CPU)")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-0.8B",
        help="Model name or local path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help='User text; formatted as chat message body (apply_chat_template).',
    )
    parser.add_argument(
        "--messages",
        type=str,
        help="JSON string for chat messages. Overrides --prompt if provided.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--do-sample", type=parse_bool, default=True)
    parser.add_argument(
        "--stream",
        type=parse_bool,
        default=False,
        help="Stream output tokens to stdout.",
    )
    parser.add_argument(
        "--benchmark",
        type=parse_bool,
        default=False,
        help="Print basic tokens/sec benchmarking info.",
    )
    parser.add_argument(
        "--enable-thinking",
        type=parse_bool,
        default=False,
        help="Enable thinking/reasoning mode. Default is off (no thinking).",
    )
    return parser


def resolve_input(prompt: Optional[str], messages_json: Optional[str]) -> tuple[Optional[str], Optional[list]]:
    """Returns (prompt, messages) for TinyQwenEngine. Prefer messages so chat_template is used."""
    if messages_json:
        return None, parse_messages(messages_json)
    if prompt is not None:
        # Match demo-style multimodal message body; works for text-only Qwen3.5 instruct.
        return None, [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
    return None, None


def main() -> None:
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    parser = build_arg_parser()
    args = parser.parse_args()
    prompt, messages = resolve_input(args.prompt, args.messages)

    engine = TinyQwenEngine(args.model)
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        enable_thinking=args.enable_thinking,
    )

    if args.stream:
        for piece in engine.generate_stream(
            prompt=prompt,
            messages=messages,
            gen_config=gen_config,
        ):
            print(piece, end="", flush=True)
        print()
        return

    result = engine.generate(
        prompt=prompt,
        messages=messages,
        gen_config=gen_config,
        benchmark=args.benchmark,
    )
    print(result["text"])
    if result.get("tool_calls"):
        print(result["tool_calls"])
    if result.get("metrics"):
        metrics = result["metrics"]
        print(f"elapsed_s={metrics['elapsed_s']:.3f} tokens_per_s={metrics['tokens_per_s']:.2f}")

if __name__ == "__main__":
    main()
