"""Generation backends.

Two paths, deliberately:

- **vLLM** for the BF16 student. Fast, and the student is the model generated from most often
  (benchmarks before and after training, plus trajectory rollouts).
- **Transformers ``generate()``** for teachers. Slower, but it uses the exact bitsandbytes loader
  from :mod:`opd.models` that OPD itself uses, so a benchmarked teacher is literally the teacher
  that supervises. Routing quantized teachers through vLLM would change the teacher implementation
  between measurement and use.

Both take pre-rendered prompt token ids. vLLM builds its own tokenizer internally and would not see
the non-thinking shim, so ids -- not text -- are the safe interchange format.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass(frozen=True)
class Completion:
    index: int
    token_ids: list[int]
    text: str
    finish_reason: str

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def load_vllm():
    try:
        from vllm import LLM, SamplingParams
    except ImportError as error:  # pragma: no cover - depends on the Linux GPU environment
        raise RuntimeError("vLLM is not installed. Run: uv sync --extra vllm") from error
    return LLM, SamplingParams


def build_vllm_engine(
    model_id: str,
    revision: str,
    gpu_memory_utilization: float,
    max_model_length: int,
    seed: int,
    dtype: str = "bfloat16",
):  # pragma: no cover - requires a GPU
    """Build a vLLM engine.

    ``dtype`` must be ``"auto"`` for a pre-quantized checkpoint: AWQ ships fp16 and its kernels are
    built for it, so forcing bfloat16 either errors or silently changes the arithmetic. vLLM reads
    ``quantization_config`` from the checkpoint itself, so the quantization format needs no flag.
    """
    llm_class, _ = load_vllm()
    return llm_class(
        model=model_id,
        revision=revision,
        tokenizer=model_id,
        tokenizer_revision=revision,
        dtype=dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_length,
        seed=seed,
        generation_config="vllm",
    )


def generate_vllm(
    engine: Any,
    prompt_token_ids: list[list[int]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seeds: list[int] | None = None,
) -> list[Completion]:  # pragma: no cover - requires a GPU
    """Generate with vLLM.

    ``seeds`` assigns a per-request seed so a result depends on its prompt rather than on batch
    composition or restart order.
    """
    _, sampling_params_class = load_vllm()
    if seeds is not None and len(seeds) != len(prompt_token_ids):
        raise ValueError("seeds must be the same length as prompt_token_ids")

    def params(index: int) -> Any:
        return sampling_params_class(
            n=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p if temperature > 0 else 1.0,
            top_k=top_k if top_k > 0 else -1,
            seed=None if seeds is None else seeds[index],
            skip_special_tokens=False,
        )

    outputs = engine.generate(
        [{"prompt_token_ids": ids} for ids in prompt_token_ids],
        [params(index) for index in range(len(prompt_token_ids))],
    )
    completions: list[Completion] = []
    for index, output in enumerate(outputs):
        best = output.outputs[0]
        completions.append(
            Completion(
                index=index,
                token_ids=list(best.token_ids),
                text=best.text,
                finish_reason=str(best.finish_reason),
            )
        )
    return completions


@torch.no_grad()
def generate_hf(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_token_ids: list[list[int]],
    max_new_tokens: int,
    batch_size: int,
    greedy: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 0,
    progress: bool = True,
) -> tuple[list[Completion], float]:
    """Generate with Transformers. Returns the completions and the elapsed seconds.

    Prompts are left-padded (the tokenizer is configured that way in :func:`opd.models
    .load_tokenizer`) so every sequence in a batch ends at the same position and the generated
    tokens start at a single, known offset.

    Per-batch progress goes to stderr so stdout stays a clean JSON document. This path can run for
    tens of minutes on a quantized teacher, and the per-batch numbers are what tell you whether to
    change the batch size: ``steps`` is how many decode steps the batch actually ran, and
    ``generate()`` only stops when *every* sequence in the batch has finished, so ``steps`` near
    the cap while ``mean_new`` is far below it means stragglers are holding the batch open.
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    completions: list[Completion] = []

    # Group by length so a batch is not dominated by padding for its longest member.
    order = sorted(range(len(prompt_token_ids)), key=lambda i: len(prompt_token_ids[i]))
    total_batches = (len(order) + batch_size - 1) // batch_size
    batch_number = 0
    started = time.perf_counter()
    for start in range(0, len(order), batch_size):
        batch_number += 1
        batch_started = time.perf_counter()
        chunk = order[start : start + batch_size]
        widest = max(len(prompt_token_ids[i]) for i in chunk)
        input_ids = torch.full((len(chunk), widest), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(chunk), widest), dtype=torch.long)
        for row, index in enumerate(chunk):
            ids = prompt_token_ids[index]
            input_ids[row, widest - len(ids) :] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, widest - len(ids) :] = 1

        generated = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=None if greedy else temperature,
            top_p=None if greedy else top_p,
            top_k=None if greedy else (top_k or None),
            pad_token_id=pad_id,
            use_cache=True,
        )
        steps = int(generated.shape[1] - widest)
        batch_new_tokens = 0
        finished_early = 0
        for row, index in enumerate(chunk):
            new_tokens = generated[row, widest:].tolist()
            # generate() right-pads finished sequences; drop everything from the first EOS on.
            if eos_id in new_tokens:
                new_tokens = new_tokens[: new_tokens.index(eos_id) + 1]
            finish_reason = "length" if len(new_tokens) >= max_new_tokens else "stop"
            batch_new_tokens += len(new_tokens)
            finished_early += finish_reason == "stop"
            completions.append(
                Completion(
                    index=index,
                    token_ids=new_tokens,
                    text=tokenizer.decode(new_tokens, skip_special_tokens=True),
                    finish_reason=finish_reason,
                )
            )

        if progress:
            batch_seconds = time.perf_counter() - batch_started
            done = time.perf_counter() - started
            eta = done / batch_number * (total_batches - batch_number)
            peak = f"{torch.cuda.max_memory_allocated() / 1024**3:5.1f}" if torch.cuda.is_available() else "  n/a"
            print(
                f"[gen] batch {batch_number:>3}/{total_batches:<3}"
                f" n={len(chunk):<3} prompt={widest:<5}"
                f" steps={steps:<5} mean_new={batch_new_tokens / len(chunk):>6.0f}"
                f" stop={finished_early:>3}/{len(chunk):<3}"
                f" {batch_seconds:>6.1f}s"
                f" {batch_new_tokens / max(batch_seconds, 1e-9):>7.1f} tok/s"
                f" peak={peak} GiB"
                f" eta={eta / 60:>5.1f}m",
                file=sys.stderr,
                flush=True,
            )

    elapsed = time.perf_counter() - started
    if progress:
        total_new = sum(len(item.token_ids) for item in completions)
        print(
            f"[gen] done {len(completions)} completions,"
            f" {total_new} tokens in {elapsed / 60:.1f}m"
            f" ({total_new / max(elapsed, 1e-9):.1f} tok/s overall)",
            file=sys.stderr,
            flush=True,
        )

    completions.sort(key=lambda item: item.index)
    return completions, elapsed
