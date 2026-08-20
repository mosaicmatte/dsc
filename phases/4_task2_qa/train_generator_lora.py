#!/usr/bin/env python3
"""Task B7 (optional) — LoRA fine-tune the generator on train.json.

READ THIS FIRST
---------------
LoRA is PERMITTED but it does NOT reduce the base model's parameter count for
eligibility. A LoRA-tuned Qwen2.5-3B counts as 3.09B, not 30M. LoRA saves
TRAINING MEMORY, not budget. Check `python src/params.py` before starting.

WHEN IT IS WORTH THE TIME
-------------------------
Only after Tasks B5 and B6 are done. Prompt format and context size are far
cheaper levers and often move the metric more than a LoRA run on ~10k examples.
If your error analysis says the model is retrieving the right passage and still
answering in the wrong FORMAT, LoRA is the right fix. If it says the model is
hallucinating, fix retrieval or grounding instead — fine-tuning on 10k examples
will not teach it the law.

WHAT IT TRAINS ON
-----------------
(prompt built exactly as at inference, gold answer) pairs, with the loss masked
to the answer tokens only. Training on the prompt tokens too is a common bug: it
wastes capacity teaching the model to reproduce legal passages it is already
being shown.

USAGE
  python phases/4_task2_qa/train_generator_lora.py \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --run work/experiments/runs/task2-retrieval-train.jsonl --epochs 1
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(  # repo root: phases/<n>_<name>/ -> ../..
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prompts as P  # noqa: E402

from src import config, io_utils, params  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--run", required=True, help="retrieval run over TRAIN questions")
    ap.add_argument("--queries", default="data/processed/task2_train_split.jsonl")
    ap.add_argument("--corpus", default="data/processed/corpus_article.jsonl")
    ap.add_argument("--answer-field", default="answer")
    ap.add_argument("--out", default="models/generator-lora")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--prompt", default="grounded")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    print(params.report([("generator", a.model)]))
    print("(LoRA does not change the number above — it is the base model that counts)\n")

    config.set_seed(a.seed)

    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from peft import LoraConfig, get_peft_model  # type: ignore
    from transformers import (AutoModelForCausalLM, AutoTokenizer,  # type: ignore
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)

    run = io_utils.load_run(a.run)
    queries = io_utils.load_queries(a.queries)
    doc_ids, texts, _ = io_utils.load_corpus(a.corpus)
    dtext = dict(zip(doc_ids, texts))

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def build(q):
        top = run.get(q["qid"], [])[:a.top_k]
        msgs = P.as_chat(q["text"], [dtext.get(d, "") for d, _ in top],
                         a.prompt, [d for d, _ in top])
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
        except Exception:
            return P.build(q["text"], [dtext.get(d, "") for d, _ in top],
                           a.prompt, [d for d, _ in top])

    rows = []
    for q in queries:
        gold = q.get(a.answer_field)
        if not gold:
            continue
        prompt = build(q)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(str(gold) + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p_ids + a_ids)[:a.max_length]
        # mask the prompt: loss only on the answer tokens
        labels = ([-100] * len(p_ids) + a_ids)[:a.max_length]
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                     "labels": labels})
    if not rows:
        raise SystemExit(f"no examples — is --answer-field '{a.answer_field}' right?")
    print(f"{len(rows)} training examples, loss masked to answer tokens only")

    model = AutoModelForCausalLM.from_pretrained(
        a.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None)
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr,
        logging_steps=10, save_strategy="epoch", report_to=[],
        bf16=torch.cuda.is_available(), seed=a.seed)
    Trainer(model=model, args=args, train_dataset=Dataset.from_list(rows),
            data_collator=DataCollatorForSeq2Seq(tok, padding=True)).train()

    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    cfg = vars(a)
    run_id = config.make_run_id(cfg, prefix="train-lora")
    config.freeze(cfg, run_id)
    print(f"\nsaved adapter -> {a.out}\nfroze config -> configs/{run_id}.yaml")
    print(f"\nEVALUATE:\n  python phases/4_task2_qa/baseline_generative.py "
          f"--model {a.out} --top-k {a.top_k} --prompt {a.prompt}")


if __name__ == "__main__":
    main()
