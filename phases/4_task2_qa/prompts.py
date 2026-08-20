"""Prompt variants for the Task 2 reader. Ablated in ``ablate_context.py``.

Design notes, so the variants are a controlled experiment rather than a pile:

* ``minimal``  — no instruction beyond the task. The control condition. Always
                 run it: if the elaborate prompts do not beat it, they are noise.
* ``grounded`` — explicitly forbids using outside knowledge and permits "not
                 found". This is the anti-hallucination lever from the phase
                 README; it usually helps most when retrieval is weak.
* ``cited``    — requires naming the source điều. Makes a wrong citation visible,
                 which is worth points if the metric rewards grounding and worth
                 error analysis regardless.
* ``concise``  — constrains answer length. Matters when the metric is token-F1:
                 a long correct answer is penalised on precision against a short
                 gold string.

Vietnamese prompts throughout — the models are Vietnamese-trained and instruction
following degrades noticeably when the prompt language does not match the content.
"""
from __future__ import annotations

from typing import Dict, List

SYSTEM = "Bạn là trợ lý pháp lý. Chỉ trả lời dựa trên các đoạn văn bản được cung cấp."

# TODO(TEAM/phase4-B6): once `00_task2_eval_notes.md` tells you the answer format,
# add a variant that matches it exactly. If gold answers are one-line spans, a
# variant that forbids full sentences will beat all four of these. If they are
# structured (e.g. "Có/Không + căn cứ"), encode that structure in the template
# rather than hoping the model infers it.
TEMPLATES: Dict[str, str] = {
    "minimal":
        "Các đoạn văn bản pháp luật:\n{context}\n\n"
        "Câu hỏi: {question}\n"
        "Trả lời:",

    "grounded":
        "Các đoạn văn bản pháp luật:\n{context}\n\n"
        "Câu hỏi: {question}\n\n"
        "Chỉ sử dụng thông tin trong các đoạn văn bản trên. "
        "Không sử dụng kiến thức bên ngoài. "
        "Nếu các đoạn văn bản không chứa câu trả lời, hãy trả lời "
        "\"Không tìm thấy thông tin\".\n"
        "Trả lời:",

    "cited":
        "Các đoạn văn bản pháp luật:\n{context}\n\n"
        "Câu hỏi: {question}\n\n"
        "Chỉ sử dụng thông tin trong các đoạn văn bản trên. "
        "Nêu rõ điều khoản làm căn cứ.\n"
        "Trả lời (kèm căn cứ):",

    "concise":
        "Các đoạn văn bản pháp luật:\n{context}\n\n"
        "Câu hỏi: {question}\n\n"
        "Chỉ sử dụng thông tin trong các đoạn văn bản trên. "
        "Trả lời ngắn gọn, không giải thích thêm.\n"
        "Trả lời:",
}


def format_context(passages: List[str], doc_ids: List[str] | None = None,
                   max_chars: int = 1500) -> str:
    """Number the passages so a `cited` answer can refer to them unambiguously.

    ``max_chars`` truncates each passage. Truncation is a real trade-off: too
    tight and the answer sentence is cut off; too loose and a 1.5B model loses
    the question in the noise. Ablate it if context size alone does not explain
    your results.
    """
    out = []
    for i, p in enumerate(passages, 1):
        tag = f" ({doc_ids[i-1]})" if doc_ids else ""
        body = p if len(p) <= max_chars else p[:max_chars] + " …"
        out.append(f"[{i}]{tag} {body}")
    return "\n\n".join(out)


def build(question: str, passages: List[str], variant: str = "grounded",
          doc_ids: List[str] | None = None, max_chars: int = 1500) -> str:
    if variant not in TEMPLATES:
        raise ValueError(f"unknown prompt variant {variant!r}; "
                         f"have {sorted(TEMPLATES)}")
    return TEMPLATES[variant].format(
        context=format_context(passages, doc_ids, max_chars), question=question)


def as_chat(question: str, passages: List[str], variant: str = "grounded",
            doc_ids: List[str] | None = None, max_chars: int = 1500):
    """Chat format for instruct models. Use with `tokenizer.apply_chat_template`."""
    return [{"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": build(question, passages, variant, doc_ids, max_chars)}]
