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
* ``concise``  — constrains answer length. NOTE: the official metric is METEOR,
                 which is recall-weighted (NLTK uses alpha=0.9), and the gold
                 answers are long structured prose. So `concise` is the variant
                 most likely to LOSE here — it is kept as a control, to prove
                 that brevity hurts rather than assuming it.

Vietnamese prompts throughout — the models are Vietnamese-trained and instruction
following degrades noticeably when the prompt language does not match the content.
"""
from __future__ import annotations

from typing import Dict, List

SYSTEM = "Bạn là trợ lý pháp lý. Chỉ trả lời dựa trên các đoạn văn bản được cung cấp."

# The gold answers are long, structured prose that names its legal basis, e.g.
#   "Theo Điều 37 Nghị định 153/2020/NĐ-CP, được sửa đổi bởi khoản 26 Điều 1
#    Nghị định 65/2022/NĐ-CP quy định cụ thể:
#    - Tuân thủ quy định của pháp luật chứng khoán ...
#    - Thực hiện chế độ báo cáo ..."
# METEOR rewards covering the reference's content in the reference's order, so
# `mimic` below deliberately reproduces that shape. Measure it against the others.
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

    # Mirrors the gold answers' own structure: "Theo <căn cứ> quy định:" then
    # bullet points. Usually the strongest variant under METEOR.
    "mimic":
        "Các đoạn văn bản pháp luật:\n{context}\n\n"
        "Câu hỏi: {question}\n\n"
        "Chỉ sử dụng thông tin trong các đoạn văn bản trên. Trả lời theo đúng "
        "cấu trúc sau:\n"
        "Bắt đầu bằng \"Theo <điều, khoản, tên văn bản> quy định:\", sau đó "
        "liệt kê đầy đủ các nội dung liên quan dưới dạng gạch đầu dòng, "
        "giữ nguyên cách diễn đạt của văn bản gốc.\n"
        "Trả lời:",

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
