# Model registration — ready to paste

> Form: **https://forms.gle/HWE7tcxzWq63Kxv28**
> ("Biểu Mẫu Đăng Ký Pretrained Models DSC@UIT 2026")
>
> Window 06/08 → 18/09. **Resubmittable** — you can send this again later to add more.
> Compiled 20/08/2026 against BTC's public approved list. Everything already on that
> list is deliberately **excluded** — BTC says approved models need no re-registration.

---

## Field 1 — Email sinh viên trưởng nhóm

```
khoi.nguyenhoangkhoi@hcmut.edu.vn
```
*(must be the email registered with BTC — the one their announcements arrive at)*

## Field 2 — Tên đội thi

```
VNU-Deep Thinkers
```

## Field 3 — Tên mô hình tiền huấn luyện  *(one per line)*

```
pyvi
dangvantuan/vietnamese-embedding
vinai/phobert-large
YuITC/vietnamese-embedding-vn-legal
kiencnt2205/vietnamese-legal-reranker-bge-base
tutran27/vietnamese-legal-phapdien-embedding-v2
bqbbao6/vietnamese-legal-embedding-wsgm
Merikatorihuhu/SimCSE-finetuned-vietnamese-legal-documents
```

## Field 4 — Link mô hình  *(one per line, same order)*

```
https://github.com/trungtv/pyvi
https://huggingface.co/dangvantuan/vietnamese-embedding
https://huggingface.co/vinai/phobert-large
https://huggingface.co/YuITC/vietnamese-embedding-vn-legal
https://huggingface.co/kiencnt2205/vietnamese-legal-reranker-bge-base
https://huggingface.co/tutran27/vietnamese-legal-phapdien-embedding-v2
https://huggingface.co/bqbbao6/vietnamese-legal-embedding-wsgm
https://huggingface.co/Merikatorihuhu/SimCSE-finetuned-vietnamese-legal-documents
```

**Check the two blocks line up — 8 lines each, in the same order.** The form pairs them
by position.

---

## Why each one is on the list

| # | Model | Role | Params | Licence | Why |
|---|---|---|---|---|---|
| 1 | `pyvi` | word segmenter | n/a (rule-based) | MIT | **The one we actually need.** It is this repo's default segmenter and it is *not* on the approved list, though `underthesea` and `VnCoreNLP` are. Registering it keeps our defaults legal. |
| 2 | `dangvantuan/vietnamese-embedding` | bi-encoder | 135M | Apache-2.0 | PhoBERT-base embedder, 1.6M downloads. Small — leaves budget for a generator. Already in our model registry. |
| 3 | `vinai/phobert-large` | backbone | ~370M | MIT | Larger PhoBERT if we want to fine-tune our own bi-encoder. `phobert-base` and `-base-v2` are approved; `-large` is not. |
| 4 | `YuITC/vietnamese-embedding-vn-legal` | bi-encoder | 568M | ⚠️ none declared | **Highest-value entry.** Fine-tuned *from* `AITeamVN/Vietnamese_Embedding` (already approved) on 105,683 Vietnamese **legal** pairs. Same architecture as our planned retriever, tuned on our exact domain. |
| 5 | `kiencnt2205/vietnamese-legal-reranker-bge-base` | **cross-encoder** | 278M | ⚠️ none declared | Fills a real gap: BTC's approved list has **no Vietnamese legal-domain reranker**. From `BAAI/bge-reranker-base`, BCE-trained — which matches our Phase 3 objective — and half the size of bge-reranker-v2-m3. |
| 6 | `tutran27/vietnamese-legal-phapdien-embedding-v2` | bi-encoder | 568M | ⚠️ none declared | BGE-M3 fine-tuned on 183k + 50k pháp điển (legal-code) pairs. A second legal retriever to ablate against #4. |
| 7 | `bqbbao6/vietnamese-legal-embedding-wsgm` | bi-encoder | ~278M | ⚠️ none declared | The **word-segmented** variant of `bqbbao6/vietnamese-legal-embedding` (base version already approved). Directly tests our segmentation ablation on a legal model. |
| 8 | `Merikatorihuhu/SimCSE-finetuned-vietnamese-legal-documents` | bi-encoder | 135M | ⚠️ none declared | Small PhoBERT legal embedder (120k pairs). Cheapest legal-domain option; useful if we need budget for a 3B generator. |

All parameter counts read from the model cards on 20/08/2026.

## ⚠️ The licence caveat — say this to BTC

Entries 4–8 **do not declare a licence on their model card**. BTC requires licences that
permit research / educational / non-commercial use. They may reject them on that basis.

Two honest options:
- Submit as-is and let BTC decide. Registration is free and resubmittable, and their
  answer tells you where you stand.
- Or add a line to the model-name box: *"Các mô hình 4–8 chưa khai báo license trên
  model card; nhóm xin BTC xác nhận có được sử dụng hay không."*
  ("Models 4–8 do not declare a licence on their card; we ask BTC to confirm whether
  they may be used.")

The second is better. It is one sentence, it shows you checked, and it gets you a
decision instead of a silent rejection.

## Deliberately NOT registered

| Model | Why not |
|---|---|
| Everything in `src/dense.REGISTRY` except #2 | already on BTC's approved list — re-registering is noise |
| `CATI-AI/Qwen3-Embedding-0.6B-vietnamese-legal-v3`, `CATI-AI/Qwen3-Reranker-0.6B-vietnamese-legal` | **gated repos** — you must request access, and BTC cannot verify them either. Register only if you get access and still want them. |
| `luanngo/Qwen3-4B-VietNamese-Legal-Chat`, `thangvip/qwen3-4b-*` | 4B — at or over the ceiling, and would consume the whole budget alone |
| `jajajou/vietnamese-legal-qwen2.5-7b`, `phamff/vietnamese-legal-7b-v1` | 7B — over the ceiling. Quantization does not help; BTC says so explicitly |

## After you submit

1. Log it in [`00_model_registration.md`](00_model_registration.md) §5 with today's date.
2. Watch the [approved list](https://docs.google.com/spreadsheets/d/1c5jzsYezWho1WGLRfMKWOaFPLIk_GTnXP5vV8AOWM2Q/edit)
   for these to appear. **Do not submit a run using them until they do.**
3. When `pyvi` is approved, nothing changes — our defaults become legal.
   If BTC rejects it, switch every command to `--segmenter underthesea`.
4. Re-check the list weekly; other teams keep adding models you may want.
