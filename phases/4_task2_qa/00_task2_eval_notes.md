# Task B1 — Task 2 evaluation code notes (ANSWER IN WRITING)

> **TODO(TEAM/phase4-B1): answer every question below before writing any reader code.**
>
> The metric is already confirmed: **METEOR primary, ROUGE-L secondary**, macro-averaged,
> over plain `.split()` tokens (BTC's scorer has the Vietnamese tokenizer commented out).
> Their code is vendored at
> [`../0_harness/btc_eval/scoring_legalqa.py`](../0_harness/btc_eval/scoring_legalqa.py)
> and summarised in
> [`docs/reference/09_official_rules.md`](../../docs/reference/09_official_rules.md) §5.
> Use this worksheet for the parts only the real data can answer.

Same discipline as Phase 0. Line references, not recollections. **Do this before writing
any reader code** — the answer format determines whether Baseline A is even possible.

---

## Q1. What is the answer field, exactly?

```jsonc
 "82051": {
        "question": "Vận chuyển động vật ra khỏi địa bàn cấp tỉnh mà không có Giấy chứng nhận kiểm dịch động vật, sản phẩm động vật thì bị xử phạt thế nào?",
        "answer": "Căn cứ khoản 3, khoản 5 Điều 17 Nghị định 90/2017/NĐ-CP, được sửa đổi bởi điểm a khoản 9 Điều 3 Nghị định 07/2022/NĐ-CP, khoản 7 Điều 2 Nghị định 04/2020/NĐ-CP quy định về vi phạm quy định chung về Giấy chứng nhận kiểm dịch động vật, sản phẩm động vật vận chuyển ra khỏi địa bàn cấp tỉnh như sau:\nVi phạm quy định chung về Giấy chứng nhận kiểm dịch động vật, sản phẩm động vật vận chuyển ra khỏi địa bàn cấp tỉnh\n1. Phạt tiền từ 4.000.000 đồng đến 5.000.000 đồng đối với hành vi mua bán, tẩy xóa, sửa chữa Giấy chứng nhận kiểm dịch động vật, sản phẩm động vật.\n2. Phạt tiền từ 5.000.000 đồng đến 6.000.000 đồng đối với hành vi cho thuê, cho mượn, thuê, mượn Giấy chứng nhận kiểm dịch động vật, sản phẩm động vật.\n3. Phạt tiền từ 6.000.000 đồng đến 8.000.000 đồng đối với hành vi không có Giấy chứng nhận kiểm dịch động vật, sản phẩm động vật.\n4. Hình thức xử phạt bổ sung:\nTịch thu Giấy chứng nhận kiểm dịch động vật, sản phẩm động vật đối với hành vi mua bán quy định tại khoản 1, khoản 2 Điều này.\n5. Biện pháp khắc phục hậu quả:\na) Buộc kiểm dịch lại động vật, sản phẩm động vật đối với hành vi vi phạm quy định tại khoản 3 Điều này (trừ giống động vật thủy sản);\nb) Buộc tiêu hủy động vật, sản phẩm động vật đối với hành vi vi phạm quy định tại khoản 3 Điều này trong trường hợp là giống động vật thủy sản; trong trường hợp kiểm dịch lại phát hiện động vật mắc bệnh, sản phẩm động vật mang mầm bệnh truyền nhiễm nguy hiểm thuộc Danh mục bệnh động vật phải công bố dịch.\nTheo đó, người vận chuyển động vật ra khỏi địa bàn cấp tỉnh mà không có Giấy chứng nhận kiểm dịch động vật, sản phẩm động vật với mức phạt tiền từ 6.000.000 đồng đến 8.000.000 đồng.\nĐồng thời người vi phạm còn bị buộc kiểm dịch lại động vật, sản phẩm động vật đối với hành vi vi phạm, trừ giống động vật thủy sản.\nVà buộc tiêu hủy động vật trong trường hợp là giống động vật thủy sản. Trong trường hợp kiểm dịch lại phát hiện động vật mắc bệnh, sản phẩm động vật mang mầm bệnh truyền nhiễm nguy hiểm thuộc Danh mục bệnh động vật phải công bố dịch."
    },
```

> Field name: "answer"
>
> Type (span / free text / multiple choice / list of ids): free text.
>
> Line ref in the data overview: phases/0_harness/btc_eval/scoring_legalqa.py:38

## Q2. Are gold answers verbatim substrings of the retrieved passages?

Check mechanically, do not eyeball it:
```bash
python -c "
import json,sys
d=json.load(open('data/raw/<task2_train>.json'))
hit=sum(1 for r in d if r['<answer_field>'] in r.get('<context_field>',''))
print(f'{hit}/{len(d)} answers appear verbatim in their context')"
```

> ANSWER: 0/1050 verbatim
>
> *Already measured on the real data (20/08/2026), so use this to check your own
> answer rather than to skip the exercise:*
> `python phases/4_task2_qa/baseline_extractive.py --oracle-ceiling` returns
> **0/1050** on the dev split. Gold answers cite the statute and restructure it;
> none is a verbatim span. Median gold length is 309 words.
>
> **If most are verbatim → extractive (Baseline A) is viable and cheap.**
> **If few are → extraction cannot reach the gold answers; go generative.**

## Q3. What is the metric?

> ANSWER (with line ref):
> - Exact match? token-F1? ROUGE? something custom?
> - Is the answer normalised before comparison (lowercase, punctuation, diacritics)?
> - If token-F1: what tokeniser? Syllable-level or word-segmented? This changes the
>   score materially in Vietnamese.

> ANSWER:
> - The metrics are **METEOR** and **ROUGE-L**. (Line ref: phases/0_harness/btc_eval/scoring_legalqa.py:6-8)
> - The answer is **NOT** normalised before comparison. (Line ref: phases/0_harness/btc_eval/scoring_legalqa.py:28)

## Q4. Output format

```jsonc
{"9001": {"answer": "Theo Điều 37 ... quy định cụ thể: - ..."}}
```
> ANSWER:
> - Filename: submission.json  Zipped: submission.zip (Line ref: make_submission_task2.py:7)
> - Must every question appear? Yes. If otherwise, the scorer throws an exception, causing the submission to fail. (Line ref: scoring_legalqa.py:48-50)
> - Is an empty answer allowed, and what does it score? Yes, but it scores 0.0.
> - Is a supporting-passage id also required? No. Only the text answer string is required. (phases/0_harness/btc_eval/scoring_legalqa.py:38)

## Q5. Is retrieval scored separately, or only the final answer?

> ANSWER: Only the final answer is scored.
>
> If only the answer is scored, retrieval quality is still your ceiling — measure it
> anyway (Task B2), you just will not be graded on it directly.

---

## Decision, based on the above

> Approach: **generative**
> Because: The gold answers are not verbatim strings of retrieval passages (0/1050). Thus, extraction can not reach gold answers.
