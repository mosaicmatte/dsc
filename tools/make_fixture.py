#!/usr/bin/env python3
"""Generate a synthetic Vietnamese legal fixture — for practice and for tests.

WHY IT EXISTS
-------------
This produces raw files in BTC's ACTUAL shapes, confirmed against the Task 1/2
overview documents:

    selected-contexts.zip -> context_*.json  {"id": int, "name", "link", "passage"}
    train.json                                {"<qid>": {"question", "answer": [ids]}}
    task2_train.json                          {"<qid>": {"question", "answer": "prose"}}

so you can run the entire Phase 0-1 chain today and learn what every script does
before the real data raises the stakes.

It is also the fixture `phases/0_harness/smoke_test.py` runs against.

DELIBERATELY NON-UNIFORM
------------------------
Each law gets its own decree number and distinguishing vocabulary, and article
lengths vary. Uniform synthetic data produces tied scores for every query, which
makes the cutoff rules look broken when they are not — and hides real regressions
behind artificial ties. A third of the queries name a decree outright (the
lexical-match case, one dominant score) and the rest paraphrase (flatter scores),
so both retrieval regimes are represented.

WHAT IT IS NOT
--------------
It is not a difficulty proxy. BM25 scores ~1.0 recall here and will not on real
data. Use it to learn the mechanics, never to predict a score.

USAGE
  python tools/make_fixture.py --out data/fixture
  python phases/0_harness/ingest.py \
      --raw-corpus data/fixture/selected-contexts.zip \
      --raw-queries data/fixture/train.json --out-dir data/fixture/processed
"""
from __future__ import annotations

import argparse
import json
import os
import random
import zipfile

TOPICS = [
    ("lao động", ["nghỉ hằng năm", "hợp đồng lao động", "tiền lương làm thêm giờ",
                  "kỷ luật lao động", "bảo hiểm xã hội"]),
    ("giao thông", ["xử phạt nồng độ cồn", "tốc độ tối đa", "giấy phép lái xe",
                    "đăng kiểm xe cơ giới", "vượt đèn đỏ"]),
    ("đất đai", ["cấp giấy chứng nhận", "thu hồi đất", "chuyển mục đích sử dụng",
                 "bồi thường tái định cư", "hạn mức giao đất"]),
    ("doanh nghiệp", ["đăng ký kinh doanh", "giải thể doanh nghiệp", "vốn điều lệ",
                      "người đại diện pháp luật", "chia tách sáp nhập"]),
]
MARKS = ["nội thành", "nông thôn", "khu công nghiệp", "vùng biên giới"]


def build(n_queries: int = 60, seed: int = 11):
    rng = random.Random(seed)
    corpus, queries, task2 = [], {}, {}
    for ti, (topic, subs) in enumerate(TOPICS):
        for li in range(4):
            decree = f"{100+ti*10+li}/2019/NĐ-CP"
            mark = MARKS[li]
            body = [f"Nghị định {decree} về {topic} áp dụng tại {mark}"]
            for ai, s in enumerate(subs, 1):
                body.append(f"Điều {ai}. Quy định về {s}")
                body.append(f"1. Nội dung {s} tại {mark} thực hiện theo "
                            f"Nghị định {decree}.")
                for extra in range(rng.randint(0, 3)):
                    body.append(f"{extra+2}. Cơ quan có thẩm quyền hướng dẫn chi "
                                f"tiết về {s} trong phạm vi {mark}.")
            corpus.append({"link": f"https://thuvienphapluat.vn/van-ban/{decree}",
                           "name": f"Nghi-dinh-{decree.replace('/', '-')}",
                           "passage": "\n".join(body),
                           "id": 700 + ti * 10 + li})

    for qi in range(n_queries):
        ti, li = qi % 4, (qi // 4) % 4
        topic, subs = TOPICS[ti]
        s = subs[qi // 4 % 5]
        mark, decree = MARKS[li], f"{100+ti*10+li}/2019/NĐ-CP"
        if qi % 3 == 0:
            q = f"Nghị định {decree} quy định về {s} như thế nào?"
            rel = [str(700 + ti * 10 + li)]
        else:
            q = f"Quy định về {s} tại {mark} trong lĩnh vực {topic} ra sao?"
            rel = ([str(700 + ti * 10 + li)]
                   + ([str(700 + ti * 10 + (li + 1) % 4)] if qi % 5 == 0 else []))
        queries[str(147000 + qi)] = {"question": q, "answer": rel}
        task2[str(147000 + qi)] = {
            "question": q,
            "answer": (f"Theo Nghị định {decree} về {topic} quy định cụ thể:\n"
                       f"- Nội dung {s} tại {mark} được thực hiện theo quy định "
                       f"của pháp luật hiện hành.\n"
                       f"- Cơ quan nhà nước có thẩm quyền hướng dẫn chi tiết "
                       f"về {s}."),
        }
    return corpus, queries, task2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/fixture")
    ap.add_argument("--n-queries", type=int, default=60)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    corpus, queries, task2 = build(a.n_queries, a.seed)
    # corpus: context_*.json inside selected-contexts.zip, exactly as BTC ships it
    ctx_dir = os.path.join(a.out, "contexts")
    os.makedirs(ctx_dir, exist_ok=True)
    for i, rec in enumerate(corpus):
        json.dump(rec, open(f"{ctx_dir}/context_{i}.json", "w"), ensure_ascii=False)
    with zipfile.ZipFile(f"{a.out}/selected-contexts.zip", "w",
                         zipfile.ZIP_DEFLATED) as z:
        for i in range(len(corpus)):
            z.write(f"{ctx_dir}/context_{i}.json", f"context_{i}.json")

    # queries: JSON objects keyed by question id
    json.dump(queries, open(f"{a.out}/train.json", "w"), ensure_ascii=False, indent=1)
    json.dump(task2, open(f"{a.out}/task2_train.json", "w"),
              ensure_ascii=False, indent=1)
    test = {k: {"question": v["question"]} for k, v in list(queries.items())[:20]}
    json.dump(test, open(f"{a.out}/public_official.json", "w"),
              ensure_ascii=False, indent=1)

    print(f"wrote fixture to {a.out}/")
    for f in ("selected-contexts.zip", "train.json", "task2_train.json",
              "public_official.json"):
        print(f"  {f}")
    print(f"\n{len(corpus)} documents · {len(queries)} Task 1 queries · "
          f"{len(task2)} Task 2 questions · {len(test)} unlabelled test queries")
    print("\nNOT a difficulty proxy — BM25 scores near-perfect recall here and will\n"
          "not on real data. Use it to learn the mechanics only.")
    print(f"\nNext: see docs/walkthrough.md, or run\n"
          f"  python phases/0_harness/ingest.py "
          f"--raw-corpus {a.out}/selected-contexts.zip "
          f"--raw-queries {a.out}/train.json --out-dir {a.out}/processed")


if __name__ == "__main__":
    main()
