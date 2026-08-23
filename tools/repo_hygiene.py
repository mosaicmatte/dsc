#!/usr/bin/env python3
"""Chặn những thứ không bao giờ được vào repo. Chạy tự động trong CI mỗi PR.

    python tools/repo_hygiene.py

Ba lỗi dưới đây, một khi đã commit và push, rất khó gỡ ra — lịch sử git giữ lại
file mãi mãi kể cả sau khi xoá. Rẻ nhất là chặn ngay từ Pull Request.

  1. Dữ liệu BTC (455 MB), trọng số mô hình, index — không bao giờ vào repo.
  2. File lớn bất thường lọt qua .gitignore.
  3. File nộp bài sai định dạng — Codabench sẽ chấm 0 mà không báo gì.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

FORBIDDEN_PREFIX = ("data/raw/", "data/processed/", "models/", "indexes/", ".venv/")
BIG_MB = 5.0
BIG_ALLOWED = ("work/submissions/",)


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [l for l in out.splitlines() if l.strip()]


def main() -> int:
    problems: list[str] = []
    files = tracked_files()

    for f in files:
        if any(f.startswith(p) for p in FORBIDDEN_PREFIX) and not f.endswith(".gitkeep"):
            problems.append(f"{f} — thư mục này không bao giờ được commit")
        full = os.path.join(ROOT, f)
        if not os.path.isfile(full):
            continue
        mb = os.path.getsize(full) / 1e6
        if mb > BIG_MB and not any(f.startswith(p) for p in BIG_ALLOWED):
            problems.append(f"{f} — {mb:.1f} MB, vượt ngưỡng {BIG_MB} MB")

    for f in files:
        if not (f.startswith("work/submissions/") and f.endswith(".zip")):
            continue
        try:
            with zipfile.ZipFile(os.path.join(ROOT, f)) as z:
                names = [n for n in z.namelist() if not n.endswith("/")]
                if names != ["submission.json"]:
                    problems.append(f"{f} — phải chứa đúng một submission.json, "
                                    f"đang có {names}")
                    continue
                sub = json.loads(z.read("submission.json").decode("utf-8"))
        except Exception as e:                                   # noqa: BLE001
            problems.append(f"{f} — không đọc được ({type(e).__name__})")
            continue

        bad = [q for q, v in sub.items()
               if not isinstance(v, dict) or set(v) != {"answer"}]
        if bad:
            problems.append(f'{f} — {len(bad)} câu sai dạng {{"answer": ...}}, '
                            f"ví dụ {bad[:3]}")
            continue
        over = [q for q, v in sub.items()
                if isinstance(v["answer"], list) and len(v["answer"]) > 5]
        if over:
            problems.append(f"{f} — {len(over)} câu trả về hơn 5 mã văn bản "
                            f"(bị chấm 0), ví dụ {over[:3]}")
        nonstr = [q for q, v in sub.items()
                  if isinstance(v["answer"], list)
                  and any(not isinstance(d, str) for d in v["answer"])]
        if nonstr:
            problems.append(f"{f} — {len(nonstr)} câu dùng mã văn bản không phải "
                            f"chuỗi (bị chấm 0 âm thầm), ví dụ {nonstr[:3]}")

    print(f"đã kiểm tra {len(files)} file đang được git theo dõi")
    if problems:
        print("\nVỆ SINH REPO THẤT BẠI:\n")
        for p in problems:
            print(f"  * {p}")
        print("")
        return 1
    print("vệ sinh repo: đạt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
