#!/usr/bin/env python3
"""Nộp kết quả cho nhóm bằng MỘT lệnh — kiểm tra, commit, push, mở PR.

    python tools/handoff.py

Chạy xong một thí nghiệm thì gõ lệnh trên. Script sẽ tự tìm lần chạy mới nhất
trong ``work/experiments/runs.csv``, kiểm tra mọi thứ có hợp lệ không, chọn đúng
những file cần commit, viết sẵn câu commit kèm con số, push lên nhánh của mình,
rồi in ra link mở Pull Request.

VÌ SAO CẦN SCRIPT NÀY THAY VÌ TỰ GÕ GIT
---------------------------------------
Nộp thủ công có sáu bước và mỗi bước có một cách hỏng riêng: quên
``work/configs/`` nên không ai chạy lại được; lỡ tay ``git add .`` rồi commit
nhầm 455 MB dữ liệu; commit file nộp bài mà nó sai định dạng và không ai phát
hiện cho đến khi Codabench chấm 0 điểm. Script làm đúng sáu bước đó, và từ chối
làm nếu phát hiện một trong các lỗi trên.

CÁCH DÙNG
---------
    python tools/handoff.py                    # lần chạy mới nhất
    python tools/handoff.py --run-id bm25-thu-01
    python tools/handoff.py --check            # chỉ kiểm tra, không commit gì
    python tools/handoff.py --phase 1          # kèm file run + phân tích lỗi
    python tools/handoff.py --yes              # không hỏi lại, dùng khi đã quen

NHỮNG GÌ SCRIPT TỪ CHỐI LÀM
---------------------------
* Lần chạy chưa có dòng trong ``runs.csv`` — chưa đo thì chưa có gì để nộp.
* Đang đứng trên nhánh ``main`` — làm việc trên nhánh riêng, gộp qua PR.
* Có file trong ``data/``, ``models/``, ``indexes/`` đang được stage.
* Có file lạ lớn hơn 5 MB (trừ ``work/submissions/*.zip``).
* File nộp bài không đúng định dạng BTC yêu cầu.

Xem thêm ``docs/teamwork.md``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import metrics  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_CSV = "work/experiments/runs.csv"
REPO_URL = "https://github.com/mosaicmatte/dsc"

# Thư mục không bao giờ được commit. Dữ liệu BTC 455 MB, trọng số mô hình hàng GB,
# index tự sinh lại được — cả ba đều làm repo hỏng và không ai gỡ ra được nữa.
FORBIDDEN_DIRS = ("data/", "models/", "indexes/", ".venv/")
BIG_FILE_MB = 5.0
BIG_FILE_ALLOWED = ("work/submissions/",)


# ---------------------------------------------------------------- tiện ích git
def git(*args, check=True, raw=False) -> str:
    """`raw=True` giữ nguyên khoảng trắng đầu dòng — bắt buộc cho
    `git status --porcelain`, vì cột trạng thái của nó có thể bắt đầu bằng dấu
    cách (' M Makefile'). Strip cả chuỗi sẽ ăn mất ký tự đầu của dòng đầu tiên
    và biến 'Makefile' thành 'akefile'."""
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"lệnh git thất bại: git {' '.join(args)}\n{r.stderr.strip()}")
    return r.stdout if raw else r.stdout.strip()


def changed_files() -> list[str]:
    """Mọi file đã sửa hoặc chưa được theo dõi, trừ file bị .gitignore chặn."""
    out = git("status", "--porcelain", "--untracked-files=all", raw=True)
    files = []
    for line in out.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip().strip('"')
        if " -> " in path:            # file được đổi tên
            path = path.split(" -> ", 1)[1]
        files.append(path)
    return files


# ------------------------------------------------------------------ kiểm tra
def read_run(run_id: str | None) -> dict:
    path = os.path.join(ROOT, RUNS_CSV)
    if not os.path.exists(path):
        raise SystemExit(f"không tìm thấy {RUNS_CSV} — chạy một thí nghiệm trước đã")
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("run_id")]
    if not rows:
        raise SystemExit(f"{RUNS_CSV} chưa có dòng nào — chạy một thí nghiệm trước đã")
    if run_id is None:
        # Mặc định lấy lần chạy có điểm gần nhất, không phải dòng cuối cùng —
        # dòng cuối thường là dòng ghi lại việc nộp bài, không có điểm dev.
        scored = [r for r in rows
                  if r.get("dev_official") or r.get("dev_R") or r.get("dev_P")]
        return (scored or rows)[-1]
    for r in reversed(rows):
        if r["run_id"] == run_id:
            return r
    ids = [r["run_id"] for r in rows[-8:]]
    raise SystemExit(f"không có run_id {run_id!r} trong {RUNS_CSV}.\n"
                     f"những run_id gần đây: {', '.join(ids)}")


def check_submission_zip(path: str) -> list[str]:
    """Mở file zip ra kiểm tra đúng những gì chương trình chấm của BTC đòi hỏi."""
    problems = []
    try:
        with zipfile.ZipFile(os.path.join(ROOT, path)) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            if names != ["submission.json"]:
                problems.append(f"{path}: bên trong phải có đúng một file tên "
                                f"submission.json, đang thấy {names}")
                return problems
            sub = json.loads(z.read("submission.json").decode("utf-8"))
    except (zipfile.BadZipFile, json.JSONDecodeError, KeyError) as e:
        return [f"{path}: không đọc được ({type(e).__name__})"]

    if not isinstance(sub, dict) or not sub:
        return [f"{path}: submission.json phải là một JSON object không rỗng"]

    bad_shape = [q for q, v in sub.items()
                 if not isinstance(v, dict) or set(v) != {"answer"}]
    if bad_shape:
        problems.append(f'{path}: {len(bad_shape)} câu không đúng dạng '
                        f'{{"answer": ...}}, ví dụ {bad_shape[:3]}')
        return problems

    answers = {q: v["answer"] for q, v in sub.items()}
    if all(isinstance(a, list) for a in answers.values()):        # Task 1
        problems += [f"{path}: {p}" for p in
                     metrics.check_submittable(answers, list(answers))]
    elif all(isinstance(a, str) for a in answers.values()):       # Task 2
        empty = [q for q, a in answers.items() if not a.strip()]
        if empty:
            problems.append(f"{path}: {len(empty)} câu trả lời rỗng, "
                            f"ví dụ {empty[:3]}")
    else:
        problems.append(f"{path}: trộn lẫn kiểu dữ liệu — Task 1 phải là list "
                        f"mã văn bản, Task 2 phải là chuỗi văn xuôi")
    return problems


def run_checks(run: dict, files: list[str], branch: str) -> list[str]:
    problems = []

    if branch in ("main", "master"):
        problems.append(
            f"đang đứng trên nhánh {branch!r}. Làm trên nhánh riêng rồi gộp qua PR:\n"
            f"      git checkout -b <tên-mình>/<việc-đang-làm>")

    for f in files:
        if any(f.startswith(d) for d in FORBIDDEN_DIRS):
            problems.append(f"{f} nằm trong thư mục không bao giờ được commit")
            continue
        full = os.path.join(ROOT, f)
        if not os.path.isfile(full):
            continue
        mb = os.path.getsize(full) / 1e6
        if mb > BIG_FILE_MB and not any(f.startswith(p) for p in BIG_FILE_ALLOWED):
            problems.append(f"{f} nặng {mb:.1f} MB — quá lớn để commit. "
                            f"Nếu thật sự cần thì thêm riêng bằng `git add -f`.")

    score = run.get("dev_official") or run.get("dev_R") or run.get("dev_P")
    if not score and not run.get("leaderboard"):
        problems.append(f"run {run['run_id']!r} chưa có điểm nào trong {RUNS_CSV}. "
                        f"Chạy trên tập dev trước khi nộp cho nhóm.")

    for f in files:
        if f.startswith("work/submissions/") and f.endswith(".zip"):
            problems += check_submission_zip(f)
    return problems


# ------------------------------------------------------------------ chọn file
def files_to_stage(run: dict, files: list[str], phase: str | None) -> tuple[list, list]:
    """Trả về (file commit bình thường, file phải ép bằng -f)."""
    keep_prefix = ("src/", "phases/", "tools/", "tests/", "docs/", ".github/",
                   "work/configs/", "work/analysis/", "paper/")
    normal = [f for f in files
              if f == RUNS_CSV
              or f.startswith(keep_prefix)
              or (f.startswith("work/submissions/") and f.endswith(".zip"))
              or f in ("README.md", "START_HERE.md", "Makefile",
                       "requirements.txt", ".gitignore", ".gitattributes")]

    forced = []
    if phase:
        # Khôi cần file run trên tập dev để chạy tools/error_analysis.py, nhưng
        # thư mục runs/ bị .gitignore chặn vì mỗi lần chạy sinh một file mới.
        # Quy ước: cuối mỗi phase ép commit đúng bản tốt nhất.
        cand = f"work/experiments/runs/{run['run_id']}.jsonl"
        if os.path.isfile(os.path.join(ROOT, cand)):
            forced.append(cand)
        else:
            print(f"  ! không tìm thấy {cand} — bỏ qua file run", file=sys.stderr)
    return sorted(set(normal)), forced


def commit_message(run: dict, phase: str | None) -> str:
    bits = []
    for k, label in (("dev_R", "dev_R"), ("dev_official", "dev"), ("dev_P", "dev_P")):
        if run.get(k):
            bits.append(f"{label}={run[k]}")
            break
    if run.get("dev_P") and run.get("dev_R"):
        bits.append(f"dev_P={run['dev_P']}")
    head = f"{run['run_id']}: " + (", ".join(bits) if bits else "kết quả mới")
    body = []
    for k in ("task", "phase", "chunking", "retriever", "reranker", "cutoff_rule"):
        if run.get(k):
            body.append(f"{k}: {run[k]}")
    if run.get("notes"):
        body.append(f"notes: {run['notes']}")
    if phase:
        body.append(f"kèm file run của Phase {phase} để chạy phân tích lỗi")
    return head + ("\n\n" + "\n".join(body) if body else "")


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default=None,
                    help="mặc định: dòng cuối cùng trong runs.csv")
    ap.add_argument("--phase", default=None,
                    help="kèm file run tốt nhất của phase này cho Khôi phân tích lỗi")
    ap.add_argument("--check", action="store_true", help="chỉ kiểm tra, không commit")
    ap.add_argument("--yes", action="store_true", help="không hỏi xác nhận")
    ap.add_argument("--no-push", action="store_true", help="commit nhưng không push")
    a = ap.parse_args()

    run = read_run(a.run_id)
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    files = changed_files()

    print(f"\nrun_id  : {run['run_id']}")
    print(f"nhánh   : {branch}")
    score = run.get("dev_official") or run.get("dev_R") or "—"
    print(f"điểm dev: {score}")

    problems = run_checks(run, files, branch)
    if problems:
        sys.stdout.flush()
        print("\nKHÔNG NỘP ĐƯỢC — sửa những điều sau:\n", file=sys.stderr)
        for p in problems:
            print(f"  * {p}", file=sys.stderr)
        print("", file=sys.stderr)
        raise SystemExit(1)

    normal, forced = files_to_stage(run, files, a.phase)
    skipped = [f for f in files if f not in normal and f not in forced]

    print(f"\nsẽ commit ({len(normal) + len(forced)} file):")
    for f in normal:
        print(f"  + {f}")
    for f in forced:
        print(f"  + {f}   (ép qua .gitignore)")
    if skipped:
        print(f"\nbỏ qua ({len(skipped)} file — không thuộc loại cần chia sẻ):")
        for f in skipped[:10]:
            print(f"  - {f}")
        if len(skipped) > 10:
            print(f"  - … và {len(skipped) - 10} file nữa")

    msg = commit_message(run, a.phase)
    print("\ncâu commit:")
    for line in msg.splitlines():
        print(f"  | {line}")

    if a.check:
        print("\n--check: mọi thứ hợp lệ, chưa commit gì cả.\n")
        return
    if not normal and not forced:
        print("\nkhông có file nào để commit.\n")
        return
    if not a.yes:
        if input("\nnộp cho nhóm? [y/N] ").strip().lower() not in ("y", "yes"):
            print("đã huỷ, không thay đổi gì.\n")
            return

    for f in normal:
        git("add", "--", f)
    for f in forced:
        git("add", "-f", "--", f)
    git("commit", "-m", msg)
    print(f"\nđã commit: {git('rev-parse', '--short', 'HEAD')}")

    if a.no_push:
        print("--no-push: chưa đẩy lên. Khi nào sẵn sàng thì "
              f"`git push -u origin {branch}`\n")
        return
    git("push", "-u", "origin", branch)
    print(f"đã push lên nhánh {branch}\n")
    print("Bước cuối — mở Pull Request để Khôi gộp và nộp bài:")
    print(f"  {REPO_URL}/compare/main...{branch}?expand=1\n")


if __name__ == "__main__":
    main()
