# Làm việc chung — dữ liệu đi từ máy mình đến Khôi bằng đường nào

> Trang duy nhất trong repo viết bằng tiếng Việt, vì nó là quy trình phối hợp
> của nhóm chứ không phải kiến thức kỹ thuật. Mọi trang khác vẫn là tiếng Anh.

Nhóm chia làm ba: **Thư · Vinh** (Task 1), **Long · Nguyên** (Task 2), **Khôi**
(luật, đo lường, nộp bài, bài báo). Khôi là người duy nhất nộp bài lên Codabench
và là người viết phân tích lỗi cho bài báo, nên anh ấy cần nhận được kết quả của
bốn người kia. Trang này nói rõ nhận bằng cách nào.

---

## Nguyên tắc: không ai gửi file qua chat

**Git là đường truyền duy nhất.** Không gửi file qua Messenger, Zalo, Drive hay
email. Lý do không phải là hình thức:

* File gửi qua chat không có phiên bản. Ba tuần nữa, khi bài báo cần biết con số
  0.78 đến từ lần chạy nào với config nào, không ai tìm lại được.
* File gửi qua chat không đi kèm code sinh ra nó. Khôi nhận được con số nhưng
  không tái tạo được, mà BTC thì bắt buộc phải nộp gói tái lập thực nghiệm.
* Người gửi hay quên. Git thì không.

Nếu một thứ đáng để Khôi nhận, nó đáng được commit.

---

## Ba loại dữ liệu, ba cách xử lý

| Loại | Đi qua Git? | Vì sao |
|---|---|---|
| `work/experiments/runs.csv` | **có** | Bảng số liệu chính. Mọi script tự ghi vào đây. |
| `work/configs/*.yaml` | **có** | Config đóng băng — thứ duy nhất tái tạo được lần chạy. |
| `work/analysis/*.md` | **có** | Phân tích lỗi, bảng ablation. Đây là nội dung bài báo. |
| `work/submissions/*.zip` | **có** | Nhỏ (Task 1 ~22 KB, Task 2 ~2 MB) và Khôi không tự tạo lại được nếu không có GPU. |
| `src/`, `phases/`, `tools/` | **có** | Code. Hiển nhiên. |
| `work/experiments/runs/*.jsonl` | **chỉ bản tốt nhất** | 1–3 MB mỗi file. Xem mục dưới. |
| `work/experiments/predictions/*.jsonl` | **không** | 10–13 MB mỗi file. Khôi không cần — anh ấy chỉ cần điểm và file zip. |
| `data/` | **không bao giờ** | 455 MB. Ai cũng tải trực tiếp từ Drive của BTC. |
| `models/`, `indexes/` | **không bao giờ** | Trọng số và index tự sinh lại được từ config. |

---

## Quy trình của mỗi người

### 1. Làm trên nhánh của mình, không làm thẳng trên `main`

```bash
git checkout main
git pull
git checkout -b thu/bm25-grid      # tên: <tên mình>/<việc đang làm>
```

Năm người cùng sửa `main` thì sẽ đè lên nhau. Nhánh riêng thì không.

### 2. Chạy thí nghiệm như bình thường

Script tự ghi một dòng vào `runs.csv`. Không cần làm gì thêm.

### 3. Commit và push

```bash
git add work/experiments/runs.csv work/configs/ work/analysis/
git add src/ phases/                          # nếu có sửa code
git commit -m "bm25: quét lưới k1/b, dev_R 0.741 -> 0.7683"
git push -u origin thu/bm25-grid
```

Trong câu commit **luôn có con số**. `"sửa bm25"` không nói gì;
`"dev_R 0.741 -> 0.7683"` nói tất cả.

### 4. Mở Pull Request vào `main`

Trên GitHub, bấm *Compare & pull request*. Khôi xem rồi gộp. Đây cũng là lúc
Khôi biết có kết quả mới mà không cần ai nhắn.

### 5. Ghi một dòng vào tài liệu của cặp

`run_id`, điểm dev, một câu đã đổi gì. Chỉ vậy thôi. Tài liệu là nhật ký, không
phải nơi lưu số — số nằm ở `runs.csv`.

---

## `runs.csv` bị xung đột thì làm sao

Gần như sẽ không bị nữa. File này chỉ được **ghi thêm dòng vào cuối**, không bao
giờ sửa dòng cũ, nên `.gitattributes` đã khai báo:

```
work/experiments/runs.csv merge=union
```

Git sẽ tự gộp bằng cách **giữ cả hai bên**. Không cần làm gì.

Nếu vì lý do nào đó vẫn thấy xung đột: **luôn giữ CẢ HAI dòng**, xoá ba dòng
đánh dấu `<<<<<<<`, `=======`, `>>>>>>>`. Không bao giờ chọn một bên — chọn một
bên là xoá mất một thí nghiệm của người khác.

---

## Khi Khôi cần file run để phân tích lỗi

`tools/error_analysis.py` cần file run trên tập dev, mà thư mục
`work/experiments/runs/` bị `.gitignore` chặn (mỗi lần chạy là một file mới, để
mở thì repo phình rất nhanh).

Nên quy ước: **cuối mỗi phase, commit đúng một file — bản tốt nhất.**

```bash
git add -f work/experiments/runs/<run_id_tot_nhat>.jsonl
git commit -m "run: bản tốt nhất Phase 1 để Khôi phân tích lỗi"
```

`-f` là bắt buộc, vì nó ghi đè `.gitignore` cho đúng một file đó. Đừng bỏ chặn
cả thư mục.

---

## Khi Khôi cần nộp bài

Không cần hỏi ai. File zip đã nằm trong Git:

```bash
git pull
ls work/submissions/*.zip
```

Đó là lý do `work/submissions/*.zip` được cho phép commit. Người tạo ra nó chỉ
cần `git add work/submissions/<ten>.zip` cùng lúc với `runs.csv`.

> **Lưu ý:** file `.json` cạnh file `.zip` vẫn bị chặn. Nó là cùng một nội dung
> nhưng lớn gấp năm lần, và Codabench chỉ nhận `.zip`.

---

## Việc Khôi làm khi nhận được

1. `git pull` — không cần ai báo.
2. Gộp PR, xem `runs.csv` đã có dòng mới chưa.
3. Chạy `make budget` nếu người kia đổi mô hình — kiểm tra trần 4 tỷ tham số.
4. Nộp file zip lên Codabench, rồi điền cột `leaderboard`:
   ```bash
   python -c "from src.exp_log import update_leaderboard as u; u('<run_id>', 0.xxxx)"
   ```
5. Kiểm tra dev có còn trung thực không:
   ```bash
   python -c "from src.exp_log import correlation as c; print(c())"
   ```
   Điểm dev tăng mà leaderboard không tăng theo nghĩa là tập dev đang nói dối, và
   mọi quyết định của cả bốn người kia đang dựa trên số sai. Đây là việc quan
   trọng nhất Khôi làm.

---

## Bảng tra nhanh — "tôi vừa chạy xong, giờ làm gì"

| Vừa làm gì | Commit cái gì |
|---|---|
| Chạy một thí nghiệm | `runs.csv` + `work/configs/` |
| Sửa code | thêm `src/` hoặc `phases/` |
| Tạo file nộp bài | thêm `work/submissions/*.zip` |
| Xong một phase | thêm `work/analysis/*.md` và `git add -f` file run tốt nhất |
| Tải dữ liệu BTC về | **không commit gì cả** — dữ liệu không bao giờ vào Git |
| Huấn luyện xong một mô hình | **không commit trọng số** — commit config đã dùng để huấn luyện |
