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

## Quy trình của mỗi người — một lệnh

### 1. Làm trên nhánh của mình, không làm thẳng trên `main`

```bash
git checkout main
git pull
git checkout -b thu/bm25-grid      # tên: <tên mình>/<việc đang làm>
```

Năm người cùng sửa `main` thì sẽ đè lên nhau. Nhánh riêng thì không.

### 2. Chạy thí nghiệm như bình thường

Script tự ghi một dòng vào `runs.csv`. Không cần làm gì thêm.

### 3. Nộp cho nhóm

```bash
make send
```

Chỉ vậy. Lệnh này gọi `tools/handoff.py`, và nó làm giúp toàn bộ phần còn lại:

* tìm lần chạy có điểm gần nhất trong `runs.csv`;
* **kiểm tra** — xem mục dưới;
* chọn đúng những file cần chia sẻ, bỏ qua những file không thuộc loại đó;
* viết sẵn câu commit **có kèm con số** (`bm25-thu-01: dev_R=0.7683`);
* commit, push lên nhánh của mình;
* in ra link mở Pull Request để Khôi gộp.

Trước khi commit nó in ra danh sách file và câu commit rồi hỏi lại một lần, nên
cứ chạy thử thoải mái. Muốn xem trước mà chưa commit gì thì thêm `--check`:

```bash
python tools/handoff.py --check
```

Cuối mỗi phase, thêm `--phase` để đính kèm file run cho Khôi chạy phân tích lỗi:

```bash
python tools/handoff.py --phase 1
```

### Script sẽ TỪ CHỐI nộp nếu

| Phát hiện | Vì sao chặn |
|---|---|
| Đang đứng trên `main` | Năm người sửa cùng một nhánh sẽ đè lên nhau |
| Lần chạy chưa có điểm trong `runs.csv` | Chưa đo thì chưa có gì để nộp |
| Có file trong `data/`, `models/`, `indexes/` | Commit rồi thì git giữ mãi mãi, gỡ ra rất khó |
| File lạ lớn hơn 5 MB | Thường là dữ liệu lọt qua `.gitignore` |
| File nộp bài sai định dạng | Codabench chấm 0 mà không báo lỗi gì |

Riêng ô cuối, script **mở file zip ra kiểm tra thật**: bên trong đúng một
`submission.json`, mỗi câu đúng dạng `{"answer": ...}`, Task 1 không quá 5 mã và
mọi mã đều là chuỗi, Task 2 không có câu trả lời rỗng. Đây đúng là bốn cách một
bài nộp bị chấm 0 mà nhìn bằng mắt không thấy gì sai.

### 4. Mở Pull Request

Script in sẵn link. Bấm vào, viết một dòng mô tả, gửi. Khôi xem rồi gộp — đây
cũng là lúc Khôi biết có kết quả mới mà không cần ai nhắn.

### 5. Ghi một dòng vào tài liệu của cặp

`run_id`, điểm dev, một câu đã đổi gì. Chỉ vậy thôi. Tài liệu là nhật ký, không
phải nơi lưu số — số nằm ở `runs.csv`.

---

## Máy tự kiểm tra mọi Pull Request

Mỗi PR sẽ tự chạy `.github/workflows/ci.yml` trên GitHub, gồm hai việc:

* **`python tests/test_cases.py`** — 114 trường hợp kiểm thử: khớp điểm với
  chương trình chấm của BTC, giới hạn 5 mã, quy kết lỗi, cắt văn bản, chuẩn hoá
  tiếng Việt.
* **`python tools/repo_hygiene.py`** — quét toàn bộ file đang được git theo dõi,
  chặn dữ liệu lọt vào repo, file quá lớn, và file nộp bài sai định dạng.

Nếu ô kiểm tra trên PR hiện màu đỏ thì **đừng gộp**. Bấm vào xem log, nó nói rõ
file nào sai và sai chỗ nào. Chạy trước ở máy mình bằng `make test` và
`make hygiene`.

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
python tools/handoff.py --phase 1
```

Script tự tìm file run của lần chạy đó và thêm bằng `git add -f` — `-f` là bắt
buộc vì nó ghi đè `.gitignore` cho đúng một file. Đừng bỏ chặn cả thư mục:
mỗi lần chạy sinh một file mới, để mở thì repo phình rất nhanh.

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

Trong hầu hết trường hợp câu trả lời là `make send`. Bảng này để biết nó đang
làm gì thay mình, và khi nào cần thêm gì.

| Vừa làm gì | Gõ gì |
|---|---|
| Chạy một thí nghiệm | `make send` |
| Sửa code | `make send` — nó tự kèm `src/`, `phases/`, `tools/` |
| Tạo file nộp bài | `make send` — nó tự kèm file `.zip` và kiểm tra định dạng |
| Xong một phase | `python tools/handoff.py --phase <n>` — kèm thêm file run |
| Muốn xem trước mà chưa commit | `python tools/handoff.py --check` |
| Tải dữ liệu BTC về | **không gõ gì cả** — dữ liệu không bao giờ vào Git |
| Huấn luyện xong một mô hình | `make send` — nó kèm config, **không** kèm trọng số |
| PR đang đỏ | `make test` và `make hygiene` ở máy mình để xem hỏng chỗ nào |
