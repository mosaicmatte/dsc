# Error analysis — phase 1

Run: `bm25-article-real.jsonl`  ·  cutoff: `top_k` (k=5, α=0.85)  ·  1049 dev questions

## 1. Headline

| | |
|---|---|
| **Recall (primary)** | **0.7410** |
| Precision (tiebreak) | 0.1575 |
| questions with perfect recall | 758 / 1049 |
| questions with zero recall | 250 / 1049 |
| avg slots used (of 5) | 5.00 |
| avg slots wasted on non-relevant | 4.21 |

## 2. Integrity alarms

None. No missing gold ids, no chunk ids, no over-cap or empty answers, no duplicates.

## 3. Where the missing recall went

Recall is **0.7410**, so **0.2590** is missing. Every point of it is attributed below; the components sum to the gap exactly, by construction.

| cause | lost recall | share of gap | what fixes it |
|---|---|---|---|
| `ranking` | 0.1985 | 76.7% | RERANKER / FUSION — retrieved but ranked below position 5 |
| `retrieval` | 0.0605 | 23.3% | RETRIEVER — the document is not in the run at all |
| `cap` | 0.0000 |  0.0% | IMPOSSIBLE — more than 5 gold documents; the cap forbids full recall |
| `cutoff` | 0.0000 |  0.0% | CUTOFF RULE — in the top-5 and not returned; free to fix |
| `zeroed` | 0.0000 |  0.0% | SELF-INFLICTED — empty or >5 ids; clamp the cutoff |
| **total** | **0.2590** | 100% | |

Sanity: recall 0.7410 + gap 0.2590 = 1.0000

### Nested ceilings

Each row is the best recall achievable if everything below it were perfect.

| ceiling | value | meaning |
|---|---|---|
| cap ceiling | 1.0000 | best possible given ≤5 ids per question |
| retrieval ceiling | 0.9395 | ...and given what the retriever returned |
| prefix ceiling | 0.7410 | ...and given the current ranking, with a perfect cutoff |
| **achieved** | **0.7410** | ...and given the cutoff you used |

## 4. What every cutoff would have scored

| rule | param | recall | precision | avg set |
|---|---|---|---|---|
| ratio | 0.60 | 0.7410 | 0.1618 | 4.96 |
| ratio | 0.50 | 0.7410 | 0.1583 | 5.00 |
| top_k | 5 | 0.7410 | 0.1575 | 5.00 |
| ratio | 0.70 | 0.7401 | 0.1754 | 4.84 |
| ratio | 0.75 | 0.7367 | 0.1923 | 4.69 |
| ratio | 0.80 | 0.7324 | 0.2196 | 4.44 |
| ratio | 0.85 | 0.7267 | 0.2677 | 4.01 |
| top_k | 4 | 0.6964 | 0.1847 | 4.00 |
| ratio | 0.90 | 0.6921 | 0.3367 | 3.30 |
| top_k | 3 | 0.6473 | 0.2285 | 3.00 |
| ratio | 0.95 | 0.6010 | 0.4141 | 2.18 |
| gap |  | 0.5739 | 0.4024 | 1.85 |
| top_k | 2 | 0.5721 | 0.3022 | 2.00 |
| ratio | 0.98 | 0.5094 | 0.4418 | 1.45 |
| top_k | 1 | 0.4295 | 0.4500 | 1.00 |

## 5. Breakdowns

### By number of gold documents

| \|gold\| | questions | recall | precision | cap loss | retrieval loss | ranking loss | cutoff loss |
|---|---|---|---|---|---|---|---|
| 1 | 966 | 0.7578 | 0.1516 | 0.0000 | 0.0518 | 0.1905 | 0.0000 |
| 2 | 73 | 0.5822 | 0.2329 | 0.0000 | 0.1370 | 0.2808 | 0.0000 |
| 3 | 8 | 0.2917 | 0.1750 | 0.0000 | 0.3333 | 0.3750 | 0.0000 |
| 4 | 2 | 0.2500 | 0.2000 | 0.0000 | 0.3750 | 0.3750 | 0.0000 |

> Questions with more than 5 gold documents can never reach recall 1.0. If that row is large, your headline recall has a hard ceiling below 1.

### By where the best gold document ranked

| best gold rank | questions | recall | what it means |
|---|---|---|---|
| 1 | 472 | 0.9753 | already top — pure cutoff/precision question |
| 11-25 | 81 | 0.0000 | reranker territory |
| 2-5 | 327 | 0.9694 | inside the cap — cutoff decides whether you get it |
| 26-50 | 17 | 0.0000 | deep; reranking at depth 50 needed |
| 51-100 | 14 | 0.0000 | very deep; check retrieval quality |
| 6-10 | 84 | 0.0000 | just outside — a reranker should reach these |
| not retrieved | 54 | 0.0000 | RETRIEVER failure — reranking cannot help |

### By question length (words)

| length | questions | recall | precision |
|---|---|---|---|
| 1-10 | 60 | 0.6000 | 0.1333 |
| 11-20 | 531 | 0.6933 | 0.1461 |
| 21-40 | 454 | 0.8153 | 0.1740 |
| >40 | 4 | 0.7500 | 0.1500 |

### By answer-set size actually returned

| slots used | questions | recall | precision |
|---|---|---|---|
| 5 | 1049 | 0.7410 | 0.1575 |

## 7. Worksheet — 20 worst questions

Fill in **CATEGORY** by hand. Categories: `granularity`, `lexical-mismatch`, `numeric`, `multi-article`, `negation`, `too-general`, `too-specific`, `ambiguous-query`, `label-noise`, `cutoff`, `impossible`

> `dominant loss` already tells you which *kind* of fix applies. Your job is to say *why* the model made that mistake — that is what a script cannot do and what the paper needs.

### 1. `83554` — recall 0.00, precision 0.00

**Q:** Phân bổ dự toán chi đầu tư phát triển được quy định như thế nào?

- state: `ok`
- gold: 1  ·  retrieved: 1  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **ranking** → RERANKER / FUSION — retrieved but ranked below position 5

| gold doc | rank in run | in answer? |
|---|---|---|
| `100125` | 32 | no |

**Returned (5):** `139358`, `42223`, `64378`, `166280`, `280879`

<details><summary>gold `100125`</summary>

> Thong-tu-122-2021-TT-BTC-to-chuc-thuc-hien-du-toan-ngan-sach-nha-nuoc-2022-501044 Thong-tu-122-2021-TT-BTC-to-chuc-thuc-hien-du-toan-ngan-sach-nha-nuoc-2022-501044 BỘ TÀI CHÍNH  -------  CỘNG HÒA XÃ H …

</details>

<details><summary>we returned `139358` instead</summary>

> Nghi-dinh-147-2020-ND-CP-quy-dinh-to-chuc-va-hoat-dong-cua-Quy-dau-tu-phat-trien-dia-phuong-459854 Nghi-dinh-147-2020-ND-CP-quy-dinh-to-chuc-va-hoat-dong-cua-Quy-dau-tu-phat-trien-dia-phuong-459854 CH …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 2. `55892` — recall 0.00, precision 0.00

**Q:** Mục tiêu của Chiến lược nợ công giai đoạn 2021 - 2030?

- state: `ok`
- gold: 1  ·  retrieved: 1  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **ranking** → RERANKER / FUSION — retrieved but ranked below position 5

| gold doc | rank in run | in answer? |
|---|---|---|
| `195324` | 18 | no |

**Returned (5):** `202799`, `195242`, `75514`, `84495`, `136617`

<details><summary>gold `195324`</summary>

> Quyet-dinh-460-QD-TTg-2022-Chien-luoc-no-cong-den-2030-510114 Quyet-dinh-460-QD-TTg-2022-Chien-luoc-no-cong-den-2030-510114 THỦ TƯỚNG CHÍNH PHỦ  -------  CỘNG HÒA XÃ HỘI  CHỦ NGHĨA VIỆT NAM  Độc lập - …

</details>

<details><summary>we returned `202799` instead</summary>

> Quyet-dinh-506-QD-BTP-2023-thuc-hien-Chien-luoc-phat-trien-thong-ke-Viet-Nam-2021-2030-562049 Quyet-dinh-506-QD-BTP-2023-thuc-hien-Chien-luoc-phat-trien-thong-ke-Viet-Nam-2021-2030-562049 BỘ TƯ PHÁP   …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 3. `1454` — recall 0.00, precision 0.00

**Q:** Đại hội đại biểu toàn quốc của Hội Xuất bản Việt Nam biểu quyết theo nguyên tắc nào?

- state: `ok`
- gold: 1  ·  retrieved: 1  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **ranking** → RERANKER / FUSION — retrieved but ranked below position 5

| gold doc | rank in run | in answer? |
|---|---|---|
| `296173` | 64 | no |

**Returned (5):** `209151`, `228519`, `94023`, `201504`, `68073`

<details><summary>gold `296173`</summary>

> Quyet-dinh-599-QD-BNV-nam-2012-phe-duyet-Dieu-le-sua-doi-bo-sung-Hoi-Xuat-ban-173096 Quyet-dinh-599-QD-BNV-nam-2012-phe-duyet-Dieu-le-sua-doi-bo-sung-Hoi-Xuat-ban-173096 BỘ NỘI VỤ  --------  CỘNG  HÒA …

</details>

<details><summary>we returned `209151` instead</summary>

> Quyet-dinh-422-QD-TTg-2021-phe-duyet-Dieu-le-Lien-hiep-cac-to-chuc-huu-nghi-Viet-Nam-468817 Quyet-dinh-422-QD-TTg-2021-phe-duyet-Dieu-le-Lien-hiep-cac-to-chuc-huu-nghi-Viet-Nam-468817 THỦ TƯỚNG  CHÍNH …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 4. `78318` — recall 0.00, precision 0.00

**Q:** Hành vi sử dụng hóa đơn, chứng từ không hợp pháp là những hành vi nào?

- state: `ok`
- gold: 1  ·  retrieved: 1  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **ranking** → RERANKER / FUSION — retrieved but ranked below position 5

| gold doc | rank in run | in answer? |
|---|---|---|
| `231881` | 44 | no |

**Returned (5):** `47011`, `272614`, `99418`, `87086`, `294093`

<details><summary>gold `231881`</summary>

> Nghi-dinh-123-2020-ND-CP-quy-dinh-hoa-don-chung-tu-445980 Nghi-dinh-123-2020-ND-CP-quy-dinh-hoa-don-chung-tu-445980 CHÍNH PHỦ  --------  CỘNG HÒA XÃ HỘI CHỦ  NGHĨA VIỆT NAM  Độc lập - Tự do - Hạnh phú …

</details>

<details><summary>we returned `47011` instead</summary>

> Thong-bao-7592-TB-CTTPHCM-2023-canh-bao-tinh-trang-vi-pham-khi-su-dung-hoa-don-Cuc-Thue-Ho-Chi-Minh- Thong-bao-7592-TB-CTTPHCM-2023-canh-bao-tinh-trang-vi-pham-khi-su-dung-hoa-don-Cuc-Thue-Ho-Chi-Minh …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 5. `45928` — recall 0.00, precision 0.00

**Q:** Thời giờ nghỉ ngơi đối với người lao động được quy định như thế nào?

- state: `ok`
- gold: 1  ·  retrieved: 1  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **ranking** → RERANKER / FUSION — retrieved but ranked below position 5

| gold doc | rank in run | in answer? |
|---|---|---|
| `129823` | 9 | no |

**Returned (5):** `288109`, `84412`, `245505`, `222158`, `208105`

<details><summary>gold `129823`</summary>

> Bo-Luat-lao-dong-2019-333670 Bo-Luat-lao-dong-2019-333670 QUỐC HỘI  --------  CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM  Độc lập - Tự do - Hạnh phúc  ---------------  Bộ luật số: 45/2019/QH14  Hà Nội, ngày 2 …

</details>

<details><summary>we returned `288109` instead</summary>

> Bo-luat-hang-hai-Viet-Nam-2015-298374 Bo-luat-hang-hai-Viet-Nam-2015-298374 QUỐC HỘI  --------  CỘNG HÒA XÃ  HỘI CHỦ NGHĨA VIỆT NAM  Độc lập - Tự do - Hạnh phúc  ---------------  Luật số:  95/2015/QH1 …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 6. `126112` — recall 0.00, precision 0.00

**Q:** Các trường hợp bị thu hồi đất do vi phạm pháp luật về đất đai theo Dự thảo Luật Đất đai (Sửa đổi)?

- state: `ok`
- gold: 1  ·  retrieved: 1  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **ranking** → RERANKER / FUSION — retrieved but ranked below position 5

| gold doc | rank in run | in answer? |
|---|---|---|
| `184038` | 6 | no |

**Returned (5):** `285773`, `80269`, `9248`, `114181`, `208565`

<details><summary>gold `184038`</summary>

> QUỐC HỘI  --------  CỘNG  HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM  Độc lập - Tự do - Hạnh phúc  ---------------  Luật  số: ……/2023/QH15    DỰ THẢO    QUỐC HỘI  --------  CỘNG  HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM  Độc …

</details>

<details><summary>we returned `285773` instead</summary>

> Quyet-dinh-1257-QD-BTNMT-2022-Ke-hoach-xay-dung-du-an-Luat-Dat-dai-sua-doi-516966 Quyet-dinh-1257-QD-BTNMT-2022-Ke-hoach-xay-dung-du-an-Luat-Dat-dai-sua-doi-516966 BỘ TÀI NGUYÊN  VÀ  MÔI TRƯỜNG  ----- …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 7. `78362` — recall 0.00, precision 0.00

**Q:** Thủ tục công nhận tổ trưởng tổ hòa giải cơ sở như thế nào?

- state: `ok`
- gold: 1  ·  retrieved: 1  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **ranking** → RERANKER / FUSION — retrieved but ranked below position 5

| gold doc | rank in run | in answer? |
|---|---|---|
| `177035` | 21 | no |

**Returned (5):** `177925`, `305134`, `34243`, `121757`, `20100`

<details><summary>gold `177035`</summary>

> Nghi-quyet-lien-tich-01-2014-NQLT-CP-UBTUMTTQVN-huong-dan-phoi-hop-hoa-giai-o-co-so-258040 Nghi-quyet-lien-tich-01-2014-NQLT-CP-UBTUMTTQVN-huong-dan-phoi-hop-hoa-giai-o-co-so-258040 CHÍNH PHỦ - ỦY  BA …

</details>

<details><summary>we returned `177925` instead</summary>

> TÒA  ÁN NHÂN DÂN TỐI CAO  ********  CỘNG  HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM  Độc lập - Tự do - Hạnh phúc  ********  Số:  81/2002/TANDTC  Hà  Nội, ngày 10 tháng 6 năm 2002  TÒA  ÁN NHÂN DÂN TỐI CAO  ****** …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 8. `164280` — recall 0.00, precision 0.00

**Q:** Điều khiển xe ô tô chạy quá tốc độ bị phạt nguội thì có thể nộp phạt qua đường bưu điện được không?

- state: `ok`
- gold: 1  ·  retrieved: 0  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **retrieval** → RETRIEVER — the document is not in the run at all

| gold doc | rank in run | in answer? |
|---|---|---|
| `219419` | **not in top-33** | no |

**Returned (5):** `91006`, `50885`, `125243`, `174131`, `107319`

<details><summary>gold `219419`</summary>

> Nghi-dinh-118-2021-ND-CP-huong-dan-Luat-Xu-ly-vi-pham-hanh-chinh-477969 Nghi-dinh-118-2021-ND-CP-huong-dan-Luat-Xu-ly-vi-pham-hanh-chinh-477969 CHÍNH PHỦ  --------  CỘNG HÒA XÃ  HỘI CHỦ NGHĨA VIỆT NAM …

</details>

<details><summary>we returned `91006` instead</summary>

> Nghi-dinh-46-2016-ND-CP-xu-phat-vi-pham-hanh-chinh-giao-thong-duong-bo-duong-sat-288330 Nghi-dinh-46-2016-ND-CP-xu-phat-vi-pham-hanh-chinh-giao-thong-duong-bo-duong-sat-288330 CHÍNH PHỦ  -------  CỘNG …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 9. `35752` — recall 0.00, precision 0.00

**Q:** Vi phạm các quy định về thực hiện nghĩa vụ quân sự sẽ xử phạt như thế nào?

- state: `ok`
- gold: 1  ·  retrieved: 1  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **ranking** → RERANKER / FUSION — retrieved but ranked below position 5

| gold doc | rank in run | in answer? |
|---|---|---|
| `122159` | 14 | no |

**Returned (5):** `230077`, `173521`, `255913`, `177925`, `32881`

<details><summary>gold `122159`</summary>

> Nghi-dinh-120-2013-ND-CP-xu-phat-vi-pham-hanh-chinh-quoc-phong-co-yeu-209606 Nghi-dinh-120-2013-ND-CP-xu-phat-vi-pham-hanh-chinh-quoc-phong-co-yeu-209606 CHÍNH PHỦ  --------  CỘNG HÒA XÃ HỘI  CHỦ NGHĨ …

</details>

<details><summary>we returned `230077` instead</summary>

> TÒA  ÁN NHÂN DÂN TỐI CAO  -------  CỘNG  HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM  Độc lập - Tự do - Hạnh phúc  ---------------  Số: 64/TANDTC-PC  V/v  thông báo kết quả giải đáp trực tuyến một số vướng mắc về h …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 10. `55900` — recall 0.00, precision 0.00

**Q:** Khái niệm đầu tư kinh doanh

- state: `ok`
- gold: 1  ·  retrieved: 0  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **retrieval** → RETRIEVER — the document is not in the run at all

| gold doc | rank in run | in answer? |
|---|---|---|
| `2113` | **not in top-76** | no |

**Returned (5):** `289701`, `82140`, `243216`, `206557`, `28054`

<details><summary>gold `2113`</summary>

> Luat-Dau-tu-so-61-2020-QH14-321051 Luat-Dau-tu-so-61-2020-QH14-321051 QUỐC HỘI  -------  CỘNG HÒA XÃ HỘI  CHỦ NGHĨA VIỆT NAM  Độc lập - Tự do - Hạnh phúc  ---------------  Luật số:  61/2020/QH14  Hà N …

</details>

<details><summary>we returned `289701` instead</summary>

> Quyet-dinh-1658-QD-TTg-2021-phe-duyet-Chien-luoc-quoc-gia-ve-tang-truong-xanh-489788 Quyet-dinh-1658-QD-TTg-2021-phe-duyet-Chien-luoc-quoc-gia-ve-tang-truong-xanh-489788 THỦ TƯỚNG CHÍNH  PHỦ  -------  …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 11. `135152` — recall 0.00, precision 0.00

**Q:** Hành vi nào bị cấm trong hoạt động hóa đơn?

- state: `ok`
- gold: 1  ·  retrieved: 0  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **retrieval** → RETRIEVER — the document is not in the run at all

| gold doc | rank in run | in answer? |
|---|---|---|
| `231881` | **not in top-42** | no |

**Returned (5):** `300544`, `120931`, `245154`, `161003`, `43981`

<details><summary>gold `231881`</summary>

> Nghi-dinh-123-2020-ND-CP-quy-dinh-hoa-don-chung-tu-445980 Nghi-dinh-123-2020-ND-CP-quy-dinh-hoa-don-chung-tu-445980 CHÍNH PHỦ  --------  CỘNG HÒA XÃ HỘI CHỦ  NGHĨA VIỆT NAM  Độc lập - Tự do - Hạnh phú …

</details>

<details><summary>we returned `300544` instead</summary>

> Nghi-dinh-38-2014-ND-CP-quan-ly-hoa-chat-thuoc-dien-kiem-soat-cua-Cong-uoc-Cam-vu-khi-hoa-hoc-228675 Nghi-dinh-38-2014-ND-CP-quan-ly-hoa-chat-thuoc-dien-kiem-soat-cua-Cong-uoc-Cam-vu-khi-hoa-hoc-22867 …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 12. `144746` — recall 0.00, precision 0.00

**Q:** Việc chứng nhận đối với thực phẩm xuất khẩu thuộc về thẩm quyền của cơ quan nào?

- state: `ok`
- gold: 1  ·  retrieved: 1  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **ranking** → RERANKER / FUSION — retrieved but ranked below position 5

| gold doc | rank in run | in answer? |
|---|---|---|
| `219533` | 14 | no |

**Returned (5):** `232471`, `129920`, `19512`, `260712`, `246194`

<details><summary>gold `219533`</summary>

> Luat-an-toan-thuc-pham-2010-108074 Luat-an-toan-thuc-pham-2010-108074 QUỐC  HỘI  -------  CỘNG  HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM  Độc lập – Tự do – Hạnh phúc  ---------  Luật  số: 55/2010/QH12  Hà  Nội,  …

</details>

<details><summary>we returned `232471` instead</summary>

> Quyet-dinh-1390-QD-BCT-2020-cau-hoi-kiem-tra-de-xac-nhan-da-tap-huan-ve-an-toan-thuc-pham-448394 Quyet-dinh-1390-QD-BCT-2020-cau-hoi-kiem-tra-de-xac-nhan-da-tap-huan-ve-an-toan-thuc-pham-448394 BỘ CÔN …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 13. `11032` — recall 0.00, precision 0.00

**Q:** Những vấn đề cần lưu ý khi nộp hồ sơ làm thủ tục chấm dứt giám hộ?

- state: `ok`
- gold: 1  ·  retrieved: 0  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **retrieval** → RETRIEVER — the document is not in the run at all

| gold doc | rank in run | in answer? |
|---|---|---|
| `146687` | **not in top-87** | no |

**Returned (5):** `36157`, `55016`, `221004`, `220347`, `54992`

<details><summary>gold `146687`</summary>

> Quyet-dinh-528-QD-BTP-2023-cong-bo-thu-tuc-hanh-chinh-linh-vuc-ho-tich-thuoc-Bo-Tu-phap-563719 Quyet-dinh-528-QD-BTP-2023-cong-bo-thu-tuc-hanh-chinh-linh-vuc-ho-tich-thuoc-Bo-Tu-phap-563719 BỘ TƯ PHÁP …

</details>

<details><summary>we returned `36157` instead</summary>

> Thong-tu-24-2017-TT-NHNN-trinh-tu-thu-tuc-thu-hoi-Giay-phep-thanh-ly-tai-san-cua-to-chuc-tin-dung-34 Thong-tu-24-2017-TT-NHNN-trinh-tu-thu-tuc-thu-hoi-Giay-phep-thanh-ly-tai-san-cua-to-chuc-tin-dung-3 …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 14. `92448` — recall 0.00, precision 0.00

**Q:** Vụ Thi đua khen thưởng thuộc Bộ Tư pháp có nhiệm vụ và quyền hạn gì trong việc giải quyết khiếu nại tố cáo về công tác thi đua khen thưởng?

- state: `ok`
- gold: 1  ·  retrieved: 1  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **ranking** → RERANKER / FUSION — retrieved but ranked below position 5

| gold doc | rank in run | in answer? |
|---|---|---|
| `145717` | 6 | no |

**Returned (5):** `278909`, `93252`, `70971`, `20932`, `274422`

<details><summary>gold `145717`</summary>

> Quyet-dinh-826-QD-BTP-2018-chuc-nang-nhiem-vu-co-cau-to-chuc-cua-Vu-Thi-dua-Khen-thuong-383621 Quyet-dinh-826-QD-BTP-2018-chuc-nang-nhiem-vu-co-cau-to-chuc-cua-Vu-Thi-dua-Khen-thuong-383621 BỘ TƯ PHÁP …

</details>

<details><summary>we returned `278909` instead</summary>

> Quyet-dinh-1924-QD-CHK-quy-che-thi-dua-khen-thuong-2016-334843 Quyet-dinh-1924-QD-CHK-quy-che-thi-dua-khen-thuong-2016-334843 BỘ  GIAO THÔNG VẬN TẢI  CỤC HÀNG KHÔNG  VIỆT NAM  -------  CỘNG  HÒA XÃ HỘ …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 15. `516` — recall 0.00, precision 0.00

**Q:** Nguồn kinh phí để chi trả chế độ phụ cấp trách nhiệm theo nghề đối với Viện trưởng VKSNDTC như thế nào?

- state: `ok`
- gold: 1  ·  retrieved: 0  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **retrieval** → RETRIEVER — the document is not in the run at all

| gold doc | rank in run | in answer? |
|---|---|---|
| `86734` | **not in top-81** | no |

**Returned (5):** `235497`, `165830`, `89901`, `107157`, `44425`

<details><summary>gold `86734`</summary>

> Quyet-dinh-138-2005-QD-TTg-che-do-phu-cap-trach-nhiem-Kiem-sat-Dieu-tra-Kiem-tra-vien-nganh-Kiem-sat Quyet-dinh-138-2005-QD-TTg-che-do-phu-cap-trach-nhiem-Kiem-sat-Dieu-tra-Kiem-tra-vien-nganh-Kiem-sa …

</details>

<details><summary>we returned `235497` instead</summary>

> Thong-tu-lien-tich-06-2009-TTLT-BKHDT-BNV-BTC-huong-dan-thuc-hien-QD-45-200-che-do-phu-cap-uu-dai-th Thong-tu-lien-tich-06-2009-TTLT-BKHDT-BNV-BTC-huong-dan-thuc-hien-QD-45-200-che-do-phu-cap-uu-dai-t …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 16. `134762` — recall 0.00, precision 0.00

**Q:** Bảo vệ trong đơn vị sự nghiệp công lập có được ký hợp đồng làm việc không?

- state: `ok`
- gold: 1  ·  retrieved: 0  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **retrieval** → RETRIEVER — the document is not in the run at all

| gold doc | rank in run | in answer? |
|---|---|---|
| `75885` | **not in top-49** | no |

**Returned (5):** `215094`, `196732`, `93859`, `199066`, `103901`

<details><summary>gold `75885`</summary>

> Nghi-dinh-68-2000-ND-CP-thuc-hien-che-do-hop-dong-loai-cong-viec-trong-co-quan-hanh-chinh-nha-nuoc-d Nghi-dinh-68-2000-ND-CP-thuc-hien-che-do-hop-dong-loai-cong-viec-trong-co-quan-hanh-chinh-nha-nuoc- …

</details>

<details><summary>we returned `215094` instead</summary>

> Nghi-dinh-161-2018-ND-CP-tuyen-dung-su-dung-quan-ly-cong-vien-chuc-thuc-hien-che-do-hop-dong-336803 Nghi-dinh-161-2018-ND-CP-tuyen-dung-su-dung-quan-ly-cong-vien-chuc-thuc-hien-che-do-hop-dong-336803  …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 17. `135894` — recall 0.00, precision 0.00

**Q:** Kinh doanh sản phẩm là bánh kẹo cho trẻ em, nghi ngờ sản phẩm không đảm bảo an toàn thì có quyền được truy xuất nguồn gốc hàng hóa không?

- state: `ok`
- gold: 1  ·  retrieved: 1  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **ranking** → RERANKER / FUSION — retrieved but ranked below position 5

| gold doc | rank in run | in answer? |
|---|---|---|
| `219533` | 6 | no |

**Returned (5):** `232471`, `175954`, `153444`, `147980`, `78727`

<details><summary>gold `219533`</summary>

> Luat-an-toan-thuc-pham-2010-108074 Luat-an-toan-thuc-pham-2010-108074 QUỐC  HỘI  -------  CỘNG  HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM  Độc lập – Tự do – Hạnh phúc  ---------  Luật  số: 55/2010/QH12  Hà  Nội,  …

</details>

<details><summary>we returned `232471` instead</summary>

> Quyet-dinh-1390-QD-BCT-2020-cau-hoi-kiem-tra-de-xac-nhan-da-tap-huan-ve-an-toan-thuc-pham-448394 Quyet-dinh-1390-QD-BCT-2020-cau-hoi-kiem-tra-de-xac-nhan-da-tap-huan-ve-an-toan-thuc-pham-448394 BỘ CÔN …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 18. `35128` — recall 0.00, precision 0.00

**Q:** Nhiệm vụ, quyền hạn của Điều tra viên trung cấp được quy định như thế nào?

- state: `ok`
- gold: 1  ·  retrieved: 0  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **retrieval** → RETRIEVER — the document is not in the run at all

| gold doc | rank in run | in answer? |
|---|---|---|
| `234863` | **not in top-86** | no |

**Returned (5):** `20457`, `32881`, `12897`, `177925`, `161984`

<details><summary>gold `234863`</summary>

> Luat-to-chuc-co-quan-dieu-tra-hinh-su-2015-298378 Luat-to-chuc-co-quan-dieu-tra-hinh-su-2015-298378 QUỐC  HỘI  --------  CỘNG  HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM  Độc lập - Tự do – Hạnh phúc  ------------- …

</details>

<details><summary>we returned `20457` instead</summary>

> Thong-tu-02-2023-TT-BXD-huong-dan-hop-dong-xay-dung-557432 Thong-tu-02-2023-TT-BXD-huong-dan-hop-dong-xay-dung-557432 BỘ XÂY DỰNG  -------  CỘNG HÒA XÃ HỘI CHỦ  NGHĨA VIỆT NAM  Độc lập - Tự do - Hạnh  …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 19. `49710` — recall 0.00, precision 0.00

**Q:** Tổ chức giám định tư pháp theo vụ việc trong lĩnh vực công thương phải đáp ứng những điều kiện gì?

- state: `ok`
- gold: 1  ·  retrieved: 1  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **ranking** → RERANKER / FUSION — retrieved but ranked below position 5

| gold doc | rank in run | in answer? |
|---|---|---|
| `305561` | 7 | no |

**Returned (5):** `73552`, `302136`, `51941`, `58354`, `268313`

<details><summary>gold `305561`</summary>

> Thong-tu-30-2016-TT-BCT-giam-dinh-tu-phap-trong-linh-vuc-cong-thuong-324789 Thong-tu-30-2016-TT-BCT-giam-dinh-tu-phap-trong-linh-vuc-cong-thuong-324789 BỘ CÔNG  THƯƠNG  --------  CỘNG HÒA XÃ  HỘI CHỦ  …

</details>

<details><summary>we returned `73552` instead</summary>

> Thong-tu-48-2011-TT-BCT-quan-ly-chat-luong-san-pham-hang-hoa-134170 Thong-tu-48-2011-TT-BCT-quan-ly-chat-luong-san-pham-hang-hoa-134170 BỘ CÔNG THƯƠNG  --------  CỘNG HÒA XÃ HỘI CHỦ  NGHĨA VIỆT NAM  Đ …

</details>

**CATEGORY:** ______________    **WHY:** 

---

### 20. `51188` — recall 0.00, precision 0.00

**Q:** Các hội có tính chất đặc thù ở xã được xác định dựa trên cơ sở nào?

- state: `ok`
- gold: 1  ·  retrieved: 0  ·  in top-5: 0  ·  returned: 0
- slots: used 5, wasted 5, free 0
- dominant loss: **retrieval** → RETRIEVER — the document is not in the run at all

| gold doc | rank in run | in answer? |
|---|---|---|
| `231801` | **not in top-85** | no |

**Returned (5):** `224702`, `25650`, `41657`, `89392`, `64645`

<details><summary>gold `231801`</summary>

> Nhóm Antilles  Peterson - Fuchs - Pollock - Waldin -  Simmonds - Black Prince  Nhóm Guatemala  Anaheim - Benick - Chica - Dickinson  - ltzanna - Edranol - Linda - Nabal - Taylor - Trapp-Schmidt - Wagn …

</details>

<details><summary>we returned `224702` instead</summary>

> Thong-tu-162-2018-TT-BQP-che-do-phu-cap-dac-thu-doi-voi-luc-luong-Canh-ve-thuoc-Bo-Quoc-phong-399832 Thong-tu-162-2018-TT-BQP-che-do-phu-cap-dac-thu-doi-voi-luc-luong-Canh-ve-thuoc-Bo-Quoc-phong-39983 …

</details>

**CATEGORY:** ______________    **WHY:** 

---

## 8. Tally (fill in after categorising)

| category | count | what would fix it |
|---|---|---|
| `granularity` | | _right văn bản, wrong điều (or the reverse)_ |
| `lexical-mismatch` | | _query and source use different words for the same thing_ |
| `numeric` | | _wrong article/decree number matched (Điều 12 vs Điều 112)_ |
| `multi-article` | | _answer spans several documents; we returned one_ |
| `negation` | | _retrieved text states the opposite condition_ |
| `too-general` | | _retrieved a broad law where a specific decree was wanted_ |
| `too-specific` | | _retrieved a narrow provision where the parent was wanted_ |
| `ambiguous-query` | | _question under-specified; several answers defensible_ |
| `label-noise` | | _our answer looks right and the gold looks wrong_ |
| `cutoff` | | _gold ranked inside the top-5 but we did not return it_ |
| `impossible` | | _more than 5 gold documents — the cap forbids full recall_ |

## 9. One paragraph for the paper

_Which method was insufficient, why, and what should the next one fix?_

Template — replace the numbers with yours:

> At phase 1 the system reached Recall 0.7410. Of the 0.2590 shortfall, 0.0605 was documents the retriever never returned, 0.1985 was documents retrieved but ranked outside the five available slots, 0.0000 was documents present in the top five that the cutoff rule discarded, and 0.0000 was unreachable because the question has more than five relevant documents. Manual inspection of 20 failures attributed them chiefly to ___. This motivated ___ in the next phase.

> 
