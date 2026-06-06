# Lộ trình học kỹ thuật — AvatarShield (IVP501)

> Mục tiêu: hiểu **trực giác** từng kỹ thuật + flow tổng thể, không cần nhớ code
> nằm ở đâu. Học theo **thứ tự phụ thuộc kiến thức** (dưới lên), không theo thứ tự
> pipeline chạy.

## Cách dùng file này

- Mỗi kỹ thuật là 1 ô `[ ]`. Khi bạn confirm "đã hiểu", ô được tick `[x]`.
- Trạng thái mỗi tầng: ⬜ chưa học · 🟡 đang giảng, chờ confirm · ✅ đã confirm hiểu.
- Cột "Đã giảng?" đánh dấu phần nội dung đã được trình bày trong chat.
- **Đối chiếu ảnh:** mỗi tầng có mục *Ảnh demo* — mở file PNG tương ứng khi học
  tới đó; mọi ảnh cùng **một frame** ([`13-year-old-Yejin-solo-dance.mp4`](samples/input/13-year-old-Yejin-solo-dance.mp4)
  frame **228**, theme `porcelain-pink`) nên bạn thấy rõ *bước trước → bước sau* trên cùng người.

### Tạo lại bộ ảnh demo

```bash
cd final-project
python scripts/render_pipeline_steps.py \
  --video samples/input/13-year-old-Yejin-solo-dance.mp4 \
  --frame 228 --theme porcelain-pink
```

Thư mục output: [`data/output/demo_steps/`](data/output/demo_steps/)

**Video đầy đủ** (cùng clip + theme):

```bash
python scripts/render_clip.py \
  --video samples/input/13-year-old-Yejin-solo-dance.mp4 \
  --theme porcelain-pink
```

→ [`…_porcelain-pink.mp4`](data/output/13-year-old-Yejin-solo-dance_porcelain-pink.mp4) (287 frame @ 24 fps)

> **Bối cảnh frame 228** (đọc một lần, rồi mọi ảnh cùng cảnh): clip **dọc 1080×1920**, studio nhảy —
> **tường gradient xanh lam → hồng tím**, sàn xám phản chiếu ánh màu. Dancer **full-body** giữa khung,
> áo crop đen + quần short đen + giày combat; tư thế nhảy (chân rộng, tay gần ngực). **Mặt lớn hơn clip cũ**
> (~1/6 chiều cao) — khung xanh bbox Haar ôm mặt frontal. Frame chọn = bbox Haar **lớn nhất** trong video.

### Bản đồ ảnh — học tới đâu mở tới đó

| Ảnh | Đường dẫn | Tầng | Trực giác một câu |
|---|---|---|---|
| Tổng quan | [demo_steps_grid.png](data/output/demo_steps/demo_steps_grid.png) | 0→5 | Comic strip: studio thật → mask → phẳng → nét → palette → chỉ đầu anime |
| Gốc | [00_original.png](data/output/demo_steps/00_original.png) | 0, 4 | Clip dọc studio + hộp xanh trên mặt |
| HSV | [01_hsv_mask.png](data/output/demo_steps/01_hsv_mask.png) | 1 | Silhouette trắng — da/tay/chân; **thân áo đen** thành lỗ đen giữa blob |
| YCbCr | [02_ycbcr_mask.png](data/output/demo_steps/02_ycbcr_mask.png) | 1 | Silhouette **rộng hơn** HSV; giữ má, có thể bắt vùng da sáng trên tường hồng |
| Da cuối | [03_skin_mask.png](data/output/demo_steps/03_skin_mask.png) | 1, 2 | **Oval trắng** trong bbox mặt (mặt gần nên đọc được, không còn cả người) |
| Bộ lọc | [04_spatial_filters.png](data/output/demo_steps/04_spatial_filters.png) | 2 | Zoom mặt: trái mờ nhòe, giữa sạch hạt, phải da như sơn |
| Feather | [05_head_feather.png](data/output/demo_steps/05_head_feather.png) | 2, 4 | Đầu **tối mờ** + halo trên tường; chân/áo/sàn studio sáng bình thường |
| Bilateral | [06_bilateral.png](data/output/demo_steps/06_bilateral.png) | 2, 3 | Cả frame mịn — gradient tường xanh–hồng vẫn đọc được |
| K-means L | [07_kmeans_L.png](data/output/demo_steps/07_kmeans_L.png) | 1, 3 | Poster 3–4 band: người, tường, sàn thành mảng phẳng |
| XDoG | [08_xdog.png](data/output/demo_steps/08_xdog.png) | 3 | Poster + **viền đen** quanh người, mũi, viền giày |
| Cel core | [09_cel_shade_core.png](data/output/demo_steps/09_cel_shade_core.png) | 3 | Anime draft: màu **rực hơn** 07, nét rõ |
| Reinhard | [10_reinhard_regions.png](data/output/demo_steps/10_reinhard_regions.png) | 1, 3 | Da **porcelain hồng**, môi/mắt đậm; tóc tím-đen, vẫn có highlight |
| Composite | [11_composite.png](data/output/demo_steps/11_composite.png) | 2, 3 | **Chỉ đầu** anime; crop top, short, sàn, tường gradient **y nguyên** 00 |
| EMA | [12_temporal_ema.png](data/output/demo_steps/12_temporal_ema.png) | 5 | Hai bản cel cạnh nhau — khác **nhẹ**; xem video mới thấy "boiling" |

### Trực giác từng ảnh (chi tiết — đọc rồi mở ảnh để khớp)

Mỗi mục: **Giống như** · **Mắt thấy** · **So với bước trước** · **Lưu ý**

**[00 — Gốc](data/output/demo_steps/00_original.png)** — Giống frame TikTok/Reels studio. Mắt thấy: dancer đen trên nền xanh–hồng; hộp xanh chặt mặt (Haar). So với sau: đây là "thật", mọi bước sau chỉ sửa đầu hoặc mask. Lưu ý: bbox không phải mask da.

**[01 — HSV](data/output/demo_steps/01_hsv_mask.png)** — Giống **cắt stencil** theo "màu da + độ sáng". Mắt thấy: nền đen; silhouette **cả người** trắng, **lỗ đen** ở crop top (vải đen không match hue da). So với 00: không còn màu. Lưu ý: tường hồng có thể có chấm trắng nếu chroma gần da.

**[02 — YCbCr](data/output/demo_steps/02_ycbcr_mask.png)** — Giống stencil theo **tông da** bất kể sáng tối. Mắt thấy: blob **đầy hơn 01** quanh mặt/cổ/tay. So với 01: ít mất da dưới hàm. Lưu ý: vùng sáng trên tường hồng đôi khi trắng — 03 giao để loại.

**[03 — Da cuối](data/output/demo_steps/03_skin_mask.png)** — Giống **con tem** mặt: 01∩02 + morph + oval. Mắt thấy: **oval trắng** rõ trong khung đen (mặt gần camera). So với 01/02: không còn silhouette cả người. Lưu ý: mask logic, không preview đẹp.

**[04 — Bộ lọc](data/output/demo_steps/04_spatial_filters.png)** — Ba kính: sương / kính cường lực / app làm mịn da. **Trái:** mọi cạnh mờ. **Giữa:** ít hạt, cạnh sắc. **Phải:** da phẳng, mũi/môi rõ — cảm giác trước cel-shade.

**[05 — Feather](data/output/demo_steps/05_head_feather.png)** — Giống **kính mờ hình bầu dục** trên đầu. Mắt thấy: đầu **tối/xám** + vòng mờ trên tường; chân, giày, sàn studio sáng bình thường. So với 00: preview alpha blend, chưa cel-shade.

**[06 — Bilateral](data/output/demo_steps/06_bilateral.png)** — Giống filter làm mịn **cả khung**. Mắt thấy: texture da/áo mờ; gradient tường vẫn đọc. So với 00: chưa band, chưa nét ink.

**[07 — K-means L](data/output/demo_steps/07_kmeans_L.png)** — Giống **tranh giấy 3–4 lớp sáng**. Mắt thấy: người + tường xanh–hồng + sàn thành **mảng phẳng**. So với 06: từ mịn → **bậc thang** sáng/tối.

**[08 — XDoG](data/output/demo_steps/08_xdog.png)** — Giống **tô viền đen** lên 07. Mắt thấy: contour người, mũi, tóc, viền giày; má không lốm đốm.

**[09 — Cel core](data/output/demo_steps/09_cel_shade_core.png)** — Giống **frame hoạt hình** trước tô palette. Mắt thấy: như 08, màu **tươi hơn**. So với 10: da chưa porcelain hồng.

**[10 — Reinhard](data/output/demo_steps/10_reinhard_regions.png)** — Giống **đổi palette**: da hồng, môi/mắt accent, tóc tím-đen. Mắt thấy: full-body poster; tóc đổi màu nhưng còn highlight. So với 11: cả frame stylized, chưa ghép nền gốc.

**[11 — Composite](data/output/demo_steps/11_composite.png)** — Giống **dán đầu anime lên frame studio**. Mắt thấy: đầu stylized; crop top, short, giày, tường gradient **y hệt 00**. Output quan trọng nhất.

**[12 — EMA](data/output/demo_steps/12_temporal_ema.png)** — Hai bản in cùng khung: trái hơi "run", phải ổn hơn. Mắt thấy: **gần giống** trên ảnh tĩnh. Lưu ý: "boiling" thật cần **video** phase8 (`ivp_per_frame` vs `ivp_full`).

**Ảnh bổ sung (chạy script phase, chưa nằm trong `demo_steps/`):**

| Mục đích | Script | Output gợi ý |
|---|---|---|
| One Euro bbox (video) | `scripts/phase1_test.py` | `data/output/phase1_bbox.mp4` |
| So sánh 5 theme | `scripts/phase6_render_previews.py` | `data/output/phase6_grid.png` |
| Evaluation privacy | `scripts/eval_*.py` | `RESULTS.md`, `data/output/phase8/clip/` |

---

## Bản đồ phụ thuộc

```
TẦNG 5  Temporal: EMA → One Euro filter ............... (video hết "boiling")
            ▲
TẦNG 4  Detection & Tracking: Haar → CSRT ............. (tìm & bám mặt)
            ▲
TẦNG 3  Thuật toán lõi: K-means · XDoG · Reinhard · Warp
            ▲
TẦNG 2  Lọc không gian & hình thái: Gaussian/Median/Bilateral · Morphology · Feather
            ▲
TẦNG 1  Color spaces: BGR · HSV · YCbCr · Lab ......... (XƯƠNG SỐNG)
            ▲
TẦNG 0  Nền tảng ảnh số: numpy array · mask · vẽ hình
```

## Bảng tiến độ tổng

| Tầng | Chủ đề | Trạng thái | Ảnh tới tầng này |
|---|---|---|---|
| 0 | Nền tảng ảnh số | ✅ | → [00_original](data/output/demo_steps/00_original.png) |
| 1 | Color spaces | ✅ | → [03_skin_mask](data/output/demo_steps/03_skin_mask.png) |
| 2 | Lọc không gian & hình thái | 🟡 | → [05_head_feather](data/output/demo_steps/05_head_feather.png) |
| 3 | Thuật toán lõi | ⬜ | → [11_composite](data/output/demo_steps/11_composite.png) |
| 4 | Detection & Tracking | ⬜ | → [00_original](data/output/demo_steps/00_original.png) (bbox) |
| 5 | Temporal filtering | ⬜ | → [12_temporal_ema](data/output/demo_steps/12_temporal_ema.png) |
| 6 | Evaluation | ⬜ | → `RESULTS.md` (không có ảnh từng bước) |

---

## TẦNG 0 — Nền tảng ảnh số ✅

- [x] Ảnh = mảng `numpy` shape `(H, W, 3)`, dtype `uint8` (0–255), thứ tự kênh **BGR**.
- [x] Mask nhị phân = ảnh 1 kênh chỉ chứa `0` / `255` để đánh dấu vùng.
- [x] Phép trên mask: `cv2.bitwise_and / or / not` = giao / hợp / phủ định.

**Câu chốt:** mọi kỹ thuật sau chỉ là "biến đổi mảng số này theo một quy tắc".

### Ảnh demo — Tầng 0

| Ảnh | Mô tả đối chiếu |
|---|---|
| [00_original.png](data/output/demo_steps/00_original.png) | Đây là **ảnh màu BGR** đầu vào: mỗi pixel = 3 số 0–255. Không có mask — toàn frame là dữ liệu. |
| [01_hsv_mask.png](data/output/demo_steps/01_hsv_mask.png) · [02_ycbcr_mask.png](data/output/demo_steps/02_ycbcr_mask.png) | **Mask nhị phân**: trắng = vùng được chọn, đen = loại. Hai mask độc lập trước khi `bitwise_and`. |
| [03_skin_mask.png](data/output/demo_steps/03_skin_mask.png) | **Giao hai mask** (AND): chỉ pixel mà *cả* HSV *và* YCbCr đồng ý mới trắng — minh họa `bitwise_and`. |

---

## TẦNG 1 — Color spaces ✅

> **Ý tưởng lớn nhất:** tách *độ sáng* ra khỏi *màu sắc*. Trong BGR, sáng và màu
> trộn vào cả 3 kênh → cùng màu da nhưng sáng/tối cho 3 số khác hẳn nhau. Các hệ
> dưới đây đều gom độ sáng vào **1 kênh** để "màu da" thành vùng ổn định.

| Hệ | Kênh độ sáng | Kênh màu |
|---|---|---|
| HSV | **V** | H (màu gì), S (đậm/nhạt) |
| YCbCr | **Y** | Cb, Cr |
| Lab | **L** | a (lục↔đỏ), b (lam↔vàng) |

- [x] **HSV** — hình trụ màu: H = góc (màu gì, **vòng tròn** nên đỏ ở cả ~0 và ~179),
  S = bán kính (rực/nhạt), V = chiều cao (sáng/tối).
  - Gotcha: OpenCV H ∈ [0,179]; da ấm nằm ở *cả hai* đầu → phải OR hai khoảng hue.
  - Điểm yếu: hue/sat của da vẫn dịch nhẹ khi bóng mạnh → rớt da trong bóng.
  - **Ảnh:** [01_hsv_mask.png](data/output/demo_steps/01_hsv_mask.png) — so với gốc: vùng má/cằm sáng thường trắng; **bóng dưới mũi/cằm** dễ bị đen (mất da).
- [x] **YCbCr** — Y = luma, Cb/Cr = chroma. Màu da mọi sắc tộc tạo **cụm chặt trong
  (Cb,Cr)** bất kể sáng tối → **bền với ánh sáng**.
  - Gotcha: OpenCV trả thứ tự `Y, Cr, Cb`. Điểm yếu: bắt nhầm nền cam (gỗ, tường vàng).
  - **Ảnh:** [02_ycbcr_mask.png](data/output/demo_steps/02_ycbcr_mask.png) — thường **giữ da trong bóng** tốt hơn HSV; có thể bắt thêm vùng nền ấm (so khung áo/nền).
- [x] **Giao HSV ∩ YCbCr** — "hai nhân chứng độc lập, chỉ tin khi cả hai cùng khai".
  Bóng-da được YCbCr cứu; nền-cam bị HSV loại → mask da sạch hơn dùng riêng.
  - **Ảnh:** [03_skin_mask.png](data/output/demo_steps/03_skin_mask.png) — nhỏ hơn, sạch hơn từng mask riêng; chỉ còn blob mặt (sau morph + CC + face oval).
- [x] **Lab** — gần cảm nhận mắt người. L = shading (đổ bóng), a/b = tông màu.
  - Dùng 1: posterize **chỉ kênh L** → cel-shade mà không hỏng hue.
  - Dùng 2: color transfer (Reinhard rút gọn = mean-shift). Da/môi/mắt/mày: kéo
    trung bình **cả L,a,b** về tông đích, NHƯNG giữ độ lệch chuẩn → vẫn còn nếp
    sáng-tối (shading) tự nhiên. ("Giữ shading" = giữ std, KHÔNG phải giữ nguyên L.)
  - Tóc thì khác: **chroma-only** (chỉ kéo a,b, giữ nguyên L) để giữ độ bóng tóc gốc.
  - Gotcha: OpenCV co L về [0,255], a,b lấy 128 làm tâm.
  - **Ảnh L:** [07_kmeans_L.png](data/output/demo_steps/07_kmeans_L.png) — chỉ **L** bị gom 3 band; màu da/hair vẫn tự nhiên quanh band.
  - **Ảnh Reinhard:** [10_reinhard_regions.png](data/output/demo_steps/10_reinhard_regions.png) — so [09_cel_shade_core](data/output/demo_steps/09_cel_shade_core.png): tông porcelain-pink lên da/môi; tóc đổi hue nhưng **độ sáng tóc** gần gốc hơn da.

**Bảng quyết định (học thuộc):**

| Muốn làm gì | Dùng | Vì sao |
|---|---|---|
| Tìm vùng da | HSV ∩ YCbCr | bù điểm yếu nhau |
| Posterize sáng (cel-shade) | Lab, kênh L | tách sáng khỏi màu |
| Đổi tông da/môi/mắt theo theme | Lab, mean-shift cả L,a,b | giữ std → còn shading |
| Đổi màu tóc | Lab, chỉ a,b | giữ L = giữ độ bóng tóc gốc |
| Tăng độ rực | HSV, kênh S | rực = bán kính trụ — thấy ở [09](data/output/demo_steps/09_cel_shade_core.png) đậm hơn [07](data/output/demo_steps/07_kmeans_L.png) |

Code cốt lõi:
```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)   # thứ tự Y, Cr, Cb
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)     # L∈[0,255], a,b tâm 128
skin = cv2.bitwise_and(mask_hsv, mask_ycc)     # chỉ giữ pixel cả 2 đồng ý
```

**Câu chốt cả tầng:** *đổi sang không gian tách đúng thứ cần chỉnh → sửa đúng kênh
→ đổi về BGR.*

**Chuỗi ảnh học Tầng 1:** [00](data/output/demo_steps/00_original.png) → [01](data/output/demo_steps/01_hsv_mask.png) → [02](data/output/demo_steps/02_ycbcr_mask.png) → [03](data/output/demo_steps/03_skin_mask.png) → [07](data/output/demo_steps/07_kmeans_L.png) → [10](data/output/demo_steps/10_reinhard_regions.png)

---

## TẦNG 2 — Lọc không gian & hình thái 🟡

- [x] **Convolution / kernel** — trượt ma trận nhỏ qua ảnh, mỗi pixel = tổng có
  trọng số vùng lân cận. Nền của mọi bộ lọc.
  - **Ảnh:** [04_spatial_filters.png](data/output/demo_steps/04_spatial_filters.png) — cả ba cột đều là convolution; khác **kernel / trọng số**.
- [ ] **Gaussian blur** — làm mờ đều theo trọng số chuông. Dùng cho feather, nền DoG.
  - **Ảnh:** cột **trái** của [04](data/output/demo_steps/04_spatial_filters.png); mờ đều cả cạnh tóc/da.
  - **Ảnh feather:** [05_head_feather.png](data/output/demo_steps/05_head_feather.png) — Gaussian trên **mask** (không phải màu), tạo viền mờ khi blend.
- [ ] **Median blur** — thay pixel bằng trung vị lân cận; khử nhiễu hạt, giữ cạnh.
  - **Ảnh:** cột **giữa** [04](data/output/demo_steps/04_spatial_filters.png) — ít hạt nhiễu hơn Gaussian, cạnh còn khá sắc.
- [ ] **Bilateral filter** — *làm phẳng nhưng GIỮ cạnh*: chỉ trộn pixel vừa gần về
  vị trí vừa gần về màu → da phẳng mà viền mặt còn nguyên. **Trái tim cel-shade.**
  - **Ảnh:** cột **phải** [04](data/output/demo_steps/04_spatial_filters.png) vs [06_bilateral.png](data/output/demo_steps/06_bilateral.png) (iterated ×3 trên cả frame).
- [ ] **Morphology (open/close)** trên mask: open = xoá đốm nhiễu, close = lấp lỗ.
  - **Ảnh:** so [01](data/output/demo_steps/01_hsv_mask.png)/[02](data/output/demo_steps/02_ycbcr_mask.png) (thô, lỗ hổng) với [03](data/output/demo_steps/03_skin_mask.png) (đặc, một blob mặt) — close lấp lỗ mắt/miệng, open xoá chấm nhiễu.
- [ ] **Connected components** — giữ blob lớn nhất (loại các mảng da rời rạc ngoài mặt).
  - **Ảnh:** [03](data/output/demo_steps/03_skin_mask.png) — chỉ còn một vùng da trong bbox mặt; tay/nền da (nếu có) bị loại.
- [ ] **Feather** — Gaussian blur lên mask {0,255} → alpha mượt [0,1] để blend không viền.
  - **Ảnh:** [05_head_feather.png](data/output/demo_steps/05_head_feather.png) — vùng đầu hơi tối/xám = preview alpha; [11_composite](data/output/demo_steps/11_composite.png) không có đường cắt cứng quanh tóc.

**Chuỗi ảnh học Tầng 2:** [03](data/output/demo_steps/03_skin_mask.png) → [04](data/output/demo_steps/04_spatial_filters.png) → [06](data/output/demo_steps/06_bilateral.png) → [05](data/output/demo_steps/05_head_feather.png) → [11](data/output/demo_steps/11_composite.png)

---

## TẦNG 3 — Bốn thuật toán lõi ⬜

- [ ] **K-means quantization** — gom giá trị kênh L về K cụm; thay mỗi pixel bằng
  tâm cụm → gradient mượt thành vài "band" phẳng = bản chất hiệu ứng cel.
  - **Ảnh:** [06](data/output/demo_steps/06_bilateral.png) → [07](data/output/demo_steps/07_kmeans_L.png) — da chuyển từ mịn sang **3 tầng sáng** rõ.
- [ ] **DoG → XDoG** — hiệu 2 ảnh Gaussian khác sigma = phát hiện cạnh; XDoG thêm
  ngưỡng mềm `tanh` cho nét "vẽ tay" thay vì lốm đốm.
  - **Ảnh:** [07](data/output/demo_steps/07_kmeans_L.png) → [08](data/output/demo_steps/08_xdog.png) — nét đen dọc mũi, cằm, tóc; không phải noise rải trên má.
- [ ] **Reinhard color transfer** — kéo *trung bình* màu vùng về tông đích trong Lab,
  giữ độ lệch chuẩn (= giữ shading). Ở đây rút gọn còn "mean-shift".
  - **Ảnh:** [09](data/output/demo_steps/09_cel_shade_core.png) → [10](data/output/demo_steps/10_reinhard_regions.png) — theme porcelain-pink: da sáng hơn, môi/mắt đậm hơn; so vùng tóc vs da (tóc giữ L).
**Chuỗi ảnh học Tầng 3 (cel-shade):** [06](data/output/demo_steps/06_bilateral.png) → [07](data/output/demo_steps/07_kmeans_L.png) → [08](data/output/demo_steps/08_xdog.png) → [09](data/output/demo_steps/09_cel_shade_core.png) → [10](data/output/demo_steps/10_reinhard_regions.png) → [11](data/output/demo_steps/11_composite.png)

---

## TẦNG 4 — Detection & Tracking ⬜

- [ ] **Haar cascade (Viola–Jones)** — integral image + đặc trưng Haar + cascade
  AdaBoost. Phát hiện mặt từng-frame, không GPU. (frontal → profile → flip).
  - **Ảnh:** [00_original.png](data/output/demo_steps/00_original.png) — khung xanh = bbox Haar sau One Euro trên frame 228 (Yejin solo dance).
- [ ] **Skin-coverage gate** — loại bbox có < 30% da bên trong (chống bắt nhầm
  tường/áo len khi mắt nhắm).
  - **Ảnh:** không có frame riêng — hiểu qua logic: nếu [03](data/output/demo_steps/03_skin_mask.png) gần rỗng trong bbox → reject detection.
- [ ] **Correlation filter tracker (CSRT/KCF)** — bám patch từ lần Haar cuối qua các
  frame Haar miss, có giới hạn tuổi chống trôi. Không train runtime.
  - **Ảnh:** cần **video** — chạy `scripts/phase1_test.py` → `phase1_bbox.mp4` (bbox mượt khi Haar miss vài frame).

**Chuỗi ảnh học Tầng 4:** [00_original](data/output/demo_steps/00_original.png) + [05_head_feather](data/output/demo_steps/05_head_feather.png) (geometry từ bbox) + `phase1_bbox.mp4` (thời gian)

---

## TẦNG 5 — Temporal filtering ⬜

- [ ] **EMA** — `new = α·prev + (1-α)·curr`. Trung bình trượt mũ; nền mọi làm mượt
  theo thời gian (dùng cho K-means centers để hết flicker, cho eye tracker).
  - **Ảnh:** [12_temporal_ema.png](data/output/demo_steps/12_temporal_ema.png) — **trái** EMA off, **phải** EMA 0.85; nhìn band sáng trên má/cằm: phải ít "nhảy" hơn (cùng frame, khác lịch sử 30 frame trước).
- [ ] **One Euro filter** — low-pass *thích nghi theo tốc độ*: `cutoff = min + β·|ẋ|`.
  Chậm → mượt mạnh, nhanh → bám sát (ít trễ). Làm bbox mặt hết rung.
  - **Ảnh:** một frame không đủ — xem `phase1_bbox.mp4` hoặc so `ivp_per_frame` vs `ivp_full` trong `data/output/phase8/clip/` (`RESULTS.md` §7).
- [ ] **Vì sao cần temporal** — K-means init ngẫu nhiên → band nhảy giữa frame =
  "boiling"; sort centers + EMA blend để khoá band lại.
  - **Ảnh:** [12](data/output/demo_steps/12_temporal_ema.png) + bảng MAD trong `RESULTS.md` (số liệu, không ảnh từng frame).

**Chuỗi ảnh học Tầng 5:** [12_temporal_ema](data/output/demo_steps/12_temporal_ema.png) → video IVP smoothing ON/OFF (phase8)

---

## TẦNG 6 — Evaluation (hiểu khái niệm) ⬜

- [ ] **FFT high-frequency ratio** — biến đổi Fourier; đo % năng lượng ở tần số cao
  (texture/chi tiết). Nghịch lý: với cel-shade nó *tăng* vì line-art XDoG là tần số
  cao → metric này không faithful cho cel-shade.
  - **Đối chiếu:** [08_xdog](data/output/demo_steps/08_xdog.png) / [09](data/output/demo_steps/09_cel_shade_core.png) (nhiều HF geometric) vs [06_bilateral](data/output/demo_steps/06_bilateral.png) (ít texture) — rồi đọc `RESULTS.md` § FFT.
- [ ] **FaceNet re-id** — nhúng mặt thành vector, đo khoảng cách cosine, < 0.7 coi
  là "cùng người". Metric privacy chính của dự án.
  - **Đối chiếu:** [11_composite](data/output/demo_steps/11_composite.png) (đích IVP) vs baseline trong `data/output/phase8/clip/` — số re-id trong `RESULTS.md` § Privacy.

**Chuỗi học Tầng 6:** đọc `RESULTS.md` + (tuỳ chọn) render `eval_render_baselines.py` rồi mở grid `phase8/clip/grid_frame_*.png` nếu đã chạy eval.

---

## Nhật ký học

| Ngày | Tầng/mục confirm hiểu | Ghi chú |
|---|---|---|
| 2026-06-03 | Tầng 0 | nền tảng, đã nắm |
| 2026-06-03 | Tầng 1 | color spaces — confirm hiểu |
| 2026-06-04 | Ảnh demo | `data/output/demo_steps/` — `clip.mp4` frame 262, porcelain-pink |
| 2026-06-05 | Ảnh demo | `13-year-old-Yejin-solo-dance.mp4` frame 228 — re-render + cập nhật mô tả STUDY |
| 2026-06-05 | Video IVP | `…_porcelain-pink.mp4` (cel-shade + theme palette) — `render_clip.py` |
