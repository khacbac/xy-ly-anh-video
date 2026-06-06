# Track A Presentation Script — AvatarShield (Classical IVP)

**Tổng thời lượng:** ~6–7 phút (9 slide, 40–50s/slide)
**Speaker:** Bạn (Track A)
**Bắt đầu:** Slide 5 — kết thúc: Slide 13 — pass mic ở Slide 14

---

## SLIDE 5 — Classical IVP Pipeline (road-map) · ~50s

> "Cảm ơn [member trước]. Mình là Bac, trình bày Track A — Classical IVP, codename **AvatarShield**.
>
> Slide này là **road-map**. Trên cùng là 7 panel head-crop từ chính clip demo (Yejin, frame 262) — show output các stage có visible visual diff. Phía dưới là sơ đồ khối đầy đủ với One Euro / Morphology / HSV Saturation Lift (3 stage không có panel riêng vì hoặc là temporal, hoặc visual diff quá nhỏ). Mình đi nhanh từ trái sang phải:
>
> 1. **Frame + Haar Detect** — RGB vào, Haar cascade trả bbox head (panel 1, hộp xanh).
> 2. **One Euro filter** — smooth bbox `(cx, cy, w, h)` **ngay sau detect**, hold-last-good ≤ 6 frame nếu Haar miss. *(Temporal — không có panel.)*
> 3. **Skin mask + Morphology** — HSV ∩ YCbCr threshold rồi connected-components dọn noise (panel 2).
> 4. **Head Ellipse** — Gaussian-feather ellipse từ bbox → ROI mask mềm, không răng cưa biên (panel 3).
> 5. **Cel-Shade Core** — 4 sub-step **bắt buộc đúng thứ tự**: bilateral × 3 dập texture (panel 4) → Lab K-means posterize L vào 3 band (panel 5) → XDoG line art + HSV saturation lift (visual diff nhỏ với K-means panel; gộp vào panel 6 cùng Reinhard). **EMA α=0.85** áp lên K-means cluster center giữa các frame — đây là half-2 của temporal stability (half-1 là One Euro ở step 2).
> 6. **Reinhard per region** — Lab color transfer riêng cho skin / lips / eyes / brows / hair, hair chỉ chuyển chroma → self-recognition cue. Visual diff vs cel-shade là chroma shift nhỏ; effect rõ nhất ở composite step.
> 7. **Alpha Composite** — feathered head lên frame gốc, body + background giữ nguyên → visual story preserved (panel 7).
>
> Frame in, composite out, 0 neural model, CPU-only."

**Tip:** lướt tay theo strip 7 panel — chỉ rõ panel 6 = XDoG + HSV lift + Reinhard (gộp 3 stage visual-close để tránh panel trùng). Nhấn mạnh **2 điểm temporal stability không cùng chỗ** (One Euro early, EMA inside Cel-Shade Core) — đó là detail dễ bị hỏi. Deep-dive Cel-Shade Core để dành slide 8.

---

## SLIDE 6 — Why Pure-Classical IVP? · ~45s

> "Hiện tại các approach có sẵn đều có vấn đề:
>
> - **GAN stylization** kiểu VToonify đẹp nhưng *không giải thích được* — defense bị hỏi 'tại sao pixel này màu này' là tịt.
> - **Full-face replacement** kiểu DeepPrivacy thay nguyên mặt — subject mất *agency*, không nhận ra chính mình trong video của mình.
> - **AR filter** của Snapchat/TikTok chỉ là cosmetic, không phải formal anonymization.
> - Tất cả đều là **black-box** — heavy lifting bị giấu đi.
>
> Pure-classical của mình ngược lại — chỉ OpenCV, NumPy, Pillow, **0 neural model**. Chạy CPU-only 2.7 fps trên M1. Mỗi stage là 1 IVP operator trong textbook — defendable end-to-end.
>
> Quan trọng nhất: identity mới sinh từ **theme palette + self-recognition cues** — subject vẫn nhận ra mình, người lạ thì không."

**Pointer:** chỉ vào ảnh phải — "đây là input bên trái, output AvatarShield bên phải, cùng 1 frame".

---

## SLIDE 7 — Pipeline Overview · ~45s

> "Pipeline 6 stage:
>
> 1. **Detect** — Haar cascade frontal + profile, classical Viola-Jones.
> 2. **Skin mask** — giao của HSV và YCbCr threshold, plus morphology để dọn noise.
> 3. **Cel-shade core** — 4 bước nhỏ, mình deep-dive ở slide kế.
> 4. **Recolor** — Reinhard color transfer trong Lab space, **per region** — skin, lips, eyes, brows, hair riêng biệt. Hair chỉ chuyển chroma, giữ L của user — đây là self-recognition cue.
> 5. **Stabilize** — One Euro filter trên bbox + EMA trên K-means centers — chống flicker giữa các frame.
> 6. **Composite** — alpha blend ellipse đầu lên frame gốc, body và background giữ nguyên.
>
> Top là 5 frame demo: original → skin mask → bilateral → K-means → composite."

**Pointer:** lướt tay qua từng frame trong strip khi đọc.

---

## SLIDE 8 — Cel-Shade Core · ~45s

> "Slide 5 đã list 4 bước. Slide này show **tại sao thứ tự này là duy nhất đúng**:
>
> 1. **Bilateral PHẢI trước K-means** — nếu để micro-texture lại, cluster luminance sẽ ăn vào noise → flat band bị răng cưa. Bilateral × 3 dập texture nhưng preserve edge.
> 2. **Lab K-means trên kênh L** — posterize luminance vào 3 cluster → cel-shade flat band.
> 3. **XDoG PHẢI sau K-means** — vẽ line art trên ảnh đã flat, contour mới sạch. Vẽ trên ảnh gốc thì line bị nhiễu bởi micro-edge.
> 4. **HSV saturation lift cuối** — sau khi đã có structure (flat + line), mới boost chroma → look ấm, không phá band.
>
> Mỗi bước đều **tham số hóa bởi theme palette** — chuyển theme là 4 scalar (edge strength, saturation boost, luminance levels, chroma levels) đổi theo."

---

## SLIDE 9 — 5 Theme Palettes × Per-Region Reinhard · ~45s

> "5 theme palette: porcelain-pink, tan-amber, ivory-violet, bronze-teal, peach-noir. Mỗi theme = **5 Lab triple** cho skin/lips/eyes/brows/hair, plus 4 style scalar.
>
> Reinhard transfer chạy **per region** — tại sao? Vì global Reinhard sẽ wash skin tone vào hair. Per-region giữ semantic: môi vẫn là môi, lông mày vẫn là lông mày.
>
> Quan trọng nhất: **hair chỉ chuyển chroma**, preserve luminance L của user. Đây là *self-recognition cue* — subject nhìn vào sẽ thấy 'a stylized themselves', không phải người lạ. Người ngoài thì thấy 1 identity hoàn toàn scrambled."

**Pointer:** chỉ vào swatch grid — "5 hàng là 5 theme, 5 cột là 5 region".

---

## SLIDE 10 — Temporal Stability · ~40s

> "Classical pipeline thường bị **boiling** frame-by-frame — K-means cluster nhảy lung tung. Mình dùng 2 cơ chế:
>
> - **One Euro filter** trên bbox `(cx, cy, w, h)` — adaptive low-pass, plus hold-last-good ≤ 6 frame nếu Haar miss.
> - **EMA α=0.85** trên 3 K-means cluster center — posterize band locked across frames.
>
> Ablation trên 667 frame: head-region MAD giảm từ 20.17 xuống 17.66 — **excess MAD giảm ~30%**, std halved. Bên trái là EMA off, bên phải là EMA on — cùng source, eye-check thấy ngay no boiling."

---

## SLIDE 11 — Privacy Results · ~50s (slide quan trọng nhất)

> "Đây là kết quả chính. Setup: FaceNet vggface2, cosine distance threshold 0.7, every 15th frame, n=33 sample trên clip.mp4.
>
> 3 baseline: Gaussian blur σ=12, mosaic 8×8, và AvatarShield IVP.
>
> | Baseline | n paired | re-id rate |
> | Blur | 5 | 0.800 |
> | Mosaic | 17 | 0.941 |
> | **IVP** | **24** | **0.667** |
>
> Hai headline:
>
> 1. **IVP đạt re-id thấp nhất** — 0.667 — thấp hơn cả blur lẫn mosaic.
> 2. **VÀ giữ detector survivability cao nhất** — 24/33 frame Haar vẫn detect được.
>
> Để ý cột n — blur có n=5, nghĩa là detector chết trên 28/33 frame. Blur 'thắng' re-id là vì **detector không detect nổi mặt nữa** — false win. Mosaic cũng vậy, n=17.
>
> IVP là baseline duy nhất giữ được cả 2: head locate-able **và** identity hidden. Đây là trade-off đúng theo design — preserve visual story, scramble identity."

**Pointer:** chỉ vào row **AvatarShield IVP** trong table, nhấn mạnh **n=24** và **0.667**.

---

## SLIDE 12 — Joint Trade-Off · ~40s

> "Để confirm trade-off không phải accident, mình plot **FFT high-frequency ratio × re-id rate** trên Yejin clip, n=20.
>
> - **Blur** ở top-left — texture suppression cao nhất nhưng detector chết.
> - **Mosaic** ở giữa — block cả 2 trục.
> - **Control và IVP** ở top-right — vì XDoG line art **là** HF signal by design, IVP không thua control về HF.
>
> Insight: **HF ratio alone rewards blur unfairly**. Re-id với detector-survivability count mới là honest metric. IVP wins joint trade-off mà không fake bất kỳ trục nào."

**Pointer:** chỉ vào điểm IVP (top-right) và điểm Blur (top-left) — "đây là 2 cực".

---

## SLIDE 13 — Limitations & Non-Claims · ~40s

> "Honest disclosure — đây là những gì mình **không** claim:
>
> - **n=1 source clip** cho headline number. Dancing clip thì Haar fail khi face nhỏ hơn 5% chiều cao frame.
> - **Single embedder** — FaceNet only, chưa benchmark ArcFace.
> - **No human study** — self-recognition study A/B deferred.
> - **Threat model**: defend *casual* face matching, KHÔNG defend motivated attacker với reference photo.
>
> AvatarShield **không phải** K-anonymity guarantee. Đây là **visual-story-preserving anonymization** cho child-safe sharing, defended bằng transparent classical IVP operators — không phải bằng black-box trust.
>
> Đó là Track A. Mình pass mic cho [member 2] để present Track B — ML-Assisted SafeSwap."

**Tip:** câu cuối là handoff — gật nhẹ, lùi 1 bước, để member kia tiến lên.

---

# Defense Q&A — câu hỏi có thể bị hỏi

| Câu hỏi | Trả lời ngắn |
|---|---|
| "Tại sao không dùng MediaPipe FaceMesh?" | Vì spec yêu cầu pure-classical — MediaPipe là neural. Haar đủ tốt cho frontal face >5% frame. |
| "Tại sao Haar mà không HOG hay cascade khác?" | Haar có sẵn OpenCV, frontal+profile cascade đủ recall cho clip face >5% chiều cao frame. HOG chậm hơn mà không gain accuracy ở use-case này. |
| "n=1 clip thì làm sao trust kết quả?" | Đây là research prototype, không phải production claim. Slide Limitations đã disclose. |
| "Tại sao threshold cosine 0.7?" | Default của FaceNet vggface2; cùng threshold cho all 3 baseline → fair comparison. |
| "EMA 0.85 chọn sao?" | Grid search trên dev clip; 0.85 = sweet spot giữa flicker reduction và lag với head motion. |
| "Vì sao hair chỉ chuyển chroma?" | Preserve L = preserve head motion + silhouette → subject nhận ra mình. Style toàn bộ L sẽ phá self-recognition cue. |
| "2.7 fps có deploy production được không?" | Không — research prototype. Real-time / on-device deferred (đã ghi Limitations). |
| "So với DeepPrivacy?" | DeepPrivacy mất agency (subject không nhận ra mình). AvatarShield preserve self-recognition cue qua luminance. |

---

# Timing checklist

| Slide | Topic | Phút tích lũy |
|:---:|---|:---:|
| 5 | Intro pipeline (road-map + strip) | 0:50 |
| 6 | Why pure-classical | 1:35 |
| 7 | Pipeline overview | 2:20 |
| 8 | Cel-shade core | 3:05 |
| 9 | Theme palettes | 3:50 |
| 10 | Temporal stability | 4:30 |
| 11 | **Privacy results** | 5:20 |
| 12 | Joint trade-off | 6:00 |
| 13 | Limitations + handoff | 6:40 |

**Cushion:** ~20s nếu chậm hoặc gặp câu hỏi giữa chừng. Target tổng 6:50 — dưới hard limit 7 phút. Nếu cần cắt: gộp slide 7 (Pipeline Overview) vào slide 5 — sau khi expand strip thì slide 7 phần lớn là duplicate của slide 5.

---

# Demo plan (nếu có thời gian show video)

- File chính: `data/output/13-year-old-Yejin-solo-dance_porcelain-pink_cartoon.mp4`
- Mở **trước khi vào slide 11**, queue ở frame ~5s
- Sau khi nói xong slide 11, click "show video", chạy 8–10s, pause, quay lại slides
- Câu chốt: "Đây là raw output — chưa edit, chưa post-process, render từ pipeline trên slide 7."
