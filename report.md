# Báo cáo Day 17 – Track 3: Memory Systems for AI Agent

**Sinh viên:** Nguyễn Đức Thành – 2A202600838

---

## 1. Mục tiêu

Xây dựng và so sánh hai AI agent để hiểu rõ trade-off giữa **độ nhớ dài hạn**, **chất lượng phản hồi**, và **chi phí token**:

- **Baseline Agent** – chỉ nhớ trong cùng thread, quên hoàn toàn khi mở session mới.
- **Advanced Agent** – có ba lớp memory: short-term (thread), persistent (`User.md`), và compact memory.

---

## 2. Kiến trúc hệ thống

```
src/
├── config.py          # Cấu hình chung: paths, compact threshold, provider
├── model_provider.py  # Khởi tạo LLM cho 6 provider (openai, gemini, anthropic, ollama, openrouter, custom)
├── memory_store.py    # estimate_tokens, UserProfileStore, CompactMemoryManager
├── agent_baseline.py  # Agent A – within-session memory only
├── agent_advanced.py  # Agent B – User.md + compact memory
├── benchmark.py       # Standard + Stress benchmark, bảng so sánh, phân tích
└── test_agents.py     # 4 test: CRUD User.md, compact trigger, cross-session recall, prompt load
```

### Ba lớp memory của Advanced Agent

| Lớp | Cơ chế | Phạm vi |
|-----|--------|---------|
| Short-term | Danh sách messages trong thread | Trong một thread |
| Persistent | File `User.md` per user, upsert theo key | Vĩnh viễn, qua mọi session |
| Compact | Khi tổng token > ngưỡng → nén messages cũ thành summary (tối đa 600 chars), giữ lại N messages gần nhất | Trong một thread dài |

---

## 3. Kết quả benchmark

### 3.1 Standard Benchmark – `data/conversations.json` (10 conversations)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|:-----------------:|:-----------------------:|:--------------------:|:----------------:|:---------------------:|:-----------:|
| Baseline | 3 412             | 24 222                  | 0.000                | 0.400            | 0                     | 0           |
| Advanced | 6 150             | 43 675                  | **0.250**            | **0.617**        | 3 132                 | 0           |

### 3.2 Long-Context Stress Benchmark – `data/advanced_long_context.json` (1 conversation, 15 turns dài)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|:-----------------:|:-----------------------:|:--------------------:|:----------------:|:---------------------:|:-----------:|
| Baseline | 466               | 24 118                  | 0.000                | 0.400            | 0                     | 0           |
| Advanced | 633               | **19 271**              | **0.500**            | **0.708**        | 252                   | 1           |

---

## 4. Phân tích kết quả

### 4.1 Tại sao Advanced có recall tốt hơn Baseline?

Baseline chỉ giữ lịch sử trong thread hiện tại. Khi recall question được hỏi ở thread mới (cross-session), Baseline không có bất kỳ thông tin nào về người dùng → recall = 0.

Advanced lưu facts ổn định (tên, nghề nghiệp, nơi ở, phong cách trả lời, đồ uống yêu thích...) vào `User.md` ngay khi người dùng chia sẻ. Khi mở thread mới, Advanced đọc `User.md` và có thể trả lời đúng các câu recall → recall = 0.25 (standard) và 0.5 (stress).

### 4.2 Tại sao Advanced tốn hơn ở hội thoại ngắn?

Mỗi lượt của Advanced phải kéo theo:
- Toàn bộ nội dung `User.md` (~100–300 tokens)
- Compact summary (nếu có)
- N messages gần nhất

Ở hội thoại ngắn (ít turns, chưa đủ để compact), chi phí đọc profile cố định này vượt qua lợi ích nén context. Kết quả: Advanced prompt tokens (43 675) > Baseline (24 222) trong standard benchmark.

### 4.3 Tại sao compact memory giúp Advanced ở hội thoại dài?

Baseline giữ toàn bộ lịch sử → prompt tokens tăng **tuyến tính** O(n) theo số lượt.

Advanced khi vượt ngưỡng (mặc định 2 000 tokens) sẽ nén messages cũ thành summary, chỉ giữ lại 6 messages gần nhất. Summary được giới hạn tối đa 600 characters để tránh tích lũy vô hạn. Kết quả ở stress benchmark:

- Baseline: 24 118 prompt tokens (tăng đều theo chiều dài thread)
- Advanced: 19 271 prompt tokens (compact sau 1 lần → tiết kiệm ~20%)

### 4.4 Memory file tăng trưởng và rủi ro

`User.md` tăng ~300 bytes mỗi conversation (3 132 bytes sau 10 conversations). Nếu người dùng có hàng nghìn sessions, file có thể phình to đáng kể và làm tăng prompt cost trở lại. Các giải pháp đề xuất (bonus):

- **Confidence threshold** – chỉ ghi vào `User.md` khi chắc chắn là fact (không phải câu hỏi hay giả thiết)
- **Conflict handling** – khi người dùng đính chính, ghi đè thông tin cũ bằng `upsert_fact()` thay vì thêm mới
- **Memory decay** – giảm ưu tiên facts không được nhắc lại theo thời gian

---

## 5. Bonus: Confidence Threshold + Conflict Handling

### 5.1 Vấn đề cần giải quyết

Hệ thống memory ngây thơ sẽ ghi vào `User.md` mọi thứ trích được từ tin nhắn người dùng. Điều này dẫn đến hai lỗi phổ biến:

1. **Lưu fact sai**: câu hỏi ("Bạn có biết Python không?") hay câu đùa ("Hay là chuyển sang PM cho đỡ?") bị hiểu thành fact thật.
2. **Giữ thông tin cũ sai**: khi người dùng đính chính ("Không còn ở Huế nữa, giờ ở Đà Nẵng"), fact cũ vẫn tồn tại song song với fact mới.

### 5.2 Thiết kế Confidence Threshold

Hàm `extract_profile_updates_scored()` (trong [memory_store.py](src/memory_store.py)) gán mỗi fact một điểm tin cậy:

| Loại tình huống | Score | Ví dụ |
|-----------------|-------|-------|
| Câu đính chính rõ ràng | **0.95** | "đính chính", "không còn X nữa", "giờ là Y" |
| Khai báo trực tiếp | **0.80** | "tên là DũngCT", "làm MLOps engineer" |
| Từ khoá xuất hiện gián tiếp | **0.60** | "Python" đề cập trong câu thảo luận |

Hàm `extract_profile_updates(message, confidence_threshold=0.70)` chỉ trả về facts có score ≥ ngưỡng. Mặc định **0.70** lọc sạch các từ khoá gián tiếp (0.60) mà giữ lại mọi khai báo tường minh và đính chính.

**Tác động lên recall và token cost:**
- Recall tăng: `User.md` không bị nhiễm bởi facts sai từ câu hỏi hay đùa.
- Token cost giảm nhẹ: file `User.md` nhỏ hơn → prompt mỗi lượt nhỏ hơn.

**Rủi ro mới:**
- Ngưỡng quá cao (ví dụ 0.90) có thể lọc mất cả khai báo trực tiếp (score 0.80), gây recall giảm.
- Cần điều chỉnh ngưỡng theo domain; giá trị 0.70 là compromise cho tiếng Việt thông thường.

### 5.3 Thiết kế Conflict Handling

Khi `is_correction=True`, `upsert_fact()` ghi đè fact cũ **và** thêm annotation `_(đã đính chính)_` vào dòng trong `User.md`:

```
- **location**: Đà Nẵng  _(đã đính chính)_
```

Điều này đảm bảo:
- **Không giữ hai fact mâu thuẫn cùng lúc** (lỗi phổ biến ở agent đơn giản chỉ append).
- Lịch sử đính chính có thể audit được mà không cần event log riêng.

**Tác động lên recall và token cost:**
- Recall tăng đáng kể ở benchmark có correction (conv-03, conv-06): agent trả lời đúng thông tin mới nhất thay vì thông tin cũ.
- Token cost không đổi (annotation ngắn).

**Rủi ro mới:**
- Nếu agent nhầm nhận câu ví dụ là câu đính chính ("Giả sử mình ở Hà Nội..."), fact đúng bị ghi đè. Cần kết hợp với guardrail kiểm tra ngữ cảnh ví dụ vs. tuyên bố thật.

---

## 6. Kết quả test

```
PASSED test_user_markdown_read_write_edit                          – User.md CRUD hoạt động đúng
PASSED test_compact_trigger                                        – Compact kích hoạt khi tổng tokens > ngưỡng (7 compactions)
PASSED test_cross_session_recall                                   – Advanced nhớ qua session, Baseline không
PASSED test_compact_reduces_prompt_load_on_long_thread             – Advanced (3 257t) < Baseline (5 348t) sau compaction
PASSED test_confidence_threshold_filters_low_quality_facts  [BONUS] – Score 0.60 bị lọc, score 0.80 qua ngưỡng
PASSED test_conflict_handling_correction_overrides          [BONUS] – Fact cũ bị ghi đè, annotation đính chính xuất hiện

6 passed in 0.26s
```

---

## 7. Kết luận

| Tiêu chí | Baseline | Advanced |
|----------|----------|----------|
| Cross-session recall | Không | Có |
| Chi phí ở hội thoại ngắn | Thấp hơn | Cao hơn (do profile overhead) |
| Chi phí ở hội thoại dài | Tăng tuyến tính | Tăng chậm hơn (compact) |
| Độ phức tạp hệ thống | Thấp | Cao (cần guardrail cho User.md) |

**Bài học cốt lõi:** Memory system mạnh hơn không đồng nghĩa rẻ hơn ở mọi trường hợp. Advanced Agent thực sự có lợi thế rõ rệt khi hội thoại đủ dài để compact memory phát huy và khi recall cross-session là yêu cầu bắt buộc.
