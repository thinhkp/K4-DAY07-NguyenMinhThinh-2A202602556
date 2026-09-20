# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Minh Thịnh
**Nhóm:** DeltaX
**Ngày:** 20/9

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
Độ tương tự cosine cao cho biết hai vector embedding có hướng gần nhau, nghĩa là hai đoạn văn có nội dung hoặc ý nghĩa gần nhau. Giá trị này phản ánh mức độ tương đồng về ngữ nghĩa, không chỉ dựa trên việc hai câu có dùng cùng từ hay không.

**Ví dụ có độ tương tự CAO:**
- Câu A: Người mua có thể yêu cầu hoàn tiền khi sản phẩm bị lỗi.
- Câu B: Khách hàng được phép đề nghị hoàn lại tiền nếu hàng hóa không hoạt động.
- Tại sao tương đồng: Hai câu dùng một số từ khác nhau nhưng cùng diễn đạt ý người mua được yêu cầu hoàn tiền khi sản phẩm bị lỗi.

**Ví dụ có độ tương tự THẤP:**
- Câu A: Người mua có thể yêu cầu hoàn tiền khi sản phẩm bị lỗi.
- Câu B: Người bán phải cung cấp bằng chứng trong vòng 24 giờ khi Shopee yêu cầu.
- Tại sao khác: Hai câu nói về hai đối tượng và hai hành động khác nhau: một câu nói về quyền hoàn tiền của người mua, còn câu kia nói về nghĩa vụ cung cấp bằng chứng của người bán.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
Cosine similarity đo góc giữa hai vector nên tập trung vào hướng biểu diễn, tức thông tin ngữ nghĩa, và ít bị ảnh hưởng bởi độ lớn hoặc độ dài vector. Điều này phù hợp với text embedding vì các câu có độ dài khác nhau vẫn có thể cùng chủ đề hoặc cùng ý nghĩa.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
>
> Mỗi chunk mới đóng góp thêm `chunk_size - overlap = 500 - 50 = 450` ký tự. Vì vậy:
>
> `ceil((10,000 - 50) / (500 - 50)) = ceil(9,950 / 450) = ceil(22.11...) = 23`
>
> *Đáp án:* **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
Khi `overlap = 100`, số chunk là `ceil((10,000 - 100) / (500 - 100)) = ceil(9,900 / 400) = ceil(24.75) = 25 chunks`, tăng từ 23 lên 25. Overlap lớn giúp nội dung ở ranh giới giữa hai chunk được lặp lại, làm giảm nguy cơ mất ngữ cảnh khi truy xuất; đổi lại, số chunk và chi phí lưu trữ/embedding cũng tăng.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
Hàm dùng regex `(?<=[.!?])(?:\s+|\n+)` để tách tại vị trí sau dấu chấm, dấu chấm than hoặc dấu hỏi, nhờ đó vẫn giữ dấu câu trong từng câu. Các câu sau khi được làm sạch khoảng trắng sẽ được gom thành từng chunk với tối đa `max_sentences_per_chunk` câu. Nếu văn bản rỗng hoặc chỉ có khoảng trắng, hàm trả về danh sách rỗng; cách tách đơn giản này vẫn có thể nhận diện chưa chính xác các viết tắt hoặc số thập phân.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
Thuật toán thử các separator theo thứ tự ưu tiên `\n\n`, `\n`, `. `, khoảng trắng và cuối cùng là chuỗi rỗng. Các phần nhỏ được gom liền kề nếu vẫn không vượt quá `chunk_size`; phần còn quá dài sẽ được đệ quy xử lý bằng separator tiếp theo. Base case là trả về nguyên văn bản nếu đã đủ ngắn, hoặc cắt cứng theo `chunk_size` khi không còn separator; văn bản rỗng trả về danh sách rỗng.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
`EmbeddingStore` lưu mỗi `Document` thành một record in-memory gồm `id`, `content`, bản sao `metadata` và embedding được tạo từ nội dung. `add_documents` chỉ lưu các document được truyền vào, không tự chia nhỏ tài liệu. Khi `search` chạy, query được embedding bằng cùng embedding function, sau đó tính dot product với các embedding đã lưu, sắp xếp score giảm dần và trả về tối đa `top_k` record; các embedding dài được loại khỏi kết quả.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
`search_with_filter` lọc record theo toàn bộ cặp khóa-giá trị trong `metadata_filter` trước khi tính similarity, ví dụ chỉ giữ các tài liệu có `audience="buyer"`; nếu không có filter thì dùng toàn bộ store. `delete_document` dựa vào `metadata["doc_id"]` để xóa tất cả các chunk thuộc cùng một file gốc và trả về `True` nếu có record bị xóa, ngược lại trả về `False`.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
`answer` trước tiên truy xuất tối đa `top_k` kết quả từ `EmbeddingStore`, sau đó ghép nội dung thành context với số thứ tự `[1]`, `[2]`, ... và nguồn lấy từ `source_url`, `source`, `doc_id` hoặc id của record. Prompt yêu cầu LLM chỉ dùng thông tin trong context, không suy đoán, nói rõ khi thiếu dữ liệu và trích dẫn số nguồn; nếu store không có kết quả, agent trả thông báo không tìm thấy thay vì gọi LLM.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
# Dán kết quả (output) của: pytest tests/ -v
pytest tests/ -v
===================================================================== test session starts =====================================================================
platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.6.0 -- D:\workspace\vinai\K4-L3B-Data-Foundations\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\workspace\vinai\K4-L3B-Data-Foundations
collected 42 items                                                                                                                                             

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                                                    [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                                             [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                                                      [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                                                       [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                                            [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                                            [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                                                  [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                                                   [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                                                 [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                                                   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                                                   [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                                              [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                                          [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                                                    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                                           [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                                               [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                                                         [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                                               [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                                                   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                                                     [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                                                       [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                                             [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                                                  [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                                                    [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                                                        [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                                                     [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                                              [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                                             [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                                                        [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                                                    [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                                               [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                                                   [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                                                         [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                                                   [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                                                [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                                              [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                                             [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                                                 [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                                            [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                                                     [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                                           [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                                               [100%]

===================================================================== 42 passed in 0.08s ======================================================================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Người mua có thể yêu cầu hoàn tiền khi hàng bị lỗi. | Khách hàng được phép đề nghị hoàn lại tiền nếu sản phẩm hư hỏng. | Cao | -0.1150 | Không kết luận bằng mock |
| 2 | Người mua phải gửi bằng chứng hình ảnh. | Người bán cần cung cấp chứng từ nhập khẩu. | Thấp | -0.1053 | Không kết luận bằng mock |
| 3 | Tiền hoàn được chuyển vào ví ShopeePay. | Khoản hoàn tiền được gửi vào tài khoản ShopeePay. | Cao | -0.1706 | Không kết luận bằng mock |
| 4 | Yêu cầu trả hàng được xử lý trong 3–5 ngày. | Người bán phải cung cấp thông tin sản phẩm chính xác. | Thấp | 0.0335 | Không kết luận bằng mock |
| 5 | Đơn hàng thông thường được yêu cầu trả hàng trong 15 ngày. | Người mua có 15 ngày để gửi yêu cầu hoàn tiền. | Cao | -0.0729 | Không kết luận bằng mock |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
Kết quả bất ngờ là các cặp câu có ý nghĩa gần nhau như cặp 1, 3 và 5 vẫn có điểm âm hoặc điểm rất thấp, trong khi cặp 4 không liên quan lại có điểm dương. Tuy nhiên, các điểm này được tính bằng `MockEmbedder`, vốn tạo vector từ MD5 nên không hiểu ngữ nghĩa; vì vậy chúng chỉ cho thấy pipeline tính similarity hoạt động, không thể dùng để đánh giá chất lượng biểu diễn ý nghĩa. Khi benchmark retrieval, cần dùng embedding thật như local multilingual, OpenAI hoặc Gemini.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Với thực phẩm tươi sống và đông lạnh, người mua phải gửi yêu cầu Trả hàng/Hoàn tiền trong thời hạn bao lâu? | `buyer-return-processing#2`: hướng dẫn theo dõi và xử lý yêu cầu trả hàng/hoàn tiền | 0.2105 | Không | Gold answer là trong vòng 24 giờ kể từ khi giao hàng thành công, nhưng evidence này không xuất hiện trong top-3. |
| 2 | Shopee có hỗ trợ yêu cầu đổi hàng không, và người mua có thể làm gì nếu hàng nhận được có vấn đề? | `buyer-return-processing#8`: kiểm tra kiện hàng hoàn trả và quay video làm bằng chứng | 0.2199 | Không | Gold answer là Shopee chưa hỗ trợ đổi hàng; người mua có thể từ chối nhận khi đồng kiểm hoặc gửi yêu cầu Trả hàng/Hoàn tiền. |
| 3 | Người mua có thể gửi yêu cầu Trả hàng/Hoàn tiền bằng những cách nào? | `buyer-return-conditions#2`: các lý do và điều kiện trả hàng/hoàn tiền | 0.2526 | Không | Gold answer gồm gửi trực tiếp tại trang đơn hàng hoặc qua mục Trò Chuyện Với Shopee; evidence không nằm trong top-3. |
| 4 | Sau khi Shopee chấp nhận hoàn tiền, tiền hoàn về thẻ tín dụng hoặc thẻ ghi nợ mất bao lâu? | `buyer-refund-timeline#0`: bảng phương thức thanh toán và thời gian hoàn tiền | 0.1927 | Có | Khoảng 7–14 ngày làm việc, tùy theo ngân hàng. Evidence xuất hiện ở top-1. |
| 5 | Một yêu cầu hoàn tiền cần được phản hồi trong bao lâu? | `seller-mall-return-obligations#3`: chi phí vận chuyển và khiếu nại sản phẩm hoàn trả | 0.1336 | Không | Gold answer là người bán có 02 ngày lịch để phản hồi; không phản hồi được xem là đồng ý. Tài liệu đúng ở top-1 nhưng chunk không chứa evidence cần tìm. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 1 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
Qua benchmark, tôi nhận ra lấy đúng tài liệu chưa chắc đã lấy đúng section chứa câu trả lời. Chunking theo kích thước và embedding mock có thể đưa các đoạn cùng chủ đề nhưng thiếu con số hoặc điều kiện cần thiết lên top-k. Khi so sánh với các chiến lược khác, cần ưu tiên evidence chứa đáp án thay vì chỉ kiểm tra `doc_id`.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5/ 5 |
| Hướng tiếp cận của tôi (My Approach) | 10/ 10 |
| Hoàn thiện code (Core Implementation — tests) | 30/ 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5/ 5 |
| Kết quả truy xuất của tôi (Competition Results) | 8/ 10 |
| **Tổng phần cá nhân** | **58
/ 60** |
