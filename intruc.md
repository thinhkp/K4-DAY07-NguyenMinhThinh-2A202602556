# Tóm tắt công việc Lab 07 — Embedding & Vector Store

## 1. Mục tiêu và phạm vi

Hoàn thiện một hệ thống RAG nhỏ gồm:

- Ba chiến lược chunking và công cụ so sánh chúng.
- Vector store in-memory có embedding, tìm kiếm, lọc metadata và xóa tài liệu.
- RAG agent truy xuất ngữ cảnh rồi gọi LLM để trả lời có trích dẫn nguồn.
- Corpus chính sách thương mại điện tử có provenance đầy đủ.
- Bộ benchmark gồm đúng 5 câu hỏi để so sánh chiến lược retrieval.

Repo này là biến thể **K4-L3B**. Chủ đề dữ liệu bắt buộc là chính sách **đổi trả, bảo hành hoặc quy định người bán/người mua** trên nền tảng thương mại điện tử. Dữ liệu phải là nguồn công khai hoặc được phép chia sẻ.

Phần cá nhân: **60 điểm**. Phần nhóm: **40 điểm**. Bộ test dùng embedding giả lập nên không cần API key để hoàn thành phần code; Internet cần cho việc thu thập dữ liệu.

## 2. Kết quả cần nộp

| Hạng mục | Người làm | Yêu cầu |
|---|---|---|
| `src/` | Mỗi người | Hoàn thiện TODO; `pytest tests/ -v` phải có **42 passed** |
| `data/<chu-de>/` | Nhóm | 5–10 file `.md` đã làm sạch và `sources.csv` khớp 1-1 |
| `bench.py` | Mỗi người | Chạy 5 query, in top-3 cùng score và `doc_id` |
| `ket_qua_benchmark.txt` | Mỗi người | Lưu output benchmark của chiến lược cá nhân |
| `report/REPORT_CANHAN.md` | Mỗi người | Điền phần cá nhân và kết quả riêng |
| `report/REPORT_NHOM.md` | Nhóm | Điền corpus, chiến lược, 5 query, gold answer và so sánh |
| Repo GitHub | Mỗi người | Đúng tên `K4-DAY07-HoVaTen-MSSV`, push và nộp link trên vlearn |

## 3. Lộ trình 4 giờ và checkpoint

| Thời gian | Giai đoạn | Việc chính | Checkpoint |
|---|---|---|---|
| 0:00–1:00 | Dữ liệu | Setup, chọn chủ đề, crawl và làm sạch corpus | CP1 0:20, CP2 1:00 |
| 1:00–2:30 | Code cá nhân | Warm-up, hoàn thiện `src/` | CP3 1:45, CP4 2:30 |
| 2:30–3:00 | Chiến lược | Viết 5 query, tạo benchmark, chọn chiến lược riêng | CP5 3:00 |
| 3:00–3:25 | So sánh | Chạy benchmark, đánh giá top-3 và phân tích failure case | CP6 3:25 |
| 3:25–4:00 | Demo và nộp | Hoàn thiện báo cáo, demo, commit, push | CP7 4:00 |

## 4. Giai đoạn 1 — Setup và baseline

### Chuẩn bị môi trường

1. Fork đúng repo K4-L3B, không clone trực tiếp repo gốc.
2. Clone fork, mở repo trong VS Code.
3. Dùng Python 3.11 (3.10+ vẫn chạy test).
4. Trên Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Nếu PowerShell chặn script:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

`requirements.txt` chỉ cần `pytest` và `python-dotenv`. Không cần cài ChromaDB, sentence-transformers hay LLM SDK để pass test.

### CP1 — 0:20

Chạy:

```powershell
pytest tests/ -v
```

Baseline đúng là **31 failed, 11 passed**. Các lỗi ban đầu chủ yếu là `NotImplementedError`; 11 test pass kiểm tra cấu trúc và `FixedSizeChunker` có sẵn. Nếu gặp `ModuleNotFoundError`, kiểm tra venv và `pip install`.

## 5. Giai đoạn 1 — Thu thập dữ liệu L3B

Đọc [`docs/DATA_COLLECTION.md`](docs/DATA_COLLECTION.md) trước khi crawl. Có thể dùng crawler:

```powershell
Copy-Item scripts\urls.example.csv data\urls.csv
# Điền 5–10 URL vào data\urls.csv
python scripts\fetch_public_pages.py data\urls.csv --output-dir data\<ten-chu-de>
```

### Ràng buộc corpus

- Có 5–10 tài liệu Markdown.
- Mỗi tài liệu có metadata:
  - `audience`: `buyer`, `seller` hoặc `both`.
  - `source_url`.
  - `retrieved_at`.
  - `document_version` hoặc `not-stated` nếu nguồn không nêu phiên bản.
  - Ít nhất một trường hữu ích khác, ví dụ `category`, `language`, `platform`.
- `doc_id` trong frontmatter phải khớp tên file.
- `sources.csv` phải khớp 1-1 với các file Markdown.
- Có ít nhất **hai giá trị `audience` khác nhau**, nếu không metadata filter không có ý nghĩa.
- Trong 5 query phải có ít nhất một query cần:
  - `metadata_filter={"audience": "buyer"}`, hoặc
  - `metadata_filter={"audience": "seller"}`.
- Gold answer phải trích được từ tài liệu thật, không suy đoán chính sách.

Crawler đã kiểm tra `robots.txt`, giới hạn tốc độ request và chỉ nhận HTML/text. Không vượt qua `robots.txt`; hãy đổi nguồn nếu bị chặn. Sau khi crawl phải:

- Xóa menu, điều hướng, tin tức và nội dung thừa.
- Giữ lại điều khoản, con số, điều kiện và mốc thời gian.
- Kiểm tra thủ công ngôn ngữ và nội dung sau khi tải.
- Nếu một trang gộp chính sách buyer và seller nhưng cần filter theo audience, tách thành các file riêng.

### CP2 — 1:00

Kiểm tra 5–10 file, metadata đầy đủ, `sources.csv` khớp 1-1 và audience có ít nhất hai giá trị. Điền **Data Inventory** và **Metadata Schema** vào `REPORT_NHOM.md`.

## 6. Giai đoạn 2 — Warm-up và hoàn thiện `src/`

Không đổi chữ ký các hàm/class; chỉ thay TODO và `raise NotImplementedError`.

### Warm-up — điền `REPORT_CANHAN.md` mục 1

- Giải thích cosine similarity cao nghĩa là gì.
- Cho một cặp câu khác từ nhưng cùng nghĩa (similarity cao) và một cặp ít liên quan (similarity thấp).
- Giải thích vì sao cosine phù hợp với text embedding hơn Euclidean distance.
- Với `length=10000`, `chunk_size=500`, `overlap=50`, dùng:

```text
ceil((độ_dài - overlap) / (chunk_size - overlap))
```

- Kiểm tra lại bằng `FixedSizeChunker`.
- Giải thích khi overlap tăng lên 100 thì số chunk thay đổi thế nào và vì sao overlap lớn có thể hữu ích.

### `src/chunking.py`

#### `SentenceChunker.chunk`

- Tách sau dấu `.`, `!`, `?` và newline nhưng phải giữ dấu câu.
- Gom tối đa `max_sentences_per_chunk` câu/chunk.
- Strip khoảng trắng.
- Text rỗng trả `[]`.
- Ghi nhận giới hạn: viết tắt như `TS.`, `v.v.` và số thập phân có thể bị tách sai.

#### `RecursiveChunker.chunk` và `_split`

- Ưu tiên separator: `["\n\n", "\n", ". ", " ", ""]`.
- Đệ quy xuống separator nhỏ hơn khi mảnh còn quá dài.
- Gom các mảnh liền kề cho tới gần `chunk_size`, tránh chunk vụn.
- Xử lý base case `separators=[]` bằng cách cắt cứng theo `chunk_size`.

#### `compute_similarity`

- Dùng công thức cosine trong docstring và helper `_dot`.
- Vector rỗng hoặc độ dài 0 phải trả `0.0`, không được `ZeroDivisionError`.

#### `ChunkingStrategyComparator.compare`

Gọi cả ba chunker trên cùng text và trả đúng cấu trúc:

```python
{
    "fixed_size": {"count": ..., "avg_length": ..., "chunks": ...},
    "by_sentences": {"count": ..., "avg_length": ..., "chunks": ...},
    "recursive": {"count": ..., "avg_length": ..., "chunks": ...},
}
```

Text rỗng phải chặn chia cho 0.

### CP3 — 1:45

```powershell
pytest tests/ -k "Chunker or Similarity or Compare" -v
```

Kỳ vọng **23 passed**.

### `src/store.py` — `EmbeddingStore`

Chỉ dùng in-memory; đặt `_use_chroma = False`. Không triển khai nhánh ChromaDB.

1. `_make_record`:
   - Copy metadata, không giữ trực tiếp object metadata của caller.
   - Luôn có `metadata["doc_id"]`.
   - `doc_id` là file gốc, không phải id chunk kiểu `file#0`.
2. `_search_records`:
   - Tìm similarity trên tập record được truyền vào.
   - Dùng chung cho `search` và `search_with_filter`.
   - Không đưa vector `embedding` vào kết quả trả về.
3. `add_documents`:
   - Không tự chunk; một `Document` tương ứng một record.
4. `search`:
   - Tìm top-k trên toàn bộ record.
5. `search_with_filter`:
   - Lọc candidate **trước**, sau đó mới search.
   - Metadata của mọi chunk phải có đủ field để lọc.
6. `delete_document`:
   - Xóa mọi chunk có `metadata["doc_id"]` khớp.
   - Trả `True` nếu có xóa, `False` nếu không tìm thấy.

### `src/agent.py` — `KnowledgeBaseAgent.answer`

Luồng xử lý:

1. Truy xuất top-k, có thể truyền `metadata_filter`.
2. Dựng prompt với các chunk được đánh số `[1]`, `[2]`, `[3]` và kèm nguồn.
3. Yêu cầu LLM chỉ dùng context, trả lời “không tìm thấy” nếu context không đủ.
4. Yêu cầu trích dẫn số chunk để bảo đảm source traceability.
5. Store rỗng phải trả thông báo rõ ràng và không gọi LLM vô ích.

### CP4 — 2:30

```powershell
pytest tests/ -v
python main.py "Chunking là gì?"
```

Phải đạt **42 passed** và `main.py` chạy từ đầu đến cuối. Có thể thấy cảnh báo thiếu file mẫu; đó không phải lỗi. Dán output test thật vào `REPORT_CANHAN.md` mục 3.

## 7. Giai đoạn 3 — Chiến lược và benchmark

### Phân vai nhóm

- **R1 Data:** chốt chủ đề, chia URL, kiểm metadata, duy trì `sources.csv`.
- **R2 Benchmark:** viết đúng 5 query và gold answer, kiểm chứng answer từ corpus.
- **R3 Strategy:** bảo đảm chiến lược không trùng, thử chunk theo heading, chạy baseline.
- Nhóm 4 người có thể thêm **Report & Demo Lead**.

Mỗi thành viên vẫn tự code và chạy benchmark riêng. Trên cùng corpus, mỗi người dùng một chiến lược khác nhau; tối thiểu một người dùng chunk theo heading/section. Gợi ý: Fixed Size có overlap, Sentence, Recursive hoặc custom heading.

### 5 benchmark query

Đúng 5 câu, đa dạng: hỏi số liệu, điều kiện, quy trình, liệt kê. Ít nhất một câu phải cần metadata filter `buyer` hoặc `seller`; nên thiết kế corpus có tài liệu cùng chủ đề nhưng khác audience để chứng minh filter có tác dụng.

### Chunk theo heading

- Tách trước mỗi heading như `## Điều ...`.
- Mỗi section là một chunk nếu đủ ngắn.
- Section quá dài thì hạ xuống recursive.
- Khi chia section dài, gắn lại heading vào mọi mảnh con để không mất ngữ cảnh.

### `bench.py`

Phải:

1. Đọc file `.md`, tách frontmatter và body.
2. Chunk body ở ngoài store.
3. Tạo `Document` cho từng chunk:

```python
Document(
    id=f"{path.stem}#{i}",
    content=chunk,
    metadata={**frontmatter, "doc_id": path.stem},
)
```

4. Nạp vào `EmbeddingStore`.
5. Chạy 5 query qua `search_with_filter()`.
6. In top-3, score và `doc_id`.

Không nạp nguyên file như một document. Frontmatter phải được trải vào mọi chunk. Mỗi người chỉ đổi dòng chọn chunker để so sánh công bằng. Nếu dùng OpenAI embedding, nên cache theo hash nội dung.

### CP5 — 3:00

`python bench.py` phải in số chunk đã nạp và top-3 cho cả 5 query. Điền 5 query/gold answer vào `REPORT_NHOM.md` và bảo đảm mỗi thành viên dùng chiến lược riêng.

## 8. Giai đoạn 4 — Đánh giá và phân tích

`MockEmbedder` dùng MD5 nên không hiểu ngữ nghĩa. Nếu có điều kiện, dùng embedding thật:

- Local: `pip install -r requirements-local.txt`, `.env` đặt `EMBEDDING_PROVIDER=local`.
- OpenAI: cài `openai`, đặt provider, API key và model.
- Gemini: cài `google-genai`, đặt provider, API key và model.

Không để API key trong Git; `.env` đã nằm trong `.gitignore`. Kiểm tra dòng `Embedding backend` mà `main.py` in ra. Nếu dùng mock, phải ghi rõ trong báo cáo và tập trung phân tích thêm `count`, `avg_length`, độ mạch lạc.

Đánh giá ở hai mức:

- **Retrieval:** top-3 có chunk đúng section và có thông tin trả lời không.
- **Answer:** agent có trả lời đúng, đủ và trích nguồn không.

Không chỉ kiểm tra `doc_id`: nhiều chunk cùng file có thể làm kết quả nhìn có vẻ đúng nhưng chunk thực tế không chứa đáp án. Phân tích các failure case như chunk đúng chủ đề nhưng thiếu số liệu, đúng file nhưng sai section, hoặc filter loại nhầm tài liệu.

### CP6 — 3:25

- Mỗi người có `ket_qua_benchmark.txt`.
- Điền bảng top-3 vào `REPORT_CANHAN.md` mục 5.
- Nhóm có bảng so sánh và ít nhất một failure case trong `REPORT_NHOM.md` mục 2 và 4.

## 9. Demo và hoàn tất báo cáo

Demo 6–8 phút:

1. Chủ đề và corpus: 1 phút.
2. Mỗi thành viên trình bày chiến lược: 2 phút.
3. So sánh và giải thích chiến lược thắng: 3 phút.
4. Demo 1–2 query trực tiếp: 2 phút.

Chuẩn bị sẵn terminal với `bench.py`; không debug lần đầu trong demo. Có thể chuẩn bị câu trả lời cho:

- Đổi domain thì chiến lược nào còn phù hợp?
- Metadata filter giúp ở đâu và có thể làm mất recall ở đâu?
- Nhóm học được gì từ chiến lược của thành viên khác?

`REPORT_CANHAN.md` cần có:

- Warm-up.
- Cách triển khai chunker, store và agent.
- Output `pytest` thật.
- 5 dự đoán similarity và nhận xét.
- Kết quả 5 query của cá nhân.

`REPORT_NHOM.md` cần có:

- Chủ đề, lý do chọn, data inventory và metadata schema.
- Baseline `ChunkingStrategyComparator` trên 2–3 tài liệu.
- Chiến lược của từng thành viên và so sánh.
- Đúng 5 query, gold answer và chunk chứa đáp án.
- Chất lượng retrieval, tác dụng của metadata filter, failure case.
- Insight demo và bài học nhóm.

## 10. Checklist CP7 trước khi nộp

- [ ] `pytest tests/ -v` → **42 passed**.
- [ ] Không còn `raise NotImplementedError` trong phần cần hoàn thiện.
- [ ] Có 5–10 tài liệu trong `data/<chu-de>/`.
- [ ] Mọi file có metadata L3B: `audience`, `source_url`, `retrieved_at`, `document_version` và field bổ sung.
- [ ] `sources.csv` khớp 1-1 với file Markdown.
- [ ] Corpus có ít nhất hai giá trị audience.
- [ ] Có ít nhất một query cần filter `buyer` hoặc `seller`.
- [ ] Có ít nhất một thành viên chunk theo heading/section.
- [ ] `bench.py` chạy được, `ket_qua_benchmark.txt` đã lưu.
- [ ] Hai báo cáo đã điền đầy đủ; output pytest là thật.
- [ ] Không commit `.venv/`, `.env` hoặc API key.
- [ ] Repo đúng tên `K4-DAY07-HoVaTen-MSSV`.

## 11. Lệnh commit và nộp

```powershell
pytest tests/ -v
git status
git add .
git commit -m "Nop bai Lab 07"
git branch -M main
git remote add origin https://github.com/<tai-khoan>/K4-DAY07-<HoVaTen>-<MSSV>.git
git push -u origin main
```

Nộp **link repo GitHub trên vlearn**, không nộp file ZIP.

## 12. Lỗi thường gặp

| Triệu chứng | Nguyên nhân/cách xử lý |
|---|---|
| `ModuleNotFoundError: src` | Chạy lệnh từ thư mục gốc repo và kích hoạt venv |
| Store test sập khi có ChromaDB | Luôn dùng in-memory, `_use_chroma = False` |
| `search_with_filter` trả rỗng | Lọc trước khi search và trải metadata vào từng chunk |
| `delete_document` luôn `False` | Record thiếu `metadata["doc_id"]` của file gốc |
| Recursive tạo nhiều chunk vụn | Thiếu bước gom các mảnh liền kề |
| Recursive fail với `separators=[]` | Thêm base case cắt cứng |
| `ZeroDivisionError` trong comparator | Xử lý text rỗng/count bằng 0 |
| Chunk sentence mất dấu câu | Dùng regex tách sau dấu câu, không nuốt delimiter |
| Crawler bị `robots.txt` chặn | Đổi URL, không vượt robots |
| Crawler `LookupError: unknown encoding` | Bỏ URL charset lỗi khỏi CSV và xử lý riêng |
| Score mock âm hoặc retrieval sai ngữ nghĩa | Ghi rõ giới hạn mock hoặc bật embedding thật |
| Filter không tạo khác biệt | Corpus chỉ có một audience hoặc hai đáp án nằm chung file; hãy tách tài liệu |

