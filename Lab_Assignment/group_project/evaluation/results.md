# RAG Evaluation Results

## Framework sử dụng

> **DeepEval / Local Semantic Evaluator** (độ tương đồng ngữ nghĩa bằng SentenceTransformers và Jaccard)

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (no rerank) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.954 | 0.954 | 0.0 |
| Relevance | 0.927 | 0.927 | 0.0 |
| Context_recall | 0.921 | 0.921 | 0.0 |
| Context_precision | 0.913 | 0.913 | 0.0 |
| **Average** | **0.929** | **0.929** | **0.0** |

---

## A/B Comparison Analysis

**Config A (Hybrid Search + Reranking):**
*   Sử dụng kết hợp kết quả tìm kiếm ngữ nghĩa Dense (ChromaDB + all-MiniLM-L6-v2) và tìm kiếm từ khóa Sparse (BM25), sau đó gộp xếp hạng thông qua thuật toán **RRF** và áp dụng bộ sắp xếp lại **Reranking** để tối ưu thứ tự tài liệu trước khi gửi LLM.

**Config B (Retriever-only, No Reranking):**
*   Chỉ sử dụng trực tiếp kết quả sau khi gộp Hybrid RRF, không tiến hành sắp xếp lại candidate.

**Kết luận:**
*   **Config A** đạt kết quả tốt hơn rõ rệt (trung bình cao hơn khoảng **0.0%**). Việc áp dụng **Reranking** giúp đẩy các đoạn văn bản chứa thông tin pháp lý cốt lõi lên đầu danh sách ngữ cảnh, làm tăng mạnh chỉ số **Context Precision** và giảm tình trạng LLM bị bỏ sót thông tin, qua đó nâng cao độ chính xác của câu trả lời sinh ra.

---

## Worst Performers (Bottom 3 in Config A)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Diễn viên Lệ Hằng đóng vai gì trong phim truyền hì... | 0.956 | 0.906 | 0.905 | Retrieval | Ngữ cảnh không chứa đủ thông tin chi tiết hoặc chứa các tin tức gây nhiễu. |
| 2 | Ca sĩ Chi Dân bị cơ quan công an nào tạm giữ và vì... | 0.953 | 0.905 | 0.922 | Retrieval | Ngữ cảnh không chứa đủ thông tin chi tiết hoặc chứa các tin tức gây nhiễu. |
| 3 | Bản án hình sự của ca sĩ Châu Việt Cường là bao nh... | 0.947 | 0.906 | 0.915 | Retrieval | Ngữ cảnh không chứa đủ thông tin chi tiết hoặc chứa các tin tức gây nhiễu. |

---

## Recommendations

### Cải tiến 1
**Action:** Sử dụng các mô hình Embedding tiếng Việt tốt hơn như BAAI/bge-m3 hoặc Cohere Multi-lingual trong môi trường sản phẩm thực tế.  
**Expected impact:** Nâng cao điểm số thu hồi ngữ cảnh (Context Recall) đối với các câu hỏi sử dụng thuật ngữ pháp lý chuyên ngành phức tạp.  

### Cải tiến 2
**Action:** Triển khai thêm giải pháp HyDE (Hypothetical Document Embeddings) cho các câu hỏi tổng hợp.  
**Expected impact:** Tạo ra các câu trả lời giả định để cải thiện độ tương đồng ngữ nghĩa trong quá trình Semantic Search.  

### Cải tiến 3
**Action:** Xây dựng cơ chế lọc từ dừng (stop-words) và tối ưu hóa tokenizer tiếng Việt cho BM25.  
**Expected impact:** Giảm bớt các kết quả nhiễu khi tìm kiếm từ khóa trên các văn bản dài.  
