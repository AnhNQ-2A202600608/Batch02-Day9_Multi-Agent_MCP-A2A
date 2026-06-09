import sys
import os
from pathlib import Path

# Thêm thư mục gốc của dự án vào sys.path để chạy trực tiếp không bị lỗi import
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Đảm bảo stdout hỗ trợ UTF-8 trên Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"

_RETRIEVAL_WARNING_PRINTED = os.environ.get("RAG_SILENT") == "1" or "unittest" in sys.modules or "pytest" in sys.modules



def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    use_hyde: bool = False,
) -> list[dict]:
    global _RETRIEVAL_WARNING_PRINTED
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Lazy load các module để tối ưu hóa thời gian import ban đầu
    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search
    from src.task7_reranking import rerank, rerank_rrf, cosine_sim
    from src.task8_pageindex_vectorless import pageindex_search

    # Step 1: Chạy song song semantic + lexical
    # Truy vấn nhiều ứng viên hơn (top_k * 2) để chuẩn bị cho RRF & Reranking
    dense_results = semantic_search(query, top_k=top_k * 2, use_hyde=use_hyde)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # Step 2: Merge bằng RRF (Reciprocal Rank Fusion)
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Rerank
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        # Nếu không sử dụng Reranking, ta vẫn gán score bằng Cosine Similarity
        # để đảm bảo so khớp chính xác với ngưỡng score_threshold (0.3)
        try:
            from src.task5_semantic_search import get_semantic_model
            model = get_semantic_model()
            query_emb = model.encode(query).tolist()
            candidate_contents = [c["content"] for c in merged[:top_k]]
            candidate_embs = model.encode(candidate_contents)
            
            final_results = []
            for idx, item in enumerate(merged[:top_k]):
                sim = cosine_sim(query_emb, candidate_embs[idx].tolist())
                sim = max(0.0, min(1.0, sim))
                new_item = item.copy()
                new_item["score"] = sim
                final_results.append(new_item)
        except Exception:
            final_results = merged[:top_k]

    # Step 4: Check threshold → fallback sang PageIndex
    if not final_results or final_results[0]["score"] < score_threshold:
        best_score = final_results[0]["score"] if final_results else 0.0
        if not _RETRIEVAL_WARNING_PRINTED:
            print(f"  [WARNING] Hybrid score ({best_score:.3f}) < threshold ({score_threshold}). Fallback -> PageIndex (subsequent warnings silent)")
            _RETRIEVAL_WARNING_PRINTED = True
        
        fallback_results = pageindex_search(query, top_k=top_k)
        # Đảm bảo mỗi kết quả từ PageIndex được đánh dấu source hợp lệ
        for item in fallback_results:
            item["source"] = "pageindex"
        return fallback_results[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
