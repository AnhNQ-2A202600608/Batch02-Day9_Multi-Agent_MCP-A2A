"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import sys
import os
import math
from pathlib import Path
import requests
from dotenv import load_dotenv

# Thêm thư mục gốc của dự án vào sys.path để chạy trực tiếp không bị lỗi import
ROOT_DIR = Path(__file__).resolve().parents[1] if "__file__" in locals() else Path(".").resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Đảm bảo stdout hỗ trợ UTF-8 trên Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")

_JINA_CALL_PRINTED = os.environ.get("RAG_SILENT") == "1" or "unittest" in sys.modules or "pytest" in sys.modules
_JINA_WARNING_PRINTED = os.environ.get("RAG_SILENT") == "1" or "unittest" in sys.modules or "pytest" in sys.modules



def cosine_sim(v1: list[float], v2: list[float]) -> float:
    """Tính cosine similarity giữa 2 vector."""
    if not v1 or not v2:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def local_jaccard_similarity(str1: str, str2: str) -> float:
    """Tính độ tương đồng từ vựng Jaccard làm fallback cho cross-encoder."""
    words1 = set(str1.lower().split())
    words2 = set(str2.lower().split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    global _JINA_CALL_PRINTED, _JINA_WARNING_PRINTED
    """
    Rerank candidates sử dụng cross-encoder model.
    Hỗ trợ Jina Reranker API và fallback local Jaccard similarity nếu không có API key.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by score descending.
    """
    if not candidates:
        return []

    # Option A: Jina Reranker API nếu có API key
    if JINA_API_KEY:
        try:
            if not _JINA_CALL_PRINTED:
                print("Calling Jina Reranker API (subsequent calls silent)...")
                _JINA_CALL_PRINTED = True
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {JINA_API_KEY}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": top_k
                },
                timeout=5
            )
            if response.status_code == 200:
                reranked = response.json()["results"]
                results = []
                for r in reranked:
                    idx = r["index"]
                    item = candidates[idx].copy()
                    item["score"] = float(r["relevance_score"])
                    results.append(item)
                return results
            else:
                if response.status_code in (402, 403):
                    if not _JINA_WARNING_PRINTED:
                        print("Jina API quota exceeded (Hết hạn mức token). Tự động dùng local semantic rerank (subsequent warnings silent).")
                        _JINA_WARNING_PRINTED = True
                else:
                    if not _JINA_WARNING_PRINTED:
                        print(f"Jina API returned status code {response.status_code}, falling back to local rerank (subsequent warnings silent).")
                        _JINA_WARNING_PRINTED = True
        except Exception as e:
            if not _JINA_WARNING_PRINTED:
                print(f"Jina API failed: {e}. Falling back to local rerank (subsequent warnings silent).")
                _JINA_WARNING_PRINTED = True

    # Option B: Fallback local (Sử dụng SentenceTransformer và cosine similarity để tăng độ chính xác)
    if not JINA_API_KEY:
        if not _JINA_WARNING_PRINTED:
            print("Using local semantic similarity for reranking (subsequent calls silent)...")
            _JINA_WARNING_PRINTED = True
    try:
        from src.task5_semantic_search import get_semantic_model
        model = get_semantic_model()
        
        # Encode query và candidates
        query_emb = model.encode(query).tolist()
        candidate_contents = [c["content"] for c in candidates]
        candidate_embs = model.encode(candidate_contents)
        
        results = []
        for idx, item in enumerate(candidates):
            sim = cosine_sim(query_emb, candidate_embs[idx].tolist())
            # Giới hạn score nằm trong khoảng [0.0, 1.0]
            sim = max(0.0, min(1.0, sim))
            
            # Sử dụng trực tiếp điểm tương đồng ngữ nghĩa để làm thước đo chính xác tương thích với threshold
            combined_score = sim
            new_item = item.copy()
            new_item["score"] = combined_score
            results.append(new_item)
    except Exception as e:
        print(f"[WARNING] Local semantic reranking failed: {e}. Falling back to Jaccard similarity.")
        results = []
        for item in candidates:
            jaccard = local_jaccard_similarity(query, item["content"])
            combined_score = 0.3 * item.get("score", 0.0) + 0.7 * jaccard
            new_item = item.copy()
            new_item["score"] = combined_score
            results.append(new_item)

    # Sort descending
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    # Nếu candidates thiếu embedding, cần tạo hoặc bỏ qua MMR
    for c in candidates:
        if "embedding" not in c:
            print("[WARNING] Candidates thiếu embedding cho MMR, falling back to original list.")
            return candidates[:top_k]

    selected = []
    remaining = list(range(len(candidates)))

    # Chọn phần tử đầu tiên có điểm cosine similarity cao nhất với query
    first_idx = max(remaining, key=lambda idx: cosine_sim(query_embedding, candidates[idx]["embedding"]))
    selected.append(first_idx)
    remaining.remove(first_idx)

    # Lặp để chọn tiếp các phần tử
    for _ in range(min(top_k - 1, len(candidates) - 1)):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            # Tương đồng với query
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Tương đồng lớn nhất với các tài liệu đã chọn
            max_sim_to_selected = max(
                cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                for sel_idx in selected
            )

            # Công thức MMR
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores = {}  # content -> score
    content_map = {}  # content -> full dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            # Giữ lại item đầy đủ để tạo kết quả trả về
            if key not in content_map or item.get("score", 0) > content_map[key].get("score", 0):
                content_map[key] = item

    # Sắp xếp theo điểm RRF
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        results.append(item)

    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # MMR cần query_embedding, nếu gọi từ đây ta sẽ lấy từ model embedding
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = model.encode(query).tolist()
        
        # Thêm embedding cho candidates nếu thiếu
        missing_emb_indices = [i for i, c in enumerate(candidates) if "embedding" not in c]
        if missing_emb_indices:
            embeddings = model.encode([candidates[i]["content"] for i in missing_emb_indices])
            for idx, emb in zip(missing_emb_indices, embeddings):
                candidates[idx]["embedding"] = emb.tolist()
                
        return rerank_mmr(query_embedding, candidates, top_k)
    elif method == "rrf":
        # RRF cần ranked_lists. Nếu chỉ truyền candidates đơn lẻ, ta xem như chỉ có 1 list
        return rerank_rrf([candidates], top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội sản xuất trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
