"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25
"""

import sys
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

CORPUS = None

def get_corpus():
    global CORPUS
    if CORPUS is None:
        try:
            from src.task4_chunking_indexing import load_documents, chunk_documents
            _docs = load_documents()
            CORPUS = chunk_documents(_docs)
        except Exception as e:
            print(f"[WARNING] Không thể tải corpus: {e}")
            CORPUS = []
    return CORPUS

_bm25_index = None


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    # Tokenize đơn giản bằng split() cho tiếng Việt
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global _bm25_index
    corpus = get_corpus()
    if not corpus:
        return []

    if _bm25_index is None:
        _bm25_index = build_bm25_index(corpus)

    tokenized_query = query.lower().split()
    scores = _bm25_index.get_scores(tokenized_query)

    # Lấy các index có score cao nhất
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

    results = []
    for idx in ranked_indices[:top_k]:
        if scores[idx] > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
