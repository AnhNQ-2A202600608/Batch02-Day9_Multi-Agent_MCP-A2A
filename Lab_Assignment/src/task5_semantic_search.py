"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import os
import warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

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

CHROMA_DB_PATH = ROOT_DIR / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


_SEMANTIC_MODEL = None

def get_semantic_model():
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SEMANTIC_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _SEMANTIC_MODEL


def generate_hyde_document(query: str) -> str:
    """
    Sinh tài liệu giả định (Hypothetical Document) bằng LLM hoặc fallback local heuristic.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-xxx") or len(api_key) < 15:
        from src.task10_generation import local_heuristic_generation
        return local_heuristic_generation(query, [])

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            f"Hãy viết một đoạn văn ngắn (dưới 150 từ) bằng tiếng Việt trả lời giả định cho câu hỏi sau. "
            f"Đoạn văn này sẽ được dùng làm tài liệu tham khảo để đối sánh vector.\n"
            f"Câu hỏi: {query}\n"
            f"Câu trả lời giả định:"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
            timeout=5
        )
        return response.choices[0].message.content
    except Exception:
        from src.task10_generation import local_heuristic_generation
        return local_heuristic_generation(query, [])


def semantic_search(query: str, top_k: int = 10, use_hyde: bool = False) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not CHROMA_DB_PATH.exists():
        print(f"[WARNING] Vector store không tồn tại tại {CHROMA_DB_PATH}")
        return []

    # 1. Khởi tạo Chroma Client và lấy collection
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    try:
        collection = client.get_collection(name="DrugLawDocs")
    except Exception as e:
        print(f"[WARNING] Không thể load collection: {e}")
        return []

    # 2. Embed query bằng cùng model ở Task 4
    model = get_semantic_model()
    if use_hyde:
        hyde_doc = generate_hyde_document(query)
        query_embedding = model.encode(hyde_doc).tolist()
    else:
        query_embedding = model.encode(query).tolist()

    # 3. Query vector store
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # 4. Parse kết quả
    # Chroma returns a list of lists for documents, metadatas, distances
    hits = []
    if not results or not results["documents"] or len(results["documents"][0]) == 0:
        return hits

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    for i in range(len(documents)):
        distance = distances[i]
        # Cosine distance = 1 - cosine similarity. Do đó similarity = 1 - distance
        score = 1.0 - distance
        
        hits.append({
            "content": documents[i],
            "score": float(score),
            "metadata": metadatas[i] if metadatas[i] else {}
        })

    # Sắp xếp lại theo score giảm dần
    hits = sorted(hits, key=lambda x: x["score"], reverse=True)
    return hits


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
