import os
import sys
import warnings
import logging

# Tắt toàn bộ log cảnh báo phiền phức từ HuggingFace/transformers
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Đảm bảo đường dẫn import hoạt động
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

load_dotenv()

# Import các hàm từ RAG pipeline cá nhân
from src.multi_agent_system import MultiAgentConfig, orchestrate_query
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import reorder_for_llm, format_context, local_heuristic_generation

# =============================================================================
# OPTIMIZATION: CACHING HEAVY MODELS & CORPUS DATA (Tăng tốc độ load trang)
# =============================================================================
@st.cache_resource
def get_cached_semantic_model():
    from sentence_transformers import SentenceTransformer
    import src.task5_semantic_search as t5
    return SentenceTransformer(t5.EMBEDDING_MODEL)

@st.cache_resource
def get_cached_corpus_and_bm25():
    import src.task6_lexical_search as t6
    from src.task4_chunking_indexing import load_documents, chunk_documents
    _docs = load_documents()
    corpus = chunk_documents(_docs)
    bm25_index = t6.build_bm25_index(corpus)
    return corpus, bm25_index

def warmup_local_retrieval():
    """Chỉ warmup BM25/corpus nhẹ, tránh tải model embedding lớn ngay lúc boot app."""
    try:
        import src.task6_lexical_search as t6
        cached_corpus, cached_bm25 = get_cached_corpus_and_bm25()
        t6.CORPUS = cached_corpus
        t6._bm25_index = cached_bm25
    except Exception:
        pass


warmup_local_retrieval()

import re

# Helper function to extract metadata details
def get_source_details(source_filename: str, doc_type: str, chunk_content: str):
    """
    Trích xuất chi tiết nguồn thông tin:
    - Tên văn bản/bài viết
    - Phần/Trang/Điều luật liên quan
    - URL liên kết
    - Toàn bộ nội dung văn bản gốc
    """
    display_name = source_filename.replace(".md", "")
    section = "N/A"
    link_url = None
    full_text = ""
    
    # Đọc file markdown gốc từ data/standardized/
    md_path = Path("data/standardized") / doc_type / source_filename
    if md_path.exists():
        try:
            full_text = md_path.read_text(encoding="utf-8")
        except Exception:
            pass
            
    # 1. Nếu là văn bản pháp luật
    if doc_type == "legal":
        # Tìm các ký tự chỉ "Điều XX" hoặc "Khoản XX" trong chunk để định vị phần/trang
        match_dieu = re.search(r"(Điều\s+\d+)", chunk_content)
        match_khoan = re.search(r"(Khoản\s+\d+)", chunk_content)
        
        sections = []
        if match_dieu:
            sections.append(match_dieu.group(1))
        if match_khoan:
            sections.append(match_khoan.group(1))
            
        section = " - ".join(sections) if sections else "Thông tin chung"
            
    # 2. Nếu là bài báo tin tức
    elif doc_type == "news":
        json_name = source_filename.replace(".md", ".json")
        json_path = Path("data/landing/news") / json_name
        if json_path.exists():
            try:
                import json
                data = json.loads(json_path.read_text(encoding="utf-8"))
                link_url = data.get("url")
                display_name = data.get("title", display_name)
            except Exception:
                pass
                
        section = "Tin tức báo chí"
        
    return {
        "display_name": display_name,
        "section": section,
        "link_url": link_url,
        "full_text": full_text
    }


def should_show_citations(answer: str) -> bool:
    """
    Xác định xem có nên hiển thị nguồn trích dẫn hay không.
    Ẩn nguồn nếu câu trả lời là chào hỏi xã giao, không đủ thông tin xác minh,
    hoặc không thực sự trích dẫn tài liệu cụ thể nào.
    """
    if not answer:
        return False
        
    answer_lower = answer.lower()
    
    # 1. Ẩn nếu là lời chào xã giao ngắn
    greetings = ["xin chào", "chào bạn", "chào anh", "chào chị", "hello", "hi", "hey", "chúc một ngày"]
    if any(g in answer_lower for g in greetings) and len(answer.split()) < 25:
        return False
        
    # 2. Ẩn nếu chứa các câu từ thể hiện không tìm thấy/không thể xác minh thông tin
    unverified_phrases = [
        "tôi không thể xác minh",
        "tôi không tìm thấy",
        "không thể xác minh",
        "không tìm thấy thông tin",
        "không có thông tin",
        "không đề cập",
        "không được đề cập",
        "tôi không biết",
        "chưa có thông tin"
    ]
    if any(phrase in answer_lower for phrase in unverified_phrases):
        return False
        
    # 3. Ẩn nếu không chứa bất kỳ thẻ trích dẫn nào trong câu trả lời (dạng [bo-luat...docx] hoặc [article_01.md])
    has_brackets = bool(re.search(r"\[[^\]]+\.(docx|md)\]", answer, re.IGNORECASE))
    if not has_brackets:
        return False
        
    return True


def render_sources(sources, key_prefix="hist", msg_idx=0):
    """
    Hiển thị các nguồn trích dẫn bằng Streamlit Expander.
    """
    if not sources:
        return
        
    with st.expander(f"🔍 Xem nguồn trích dẫn ({len(sources)} đoạn văn bản)"):
        for s_idx, source in enumerate(sources, 1):
            source_name = source.get("metadata", {}).get("source", "Không rõ nguồn")
            doc_type = source.get("metadata", {}).get("type", "unknown")
            score = source.get("score", 0.0)
            retrieval_method = source.get("source", "hybrid")
            
            # Lấy thông tin nguồn chi tiết
            details = get_source_details(source_name, doc_type, source["content"])
            
            st.markdown(f"""
            **[{s_idx}] {details['display_name']}** 
            <span class="badge badge-source">{retrieval_method.upper()}</span>
            <span class="badge badge-score">Score: {score:.3f}</span>
            <span class="badge badge-type">Type: {doc_type.upper()}</span>
            <br>
            📍 **Vị trí/Phần:** `{details['section']}`
            """, unsafe_allow_html=True)
            st.caption(source["content"])
            
            # Hiển thị nút liên kết / xem tài liệu gốc
            col1, col2 = st.columns([1, 1])
            with col1:
                key = f"btn_{key_prefix}_{msg_idx}_{s_idx}"
                if st.button("📖 Xem văn bản gốc", key=key, use_container_width=True):
                    st.session_state.selected_doc = {
                        "title": details["display_name"],
                        "section": details["section"],
                        "chunk_content": source["content"],
                        "full_text": details["full_text"]
                    }
                    st.rerun()
            with col2:
                if details["link_url"]:
                    st.link_button("🔗 Mở link bài viết", url=details["link_url"], use_container_width=True)
            st.markdown("<hr style='margin: 0.4rem 0; border: 0.5px solid #F1F5F9;'>", unsafe_allow_html=True)


def render_agent_trace(agent_trace):
    """Hiển thị log supervisor-worker để dễ demo kiến trúc hệ thống."""
    if not agent_trace:
        return

    with st.expander(f"🤖 Supervisor Trace ({len(agent_trace)} steps)"):
        for idx, item in enumerate(agent_trace, 1):
            agent_name = item.get("agent", "unknown")
            agent_role = item.get("role", "agent")
            summary = item.get("summary", "")
            metadata = item.get("metadata", {})
            results_count = item.get("results_count", 0)

            st.markdown(
                f"**[{idx}] {agent_name}** (`{agent_role}`)  \n"
                f"{summary}  \n"
                f"`results={results_count}` `metadata={metadata}`"
            )


def render_overview_card(title: str, value: str, detail: str):
    st.markdown(
        f"""
        <div class="overview-card">
            <div class="overview-label">{title}</div>
            <div class="overview-value">{value}</div>
            <div class="overview-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step_badge(step: str, active: bool = False):
    tone = "active-step" if active else "idle-step"
    st.markdown(f'<div class="step-badge {tone}">{step}</div>', unsafe_allow_html=True)


# Cấu hình trang Streamlit
st.set_page_config(
    page_title="RAG Chatbot Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện premium
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Font chính */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Cấu hình background toàn trang */
    .stApp {
        background-color: #F8FAFC !important;
    }
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0B0F19 !important;
        }
    }
    
    /* Cấu hình Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    @media (prefers-color-scheme: dark) {
        [data-testid="stSidebar"] {
            background-color: #0F172A !important;
            border-right: 1px solid #1E293B !important;
        }
    }
    
    /* Tiêu đề chính giống Google Gemini */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4285F4 0%, #9B72CB 30%, #D96570 70%, #F48120 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 0.2rem;
        letter-spacing: -0.04em;
        text-shadow: 0 4px 12px rgba(0, 0, 0, 0.01);
    }
    @media (prefers-color-scheme: dark) {
        .main-header {
            background: linear-gradient(90deg, #66A6FF 0%, #B19FFB 30%, #F5A3B3 70%, #F8B375 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }
    }
    
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    @media (prefers-color-scheme: dark) {
        .sub-title {
            color: #94A3B8;
        }
    }

    .hero-panel {
        background: linear-gradient(135deg, #fff7ed 0%, #fffbeb 48%, #f8fafc 100%);
        border: 1px solid #FED7AA;
        border-radius: 24px;
        padding: 24px 24px 18px 24px;
        margin-bottom: 18px;
        box-shadow: 0 18px 40px -28px rgba(194, 65, 12, 0.35);
    }
    .hero-kicker {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid #FDBA74;
        background: rgba(251, 146, 60, 0.12);
        color: #9A3412;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        color: #111827;
        line-height: 1.05;
        margin-bottom: 6px;
    }
    .hero-text {
        font-size: 1rem;
        line-height: 1.6;
        color: #475569;
        max-width: 70ch;
    }
    .overview-card {
        background: #ffffff;
        border: 1px solid #E5E7EB;
        border-radius: 18px;
        padding: 16px 18px;
        min-height: 124px;
        box-shadow: 0 14px 28px -26px rgba(15, 23, 42, 0.38);
    }
    .overview-label {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9CA3AF;
        margin-bottom: 10px;
    }
    .overview-value {
        font-size: 1.2rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 6px;
    }
    .overview-detail {
        color: #6B7280;
        line-height: 1.45;
        font-size: 0.92rem;
    }
    .step-badge {
        border-radius: 999px;
        padding: 10px 14px;
        text-align: center;
        font-size: 0.88rem;
        font-weight: 700;
        border: 1px solid #E5E7EB;
        background: #FFFFFF;
        color: #6B7280;
        margin-bottom: 8px;
    }
    .active-step {
        color: #9A3412;
        background: #FFF7ED;
        border-color: #FDBA74;
    }
    .idle-step {
        color: #4B5563;
        background: #F8FAFC;
        border-color: #E2E8F0;
    }
    @media (prefers-color-scheme: dark) {
        .hero-panel {
            background: linear-gradient(135deg, rgba(120, 53, 15, 0.35) 0%, rgba(17, 24, 39, 0.95) 60%);
            border-color: #7C2D12;
        }
        .hero-kicker { color: #FDBA74; border-color: #9A3412; }
        .hero-title { color: #F8FAFC; }
        .hero-text { color: #CBD5E1; }
        .overview-card {
            background: #111827;
            border-color: #1F2937;
        }
        .overview-value { color: #F9FAFB; }
        .overview-detail { color: #94A3B8; }
        .step-badge { background: #0F172A; border-color: #1E293B; color: #CBD5E1; }
        .active-step { background: rgba(124, 45, 18, 0.35); color: #FDBA74; border-color: #9A3412; }
    }
    
    /* --- KEYFRAME ANIMATIONS (DƯỚI CÙNG TRANG VÀ HOVER EFFECT) --- */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(16px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        to {
            box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
        }
    }
    
    /* Hiệu ứng nhịp đập cho chấm xanh lá hoạt động */
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.6s infinite cubic-bezier(0.66, 0, 0, 1);
        vertical-align: middle;
        margin-right: 6px;
    }
    
    /* Bong bóng chat slide-up và fade-in */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 1.25rem !important;
        margin-bottom: 1rem !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.01) !important;
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.25s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.25s ease;
        animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
        border-color: #3B82F6 !important;
        box-shadow: 0 12px 20px -8px rgba(37, 99, 235, 0.08) !important;
    }
    
    @media (prefers-color-scheme: dark) {
        .stChatMessage {
            border: 1px solid #1E293B !important;
            background-color: #0F172A !important;
        }
        .stChatMessage:hover {
            border-color: #60A5FA !important;
            box-shadow: 0 12px 20px -8px rgba(0, 0, 0, 0.3) !important;
        }
    }
    
    /* Thiết kế ô nhập chat dạng viên thuốc (floating pill-shaped stChatInput) */
    div[data-testid="stChatInput"] {
        border-radius: 9999px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05) !important;
        background-color: #FFFFFF !important;
        padding: 6px 12px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #3B82F6 !important;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.15) !important;
    }
    @media (prefers-color-scheme: dark) {
        div[data-testid="stChatInput"] {
            border-color: #1E293B !important;
            background-color: #1E293B !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3) !important;
        }
        div[data-testid="stChatInput"]:focus-within {
            border-color: #60A5FA !important;
            box-shadow: 0 10px 25px -5px rgba(96, 165, 250, 0.15) !important;
        }
    }
    
    /* Expander styling */
    .stExpander {
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.01) !important;
        background-color: #F8FAFC !important;
        margin-top: 0.8rem !important;
        animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    @media (prefers-color-scheme: dark) {
        .stExpander {
            border: 1px solid #1E293B !important;
            background-color: #1E293B60 !important;
        }
    }
    
    /* Cấu hình khung hiển thị tài liệu đọc (Tab bên phải) slide-in mượt */
    .doc-viewer-container {
        animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    
    /* Custom tabs for source content */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: transparent;
        border-bottom: 2px solid #E2E8F0;
    }
    @media (prefers-color-scheme: dark) {
        .stTabs [data-baseweb="tab-list"] {
            border-bottom: 2px solid #1E293B;
        }
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px !important;
        background-color: transparent !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #64748B !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.2s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #2563EB !important;
    }
    @media (prefers-color-scheme: dark) {
        .stTabs [data-baseweb="tab"]:hover {
            color: #60A5FA !important;
        }
    }
    
    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        border-bottom: 2px solid #2563EB !important;
    }
    @media (prefers-color-scheme: dark) {
        .stTabs [aria-selected="true"] {
            color: #60A5FA !important;
            border-bottom: 2px solid #60A5FA !important;
        }
    }
    
    /* Highlight văn bản trích dẫn */
    mark {
        background-color: rgba(245, 158, 11, 0.12) !important;
        color: #D97706 !important;
        border-bottom: 2px solid #F59E0B;
        padding: 2px 4px;
        font-weight: 600;
        border-radius: 4px;
    }
    @media (prefers-color-scheme: dark) {
        mark {
            background-color: rgba(245, 158, 11, 0.25) !important;
            color: #FBBF24 !important;
            border-bottom-color: #F59E0B;
        }
    }
    
    /* Button animations (Spring physics feedback) */
    .stButton > button {
        border-radius: 8px !important;
        transition: transform 0.1s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.2s ease !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
    }
    .stButton > button:active {
        transform: translateY(1px) scale(0.98);
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.05);
    }
    
    /* Badge tags */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 6px;
        margin-right: 0.4rem;
        letter-spacing: 0.02em;
    }
    .badge-source {
        background-color: #DBEAFE;
        color: #1E40AF;
    }
    .badge-score {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .badge-type {
        background-color: #E0F2FE;
        color: #0369A1;
    }
    .badge-fallback {
        background-color: #FEF3C7;
        color: #92400E;
    }
</style>
""", unsafe_allow_html=True)



# =============================================================================
# CONVERSATIONAL QUERY CONTEXTUALIZATION (Query Rewriting)
# =============================================================================

def is_query_out_of_scope(query: str) -> bool:
    """
    Kiểm tra xem câu hỏi có nằm ngoài phạm vi hỗ trợ hay không.
    Phạm vi hỗ trợ: Luật phòng chống ma túy, tin tức ma túy/nghệ sĩ liên quan ma túy, hoặc câu chào.
    """
    query_lower = query.lower().strip()
    
    # 1. Nếu là câu chào xã giao ngắn
    greetings = ["hello", "hi", "xin chào", "chào bạn", "chào", "chao ban", "chao", "hey", "chúc một ngày"]
    if query_lower in greetings or (any(g in query_lower for g in greetings) and len(query.split()) <= 4):
        return False
        
    # 2. Danh sách từ khóa chặn nhanh (Chặn các chủ đề ngoại lệ rõ ràng như lập trình, nấu ăn, thời tiết...)
    out_of_scope_keywords = [
        "viết code", "viết script", "lập trình", "code python", "code c#", "code java", "code javascript",
        "công thức nấu", "nấu ăn", "nấu món", "cách làm bánh", "cách nấu", "hướng dẫn nấu", "món ăn",
        "dự báo thời tiết", "thời tiết hôm nay", "giá vàng hôm nay", "chơi game", "tải game", "tải phần mềm"
    ]
    if any(keyword in query_lower for keyword in out_of_scope_keywords):
        return True

    # 3. Danh sách các từ khóa liên quan đến phạm vi hỗ trợ
    in_scope_keywords = [
        "ma túy", "ma tuy", "cai nghiện", "cai nghien", "tàng trữ", "tang tru", 
        "vận chuyển", "van chuyen", "mua bán", "mua ban", "sử dụng", "su dung", 
        "hình phạt", "hinh phat", "phạt tù", "phat tu", "bộ luật", "bo luat", 
        "luật", "luat", "điều", "dieu", "khoản", "khoan", "nghệ sĩ", "nghe si", 
        "ca sĩ", "ca si", "diễn viên", "dien vien", "người mẫu", "nguoi mau",
        "chi dân", "an tây", "andrea", "hữu tín", "lệ hằng", "châu việt cường", 
        "chất cấm", "chat cam", "heroin", "cocaine", "ketamine", "thuốc lắc", 
        "thuoc lac", "cần sa", "can sa", "bắt", "khởi tố", "tạm giữ", "án tù", 
        "bo-luat", "luat-phong-chong", "article_"
    ]
    
    if any(keyword in query_lower for keyword in in_scope_keywords):
        return False
        
    # 3. Sử dụng OpenAI GPT để phân loại chính xác nếu có API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    use_openai = api_key and not api_key.startswith("sk-xxx") and len(api_key) > 15
    if use_openai:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = (
                "Phân loại câu hỏi sau có liên quan đến chủ đề luật phòng chống ma túy, "
                "chất ma túy, hành vi liên quan đến ma túy, tin tức ma túy, nghệ sĩ liên quan đến ma túy, "
                "hoặc câu chào hỏi hay không?\n"
                f"Câu hỏi: \"{query}\"\n"
                "Chỉ trả lời duy nhất chữ 'YES' (nếu liên quan) hoặc 'NO' (nếu không liên quan)."
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5
            )
            decision = response.choices[0].message.content.strip().upper()
            if "YES" in decision:
                return False
            elif "NO" in decision:
                return True
        except Exception:
            pass
            
    # Mặc định nếu không khớp từ khóa và không có API key thì coi như out of scope
    return True


def contextualize_query_local(query: str, history: list) -> str:
    """
    Sử dụng OpenAI hoặc quy tắc heuristic để viết lại câu hỏi follow-up
    thành câu hỏi độc lập đầy đủ ngữ cảnh dựa trên lịch sử hội thoại.
    """
    if not history:
        return query
        
    api_key = os.getenv("OPENAI_API_KEY", "")
    use_openai = api_key and not api_key.startswith("sk-xxx") and len(api_key) > 15
    
    # Chỉ viết lại nếu câu hỏi quá ngắn hoặc chứa từ chỉ định
    query_lower = query.lower()
    pronouns = ["nó", "họ", "anh ấy", "cô ấy", "ông này", "bà này", "đó", "này", "ở đâu", "bao nhiêu", "như thế nào", "tại sao"]
    is_follow_up = len(query.split()) < 5 or any(p in query_lower for p in pronouns)
    
    if not is_follow_up:
        return query
        
    if use_openai:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # Xây dựng lịch sử tóm tắt
            history_text = ""
            for msg in history[-3:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"
                
            prompt = (
                "Given the following conversation history and a follow-up question, "
                "rewrite the follow-up question to be a standalone question in Vietnamese that "
                "contains all necessary context. Do NOT answer the question, just rewrite it.\n\n"
                f"History:\n{history_text}\n"
                f"Follow-up: {query}\n"
                "Standalone Question:"
            )
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100
            )
            rewritten = response.choices[0].message.content.strip()
            print(f"Contextualized query: '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            print(f"Failed to contextualize query via API: {e}")
            
    # Heuristic Fallback: Ghép với chủ đề câu hỏi trước đó nếu phát hiện follow-up
    last_user_msg = next((msg["content"] for msg in reversed(history) if msg["role"] == "user"), "")
    if last_user_msg:
        # Nếu câu hỏi trước hỏi về 1 nghệ sĩ cụ thể
        for name in ["Chi Dân", "An Tây", "Andrea", "Hữu Tín", "Lệ Hằng", "Châu Việt Cường"]:
            if name.lower() in last_user_msg.lower() and name.lower() not in query_lower:
                return f"{query} liên quan đến {name}"
        # Ghép mặc định với câu hỏi trước
        return f"{query} ({last_user_msg})"
        
    return query


# =============================================================================
# MULTI-TURN GENERATION WITH CUSTOM PIPELINE CONFIGS
# =============================================================================

def generate_answer_with_history(
    query: str,
    retrieved_chunks: list,
    history: list,
    temperature: float = 0.3,
    top_p: float = 0.9
) -> dict:
    """
    Sinh câu trả lời có trích dẫn nguồn, hỗ trợ đa lượt hội thoại (multi-turn)
    bằng cách gửi kèm lịch sử chat trước đó tới OpenAI.
    """
    result = orchestrate_query(
        query=query,
        history=history,
        config=MultiAgentConfig(
            top_k=max(len(retrieved_chunks), 5) if retrieved_chunks else 5,
            temperature=temperature,
            top_p=top_p,
            include_trace=True,
        ),
    )
    return result


def run_single_agent_query(
    query: str,
    history: list,
    top_k: int,
    score_threshold: float,
    use_reranking: bool,
    use_hyde: bool,
    temperature: float,
    top_p: float,
) -> dict:
    """Luồng classic RAG: 1 retriever pipeline + 1 synthesizer."""
    try:
        chunks = retrieve(
            query,
            top_k=top_k,
            score_threshold=score_threshold,
            use_reranking=use_reranking,
            use_hyde=use_hyde,
        )
    except Exception:
        from src.task6_lexical_search import lexical_search

        chunks = lexical_search(query, top_k=top_k)
        for item in chunks:
            item["source"] = item.get("source", "hybrid")
    reordered_chunks = reorder_for_llm(chunks)
    context = format_context(reordered_chunks)
    api_key = os.getenv("OPENAI_API_KEY", "")
    use_openai = api_key and not api_key.startswith("sk-xxx") and len(api_key) > 15

    if not use_openai:
        answer = local_heuristic_generation(query, reordered_chunks)
    else:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Bạn là single-agent RAG assistant. Chỉ dùng context được cung cấp. "
                        "Mỗi nhận định thực tế phải có citation dạng [source.md] hoặc [source.docx]. "
                        "Nếu thiếu bằng chứng, hãy nói rõ không thể xác minh."
                    ),
                }
            ]
            for msg in history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append(
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\n---\n\nQuestion: {query}",
                }
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                timeout=15,
            )
            answer = response.choices[0].message.content
        except Exception:
            answer = local_heuristic_generation(query, reordered_chunks)

    agent_trace = [
        {
            "agent": "single_retriever",
            "summary": "Chạy retrieval pipeline classic: semantic + lexical + fallback.",
            "results_count": len(chunks),
            "metadata": {
                "mode": "single-agent",
                "use_reranking": use_reranking,
                "use_hyde": use_hyde,
            },
        },
        {
            "agent": "single_synthesizer",
            "summary": "Tổng hợp câu trả lời từ một context hợp nhất.",
            "results_count": len(reordered_chunks),
            "metadata": {
                "mode": "single-agent",
                "llm": "openai" if use_openai else "local",
            },
        },
    ]
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "agent_trace": agent_trace,
        "planner_route": "classic",
    }


# =============================================================================
# STREAMLIT UI LAYOUT
# =============================================================================

st.markdown(
    """
    <div class="hero-panel">
        <div class="hero-kicker">Drug Law Demo Console</div>
        <div class="hero-title">Chatbot RAG về luật ma túy và tin tức nghệ sĩ</div>
        <div class="hero-text">
            Giao diện này được thiết kế để người xem thấy rõ hệ thống đang chạy theo kiến trúc nào,
            lấy nguồn từ đâu và tổng hợp câu trả lời như thế nào. Bạn có thể chuyển ngay giữa
            <b>Single-Agent</b> và <b>Multi-Agent</b> để demo sự khác biệt.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Điều Khiển Demo")
system_mode = st.sidebar.radio(
    "1. Chọn kiến trúc",
    options=["Multi-Agent", "Single-Agent"],
    captions=[
        "Supervisor giao việc cho 2-3 worker rồi tổng hợp lại.",
        "Một pipeline classic: retrieve rồi generate.",
    ],
)

preset_mode = st.sidebar.radio(
    "2. Chọn preset",
    options=["Demo nhanh", "Cân bằng", "Chuyên sâu", "Tùy chỉnh"],
    captions=[
        "Ưu tiên tốc độ và ít bước hơn.",
        "Cân bằng giữa tốc độ và độ chính xác.",
        "Tìm nhiều nguồn hơn, phù hợp khi trình diễn kỹ.",
        "Tự cấu hình toàn bộ tham số.",
    ],
)

if preset_mode == "Demo nhanh":
    use_reranking = system_mode == "Multi-Agent"
    use_hyde = False
    score_threshold = 0.35
    top_k = 3
    temperature = 0.2
    top_p = 0.85
elif preset_mode == "Cân bằng":
    use_reranking = True
    use_hyde = False
    score_threshold = 0.3
    top_k = 5
    temperature = 0.3
    top_p = 0.9
elif preset_mode == "Chuyên sâu":
    use_reranking = True
    use_hyde = True
    score_threshold = 0.25
    top_k = 6
    temperature = 0.25
    top_p = 0.9
else:
    with st.sidebar.expander("3. Tùy chỉnh nâng cao", expanded=True):
        use_reranking = st.checkbox("Bật reranking", value=True)
        use_hyde = st.checkbox("Bật HyDE", value=False)
        score_threshold = st.slider("Ngưỡng fallback", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        top_k = st.slider("Số nguồn lấy về", min_value=1, max_value=10, value=5)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1)

show_agent_trace = st.sidebar.checkbox(
    "Hiển thị trace agent",
    value=system_mode == "Multi-Agent",
    help="Bật để người xem thấy hệ thống đang chạy qua những bước nào.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    **Tóm tắt cấu hình**
    
    - Kiến trúc: `{system_mode}`
    - Preset: `{preset_mode}`
    - Top K: `{top_k}`
    - Reranking: `{'On' if use_reranking else 'Off'}`
    - HyDE: `{'On' if use_hyde else 'Off'}`
    """,
)

with st.sidebar.expander("Giải thích ngắn gọn", expanded=False):
    st.markdown(
        """
        - `Single-Agent`: một retriever pipeline cổ điển rồi sinh câu trả lời.
        - `Multi-Agent`: supervisor định tuyến, 2-3 worker truy xuất song song, critic gộp kết quả, rồi synthesizer trả lời.
        - `HyDE`: sinh truy vấn giả định để retrieval ngữ nghĩa tốt hơn.
        - `Reranking`: sắp xếp lại nguồn trước khi sinh câu trả lời.
        """
    )

with st.sidebar.expander("Trạng thái API", expanded=False):
    st.caption("Thiếu API key thì hệ thống sẽ tự dùng fallback local.")
    api_keys_status = {
        "OpenAI API": os.getenv("OPENAI_API_KEY", ""),
        "Jina AI Reranker": os.getenv("JINA_API_KEY", ""),
        "PageIndex SDK": os.getenv("PAGEINDEX_API_KEY", "")
    }

    for name, key in api_keys_status.items():
        if key and not key.startswith("sk-xxx") and not key.startswith("jina_xxx") and not key.startswith("pi_xxx") and len(key) > 10:
            st.success(f"✔️ {name}: hoạt động")
        else:
            st.warning(f"⚠️ {name}: fallback local")

# Nút Xóa lịch sử chat
if st.sidebar.button("🗑️ Xóa Lịch Sử Chat"):
    st.session_state.messages = []
    st.session_state.selected_doc = None
    st.session_state.last_sources = []
    st.session_state.last_agent_trace = []
    st.rerun()

# Khởi tạo các session state cần thiết
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "last_agent_trace" not in st.session_state:
    st.session_state.last_agent_trace = []

with st.sidebar:
    if show_agent_trace and st.session_state.last_agent_trace:
        render_agent_trace(st.session_state.last_agent_trace)

summary_col1, summary_col2, summary_col3 = st.columns(3)
with summary_col1:
    render_overview_card(
        "Kiến trúc đang chạy",
        system_mode,
        "Chuyển nhanh giữa classic pipeline và điều phối nhiều agent.",
    )
with summary_col2:
    render_overview_card(
        "Preset demo",
        preset_mode,
        f"Top K = {top_k}, Reranking = {'On' if use_reranking else 'Off'}, HyDE = {'On' if use_hyde else 'Off'}.",
    )
with summary_col3:
    render_overview_card(
        "Kiểu trả lời",
        "Có citation",
        "Mỗi câu trả lời đều cố gắng gắn nguồn pháp luật hoặc bài báo tương ứng.",
    )

st.markdown("**Luồng xử lý hiện tại**")
flow_cols = st.columns(5)
with flow_cols[0]:
    render_step_badge("1. Nhận câu hỏi", active=True)
with flow_cols[1]:
    render_step_badge("2. Supervisor lập kế hoạch" if system_mode == "Multi-Agent" else "2. Retrieval classic", active=True)
with flow_cols[2]:
    render_step_badge("3. 2-3 Worker song song" if system_mode == "Multi-Agent" else "3. Gộp nguồn", active=system_mode == "Multi-Agent")
with flow_cols[3]:
    render_step_badge("4. Critic + Rerank" if system_mode == "Multi-Agent" else "4. Sắp lại context", active=True)
with flow_cols[4]:
    render_step_badge("5. Sinh câu trả lời", active=True)

# Chia giao diện chính thành 2 cột: Cột Trò chuyện (trái) và Cột Tài liệu nguồn (phải)
chat_col, doc_col = st.columns([5, 4])

with chat_col:
    # Hiển thị lịch sử chat
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Hiển thị các nguồn trích dẫn
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                if should_show_citations(message["content"]):
                    render_sources(message["sources"], key_prefix="hist", msg_idx=idx)
            if show_agent_trace and message["role"] == "assistant" and message.get("agent_trace"):
                render_agent_trace(message["agent_trace"])

    # Sinh câu trả lời của Trợ lý nếu tin nhắn cuối cùng trong lịch sử là của người dùng
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        latest_prompt = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Kiểm tra câu hỏi ngoài phạm vi hỗ trợ (Ngoại lệ)
            if is_query_out_of_scope(latest_prompt):
                answer = (
                    "Xin lỗi, tôi là RAG Chatbot Assistant. Tôi chỉ có thể giải đáp các thắc mắc "
                    "liên quan đến Luật phòng chống ma túy Việt Nam và tin tức liên quan đến các nghệ sĩ vi phạm pháp luật "
                    "về ma túy. Câu hỏi của bạn nằm ngoài phạm vi hỗ trợ của tôi."
                )
                message_placeholder.markdown(answer)
                st.session_state.last_sources = []
                st.session_state.selected_doc = None
                st.session_state.last_agent_trace = []
                
                # Lưu vào lịch sử hội thoại
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": [],
                    "agent_trace": []
                })
                st.rerun()
            
            # Rewrite câu hỏi dựa trên lịch sử nếu có
            contextualized = contextualize_query_local(latest_prompt, st.session_state.messages[:-1])
            
            if system_mode == "Multi-Agent":
                with st.spinner("🤖 Multi-agent đang lập kế hoạch, truy xuất và tổng hợp câu trả lời..."):
                    orchestration = orchestrate_query(
                        query=contextualized,
                        history=st.session_state.messages[:-1],
                        config=MultiAgentConfig(
                            top_k=top_k,
                            score_threshold=score_threshold,
                            use_reranking=use_reranking,
                            use_hyde=use_hyde,
                            temperature=temperature,
                            top_p=top_p,
                            include_trace=show_agent_trace
                        )
                    )
            else:
                with st.spinner("🧭 Single-agent đang truy xuất nguồn và tạo câu trả lời..."):
                    orchestration = run_single_agent_query(
                        query=contextualized,
                        history=st.session_state.messages[:-1],
                        top_k=top_k,
                        score_threshold=score_threshold,
                        use_reranking=use_reranking,
                        use_hyde=use_hyde,
                        temperature=temperature,
                        top_p=top_p,
                    )

            chunks = orchestration["sources"]
            answer = orchestration["answer"]
            agent_trace = orchestration.get("agent_trace", []) if show_agent_trace else []
            message_placeholder.markdown(answer)
            
            # Lưu các nguồn được truy xuất và tự động đặt tài liệu đầu tiên làm mặc định hiển thị
            if chunks and should_show_citations(answer):
                st.session_state.last_sources = chunks
                st.session_state.last_agent_trace = agent_trace
                
                # Tự động chọn tài liệu đầu tiên
                first_source = chunks[0]
                s_name = first_source.get("metadata", {}).get("source", "Không rõ nguồn")
                d_type = first_source.get("metadata", {}).get("type", "unknown")
                details = get_source_details(s_name, d_type, first_source["content"])
                st.session_state.selected_doc = {
                    "title": details["display_name"],
                    "section": details["section"],
                    "chunk_content": first_source["content"],
                    "full_text": details["full_text"]
                }
                
                render_sources(chunks, key_prefix="live", msg_idx=999)
                if show_agent_trace:
                    render_agent_trace(agent_trace)
            else:
                st.session_state.last_sources = []
                st.session_state.selected_doc = None
                st.session_state.last_agent_trace = agent_trace
                        
            # Lưu vào lịch sử hội thoại
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": chunks,
                "agent_trace": agent_trace
            })
            
            # Reload lại trang để cột bên cạnh hiển thị tài liệu gốc ngay lập tức
            st.rerun()

# Ô nhập tin nhắn ghim ở cuối trang (ngoài chat_col để tránh giật/nhảy vị trí viết)
chat_placeholder = (
    "Ví dụ: Hình phạt cho tội tàng trữ ma túy là gì?"
    if system_mode == "Single-Agent"
    else "Ví dụ: So sánh thông tin pháp luật và tin tức liên quan đến một vụ việc ma túy"
)

if prompt := st.chat_input(chat_placeholder):
    # 1. Lưu tin nhắn của người dùng vào session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 2. Reload lại trang để kích hoạt nhánh xử lý sinh câu trả lời của trợ lý
    st.rerun()

with doc_col:
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 1.5rem; font-weight: 700; color: #1E3A8A;">📄 Chi tiết Tài liệu Nguồn</div>', unsafe_allow_html=True)
    
    if st.session_state.last_sources:
        # Nhúng class CSS để kích hoạt animation slide-up cho khu vực xem tài liệu
        st.markdown('<div class="doc-viewer-container">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["📖 Nội dung tài liệu gốc", "🔍 Danh sách nguồn trích dẫn"])
        
        with tab1:
            if st.session_state.selected_doc:
                doc = st.session_state.selected_doc
                st.markdown(f"### {doc['title']}")
                if doc['section'] and doc['section'] != "N/A":
                    st.info(f"📍 Phần trích dẫn: **{doc['section']}**")
                    
                st.markdown("**Nội dung văn bản gốc (Phần trích dẫn được bôi màu vàng bên dưới):**")
                
                highlighted_text = doc['full_text']
                if doc['chunk_content'] and doc['chunk_content'] in doc['full_text']:
                    # Sử dụng thẻ mark thuần để CSS custom bôi màu mềm mại hơn
                    mark_tag = f'<mark>{doc["chunk_content"]}</mark>'
                    highlighted_text = doc['full_text'].replace(doc['chunk_content'], mark_tag)
                
                with st.container(height=550):
                    st.markdown(highlighted_text, unsafe_allow_html=True)
            else:
                st.info("💡 Hãy click nút **'📖 Xem văn bản gốc'** của các nguồn trích dẫn trong phần chat hoặc tab bên cạnh để đọc tài liệu gốc tại đây.")
                
        with tab2:
            st.markdown("Các đoạn trích dẫn được hệ thống RAG tìm thấy cho câu hỏi hiện tại:")
            for s_idx, source in enumerate(st.session_state.last_sources, 1):
                source_name = source.get("metadata", {}).get("source", "Không rõ nguồn")
                doc_type = source.get("metadata", {}).get("type", "unknown")
                score = source.get("score", 0.0)
                retrieval_method = source.get("source", "hybrid")
                
                # Lấy thông tin nguồn chi tiết
                details = get_source_details(source_name, doc_type, source["content"])
                
                st.markdown(f"""
                **[{s_idx}] {details['display_name']}** 
                <span class="badge badge-source">{retrieval_method.upper()}</span>
                <span class="badge badge-score">Score: {score:.3f}</span>
                <span class="badge badge-type">Type: {doc_type.upper()}</span>
                <br>
                📍 **Vị trí/Phần:** `{details['section']}`
                """, unsafe_allow_html=True)
                st.caption(source["content"])
                
                # Nút chuyển tài liệu gốc
                col1, col2 = st.columns([1, 1])
                with col1:
                    key = f"btn_tab2_{s_idx}"
                    if st.button("📖 Xem văn bản gốc", key=key, use_container_width=True):
                        st.session_state.selected_doc = {
                            "title": details["display_name"],
                            "section": details["section"],
                            "chunk_content": source["content"],
                            "full_text": details["full_text"]
                        }
                        st.rerun()
                with col2:
                    if details["link_url"]:
                        st.link_button("🔗 Mở link bài viết", url=details["link_url"], use_container_width=True)
                st.markdown("<hr style='margin: 0.4rem 0; border: 0.5px solid #F1F5F9;'>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("💡 Chưa có tài liệu nguồn nào được truy xuất cho câu trả lời này (hoặc câu trả lời là lời chào / thông tin không xác minh được).")
