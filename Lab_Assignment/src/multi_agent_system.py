"""
Multi-agent orchestration layer for the RAG chatbot.
"""

from __future__ import annotations

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

_SEMANTIC_AVAILABLE: bool | None = None


@dataclass
class AgentReport:
    agent: str
    summary: str
    results_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiAgentConfig:
    top_k: int = 5
    score_threshold: float = 0.3
    use_reranking: bool = True
    use_hyde: bool = False
    temperature: float = 0.3
    top_p: float = 0.9
    include_trace: bool = True


LEGAL_KEYWORDS = {
    "luật", "điều", "khoản", "nghị định", "hình phạt", "tội", "phạt tù",
    "cai nghiện", "quy định", "quản lý", "bắt buộc", "bộ luật", "xử lý",
}
NEWS_KEYWORDS = {
    "nghệ sĩ", "ca sĩ", "diễn viên", "người mẫu", "bị bắt", "bị tạm giữ",
    "bài báo", "tin tức", "chi dân", "an tây", "hữu tín", "lệ hằng",
    "châu việt cường", "andrea",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _keyword_overlap(query: str, content: str) -> float:
    query_tokens = set(_tokenize(query))
    content_tokens = set(_tokenize(content))
    if not query_tokens or not content_tokens:
        return 0.0
    return len(query_tokens & content_tokens) / len(query_tokens)


def _filter_by_type(results: list[dict], doc_type: str) -> list[dict]:
    return [item for item in results if item.get("metadata", {}).get("type") == doc_type]


def _safe_semantic_search(query: str, top_k: int, use_hyde: bool) -> list[dict]:
    global _SEMANTIC_AVAILABLE
    if _SEMANTIC_AVAILABLE is False:
        return []
    try:
        from src.task5_semantic_search import semantic_search

        results = semantic_search(query, top_k=top_k, use_hyde=use_hyde)
        _SEMANTIC_AVAILABLE = True
        return results
    except Exception:
        _SEMANTIC_AVAILABLE = False
        return []


def _safe_lexical_search(query: str, top_k: int) -> list[dict]:
    try:
        from src.task6_lexical_search import lexical_search

        return lexical_search(query, top_k=top_k)
    except Exception:
        return []


def _safe_retrieve(query: str, config: MultiAgentConfig) -> list[dict]:
    try:
        from src.task9_retrieval_pipeline import retrieve

        return retrieve(
            query,
            top_k=config.top_k * 2,
            score_threshold=config.score_threshold,
            use_reranking=config.use_reranking,
            use_hyde=config.use_hyde,
        )
    except Exception:
        fallback = _safe_lexical_search(query, top_k=config.top_k * 2)
        for item in fallback:
            item["source"] = item.get("source", "hybrid")
        return fallback


def _dedupe_results(results: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    for item in results:
        metadata = item.get("metadata", {})
        key = (
            metadata.get("source", ""),
            metadata.get("type", ""),
            int(metadata.get("chunk_index", -1)),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


class PlannerAgent:
    name = "planner"

    def run(self, query: str, history: list[dict] | None = None) -> dict[str, Any]:
        query_lower = query.lower()
        legal_hits = sum(1 for keyword in LEGAL_KEYWORDS if keyword in query_lower)
        news_hits = sum(1 for keyword in NEWS_KEYWORDS if keyword in query_lower)

        if legal_hits and news_hits:
            route = "cross_domain"
        elif legal_hits:
            route = "legal"
        elif news_hits:
            route = "news"
        else:
            route = "hybrid"

        requires_breadth = any(
            phrase in query_lower
            for phrase in ("so sánh", "liên quan", "tác động", "gồm những ai", "tổng hợp")
        )
        if requires_breadth and route != "cross_domain":
            route = "hybrid"

        recent_context = ""
        if history:
            last_user = next((msg["content"] for msg in reversed(history) if msg["role"] == "user"), "")
            if last_user and last_user != query:
                recent_context = last_user

        return {
            "route": route,
            "legal_weight": 1.15 if route in {"legal", "cross_domain"} else 1.0,
            "news_weight": 1.15 if route in {"news", "cross_domain"} else 1.0,
            "hybrid_weight": 1.1 if route == "hybrid" else 1.0,
            "recent_context": recent_context,
            "summary": f"Planner routed query to `{route}` mode.",
        }


class LegalRetrievalAgent:
    name = "legal_retriever"

    def run(self, query: str, config: MultiAgentConfig) -> tuple[list[dict], AgentReport]:
        candidates: list[dict] = []
        candidates.extend(_filter_by_type(_safe_semantic_search(query, top_k=config.top_k * 2, use_hyde=config.use_hyde), "legal"))
        candidates.extend(_filter_by_type(_safe_lexical_search(query, top_k=config.top_k * 2), "legal"))
        candidates.extend(_filter_by_type(_safe_retrieve(query, config), "legal"))
        candidates = _dedupe_results(candidates)
        report = AgentReport(
            agent=self.name,
            summary=f"Collected {len(candidates)} legal candidates.",
            results_count=len(candidates),
            metadata={"focus": "legal"},
        )
        return candidates[: config.top_k * 2], report


class NewsRetrievalAgent:
    name = "news_retriever"

    def run(self, query: str, config: MultiAgentConfig) -> tuple[list[dict], AgentReport]:
        candidates: list[dict] = []
        candidates.extend(_filter_by_type(_safe_semantic_search(query, top_k=config.top_k * 2, use_hyde=config.use_hyde), "news"))
        candidates.extend(_filter_by_type(_safe_lexical_search(query, top_k=config.top_k * 2), "news"))
        candidates.extend(_filter_by_type(_safe_retrieve(query, config), "news"))
        candidates = _dedupe_results(candidates)
        report = AgentReport(
            agent=self.name,
            summary=f"Collected {len(candidates)} news candidates.",
            results_count=len(candidates),
            metadata={"focus": "news"},
        )
        return candidates[: config.top_k * 2], report


class HybridRetrievalAgent:
    name = "hybrid_retriever"

    def run(self, query: str, config: MultiAgentConfig) -> tuple[list[dict], AgentReport]:
        candidates = _safe_retrieve(query, config)
        report = AgentReport(
            agent=self.name,
            summary=f"Collected {len(candidates)} hybrid candidates.",
            results_count=len(candidates),
            metadata={"focus": "hybrid"},
        )
        return candidates, report


class CriticAgent:
    name = "critic"

    def run(
        self,
        query: str,
        plan: dict[str, Any],
        legal_results: list[dict],
        news_results: list[dict],
        hybrid_results: list[dict],
        config: MultiAgentConfig,
    ) -> tuple[list[dict], AgentReport]:
        merged = _dedupe_results(legal_results + news_results + hybrid_results)

        weighted: list[dict] = []
        for item in merged:
            new_item = item.copy()
            metadata = new_item.get("metadata", {})
            doc_type = metadata.get("type", "unknown")
            base_score = float(new_item.get("score", 0.0))
            overlap_score = _keyword_overlap(query, new_item.get("content", ""))

            if doc_type == "legal":
                domain_weight = plan["legal_weight"]
            elif doc_type == "news":
                domain_weight = plan["news_weight"]
            else:
                domain_weight = 1.0

            if new_item.get("source") == "hybrid":
                domain_weight *= plan["hybrid_weight"]

            final_score = (base_score * 0.75 + overlap_score * 0.25) * domain_weight
            new_item["score"] = round(final_score, 4)
            new_item["agent"] = self.name
            weighted.append(new_item)

        weighted.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        final_chunks = weighted[: config.top_k]
        report = AgentReport(
            agent=self.name,
            summary=f"Ranked {len(merged)} unique candidates and kept top {len(final_chunks)}.",
            results_count=len(final_chunks),
            metadata={"route": plan["route"]},
        )
        return final_chunks, report


class SynthesizerAgent:
    name = "synthesizer"

    def run(
        self,
        query: str,
        chunks: list[dict],
        history: list[dict] | None,
        config: MultiAgentConfig,
        plan: dict[str, Any],
    ) -> tuple[str, AgentReport]:
        from src.task10_generation import format_context, local_heuristic_generation, reorder_for_llm

        reordered = reorder_for_llm(chunks)
        context = format_context(reordered)
        api_key = os.getenv("OPENAI_API_KEY", "")
        use_openai = api_key and not api_key.startswith("sk-xxx") and len(api_key) > 15

        if not use_openai:
            answer = local_heuristic_generation(query, reordered)
            report = AgentReport(
                agent=self.name,
                summary="Used local heuristic synthesis.",
                results_count=len(reordered),
                metadata={"llm_mode": "local", "route": plan["route"]},
            )
            return answer, report

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            system_prompt = (
                "You are the final synthesis agent in a Vietnamese multi-agent RAG system.\n"
                "Use only the provided context.\n"
                "Every factual statement must contain an inline citation like [source.md].\n"
                "If evidence is incomplete, say 'Tôi không thể xác minh thông tin này từ nguồn hiện có'.\n"
                f"Planner route: {plan['route']}."
            )

            messages = [{"role": "system", "content": system_prompt}]
            if history:
                messages.extend(history[-6:])
            messages.append(
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\n---\n\nQuestion: {query}",
                }
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=config.temperature,
                top_p=config.top_p,
                timeout=15,
            )
            answer = response.choices[0].message.content
            report = AgentReport(
                agent=self.name,
                summary="Used OpenAI synthesis with citations.",
                results_count=len(reordered),
                metadata={"llm_mode": "openai", "route": plan["route"]},
            )
            return answer, report
        except Exception:
            answer = local_heuristic_generation(query, reordered)
            report = AgentReport(
                agent=self.name,
                summary="OpenAI failed, fell back to local heuristic synthesis.",
                results_count=len(reordered),
                metadata={"llm_mode": "fallback", "route": plan["route"]},
            )
            return answer, report


class MultiAgentOrchestrator:
    def __init__(self) -> None:
        self.planner = PlannerAgent()
        self.legal_agent = LegalRetrievalAgent()
        self.news_agent = NewsRetrievalAgent()
        self.hybrid_agent = HybridRetrievalAgent()
        self.critic = CriticAgent()
        self.synthesizer = SynthesizerAgent()

    def run(
        self,
        query: str,
        history: list[dict] | None = None,
        config: MultiAgentConfig | None = None,
    ) -> dict[str, Any]:
        config = config or MultiAgentConfig()
        plan = self.planner.run(query, history=history)
        reports = [
            AgentReport(
                agent=self.planner.name,
                summary=plan["summary"],
                metadata={"route": plan["route"]},
            )
        ]

        selected_agents = [self.hybrid_agent]
        if plan["route"] == "legal":
            selected_agents.append(self.legal_agent)
        elif plan["route"] == "news":
            selected_agents.append(self.news_agent)
        else:
            selected_agents.extend([self.legal_agent, self.news_agent])

        futures = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            for agent in selected_agents:
                futures[executor.submit(agent.run, query, config)] = agent.name

            legal_results: list[dict] = []
            news_results: list[dict] = []
            hybrid_results: list[dict] = []

            for future in as_completed(futures):
                agent_name = futures[future]
                results, report = future.result()
                reports.append(report)
                if agent_name == self.legal_agent.name:
                    legal_results = results
                elif agent_name == self.news_agent.name:
                    news_results = results
                else:
                    hybrid_results = results

        final_chunks, critic_report = self.critic.run(
            query=query,
            plan=plan,
            legal_results=legal_results,
            news_results=news_results,
            hybrid_results=hybrid_results,
            config=config,
        )
        reports.append(critic_report)

        answer, synthesis_report = self.synthesizer.run(
            query=query,
            chunks=final_chunks,
            history=history,
            config=config,
            plan=plan,
        )
        reports.append(synthesis_report)

        retrieval_source = final_chunks[0].get("source", "hybrid") if final_chunks else "none"
        return {
            "answer": answer,
            "sources": final_chunks,
            "retrieval_source": retrieval_source,
            "agent_trace": [report.__dict__ for report in reports] if config.include_trace else [],
            "planner_route": plan["route"],
        }


def orchestrate_query(
    query: str,
    history: list[dict] | None = None,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    orchestrator = MultiAgentOrchestrator()
    return orchestrator.run(query=query, history=history, config=config)
