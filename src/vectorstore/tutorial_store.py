from __future__ import annotations

import hashlib
import re
import time
from types import SimpleNamespace
from typing import Any

from ..config import get_settings
from ..utils.logger import get_logger
from .chroma_client import VectorStoreManager

logger = get_logger(__name__)

_TUTORIAL_VECTORSTORES: dict[str, Any] = {}
_DEFAULT_TUTORIAL_COLLECTION_NAME = "recipe_tutorials"
_TUTORIAL_QUERY_STRIP_MARKERS = (
    "怎么做",
    "做法",
    "步骤",
    "分步骤",
    "制作",
    "教程",
    "如何做",
    "详细一点",
    "再详细一点",
    "再展开说说",
    "展开说说",
    "再展开",
    "具体步骤",
    "具体怎么做",
    "讲细一点",
    "说细一点",
    "展开一下",
    "这个",
    "这道菜",
    "这道",
    "它",
    "那个",
    "上一道",
    "上一个",
)


def _resolve_collection_name(collection_name: str | None = None) -> str:
    settings = get_settings()
    configured_name = str(getattr(settings, "tutorial_collection_name", "") or "").strip()
    return str(collection_name or configured_name or _DEFAULT_TUTORIAL_COLLECTION_NAME).strip()


def _resolve_collection_names(
    collection_name: str | None = None,
    collection_names: list[str] | None = None,
) -> list[str]:
    raw_names = list(collection_names or [])
    if collection_name:
        raw_names.append(collection_name)
    if not raw_names:
        raw_names.append(_resolve_collection_name())

    normalized_names: list[str] = []
    seen_names: set[str] = set()
    for raw_name in raw_names:
        normalized_name = _resolve_collection_name(raw_name)
        if not normalized_name or normalized_name in seen_names:
            continue
        seen_names.add(normalized_name)
        normalized_names.append(normalized_name)
    return normalized_names


def get_tutorial_vectorstore(collection_name: str | None = None) -> Any | None:
    return _TUTORIAL_VECTORSTORES.get(_resolve_collection_name(collection_name))


def get_or_connect_tutorial_vectorstore(collection_name: str | None = None) -> Any | None:
    resolved_collection_name = _resolve_collection_name(collection_name)
    cached = _TUTORIAL_VECTORSTORES.get(resolved_collection_name)
    if cached is not None:
        return cached

    try:
        from langchain_community.embeddings import DashScopeEmbeddings
        from langchain_community.vectorstores import Chroma
    except Exception as exc:
        logger.error(f"tutorial vectorstore imports failed: {exc}", exc_info=True)
        return None

    settings = get_settings()
    try:
        manager = VectorStoreManager()
        client = manager.get_client(settings.chroma_db_path)
        embeddings = DashScopeEmbeddings(
            model=settings.embedding_model,
            dashscope_api_key=settings.dashscope_api_key,
        )
        tutorial_vectorstore = Chroma(
            collection_name=resolved_collection_name,
            embedding_function=embeddings,
            client=client,
            persist_directory=settings.chroma_db_path,
        )
        _TUTORIAL_VECTORSTORES[resolved_collection_name] = tutorial_vectorstore
        return tutorial_vectorstore
    except Exception as exc:
        logger.error(f"tutorial vectorstore connection failed: {exc}", exc_info=True)
        return None


def get_tutorial_collection_count(collection_name: str | None = None, vectorstore: Any | None = None) -> int | None:
    resolved_vectorstore = vectorstore or get_or_connect_tutorial_vectorstore(collection_name)
    collection = getattr(resolved_vectorstore, "_collection", None) if resolved_vectorstore is not None else None
    if collection is None or not hasattr(collection, "count"):
        return None

    try:
        return int(collection.count())
    except Exception as exc:
        logger.warning(f"tutorial collection count failed: {exc}", exc_info=True)
        return None


def _resolve_chunk_id(chunk: dict[str, Any], text: str) -> str:
    chunk_id = str(chunk.get("id") or "").strip()
    if chunk_id:
        return chunk_id
    tutorial_id = str((chunk.get("metadata") or {}).get("tutorial_id") or "tutorial_chunk").strip() or "tutorial_chunk"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{tutorial_id}::{digest}"


def _extract_step_segments(text: str) -> list[str]:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return []
    matches = re.findall(r"第\d+步[^。]*?(?:。|$)", normalized_text)
    return [segment.strip().rstrip("。") for segment in matches if segment.strip()]


def _normalize_tutorial_query_terms(query: str) -> list[str]:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return []

    cleaned_query = normalized_query
    for phrase in _TUTORIAL_QUERY_STRIP_MARKERS:
        cleaned_query = cleaned_query.replace(phrase, " ")

    terms: list[str] = []
    for candidate in [cleaned_query.strip(), normalized_query.strip()]:
        if candidate and candidate not in terms:
            terms.append(candidate)
    for candidate in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", cleaned_query):
        normalized_candidate = candidate.strip()
        if normalized_candidate and normalized_candidate not in terms:
            terms.append(normalized_candidate)
    return terms


def _score_tutorial_title_match(query: str, doc: dict[str, Any]) -> tuple[int, list[str]]:
    query_terms = _normalize_tutorial_query_terms(query)
    if not query_terms:
        return 0, []

    titles = [
        str(doc.get("name") or "").strip(),
        str(doc.get("tutorial_title") or "").strip(),
        str(doc.get("recipe_name") or "").strip(),
    ]
    matched_terms: list[str] = []
    score = 0
    for term in query_terms:
        normalized_term = str(term).strip()
        if not normalized_term:
            continue
        if any(normalized_term in title for title in titles if title):
            matched_terms.append(normalized_term)
            score += 100 + len(normalized_term)
    return score, matched_terms


def _similarity_search_with_retry(
    vectorstore: Any,
    *,
    query: str,
    k: int,
    max_attempts: int = 4,
    base_delay_seconds: float = 1.0,
) -> list[Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return vectorstore.similarity_search(query=query, k=k)
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            sleep_seconds = round(base_delay_seconds * (2 ** (attempt - 1)), 2)
            logger.warning(
                f"tutorial similarity_search retrying: attempt={attempt}/{max_attempts}, sleep={sleep_seconds}s, error={exc}"
            )
            time.sleep(sleep_seconds)
    if last_error is not None:
        raise last_error
    return []


def _keyword_search_tutorial_chunks(vectorstore: Any, query: str, *, top_k: int) -> list[Any]:
    collection = getattr(vectorstore, "_collection", None)
    if collection is None or not hasattr(collection, "get"):
        return []

    try:
        payload = collection.get(include=["documents", "metadatas"])
    except Exception as exc:
        logger.error(f"tutorial keyword fallback failed to read collection: {exc}", exc_info=True)
        return []

    documents = list(payload.get("documents") or [])
    metadatas = list(payload.get("metadatas") or [])
    query_terms = _normalize_tutorial_query_terms(query)
    if not query_terms:
        return []

    scored_chunks: list[tuple[float, Any]] = []
    for metadata, page_content in zip(metadatas, documents):
        normalized_metadata = dict(metadata or {})
        normalized_page_content = str(page_content or "").strip()
        title = str(
            normalized_metadata.get("recipe_name")
            or normalized_metadata.get("tutorial_title")
            or normalized_metadata.get("tutorial_id")
            or ""
        ).strip()
        haystack = f"{title}\n{normalized_page_content}"
        score = 0.0
        for term in query_terms:
            if not term:
                continue
            if term in title:
                score += 20.0 + len(term)
            if term in haystack:
                score += 5.0 + haystack.count(term)
        if score <= 0:
            continue
        scored_chunks.append(
            (
                score,
                SimpleNamespace(metadata=normalized_metadata, page_content=normalized_page_content),
            )
        )

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored_chunks[:top_k]]


def _search_single_tutorial_collection(
    query: str,
    *,
    top_k: int = 5,
    collection_name: str | None = None,
    vectorstore: Any | None = None,
    chunk_fetch_k: int | None = None,
) -> list[dict[str, Any]]:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return []

    resolved_collection_name = _resolve_collection_name(collection_name)
    resolved_vectorstore = vectorstore or get_or_connect_tutorial_vectorstore(resolved_collection_name)
    if resolved_vectorstore is None:
        return []

    raw_limit = max(int(chunk_fetch_k or (top_k * 3)), int(top_k), 1)
    try:
        raw_docs = _similarity_search_with_retry(
            resolved_vectorstore,
            query=normalized_query,
            k=raw_limit,
        )
    except Exception as exc:
        logger.error(f"tutorial similarity search failed: {exc}", exc_info=True)
        raw_docs = _keyword_search_tutorial_chunks(resolved_vectorstore, normalized_query, top_k=raw_limit)
        if raw_docs:
            logger.warning("tutorial retrieval falling back to local keyword search")
        else:
            return []

    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for rank, raw_doc in enumerate(raw_docs):
        metadata = dict(getattr(raw_doc, "metadata", {}) or {})
        page_content = str(getattr(raw_doc, "page_content", "") or "").strip()
        tutorial_id = str(metadata.get("tutorial_id") or metadata.get("recipe_id") or metadata.get("id") or "").strip()
        if not tutorial_id:
            continue

        if tutorial_id not in grouped:
            order.append(tutorial_id)
            grouped[tutorial_id] = {
                "id": tutorial_id,
                "name": str(metadata.get("recipe_name") or metadata.get("tutorial_title") or tutorial_id).strip() or tutorial_id,
                "tutorial_title": str(metadata.get("tutorial_title") or metadata.get("recipe_name") or tutorial_id).strip() or tutorial_id,
                "recipe_name": str(metadata.get("recipe_name") or metadata.get("tutorial_title") or tutorial_id).strip() or tutorial_id,
                "description": "",
                "text_parts": [],
                "step_items": [],
                "difficulty": metadata.get("difficulty", ""),
                "time": metadata.get("time", 0),
                "calories": metadata.get("calories", 0),
                "cuisine": metadata.get("cuisine", ""),
                "tags": metadata.get("tags", ""),
                "health_goals": metadata.get("health_goals", ""),
                "scenarios": metadata.get("scenarios", ""),
                "entity_type": metadata.get("entity_type", "tutorial_chunk"),
                "source_type": metadata.get("source_type", "tutorial"),
                "collection_name": resolved_collection_name,
                "score": max(0.0, 1.0 - (rank * 0.01)),
            }

        group = grouped[tutorial_id]
        chunk_type = str(metadata.get("chunk_type") or "").strip()
        if page_content and page_content not in group["text_parts"]:
            group["text_parts"].append(page_content)
        if chunk_type == "overview" and page_content and not group["description"]:
            group["description"] = page_content
        if chunk_type == "steps":
            step_start = int(metadata.get("step_start") or metadata.get("chunk_index") or 0)
            for step_segment in _extract_step_segments(page_content) or ([page_content] if page_content else []):
                if step_segment and step_segment not in [item[1] for item in group["step_items"]]:
                    group["step_items"].append((step_start, step_segment))

    normalized_docs: list[dict[str, Any]] = []
    for tutorial_id in order[:top_k]:
        group = grouped[tutorial_id]
        ordered_steps = [item[1] for item in sorted(group["step_items"], key=lambda item: item[0])]
        normalized_docs.append(
            {
                "id": group["id"],
                "name": group["name"],
                "tutorial_title": group["tutorial_title"],
                "recipe_name": group["recipe_name"],
                "description": group["description"] or (group["text_parts"][0] if group["text_parts"] else ""),
                "text": "\n".join(group["text_parts"]),
                "steps": ordered_steps,
                "difficulty": group["difficulty"],
                "time": group["time"],
                "calories": group["calories"],
                "cuisine": group["cuisine"],
                "tags": group["tags"],
                "health_goals": group["health_goals"],
                "scenarios": group["scenarios"],
                "entity_type": group["entity_type"],
                "source_type": group["source_type"],
                "collection_name": group["collection_name"],
                "score": group["score"],
            }
        )
    return normalized_docs


def search_tutorial_documents(
    query: str,
    *,
    top_k: int = 5,
    collection_name: str | None = None,
    collection_names: list[str] | None = None,
    vectorstore: Any | None = None,
    vectorstores: dict[str, Any] | None = None,
    chunk_fetch_k: int | None = None,
    direct_topic_match_required: bool = True,
) -> list[dict[str, Any]]:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return []

    query_terms = _normalize_tutorial_query_terms(normalized_query)
    if not query_terms:
        return []

    resolved_collection_names = _resolve_collection_names(
        collection_name=collection_name,
        collection_names=collection_names,
    )
    aggregated_docs: list[dict[str, Any]] = []
    for resolved_collection_name in resolved_collection_names:
        resolved_vectorstore = None
        if isinstance(vectorstores, dict):
            resolved_vectorstore = vectorstores.get(resolved_collection_name)
        elif len(resolved_collection_names) == 1:
            resolved_vectorstore = vectorstore

        aggregated_docs.extend(
            _search_single_tutorial_collection(
                normalized_query,
                top_k=top_k,
                collection_name=resolved_collection_name,
                vectorstore=resolved_vectorstore,
                chunk_fetch_k=chunk_fetch_k,
            )
        )

    if not aggregated_docs:
        return []

    merged_docs: dict[str, dict[str, Any]] = {}
    for doc in aggregated_docs:
        doc_copy = dict(doc)
        collection_name_value = str(doc_copy.get("collection_name") or "").strip()
        topic_match_score, topic_match_terms = _score_tutorial_title_match(normalized_query, doc_copy)
        doc_copy["topic_match_score"] = topic_match_score
        doc_copy["topic_match_terms"] = topic_match_terms
        doc_copy["direct_topic_match"] = bool(topic_match_terms)
        doc_copy["matched_collection_names"] = [collection_name_value] if collection_name_value else []

        tutorial_id = str(doc_copy.get("id") or "").strip()
        if not tutorial_id:
            continue

        existing = merged_docs.get(tutorial_id)
        if existing is None:
            merged_docs[tutorial_id] = doc_copy
            continue

        combined_collection_names = list(
            dict.fromkeys(existing.get("matched_collection_names", []) + doc_copy["matched_collection_names"])
        )
        combined_topic_terms = list(
            dict.fromkeys(existing.get("topic_match_terms", []) + topic_match_terms)
        )

        existing_score = (
            int(bool(existing.get("direct_topic_match"))),
            int(existing.get("topic_match_score") or 0),
            float(existing.get("score") or 0.0),
        )
        candidate_score = (
            int(bool(doc_copy.get("direct_topic_match"))),
            int(topic_match_score),
            float(doc_copy.get("score") or 0.0),
        )
        if candidate_score > existing_score:
            doc_copy["matched_collection_names"] = combined_collection_names
            doc_copy["topic_match_terms"] = combined_topic_terms
            merged_docs[tutorial_id] = doc_copy
        else:
            existing["matched_collection_names"] = combined_collection_names
            existing["topic_match_terms"] = combined_topic_terms
            existing["direct_topic_match"] = bool(combined_topic_terms)
            existing["topic_match_score"] = max(
                int(existing.get("topic_match_score") or 0),
                int(topic_match_score),
            )

    sorted_docs = sorted(
        merged_docs.values(),
        key=lambda doc: (
            int(bool(doc.get("direct_topic_match"))),
            int(doc.get("topic_match_score") or 0),
            float(doc.get("score") or 0.0),
        ),
        reverse=True,
    )
    direct_match_docs = [doc for doc in sorted_docs if doc.get("direct_topic_match")]
    if direct_topic_match_required and not direct_match_docs:
        return []
    if direct_topic_match_required:
        return direct_match_docs[:top_k]
    return sorted_docs[:top_k]


def _add_texts_with_retry(
    vectorstore: Any,
    *,
    texts: list[str],
    metadatas: list[dict[str, Any]],
    ids: list[str],
    retry_label: str,
    max_attempts: int = 4,
    base_delay_seconds: float = 1.0,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            return
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            sleep_seconds = round(base_delay_seconds * (2 ** (attempt - 1)), 2)
            logger.warning(
                f"tutorial add_texts retrying: label={retry_label}, attempt={attempt}/{max_attempts}, sleep={sleep_seconds}s, error={exc}"
            )
            time.sleep(sleep_seconds)
    if last_error is not None:
        raise last_error


def _get_existing_tutorial_chunk_ids(vectorstore: Any, ids: list[str]) -> set[str]:
    normalized_ids = [str(chunk_id).strip() for chunk_id in ids if str(chunk_id).strip()]
    if not normalized_ids:
        return set()

    collection = getattr(vectorstore, "_collection", None)
    if collection is not None and hasattr(collection, "get"):
        try:
            payload = collection.get(ids=normalized_ids)
            return {str(chunk_id).strip() for chunk_id in list(payload.get("ids") or []) if str(chunk_id).strip()}
        except Exception as exc:
            logger.warning(f"tutorial existing-id probe via collection.get failed: {exc}")

    getter = getattr(vectorstore, "get", None)
    if callable(getter):
        try:
            payload = getter(ids=normalized_ids)
            return {str(chunk_id).strip() for chunk_id in list(payload.get("ids") or []) if str(chunk_id).strip()}
        except Exception as exc:
            logger.warning(f"tutorial existing-id probe via vectorstore.get failed: {exc}")

    return set()


def batch_add_tutorial_chunks(
    tutorial_chunks: list[dict[str, Any]],
    vectorstore: Any | None = None,
    collection_name: str | None = None,
) -> tuple[int, int, list[str]]:
    if not tutorial_chunks:
        return 0, 0, []

    resolved_vectorstore = vectorstore or get_or_connect_tutorial_vectorstore(collection_name)
    if resolved_vectorstore is None:
        error_message = "tutorial vectorstore is not initialized"
        return 0, len(tutorial_chunks), [error_message]

    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
    errors: list[str] = []

    for index, chunk in enumerate(tutorial_chunks):
        if not isinstance(chunk, dict):
            errors.append(f"chunk_{index}: invalid chunk payload")
            continue
        text = str(chunk.get("text") or chunk.get("page_content") or "").strip()
        metadata = chunk.get("metadata") or {}
        if not text:
            errors.append(f"chunk_{index}: empty text")
            continue
        if not isinstance(metadata, dict):
            errors.append(f"chunk_{index}: invalid metadata")
            continue
        texts.append(text)
        metadatas.append(metadata)
        ids.append(_resolve_chunk_id(chunk, text))

    if not texts:
        return 0, len(errors), errors

    existing_ids = _get_existing_tutorial_chunk_ids(resolved_vectorstore, ids)
    skipped_count = len(existing_ids)
    if skipped_count:
        logger.info(f"tutorial chunk ids already exist, skipping reinsert: count={skipped_count}")

    pending_records = [
        (text, metadata, chunk_id)
        for text, metadata, chunk_id in zip(texts, metadatas, ids)
        if chunk_id not in existing_ids
    ]
    if not pending_records:
        return skipped_count, len(errors), errors

    texts = [record[0] for record in pending_records]
    metadatas = [record[1] for record in pending_records]
    ids = [record[2] for record in pending_records]

    try:
        _add_texts_with_retry(
            resolved_vectorstore,
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            retry_label="tutorial_batch",
        )
        return skipped_count + len(texts), len(errors), errors
    except Exception as exc:
        logger.warning(f"tutorial batch add failed, falling back to single inserts: {exc}")

    success_count = skipped_count
    fallback_errors = list(errors)
    for text, metadata, chunk_id in zip(texts, metadatas, ids):
        try:
            _add_texts_with_retry(
                resolved_vectorstore,
                texts=[text],
                metadatas=[metadata],
                ids=[chunk_id],
                retry_label=chunk_id,
            )
            success_count += 1
        except Exception as exc:
            fallback_errors.append(f"{chunk_id}: {exc}")

    return success_count, len(fallback_errors), fallback_errors
