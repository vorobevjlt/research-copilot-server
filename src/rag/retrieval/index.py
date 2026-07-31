from src.services.llm import openAI
from fastapi import HTTPException
from src.services.supabase import supabase
from src.rag.retrieval.utils import (
    get_project_settings,
    get_project_document_ids,
    build_context_from_retrieved_chunks,
    generate_query_variations,
)
from typing import List, Dict
from src.rag.retrieval.utils import rrf_rank_and_fuse


def retrieve_context(project_id, user_query):
    try:
        project_settings = get_project_settings(project_id)

        document_ids = get_project_document_ids(project_id)

        strategy = project_settings["rag_strategy"]
        chunks = []
        if strategy == "basic":
            chunks = vector_search(user_query, document_ids, project_settings)
            print(f"Vector search resulted in: {len(chunks)} chunks")

        elif strategy == "hybrid":
            chunks = hybrid_search(user_query, document_ids, project_settings)
            print(f"Hybrid search resulted in: {len(chunks)} chunks")

        elif strategy == "multi-query-vector":
            chunks = multi_query_vector_search(
                user_query, document_ids, project_settings
            )
            print(f"Multi-query vector search resulted in: {len(chunks)} chunks")

        elif strategy == "multi-query-hybrid":
            chunks = multi_query_hybrid_search(
                user_query, document_ids, project_settings
            )
            print(f"Multi-query hybrid search resulted in: {len(chunks)} chunks")

        chunks = chunks[: project_settings["final_context_size"]]

        texts, images, tables, citations = build_context_from_retrieved_chunks(chunks)

        return texts, images, tables, citations
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed in RAG's Retrieval: {str(e)}"
        )


def vector_search(user_query, document_ids, project_settings):
    user_query_embedding = openAI["embeddings"].embed_documents([user_query])[0]
    vector_search_result_chunks = supabase.rpc(
        "vector_search_document_chunks",
        {
            "query_embedding": user_query_embedding,
            "filter_document_ids": document_ids,
            "match_threshold": project_settings["similarity_threshold"],
            "chunks_per_search": project_settings["chunks_per_search"],
        },
    ).execute()
    return vector_search_result_chunks.data if vector_search_result_chunks.data else []


def keyword_search(query, document_ids, settings):
    keyword_search_result_chunks = supabase.rpc(
        "keyword_search_document_chunks",
        {
            "query_text": query,
            "filter_document_ids": document_ids,
            "chunks_per_search": settings["chunks_per_search"],
        },
    ).execute()

    return (
        keyword_search_result_chunks.data if keyword_search_result_chunks.data else []
    )


def hybrid_search(query: str, document_ids: List[str], settings: dict) -> List[Dict]:
    vector_results = vector_search(query, document_ids, settings)
    keyword_results = keyword_search(query, document_ids, settings)

    print(f"📈 Vector search returned: {len(vector_results)} chunks")
    print(f"📈 Keyword search returned: {len(keyword_results)} chunks")

    return rrf_rank_and_fuse(
        [vector_results, keyword_results],
        [settings["vector_weight"], settings["keyword_weight"]],
    )


def multi_query_vector_search(user_query, document_ids, project_settings):
    queries = generate_query_variations(
        user_query, project_settings["number_of_queries"]
    )
    print(f"Generated {len(queries)} query variations")

    all_chunks = []
    for index, query in enumerate(queries):
        chunks = vector_search(query, document_ids, project_settings)
        all_chunks.append(chunks)
        print(
            f"Vector search for query {index+1}/{len(queries)}: {query} resulted in: {len(chunks)} chunks"
        )

    final_chunks = rrf_rank_and_fuse(all_chunks)
    print(f"RRF Fusion returned {len(final_chunks)} chunks")
    return final_chunks


def multi_query_hybrid_search(user_query, document_ids, project_settings):
    queries = generate_query_variations(
        user_query, project_settings["number_of_queries"]
    )
    print(f"Generated {len(queries)} query variations for hybrid search")

    all_chunks = []
    for index, query in enumerate(queries):
        chunks = hybrid_search(query, document_ids, project_settings)
        all_chunks.append(chunks)
        print(
            f"Hybrid search for query {index+1}/{len(queries)}: {query} resulted in: {len(chunks)} chunks"
        )

    final_chunks = rrf_rank_and_fuse(all_chunks)
    print(f"RRF Fusion returned {len(final_chunks)} chunks")
    return final_chunks
