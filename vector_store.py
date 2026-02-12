"""
NeuroFlow Vector Store — ChromaDB
Manages persistent vector collections for tasks and interventions.
"""

import os
from typing import Optional
import chromadb


CHROMA_PATH = os.path.join(os.path.dirname(__file__), "data", "chroma")


def _get_client() -> chromadb.ClientAPI:
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def _tasks_collection():
    client = _get_client()
    return client.get_or_create_collection(
        name="tasks_collection",
        metadata={"description": "Task context embeddings for similarity search"},
    )


def _interventions_collection():
    client = _get_client()
    return client.get_or_create_collection(
        name="interventions_collection",
        metadata={"description": "Successful intervention patterns"},
    )


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def add_task_embedding(
    task_id: str,
    description: str,
    metadata: Optional[dict] = None,
) -> None:
    """Store a task description (ChromaDB auto-embeds with its default model)."""
    col = _tasks_collection()
    meta = metadata or {}
    # ChromaDB metadata values must be str, int, float, or bool
    safe_meta = {k: str(v) for k, v in meta.items()}
    col.upsert(
        ids=[task_id],
        documents=[description],
        metadatas=[safe_meta],
    )


def query_similar_tasks(query: str, n_results: int = 5) -> list[dict]:
    """Find tasks similar to the query description."""
    col = _tasks_collection()
    if col.count() == 0:
        return []
    results = col.query(query_texts=[query], n_results=min(n_results, col.count()))
    tasks = []
    for i, doc_id in enumerate(results["ids"][0]):
        tasks.append({
            "task_id": doc_id,
            "description": results["documents"][0][i],
            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            "distance": results["distances"][0][i] if results["distances"] else None,
        })
    return tasks


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------

def add_intervention(
    intervention_id: str,
    pattern_type: str,
    intervention_text: str,
    success: bool = False,
    context: str = "",
) -> None:
    col = _interventions_collection()
    col.upsert(
        ids=[intervention_id],
        documents=[intervention_text],
        metadatas=[{
            "pattern_type": pattern_type,
            "success": str(success),
            "context": context,
        }],
    )


def query_similar_interventions(
    query: str, n_results: int = 3
) -> list[dict]:
    col = _interventions_collection()
    if col.count() == 0:
        return []
    results = col.query(query_texts=[query], n_results=min(n_results, col.count()))
    interventions = []
    for i, doc_id in enumerate(results["ids"][0]):
        interventions.append({
            "intervention_id": doc_id,
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            "distance": results["distances"][0][i] if results["distances"] else None,
        })
    return interventions
