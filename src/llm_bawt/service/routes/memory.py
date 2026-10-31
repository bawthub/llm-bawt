"""Memory management routes."""

from fastapi import APIRouter, Depends, HTTPException, Query

from llm_bawt.memory.errors import MemoryUpdateConflictError, MemoryUpdateDuplicateError

from ..dependencies import get_effective_bot_id, get_service, require_memory_client
from ..logging import get_service_logger
from ..schemas import (
    ConsolidateRequest,
    ConsolidateResponse,
    MemoryDeleteResponse,
    MemoryForgetRequest,
    MemoryForgetResponse,
    MemoryItem,
    MemoryUpdateRequest,
    MemoryRestoreResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryStatsResponse,
    MessagePreview,
    MessagesPreviewResponse,
    RegenerateEmbeddingsResponse,
)

router = APIRouter()
log = get_service_logger(__name__)


def _build_preview_response(bot_id: str, messages: list[dict]) -> MessagesPreviewResponse:
    return MessagesPreviewResponse(
        bot_id=bot_id,
        messages=[
            MessagePreview(
                id=msg["id"],
                role=msg.get("role", "?"),
                content=msg.get("content", ""),
                timestamp=msg.get("timestamp"),
            )
            for msg in messages
        ],
        total_count=len(messages),
    )


def _require_memory_client_for_bot(bot_id: str):
    client = get_service().get_memory_client(bot_id)
    if not client:
        raise HTTPException(status_code=503, detail="Memory service unavailable")
    return client


@router.get("/v1/memory/stats", response_model=MemoryStatsResponse, tags=["Memory"])
def get_memory_stats(
    bot_id: str = Depends(get_effective_bot_id),
    client=Depends(require_memory_client),
):
    """Get memory statistics for a bot."""
    try:
        stats = client.stats()
        return MemoryStatsResponse(
            bot_id=bot_id,
            messages=stats.get("messages", {}),
            memories=stats.get("memories", {}),
        )
    except Exception as e:
        log.error(f"Failed to get memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/memory/search", response_model=MemorySearchResponse, tags=["Memory"])
def search_memory(request: MemorySearchRequest):
    """Search memories."""
    effective_bot_id = request.bot_id or get_effective_bot_id()

    try:
        client = _require_memory_client_for_bot(effective_bot_id)

        results = []
        if request.method in ("embedding", "all"):
            memories = client.search(
                request.query,
                n_results=request.limit,
                min_relevance=request.min_importance,
            )
            for mem in memories:
                results.append(
                    MemoryItem(
                        id=str(getattr(mem, "id", "")),
                        content=str(getattr(mem, "content", "")),
                        importance=float(getattr(mem, "importance", 0.5)),
                        relevance=getattr(mem, "relevance", None),
                        tags=list(getattr(mem, "tags", []) or []),
                        created_at=getattr(mem, "created_at", None),
                        access_count=0,
                    )
                )
        elif request.method == "high-importance":
            memories = client.list_memories(
                limit=request.limit,
                min_importance=request.min_importance or 0.7,
            )
            for mem in memories:
                results.append(
                    MemoryItem(
                        id=str(mem.get("id", "")),
                        content=mem.get("content", ""),
                        importance=mem.get("importance", 0.5),
                        tags=mem.get("tags", []),
                        created_at=mem.get("created_at"),
                    )
                )

        return MemorySearchResponse(
            bot_id=effective_bot_id,
            method=request.method,
            query=request.query,
            results=results,
            total_count=len(results),
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to search memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/memory", response_model=MemorySearchResponse, tags=["Memory"])
def list_memories(
    limit: int = Query(20, description="Max results"),
    bot_id: str = Depends(get_effective_bot_id),
    client=Depends(require_memory_client),
):
    """List all memories for a bot (ordered by importance)."""
    try:
        memories = client.list_memories(limit=limit, min_importance=0.0)

        results = [
            MemoryItem(
                id=str(mem.get("id", "")),
                content=mem.get("content", ""),
                importance=mem.get("importance", 0.5),
                tags=mem.get("tags", []),
                created_at=mem.get("created_at"),
                access_count=mem.get("access_count", 0),
            )
            for mem in memories
        ]

        return MemorySearchResponse(
            bot_id=bot_id,
            method="list",
            query="",
            results=results,
            total_count=len(results),
        )
    except Exception as e:
        log.error(f"Failed to list memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/memory/message", tags=["Memory"])
def get_message_by_id(
    message_id: str = Query(..., description="Message ID (supports prefix match)"),
    bot_id: str = Depends(get_effective_bot_id),
    client=Depends(require_memory_client),
):
    """Get a specific message by ID."""
    try:
        message = client.get_message_by_id(message_id)
        if message:
            return {"message": message}
        return {"error": f"Message '{message_id}' not found"}
    except Exception as e:
        log.error(f"Failed to get message for bot '{bot_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/v1/memory/{memory_id}", response_model=MemoryDeleteResponse, tags=["Memory"])
def delete_memory(
    memory_id: str,
    bot_id: str = Depends(get_effective_bot_id),
    client=Depends(require_memory_client),
):
    """Delete a specific memory by ID."""
    try:
        success = client.delete_memory(memory_id)
        if success:
            return MemoryDeleteResponse(
                success=True,
                memory_id=memory_id,
                message=f"Memory '{memory_id}' deleted",
            )
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to delete memory for bot '{bot_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/v1/memory/{memory_id}", response_model=MemoryItem, tags=["Memory"])
def update_memory(
    memory_id: str,
    request: MemoryUpdateRequest,
    bot_id: str = Depends(get_effective_bot_id),
    client=Depends(require_memory_client),
):
    """Update an existing memory by ID."""
    try:
        if request.content is None and request.importance is None and request.tags is None:
            raise HTTPException(status_code=400, detail="Provide at least one field: content, importance, or tags")

        updated = client.update_memory(
            memory_id=memory_id,
            content=request.content,
            importance=request.importance,
            tags=request.tags,
        )
        if not updated:
            raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found")

        return MemoryItem(
            id=str(getattr(updated, "id", "")),
            content=str(getattr(updated, "content", "")),
            importance=float(getattr(updated, "importance", 0.5)),
            relevance=getattr(updated, "relevance", None),
            tags=list(getattr(updated, "tags", []) or []),
            created_at=getattr(updated, "created_at", None),
            last_accessed=getattr(updated, "last_accessed", None),
            source_message_ids=list(getattr(updated, "source_message_ids", []) or []),
        )
    except HTTPException:
        raise
    except (MemoryUpdateDuplicateError, MemoryUpdateConflictError) as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        log.error(f"Failed to update memory for bot '{bot_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/memory/forget", response_model=MemoryForgetResponse, tags=["Memory"])
def forget_messages(
    request: MemoryForgetRequest,
    bot_id: str = Depends(get_effective_bot_id),
    client=Depends(require_memory_client),
):
    """Forget recent messages (soft delete)."""
    try:
        if request.message_id:
            success = client.ignore_message_by_id(request.message_id)
            if success:
                return MemoryForgetResponse(
                    success=True,
                    messages_ignored=1,
                    memories_deleted=0,
                    message=f"Ignored message {request.message_id[:8]}...",
                )
            raise HTTPException(status_code=404, detail=f"Message {request.message_id} not found")

        if request.count:
            result = client.forget_recent_messages(request.count)
        elif request.minutes:
            result = client.forget_messages_since_minutes(request.minutes)
        else:
            raise HTTPException(status_code=400, detail="Must specify count, minutes, or message_id")

        messages_ignored = int(result.get("messages_ignored", 0))
        memories_deleted = int(result.get("memories_deleted", 0))

        return MemoryForgetResponse(
            success=True,
            messages_ignored=messages_ignored,
            memories_deleted=memories_deleted,
            message=f"Ignored {messages_ignored} messages, deleted {memories_deleted} memories",
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to forget messages for bot '{bot_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/memory/restore", response_model=MemoryRestoreResponse, tags=["Memory"])
def restore_messages(
    bot_id: str = Depends(get_effective_bot_id),
    client=Depends(require_memory_client),
):
    """Restore ignored messages."""
    try:
        restored = client.restore_ignored_messages()
        return MemoryRestoreResponse(
            success=True,
            messages_restored=restored,
            message=f"Restored {restored} messages",
        )
    except Exception as e:
        log.error(f"Failed to restore messages for bot '{bot_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/memory/preview/recent", response_model=MessagesPreviewResponse, tags=["Memory"])
def preview_recent_messages(
    count: int = Query(10, description="Number of recent messages to preview"),
    bot_id: str = Depends(get_effective_bot_id),
    client=Depends(require_memory_client),
):
    """Preview recent messages before forgetting."""
    try:
        messages = client.preview_recent_messages(count)
        return _build_preview_response(bot_id, messages)
    except Exception as e:
        log.error(f"Failed to preview messages for bot '{bot_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/memory/preview/minutes", response_model=MessagesPreviewResponse, tags=["Memory"])
def preview_messages_since_minutes(
    minutes: int = Query(..., description="Number of minutes to look back"),
    bot_id: str = Depends(get_effective_bot_id),
    client=Depends(require_memory_client),
):
    """Preview messages from last N minutes before forgetting."""
    try:
        messages = client.preview_messages_since_minutes(minutes)
        return _build_preview_response(bot_id, messages)
    except Exception as e:
        log.error(f"Failed to preview messages by minutes for bot '{bot_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/memory/preview/ignored", response_model=MessagesPreviewResponse, tags=["Memory"])
def preview_ignored_messages(
    bot_id: str = Depends(get_effective_bot_id),
    client=Depends(require_memory_client),
):
    """Preview ignored messages before restoring."""
    try:
        messages = client.preview_ignored_messages()
        return _build_preview_response(bot_id, messages)
    except Exception as e:
        log.error(f"Failed to preview ignored messages for bot '{bot_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/memory/regenerate-embeddings", response_model=RegenerateEmbeddingsResponse, tags=["Memory"])
def regenerate_embeddings(
    bot_id: str = Depends(get_effective_bot_id),
    client=Depends(require_memory_client),
):
    """Run one bounded content-and-meaning embedding repair pass."""
    try:
        result = client.regenerate_embeddings()

        if "error" in result:
            return RegenerateEmbeddingsResponse(
                success=False,
                updated=0,
                failed=0,
                message=result["error"],
            )

        return RegenerateEmbeddingsResponse(
            success=True,
            updated=result.get("updated", 0),
            failed=result.get("failed", 0),
            embedding_dim=result.get("embedding_dim"),
            message=(f"Checked {result.get('examined', 0)} of up to 25 rows; "
                     f"updated {result.get('updated', 0)} embeddings. "
                     "Run again to continue any remaining repairs."),
        )
    except Exception as e:
        log.error(f"Failed to regenerate embeddings for bot '{bot_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/memory/consolidate", response_model=ConsolidateResponse, tags=["Memory"])
def consolidate_memories(
    request: ConsolidateRequest,
    bot_id: str = Depends(get_effective_bot_id),
    client=Depends(require_memory_client),
):
    """Find and merge redundant memories."""
    try:
        result = client.consolidate_memories(
            dry_run=request.dry_run,
            similarity_threshold=request.similarity_threshold,
        )

        return ConsolidateResponse(
            success=True,
            dry_run=bool(result.get("dry_run", request.dry_run)),
            clusters_found=int(result.get("clusters_found", 0)),
            clusters_merged=int(result.get("clusters_merged", 0)),
            memories_consolidated=int(result.get("memories_consolidated", 0)),
            new_memories_created=int(result.get("new_memories_created", 0)),
            errors=list(result.get("errors", [])),
            message=(
                f"{'Would merge' if request.dry_run else 'Merged'} "
                f"{int(result.get('clusters_merged', 0))} clusters"
            ),
        )
    except Exception as e:
        log.error(f"Failed to consolidate memories for bot '{bot_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e))
