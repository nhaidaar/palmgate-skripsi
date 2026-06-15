import logging


log = logging.getLogger("palmgate")


def match_embedding_and_log(palm_processor, db, embedding, threshold, duration_ms: int | None = None):
    stored = db.get_all_embeddings()
    result = palm_processor.compute_similarity(embedding, stored, threshold)
    description = None
    if result["status"] == "DENIED" and result.get("closest_match"):
        description = f"similar to {result['closest_match']}"
    db.add_access_log(
        user_id=result["user_id"],
        matched_name=result["name"] if result["status"] == "ALLOWED" else "Unknown",
        status=result["status"],
        similarity=result["similarity"],
        duration_ms=duration_ms,
        description=description,
    )
    log.info(
        "%s | user=%s | similarity=%.4f",
        result["status"],
        result["name"],
        result["similarity"],
    )
    return result
