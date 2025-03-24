def wisdom_teller(input, user_id, pending=None):
    
    logger.info(f"Saving wisdom: {user_id}")
    namespace ="wisdoms_"+{user_id}
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    embedding_text = input
    embedding = EMBEDDINGS.embed_query(embedding_text)
    metadata = {    
        "created_at": created_at
    }
    id = convo_id
    try:
        upsert_result = index.upsert([(convo_id,embedding, metadata)], namespace=namespace)
        logger.info(f"Saved wisdom: {convo_id} - Result: {upsert_result}")
       
    except Exception as e:
        logger.error(f"Child node upsert failed: {e}")
        return False
    logger.info(f"Saved")
    return True

def wisdom_wishper(input,windows=1000):
    wisdom=""
    return wisdoms