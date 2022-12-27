"""start batch extractor."""
import time
import logging
import coloredlogs
import msgpack
import traceback
import redis
import torch
from multi_document_mrc import MultiDocMRC
from multi_document_mrc.service.config import (
    QUEUE_KEY,
    BATCH_SIZE,
    OUTPUT_KEY,
    OUTPUT_TIMEOUT,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    REDIS_PW,
    INTERNAL_REDIS_HOST,
    INTERNAL_REDIS_PORT,
    INTERNAL_REDIS_DB,
    INTERNAL_REDIS_PW,
    GPU_MEM_LIMIT,
    MODEL_PATH,
    MAX_SEQ_LENGTH,
    DEFAULT_THRESHOLD,
    MAX_KNOWLEDGES,
)

coloredlogs.install(
    level="INFO", fmt="%(asctime)s %(name)s[%(process)d] %(levelname)-8s %(message)s"
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Init redis + model
if REDIS_HOST is None:
    REDIS_CLIENT = None
else:
    REDIS_CLIENT = redis.Redis(
        host=REDIS_HOST, port=int(REDIS_PORT), db=int(REDIS_DB), password=REDIS_PW
    )

INTERNAL_REDIS_CLIENT = redis.Redis(
    host=INTERNAL_REDIS_HOST,
    port=int(INTERNAL_REDIS_PORT),
    db=int(INTERNAL_REDIS_DB),
    password=INTERNAL_REDIS_PW,
)

# Set limit GPU memory
TOTAL_GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(fraction=round((GPU_MEM_LIMIT / TOTAL_GPU_MEMORY), 2), device=0)

MODEL = MultiDocMRC(
    models_path=MODEL_PATH,
    max_seq_length=int(MAX_SEQ_LENGTH)
)


def continue_loop_extractor():
    """Batch process loop extractor."""
    logger.info("Start Knowledge grounded response extractor batch service")
    if REDIS_CLIENT is not None:
        while True:
            get_input_from_redis_and_process(REDIS_CLIENT)
            get_input_from_redis_and_process(INTERNAL_REDIS_CLIENT)
    else:
        while True:
            get_input_from_redis_and_process(INTERNAL_REDIS_CLIENT)


def get_input_from_redis_and_process(redis_db):
    """Get input from redis and process."""
    try:
        queue = redis_db.lrange(QUEUE_KEY, 0, int(BATCH_SIZE) - 1)
    except Exception:
        return
    if len(queue) == 0:
        return

    data, item_ids = [], []
    try:
        start = time.time()
        # Step 1: Batch data
        for q in queue:
            # Get queue data
            q = msgpack.loads(q, encoding="utf-8")  # Deserialize = mgspack
            item_ids.append(q.get("id"))
            # data in q is list[text]
            data.append(q.get("data"))
    except Exception as e:
        print(e)
        logger.critical(f"Knowledge grounded response extractor - Dequeue error:\n {traceback.format_exc()}")

    total_items = len(data)
    if total_items > 0:
        try:
            results = MODEL.generate_responses(
                data,
                n_contexts=MAX_KNOWLEDGES,
                threshold=DEFAULT_THRESHOLD,
                silent=False
            )
            results = [s["text"] for s in results]
            serialize_msgpack(redis_db, results, item_ids)

        except Exception as e:
            print(e)
            logger.critical(
                f"Knowledge grounded response extractor service error:\n {traceback.format_exc()}"
            )

        # Step 3: Delete processed item in queue
        redis_db.ltrim(QUEUE_KEY, total_items, -1)
        logger.info(
            f"Finish knowledge grounded response extracting {total_items} items in {time.time() - start}s"
        )

    time.sleep(0.01)
    return


def serialize_msgpack(redis_db, results, item_ids):
    """serialize_msgpack."""
    for result, item_id in zip(results, item_ids):  # Serialize = mgspack
        redis_db.set(OUTPUT_KEY + item_id, msgpack.dumps(result))
        # Set timeout cho key
        redis_db.expire(OUTPUT_KEY + item_id, int(OUTPUT_TIMEOUT))


if __name__ == "__main__":
    continue_loop_extractor()
