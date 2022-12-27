"""Flask service."""
from flask_cors import CORS
import flask
import logging
import coloredlogs
import copy
import redis
import msgpack
import uuid
import time
from multi_document_mrc.service.config import (
    INTERNAL_REDIS_HOST,
    INTERNAL_REDIS_PORT,
    INTERNAL_REDIS_DB,
    INTERNAL_REDIS_PW,
    SERVICE_PORT,
    SERVICE_HOST,
    QUEUE_KEY,
    OUTPUT_KEY,
    REQUEST_TIMEOUT,
    SLEEP_BETWEEN_PUSH_GET,
    MODEL_VERSION
)

# Thời gian nghỉ giữa các lần đọc kết quả từ redis
SLEEP_TIME = 0.025

coloredlogs.install(
    level="INFO", fmt="%(asctime)s %(name)s[%(process)d] %(levelname)-8s %(message)s"
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# init redis client
INTERNAL_REDIS_CLIENT = redis.Redis(
    host=INTERNAL_REDIS_HOST,
    port=int(INTERNAL_REDIS_PORT),
    db=int(INTERNAL_REDIS_DB),
    password=INTERNAL_REDIS_PW,
)

# init flask
app = flask.Flask(__name__)
CORS(app)
RESPONSE = {
    "status": "success",
    "code": 200,
    "knowledge_based_response": "",
    "version": MODEL_VERSION
}


def get_outputs_from_redis(
    redis_db, redis_key, timeout=int(REQUEST_TIMEOUT), sleep=SLEEP_TIME
):
    """Get text from redis.

    Args:
        redis_db (redis.Redis): redis client
        redis_key (str): key to get the result from
        timeout (float): threshold to stop polling the result
        sleep (float):

    Returns:
        str: text
    """
    start_time = time.time()
    while True:
        last_time = time.time() - start_time
        output = redis_db.get(redis_key)
        if output is not None:
            # deserialize embedding
            result = msgpack.loads(output, encoding="utf-8")
            # delete the result from the database and break from the polling loop
            redis_db.delete(redis_key)
            break
        elif last_time > timeout:
            result = None
            logger.error(f"Can't get key {redis_key} from redis.")
            break
        time.sleep(sleep)
    return result


def push_message_to_redis(redis_db, redis_key, knowledge_question):
    """Push message to redis.

    Args:
        redis_db (redis.Redis): redis client
        redis_key (str): key of the queue
        dialog_history (list): list of text

    Returns:
        str: id of message pushed to queue

    """
    message_id = str(uuid.uuid4())
    sample = {"id": message_id, "data": knowledge_question}
    redis_db.rpush(redis_key, msgpack.dumps(sample))

    return message_id


def get_model_response(knowledge_question: dict):
    """Push dialog_history to redis and get response."""
    message_id = push_message_to_redis(INTERNAL_REDIS_CLIENT, QUEUE_KEY, knowledge_question)
    time.sleep(float(SLEEP_BETWEEN_PUSH_GET))
    response = get_outputs_from_redis(INTERNAL_REDIS_CLIENT, OUTPUT_KEY + message_id)
    return response


@app.route("/knowledge_grounded_response", methods=["POST"])
def model_predict():
    """Push text in Redis, after get ouput from redis."""
    # Get params from request
    res = copy.deepcopy(RESPONSE)
    data = flask.request.get_json(force=True)
    try:
        knowledge_question = data.get("data")
        sender_id = data.get("sender_id")
        bot_id = data.get("bot_id")
        res.update(bot_id=bot_id)
        res.update(sender_id=sender_id)
    except Exception:
        logger.exception(
            f"Knowledge grounded response generator: False to get data in request {data}"
        )
        res.update(response="")
        return flask.jsonify(res)

    try:
        response = get_model_response(knowledge_question)
    except Exception:
        response = ""
        logger.exception(
            "Can't generate reponse due to neural response generator service."
        )
    res.update(knowledge_based_response=response)
    return flask.jsonify(res)


if __name__ == "__main__":
    app.run(host=str(SERVICE_HOST), port=int(SERVICE_PORT))
