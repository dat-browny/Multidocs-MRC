"""Config."""
import os
import yaml
from multi_document_mrc import get_root_path


ROOT_PATH = get_root_path()
CONFIG_PATH = os.path.join(ROOT_PATH, "service", "config_service.yml")
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# init config
SERVICE_PORT = os.environ.get("SERVICE_PORT") or config["SERVICE_PORT"]
SERVICE_HOST = os.environ.get("SERVICE_HOST") or config["SERVICE_HOST"]

# Config
# internal container redis db
INTERNAL_REDIS_HOST = os.environ.get("INTERNAL_REDIS_HOST") or config.get(
    "INTERNAL_REDIS_HOST"
)
INTERNAL_REDIS_PORT = os.environ.get("INTERNAL_REDIS_PORT") or config.get(
    "INTERNAL_REDIS_PORT"
)
INTERNAL_REDIS_DB = os.environ.get("INTERNAL_REDIS_DB") or config.get(
    "INTERNAL_REDIS_DB"
)
INTERNAL_REDIS_PW = os.environ.get("INTERNAL_REDIS_PW") or config.get(
    "INTERNAL_REDIS_PW"
)
# external container redis db
REDIS_HOST = os.environ.get("REDIS_HOST") or config.get("REDIS_HOST")
REDIS_PORT = os.environ.get("REDIS_PORT") or config.get("REDIS_PORT")
REDIS_DB = os.environ.get("REDIS_DB") or config.get("REDIS_DB")
REDIS_PW = os.environ.get("REDIS_PW") or config.get("REDIS_PW")

QUEUE_KEY = os.environ.get("QUEUE_KEY") or config.get("QUEUE_KEY")
OUTPUT_KEY = os.environ.get("OUTPUT_KEY") or config.get("OUTPUT_KEY")
OUTPUT_TIMEOUT = os.environ.get("OUTPUT_TIMEOUT") or config.get("OUTPUT_TIMEOUT")
OUTPUT_TIMEOUT = int(OUTPUT_TIMEOUT)
BATCH_SIZE = os.environ.get("BATCH_SIZE") or config.get("BATCH_SIZE")
BATCH_SIZE = int(BATCH_SIZE)
GPU_MEM_LIMIT = os.environ.get("GPU_MEM_LIMIT") or config.get("GPU_MEM_LIMIT")
GPU_MEM_LIMIT = int(GPU_MEM_LIMIT)
MODEL_PATH = os.environ.get("MODEL_PATH") or config.get("MODEL_PATH")
DEFAULT_THRESHOLD = os.environ.get("DEFAULT_THRESHOLD") or config.get(
    "DEFAULT_THRESHOLD"
)
DEFAULT_THRESHOLD = float(DEFAULT_THRESHOLD)
MAX_SEQ_LENGTH = os.environ.get("MAX_SEQ_LENGTH") or config.get("MAX_SEQ_LENGTH")
MAX_SEQ_LENGTH = int(MAX_SEQ_LENGTH)
SLEEP_BETWEEN_PUSH_GET = os.environ.get("SLEEP_BETWEEN_PUSH_GET") or config.get(
    "SLEEP_BETWEEN_PUSH_GET"
)
SLEEP_BETWEEN_PUSH_GET = float(SLEEP_BETWEEN_PUSH_GET)
REQUEST_TIMEOUT = os.environ.get("REQUEST_TIMEOUT") or config.get("REQUEST_TIMEOUT")
REQUEST_TIMEOUT = int(REQUEST_TIMEOUT)
BREAK_TIME_BETWEEN_2_CALLS = os.environ.get("BREAK_TIME_BETWEEN_2_CALLS") or config.get("BREAK_TIME_BETWEEN_2_CALLS")
BREAK_TIME_BETWEEN_2_CALLS = float(BREAK_TIME_BETWEEN_2_CALLS)
MAX_KNOWLEDGES = os.environ.get("MAX_KNOWLEDGES") or config.get("MAX_KNOWLEDGES")
MAX_KNOWLEDGES = int(MAX_KNOWLEDGES)
MODEL_VERSION = os.environ.get("MODEL_VERSION") or config.get("MODEL_VERSION")
