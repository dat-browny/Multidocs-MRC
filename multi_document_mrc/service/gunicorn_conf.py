"""file gunicorn.conf.py."""
# coding=utf-8
import os
import multiprocessing
from multi_document_mrc.service.config import SERVICE_HOST, SERVICE_PORT

loglevel = "info"
# errorlog = os.path.join(_VAR, 'log/api-error.log')
# accesslog = os.path.join(_VAR, 'log/api-access.log')
errorlog = "-"
accesslog = "-"

# bind = 'unix:%s' % os.path.join(_VAR, 'run/gunicorn.sock')
bind = str(SERVICE_HOST) + ":" + str(SERVICE_PORT)
workers = os.environ.get("WORKERS") or multiprocessing.cpu_count() * 2 + 1

timeout = 3 * 60  # 3 minutes
keepalive = 24 * 60 * 60  # 1 day

capture_output = True
