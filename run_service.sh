#!/bin/bash
source activate vnpt
redis-server --port 6380 &
python multi_document_mrc/service/start_batch_extractor.py &
gunicorn -c multi_document_mrc/service/gunicorn_conf.py multi_document_mrc.service.wsgi_service:app
