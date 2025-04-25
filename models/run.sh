#!/bin/bash

# Start Triton Server in the background
docker run --rm --gpus='device=5' --name ocr-document-temp -d -it --ipc=host -p 5002:8000 -p 5003:8001 -p 5004:8002 -v /raid/new_OCR/triton_pipeline/document_tr_new:/models  nvcr.io/nvidia/tritonserver:22.12-py3 bash -c "export PYTHONIOENCODING=UTF-8-SIG  && tritonserver --model-repository=/models   --strict-model-config=false   --model-control-mode=poll   --repository-poll-secs=10   --backend-config=tensorflow,version=2  --log-verbose=1 --exit-on-error=false"

# Wait for the server to be up (this might need adjustment)
sleep 30  # Waits 30 seconds. Adjust this based on how long your server typically takes to start.

# Run warm-up script
python3 /raid/new_OCR/triton_pipeline/document_tr_new/warmup.py

docker logs -f ocr-document-temp