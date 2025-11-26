Notification Classifier — Quick server

You need to download `model/model.safetensors` from Release.

Quick start (run locally):

1) Create and activate your environment, then install deps:

```bash
cd /home/nova/classifier
pip install -r requirements.txt
```

2) Start the FastAPI server (uses GPU if available):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

3) Test the server with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"texts": ["YOUR TEXT", "YOUR TEXT"]}'
```
