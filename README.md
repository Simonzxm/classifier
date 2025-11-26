Notification Classifier — Quick server

Files added:
- `main.py`: FastAPI app that loads `./model` and serves a `/predict` endpoint.
- `requirements.txt`: Python dependencies to run the server.

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
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"texts": ["尊敬的用户，您的账户余额将于 24 小时内到期，请及时续费。","这是来自 XX 报道的新闻摘要。"]}'
```

What I can do next (pick one or tell me to proceed):
- Package the model to ONNX for faster CPU inference and provide conversion script.
- Add a simple batching/gunicorn wrapper for production.
- Write a small test harness that sends a batch of validation texts and saves predictions to CSV.
- Nothing else — you're set to run the server locally.
