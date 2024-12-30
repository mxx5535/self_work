conda activate faiss
nohup python app.py >> /www/logs/faiss_server/main_logger.log 2>&1 &
tail -f /www/logs/faiss_server/main_logger.log

conda activate faiss
nohup python bge_reranker.py >> ./bge_reranker.log 2>&1 &
curl --location --request POST 'http://127.0.0.1:5000/v1/rerank' --header 'Content-Type: application/json' --data '[["有回放吗?", "今天有回放吗？"], ["有回放吗？", "今天上课吗？"]]'

conda activate faiss
nohup python bge_embedding.py >> ./bge_embedding.log 2>&1 &
curl --location --request POST 'http://127.0.0.1:50072/embedding' --header 'Content-Type: application/json' --data '{"text": "你好"}'

conda activate fastchat
nohup python -m fastchat.serve.controller --host 0.0.0.0 >> ./controller.log 2>&1 &

nohup python -m fastchat.serve.model_worker --model-names qwen2-chat --model-path /root/autodl-tmp/qwen/Qwen/Qwen2___5-7B-Instruct --host 0.0.0.0 >> ./model_worker.log 2>&1 &

nohup python -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8000 >> ./api_server.log 2>&1 &

curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{"model": "qwen2-chat", "messages": [{"role": "user", "content": "Hello! What is your name?"}]}'

