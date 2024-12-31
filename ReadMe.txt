conda activate faiss
nohup python app.py >> /www/logs/faiss_server/main_logger.log 2>&1 &
tail -f /www/logs/faiss_server/main_logger.log

conda activate faiss
nohup python bge_reranker.py >> ./bge_reranker.log 2>&1 &
curl --location --request POST 'http://127.0.0.1:5000/v1/rerank' --header 'Content-Type: application/json' --data '[["有回放吗?", "今天有回放吗？"], ["有回放吗？", "今天上课吗？"]]'

conda activate faiss
nohup python bge_embedding.py >> ./bge_embedding.log 2>&1 &
curl --location --request POST 'http://127.0.0.1:50072/embedding' --header 'Content-Type: application/json' --data '{"text": "你好"}'

conda activate ai_qiniu_chatbot
nohup litellm --model ollama/qwen2.5:7b >> ./litellm.log 2>&1 &
curl http://localhost:4000/v1/chat/completions   -H "Content-Type: application/json"   -d '{"model": "qwen2.5:7b", "messages": [{"role": "user", "content": "Hello! What is your name?"}]}'
