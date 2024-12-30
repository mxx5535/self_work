# -*- coding: utf-8 -*-
# @Time : 2024/12/28 13:19
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : embeddings.py
# @Project : ai_qiniu_chatbot
import requests

class BaseEmbedding():
    def embed_document(self, text: str) -> list[float]:
        return self._embed_document(text)

    def _embed_document(self, text: str):
        raise NotImplementedError

class BGEEmbedding(BaseEmbedding):
    def __init__(self, url: str = 'http://127.0.0.1:50072/embedding'):
        self.url = url

    def _embed_document(self, text: str) -> list[float]:
        # 发送post请求
        response = requests.post(self.url, json={"text": text}).json()
        return response['embedding']

if __name__ == '__main__':
    instance_embedding = BGEEmbedding()
    result = instance_embedding.embed_document(text = '你好')
    print(result)