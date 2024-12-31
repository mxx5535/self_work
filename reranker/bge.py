# -*- coding: utf-8 -*-
# @Time : 2024/12/28 15:39
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : bge.py
# @Project : ai_qiniu_chatbot
import requests, json
from .base import BaseReranker

class BGEReranker(BaseReranker):
    def _rerank(self, query: str, docs: list[str], doc_details: list[dict], top_k: int = 10) -> tuple[list[str], list[float]]:
        pairs = [[query, doc] for doc in docs]
        headers = {'Content-Type': 'application/json'}
        rerank_scores = requests.post('http://127.0.0.1:5000/v1/rerank', data=json.dumps(pairs), headers=headers).json()['pairs_score']

        rerank_doc_details = []
        for doc_detail, score in zip(doc_details, rerank_scores):
            doc_detail['rerank_score'] = score
            rerank_doc_details.append(doc_detail)
        rerank_doc_details.sort(key=lambda x: x['rerank_score'], reverse=True)

        return rerank_doc_details[:top_k]