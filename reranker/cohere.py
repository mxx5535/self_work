# -*- coding: utf-8 -*-
# @Time : 2024/12/28 15:39
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : cohere.py
# @Project : ai_qiniu_chatbot
from .base import BaseReranker

class CohereReranker(BaseReranker):
    def _rerank(self, query: str, docs: list[str], ids: list[float] = [], top_n: int = 10) -> tuple[list[str], list[float]]:
        raise Exception('no apikey')

        # Get your cohere API key on: www.cohere.com
        co = cohere.Client("{apiKey}")

        rerank_hits = co.rerank(query=query, documents=docs,
                                model='rerank-multilingual-v2.0', top_n=top_n)
        return [hit.document for hit in rerank_hits][:top_n], [hit.relevance_score for hit in rerank_hits][:top_n]