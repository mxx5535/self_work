# -*- coding: utf-8 -*-
# @Time : 2024/12/28 15:38
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : base.py
# @Project : ai_qiniu_chatbot
import requests


class BaseReranker:
    def rerank(self,  query: str, docs: list[str], doc_details: list[dict], top_k: int = 10) -> tuple[
        list[str], list[float]]:
        if len(docs)==0:
            return []
        return self._rerank(query, docs, doc_details, top_k)

