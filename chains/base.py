# -*- coding: utf-8 -*-
# @Time : 2024/12/28 15:33
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : base.py
# @Project : ai_qiniu_chatbot
import time
from vectorstores.vectorstores import BaseVectorstore
from embeddings.embeddings import BaseEmbedding

class BaseChain:
    def invoke(self, *args, **kwargs):
        return self._invoke(*args, **kwargs)

    def _invoke(self, *args, **kwargs):
        raise NotImplementedError

# qq检索
class QRetrievalChain(BaseChain):
    def __init__(self, vectorstore: BaseVectorstore, embedding: BaseEmbedding):
        self.vectorstore = vectorstore
        self.embedding = embedding

    def _invoke(self, *args, **kwargs):
        index_name = kwargs['index_name']
        query = kwargs['query']
        filter = kwargs.get('filter')
        top_k = kwargs.get('top_k')

        start = time.time()
        query_embedding = self.embedding.embed_document(query)
        print('embedding time_cost_s:', time.time() - start)
        print(query_embedding)
        result = self.vectorstore.search(index_name=index_name, embedding=query_embedding, top_k=top_k or 10, filter=filter)
        print('faiss time_cost_s:', time.time() - start)
        return result
