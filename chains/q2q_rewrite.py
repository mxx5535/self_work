# -*- coding: utf-8 -*-
# @Time : 2024/12/28 15:51
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : q2q_rewrite.py
# @Project : ai_qiniu_chatbot
import time
from .base import QRetrievalChain
from vectorstores.vectorstores import BaseVectorstore
from embeddings.embeddings import BaseEmbedding
from chats.rewrite import BaseRewriter

class QRetrievalChainWithRewrite(QRetrievalChain):
    def __init__(self, vectorstore: BaseVectorstore, embedding: BaseEmbedding, rewriter: BaseRewriter):
        super().__init__(vectorstore, embedding)
        self.rewriter = rewriter

    def _invoke(self, *args, **kwargs):
        index_name = kwargs['index_name']
        query = kwargs['query']
        filter = kwargs.get('filter')
        top_k = kwargs.get('top_k')

        # rewrite
        rewrite_queries = self.rewriter.rewrite(query)

        # retrieval
        retrieval_questions = []
        retrieval_details = []
        for rewrite_query in rewrite_queries:
            query_embedding = self.embedding.embed_document(rewrite_query)
            retrieval_result = self.vectorstore.search(index_name=index_name, embedding=query_embedding, top_k=top_k or 10,
                                             filter=filter)

            for i in retrieval_result['result']:
                retrieval_questions.append(i['meta_data']['prompt'])
                retrieval_details.append([i, i['score'], rewrite_query])

        # rerank
        retrieval_details = sorted(retrieval_details, key = lambda x : x[1])
        return retrieval_details
