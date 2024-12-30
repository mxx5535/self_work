# -*- coding: utf-8 -*-
# @Time : 2024/12/28 15:56
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : q2q_rewrite_rerank.py
# @Project : ai_qiniu_chatbot
# qq检索加入重写和重排
from .base import QRetrievalChain
from vectorstores.vectorstores import BaseVectorstore
from embeddings.embeddings import BaseEmbedding
from reranker.bge import BaseReranker
from chats.rewrite import BaseRewriter

class QRetrievalChainWithRewriteRerank(QRetrievalChain):
    def __init__(self, vectorstore: BaseVectorstore, embedding: BaseEmbedding, rewriter: BaseRewriter, reranker: BaseReranker):
        super().__init__(vectorstore, embedding)
        self.rewriter = rewriter
        self.reranker = reranker

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
            query_embedding = self.embedding.embed_documents([rewrite_query])[0]
            retrieval_result = self.vectorstore.search(index_name=index_name, embedding=query_embedding, top_k=top_k or 10,
                                             filter=filter)

            for i in retrieval_result['result']:
                retrieval_questions.append(i['meta_data']['question'])
                retrieval_details.append(i)

        # rerank
        reranked_results = self.reranker.rerank(query, docs=retrieval_questions, doc_details=retrieval_details, top_k=10)
        return retrieval_details, reranked_results