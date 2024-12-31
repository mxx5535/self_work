# -*- coding: utf-8 -*-
# @Time : 2024/12/30 21:23
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : rag_rewrite_rerank_chain.py
# @Project : self_work
from .base import QRetrievalChain
from vectorstores.vectorstores import BaseVectorstore
from embeddings.embeddings import BaseEmbedding
from reranker.base import BaseReranker
from chats.rewrite import BaseRewriter
from risk.base import BaseContentFilter
from chats.base import BaseChat
from prompt_templates.geralprompt import BasePrompt
from .rag_chain import RAGChain

# rag中加入重写和重排
class RAGChainWithRewriteRerank(RAGChain):
    def __init__(self, vectorstore: BaseVectorstore, embedding: BaseEmbedding, chat: BaseChat, content_filter: BaseContentFilter, prompt_template:BasePrompt, rewriter: BaseRewriter, reranker: BaseReranker):
        super().__init__(vectorstore, embedding, chat, content_filter, prompt_template)
        self.rewriter = rewriter
        self.reranker = reranker

    def _invoke(self, *args, **kwargs):
        index_name = kwargs['index_name']
        query = kwargs['query']
        filter = kwargs.get('filter')
        top_k = kwargs.get('top_k')
        prompt_kwargs = kwargs['prompt_kwargs']
        sensitive_words = kwargs['sensitive_words']

        # rewrite
        rewrite_queries = self.rewriter.rewrite(query)

        # retrieval
        retrieval_questions = []
        retrieval_details = []
        for rewrite_query in rewrite_queries:
            query_embedding = self.embedding.embed_document(rewrite_query)
            retrieval_result = self.vectorstore.search(index_name=index_name, embedding=query_embedding,
                                                       top_k=top_k or 10,
                                                       filter=filter)

            for i in retrieval_result['result']:
                retrieval_questions.append(i['meta_data']['prompt'])
                retrieval_details.append(i)

        # rerank
        reranked_results = self.reranker.rerank(query, docs=retrieval_questions, doc_details=retrieval_details, top_k=10)

        context = '\n'.join(set([i['meta_data']['answer'] for i in reranked_results]))

        # chat
        # prompt组装
        prompt_kwargs['context'] = context
        prompt = self.prompt_template.get_prompt(**prompt_kwargs)
        chat_results = self.chat.chat(prompt)

        # filter
        filtered_results = self.content_filter.judge(sensitive_words, [chat_results])
        if not all(filtered_results):
            return chat_results
        else:
            return '违规内容已被过滤'
