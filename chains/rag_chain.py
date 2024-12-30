# -*- coding: utf-8 -*-
# @Time : 2024/12/28 19:05
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : rag_chain.py
# @Project : ai_qiniu_chatbot
# rag标准流程
from .base import QRetrievalChain
from vectorstores.vectorstores import BaseVectorstore
from embeddings.embeddings import BaseEmbedding
from reranker.base import BaseReranker
from chats.rewrite import BaseRewriter
from risk.base import BaseContentFilter
from chats.base import BaseChat
from prompt_templates.geralprompt import BasePrompt

class RAGChain(QRetrievalChain):
    def __init__(self, vectorstore: BaseVectorstore, embedding: BaseEmbedding, chat: BaseChat, content_filter: BaseContentFilter, prompt_template:BasePrompt):
        self.chat = chat
        self.vectorstore = vectorstore
        self.embedding = embedding
        self.content_filter = content_filter
        self.prompt_template = prompt_template

    def _invoke(self, *args, **kwargs):
        index_name = kwargs['index_name']
        query = kwargs['query']
        filter = kwargs.get('filter')
        top_k = kwargs.get('top_k')
        prompt_kwargs = kwargs['prompt_kwargs']
        sensitive_words = kwargs['sensitive_words']

        # retrieval
        query_embedding = self.embedding.embed_document(query)
        retrieval_results = self.vectorstore.search(index_name=index_name, embedding=query_embedding, top_k=top_k or 10, filter=filter)
        context = '\n'.join([i['meta_data']['answer'] for i in retrieval_results['result']])

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




