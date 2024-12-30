# -*- coding: utf-8 -*-
# @Time : 2024/12/28 16:31
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : chain_test.py
# @Project : ai_qiniu_chatbot
from chats.azure_chat import AzureChatOpenAI
from chats.rewrite import BaseRewriter
from chains.base import QRetrievalChain
from embeddings.embeddings import BGEEmbedding
from vectorstores.vectorstores import MyFaissVectorstore
from chains.q2q_rewrite import QRetrievalChainWithRewrite
from prompt_templates.geralprompt import GeneratePrompt
from risk.rule_risk import KeyWordFilter
from chains.rag_chain import RAGChain


def q_retrieval_chain():
    faiss_vectorstore = MyFaissVectorstore()
    bge_embedding = BGEEmbedding()
    q_retrieval_chain = QRetrievalChain(faiss_vectorstore, bge_embedding)
    kwargs = {
        'index_name': 'qiniu_20241228',
        'query': '老师怎么上课呀',
        'top_k': 10,
        'filter': None
    }
    result = q_retrieval_chain.invoke(**kwargs)
    print(result)

def q_retrieval_chain_with_rewrite():
    chat = AzureChatOpenAI()
    faiss_vectorstore = MyFaissVectorstore()
    bge_embedding = BGEEmbedding()
    rewriter = BaseRewriter(chat)
    q_retrieval_chain = QRetrievalChainWithRewrite(faiss_vectorstore, bge_embedding, rewriter)
    kwargs = {
        'index_name': 'qiniu_20241228',
        'query': '老师有回放吗？',
        'top_k': 10,
        'filter': None
    }
    rerank_result = q_retrieval_chain.invoke(**kwargs)
    return rerank_result

def rag_chain(question):
    # 传入参数
    kwargs = {
        'index_name': 'qiniu_20241228',
        'query': question,
        'top_k': 10,
        'filter': None,
        'prompt_kwargs': {
            'question': question,  # query
        },
        'sensitive_words': ['骗钱']
    }

    # 执行
    chat = AzureChatOpenAI()
    faiss_vectorstore = MyFaissVectorstore()
    bge_embedding = BGEEmbedding()
    generate_prompt = GeneratePrompt()
    content_filter = KeyWordFilter()
    rag_chain = RAGChain(faiss_vectorstore, bge_embedding, chat, content_filter, generate_prompt)

    result = rag_chain.invoke(**kwargs)
    print('请求')
    print(kwargs)
    print('-*-' * 20)
    print('回复')
    print(result)


if __name__ == '__main__':
    rag_chain(question = "老师有回放吗？")
