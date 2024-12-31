# -*- coding: utf-8 -*-
# @Time : 2024/12/30 21:29
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : rag_tool_test.py
# @Project : self_work
from chats.azure_chat import AzureChatOpenAI
from embeddings.embeddings import BGEEmbedding
from vectorstores.vectorstores import MyFaissVectorstore
from prompt_templates.geralprompt import GeneratePrompt
from risk.rule_risk import KeyWordFilter
from chains.rag_chain import RAGChain
from reranker.bge import BGEReranker
from chats.rewrite import BaseRewriter
from chains.rag_rewrite_rerank_chain import RAGChainWithRewriteRerank

def rag_tool(query: str):
    '''
    其他问题通过rag调用回答，具体实现逻辑省略
    :param query:
    :return:
    '''
    print("query:", query)
    kwargs = {
        'index_name': 'qiniu_20241228',
        'query': query,
        'top_k': 10,
        'filter': None,
        'prompt_kwargs': {
            'question': query,  # query
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

    print("result:", result)

    return result

def test_rag_chain_with_rewrite_rerank(query: str):
    # 传入参数
    kwargs = {
        'index_name': 'qiniu_20241228',
        'query': query,
        'top_k': 10,
        'filter': None,
        'prompt_kwargs': {
            'question': query,  # query
            'camp_date_start_time': '后天',
            'current_day_str': '2月25日'
        },
        'sensitive_words': ['骗钱']
    }

    # 执行
    chat = AzureChatOpenAI()
    faiss_vectorstore = MyFaissVectorstore()
    bge_embedding = BGEEmbedding()
    generate_prompt = GeneratePrompt()
    content_filter = KeyWordFilter()
    rewriter = BaseRewriter(chat)
    reranker = BGEReranker()
    rag_chain_with_rewrite_rerank = RAGChainWithRewriteRerank(faiss_vectorstore, bge_embedding, chat, content_filter, generate_prompt, rewriter, reranker)

    result = rag_chain_with_rewrite_rerank.invoke(**kwargs)
    print('请求')
    print(kwargs)
    print('-*-' * 20)
    print('回复')
    print(result)


if __name__ == '__main__':
    test_rag_chain_with_rewrite_rerank("有回放吗？")