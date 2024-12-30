# -*- coding: utf-8 -*-
# @Time : 2024/12/29 17:13
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : rag_tools.py
# @Project : ai_qiniu_chatbot
from chats.azure_chat import AzureChatOpenAI
from embeddings.embeddings import BGEEmbedding
from vectorstores.vectorstores import MyFaissVectorstore
from prompt_templates.geralprompt import GeneratePrompt
from risk.rule_risk import KeyWordFilter
from chains.rag_chain import RAGChain
from typing import Annotated, TypedDict

class RAG(TypedDict):
    llms_result: str

RAGType = Annotated[RAG, "A dictionary representing the llms result based on retrival argumented generation"]

def rag_tool(question: Annotated[str, "the question"]) -> RAGType:
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
    return result + "TERMINATE"