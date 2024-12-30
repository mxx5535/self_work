# -*- coding: utf-8 -*-
# @Time : 2024/12/28 15:59
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : rewrite.py
# @Project : ai_qiniu_chatbot
from .base import BaseChat


class BaseRewriter():
    def __init__(self, chat: BaseChat, prompt: str = None):
        base_prompt = '''请将原问题优化生成3个相关的搜索查询，这些查询应与原始查询相似并且是人们可能会提出的可回答的搜索问题。请勿使用任何示例中提到的内容，确保所有生成的查询均独立于示例，仅基于提供的原始查询。
    请按照以下逗号分隔的格式提供: 'queries:<queries>'，改写后的三个query用,连接 
"original_query:{query}\n"    
"queries:"'''
        self.prompt = prompt or base_prompt
        self.chat = chat

    def rewrite(self, query: str) -> list[list[str]]:
        prompt = self.prompt.format(**{'query': query})
        # rewrite query with chatbot and embeddings
        content = self.chat.chat(prompt)

        rewrite_querys = []
        for item in content.split('\n'):
            rewrite_query = item.split(':')[-1]
            rewrite_querys.extend(rewrite_query.split(','))
        # rewrite_querys.append(query)
        return rewrite_querys
