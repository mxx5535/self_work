# -*- coding: utf-8 -*-
# @Time : 2024/12/28 16:06
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : insert_data_into_faiss.py
# @Project : ai_qiniu_chatbot

from faiss_tool.faiss_tool import FAISS_TOOL
from embeddings.embeddings import BGEEmbedding

import traceback
import pandas as pd

embedding_url = "http://127.0.0.1:50072/embedding"
faiss_server_url = 'http://127.0.0.1:9012/'
embedding_instance = BGEEmbedding(embedding_url)
faiss_server_instance = FAISS_TOOL(faiss_server_url)

data = pd.read_csv('../data/启牛话术.csv')

for index, row in data.iloc[1018:].iterrows():
    try:
        topic, question, answer = row['type'], row['question'], row['answer']
        doc_ids = faiss_server_instance.doc_add(prompt=question, embedding_instance=embedding_instance, index_name='qiniu_20241228', metadata={'topic': topic, 'answer': answer})
        print(f'{index} - {doc_ids}')
    except:
        print(traceback.format_exc())

