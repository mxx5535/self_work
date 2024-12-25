# -*- coding: utf-8 -*-
# @Time : 2024/12/25 13:51
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : insert_data_into_vector_db.py
# @Project : education_chatbot
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions as ef

client = chromadb.PersistentClient(path="../data/vector_db/qiniu_db/chromadb")  # 数据保存在磁盘

embedding_function = ef.OpenAIEmbeddingFunction(
                api_key="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",
                model_name="text-embedding-ada-002"
            )
collection = client.get_or_create_collection(name="qiniu_db_collection", embedding_function=embedding_function)

data = pd.read_csv('../data/origin_data/启牛话术.csv')

for index, row in data.iloc[724:803].iterrows():
    type_s, question, answer = row['type'], row['question'], row['answer']
    # 添加数据
    collection.add(
        documents=[question],
        metadatas=[{"doc_type": "public", "topic": type_s, "answer": answer}],
        ids=[f"{type_s}-{index}"]
    )
    print(f"{index} is done!")