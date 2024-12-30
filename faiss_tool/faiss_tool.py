# -*- coding: utf-8 -*-
# @Time : 2024/12/28 13:59
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : faiss_tool.py
# @Project : ai_qiniu_chatbot
# -*- coding: utf-8 -*-
# @Time : 2024/3/27 17:05
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : faiss_tool.py
# @Project : ai_faiss_server
import sys
sys.path.append('..')

from typing import (
    Dict,
    List,
)
import json, requests, logging
from embeddings.embeddings import BGEEmbedding
from vectorstores.vectorstores import MyFaissVectorstore

header = {"Content-Type": "application/json"}
full_logger = logging.getLogger('logger')


class FAISS_TOOL(object):
    def __init__(self, url='http://172.16.13.181:5000/'):
        self.db = MyFaissVectorstore(url)

    def doc_add(self, prompt: str, embedding_instance, index_name: str, metadata: Dict = None):
        question_embedding = embedding_instance._embed_document(prompt)
        if not question_embedding:
            raise Exception("emeddbing 服务报错！")
        if metadata:
            metadata.update({"prompt": prompt})
            metadatas = [metadata]
        else:
            metadatas = [{"prompt": prompt}]
        q_add_flag, question_ids = self.db.add(texts=[prompt], embeddings=[question_embedding], metadatas=metadatas, index_name=index_name)
        if not q_add_flag:
            raise Exception(f"faiss的add服务出错，出错原因：{question_ids}")
        return question_ids

    def doc_search(self, prompt: str, embedding_instance, index_name: str = "test", filter: Dict = None, k: int = 3):
        doc_search_list = []
        question_embedding = embedding_instance._embed_document(prompt)
        if not question_embedding:
            raise Exception(f'请检查embedding接口报错问题！')
        search_flag, response_result_list = self.db.search(embedding=question_embedding, filter=filter, top_k = k, index_name=index_name)
        if search_flag:
            for result in response_result_list:
                doc_search_list.append([result['page_content'], result['meta_data'], result['score']])
            doc_search_list = sorted(doc_search_list, key = lambda x : x[2])
            return doc_search_list
        else:
            raise Exception(f'请检查faiss的search服务出错，出错原因：{response_result_list}')

    def delete_by_ids(self, ids, index_name: str):
        flag, ids = self.db.delete_by_ids(ids = ids, index_name = index_name)
        if flag:
            return 'SUCCESS'
        else:
            return 'FAIL'

    def get_docs_by_search(self, index_name: str, doc_num: int = None, filter = None):
        result_flag, docs = self.db.get_all_docs(filter = filter, index_name = index_name)
        if result_flag:
            return docs[:doc_num]
        else:
            raise Exception("get_docs_by_search出错！")

    def delete_by_search(self, index_name: str, filter: Dict = None):
        flag, msg = self.db.delete_by_search(index_name=index_name, filter=filter)
        if flag:
            return 'SUCCESS'
        else:
            return 'FAIL'


    def update_by_id(self, id: str, embedding_instance, index_name: str = "test", new_prompt: str = "new_prompt", new_metadata: Dict = None):
        delete_result = self.delete_by_ids([id], index_name = index_name)
        if delete_result == 'SUCCESS':
            new_ids = self.doc_add(prompt=new_prompt, metadata=new_metadata, index_name=index_name, embedding_instance = embedding_instance)
            return new_ids
        else:
            raise Exception(f"update失败，错误原因：{delete_result}")

if __name__ == '__main__':
    embedding_url = "http://127.0.0.1:50072/embedding"
    faiss_server_url = 'http://127.0.0.1:9012/'
    embedding_instance = BGEEmbedding(embedding_url)
    faiss_server_instance = FAISS_TOOL(faiss_server_url)
    '''
        1. 增加文档(没有metadata)
    '''
    # doc_ids = faiss_server_instance.doc_add(prompt="这是测试样本7", embedding_instance=embedding_instance, index_name='test_ljz')
    '''
        2. 增加文档(有metadata)
    '''
    doc_ids = faiss_server_instance.doc_add(prompt="这是测试样本20241228", embedding_instance=embedding_instance, index_name='ljz_20241228', metadata={'author': 'ljz_20241228'})
    all_docs = faiss_server_instance.get_docs_by_search(index_name="ljz_20241228", doc_num=10)
    print(all_docs)
    '''
        3. 删除文档(根据文档id)
    '''
    # delete_result = faiss_server_instance.delete_by_ids(ids = ['f0d3ee9b-9163-4e69-b4f6-6397156653a5'], index_name= 'test_ljz')
    '''
        4. 删除文档(根据条件filter)
    '''
    delete_result = faiss_server_instance.delete_by_search(index_name="ljz_20241228", filter={'author': 'ljz_20241228'})
    print(delete_result)
    '''
        5. 搜索文档(all)
    '''
    all_docs = faiss_server_instance.get_docs_by_search(index_name="ljz_20241228", doc_num=10)
    print(all_docs)
    '''
        6. 搜索文档(根据条件filter)
    '''
    # filter_docs = faiss_server_instance.get_docs_by_search(index_name="ljz_20241228", doc_num=10, filter={"author": "ljz_20241228"})
    filter_docs = faiss_server_instance.get_docs_by_search(index_name="ljz_20241228", doc_num=10, filter={"author": "ljz"})
    print(filter_docs)
    '''
        7. 更新文档(根据文档id)
    '''
    # new_doc_ids = faiss_server_instance.update_by_id(id = 'f0d3ee9b-9163-4e69-b4f6-6397156653a5', index_name="test_ljz", new_prompt="测试用例11", new_metadata={"author": 'ljz'}, embedding_instance=embedding_instance)