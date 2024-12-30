# -*- coding: utf-8 -*-
# @Time : 2024/12/28 13:02
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : vectorstores.py
# @Project : ai_qiniu_chatbot
import traceback

import requests, json


class BaseVectorstore():
    pass


class MyFaissVectorstore(BaseVectorstore):
    headers = {'Content-Type': 'application/json'}

    def __init__(self, url: str = 'http://127.0.0.1:9012'):
        self.url = url

    def add(self, index_name: str, texts: list[str], embeddings: list[list[float]], metadatas: list[dict]):
        # 传入的参数组成json类型的data
        data = {
            "index_name": index_name,
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas
        }
        # 发送post请求
        try:
            response = requests.post(self.url + '/add', headers=self.headers, json=data).json()
            ids = response['result']
            flag = True
        except:
            ids = []
            flag = False
        return flag, ids

    def search(self, index_name: str, embedding: list[float], top_k: int = 10, filter: dict = None):
        data = {
            "index_name": index_name,
            "embedding": embedding,
            "top_k": top_k,
            "filter": filter
        }
        response = requests.post(self.url + '/search', headers=self.headers, json=data).json()
        return response

    def delete_by_ids(self, ids: list[str], index_name: str):
        data = {
            "ids": ids,
            "index_name": index_name
        }
        try:
            response = requests.post(self.url + '/delete_by_id', headers=self.headers, json=data).json()
            ids = response['ids']
            flag = True
        except:
            ids = []
            flag = False
        return flag ,ids


    def delete_by_search(self, index_name: str,  filter: dict = None):
        data = {
            "index_name": index_name,
            "filter": filter
        }
        try:
            response = requests.post(self.url + '/delete_by_search', headers=self.headers, json=data).json()
            msg, flag = response['msg'], True
        except:
            msg, flag = 'fail', False
        return flag, msg

    def get_all_docs(self, index_name: str, filter: dict = None):
        data = {
            "index_name": index_name,
            "filter": filter
        }
        try:
            response = requests.post(self.url + '/get_all_docs', headers=self.headers, json=data).json()
            result, flag = response['result'], True
        except:
            result, flag = [], False
        return flag, result

if __name__ == '__main__':
    faiss_instance = MyFaissVectorstore()
    print(faiss_instance.get_all_docs("prod"))