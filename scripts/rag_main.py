# -*- coding: utf-8 -*-
# @Time : 2024/12/25 14:15
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : rag_main.py
# @Project : education_chatbot
from autogen.agentchat.contrib.vectordb.base import Document, QueryResults, VectorDB, VectorDBFactory
from autogen.agentchat.contrib.vectordb.chromadb import ChromaVectorDB
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.retrieve_utils import (
    TEXT_FORMATS,
    create_vector_db_from_dir,
    get_files_from_dir,
    query_vector_db,
    split_files_to_chunks,
)
from autogen.agentchat.contrib.vectordb.utils import (
    chroma_results_to_query_results,
    filter_results_by_distance,
    get_logger,
)
import chromadb, autogen
from chromadb.utils import embedding_functions as ef
from autogen import AssistantAgent

PROMPT_QA = """You're a retrieve augmented chatbot. You answer user's questions based on your own knowledge and the
context provided by the user.
If you can't answer the question with or without the current context, you should reply exactly `UPDATE CONTEXT`.
You must give as short an answer as possible.

User's question is: {input_question}

Context is: {input_context}
"""

class MyRetrieveUserProxyAgent(RetrieveUserProxyAgent):
    def retrieve_docs(self, problem: str, n_results: int = 20, search_string: str = "", filter: dict = {}):
        """Retrieve docs based on the given problem and assign the results to the class property `_results`.
        The retrieved docs should be type of `QueryResults` which is a list of tuples containing the document and
        the distance.

        Args:
            problem (str): the problem to be solved.
            n_results (int): the number of results to be retrieved. Default is 20.
            search_string (str): only docs that contain an exact match of this string will be retrieved. Default is "".
                Not used if the vector_db doesn't support it.

        Returns:
            None.
        """
        if isinstance(self._vector_db, VectorDB):
            if not self._collection or not self._get_or_create:
                print("Trying to create collection.")
                self._init_db()
                self._collection = True
                self._get_or_create = True

            kwargs = {}
            if hasattr(self._vector_db, "type") and self._vector_db.type == "chroma":
                kwargs["where_document"] = {"$contains": search_string} if search_string else None

            kwargs['where'] = filter
            results = self._vector_db.retrieve_docs(
                queries=[problem],
                n_results=n_results,
                collection_name=self._collection_name,
                distance_threshold=self._distance_threshold,
                **kwargs,
            )
            self._search_string = search_string
            self._results = results
            print("VectorDB returns doc_ids: ", [[r[0]["id"] for r in rr] for rr in results])
            return

    @staticmethod
    def message_generator(sender, recipient, context):
        """
        Generate an initial message with the given context for the RetrieveUserProxyAgent.
        Args:
            sender (Agent): the sender agent. It should be the instance of RetrieveUserProxyAgent.
            recipient (Agent): the recipient agent. Usually it's the assistant agent.
            context (dict): the context for the message generation. It should contain the following keys:
                - `problem` (str) - the problem to be solved.
                - `n_results` (int) - the number of results to be retrieved. Default is 20.
                - `search_string` (str) - only docs that contain an exact match of this string will be retrieved. Default is "".
        Returns:
            str: the generated message ready to be sent to the recipient agent.
        """
        sender._reset()

        problem = context.get("problem", "")
        n_results = context.get("n_results", 20)
        search_string = context.get("search_string", "")
        filter = context.get("filter", {})

        sender.retrieve_docs(problem, n_results, search_string, filter)
        sender.problem = problem
        sender.n_results = n_results
        doc_contents = sender._get_context(sender._results)
        message = sender._generate_message(doc_contents, sender._task)
        return message

    def _get_context(self, results: QueryResults):
        doc_contents = ""
        self._current_docs_in_context = []
        current_tokens = 0
        _doc_idx = self._doc_idx
        _tmp_retrieve_count = 0
        for idx, doc in enumerate(results[0]):
            doc = doc[0]
            if idx <= _doc_idx:
                continue
            if doc["id"] in self._doc_ids:
                continue
            # _doc_tokens = self.custom_token_count_function(doc["content"], self._model)
            _doc_tokens = self.custom_token_count_function(doc["metadata"]["answer"], self._model)
            if _doc_tokens > self._context_max_tokens:
                func_print = f"Skip doc_id {doc['id']} as it is too long to fit in the context."
                print(func_print)
                self._doc_idx = idx
                continue
            if current_tokens + _doc_tokens > self._context_max_tokens:
                break
            func_print = f"Adding content of doc {doc['id']} to context."
            print(func_print)
            current_tokens += _doc_tokens
            # doc_contents += doc["content"] + "\n"
            doc_contents += doc['metadata']['answer'] + "\n"
            _metadata = doc.get("metadata")
            if isinstance(_metadata, dict):
                self._current_docs_in_context.append(_metadata.get("source", ""))
            self._doc_idx = idx
            self._doc_ids.append(doc["id"])
            self._doc_contents.append(doc["content"])
            _tmp_retrieve_count += 1
            if _tmp_retrieve_count >= self.n_results:
                break
        return doc_contents

config_list = [{"model": "gpt-4o-mini", "api_key": "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"}]

# 使用正确的模型名来确保维度一致
openai_ef = ef.OpenAIEmbeddingFunction(
                api_key="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",
                model_name="text-embedding-ada-002"  # 确保这是返回1536维的模型
            )

assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={"config_list": config_list, "timeout": 60, "temperature": 0}
)

vector_db = ChromaVectorDB(client=chromadb.PersistentClient(path="../data/vector_db/qiniu_db/chromadb"), embedding_function = openai_ef)

# 确保使用相同的 embedding function
ragproxyagent = MyRetrieveUserProxyAgent(
    name="ragproxyagent",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "qa",  # "code", "qa" 和 "default"
        "model": config_list[0]["model"],
        "vector_db": vector_db,
        "collection_name": "qiniu_db_collection",
        "embedding_function": openai_ef,  # 确保此处使用正确的嵌入模型
        "get_or_create": False,  # 设置为False，如果不希望重复使用已有的集合
    },
    code_execution_config=False,  # 设置为False如果不希望执行代码
    human_input_mode="NEVER"
)


if __name__ == '__main__':
    assistant.reset()
    ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem="有回放吗？", filter={"doc_type": "public"})















