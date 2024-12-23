import time
import traceback

import math
import pandas as pd

import threading
import logging
from tqdm import tqdm
from string import Template
import requests

url = "http://127.0.0.1:11430/v1/chat/completions"

prompt = '''
    你是一个文本rewrite助手，你需要将给你的文本以相同的意思，重写十句不同的话作为list返回

    注意不要回答多余废话，只按照以下格式返回：
    [
    {
       "rewrite":"重写后内容" 
    },
    {
       "rewrite":"重写后内容" 
    },
    .....
    ]

    以下是你需要重写的文本:

    $content
    '''
data = pd.read_csv('启牛话术v5.csv',index_col=0)
print(data.shape)
# 总数据量
total_data = data.shape[0]
# 分成的份数
num_parts = 10

# 每一份的大小，前 num_parts-1 份为均等
part_size = math.ceil(total_data / num_parts)
#
print(f'每个分片需要处理:{part_size}')


# # 索引范围计算函数
def get_indices_for_part(part_number):
    if part_number < 1 or part_number > num_parts:
        raise ValueError("输入的份数应在 1 到 10 之间")

    # 计算起始和结束索引
    start_index = (part_number - 1) * part_size
    end_index = min(start_index + part_size, total_data)

    return start_index, end_index


def thread_function(thread_id):
    # 配置每个线程的日志文件
    logger = logging.getLogger(f"Thread-{thread_id}")
    logger.setLevel(logging.INFO)
    # 创建文件处理器
    handler = logging.FileHandler(f"thread_{thread_id}.log")
    handler.setLevel(logging.INFO)
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # 添加处理器到logger
    logger.addHandler(handler)

    logger.info(f"Thread {thread_id} starting.")

    # 获取本线程的数据分片
    start_index, end_index = get_indices_for_part(int(thread_id) + 1)
    df_clip = data.iloc[start_index:end_index]

    violate_df = pd.DataFrame()
    has_done_num = 0
    # 调用接口或任务
    for index, row in tqdm(df_clip.iterrows()):  # 模拟5次任务
        try:
            time.sleep(0.1)
            has_done_num += 1
            logger.info(f"目前跑批进度：{has_done_num}")
            prompt_current = Template(prompt).substitute(content=row['question'])
            response_dict = eval(call_api(prompt_current))
            for i in response_dict:
                # 证明违规，需要记录下来
                row['rewrite'] = i['rewrite']
                violate_df = violate_df._append(row)
                # 每次发现都要进行保存实时记录
                violate_df.to_pickle(f'{thread_id}_result.pkl')
                logger.info(f"Thread {thread_id} 正在跑批: 已进行 {index},已保存")
        except Exception as e:
            logger.error(f"Thread {thread_id} encountered an error: {traceback.format_exc()}")
            continue

    logger.info(f"Thread {thread_id} finished. ")

    # 移除处理器（防止资源泄露）
    logger.removeHandler(handler)


def call_api(prompt_current):
    payload = {
        "model": "qwen2.5:13b",
        "max_tokens": 1000,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": prompt_current
            }
        ]
    }
    headers = {"content-type": "application/json"}

    response = requests.request("POST", url, json=payload, headers=headers)

    return response.json()['choices'][0]['message']['content']


if __name__ == '__main__':
    print('跑批开始')
    print(Template(prompt).substitute(content="这是文本"))

    threads = []
    for i in range(10):  # 创建10个线程
        thread = threading.Thread(target=thread_function, args=(i,))
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    print("All threads completed.")

