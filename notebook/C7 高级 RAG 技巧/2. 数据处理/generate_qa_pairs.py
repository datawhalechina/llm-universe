from typing import List
from openai import OpenAI
from zhipuai import ZhipuAI
import re
import json
from langchain_core.documents import Document
from tqdm import tqdm

class QaPairs():
    '''存储List[dict]类型数据'''

    def __init__(self, qa_pairs: List[dict]):
        self.qa_pairs = qa_pairs
        

    def save_json(self, path: str):
        '''将数据存储为json格式'''

        with open(path, "w", encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, ensure_ascii=False, indent=4)

    @classmethod
    def from_json(cls, path:str) -> 'QaPairs':
        '''读取json格式数据'''

        with open(path) as f:
            data = json.load(f)
        return cls(data)


llm_list = ['glm-4', 'glm-4v', 'glm-3-turbo', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4o']

PROMPT = '''
下面是上下文信息。 
 
--------------------- 
{context_str} 
--------------------- 
 
给定上下文信息，没有先验知识。 
仅根据下面的查询生成问题。 
 
你是一位老师/教授。你的任务是为即将到来的\
测验/考试设置{num_questions_per_page}个问题以及问题涉及到的原文内容\
在整个文件中，问题的性质应该是多样化的。\
将问题限制在提供的上下文信息之内。\
按照问题1：
问题

原文内容1：
内容

的形式回答
'''

def list_generate_qa_pairs(
        texts: List[str],
        num_questions_per_page: int = 2,
        model: str = 'glm-4',
) -> QaPairs:
    '''借助大模型从给定的texts里提取出问题与对应的答案'''

    if model not in llm_list:
        raise ValueError('你选择的模型暂时不被支持'
                            '''请使用'glm-4', 'glm-4v', 'glm-3-turbo', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4o' 中的一个作为model的参数''')
    elif model in llm_list[:3]:
        llm = ZhipuAI()
    else:
        llm = OpenAI()
    qa_pairs = []

    for text in tqdm(texts):
        if len(text) > 200:
            prompt = PROMPT.format(
                context_str=text,
                num_questions_per_page=num_questions_per_page
            )
            response = llm.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            matches = re.findall(
                r'问题\d+：(.*?)原文内容\d+：(.*?)((?=问题\d+：)|$)',
                response.choices[0].message.content,
                re.DOTALL
            )
            for _, match in enumerate(matches):
                qa = {
                    'query': match[0].strip(),
                    'answer': match[1].strip()
                }
                qa_pairs.append(qa)
    return QaPairs(qa_pairs=qa_pairs)

def docs_generate_qa_pairs(
        docs: List[Document], 
        num_questions_per_page: int = 2,
        model: str = 'glm-4'
) -> QaPairs:
    '''借助大模型从给定的docs里提取出问题与对应的答案'''
    list_doc = [doc.page_content for doc in docs]
    return list_generate_qa_pairs(list_doc, num_questions_per_page, model=model)


def docs_generate_pdf_qa_pairs(
        pdf_pages: List[Document],
        num_questions_per_page: int = 2,
        model: str = 'glm-4',
) -> QaPairs:
    '''
    借助大模型从给定的texts里提取出问题、答案
    返回结果为问题、答案、所属页码
    '''

    if model not in llm_list:
        raise ValueError('你选择的模型暂时不被支持'
                            '''请使用'glm-4', 'glm-4v', 'glm-3-turbo', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4o' 中的一个作为model的参数''')
    elif model in llm_list[:3]:
        llm = ZhipuAI()
    else:
        llm = OpenAI()
    qa_pairs = []

    for page in tqdm(pdf_pages):
        if len(page.page_content) > 200:
            prompt = PROMPT.format(
                context_str=page.page_content,
                num_questions_per_page=num_questions_per_page
            )
            response = llm.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            matches = re.findall(
                r'问题\d+：(.*?)原文内容\d+：(.*?)((?=问题\d+：)|$)',
                response.choices[0].message.content,
                re.DOTALL
            )
            for _, match in enumerate(matches):
                qa = {
                    'query': match[0].strip(),
                    'answer': match[1].strip(),
                    'page_num': page.metadata['page']
                }
                qa_pairs.append(qa)
    return QaPairs(qa_pairs=qa_pairs)
