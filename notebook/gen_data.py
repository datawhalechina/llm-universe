import json
from tqdm import tqdm
from openai import OpenAI
from openai import APITimeoutError
import os
import time


class APIModel:
    def __init__(self, model, api_key, base_url):
        self.__api_key = api_key
        self.__base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=self.__api_key, base_url=self.__base_url)

    def generate_prompt(self, data):
        question = data['question']
        options = [f"({chr(65+i)}): {option}" for i, option in enumerate(data['choices'])]
        options_str = '\n'.join(options)
        prompt = f"问题: {question}\n{options_str}\n"
        return prompt

    def get_compention(self, prompt, system=None, stream_output=False):
        """
        获取模型回复
        
        Args:
            prompt: 提示词
            system: 系统提示词
            stream_output: 是否使用流式输出，默认为 False
        """
        if system is None:
            system = 'You are a helpful assistant.'
        
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                print(self.model)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': system},
                        {'role': 'user', 'content': prompt}
                    ],
                    stream=False,
                    temperature=0.6,
                    timeout=3600,
                )
                print(response)
                
                if not stream_output:
                    # 非流式输出模式
                    reasoning_content = response.choices[0].message.reasoning_content
                    content = response.choices[0].message.content
                    return f"<think>\n{reasoning_content}</think>\n{content}"
                
                # 流式输出模式
                full_response = ""
                think_start = False
                answer_start = False
                
                for chunk in response:
                    print(chunk)
                    delta = chunk.choices[0].delta
                    if not delta.content and not hasattr(delta, 'reasoning_content'):
                        continue
                        
                    if hasattr(delta, 'reasoning_content'):
                        if not think_start:
                            think_start = True
                            full_response += '<think>'
                            print('\n<think>', end="", flush=True)
                        full_response += delta.reasoning_content
                        print(delta.reasoning_content, end="", flush=True)
                    
                    if delta.content:
                        if not answer_start and think_start:
                            answer_start = True
                            full_response += '</think>\n'
                            print('</think>\n', end="", flush=True)
                        full_response += delta.content
                        print(delta.content, end="", flush=True)
                
                return full_response
            
            except APITimeoutError:
                retry_count += 1
                if retry_count >= max_retries:
                    raise Exception(f"API请求超时，已达最大重试次数（{max_retries}次）")
                print(f"\n请求超时，正在重试 ({retry_count}/{max_retries})...")
                time.sleep(1)
                
            except Exception as e:
                raise e
        
        raise Exception("未知错误，重试逻辑异常")


if __name__ == "__main__":
    # 初始化 APIModel 实例
    model = APIModel(
        model="deepseek-ai/DeepSeek-R1", 
        api_key='25ca2ad7-d686-4a19-ab3a-9dd56041bf89',
        base_url='https://api-inference.modelscope.cn/v1/'
    )
    system_prompt = "你是一个高智商和高情商的专家，你被要求回答一个选择题，并选出一个正确的选项，解释原因，最终输出格式为：`答案是(选项)`。"
    # 单次交互示例
    with open('./IQ.jsonl', 'r', encoding='utf-8') as f:
        data = f.readlines()
    for i in tqdm(range(0, len(data))):
        if  i != 28 and i != 31:
            continue
        item = json.loads(data[i])
        question_prompt = model.generate_prompt(item)
        response = model.get_compention(question_prompt, system_prompt, stream_output=False)
        with open('distill-EQ-IQ1.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'system': system_prompt,
                'input': "",
                'instruction': question_prompt,
                'output': response,
                'correct': item['answer']
            }, ensure_ascii=False) + '\n')
        time.sleep(10)
    