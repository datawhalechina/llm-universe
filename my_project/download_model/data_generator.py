import json
import os
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 加载环境变量
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# 初始化DeepSeek客户端
client = OpenAI(
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 常量定义
TAB = chr(9)  # 制表符
MAX_RETRIES = 3  # 最大重试次数
MAX_WORKERS = 5  # 并发线程数（根据API限制调整）
BATCH_SIZE = 30  # 每批处理量

def load_knowledge_graph(file_path: str) -> List[Dict[str, str]]:
    """加载知识图谱数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def expand_with_deepseek(prompt: str, max_retries: int = MAX_RETRIES) -> str:
    """调用DeepSeek生成详细解释"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个资深中医专家，需要生成专业、详实的解释。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500,
                timeout=30
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            time.sleep(2 ** attempt)  # 指数退避
    return f"【生成失败】{prompt[:50]}..."  # 失败时返回截断提示

def process_single_triplet(triplet: Dict[str, str]) -> Dict[str, Any]:
    """处理单个三元组的线程任务"""
    try:
        node1, node1_type = triplet["node_1"].split(TAB)
        node2, node2_type = triplet["node_2"].split(TAB)
        relation = triplet["relation"]

        # 动态生成指令
        instruction = {
            "包含": f"列举{node1}包含的具体中药",
            "功效": f"详细说明{node1}的药用功效",
            "出处": f"说明{node1}的来源",
            "药性": f"说明{node1}的药性",
            "药味": f"说明{node1}的药味",
            "可治疗": f"详细说明{node1}的主治",
            "归经": f"详细说明{node1}的归经",
        }.get(relation, f"说明{node1}的{relation}信息")

        input_text = f"{node1_type} {node1}的{relation}是什么？"
        base_output = f"{node1}的{relation}是：{node2}。"

        # 需要扩写的关系类型
        if relation in ["功效", "可治疗", "归经"]:
            llm_prompt = f"请详细说明以下内容：\n1. {relation}定义\n2. {node2}的详细解释\n3. 临床应用\n\n基础信息：{base_output}"
            output = expand_with_deepseek(llm_prompt)
        else:
            output = base_output

        return {
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "source": triplet  # 保留原始数据
        }
    except Exception as e:
        print(f"处理失败: {str(e)}")
        return None

def process_batch(batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """并行处理一批数据"""
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_triplet, t): t for t in batch}
        for future in tqdm(as_completed(futures), total=len(batch), desc="处理批次"):
            if (result := future.result()) is not None:
                results.append(result)
    return results

def main():
    input_file = os.path.join("data", "medicine_relations.json")
    output_file = os.path.join("data", "tcm_instruction_multi_thread.json")
    
    # 加载数据
    knowledge_graph = load_knowledge_graph(input_file)
    print(f"已加载 {len(knowledge_graph)} 条知识图谱数据")

    # 断点续传
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            processed_ids = {item["input"] for item in existing_data}
        print(f"检测到已有 {len(processed_ids)} 条处理结果")

    # 过滤未处理的数据
    new_items = [
        item for item in knowledge_graph
        if f"输入：{item['node_1'].split(TAB)[1]} {item['node_1'].split(TAB)[0]}的{item['relation']}是什么？" 
        not in processed_ids
    ]
    print(f"需要处理的新数据量: {len(new_items)}")

    # 分批并行处理
    all_results = []
    for i in tqdm(range(0, len(new_items), BATCH_SIZE), desc="总进度"):
        batch = new_items[i:i + BATCH_SIZE]
        batch_results = process_batch(batch)
        all_results.extend(batch_results)

        # 实时保存
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing.extend(batch_results)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
        else:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(batch_results, f, ensure_ascii=False, indent=2)

    print(f"处理完成！结果保存至 {output_file}")

if __name__ == "__main__":
    main()
