import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def stat_file():
    input_file = 'subset_data.jsonl'
        
    # 读取原始文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # 计算token长度和收集scores
    token_lengths = []
    scores = []
    for item in data:
        
        # 收集score数据
        if 'score' in item:
            
            if item['score'] < 7:
                continue
            scores.append(item['score'])
            
        
        # 现有的token长度计算
        total_tokens = sum([
            item.get('prompt_tokens_len', 0),
            item.get('content_tokens_len', 0),
            item.get('reasoning_content_tokens_len', 0)
        ])
        token_lengths.append(total_tokens)
        


    # 创建一个新的图形，包含两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))


    # 第一个子图：token分布
    ax1.hist(token_lengths, bins=50, edgecolor='black')
    ax1.set_title('Token Length Distribution')
    ax1.set_xlabel('Total Tokens')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Token统计信息
    token_mean = np.mean(token_lengths)
    token_std = np.std(token_lengths)
    token_25 = np.percentile(token_lengths, 25) 
    token_75 = np.percentile(token_lengths, 75)
    token_95 = np.percentile(token_lengths, 95)
    token_99 = np.percentile(token_lengths, 99)
    
    # 添加统计线和图例
    ax1.axvline(x=max(token_lengths), color='r', linestyle='--', label=f'Max: {max(token_lengths)}')
    ax1.axvline(x=token_25, color='orange', linestyle='--', label=f'25th: {int(token_25)}')
    ax1.axvline(x=token_mean, color='g', linestyle='--', 
                label=f'Mean: {int(token_mean)}\nStd: {int(token_std)}\n25th: {int(token_25)}\n75th: {int(token_75)}\n95th: {int(token_95)}\n99th: {int(token_99)}')
    ax1.axvline(x=token_75, color='orange', linestyle='--', label=f'75th: {int(token_75)}')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    
    # 第二个子图：score分布
    if scores:
        ax2.hist(scores, bins=30, edgecolor='black')
        ax2.set_title('Score Distribution')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Score统计信息
        score_mean = np.mean(scores)
        score_std = np.std(scores)
        score_25 = np.percentile(scores, 25)  # 添加25分位数
        score_75 = np.percentile(scores, 75)
        score_95 = np.percentile(scores, 95)
        score_99 = np.percentile(scores, 99)
        
        # 添加统计线和图例
        ax2.axvline(x=max(scores), color='r', linestyle='--', label=f'Max: {max(scores):.2f}')
        ax2.axvline(x=score_25, color='orange', linestyle='--', label=f'25th: {score_25:.2f}')
        ax2.axvline(x=score_mean, color='g', linestyle='--', 
                    label=f'Mean: {score_mean:.2f}\nStd: {score_std:.2f}\n25th: {score_25:.2f}\n75th: {score_75:.2f}\n95th: {score_95:.2f}\n99th: {score_99:.2f}')
        ax2.axvline(x=score_75, color='orange', linestyle='--', label=f'75th: {score_75:.2f}')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 调整子图布局以适应更大的图例
    plt.tight_layout()
    
    # 保存更大尺寸的图片以容纳图例
    fig.set_size_inches(12, 12)
    plt.savefig('distributions.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 打印统计信息
    print("\n=== Token统计 ===")
    print(f"最大token数: {max(token_lengths)}")
    print(f"平均token数: {int(token_mean)}")
    print(f"标准差: {int(token_std)}")
    print(f"25分位数: {int(token_25)}")  # 新增
    print(f"75分位数: {int(token_75)}")
    print(f"95分位数: {int(token_95)}")
    print(f"99分位数: {int(token_99)}")
    print(f"样本总数: {len(token_lengths)}")
    
    if scores:
        print("\n=== Score统计 ===")
        print(f"最大分数: {max(scores):.2f}")
        print(f"平均分数: {score_mean:.2f}")
        print(f"标准差: {score_std:.2f}")
        print(f"25分位数: {score_25:.2f}")  # 新增
        print(f"75分位数: {score_75:.2f}")
        print(f"95分位数: {score_95:.2f}")
        print(f"99分位数: {score_99:.2f}")
        print(f"分数样本总数: {len(scores)}")
        print(f"分数样本总数: {len(scores)}")
        
        score_counts = Counter(scores)
        print("\n=== Score分布统计 ===")
        for score, count in sorted(score_counts.items()):
            print(f"Score {score:.1f}: {count}个样本")


def create_formatted_item(raw_item):
    """创建统一格式的数据项"""
    return {
        'instruction': raw_item.get('input', ''),
        'input': '',
        'output': f"<think>\n{raw_item.get('reasoning_content', '')}</think>\n{raw_item.get('content', '')}"
    }

def process_file():
    # 文件配置 
    input_file = 'subset_data.jsonl'
    processed_file = 'processed_subset.jsonl'
    output_files = {
        'score_filtered': 'filter_low_data.jsonl',
        'length_8192': 'filter_length_data.jsonl', 
        'length_18678': 'filter_lengthest_data.jsonl'
    }
    
    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]
    with open(processed_file, 'r', encoding='utf-8') as f:
        processed_data = [json.loads(line) for line in f]
    print(f"原始数据量：{len(raw_data)}")
    print(f"合并数据量：{len(processed_data)}")

    # 建立raw_data的映射关系
    raw_data_map = {
        json.dumps(create_formatted_item(item), ensure_ascii=False): item
        for item in raw_data
    }

    # 过滤低分数据
    low_score_items = {
        json_str for json_str, item in raw_data_map.items()
        if item.get('score', 10) < 7
    }
    
    # 保留processed_data中的数据,但要过滤掉低分的raw_data项
    score_filtered = [
        item for item in processed_data
        if json.dumps(item, ensure_ascii=False) not in low_score_items
    ]
    
    # 写入过滤后的数据
    with open(output_files['score_filtered'], 'w', encoding='utf-8') as f:
        for item in score_filtered:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"完成分数过滤，剩余数据量：{len(score_filtered)}")

    # 过滤不同token长度的数据 (只处理raw_data中的数据)
    for output_key, max_tokens in [('length_8192', 8192), ('length_18678', 18678)]:
        filtered_data = []
        
        # 先添加processed_data中独有的数据
        for item in processed_data:
            item_str = json.dumps(item, ensure_ascii=False)
            if item_str not in raw_data_map:
                filtered_data.append(item)
        
        # 再处理raw_data中的数据
        for item in raw_data:
            total_tokens = sum([
                item.get('prompt_tokens_len', 0),
                item.get('content_tokens_len', 0),
                item.get('reasoning_content_tokens_len', 0)
            ])
            
            if total_tokens <= max_tokens:
                new_item = create_formatted_item(item)
                item_str = json.dumps(new_item, ensure_ascii=False)
                if item_str not in low_score_items:
                    filtered_data.append(new_item)

        with open(output_files[output_key], 'w', encoding='utf-8') as f:
            for item in filtered_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"完成 {max_tokens} tokens过滤，剩余数据量：{len(filtered_data)}")
if __name__ == '__main__':
    process_file()
    print("\n处理完成！")