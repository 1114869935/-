import json
import re

# 假设你的 JSON 数据存储在 papers.json 文件中
with open('papers.json', 'r') as file:
    data = json.load(file)

# 处理每个论文条目，提取 arxiv_id 并去掉版本号
processed_data = []
for paper in data:
    # 使用正则表达式提取 arxiv_id 的数字部分
    match = re.match(r'^(\d{4}\.\d+)', paper['arxiv_id'])
    if match:
        processed_id = match.group(1)
        paper['arxiv_id'] = processed_id
    processed_data.append(paper)

# 将处理后的数据保存到新的 JSON 文件中
with open('processed_papers.json', 'w') as file:
    json.dump(processed_data, file, indent=4)

print("处理完成，结果已保存到 processed_papers.json 文件中。")
