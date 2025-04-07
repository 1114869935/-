from flask import Flask, request, render_template_string
import os
import json
from models import Agent
from paper_agent import PaperAgent
from datetime import datetime, timedelta

app = Flask(__name__)

# 配置模型路径
crawler_path = "checkpoints/pasa-7b-crawler"
selector_path = "checkpoints/pasa-7b-selector"
crawler = Agent(crawler_path)
selector = Agent(selector_path)

# 定义超参数
expand_layers = 2
search_queries = 5
search_papers = 10
expand_papers = 20
threads_num = 20

# 定义优化后的 HTML 模板
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Paper Search</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        h1 {
            color: #333;
        }
        form {
            margin-bottom: 20px;
        }
        textarea {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
        }
        input[type="submit"] {
            padding: 10px 20px;
            background-color: #007BFF;
            color: white;
            border: none;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #0056b3;
        }
        h2 {
            color: #333;
            margin-top: 30px;
        }
        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            white-space: pre-wrap;
        }
        .section {
            margin-bottom: 20px;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
        }
        .paper {
            margin-bottom: 10px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        .paper h3 {
            margin: 0;
            color: #007BFF;
        }
        .paper p {
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <h1>输入Abstract进行论文搜索</h1>
    <form method="post">
        <textarea name="abstract" rows="10" cols="50" placeholder="输入Abstract"></textarea><br>
        <input type="submit" value="搜索">
    </form>
    {% if results %}
        <h2>搜索结果</h2>
        {% for section, papers in results['child'].items() %}
            <div class="section">
                <h3>{{ section }}</h3>
                {% for paper in papers %}
                    <div class="paper">
                        <h3>{{ paper['title'] }}</h3>
                        <p><strong>ArXiv ID:</strong> {{ paper['arxiv_id'] }}</p>
                        <p><strong>摘要:</strong> {{ paper['abstract'] }}</p>
                        <p><strong>来源:</strong> {{ paper['source'] }}</p>
                        <p><strong>相关得分:</strong> {{ paper['select_score'] }}</p>
                    </div>
                {% endfor %}
            </div>
        {% endfor %}
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        abstract = request.form.get('abstract')
        end_date = datetime.now().strftime("%Y%m%d")

        paper_agent = PaperAgent(
            user_query=abstract,
            crawler=crawler,
            selector=selector,
            end_date=end_date,
            expand_layers=expand_layers,
            search_queries=search_queries,
            search_papers=search_papers,
            expand_papers=expand_papers,
            threads_num=threads_num
        )

        paper_agent.run()
        results = paper_agent.root.todic()

        return render_template_string(html_template, results=results)

    return render_template_string(html_template)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)