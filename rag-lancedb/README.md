rag 测试小工具：

- LLM 使用 OpenAI API。
- Embedding Model 使用 OpenAI API。
- 向量数据库使用 LanceDB，使用 S3 作为存储。

使用流程：

1: 安装 Python 依赖：

```bash
pip install -U -r requirements.txt
```

2: 创建并启动 MinIO 容器：

```bash
docker compose up -d
```

3: 添加自己的 OpenAI API Key 以及可选的代理地址到项目根目录的 `.env` 文件中：

```bash
OPENAI_API_KEY='sk-xxxxxx'
OPENAI_BASE_URL='https://api.openai.com/v1'
OPENAI_HTTP_PROXY='http://x.x.x.x:xxxx'
```

4: 按需修改源码 `rag.py`，包括 OpenAI 使用的网址、代理地址等。

5: unset 代理变量，避免连接不到 MinIO。

6: 执行 `python3 rag.py` 启动工具。
