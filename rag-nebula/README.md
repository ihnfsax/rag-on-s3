使用说明

（可选）让 jupyter notebook 能够使用 conda 环境：

```bash
conda install ipykernel
ipython kernel install --user --name=<conda-env-name>
```

创建 `.env` 文件，里面设置 OpenAI 相关环境变量：

```bash
OPENAI_API_KEY='xxxxxxx'
OPENAI_API_BASE='https://xxxxxx.com/v1'
OPENAI_PROXY='http://xxx.xxx.xxx.xxx:xxxx'
```

启动容器：

```bash
cd rag-nebula
docker compose up -d
```

启动 jupyter notebook:

```bash
jupyter notebook
```
