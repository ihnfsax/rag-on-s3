from dotenv import load_dotenv
from openai import OpenAI
from botocore.exceptions import ClientError
import os
import fitz
import lancedb
import boto3
import botocore
import random
import pandas as pd
import logging
import pyarrow as pa
import httpx
import time

logging.basicConfig(
    filename="rag.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def gen_text_chunks(text, max_length=4096):
    chunks = []
    while len(text) > max_length:
        chunk = text[:max_length]
        text = text[max_length:]
        chunks.append(chunk)
    chunks.append(text)
    return chunks


class RAG:
    def __init__(self, vector_store_bucket, file_bucket):
        os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
        os.environ["AWS_ENDPOINT"] = "http://127.0.0.1:9000"
        os.environ["AWS_DEFAULT_REGION"] = "ap-east-1"
        session = boto3.Session()
        self.s3 = session.client(
            "s3",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
            endpoint_url="http://127.0.0.1:9000",
            region_name="ap-east-1",
            config=botocore.config.Config(s3={"addressing_style": "path"}),
        )
        db_uri = f"s3://{vector_store_bucket}/"
        self.db = lancedb.connect(db_uri)
        self.table = "knowledge_base"
        self.file_bucket = file_bucket
        self.chunk_size = 512
        self.openai = OpenAI(
            http_client=httpx.Client(proxy="http://10.252.1.45:7890"),
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://aihubmix.com/v1",
        )
        self.embedding_model = "text-embedding-3-small"
        self.embedding_size = 1536
        self.schema = pa.schema(
            [
                pa.field("embedding", pa.list_(pa.float32(), self.embedding_size)),
                pa.field("text", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("pdf_file_name", pa.string()),
                pa.field("text_file_name", pa.string()),
            ]
        )
        self.limit = 3  # 最多获取几个相似文本块作为上下文
        pd.set_option("display.max_colwidth", None)

    def load_pdf(self, pdf_path):
        logging.info(f"Loading pdf file: {pdf_path}")
        if not os.path.exists(pdf_path):
            print("The pdf file does not exist.")
            logging.info(f"The pdf file does not exist: {pdf_path}")
            return

        # 提取pdf文件中的文本
        logging.info(f"Extracting text from pdf file: {pdf_path}")
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
        except Exception:
            print("Failed to extract text from the pdf file.")
            logging.error(f"Failed to extract text from the pdf file: {pdf_path}")
            return

        # 上传pdf文件和文本文件到S3
        pdf_name = os.path.basename(pdf_path)
        text_name = pdf_name + ".text"
        if self.__upload_pdf_to_s3(pdf_path, text, text_name):
            return

        # 生成文本块
        text = text.replace("\n", " ")
        words = text.split()
        print(f"Extracted {len(words)} words from the pdf file")
        chunks = [
            " ".join(words[i : i + self.chunk_size])
            for i in range(0, len(words), self.chunk_size)
        ]
        logging.info(
            f"Extracted {len(chunks)} text chunks from the pdf file: {pdf_path}"
        )
        print(f"Extracted {len(chunks)} text chunks from the pdf file")

        # 生成文本块的embedding 并插入到 lancedb
        print("Generating embeddings...")
        for i, chunk in enumerate(chunks):
            logging.info(f"Generating embedding: {pdf_name} chunk {i}")
            embedding = (
                self.openai.embeddings.create(input=[chunk], model=self.embedding_model)
                .data[0]
                .embedding
            )
            logging.info(f"Add embedding to db: {pdf_name} chunk {i}")
            if self.__add_embedding_to_db(embedding, chunk, i, pdf_name, text_name):
                return

        print("The pdf file has been loaded successfully.")
        logging.info(f"The pdf file has been loaded successfully: {pdf_path}")

    def clear_knowledge_base(self):
        print("Are you sure to clear all data in S3? [Y/n]", end="")
        clear = input()
        if clear.lower() != "y":
            return
        try:
            logging.info(f"Deleting table knowledge_base in LanceDB")
            self.db.drop_table(self.table)
            res = []
            bucket = self.s3.list_objects(Bucket=self.file_bucket)
            for obj_version in bucket["Contents"]:
                key = obj_version["Key"]
                res.append({"Key": key})

            if len(res):
                logging.info(
                    f"Deleting all {len(res)} objects from bucket {self.file_bucket}"
                )
                self.s3.delete_objects(Bucket=self.file_bucket, Delete={"Objects": res})
        except Exception as e:
            print(e)
            logging.error(e)
            return
        logging.info("Clear knowledge base done")
        print("Clear knowledge base done.")

    def __upload_pdf_to_s3(self, pdf_path, text, text_name):
        pdf_name = os.path.basename(pdf_path)
        is_exist = False
        try:
            self.s3.head_object(Bucket=self.file_bucket, Key=pdf_name)
            is_exist = True
        except ClientError:
            pass
        if is_exist:
            print("The pdf file already exists.")
            # print("The pdf file already exists. Overwrite it? [Y/n]", end="")
            # overwrite = input()
            # if overwrite.lower() != "y":
            #     return True
            return True

        logging.info(f"Upload pdf file: {pdf_name}")
        try:
            self.s3.upload_file(pdf_path, self.file_bucket, pdf_name)
        except Exception as e:
            print(e)
            logging.error(e)
            return True
        logging.info(f"The pdf file has been uploaded successfully: {pdf_path}")

        logging.info(f"Upload text file: {text_name}")
        try:
            self.s3.put_object(Body=text, Bucket=self.file_bucket, Key=text_name)
        except Exception as e:
            print(e)
            logging.error(e)
            return True
        logging.info(f"The text file has been uploaded successfully: {text_name}")
        return False

    def __add_embedding_to_db(
        self, embedding, text_chunk, chunk_idx, pdf_name, text_name
    ):
        try:
            table = self.db.create_table(self.table, schema=self.schema, exist_ok=True)
            table.add(
                [
                    {
                        "embedding": embedding,
                        "text": text_chunk,
                        "chunk_index": chunk_idx,
                        "pdf_file_name": pdf_name,
                        "text_file_name": text_name,
                    }
                ]
            )
        except Exception as e:
            print(e)
            logging.error(e)
            return True
        return False

    def show_knowledge_base_infos(self):
        try:
            table = self.db.create_table(self.table, schema=self.schema, exist_ok=True)
            res = (
                table.search()
                .select(["pdf_file_name", "chunk_index"])
                .limit(None)
                .to_pandas()
            )
            res = res.groupby("pdf_file_name").size().reset_index()

            print("\nKnowledge base infos:")
            res.rename(columns={res.columns[1]: "chunk_size"}, inplace=True)
            print(res)
        except Exception as e:
            print(e)
            logging.error(e)

    def chat(self):
        print("Start chatting...")
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        while True:
            user_input = input("Input question (input 'exit' to exit): ")
            if user_input == "exit":
                break
            if self.__chat_with_user(user_input):
                break

    def __chat_with_user(self, user_input):
        try:
            user_embedding = (
                self.openai.embeddings.create(
                    input=[user_input], model=self.embedding_model
                )
                .data[0]
                .embedding
            )
            table = self.db.create_table(self.table, schema=self.schema, exist_ok=True)
            start_time = time.time()
            res = (
                table.search(user_embedding)
                .select(["pdf_file_name", "text", "text_file_name", "chunk_index"])
                .limit(self.limit)
                .to_pandas()
            )
            print(f"Search time: {time.time() - start_time}")
            context = []
            for _, ele in res["text"].items():
                context.append(ele)

            question = f"""
                You are an assistant for question-answering tasks. 
                Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, just say that you don't know. 
                Use four sentences maximum and keep the answer concise.
                Question: {user_input}
                Context: {context}
                Answer: 
            """

            self.messages.append({"role": "user", "content": question})

            chat_completion = self.openai.chat.completions.create(
                model="gpt-3.5-turbo", messages=self.messages
            )
            response = chat_completion.choices[0].message.content
            print("\nReferences:\n" + res["pdf_file_name"].to_string(index=False))
            print("Anwser: \n" + response + "\n\n")
            self.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            print(e)
            logging.error(e)
            return True


if __name__ == "__main__":
    load_dotenv()

    print("Initializing...")
    rag = RAG(vector_store_bucket="lance", file_bucket="blob")

    print("Welcome to the chatbot!")
    while True:
        print("\noption 1: load new pdf file")
        print("option 2: show knowledge base infos")
        print("option 3: start chatting")
        print("option 4: clear knowledge base")
        print("option 5: exit\n")
        option = input("Please select an option: ")
        if option == "1":
            pdf_path = input("Please enter the path of the pdf file: ")
            rag.load_pdf(pdf_path)
        elif option == "2":
            rag.show_knowledge_base_infos()
        elif option == "3":
            rag.chat()
        elif option == "4":
            rag.clear_knowledge_base()
        elif option == "5":
            break
        else:
            print("Invalid option!")

    print("Goodbye!")
