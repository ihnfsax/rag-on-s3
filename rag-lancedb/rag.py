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
import unicodedata

logging.basicConfig(
    filename="rag.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def find_pdf_files(directory):
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


def get_text_chunks(text, chunk_size):
    text = text.replace("\n", " ")
    words = text.split()
    word_size = 0
    chunks = []
    start = 0
    for i, word in enumerate(words):
        che_count = 0
        for char in word:
            if "\u0e00" <= char <= "\u9fa5":
                che_count += 1
        if che_count > 0:
            word_size += che_count / 2
        else:
            word_size += 1
        if word_size >= chunk_size:
            chunks.append(" ".join(words[start:i]))
            word_size = 0
            start = i
    if start < len(words) - 1:
        chunks.append(" ".join(words[start:]))
    return chunks


class RAG:
    def __init__(self, vector_store_bucket, file_bucket):
        os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
        os.environ["AWS_ENDPOINT"] = "http://127.0.0.1:9000"
        os.environ["AWS_DEFAULT_REGION"] = "ap-east-1"
        self.__init_s3()
        self.file_bucket = file_bucket
        self.__init_vector_store(vector_store_bucket)
        self.__init_openai()
        self.chunk_size = 512
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
        self.limit = 5  # 最多获取几个相似文本块作为上下文
        pd.set_option("display.max_colwidth", None)

    def __init_s3(self):
        session = boto3.Session()
        self.s3 = session.client(
            "s3",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
            endpoint_url="http://127.0.0.1:9000",
            region_name="ap-east-1",
            config=botocore.config.Config(s3={"addressing_style": "path"}),
        )

    def __init_vector_store(self, vector_store_bucket):
        self.db = lancedb.connect(f"s3://{vector_store_bucket}/")
        self.table = "knowledge_base"

    def __init_openai(self):
        openai_base_url = "https://api.openai.com/v1/"
        if len(os.environ["OPENAI_BASE_URL"]):
            openai_base_url = os.environ["OPENAI_BASE_URL"]
        if len(os.environ["OPENAI_PROXY"]):
            self.openai = OpenAI(
                http_client=httpx.Client(proxy=os.environ["OPENAI_PROXY"]),
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=openai_base_url,
            )
        else:
            self.openai = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=openai_base_url,
            )

    def load_pdf(self, pdf_path):
        logging.info(f"Loading pdf file: {pdf_path}")
        if not os.path.exists(pdf_path):
            print(f"The pdf file does not exist: {pdf_path}")
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
        print(f"Extracted {len(words)} words from the pdf file: {pdf_name}")
        chunks = get_text_chunks(text, self.chunk_size)
        logging.info(
            f"Extracted {len(chunks)} text chunks from the pdf file: {pdf_path}"
        )
        print(f"Extracted {len(chunks)} text chunks from the pdf file: {pdf_name}")

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

        print(f"The pdf file has been loaded successfully: {pdf_name}")
        logging.info(f"The pdf file has been loaded successfully: {pdf_path}")

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

    def load_folder(self, folder_path):
        logging.info(f"Loading pdf folder: {folder_path}")
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print("The pdf folder is invalid.")
            logging.info(f"The pdf folder is invalid: {folder_path}")
            return

        pdf_files = find_pdf_files(folder_path)
        if len(pdf_files) == 0:
            print("No pdf files found in the folder.")
            logging.info(f"No pdf files found in the folder: {folder_path}")
            return

        for pdf_path in pdf_files:
            self.load_pdf(pdf_path)

        print(f"Loaded {len(pdf_files)} pdf files in the folder: {folder_path}")
        logging.info(f"Loaded {len(pdf_files)} pdf files in the folder: {folder_path}")

    def create_index(self):
        try:
            table = self.db.create_table(self.table, schema=self.schema, exist_ok=True)
            start = time.time()
            print("Creating index...")
            logging.info("Creating index...")
            table.create_index(vector_column_name="embedding")
            print(f"Done. Indexing time: {time.time() - start}")
            logging.info(f"Done. Indexing time: {time.time() - start}")
        except Exception as e:
            print(e)
            logging.error(e)
        return

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
            pdf_number = res.shape[0]
            sum_chunk_size = res[0].sum()
            print(f"Total number of pdf files: {pdf_number}")
            print(f"Total number of text chunks: {sum_chunk_size}")
        except Exception as e:
            print(e)
            logging.error(e)

    def clear_knowledge_base(self):
        print("Are you sure to clear all data in S3? [Y/n] ", end="")
        clear = input()
        if clear.lower() != "y":
            return
        logging.info(f"Deleting table knowledge_base in LanceDB")
        try:
            self.db.drop_table(self.table)
        except Exception as e:
            print(e)
            logging.error(e)
        try:
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
    logging.info("RAG initialized")

    print("Welcome to the chatbot!")
    while True:
        print("\noption 1: load new pdf file")
        print("option 2: load folder of pdf files")
        print("option 3: show knowledge base infos")
        print("option 4: create index")
        print("option 5: start chatting")
        print("option 6: clear knowledge base")
        print("option 7: exit\n")
        option = input("Please select an option: ")
        if option == "1":
            pdf_path = input("Please enter the path of the pdf file: ")
            rag.load_pdf(pdf_path)
        elif option == "2":
            folder_path = input("Please enter the path of the folder: ")
            rag.load_folder(folder_path)
        elif option == "3":
            rag.show_knowledge_base_infos()
        elif option == "4":
            rag.create_index()
        elif option == "5":
            rag.chat()
        elif option == "6":
            rag.clear_knowledge_base()
        elif option == "7":
            break
        else:
            print("Invalid option!")

    print("Goodbye!")
    logging.info("RAG exited")
