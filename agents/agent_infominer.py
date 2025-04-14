from langchain_core.prompts import PromptTemplate
from datetime import datetime, timedelta
import json
from filelock import FileLock
from openai import OpenAI
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import Config
os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY

web_search_prompt_pyod = PromptTemplate.from_template("""
   You are a machine learning expert and will assist me with researching a specific use of a deep learning model in PyOD. Here is the official document you should refer to: https://pyod.readthedocs.io/en/latest/pyod.models.html
   I want to run `{algorithm_name}`. What is the Initialization function, parameters and Attributes? 
   Briefly return realted document content.
""")
web_search_prompt_pygod = PromptTemplate.from_template("""
   You are a machine learning expert and will assist me with researching a specific use of a deep learning model in PyGOD. Here is the official document you should refer to: https://docs.pygod.org/en/latest/pygod.detector.html
   I want to run `{algorithm_name}`. What is the Initialization function, parameters and Attributes? 
   Briefly return realted document content.
""")


class AgentInfominer:
    def __init__(self):
        pass

    def query_docs(self, algorithm, vectorstore, package_name,cache_path = "cache.json"):
        """Searches for relevant documentation with caching, expiration, and thread-safe cache writes."""

        lock_path = cache_path + ".lock"
        lock = FileLock(lock_path)

        # Step 1: Ensure cache file exists
        if not os.path.exists(cache_path):
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

        # Step 2: Use lock to safely read and write to cache
        with lock:
            # Load cache
            with open(cache_path, "r", encoding="utf-8") as f:
                try:
                    cache = json.load(f)
                except json.JSONDecodeError:
                    print("[Cache Error] cache.json is corrupted. Reinitializing...")
                    cache = {}

            # Check cache entry
            if algorithm in cache:
                try:
                    cached_time = datetime.fromisoformat(cache[algorithm]["query_datetime"])
                    if datetime.now() - cached_time < timedelta(days=7):
                        print(f"[Cache Hit] Using recent cache for {algorithm}")
                        print(cache[algorithm]["document"])
                        return cache[algorithm]["document"]
                    else:
                        print(f"[Cache Expired] Re-querying {algorithm}")
                except Exception:
                    print(f"[Cache Warning] Datetime parse error for {algorithm}, re-querying.")

        # Step 3: Run actual query outside lock (non-blocking for others)
        client = OpenAI()
        prompt_temp = web_search_prompt_pyod if package_name == "pyod" else web_search_prompt_pygod
        response = client.responses.create(
            model="gpt-4o",
            tools=[{"type": "web_search_preview"}],
            input=prompt_temp.invoke({"algorithm_name": algorithm}).to_string(),
            max_output_tokens=2024
        )
        algorithm_doc = response.output_text

        # Query using RAG
        # query = ""
        # if package_name == "pyod":
        #    query = f"class pyod.models.{algorithm}.{algorithm}"
        # else:
        #    query = f"class pygod.detector.{algorithm}"
        # doc_list = vectorstore.similarity_search(query, k=3)
        # algorithm_doc = "\n\n".join([doc.page_content for doc in doc_list])

        if not algorithm_doc:
            print("Error in response for " + algorithm)
            print(response)
            return ""
        print(algorithm_doc)

        # Step 4: Re-lock and write updated cache
        with lock:
            with open(cache_path, "r", encoding="utf-8") as f:
                try:
                    cache = json.load(f)
                except json.JSONDecodeError:
                    cache = {}

            cache[algorithm] = {
                "query_datetime": datetime.now().isoformat(),
                "document": algorithm_doc
            }

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

        print(f"[Cache Updated] Stored new documentation for {algorithm}")
        return algorithm_doc

if __name__ == "__main__":
    agent = AgentInfominer()
    # Example usage
    algorithm = "lscp"
    vectorstore = None  # Replace with actual vectorstore object
    package_name = "pygod"
    doc = agent.query_docs(algorithm, vectorstore, package_name)