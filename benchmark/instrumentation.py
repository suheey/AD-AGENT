import sys
import os
# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Now use absolute import



import time
import tiktoken
from langchain_openai import ChatOpenAI
from agents.agent_info_miner import AgentInfoMiner
from agents.agent_code_generator import AgentCodeGenerator
from pydantic import Field


class InstrumentedChatOpenAI(ChatOpenAI):
    class Config:
        extra = 'allow'

    # these will hold cumulative counts
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)

    def _call(self, messages, **kwargs):
        # call through to LangChain's ChatOpenAI
        response = super()._call(messages, **kwargs)

        usage = getattr(response, "usage", None)
        if usage:
            # API already tells us these counts
            self.prompt_tokens     += getattr(usage, "prompt_tokens", 0)
            self.completion_tokens += getattr(usage, "completion_tokens", 0)
            self.total_tokens      += getattr(usage, "total_tokens", 0)
        else:
            # fallback to tiktoken if for some reason usage is missing
            encoder = tiktoken.encoding_for_model(self.model_name)
            prompt_count = sum(len(encoder.encode(m.get("content",""))) for m in messages)
            completion_count = len(encoder.encode(response.content or ""))
            self.prompt_tokens     += prompt_count
            self.completion_tokens += completion_count
            self.total_tokens      += prompt_count + completion_count

        return response

    def get_token_counts(self):
        """
        Returns a tuple (prompt_tokens, completion_tokens, total_tokens)
        """
        return (
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
        )



class InstrumentedInfoMiner(AgentInfoMiner):  # type: ignore
    """
    Wraps AgentInfoMiner.query_docs to measure duration of doc retrieval.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_query_duration = 0.0

    def query_docs(self, algorithm, vectorstore, package_name, cache_path="cache.json"):
        start = time.perf_counter()
        doc = super().query_docs(algorithm, vectorstore, package_name, cache_path)
        end = time.perf_counter()
        self.last_query_duration = end - start
        return doc


class InstrumentedCodeGenerator(AgentCodeGenerator):  # type: ignore
    """
    Wraps AgentCodeGenerator.generate_code and revise_code to measure generation durations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_generation_duration = 0.0
        self.last_revision_duration = 0.0

    def generate_code(
        self,
        algorithm,
        data_path_train,
        data_path_test,
        algorithm_doc,
        input_parameters,
        package_name
    ) -> str:
        start = time.perf_counter()
        code = super().generate_code(
            algorithm,
            data_path_train,
            data_path_test,
            algorithm_doc,
            input_parameters,
            package_name
        )
        end = time.perf_counter()
        self.last_generation_duration = end - start
        return code

    def revise_code(self, code_quality, algorithm_doc: str) -> str:
        start = time.perf_counter()
        code = super().revise_code(code_quality, algorithm_doc)
        end = time.perf_counter()
        self.last_revision_duration = end - start
        return code

