#!/usr/bin/env python3
# benchmark_pre_inf_codegen.py

import sys
import os
import pandas as pd
import logging

from config.config import Config
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
logging.basicConfig(level=logging.ERROR)

# ========== Instrumentation ==========
import openai
import langchain_openai
from typing import ClassVar
import time
import glob

def _unpack_usage(usage):
    """Normalize usage fields for both dict and object forms."""
    if usage is None:
        return 0, 0, 0
    if isinstance(usage, dict):
        pt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        ct = usage.get("completion_tokens", usage.get("output_tokens", 0))
        tt = usage.get("total_tokens", pt + ct)
    else:
        pt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0)
        ct = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0)
        tt = getattr(usage, "total_tokens", pt + ct)
    return pt, ct, tt

# 1) Instrument ChatOpenAI for all entrypoints
BaseChat = langchain_openai.ChatOpenAI

class InstrumentedChatOpenAI(BaseChat):
    prompt_tokens:     ClassVar[int] = 0
    completion_tokens: ClassVar[int] = 0
    total_tokens:      ClassVar[int] = 0

    def _call(self, messages, **kwargs):
        resp = super()._call(messages, **kwargs)
        if (u := getattr(resp, "usage", None)):
            pt, ct, tt = _unpack_usage(u)
            InstrumentedChatOpenAI.prompt_tokens     += pt
            InstrumentedChatOpenAI.completion_tokens += ct
            InstrumentedChatOpenAI.total_tokens      += tt
        return resp

    def __call__(self, *args, **kwargs):
        resp = super().__call__(*args, **kwargs)
        if (u := getattr(resp, "usage", None)):
            pt, ct, tt = _unpack_usage(u)
            InstrumentedChatOpenAI.prompt_tokens     += pt
            InstrumentedChatOpenAI.completion_tokens += ct
            InstrumentedChatOpenAI.total_tokens      += tt
        return resp

    def invoke(self, prompt, **kwargs):
        resp = super().invoke(prompt, **kwargs)
        if (u := getattr(resp, "usage", None)):
            pt, ct, tt = _unpack_usage(u)
            InstrumentedChatOpenAI.prompt_tokens     += pt
            InstrumentedChatOpenAI.completion_tokens += ct
            InstrumentedChatOpenAI.total_tokens      += tt
        return resp

    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        usage = result.llm_output.get("usage") or result.llm_output.get("token_usage")
        if usage:
            pt, ct, tt = _unpack_usage(usage)
            InstrumentedChatOpenAI.prompt_tokens     += pt
            InstrumentedChatOpenAI.completion_tokens += ct
            InstrumentedChatOpenAI.total_tokens      += tt
        return result

    def _call(self, messages, **kwargs):
        # Add component tag to usage tracking
        component = self.metadata.get("component", "unknown")
        print(f"Tracking tokens for: {component}")
        
        
langchain_openai.ChatOpenAI = InstrumentedChatOpenAI

# 1b) Ensure AgentCodeGenerator’s module‐level llm uses our instrumented class
import agents.agent_code_generator as _ac
_ac.llm = InstrumentedChatOpenAI(model="gpt-4o", temperature=0)


# 2) Instrument OpenAI client for processor & InfoMiner
BaseOpenAI = openai.OpenAI

class InstrumentedOpenAI(BaseOpenAI):
    prompt_tokens:     ClassVar[int] = 0
    completion_tokens: ClassVar[int] = 0
    total_tokens:      ClassVar[int] = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # patch chat.completions.create (used by Agentprocessor)
        orig_chat = self.chat.completions.create
        def wrapped_chat(*a, **k):
            resp = orig_chat(*a, **k)
            if (u := getattr(resp, "usage", None)):
                pt, ct, tt = _unpack_usage(u)
                InstrumentedOpenAI.prompt_tokens     += pt
                InstrumentedOpenAI.completion_tokens += ct
                InstrumentedOpenAI.total_tokens      += tt
            return resp
        self.chat.completions.create = wrapped_chat

        # patch responses.create (used by AgentInfoMiner)
        if hasattr(self, "responses") and hasattr(self.responses, "create"):
            orig_resp = self.responses.create
            def wrapped_resp(*a, **k):
                resp = orig_resp(*a, **k)
                if (u := getattr(resp, "usage", None)):
                    pt, ct, tt = _unpack_usage(u)
                    InstrumentedOpenAI.prompt_tokens     += pt
                    InstrumentedOpenAI.completion_tokens += ct
                    InstrumentedOpenAI.total_tokens      += tt
                return resp
            self.responses.create = wrapped_resp

openai.OpenAI = InstrumentedOpenAI

# top‐level chat create also needs patch for any direct calls
global_client = openai.OpenAI()
openai.chat.completions.create = global_client.chat.completions.create

# Helper to reset all counters
def reset_counters():
    for C in (InstrumentedChatOpenAI, InstrumentedOpenAI):
        C.prompt_tokens     = 0
        C.completion_tokens = 0
        C.total_tokens      = 0

# ========== MODIFIED WORKFLOW ==========
from agents.agent_processor import AgentProcessor
from agents.agent_info_miner import AgentInfoMiner
from agents.agent_code_generator import AgentCodeGenerator
from agents.agent_reviewer import AgentReviewer

TARGET_DATASETS = ['arrhythmia', 'glass', 'vowels']
ALGOS = ['MO-GAAL']

rows = []
for dataset in TARGET_DATASETS:
    dataset_path = f"./data/pyod_data/{dataset}.mat"
    
    if not os.path.exists(dataset_path):
        print(f" Dataset {dataset} not found at {dataset_path}")
        continue

    for algo in ALGOS:
        reset_counters()
        start_time = time.perf_counter()
        
        try:
            # ---- Processor ----
            pre = AgentProcessor(model="gpt-4", temperature=0)
            pre.run_chatbot()  # Will be automated via input queue
            pre_in = InstrumentedOpenAI.prompt_tokens + InstrumentedChatOpenAI.prompt_tokens
            pre_out = InstrumentedOpenAI.completion_tokens + InstrumentedChatOpenAI.completion_tokens
            cfg = pre.experiment_config

            # ---- InfoMiner ----
            reset_counters()
            inf = AgentInfoMiner()
            doc = inf.query_docs(algo, None, 'pyod')
            info_in = InstrumentedOpenAI.prompt_tokens
            info_out = InstrumentedOpenAI.completion_tokens

            # ---- Code Generation & Review ----
            reset_counters()
            CodeGenerator = AgentCodeGenerator()
            generated_code = CodeGenerator.generate_code(
                algorithm=algo,
                data_path_train=cfg["dataset_train"],
                data_path_test=cfg["dataset_test"],
                algorithm_doc=doc,
                input_parameters=cfg["parameters"],
                package_name='pyod'
            )
            code_in = InstrumentedChatOpenAI.prompt_tokens
            code_out = InstrumentedChatOpenAI.completion_tokens

            reset_counters()
            reviewer = AgentReviewer()
            reviewer.test_code(generated_code, algo, 'pyod')
            review_in = InstrumentedChatOpenAI.prompt_tokens
            review_out = InstrumentedChatOpenAI.completion_tokens

            elapsed = time.perf_counter() - start_time
            success = True
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            success = False
            print(f" Failed {dataset} - {algo}: {str(e)}")

        rows.append({
            "dataset": dataset,
            "algorithm": algo,
            "pre_in": pre_in,
            "pre_out": pre_out,
            "info_in": info_in,
            "info_out": info_out,
            "code_in": code_in,
            "code_out": code_out,
            "review_in": review_in,
            "review_out": review_out,
            "time_sec": elapsed,
            "success": success
        })

# ========== FORMATTED OUTPUT ==========
df = pd.DataFrame(rows)
print("\n=== Combined Token & Time Metrics ===")
print(df[['dataset', 'algorithm', 
          'pre_in', 'pre_out',
          'info_in', 'info_out',
          'code_in', 'code_out',
          'review_in', 'review_out',
          'time_sec', 'success']])

# Save to CSV
timestamp = time.strftime("%Y%m%d-%H%M%S")
df.to_csv(f"benchmark_{timestamp}.csv", index=False)
print(f"\n Saved results to benchmark_{timestamp}.csv")