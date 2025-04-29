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

langchain_openai.ChatOpenAI = InstrumentedChatOpenAI

# 1b) Ensure AgentCoder’s module‐level llm uses our instrumented class
import agents.agent_coder as _ac
_ac.llm = InstrumentedChatOpenAI(model="gpt-4o", temperature=0)


# 2) Instrument OpenAI client for Preprocessor & InfoMiner
BaseOpenAI = openai.OpenAI

class InstrumentedOpenAI(BaseOpenAI):
    prompt_tokens:     ClassVar[int] = 0
    completion_tokens: ClassVar[int] = 0
    total_tokens:      ClassVar[int] = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # patch chat.completions.create (used by AgentPreprocessor)
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

# ========== Agents & Workflow ==========
from agents.agent_preprocessor import AgentPreprocessor
from agents.agent_infominer    import AgentInfoMiner
from agents.agent_coder        import AgentCoder

ALGOS = [
    'MO-GAAL'
    # ,'SO-GAAL','AutoEncoder','VAE','AnoGAN',
    # 'DeepSVDD','ALAD','AE1SVM','DevNet','LUNAR'
]
TRAIN_PATH = './data/glass.mat'

rows = []
for algo in ALGOS:
    print(f"\n=== Running {algo} ===")

    # ---- Preprocessor ----
    reset_counters()
    pre = AgentPreprocessor(model="gpt-4", temperature=0)
    print(f"Type: Run {algo} on {TRAIN_PATH} with contamination=0.1")
    pre.run_chatbot()
    pre_in  = InstrumentedOpenAI.prompt_tokens + InstrumentedChatOpenAI.prompt_tokens
    pre_out = InstrumentedOpenAI.completion_tokens + InstrumentedChatOpenAI.completion_tokens
    cfg     = pre.experiment_config

    # ---- InfoMiner ----
    reset_counters()
    inf = AgentInfoMiner()
    doc = inf.query_docs(algo, None, 'pyod')
    info_in  = InstrumentedOpenAI.prompt_tokens
    info_out = InstrumentedOpenAI.completion_tokens

    # ---- Coder ----
    reset_counters()
    coder = AgentCoder()
    _ = coder.generate_code(
        algorithm        = algo,
        data_path_train  = cfg["dataset_train"],
        data_path_test   = cfg["dataset_test"],
        algorithm_doc    = doc,
        input_parameters = cfg["parameters"],
        package_name     = 'pyod'
    )
    code_in  = InstrumentedChatOpenAI.prompt_tokens
    code_out = InstrumentedChatOpenAI.completion_tokens

    rows.append({
        "algorithm": algo,
        "pre_in":    pre_in,
        "pre_out":   pre_out,
        "info_in":   info_in,
        "info_out":  info_out,
        "code_in":   code_in,
        "code_out":  code_out
    })

# ========== Report ==========
df = pd.DataFrame(rows).set_index("algorithm")
print("\n--- Token Usage Table ---")
print(df)
df.to_csv("token_usage_pre_inf_codegen.csv", index=True)
