#!/usr/bin/env python3
import os
import sys
import time
import json
import builtins
import logging
from typing import ClassVar
import torch
import numpy as np
import pandas as pd
import glob

# ─── Bootstrap API key & logging ───────────────────────────────────────────────
from config.config import Config
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

# ─── Instrumentation helpers ──────────────────────────────────────────────────
def _unpack_usage(usage):
    if usage is None:
        return 0, 0, 0
    if isinstance(usage, dict):
        pt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        ct = usage.get("completion_tokens", usage.get("output_tokens", 0))
        tt = usage.get("total_tokens", pt + ct)
    else:
        pt = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0)
        ct = getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0)
        tt = getattr(usage, "total_tokens", pt + ct)
    return pt, ct, tt

# ─── Patch LangChain's ChatOpenAI ────────────────────────────────────────────
import langchain_openai
BaseChat = langchain_openai.ChatOpenAI

class InstrumentedChatOpenAI(BaseChat):
    prompt_tokens:     ClassVar[int] = 0
    completion_tokens: ClassVar[int] = 0
    total_tokens:      ClassVar[int] = 0

    def _call(self, messages, **kwargs):
        resp = super()._call(messages, **kwargs)
        if (u := getattr(resp, "usage", None)):
            pt, ct, _ = _unpack_usage(u)
            InstrumentedChatOpenAI.prompt_tokens     += pt
            InstrumentedChatOpenAI.completion_tokens += ct
        return resp

    def __call__(self, *args, **kwargs):
        resp = super().__call__(*args, **kwargs)
        if (u := getattr(resp, "usage", None)):
            pt, ct, _ = _unpack_usage(u)
            InstrumentedChatOpenAI.prompt_tokens     += pt
            InstrumentedChatOpenAI.completion_tokens += ct
        return resp

    def invoke(self, prompt, **kwargs):
        resp = super().invoke(prompt, **kwargs)
        if (u := getattr(resp, "usage", None)):
            pt, ct, _ = _unpack_usage(u)
            InstrumentedChatOpenAI.prompt_tokens     += pt
            InstrumentedChatOpenAI.completion_tokens += ct
        return resp

    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        usage = result.llm_output.get("usage") or result.llm_output.get("token_usage")
        if usage:
            pt, ct, _ = _unpack_usage(usage)
            InstrumentedChatOpenAI.prompt_tokens     += pt
            InstrumentedChatOpenAI.completion_tokens += ct
        return result

langchain_openai.ChatOpenAI = InstrumentedChatOpenAI

# ─── Patch openai.OpenAI ─────────────────────────────────────────────────────
import openai
BaseOpenAI = openai.OpenAI

class InstrumentedOpenAI(BaseOpenAI):
    prompt_tokens:     ClassVar[int] = 0
    completion_tokens: ClassVar[int] = 0
    total_tokens:      ClassVar[int] = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        orig = self.chat.completions.create
        def wrapped(*a, **k):
            resp = orig(*a, **k)
            if (u := getattr(resp, "usage", None)):
                pt, ct, _ = _unpack_usage(u)
                InstrumentedOpenAI.prompt_tokens     += pt
                InstrumentedOpenAI.completion_tokens += ct
            return resp
        self.chat.completions.create = wrapped

        if hasattr(self, "responses") and hasattr(self.responses, "create"):
            orig2 = self.responses.create
            def wrapped2(*a, **k):
                resp = orig2(*a, **k)
                if (u := getattr(resp, "usage", None)):
                    pt, ct, _ = _unpack_usage(u)
                    InstrumentedOpenAI.prompt_tokens     += pt
                    InstrumentedOpenAI.completion_tokens += ct
                return resp
            self.responses.create = wrapped2

openai.OpenAI = InstrumentedOpenAI
_global_client = openai.OpenAI()
openai.chat.completions.create = _global_client.chat.completions.create

def reset_counters():
    for C in (InstrumentedChatOpenAI, InstrumentedOpenAI):
        C.prompt_tokens     = 0
        C.completion_tokens = 0
        C.total_tokens      = 0

# ─── Import your agents ──────────────────────────────────────────────────────
from agents.agent_processor     import AgentProcessor
from agents.agent_selector      import AgentSelector
from agents.agent_info_miner    import AgentInfoMiner
from agents.agent_code_generator import AgentCodeGenerator
from agents.agent_reviewer      import AgentReviewer

# ─── Benchmark configuration ──────────────────────────────────────────────────
DATA_DIR = "./data/pygod_data"
ALGOS    = [
    "AdONE", "ANOMALOUS", "AnomalyDAE", "CONAD", "DOMINAT",
    "DONE", "GAAN", "GUIDE", "Radar", "SCAN"
]

def run_one(algo: str, train_file: str):
    workflow_start = time.perf_counter()
    
    row = {
        "algorithm": algo,
        "dataset": os.path.basename(train_file),
        "success": True,
        "error": "",
        "time_sec": 0,
        "total_time_sec": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0
    }

    # ─ Processor ──────────────────────────────────────────────────────────────
    reset_counters()
    prompt = f"Run {algo} on {train_file}"
    orig_input = builtins.input
    def fake_input(prompt_text=""):
        print(prompt_text, end="")
        return prompt
    builtins.input = fake_input

    proc_start = time.perf_counter()
    proc = AgentProcessor(model="gpt-4", temperature=0)
    try:
        proc.run_chatbot()
        row["success"] = True
        row["error"]   = ""
    except Exception as e:
        row["success"] = False
        row["error"]   = str(e)
    finally:
        builtins.input = orig_input
    proc_end = time.perf_counter()

    row["time_sec"] = proc_end - proc_start
    row["total_input_tokens"] += InstrumentedOpenAI.prompt_tokens + InstrumentedChatOpenAI.prompt_tokens
    row["total_output_tokens"] += InstrumentedOpenAI.completion_tokens + InstrumentedChatOpenAI.completion_tokens

    cfg = proc.experiment_config

    # ─ Selector ────────────────────────────────────────────────────────────────
    reset_counters()
    sel = AgentSelector(cfg)
    row["total_input_tokens"] += InstrumentedChatOpenAI.prompt_tokens + InstrumentedOpenAI.prompt_tokens
    row["total_output_tokens"] += InstrumentedChatOpenAI.completion_tokens + InstrumentedOpenAI.completion_tokens

    # ─ InfoMiner ───────────────────────────────────────────────────────────────
    reset_counters()
    inf = AgentInfoMiner()
    doc = inf.query_docs(algo, sel.vectorstore, sel.package_name)
    row["total_input_tokens"] += InstrumentedOpenAI.prompt_tokens
    row["total_output_tokens"] += InstrumentedOpenAI.completion_tokens

    # ─ Code Generator ─────────────────────────────────────────────────────────
    reset_counters()
    cg = AgentCodeGenerator()
    code = cg.generate_code(
        algorithm        = algo,
        data_path_train  = train_file,
        data_path_test   = "",  # No test file
        algorithm_doc    = doc,
        input_parameters = sel.parameters,
        package_name     = sel.package_name
    )
    row["total_input_tokens"] += InstrumentedChatOpenAI.prompt_tokens
    row["total_output_tokens"] += InstrumentedChatOpenAI.completion_tokens

    # Save the generated code to a file
    folder = "./generated_scripts"
    os.makedirs(folder, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(train_file))[0]
    path = os.path.join(folder, f"{algo}_{dataset_name}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"\n=== [Code Generator] Saved code to {path} ===")

    # ─ Reviewer ────────────────────────────────────────────────────────────────
    reset_counters()
    rev = AgentReviewer()
    err = rev.test_code(code, algo, sel.package_name)
    row["total_input_tokens"] += InstrumentedChatOpenAI.prompt_tokens
    row["total_output_tokens"] += InstrumentedChatOpenAI.completion_tokens
    if err:
        row["success"] = False
        row["error"]   = err

    workflow_end = time.perf_counter()
    row["total_time_sec"] = workflow_end - workflow_start

    return row

def main():
    all_results = []
    train_files = glob.glob(os.path.join(DATA_DIR, "*.pt"))  # PyGOD uses .pt files
    
    for algo in ALGOS:
        print(f"\n=== Running {algo} ===")
        for train_file in train_files:
            print(f"\n=== Processing {os.path.basename(train_file)} ===")
            result = run_one(algo, train_file)
            all_results.append(result)

    df = pd.DataFrame(all_results)
    
    # Create a 2D table with algorithms as rows and datasets as columns
    # For each metric (time, tokens in/out)
    metrics = ['time_sec', 'total_time_sec', 'total_input_tokens', 'total_output_tokens']
    
    for metric in metrics:
        pivot_df = df.pivot(index='algorithm', columns='dataset', values=metric)
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_fn = f"benchmark_pygod_{metric}_{ts}.csv"
        pivot_df.to_csv(out_fn)
        print(f"\nSaved {metric} results to {out_fn}")

    # Save full results
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_fn = f"benchmark_pygod_full_{ts}.csv"
    df.to_csv(out_fn, index=False)
    print(f"\nSaved full results to {out_fn}")

if __name__ == "__main__":
    main() 