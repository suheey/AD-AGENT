import logging, sys, operator, asyncio, os
from typing import TypedDict, Annotated, Sequence, List, Tuple, Any

from config.config import Config
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

# ========== langgraph ==========
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

# ========== business agents ==========
from agents.agent_preprocessor import AgentPreprocessor
from agents.agent_selector     import AgentSelector
from agents.agent_infominer    import AgentInfominer
from agents.agent_coder        import AgentCoder
from agents.agent_reviewer     import AgentReviewer
from agents.agent_evaluator    import AgentEvaluator      
from entity.code_quality       import CodeQuality

# ------------------------------------------------------------------
# Full state
# ------------------------------------------------------------------
class FullToolState(TypedDict):
    messages        : Annotated[Sequence[Any], operator.add]
    current_tool    : str
    input_parameters: dict
    data_path_train : str
    data_path_test  : str
    package_name    : str
    agent_infominer : Any
    agent_coder     : Any
    agent_reviewer  : Any
    agent_evaluator : Any          
    vectorstore     : Any
    code_quality    : Any | None
    should_rerun    : bool
    agent_preprocessor: Any
    agent_selector  : Any | None
    experiment_config: dict | None
    results         : List[Tuple[str, Any]] | None
    algorithm_doc   : str | None

# ------------------------------------------------------------------
# Node: Preprocessor
# ------------------------------------------------------------------
def call_preprocessor(state: FullToolState) -> dict:
    preprocessor = state["agent_preprocessor"]
    print("\n=== [Preprocessor] Processing user input ===")
    preprocessor.run_chatbot()
    state["experiment_config"] = preprocessor.experiment_config
    print("\n=== [Preprocessor] User input processing complete ===")
    return state

# ------------------------------------------------------------------
# Node: Selector
# ------------------------------------------------------------------
def call_selector(state: FullToolState) -> dict:
    print("\n=== [Selector] Processing user input ===")
    if state["experiment_config"] is None:
        raise ValueError("experiment_config not set, run preprocessor first!")
    print("\n=== [Selector] Selecting package & algorithm ===")
    selector = AgentSelector(state["experiment_config"])
    state.update(
        agent_selector = selector,
        input_parameters = selector.parameters,
        data_path_train = selector.data_path_train,
        data_path_test  = selector.data_path_test,
        package_name    = selector.package_name,
        vectorstore     = selector.vectorstore
    )
    print("\n=== [Selector] Selection complete ===")
    return state

# ------------------------------------------------------------------
# Node: Informiner
# ------------------------------------------------------------------
def call_informiner(state: FullToolState) -> dict:
    print(f"\n=== [Informiner] Querying documentation for {state['current_tool']} ===")
    doc = state["agent_infominer"].query_docs(
        state["current_tool"],
        state["vectorstore"],
        state["package_name"]
    )
    print(f"\n=== [Informiner] Documentation retrieved for {state['current_tool']} ===")
    return {"algorithm_doc": doc}

# ------------------------------------------------------------------
# Node: Coder  (generate / revise, **no execution**)
# ------------------------------------------------------------------
def call_coder_for_single_tool(state: FullToolState) -> dict:
    coder = state["agent_coder"]
    tool  = state["current_tool"]

    # generate code || revise code
    if state["code_quality"] is None:
        print(f"\n=== [Coder] Generating code for {tool} ===")
        code = coder.generate_code(
            algorithm       = tool,
            data_path_train = state["data_path_train"],
            data_path_test  = state["data_path_test"],
            algorithm_doc   = state["algorithm_doc"],
            input_parameters= state["input_parameters"],
            package_name    = state["package_name"]
        )
        cq = CodeQuality(code=code, algorithm=tool,
                         error_message="", auroc=-1, auprc=-1,
                         error_points=[], review_count=0)
    else:
        print( f"\n=== [Coder] Revising code for {tool} ===")
        cq = state["code_quality"]
        code = coder.revise_code(cq, state["algorithm_doc"])
        cq.code = code                                 # cover new code

    return {"code_quality": cq}

# ------------------------------------------------------------------
# Node: Reviewer  (synthetic‑data test)
# ------------------------------------------------------------------
def call_reviewer_for_single_tool(state: FullToolState) -> dict:
    reviewer = state["agent_reviewer"]
    cq       = state["code_quality"]
    tool     = state["current_tool"]

    print(f"\n=== [Reviewer] Unit test for {tool} ===")
    cq.error_message = reviewer.test_code(cq.code, tool, state["package_name"])

    if cq.error_message:
        cq.review_count += 1
    print(f"\n=== [Reviewer] Synthetic test completed for {tool} ===")
    return {"code_quality": cq}

# ------------------------------------------------------------------
# Node: Decider  (branch: rerun | evaluate)
# ------------------------------------------------------------------
def decide_next(state: FullToolState) -> dict:
    cq = state["code_quality"]
    need_rerun = bool(cq.error_message) and cq.review_count < 2
    return {"route": "coder" if need_rerun else "evaluator"}

def route_selector(state: FullToolState):
    return state["route"]

# ------------------------------------------------------------------
# Node: Evaluator  (real‑data execution & metrics)
# ------------------------------------------------------------------
def call_evaluator_for_single_tool(state: FullToolState) -> dict:
    evaluator = state["agent_evaluator"]
    cq        = state["code_quality"]
    tool      = state["current_tool"]

    print(f"\n=== [Evaluator] Real‑data run for {tool} ===")
    final_cq = evaluator.execute_code(cq.code, tool)

    # keep review_count
    final_cq.review_count = cq.review_count
    return {"code_quality": final_cq}

# ------------------------------------------------------------------
# Build single‑tool StateGraph
# ------------------------------------------------------------------
single_tool_graph = StateGraph(FullToolState)

single_tool_graph.add_node("informiner", call_informiner)
single_tool_graph.add_node("coder",      call_coder_for_single_tool)
single_tool_graph.add_node("reviewer",   call_reviewer_for_single_tool)
single_tool_graph.add_node("decider",    decide_next)
single_tool_graph.add_node("evaluator",  call_evaluator_for_single_tool)

single_tool_graph.set_entry_point("informiner")
single_tool_graph.add_edge("informiner", "coder")
single_tool_graph.add_edge("coder",      "reviewer")
single_tool_graph.add_edge("reviewer",   "decider")
single_tool_graph.add_conditional_edges(
    "decider", route_selector,
    {"coder": "coder", "evaluator": "evaluator"}
)
single_tool_graph.add_edge("evaluator", END)

compiled_single_tool = single_tool_graph.compile()

# ------------------------------------------------------------------
# process_all_tools
# ------------------------------------------------------------------
def process_all_tools(state: FullToolState) -> dict:
    if not state["agent_selector"]:
        raise ValueError("agent_selector is not set!")
    tools = state["agent_selector"].tools
    if not tools:
        state["results"] = []
        return state

    async def run_tool(tool):
        tool_state = state.copy()
        tool_state.update(
            current_tool = tool,
            code_quality = None,
            should_rerun = False
        )
        return tool, await asyncio.to_thread(
            compiled_single_tool.invoke,
            tool_state,
            config={"recursion_limit": 20}
        )

    results = []
    if "-p" in sys.argv:          # parallel
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(asyncio.gather(
            *(run_tool(t) for t in tools)
        ))
        loop.close()
    else:                         # sequential
        for t in tools:
            results.append(asyncio.run(run_tool(t)))

    state["results"] = results
    return state

# ------------------------------------------------------------------
# Build full process graph
# ------------------------------------------------------------------
full_graph = StateGraph(FullToolState)
full_graph.add_node("preprocessor",     call_preprocessor)
full_graph.add_node("selector",         call_selector)
full_graph.add_node("process_all_tools",process_all_tools)

full_graph.set_entry_point("preprocessor")
full_graph.add_edge("preprocessor",     "selector")
full_graph.add_edge("selector",         "process_all_tools")

compiled_full_graph = full_graph.compile()

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
async def main():
    # clean loader scripts
    for f in ("train_data_loader.py","test_data_loader.py",
              "head_train_data_loader.py","head_test_data_loader.py"):
        if os.path.exists(f): os.remove(f)

    state: FullToolState = {
        "messages"        : [],
        "current_tool"    : "",
        "input_parameters": {},
        "data_path_train" : "",
        "data_path_test"  : "",
        "package_name"    : "",
        "agent_infominer" : AgentInfominer(),
        "agent_coder"     : AgentCoder(),
        "agent_reviewer"  : AgentReviewer(),
        "agent_evaluator" : AgentEvaluator(),     # ★
        "vectorstore"     : None,
        "code_quality"    : None,
        "should_rerun"    : False,
        "agent_preprocessor": AgentPreprocessor(),
        "agent_selector"  : None,
        "experiment_config": None,
        "results"         : None,
        "algorithm_doc"   : None,
    }

    print("\n=== [Main] Starting full pipeline ===")
    final_state = await asyncio.to_thread(
        compiled_full_graph.invoke,
        state,
        config={"recursion_limit": 20}
    )
    print("\n=== [Main] Pipeline finished ===")

    # ---------- output results ----------
    for tool, tstate in final_state.get("results", []):
        cq: CodeQuality | None = tstate.get("code_quality")
        if cq and not cq.error_message:
            print(f"[{tool}] AUROC: {cq.auroc:.4f}  AUPRC: {cq.auprc:.4f}")
        else:
            print(f"[{tool}] Error: {cq.error_message if cq else 'Unknown'}")

if __name__ == "__main__":
    asyncio.run(main())
