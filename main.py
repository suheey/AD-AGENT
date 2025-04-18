# import logging
# import sys
# import operator
# import asyncio
# import os
# from config.config import Config
# from typing import TypedDict, Annotated, Sequence, List, Tuple, Any

# os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY
# logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

# # ===== langgraph related =====
# from langchain_core.messages import BaseMessage
# from langgraph.graph import StateGraph, END

# # ===== Business logic related =====
# from agents.agent_preprocessor import AgentPreprocessor
# from agents.agent_selector import AgentSelector
# from agents.agent_infominer import AgentInfominer
# from agents.agent_coder import AgentCoder
# from agents.agent_reviewer import AgentReviewer
# from entity.code_quality import CodeQuality

# from typing import Annotated, Sequence, TypedDict, Any, List, Tuple
# import operator

# # =================================================================
# # Define full state, adding `results` and `algorithm_doc`
# # =================================================================
# class FullToolState(TypedDict):
#     messages: Annotated[Sequence[Any], operator.add]
#     current_tool: str
#     input_parameters: dict
#     data_path_train: str
#     data_path_test: str
#     package_name: str
#     agent_infominer: Any
#     agent_coder: Any
#     agent_reviewer: Any
#     vectorstore: Any
#     code_quality: Any | None
#     should_rerun: bool
#     agent_preprocessor: Any
#     agent_selector: Any | None
#     experiment_config: dict | None
#     results: List[Tuple[str, Any]] | None
#     algorithm_doc: str | None  # New field to store queried documentation

# # =================================================================
# # Node: Preprocessor
# # =================================================================
# def call_preprocessor(state: FullToolState) -> dict:
#     preprocessor = state["agent_preprocessor"]
#     print("\n=== [Preprocessor] Starting to process user input ===")
#     preprocessor.run_chatbot()
#     state["experiment_config"] = preprocessor.experiment_config
#     print("\n=== [Preprocessor] User input processing complete ===")
#     return state

# # =================================================================
# # Node: Selector
# # =================================================================
# def call_selector(state: FullToolState) -> dict:
#     if state["experiment_config"] is None:
#         raise ValueError("experiment_config not set, please run the preprocessor first!")
#     print("\n=== [Selector] Starting to select package & algorithm ===")
#     selector_instance = AgentSelector(state["experiment_config"])
#     state["agent_selector"] = selector_instance
#     state["input_parameters"] = selector_instance.parameters
#     state["data_path_train"] = selector_instance.data_path_train
#     state["data_path_test"] = selector_instance.data_path_test
#     state["package_name"] = selector_instance.package_name
#     state["vectorstore"] = selector_instance.vectorstore
#     print("\n=== [Selector] Selection complete ===")
#     return state

# # =================================================================
# # Node: Informiner
# # Queries documentation and updates the state
# # =================================================================
# def call_informiner(state: FullToolState) -> dict:
#     infominer = state["agent_infominer"]
#     algorithm = state["current_tool"]
#     vectorstore = state["vectorstore"]
#     package_name = state["package_name"]

#     print(f"\n=== [Informiner] Querying documentation for {algorithm} ===")
#     doc = infominer.query_docs(algorithm, vectorstore, package_name)
#     print(f"\n=== [Informiner] Documentation retrieved for {algorithm} ===")
#     return {"algorithm_doc": doc}

# # =================================================================
# # Node: Coder
# # Generates and executes code for the selected algorithm
# # =================================================================
# # ----------------- Coder 节点 -----------------
# def call_coder_for_single_tool(state: FullToolState) -> dict:
#     coder = state["agent_coder"]
#     tool  = state["current_tool"]

#     # 第一次：生成；后续：修复
#     if state["code_quality"] is None:
#         print(f"\n=== [Coder] Processing {tool} (first execution) ===")
#         code = coder.generate_code(
#             algorithm=tool,
#             data_path_train=state["data_path_train"],
#             data_path_test=state["data_path_test"],
#             algorithm_doc=state["algorithm_doc"],
#             input_parameters=state["input_parameters"],
#             package_name=state["package_name"]
#         )
#         cq = CodeQuality(code=code, algorithm=tool,
#                          error_message="", auroc=-1, auprc=-1,
#                          error_points=[], review_count=0)
#     else:
#         print(f"\n=== [Coder] Revise code for {tool} ===")
#         code = coder.revise_code(state["code_quality"], state["algorithm_doc"])
#         cq   = state["code_quality"]
#         cq.code = code   # 覆盖为新版
#     print(f"\n=== [Coder] Code generated for {tool} ===")
#     return {"code_quality": cq}

# # ----------------- Reviewer 节点 -----------------
# def call_reviewer_for_single_tool(state: FullToolState) -> dict:
#     reviewer = state["agent_reviewer"]
#     tool     = state["current_tool"]
#     cq       = state["code_quality"]

#     print(f"\n=== [Reviewer] Executing code for {tool} ===")
#     exec_res = reviewer.execute_code(cq.code, tool)

#     # 将执行结果写回原对象（保持引用）
#     cq.error_message = exec_res.error_message
#     cq.auroc         = exec_res.auroc
#     cq.auprc         = exec_res.auprc
#     cq.error_points  = exec_res.error_points

#     print(f"\n=== [Reviewer] Code execution completed for {tool} ===")
#     return {"code_quality": cq}


# # =================================================================
# # Node: Decider
# # Determines whether to re-run the coder
# # =================================================================
# def decide_reviewer_result(state: FullToolState) -> dict:
#     code_quality = state["code_quality"]
#     if code_quality.error_message and code_quality.review_count < 2:
#         return {"should_rerun": True}
#     else:
#         return {"should_rerun": False}

# # Conditional edge selector for rerun logic
# def check_if_need_rerun(state: FullToolState):
#     return "need_rerun" if state["should_rerun"] else "done"

# # =================================================================
# # Build StateGraph for single-tool processing
# # informiner → coder → reviewer → decider
# # =================================================================

# single_tool_graph = StateGraph(FullToolState)

# # Add nodes
# single_tool_graph.add_node("informiner", call_informiner)
# single_tool_graph.add_node("coder", call_coder_for_single_tool)
# single_tool_graph.add_node("reviewer", call_reviewer_for_single_tool)
# single_tool_graph.add_node("decider", decide_reviewer_result)

# # Define flow
# single_tool_graph.set_entry_point("informiner")
# single_tool_graph.add_edge("informiner", "coder")
# single_tool_graph.add_edge("coder", "reviewer")
# single_tool_graph.add_edge("reviewer", "decider")
# single_tool_graph.add_conditional_edges("decider", check_if_need_rerun, {
#     "need_rerun": "coder",
#     "done": END,
# })

# # Compile the graph
# compiled_single_tool = single_tool_graph.compile()

# # =================================================================
# # 4. Node: Process all tools (parallel or sequential)
# #    Get tool list from agent_selector, build per-tool state,
# #    Call single-tool processing graph for each tool, store results in state["results"]
# # =================================================================
# def process_all_tools(state: FullToolState) -> dict:
#     if not state["agent_selector"]:
#         raise ValueError("agent_selector is not set!")
#     tools = state["agent_selector"].tools
#     if not tools:
#         print("No tool list generated. Exiting.")
#         state["results"] = []
#         return state

#     async def process_tool_async(tool, state):
#         tool_state = state.copy()
#         tool_state["current_tool"] = tool
#         tool_state["code_quality"] = None
#         tool_state["should_rerun"] = False
#         final_state = await asyncio.to_thread(
#             compiled_single_tool.invoke,
#             tool_state,
#             config={"recursion_limit": 20}
#         )
#         return tool, final_state

#     results = []
#     use_parallel = "-p" in sys.argv
#     if use_parallel:
#         # Create new event loop and process in parallel
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         tasks = [process_tool_async(tool, state) for tool in tools]
#         results = loop.run_until_complete(asyncio.gather(*tasks))
#         loop.close()
#     else:
#         # Process sequentially
#         for tool in tools:
#             results.append(asyncio.run(process_tool_async(tool, state)))
#     state["results"] = results
#     return state

# # =================================================================
# # 5. Build full process graph: preprocessor → selector → process_all_tools
# #    Combine preprocessing, planning, and tool processing
# # =================================================================
# full_graph = StateGraph(FullToolState)
# full_graph.add_node("preprocessor", call_preprocessor)
# full_graph.add_node("selector", call_selector)
# full_graph.add_node("process_all_tools", process_all_tools)
# full_graph.set_entry_point("preprocessor")
# full_graph.add_edge("preprocessor", "selector")
# full_graph.add_edge("selector", "process_all_tools")
# compiled_full_graph = full_graph.compile()

# # =================================================================
# # 6. Main function
# #    Initialize agents and initial state, call the full process graph,
# #    then print the results of each tool
# # =================================================================
# async def main():
#     if os.path.exists("train_data_loader.py"):
#         os.remove("train_data_loader.py")
#     if os.path.exists("test_data_loader.py"):
#         os.remove("test_data_loader.py")
#     if os.path.exists("head_train_data_loader.py"):
#         os.remove("head_train_data_loader.py")
#     if os.path.exists("head_test_data_loader.py"):
#         os.remove("head_test_data_loader.py")
#     infominer_instance = AgentInfominer()
#     coder_instance = AgentCoder()
#     reviewer_instance = AgentReviewer()
#     preprocessor_instance = AgentPreprocessor()
    
#     initial_state: FullToolState = {
#         "messages": [],
#         "current_tool": "",
#         "input_parameters": {},
#         "data_path_train": "",
#         "data_path_test": "",
#         "agent_infominer": infominer_instance,
#         "agent_coder": coder_instance,
#         "agent_reviewer": reviewer_instance,
#         "vectorstore": None,
#         "code_quality": None,
#         "should_rerun": False,
#         "agent_preprocessor": preprocessor_instance,
#         "agent_selector": None,
#         "experiment_config": None,
#         "results": None,
#         "algorithm_doc": None,
#     }
    
#     print("\n=== [Main] Starting full process graph ===")
#     final_state = await asyncio.to_thread(
#         compiled_full_graph.invoke,
#         initial_state,
#         config={"recursion_limit": 20}
#     )
#     print("\n=== [Main] Full process graph completed ===")
    
#     # Print results for each tool
#     results = final_state.get("results", [])
#     for tool, tool_state in results:
#         code_quality = tool_state.get("code_quality")
#         if code_quality and not code_quality.error_message:
#             print(f"[{tool}] Success, AUROC: {code_quality.auroc:.4f}, AUPRC: {code_quality.auprc:.4f}, Error Points: {[]}")
#         else:
#             err_msg = code_quality.error_message if code_quality else "Unknown"
#             print(f"[{tool}] Failed, error message: {err_msg}")

# if __name__ == "__main__":
#     asyncio.run(main())
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
from agents.agent_evaluator    import AgentEvaluator      # ★ 新增
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
    agent_evaluator : Any          # ★ 新增
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

    # 第一次：生成   后续：修复
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
        cq.code = code                                 # 覆盖新代码

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

    # 保留 review_count
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
# process_all_tools 节点（与之前相同，只改 graph 调用）
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
    # 清理 loader 脚本（方便多次运行）
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

    # ---------- 输出结果 ----------
    for tool, tstate in final_state.get("results", []):
        cq: CodeQuality | None = tstate.get("code_quality")
        if cq and not cq.error_message:
            print(f"[{tool}] AUROC: {cq.auroc:.4f}  AUPRC: {cq.auprc:.4f}")
        else:
            print(f"[{tool}] Error: {cq.error_message if cq else 'Unknown'}")

if __name__ == "__main__":
    asyncio.run(main())
