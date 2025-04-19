import logging
import sys
import operator
import asyncio
import os
from config.config import Config
from typing import TypedDict, Annotated, Sequence, List, Tuple, Any

os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

# ===== langgraph related =====
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

# ===== Business logic related =====
from agents.agent_preprocessor import AgentPreprocessor
from agents.agent_selector import AgentSelector
from agents.agent_infoMiner import AgentInfoMiner
from agents.agent_instructor import AgentInstructor
from agents.agent_reviewer import AgentReviewer
from entity.code_quality import CodeQuality

from typing import Annotated, Sequence, TypedDict, Any, List, Tuple
import operator

# =================================================================
# Define full state, adding `results` and `algorithm_doc`
# =================================================================
class FullToolState(TypedDict):
    messages: Annotated[Sequence[Any], operator.add]
    current_tool: str
    input_parameters: dict
    data_path_train: str
    data_path_test: str
    package_name: str
    agent_InfoMiner: Any
    agent_instructor: Any
    agent_reviewer: Any
    vectorstore: Any
    code_quality: Any | None
    should_rerun: bool
    agent_preprocessor: Any
    agent_selector: Any | None
    experiment_config: dict | None
    results: List[Tuple[str, Any]] | None
    algorithm_doc: str | None  # New field to store queried documentation

# =================================================================
# Node: Preprocessor
# =================================================================
def call_preprocessor(state: FullToolState) -> dict:
    preprocessor = state["agent_preprocessor"]
    print("\n=== [Preprocessor] Starting to process user input ===")
    preprocessor.run_chatbot()
    state["experiment_config"] = preprocessor.experiment_config
    print("\n=== [Preprocessor] User input processing complete ===")
    return state

# =================================================================
# Node: Selector
# =================================================================
def call_selector(state: FullToolState) -> dict:
    if state["experiment_config"] is None:
        raise ValueError("experiment_config not set, please run the preprocessor first!")
    print("\n=== [Selector] Starting to generate idea space ===")
    selector_instance = AgentSelector(state["experiment_config"])
    state["agent_selector"] = selector_instance
    state["input_parameters"] = selector_instance.parameters
    state["data_path_train"] = selector_instance.data_path_train
    state["data_path_test"] = selector_instance.data_path_test
    state["package_name"] = selector_instance.package_name
    state["vectorstore"] = selector_instance.vectorstore
    print("\n=== [Selector] Idea space generation complete ===")
    return state

# =================================================================
# Node: InfoMiner
# Queries documentation and updates the state
# =================================================================
def call_InfoMiner(state: FullToolState) -> dict:
    InfoMiner = state["agent_InfoMiner"]
    algorithm = state["current_tool"]
    vectorstore = state["vectorstore"]
    package_name = state["package_name"]

    print(f"\n=== [InfoMiner] Querying documentation for {algorithm} ===")
    doc = InfoMiner.query_docs(algorithm, vectorstore, package_name)
    print(f"\n=== [InfoMiner] Documentation retrieved for {algorithm} ===")
    return {"algorithm_doc": doc}

# =================================================================
# Node: Instructor
# Generates and executes code for the selected algorithm
# =================================================================
def call_instructor_for_single_tool(state: FullToolState) -> dict:
    instructor = state["agent_instructor"]
    tool = state["current_tool"]
    input_parameters = state["input_parameters"]
    algorithm_doc = state["algorithm_doc"]
    data_path_train = state["data_path_train"]
    data_path_test = state["data_path_test"]
    package_name = state["package_name"]

    if not state["code_quality"]:
        print(f"\n=== [Instructor] Processing {tool} (first execution) ===")
        code = instructor.generate_code(
            algorithm=tool,
            data_path_train=data_path_train,
            data_path_test=data_path_test,
            algorithm_doc=algorithm_doc,
            input_parameters=input_parameters,
            package_name=package_name
        )
    else:
        print(f"\n=== [Instructor] Re-executing updated code for {tool} ===")
        code = state["code_quality"].code

    new_code_quality = instructor.execute_generated_code(code, tool)
    if state["code_quality"]:
        new_code_quality.review_count = state["code_quality"].review_count
        new_code_quality.algorithm = state["code_quality"].algorithm
    if not new_code_quality.algorithm:
        new_code_quality.algorithm = tool
    return {"code_quality": new_code_quality}

# =================================================================
# Node: Reviewer
# Reviews code and makes revisions if errors exist
# =================================================================
def call_reviewer_for_single_tool(state: FullToolState) -> dict:
    reviewer = state["agent_reviewer"]
    code_quality = state["code_quality"]
    tool = state["current_tool"]
    algorithm_doc = state["algorithm_doc"]

    print(f"\n=== [Reviewer] Reviewing code for {tool} ===")
    if code_quality.error_message:
        code_quality.review_count += 1
        revised_code = reviewer.review_code(
            code_quality=code_quality,
            vectorstore=state["vectorstore"],
            algorithm_doc=algorithm_doc
        )
        code_quality.code = revised_code
    return {"code_quality": code_quality}

# =================================================================
# Node: Decider
# Determines whether to re-run the instructor
# =================================================================
def decide_reviewer_result(state: FullToolState) -> dict:
    code_quality = state["code_quality"]
    if code_quality.error_message and code_quality.review_count < 2:
        return {"should_rerun": True}
    else:
        return {"should_rerun": False}

# Conditional edge selector for rerun logic
def check_if_need_rerun(state: FullToolState):
    return "need_rerun" if state["should_rerun"] else "done"

# =================================================================
# Build StateGraph for single-tool processing
# InfoMiner → instructor → reviewer → decider
# =================================================================

single_tool_graph = StateGraph(FullToolState)

# Add nodes
single_tool_graph.add_node("InfoMiner", call_InfoMiner)
single_tool_graph.add_node("instructor", call_instructor_for_single_tool)
single_tool_graph.add_node("reviewer", call_reviewer_for_single_tool)
single_tool_graph.add_node("decider", decide_reviewer_result)

# Define flow
single_tool_graph.set_entry_point("InfoMiner")
single_tool_graph.add_edge("InfoMiner", "instructor")
single_tool_graph.add_edge("instructor", "reviewer")
single_tool_graph.add_edge("reviewer", "decider")
single_tool_graph.add_conditional_edges("decider", check_if_need_rerun, {
    "need_rerun": "instructor",
    "done": END,
})

# Compile the graph
compiled_single_tool = single_tool_graph.compile()

# =================================================================
# 4. Node: Process all tools (parallel or sequential)
#    Get tool list from agent_selector, build per-tool state,
#    Call single-tool processing graph for each tool, store results in state["results"]
# =================================================================
def process_all_tools(state: FullToolState) -> dict:
    if not state["agent_selector"]:
        raise ValueError("agent_selector is not set!")
    tools = state["agent_selector"].tools
    if not tools:
        print("No tool list generated. Exiting.")
        state["results"] = []
        return state

    async def process_tool_async(tool, state):
        tool_state = state.copy()
        tool_state["current_tool"] = tool
        tool_state["code_quality"] = None
        tool_state["should_rerun"] = False
        final_state = await asyncio.to_thread(
            compiled_single_tool.invoke,
            tool_state,
            config={"recursion_limit": 20}
        )
        return tool, final_state

    results = []
    use_parallel = "-p" in sys.argv
    if use_parallel:
        # Create new event loop and process in parallel
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = [process_tool_async(tool, state) for tool in tools]
        results = loop.run_until_complete(asyncio.gather(*tasks))
        loop.close()
    else:
        # Process sequentially
        for tool in tools:
            results.append(asyncio.run(process_tool_async(tool, state)))
    state["results"] = results
    return state

# =================================================================
# 5. Build full process graph: preprocessor → selector → process_all_tools
#    Combine preprocessing, planning, and tool processing
# =================================================================
full_graph = StateGraph(FullToolState)
full_graph.add_node("preprocessor", call_preprocessor)
full_graph.add_node("selector", call_selector)
full_graph.add_node("process_all_tools", process_all_tools)
full_graph.set_entry_point("preprocessor")
full_graph.add_edge("preprocessor", "selector")
full_graph.add_edge("selector", "process_all_tools")
compiled_full_graph = full_graph.compile()

# =================================================================
# 6. Main function
#    Initialize agents and initial state, call the full process graph,
#    then print the results of each tool
# =================================================================
async def main():
    if os.path.exists("train_data_loader.py"):
        os.remove("train_data_loader.py")
    if os.path.exists("test_data_loader.py"):
        os.remove("test_data_loader.py")
    if os.path.exists("head_train_data_loader.py"):
        os.remove("head_train_data_loader.py")
    if os.path.exists("head_test_data_loader.py"):
        os.remove("head_test_data_loader.py")
    InfoMiner_instance = AgentInfoMiner()
    instructor_instance = AgentInstructor()
    reviewer_instance = AgentReviewer()
    preprocessor_instance = AgentPreprocessor()
    
    initial_state: FullToolState = {
        "messages": [],
        "current_tool": "",
        "input_parameters": {},
        "data_path_train": "",
        "data_path_test": "",
        "agent_InfoMiner": InfoMiner_instance,
        "agent_instructor": instructor_instance,
        "agent_reviewer": reviewer_instance,
        "vectorstore": None,
        "code_quality": None,
        "should_rerun": False,
        "agent_preprocessor": preprocessor_instance,
        "agent_selector": None,
        "experiment_config": None,
        "results": None,
        "algorithm_doc": None,
    }
    
    print("\n=== [Main] Starting full process graph ===")
    final_state = await asyncio.to_thread(
        compiled_full_graph.invoke,
        initial_state,
        config={"recursion_limit": 20}
    )
    print("\n=== [Main] Full process graph completed ===")
    
    # Print results for each tool
    results = final_state.get("results", [])
    for tool, tool_state in results:
        code_quality = tool_state.get("code_quality")
        if code_quality and not code_quality.error_message:
            print(f"[{tool}] Success, AUROC: {code_quality.auroc:.4f}, AUPRC: {code_quality.auprc:.4f}, Error Points: {[]}")
        else:
            err_msg = code_quality.error_message if code_quality else "Unknown"
            print(f"[{tool}] Failed, error message: {err_msg}")

if __name__ == "__main__":
    asyncio.run(main())
