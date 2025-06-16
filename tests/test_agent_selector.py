import importlib
import pytest

if importlib.util.find_spec("langchain_community") is None:
    pytest.skip("langchain_community not available", allow_module_level=True)

from agents.agent_selector import AgentSelector

class DummySelector(AgentSelector):
    def __init__(self):
        pass


def test_generate_tools_pyod():
    sel = DummySelector.__new__(DummySelector)
    sel.package_name = 'pyod'
    tools = AgentSelector.generate_tools(sel, ['all'])
    assert 'ECOD' in tools

