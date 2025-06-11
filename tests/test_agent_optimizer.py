import importlib
import pytest

if importlib.util.find_spec("langchain") is None:
    pytest.skip("langchain not available", allow_module_level=True)

from agents.agent_optimizer import AgentOptimizer


def test_extract_param_dict():
    text = "Thought: ok\nAction: execute_code({'p': 1})"
    assert AgentOptimizer._extract_param_dict(text) == {'p': 1}


def test_find_float():
    assert AgentOptimizer._find_float(r'AUROC:\s*([0-9.]+)', 'AUROC: 0.77') == 0.77


def test_parse_errors():
    text = 'Failed prediction at point [1,2] with true label 0'
    pts = AgentOptimizer._parse_errors(text)
    assert pts == [{'point': [1.0, 2.0], 'true_label': 0.0}]
