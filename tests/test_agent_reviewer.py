import importlib
import pytest

if importlib.util.find_spec("langchain_openai") is None:
    pytest.skip("langchain_openai not available", allow_module_level=True)

from agents.agent_reviewer import AgentReviewer


def test_clean_markdown():
    text = "```python\nprint('x')\n```"
    assert AgentReviewer._clean_markdown(text) == "print('x')"
