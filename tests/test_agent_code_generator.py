import importlib
import pytest

if importlib.util.find_spec("langchain_openai") is None:
    pytest.skip("langchain_openai not available", allow_module_level=True)

from agents.agent_code_generator import AgentCodeGenerator, llm

class DummyResponse:
    def __init__(self, text):
        self.content = text

def test_generate_code_clean(monkeypatch):
    monkeypatch.setattr(llm, 'invoke', lambda *a, **k: DummyResponse("```python\nprint('hi')\n```"))
    ag = AgentCodeGenerator()
    code = ag.generate_code(
        algorithm='ABOD',
        data_path_train='train',
        data_path_test='test',
        algorithm_doc='doc',
        input_parameters={},
        package_name='pyod'
    )
    assert code.strip() == "print('hi')"

def test_extract_init_params_dict():
    txt = "```python\n{'a': 1}\n```"
    result = AgentCodeGenerator._extract_init_params_dict(txt)
    assert result == {'a': 1}
