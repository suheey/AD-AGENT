import importlib
import pytest

if importlib.util.find_spec("openai") is None:
    pytest.skip("openai not available", allow_module_level=True)

from agents.agent_processor import AgentProcessor


def test_extract_config(monkeypatch):
    processor = AgentProcessor()
    sample = 'FINAL: {"algorithm":["LOF"],"dataset_train":"train","dataset_test":"test","parameters":{}}'
    monkeypatch.setattr(processor, 'get_chatgpt_response', lambda msgs: sample)
    result = processor.extract_config('dummy')
    assert result == {"algorithm": ["LOF"], "dataset_train": "train", "dataset_test": "test", "parameters": {}}
