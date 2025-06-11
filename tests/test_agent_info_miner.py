import importlib
import json
import pytest

if importlib.util.find_spec("openai") is None:
    pytest.skip("openai not available", allow_module_level=True)

from agents.agent_info_miner import AgentInfoMiner

class DummyClient:
    class Response:
        def __init__(self, text):
            self.output_text = text

    def __init__(self, *a, **k):
        pass

    class responses:
        @staticmethod
        def create(*args, **kwargs):
            return DummyClient.Response("doc text")


def test_query_docs(monkeypatch, tmp_path):
    monkeypatch.setattr('agents.agent_info_miner.OpenAI', lambda *a, **k: DummyClient())
    miner = AgentInfoMiner()
    doc = miner.query_docs('Algo', vectorstore=None, package_name='pyod', cache_path=str(tmp_path/"cache.json"))
    assert doc == 'doc text'
    # cache file should be written
    with open(tmp_path/"cache.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert 'Algo' in data
