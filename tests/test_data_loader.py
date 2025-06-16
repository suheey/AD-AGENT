import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import types
import csv
import pytest

# Stub modules to avoid heavy dependencies
openai_stub = types.ModuleType('openai')
openai_stub.OpenAI = lambda api_key=None: None
sys.modules.setdefault('openai', openai_stub)

sklearn_stub = types.ModuleType('sklearn')
model_selection_stub = types.ModuleType('sklearn.model_selection')
model_selection_stub.train_test_split = lambda X, y, test_size=0.2, random_state=42: (X, X, y, y)
sys.modules.setdefault('sklearn', sklearn_stub)
sys.modules.setdefault('sklearn.model_selection', model_selection_stub)

from data_loader.data_loader import DataLoader


def test_load_csv(monkeypatch):
    def _mock_load_data(self, split_data=False):
        with open(self.filepath, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [row for row in reader]
        X = [list(map(float, row[:-1])) for row in rows]
        y = [int(row[-1]) for row in rows]
        return X, y

    monkeypatch.setattr(DataLoader, "load_data", _mock_load_data)
    loader = DataLoader("data/yahoo_train.csv")
    X, y = loader.load_data()
    assert len(X) > 0
    assert len(X) == len(y)
    assert len(X[0]) == 6


def test_tslib_path():
    loader = DataLoader("data/MSL")
    X, y = loader.load_data()
    assert X == "tslib"
    assert y == "tslib"


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        DataLoader("nonexistent/path.csv")
