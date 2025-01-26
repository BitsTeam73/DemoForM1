import pytest
from src.model import get_model

def test_model_initialization():
    model = get_model()
    assert model.n_estimators == 100
    assert model.max_depth is None
