import data_loader as dl
import pandas as pd
import pytest


def test_load_data():
    data_loader = dl.DataLoader("sample_data.txt")
    data = data_loader.load_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

@pytest.fixture
def loader():
    loader = dl.DataLoader("sample_data.txt")
    loader.load_data()
    return loader