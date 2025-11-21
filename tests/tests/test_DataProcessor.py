import DataProcessor
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_data():
    """
    Creates a mock DataFrame structured like the class expects:
    Cols: Name, sigA, sigB, Temp1, Temp2, Temp3...
    """
    data = {
        "Name": ["Sample1", "Sample2", "CorrelationRow"],
        "sigA": [0.1, 0.2, 0.0],  # Custom sigA col
        "sigB": [0.5, 0.6, 0.0],  # Custom sigB col
        "298.0": [10.0, 20.0, 7.41842325e-09],  # Temp 1 (Lowest T)
        "300.0": [11.0, 21.0, 7.35730930e-09],  # Temp 2
        "310.0": [12.0, 22.0, 7.05173954e-09]  # Temp 3
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def sample_data_incomplete():
    """
    Creates a mock DataFrame structured like the class expects:
    Cols: Name, sigA, sigB, Temp1, Temp2, Temp3...
    """
    data = {
        "Name": ["Sample1", "Sample2", "CorrelationRow"],
        "sigA": [0.1, 0.2, 0.0],  # Custom sigA col
        "sigB": [0.5, 0.6, 0.0],  # Custom sigB col
        "298.0": [10.0, 20.0, 7.41842325e-09],  # Temp 1 (Lowest T)
        "300.0": [11.0, 21.0, np.nan],  # Temp 2
        "310.0": [12.0, 22.0, 7.05173954e-09]  # Temp 3
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def base_config():
    """Basic configuration dictionary."""
    return {
        "sigA": "lowest_T",
        "sigB": "0",
        "method": "StokesEinstein",
        "solvent": "H2O"
    }

@pytest.fixture
def base_config_linear():
    """Basic configuration dictionary."""
    return {
        "sigA": "lowest_T",
        "sigB": "0",
        "method": "linear"
    }


def test_initialization(sample_data, base_config):
    """Test if the __init__ parses the DataFrame structure correctly."""
    dp = DataProcessor.DataProcessor(base_config, sample_data)

    # Test Temperatures extraction
    expected_temps = np.array([298.0, 300.0, 310.0])
    np.testing.assert_array_equal(dp.temperatures, expected_temps)

    # Test Names extraction
    assert len(dp.names) == 3
    assert dp.names.iloc[0] == "Sample1"

    # Test Experimental Values (Should exclude last row and first 3 cols)
    # Shape should be (2 rows, 3 temp cols)
    assert dp.experimental_values.shape == (2, 3)
    assert dp.experimental_values.iloc[0, 0] == 10.0  # Sample1, 298K

    # Test Correlation Time (Should be last row only, temp cols)
    assert len(dp.correlation_time) == 3
    assert dp.correlation_time.iloc[0] == 7.41842325e-09


def test_strategy_lowest_T(sample_data):
    """Test 'lowest_T' strategy for both A and B."""
    config = {"sigA": "lowest_T", "sigB": "highest_T"}
    dp = DataProcessor.DataProcessor(config, sample_data)
    dp.sigma_strategy()

    # Lowest T is the first temperature column (298.0)
    # Values should be [10.0, 20.0] (excluding correlation row)
    expected_values_A = [10.0, 20.0]
    expected_values_B = [12.0, 22.0]

    np.testing.assert_array_equal(dp.sigA, expected_values_A)
    np.testing.assert_array_equal(dp.sigB, expected_values_B)


def test_strategy_custom(sample_data):
    """Test 'custom' strategy reading from specific columns."""
    config = {"sigA": "custom", "sigB": "custom"}
    dp = DataProcessor.DataProcessor(config, sample_data)
    dp.sigma_strategy()

    # Should read from raw_df['sigA'] and raw_df['sigB']
    # Note: logic assumes we only want the rows corresponding to experimental_values
    expected_sigA = sample_data["sigA"]
    expected_sigB = sample_data["sigB"]

    np.testing.assert_array_equal(dp.sigA, expected_sigA)
    np.testing.assert_array_equal(dp.sigB, expected_sigB)


def test_strategy_zero_sigB(sample_data):
    """Test specific case where sigB is '0'."""
    config = {"sigA": "lowest_T", "sigB": "0"}
    dp = DataProcessor.DataProcessor(config, sample_data)
    dp.sigma_strategy()
    expected_sigA = [10.0, 20.0]

    assert dp.sigB == 0
    # SigA should still be set
    np.testing.assert_array_equal(dp.sigA, expected_sigA)


def test_not_implemented_errors(sample_data):
    """Ensure strategies marked as 'fit' or 'custom_temp' raise errors."""

    # Test fit
    dp_fit = DataProcessor.DataProcessor({"sigA": "fit", "sigB": "0"}, sample_data)
    with pytest.raises(NotImplementedError):
        dp_fit.sigma_strategy()

    # Test custom_temp
    dp_custom = DataProcessor.DataProcessor({"sigA": "lowest_T", "sigB": "custom_temp"}, sample_data)
    with pytest.raises(NotImplementedError):
        dp_custom.sigma_strategy()

def test_calc_correlation_time_full_data(base_config, sample_data):
    dp = DataProcessor.DataProcessor(base_config, sample_data)
    dp.calc_correlation_time()

    assert np.all(dp.correlation_time.values == sample_data.iloc[-1,3:].values)
    assert np.all(dp.correlation_time.values == [7.41842325e-09, 7.35730930e-09, 7.05173954e-09])
def test_calc_correlation_time_incomplete_data(base_config, sample_data_incomplete):
    dp = DataProcessor.DataProcessor(base_config, sample_data_incomplete)
    expected = [8.136601e-09, 7.734459e-09, 6.092625e-09]
    dp.calc_correlation_time()
    assert expected == pytest.approx(dp.correlation_time, abs=1e-11)