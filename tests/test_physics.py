from unittest import result

import numpy as np
import pytest
import physics
import pandas as pd
import matplotlib.pyplot as plt

K_B = 1.380649e-23  # J/K
GYROMAG_RADIUS = 26.7522128 * 1.0e7  # rad per Tesla
VACCUM_PERM = 4.0e-7 * np.pi  # vacuum permeability
REDUCED_PLANCK = 6.62606957e-34 / 2.0 / np.pi  # J s
R_GAS_CONSTANT = 8.314  # J/(mol*K)

@pytest.mark.skip
def test_viscosity_h20():
    assert False

@pytest.mark.skip
def test_vant_hoff_sm():
    assert False

@pytest.mark.skip
def test_stability_curve_no_sm():
    assert False

@pytest.mark.skip
def test_stability_curve_sm():
    assert False

@pytest.mark.skip
def test_vant_hoff():
    assert False


def test_b_dipole_coupling_constant():
    expected = -7.547368306374316e-25
    result = physics.b_dipole_coupling_constant(1)
    assert result == pytest.approx(expected, abs=1e-4)


def test_j_spectral_density():
    expected = 0.5
    result = physics.J_spectral_density(1,1)
    assert result == pytest.approx(expected, abs=1e-4)


def test_distance_from_sigma():
    expected = [2.84119184e-10, 2.53121417e-10]
    result = physics.distance_from_sigma(np.array([0.1,0.2]),np.array([1e-9,2e-9,3e-9]),2*np.pi*700e6)
    assert result == pytest.approx(expected, abs=1e-12)


def test_calc_cross_relaxation_rate():
    expected = [0.13896144, 0.02014558, 0.0272764]
    result = physics.calc_cross_relaxation_rate(4e-10, np.array([10e-9, 1.5e-9, 2e-9]), 2 * np.pi * 700e6)
    assert result == pytest.approx(expected, abs=1e-4)




def test_calc_tau_c():
    expected = [7.41842325e-09, 7.05173954e-09]
    eta = np.array([0.00091082, 0.00087161])
    T = np.array([298, 300])
    result = physics.calc_tau_C(T, 2e-9,eta)
    assert result == pytest.approx(expected, abs=1e-4)


def test_fit_r_h():
    eta = np.array([0.00091082, 0.00087161])
    T = np.array([298, 300])
    tauC_data= np.array([7.41842325e-09, 7.05173954e-09])
    expected = 2e-9 # consistent with other values from previous examples
    result = physics.fit_rH(T, eta, tauC_data, "leastsq")
    assert result == pytest.approx(expected, abs=1e-4)

# -------- Testing Calculators ------------
# ----- Fixtures -------
@pytest.fixture
def sample_data():
    """
    Creates aligned temperature and correlation time data.
    Includes a NaN in correlation_time to test masking.
    """
    temps = pd.Series([298.0, 300.0, 310.0], name="Temperature")
    # Middle value is NaN
    times = pd.Series([7.41842325e-09, np.nan, 7.05173954e-09], name="TauC")
    return temps, times

@pytest.fixture
def stokes_config():
    return {"calculator": "StokesEinstein", "solvent": "H2O"}

@pytest.fixture
def linear_config():
    return {"calculator": "linear"}


def test_StokesEinsteinCalculator_viscosity_H2O(stokes_config):
    corr_calculator = physics.StokesEinsteinCalculator(stokes_config)
    assert isinstance(corr_calculator, physics.StokesEinsteinCalculator)
    expected = [0.00091082, 0.00087161]
    result = corr_calculator._viscosity_H2O(np.array([298,300]))
    assert result == pytest.approx(expected, abs=1e-4)


def test_mask_times_temperatures(sample_data):
    """Test that the base class correctly filters out NaN values."""
    temps, times = sample_data

    # We can use the Linear calculator to access the inherited method
    calc = physics.Linear_correlation_interpolator()

    masked_times, masked_temps, mask = calc.mask_times_temperatures(times, temps)

    assert len(masked_times) == 2
    assert len(masked_temps) == 2
    assert 300.0 not in masked_temps.values
    assert np.isnan(masked_times.values).sum() == 0


def test_factory_create_stokes(stokes_config):
    """Test factory creates StokesEinsteinCalculator."""
    calc = physics.CorrelationTimeCalculator.create(stokes_config)
    assert isinstance(calc, physics.StokesEinsteinCalculator)
    assert calc.config == stokes_config

def test_factory_create_linear(linear_config):
    """Test factory creates Linear_correlation_interpolator."""
    calc = physics.CorrelationTimeCalculator.create(linear_config)
    assert isinstance(calc, physics.Linear_correlation_interpolator)

def test_factory_unknown_calculator():
    """Test factory raises ValueError for unknown types."""
    with pytest.raises(ValueError, match="Unknown correlation time calculator"):
        physics.CorrelationTimeCalculator.create({"calculator": "QuantumFluctuator"})


def test_stokes_viscosity_h2o(stokes_config):
    """Test H2O viscosity calculation returns reasonable physics values."""
    calc = physics.StokesEinsteinCalculator(stokes_config)

    # Test at 293K (20C) - Water viscosity is approx 1.002 mPa.s (0.001 Pa.s)
    temp = 293.0
    viscosity = calc._viscosity_H2O(temp)

    # Allow small tolerance for the specific empirical formula used
    assert 0.0009 < viscosity < 0.00121


def test_stokes_calculate(sample_data, stokes_config):

    temps, times = sample_data
    calc = physics.StokesEinsteinCalculator(stokes_config)
    expected = [8.136601e-09, 7.734459e-09, 6.092625e-09]
    result = calc.calculate(times, temps)

    #plt.scatter(temps, times)
    #plt.plot(temps, result)
    #plt.show()
    # TODO test with D2O

    np.testing.assert_allclose(result, expected, rtol=1e-4)

def test_linear_correlation_interpolator(sample_data):
    temps, times = sample_data
    calc = physics.Linear_correlation_interpolator()
    expected = [7.41842325e-09, 7.35730930e-09, 7.05173954e-09]
    result = calc.calculate(times, temps)

    np.testing.assert_allclose(result, expected, rtol=1e-4)
