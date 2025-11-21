import pytest
import numpy as np
import pandas as pd
import yaml
import os
from datafitter import DataFitter


@pytest.fixture
def default_config():
    """Returns a default config dictionary."""
    return {
        "frequency": 700,
        "fit_method": "leastsq",
        "data_path": "sample_data.txt",
        "sigA": "lowest_T",
        "sigB": "0",
        "correct_for_correlation_time": True,
        "max_iterations": 60,
        "Cp_model": True,
        "fit_SM": False,
        "Tm": 300,
        "fix_Tm": False,
        "ignore_lowest_T": True,
    }


@pytest.fixture
def fitter_instance(default_config):
    """Returns a data_fitter instance with a default config."""
    # Create a dummy output directory for the fitter instance
    output_dir = "tests/output"
    os.makedirs(output_dir, exist_ok=True)
    return DataFitter(default_config["data_path"], config=default_config, output_dir=output_dir)


def test_viscocity(fitter_instance):
    """Tests the viscocity calculation."""
    fitter_instance.viscocity(298)
    assert np.isclose(fitter_instance.eta, 0.0009108160211562261)


def test_calc_tau_C(fitter_instance):
    """Tests the tau_C calculation."""
    fitter_instance.viscocity(298)
    tau_c = fitter_instance.calc_tau_C(298, 2e-9)
    assert np.isclose(tau_c, 7.418390843788059e-09)


def test_b_constant(fitter_instance):
    """Tests the calculation of the dipole-dipole coupling constant."""
    b = fitter_instance.b(2e-10)
    assert np.isclose(b, -94342.10382967895)


def test_J_spectral_density(fitter_instance):
    """Tests the spectral density function."""
    j = fitter_instance.J(fitter_instance.omega0, 1e-8)
    assert np.isclose(j, 2.27e-11)


def test_which_stability_curve(fitter_instance):
    """Tests the selection of the stability curve model."""
    # Test case 1: Cp_model = True, fit_SM = True
    fitter_instance.config["Cp_model"] = True
    fitter_instance.config["fit_SM"] = True
    assert fitter_instance.which_stability_curve().__name__ == "stability_curve_SM"

    # Test case 2: Cp_model = True, fit_SM = False
    fitter_instance.config["Cp_model"] = True
    fitter_instance.config["fit_SM"] = False
    assert fitter_instance.which_stability_curve().__name__ == "stability_curve_no_Sm"

    # Test case 3: Cp_model = False, fit_SM = True
    fitter_instance.config["Cp_model"] = False
    fitter_instance.config["fit_SM"] = True
    assert fitter_instance.which_stability_curve().__name__ == "vant_Hoff_Sm"

    # Test case 4: Cp_model = False, fit_SM = False
    fitter_instance.config["Cp_model"] = False
    fitter_instance.config["fit_SM"] = False
    assert fitter_instance.which_stability_curve().__name__ == "vant_Hoff"


