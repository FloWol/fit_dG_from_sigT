import numpy as np
import lmfit
from lmfit import Parameters
import logging
from abc import ABC, abstractmethod
import warnings

# --- Physical Constants ---
K_B = 1.380649e-23  # J/K
GYROMAG_RADIUS = 26.7522128 * 1.0e7  # rad per Tesla
VACCUM_PERM = 4.0e-7 * np.pi  # vacuum permeability
REDUCED_PLANCK = 6.62606957e-34 / 2.0 / np.pi  # J s
R_GAS_CONSTANT = 8.314 # J/(mol*K)


# --- Thermodynamic Models ---

def stability_curve_no_Sm(T, Sm, Hm, Cp, Tm):
    """A simplified stability curve model for ΔG vs. T."""
    return -((Hm/R_GAS_CONSTANT)*(1/Tm-1/T) + (Cp/R_GAS_CONSTANT) * (Tm/T - 1 + np.log(T/Tm)))

def stability_curve_SM(T, Sm, Hm, Cp, Tm):
    """A stability curve model for ΔG vs. T that includes Sm."""
    return (Hm - T * Sm + Cp * (T - Tm) - T * Cp * np.log(T / Tm)) / (-R_GAS_CONSTANT * T)

def vant_Hoff_Sm(T, Sm, Hm, Cp, Tm):
    """The van't Hoff equation including Sm."""
    return Hm/(R_GAS_CONSTANT*T) - Sm/R_GAS_CONSTANT

def vant_Hoff(T, Sm, Hm, Cp, Tm):
    """The van't Hoff equation."""
    return Hm / R_GAS_CONSTANT * (1 / Tm - 1 / T)

# --- NMR Relaxation Equations ---

def b_dipole_coupling_constant(r):
    """Calculates the dipole-dipole coupling constant."""
    return -(REDUCED_PLANCK * VACCUM_PERM * GYROMAG_RADIUS ** 2) / (4 * np.pi * r ** 3)

def J_spectral_density(omega, tau):
    """Calculates the spectral density function."""
    return (tau / (1 + (omega * tau) ** 2))

def distance_from_sigma(sig, tau_C, omega0):
    """Calculates the distance between two dipoles from the cross-relaxation rate (sigma)."""
    J = (1 / 10) * (J_spectral_density(0, tau_C) - 6 * J_spectral_density(2 * omega0, tau_C))
    a = (-(REDUCED_PLANCK * VACCUM_PERM * GYROMAG_RADIUS ** 2) / (4 * np.pi)) ** 2
    warnings.warn("correction only works for 'lowest_T' setting")
    intermediate_result = (a * J[0] / sig) # J[0] as we are only interested in the lowest temperature ("lowest T")
    return intermediate_result**(1/6)

def calc_cross_relaxation_rate(r, tau_C, omega0):
    """
     Calculates the cross-relaxation rate (sigma) from the distance between two dipoles.

     Args:
         r (float): The distance in meters.

     Returns:
         float: The cross-relaxation rate.
     """

    b_val = b_dipole_coupling_constant(r)
    J_val = (1 / 10) * (J_spectral_density(0, tau_C) - 6 * J_spectral_density(2 * omega0, tau_C)) #Check if newaxis is needed
    return np.asarray(b_val**2 * J_val)


# --- Correlation Time Calculation ---

def calc_tau_C(T, rH, eta):
    """Calculates tau_C from the Stokes-Einstein equation."""
    return 4.0 * np.pi * eta * rH ** 3 / (3.0 * K_B * T)

def fit_rH(T, eta, tauC_data, fit_method="leastsq"):
    """Fits the hydrodynamic radius rH to the correlation times."""
    def residual(params, T, data):
        rH = params["rH"].value
        model = calc_tau_C(T, rH, eta)
        return data - model

    params = Parameters()
    params.add('rH', value=1)
    mini = lmfit.Minimizer(residual, params, fcn_args=(T, tauC_data))
    out = mini.minimize(method=fit_method)
    logging.info("Results of rH fit:")
    out.params.pretty_print()
    return out.params["rH"].value

class CorrelationTimeCalculator(ABC):

    @abstractmethod
    def calculate(self, correlation_times, temperatures):
        pass

    def mask_times_temperatures(self, correlation_times, temperatures):
        valid_mask = correlation_times.notnull()

        masked_temperatures = temperatures[valid_mask]
        masked_correlation_times = correlation_times[valid_mask]

        return masked_correlation_times, masked_temperatures, valid_mask

    @staticmethod
    def create(config):
        """Factory method to create a correlation time calculator."""
        calculator_name = config.get("method")
        if calculator_name == "StokesEinstein":
            return StokesEinsteinCalculator(config)

        elif calculator_name == "linear":
            return Linear_correlation_interpolator()

        else:
            raise ValueError(f"Unknown correlation time calculator: {calculator_name}")


class StokesEinsteinCalculator(CorrelationTimeCalculator):
    """
    Calculates correlation time using the Stokes-Einstein viscosity equation,
    fitting hydrodynamic radius if necessary.

    viscosity in mPa.s
    """
    def __init__(self, config: dict):
        self.config = config

        self._viscosity_map = {
            "H2O": self._viscosity_H2O,
            "D2O": self._viscosity_D2O,
        }

        solvent = config["solvent"]
        self._viscosity = self._viscosity_map[solvent]


    def _viscosity_H2O(self, T):
        """Calculates the viscosity of water at a given temperature."""
        A = 1.856e-11
        B = 4209.0
        C = 0.04527
        D = -3.376e-5
        mu = A * np.exp(B / T + C * T + D * T ** 2)
        return mu * 1.0e-3

    def _viscosity_D2O(self, T):
        raise NotImplementedError




    def calculate(self, correlation_times, temperatures):
        logging.info("Using Stokes-Einstein viscosity equation")
        viscosity = self._viscosity(temperatures)
        masked_correlation_times, masked_temperatures, mask = self.mask_times_temperatures(correlation_times, temperatures)
        viscosity_masked = viscosity[mask]

        rH = fit_rH(masked_temperatures, viscosity_masked, masked_correlation_times)
        full_correlation_times = calc_tau_C(temperatures, rH, viscosity)
        return full_correlation_times # mamybe output radius of hydration as well (it is also in the logs already)




class Linear_correlation_interpolator(CorrelationTimeCalculator):
    def calculate(self, correlation_times, temperatures):
        masked_correlation_times, masked_temperatures, mask = self.mask_times_temperatures(correlation_times, temperatures)
        lin_interp = np.interp(temperatures, masked_temperatures, masked_correlation_times)
        return lin_interp
