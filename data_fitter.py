import os
import logging
import numpy as np
import scipy as sp
import pandas as pd
import lmfit
from lmfit import minimize, Parameters
import matplotlib.pyplot as plt
from pygments.unistring import Sm


class data_fitter():
    def __init__(self, data_file, config=None, output_dir=None):
        self.config = config
        self.initial_Tm = self.config['Tm']
        self.data_file = data_file
        self.output_dir = output_dir
        self.df = pd.read_csv(data_file, sep="\t")
        self.T = self.df.columns[3:].astype(float)

        n_colors = self.df.shape[0]
        base_colors = list(plt.cm.tab20.colors)
        self.colors = [base_colors[i % len(base_colors)] for i in range(n_colors)]

        self.k_B = 1.380649e-23
        self.freq = config["frequency"]
        self.omega0 = 2.0 * np.pi * self.freq

        self.sigT = self.df.iloc[:-1, 3:]

    def viscocity(self, T):
        A = 1.856e-11
        B = 4209.0
        C = 0.04527
        D = -3.376e-5
        mu = A * np.exp(B / T + C * T + D * T ** 2)
        self.eta = mu * 1.0e-3

    def calc_tau_C(self, T, rH):
        return 4.0 * np.pi * self.eta * rH ** 3 / (3.0 * self.k_B * T)

    def fit_rH(self, T, tauC_data):
        def residual(params, T, data):
            rH = params["rH"].value
            model = self.calc_tau_C(T, rH)
            diff = data - model
            return diff

        params = Parameters()
        params.add('rH', value=2E-9)

        mini = lmfit.Minimizer(residual, params, fcn_args=(T, tauC_data))
        out1 = mini.minimize(method=self.config["fit_method"])

        logging.info("Results of rH fit:")
        withResults = out1.params
        withResults.pretty_print()

        return out1.params["rH"].value

    def correct_tau_C(self):
        if "tau_C" in self.df["Name"].values:
            tau_c_row = self.df[self.df['Name'] == 'tau_C'].iloc[0].dropna()
            tau_c_temperatures = tau_c_row.drop(labels='Name').index.astype(float).to_numpy()
            tau_c_vals = tau_c_row.drop(labels='Name').values
            self.viscocity(tau_c_temperatures)

            if len(tau_c_vals) == len(self.df.columns[3:]):
                self.tau_C = tau_c_vals
            else:
                rH = self.fit_rH(tau_c_temperatures, tau_c_vals)
                temperatures = self.df.columns[3:].to_numpy().astype(float)
                self.viscocity(temperatures)
                self.tau_C = self.calc_tau_C(temperatures, rH)
        else:
            raise ValueError("No tau_C found")

    def b(self, r):
        gyromag_radius = 26.7522128 * 1.0e7
        vaccum_perm = 4.0e-7 * np.pi
        reduced_planck = 6.62606957e-34 / 2.0 / np.pi
        return -(reduced_planck * vaccum_perm * gyromag_radius ** 2) / (4 * np.pi * r ** 3)

    def J(self, omega, tau):
        return (tau / (1 + (omega * tau) ** 2))

    def r_from_sigma(self, sig):
        gyromag_radius = 26.7522128 * 1.0e7
        vaccum_perm = 4.0e-7 * np.pi
        reduced_planck = 6.62606957e-34 / 2.0 / np.pi
        J = (1 / 10) * (self.J(0, self.tau_C) - 6 * self.J(2 * self.omega0, self.tau_C))
        a = (-(reduced_planck * vaccum_perm * gyromag_radius ** 2) / (4 * np.pi)) ** 2
        intermediate_result = (a * J[0] / sig)
        return intermediate_result ** (1 / 6)

    def R_cross(self, r):
        return np.asarray((1 / 10) * self.b(r) ** 2 * (self.J(0, self.tau_C) - 6 * self.J(2 * self.omega0, self.tau_C))[:, np.newaxis])

    def sigma_strategy(self):
        sigA_type = self.config.get("sigA")
        if sigA_type not in {"lowest_T", "fit", "true"}:
            raise ValueError(f"Invalid or missing 'sigA': {sigA_type}")
        sigB_type = self.config.get("sigB")
        if sigB_type not in {"0", "fit", "highest_T", "true"}:
            raise ValueError(f"Invalid or missing 'sigB': {sigB_type}")

        if self.config["sigA"] == "lowest_T":
            sigA = self.sigT.iloc[:, 0].values
            if self.config["correct_for_correlation_time"]:
                self.sig_A_corrected = self.R_cross(self.r_from_sigma(sigA))
            else:
                logging.info("no correction for sigma A is performed")
                self.sig_A_corrected = sigA

        if self.config["sigA"] == "fit":
            raise NotImplementedError
        if self.config["sigA"] == "custom":
            self.sigA = self.df["sigA"]
        if self.config["sigA"] == "custom_temp":
            self.sigA = self.df[self.config["custom_temp"]]

        if self.config["sigB"] == "0":
            self.sigB = 0
        if self.config["sigB"] == "highest_T":
            self.sigB = 0
            raise NotImplementedError
        if self.config["sigB"] == "custom":
            self.sigB = self.df["sigB"]
        if self.config["sigB"] == "custom_temp":
            self.sigB = self.df[self.config["custom_temp"]]

    @staticmethod
    def stability_curve_simple(T, Sm, Hm, Cp, Tm):
        R = 8.314
        return -((Hm / R) * (1 / Tm - 1 / T) + (Cp / R) * (Tm / T - 1 + np.log(T / Tm)))

    @staticmethod
    def stability_curve_SM(T, Sm, Hm, Cp, Tm):
        R = 8.314
        return (Hm + T * Sm + Cp * (T - Tm) - T * Cp * np.log(T / Tm)) / (-R * T)

    @staticmethod
    def vant_Hoff_Sm(T, Sm, Hm, Cp, Tm):
        R = 8.314
        return Hm / (R * T) + Sm / R

    @staticmethod
    def vant_Hoff(T, Sm, Hm, Cp, Tm):
        R = 8.314
        return Hm / R * (1 / Tm - 1 / T)

    def which_stability_curve(self):
        if self.config["Cp_model"]:
            if self.config["fit_SM"]:
                return self.stability_curve_SM
            else:
                logging.info("Ignore Sm values as they are not fit")
                return self.stability_curve_simple
        else:
            if self.config["fit_SM"]:
                return self.vant_Hoff_Sm
            else:
                logging.info("Ignore Sm values as they are not fit")
                return self.vant_Hoff

    def calc_K(self):
        if self.config["ignore_lowest_T"]:
            self.K = ((self.sigT - self.sigB) / (self.sig_A_corrected - self.sigT.T).T).iloc[:, 1:]
        else:
            self.K = ((self.sigT - self.sigB) / (self.sig_A_corrected.T - self.sigT))

    @staticmethod
    def delta_G(T, K):
        return -8.314 * T.to_numpy() * np.log(K.values.astype(float))

    def fit_stability_curve(self):
        for pair in range(self.K.shape[0]):
            try:
                pair_name = self.df["Name"][self.K.index[pair]]

                experimentalData = np.log(self.K.iloc[pair].values.astype(float))

                valid_indices = ~np.isnan(experimentalData) & ~np.isinf(experimentalData)
                T_fit = self.T[valid_indices]
                experimentalData_fit = experimentalData[valid_indices]

                if len(experimentalData_fit) == 0:
                    logging.info(f"No data for {pair_name}, skipping.")
                    continue

                params = Parameters()
                params.add('Sm', value=0, vary=self.config["fit_SM"], name='Sm (J/mol/K)')
                params.add('Hm', value=1, vary=True, name='Hm (J/mol)')
                params.add('Cp', value=1, vary=self.config["Cp_model"], name='Cp (J/mol/K)')

                if self.config["Tm"] is not None:
                    if self.config["fix_Tm"]:
                        params.add('Tm', value=self.initial_Tm, vary=False, name='Tm (K)')
                        logging.info("Tm is fixed, ignore any values outputted for Tm")
                    else:
                        params.add('Tm', value=self.initial_Tm, vary=True, name='Tm (K)')
                else:
                    params.add('Tm', value=1, vary=True, name='Tm (K)')

                num_params = len([p for p in params.values() if p.vary])
                stability_curve = self.which_stability_curve()

                if len(experimentalData_fit) < num_params:
                    logging.warning(f"Not enough data points ({len(experimentalData_fit)}) to fit {num_params} parameters for {pair_name}. Switching to a simpler van't Hoff model.")
                    params['Cp (J/mol/K)'].vary = False
                    params['Cp (J/mol/K)'].value = 0
                    if self.config["fit_SM"]:
                        stability_curve = self.vant_Hoff_Sm
                    else:
                        stability_curve = self.vant_Hoff

                def residual(params, T, data):
                    Sm = params["Sm (J/mol/K)"].value
                    Hm = params["Hm (J/mol)"].value
                    Cp = params["Cp (J/mol/K)"].value
                    Tm = params["Tm (K)"].value
                    model = stability_curve(T, Sm, Hm, Cp, Tm)
                    return data - model

                mini = lmfit.Minimizer(residual, params, fcn_args=(T_fit, experimentalData_fit))
                out1 = mini.minimize(method=self.config["fit_method"], max_nfev=self.config["max_iterations"])

                logging.info(f"Results of stability fit for {pair_name}:")
                withResults = out1.params
                withResults.pretty_print()

                color = self.colors[pair]
                plt.plot(self.T, self.K.iloc[pair], ".", alpha=1, label="G" + pair_name, color=color)
                plt.plot(self.T, np.exp(stability_curve(self.T, withResults["Sm (J/mol/K)"].value, withResults["Hm (J/mol)"].value,
                                                      withResults["Cp (J/mol/K)"].value, withResults["Tm (K)"].value)), label="stability" + pair_name, color=color)

            except Exception as e:
                logging.error(f"Error fitting stability curve for {pair_name} with error {e}")

        plt.xlabel("Temperature (K)")
        plt.ylabel("Equilibrium constant K(T)")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "stability_curve.pdf"), format="pdf")
        plt.show()

    def run(self):
        logging.info("Choosing/interpolating correlation time ...")
        self.correct_tau_C()
        logging.info("Picking the correct sigma ...")
        self.sigma_strategy()
        logging.info("Calculating K analytically ...")
        self.calc_K()

        for pair in range(self.K.shape[0]):
            name = self.df["Name"][self.K.index[pair]]
            plt.plot(self.T, self.sigT.iloc[pair, :], label=name)
        plt.xlabel("Temperature (K)")
        plt.ylabel("sig(T)")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "sig_T.pdf"), format="pdf")
        plt.show()

        if self.config["ignore_lowest_T"]:
            self.T = self.T[1:]
        for pair in range(self.K.shape[0]):
            name = self.df["Name"][self.K.index[pair]]
            plt.plot(self.T, self.K.iloc[pair, :], alpha=0.5, label=name, marker=".")
        plt.xlabel("Temperature (K)")
        plt.ylabel("K")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "K_T.pdf"), format="pdf")
        plt.show()

        for pair in range(self.K.shape[0]):
            name = self.df["Name"][self.K.index[pair]]
            plt.plot(self.T, self.delta_G(self.T, self.K.iloc[pair]), alpha=0.5, label=name, marker=".")
        plt.xlabel("Temperature (K)")
        plt.ylabel("ΔG [J/mol]")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "delta_G_T.pdf"), format="pdf")
        plt.show()

        logging.info("Fitting stability curve ...")
        self.fit_stability_curve()
