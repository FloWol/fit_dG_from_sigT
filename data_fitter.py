import numpy as np
import scipy as sp
import pandas as pd
import lmfit
from lmfit import minimize, Parameters
import matplotlib.pyplot as plt



class data_fitter():
    def __init__(self, data_file, config=None):
        self.config = config
        self.data_file = data_file

        self.df = pd.read_csv(data_file, sep="\t")
        self.df_filtered = self.df[~self.df["Name"].str.contains("Q", na=False)].dropna() #drop pseudo atoms and missing vals
        self.T = self.df.columns[3:].astype(float)
        self.k_B = 1.380649e-23 #J/K
        self.freq = config["frequency"]
        self.omega0 = 2.0 * np.pi * self.freq

        self.sigT = self.df_filtered.iloc[:,3:]

        self.viscocity()





    def viscocity(self):
        """
        #TODO check parameters
        for water
        :return:
        """
        A = 1.856e-11
        B = 4209.0
        C = 0.04527
        D = -3.376e-5
        mu = A * np.exp(B / self.T + C * self.T + D * self.T ** 2)
        self.eta = mu * 1.0e-3

    def calc_tau_C(self, T, rH):
        """
        from stokes law
        :param rH:
        :return:
        """
        return 4.0 * np.pi * self.eta * rH ** 3 / (3.0 * self.k_B * T)

    def fit_rH(self, T, tauC_data):
        """
        This function takes an incomplete list auf tau_C from the data file and fits the radius of hydration r_H
        to the correlation times. This can later be used to calculate the missing tau_Cs
        :return:
        """
        def residual(params, T, data):
            rH = params["rH"].value
            model = self.calc_tau_C(T, rH)

            diff = data - model
            err = np.linalg.norm(diff)
            return err

        params = Parameters()
        params.add('rH', value=1)


        mini = lmfit.Minimizer(residual, params, fcn_args=(T, tauC_data))
        out1 = mini.minimize(method=self.config["fit_method"])


        print("Results of rH fit:")
        withResults = out1.params
        withResults.pretty_print()

        return out1.params["rH"].value


    def correct_tau_C(self):
        # get correlation times and interpolate them if needed
        if "tau_C" in self.df["Name"].values:

            tau_c_row = self.df[self.df['Name'] == 'tau_C'].iloc[0].dropna()
            tau_c_temperatures = tau_c_row.drop(labels='Name').index.tolist().astype(float)
            tau_c_vals = tau_c_row.drop(labels='Name').values.tolist()

            if len(tau_c_vals) == len(self.df.columns[3:]): #tauC for every temperature
                self.tau_C = tau_c_vals
            else:

                rH = self.fit_rH(tau_c_temperatures, tau_c_vals)
                temperatures = self.df.columns[3:].to_numpy().astype(float)
                self.tau_C = self.calc_tau_C(temperatures,rH) #TODO only use the ones that are NaN from the experiment
        else:
            raise ValueError("No tau_C found")

    def b(self, r):
        # dipole dipole coupling constant
        gyromag_radius = 26.7522128 * 1.0e7  # rad per Tesla
        vaccum_perm = 4.0e-7 * np.pi  # vacuum permeability
        reduced_planck = 6.62606957e-34 / 2.0 / np.pi  # J s
        return -(reduced_planck * vaccum_perm * gyromag_radius ** 2) / (4 * np.pi * r ** 3)  # r in meter

    def J(self, omega, tau):
        return (tau / (1 + (omega * tau) ** 2))

    def r_from_sigma(self, sig):
        gyromag_radius = 26.7522128 * 1.0e7  # rad per Tesla
        vaccum_perm = 4.0e-7 * np.pi  # vacuum permeability
        reduced_planck = 6.62606957e-34 / 2.0 / np.pi  # J s
        J = (1 / 10) * (self.J(0, self.tau_C) - 6 * self.J(2 * self.omega0, self.tau_C))  # TODO tau_C needs to be at the correct temp
        a = (-(reduced_planck * vaccum_perm * gyromag_radius ** 2) / (4 * np.pi)) ** 2
        return np.power(a * J / sig, 1 / 6)



    def R_cross(self, r):
        return (1 / 10) * self.b(r) ** 2 * (self.J(0, self.tau_C) - 6 * self.J(2 * self.omega0, self.tau_C))

    def sigma_strategy(self):
        #user saftey checks
        sigA_type = self.config.get("sigA")
        if sigA_type not in {"lowest_T", "fit", "true"}:
            raise ValueError(f"Invalid or missing 'sigA': {sigA_type}")
        sigB_type = self.config.get("sigB")
        if sigB_type not in {"0", "fit", "highest_T", "true"}:
            raise ValueError(f"Invalid or missing 'sigB': {sigB_type}")

        if self.config["sigA"] == "lowest_T":
            sigA = self.df.columns[3]
            self.sig_A_corrected = self.R_cross(self.r_from_sigma(sigA))


        if self.config["sigA"] == "fit":
            self.sigA = 0 # wrong
        if self.config["sigA"] == "custom":
            self.sigA = self.df["sigA"]
        if self.config["sigA"] == "custom_temp":
            raise NotImplementedError
            self.sigA = self.df[self.config["custom_temp"]] #TODO check with temperature and file


        if self.config["sigB"] == "0":
            self.sigB = 0
        if self.config["sigB"] == "highest_T":
            self.sigB = 0
            raise NotImplementedError
        if self.config["sigB"] == "custom":
            self.sigB = self.df["sigB"]
        if self.config["sigB"] == "custom_temp":
            raise NotImplementedError
            self.sigB = self.df[self.config["custom_temp"]] #TODO check with temperature and file
        # elif self.config["sigB"] == "fit":








    def stability_curve(self, Sm, Hm, Cp, Tm):
        """
        Stability curve model for ΔG vs. T
        """
        k_B = 8.314
        return (Hm + self.T * Sm + Cp * (self.T - Tm) - self.T * Cp * np.log(self.T / Tm))/(-k_B * self.T)

    def calc_K(self):
        self.K = ((self.sigT-self.sigB)/self.sig_A_corrected-self.sigT)


    def delta_G(self, K):
        """
        Gibbs free energy ΔG = -k_B * T * ln(K)
        T: Temperature (can be array)
        K: Equilibrium constant (same shape as T or broadcastable)
        """

        return -8.314 * self.T * np.log(K)

    def fit_stability_curve(self):
        def residual(params, data):
            Sm = params["Sm"].value
            Hm = params["Hm"].value
            Cp = params["Cp"].value
            Tm = params["Tm"].value

            model = self.stability_curve(Sm, Hm, Cp, Tm)

            diff = data - model
            err = np.linalg.norm(diff)
            return err

        params = Parameters()
        params.add('Sm', value=1)
        params.add('Hm', value=1)
        params.add('Cp', value=1)
        params.add('Tm', value=1)

        experimentalData = self.delta_G(self.K)

        mini = lmfit.Minimizer(residual, params, fcn_args=(self.T, experimentalData))
        out1 = mini.minimize(method=self.config["fit_method"])


        print("Results of stability fit:")
        withResults = out1.params
        withResults.pretty_print()

        plt.plot(self.T, self.stability_curve(withResults.params["Sm"], withResults.params["Hm"],
                                              withResults.params["Cp"], withResults.params["Tm"]), label="stability")
        plt.plot(self.T, self.ln_K(), ".", alpha=0.5, label="Name")
        plt.xlabel("Temperature (K)")
        plt.ylabel("ln(K)")
        plt.legend()
        plt.show()



    def run(self):



        # correct sigA for correlation times at different temperatures
        self.correct_tau_C()
        self.sigma_strategy()
        self.calc_K()
        self.fit_stability_curve()
        # calculate ln(K) from the sigmas


        # fit the stability profiles








