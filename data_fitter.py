import os

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
        self.df_filtered = self.df.dropna() #TODO handle pseudo atoms
        self.T = self.df.columns[3:].astype(float)

        n_colors = self.df.shape[0]
        base_colors = list(plt.cm.tab20.colors)  # 20 distinct colors
        self.colors = [base_colors[i % len(base_colors)] for i in range(n_colors)]


        self.k_B = 1.380649e-23 #J/K
        self.freq = config["frequency"]
        self.omega0 = 2.0 * np.pi * self.freq

        self.sigT = self.df.iloc[:-1,3:].dropna() #TODO handle data with NaN







    def viscocity(self, T):
        """
        #TODO check parameters
        for water
        :return:
        """
        A = 1.856e-11
        B = 4209.0
        C = 0.04527
        D = -3.376e-5
        mu = A * np.exp(B / T + C * T + D * T ** 2)
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
            return diff

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
            tau_c_temperatures = tau_c_row.drop(labels='Name').index.astype(float).to_numpy()
            tau_c_vals = tau_c_row.drop(labels='Name').values
            self.viscocity(tau_c_temperatures)

            if len(tau_c_vals) == len(self.df.columns[3:]): #tauC for every temperature
                self.tau_C = tau_c_vals
            else:

                rH = self.fit_rH(tau_c_temperatures, tau_c_vals)
                temperatures = self.df.columns[3:].to_numpy().astype(float)
                self.viscocity(temperatures)
                self.tau_C = self.calc_tau_C(temperatures,rH) #TODO only use the ones that are NaN from the experiment and the actual ones
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
        a = (-(reduced_planck * vaccum_perm * gyromag_radius ** 2) / (4 * np.pi)) ** 2 #check for sign #minus is right
        intermediate_result = (a * J[0] / sig)
        #signed_intermediate = np.sign(intermediate_result)*intermediate_result #check if this is needed
        return intermediate_result**(1/6) #PFUSCH only takes lowest J not neccesarily corresponding for sig: Only correct for lowest_T



    def R_cross(self, r):
        return np.asarray((1 / 10) * self.b(r) ** 2 * (self.J(0, self.tau_C) - 6 * self.J(2 * self.omega0, self.tau_C))[:,np.newaxis])

    def sigma_strategy(self):
        #user saftey checks
        sigA_type = self.config.get("sigA")
        if sigA_type not in {"lowest_T", "fit", "true"}:
            raise ValueError(f"Invalid or missing 'sigA': {sigA_type}")
        sigB_type = self.config.get("sigB")
        if sigB_type not in {"0", "fit", "highest_T", "true"}:
            raise ValueError(f"Invalid or missing 'sigB': {sigB_type}")

        if self.config["sigA"] == "lowest_T":
            #TODO also use rows that have nan or Q atoms
            sigA = self.sigT.iloc[:,0].values #CHECK not sure about the np.abs
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







    @staticmethod
    def stability_curve_simple(T, Sm, Hm, Cp, Tm):
        """
        Stability curve model for ΔG vs. T
        """
        k_B = R = 8.314

        # return (Hm + T * Sm + Cp * (T - Tm) - T * Cp * np.log(T / Tm))/(-k_B * T) # Fitting for ln K with Sm
        return -((Hm/R)*(1/Tm-1/T) + (Cp/R) * (Tm/T - 1 + np.log(T/Tm))) # fitting for ln K
    @staticmethod
    def stability_curve_SM(T, Sm, Hm, Cp, Tm):
        k_B = R = 8.314
        return (Hm + T * Sm + Cp * (T - Tm) - T * Cp * np.log(T / Tm)) / (-k_B * T)

    def which_stability_curve(self):
        if self.config["fit_SM"] == True:
            return self.stability_curve_SM
        else:
            print("Ignore Sm values as they are not fit")
            return self.stability_curve_simple

    #TODO np.abs should not be needed
    def calc_K(self):
        if self.config["ignore_lowest_T"] == True:
            self.K = np.abs(((self.sigT-self.sigB)/(self.sig_A_corrected.T-self.sigT)).iloc[:,1:]) #PFUSCH .iloc[:,1:] is super unscientific as it removes the first diverging value same for np.abs
        else:
            self.K = np.abs(((self.sigT - self.sigB) / (self.sig_A_corrected.T - self.sigT)))

    @staticmethod
    def delta_G(T, K):
        """
        Gibbs free energy ΔG = -k_B * T * ln(K)
        T: Temperature (can be array)
        K: Equilibrium constant (same shape as T or broadcastable)
        """

        return -8.314 * T.to_numpy() * np.log(K.values.astype(float))

    def fit_stability_curve(self):
        os.makedirs("plots", exist_ok=True)
        stability_curve = self.which_stability_curve()
        def residual(params, T, data):
            Sm = params["Sm"].value
            Hm = params["Hm"].value
            Cp = params["Cp"].value
            Tm = params["Tm"].value

            model = stability_curve(T, Sm, Hm, Cp, Tm) # ln K

            diff = data - model
            err = np.linalg.norm(diff)
            # print(Sm, Hm, Cp, Tm, err)
            return diff #maybe use err

        for pair in range(0,self.K.shape[0]):
            try:
                pair_name = self.df["Name"][self.K.index[pair]]
                params = Parameters()
                params.add('Sm', value=0)
                params.add('Hm', value=1)
                params.add('Cp', value=1)
                params.add('Tm', value=1)

                experimentalData = np.log(self.K.iloc[pair].values.astype(float)) # self.delta_G(self.T, self.K.iloc[pair].values)

                mini = lmfit.Minimizer(residual, params, fcn_args=(self.T, experimentalData))
                out1 = mini.minimize(method=self.config["fit_method"])


                print(f"Results of stability fit for {pair_name}:")
                withResults = out1.params
                withResults.pretty_print()
                color = self.colors[pair]
                plt.plot(self.T, self.K.iloc[pair], ".", alpha=1, label="G" + pair_name, color=color)
                # plt.plot(self.T, self.delta_G(self.T,self.K.iloc[pair]), ".", alpha=1, label="G" + pair_name) # fits dG
                plt.plot(self.T, np.exp(stability_curve(self.T, withResults["Sm"], withResults["Hm"],
                                                      withResults["Cp"], withResults["Tm"])), label="stability" + pair_name, color=color)
                plt.xlabel("Temperature (K)")
                plt.ylabel("Equilibrium constant K(T)")
                plt.legend()
                plt.savefig(f"plots/{pair_name}.pdf", format="pdf")
                plt.show()



            except Exception as e:
                print(f"Error fitting stability curve for {pair_name} with error {e}")


        plt.xlabel("Temperature (K)")
        plt.ylabel("dG")
        plt.legend()
        plt.show()



    def run(self):


        # correct sigA for correlation times at different temperatures
        print("Choosing/interpolating correlation time ...")
        self.correct_tau_C()
        print("Picking the correct sigma ...")
        self.sigma_strategy()
        print("Calculating K analytically ...")
        self.calc_K()


        for pair in range(0,self.K.shape[0]):
            name = self.df["Name"][self.K.index[pair]]
            plt.plot(self.T, self.sigT.iloc[pair,:], label=name)
        plt.xlabel("Temperature (K)")
        plt.ylabel("sig(T)")
        plt.legend()
        plt.show()

        if self.config["ignore_lowest_T"] == True:
            self.T = self.T[1:]  # PFUSCH unscientific
        for pair in range(0, self.K.shape[0]):
            name = self.df["Name"][self.K.index[pair]]
            plt.plot(self.T, self.K.iloc[pair,:], alpha=0.5, label=name, marker=".")

        plt.xlabel("Temperature (K)")
        plt.ylabel("K")
        plt.legend()
        plt.show()

        for pair in range(0, self.K.shape[0]):
            name = self.df["Name"][self.K.index[pair]]
            plt.plot(self.T, self.delta_G(self.T, self.K)[pair,:], alpha=0.5, label=name, marker=".")

        plt.xlabel("Temperature (K)")
        plt.ylabel("ΔG [J/mol]")
        plt.legend()
        plt.show()


        print("Fitting stability curve ...")
        self.fit_stability_curve()


        #TODO check each individual step and see if things run correctly
        #TODO look into unit testing









