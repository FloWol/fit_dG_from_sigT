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
        self.df_filtered = self.df[~self.df["Name"].str.contains("Q", na=False)].dropna() #drop pseudo atoms
        self.T = self.df.columns[3:]
        self.k_B = 1.380649e-23 #J/K

        if "tau_C" in self.df["Name"]:
            self.tau_C = self.df["tau_C"] #TODO check if its as long as all the temperatures otherwise Fit rH
        else:
            self.tau_C_incomplete = self.df["tau_C"]
            self.fit_rH()
            self.tau_C = self.calc_tau_C(self.rH)


    def viscocity(self):
        """
        for water
        :return:
        """
        A = 1.856e-11
        B = 4209.0
        C = 0.04527
        D = -3.376e-5
        mu = A * sp.exp(B / self.T + C * self.T + D * self.T ** 2)
        self.eta = mu * 1.0e-3

    def calc_tau_C(self, rH):
        """
        from stokes law
        :param rH:
        :return:
        """
        self.tau_C = 4.0 * np.pi * self.eta * rH ** 3 / (3.0 * self.k_B * self.T)

    def fit_rH(self):
        """
        This function takes an incomplete list auf tau_C from the data file and fits the radius of hydration r_H
        to the correlation times. This can later be used to calculate the missing tau_Cs
        :return:
        """
        tau_Clambd =  lambda rH: 4.0 * np.pi * self.eta * rH ** 3 / (3.0 * self.k_B * self.T)
        def residual(params, data):
            rH = params["rH"].value
            model = tau_Clambd(rH)

            diff = data - model
            err = np.linalg.norm(diff)
            return diff

        params = Parameters()
        params.add('rH', value=1)
        experimentalData = self.tau_C_incomplete

        mini = lmfit.Minimizer(residual, params, fcn_args=(self.T, experimentalData))
        out1 = mini.minimize(method="leastsq") #or try amgo


        print("Results of rH fit:")
        withResults = out1.params
        withResults.pretty_print()

        self.rH = out1.params["rH"].value




    def b(self):
        pass
    def J(self, w):
        return (self.tau_C / (1 + (w * self.tau_C) ** 2))

    def R_cross(self, w, tau_c):
        return 1 / 10 * self.b ** 2 * (self.J(0, tau_c) - 6 * self.J(2 * w, tau_c))


    def run(self):
        pass





