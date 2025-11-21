import logging

from config import ModelConfig
import pandas as pd
import physics
import numpy as np


class DataProcessor():
    def __init__(self, config: ModelConfig, raw_df: pd.DataFrame):



        self.config = config
        self.raw_df = raw_df

        self.names = raw_df["Name"]
        self.temperatures = raw_df.columns[3:].astype(float)
        self.correlation_time = raw_df.iloc[-1,3:]
        self.experimental_values = raw_df.iloc[0:-1,3:]

        self.freq = config["frequency"]*1e6
        self.omega0 = 2.0 * np.pi * self.freq

        self.sigA = None
        self.sigB = None
        self.sigA_corrected = None
        self.K = None
        self.dG_from_sigma = None


    def sigma_strategy(self) -> None:
        sigA_type = self.config["sigA"]
        sigB_type = self.config["sigB"]

        # for sigma A
        if sigA_type == "lowest_T":
            self.sigA = self.experimental_values.iloc[:,0]
        elif sigA_type == "fit":
            raise NotImplementedError
        elif sigA_type == "custom":
            self.sigA = self.raw_df["sigA"]
        elif sigA_type == "custom_temp":
            raise NotImplementedError

        # for sigma B
        if sigB_type == "0":
            self.sigB = 0
        elif sigB_type == "highest_T":
            self.sigB = self.experimental_values.iloc[:, -1]
        elif sigB_type == "fit":
            raise NotImplementedError
        elif sigB_type == "custom":
            self.sigB = self.raw_df["sigB"]
        elif sigB_type == "custom_temp":
            raise NotImplementedError

    def calc_correlation_time(self):
        if len(self.correlation_time.dropna()) != len(self.temperatures):
            logging.info("Interpolating correlation time")
            correlation_time_calculator = physics.CorrelationTimeCalculator.create(self.config)
            self.correlation_time = correlation_time_calculator.calculate(self.correlation_time, self.temperatures)
        else:
            logging.info("Taking provided correlation times, as they are enough")
            # self.correlation_time already includes the ones


    def correct_sigma(self):
        """
        Only supports lowest T for now
        :return:
        """
        if self.config["correct_for_correlation_time"] == True:
            if self.config["sigA"] == "lowest_T":
                distance_A = physics.distance_from_sigma(self.sigA, self.correlation_time, self.omega0)
                self.sigA_corrected = physics.calc_cross_relaxation_rate(distance_A, self.correlation_time, self.omega0)
        else:
            self.sigA_corrected = self.sigA


    def calc_equilibrium_constant(self):
        if self.config["ignore_lowest_T"]:
            self.K = ((self.experimental_values-self.sigB)/(self.sigA_corrected-self.experimental_values.T).T).iloc[:,1:]
        else:
            self.K = ((self.experimental_values - self.sigB) / (self.sigA_corrected.T - self.experimental_values))


    @staticmethod
    def calc_Gibbs_free_energy(T, K):
        return -8.314 * T * np.log(K)


    def calc_Gibbs_from_sigma(self):
        self.dG_from_sigma = self.calc_Gibbs_free_energy(self.temperatures, self.K)


    def process_data(self):
        # sigma handling
        self.sigma_strategy()
        if self.config["correct_for_correlation_time"]:
            self.calc_correlation_time()
        self.correct_sigma()

        # Calculating values
        self.calc_equilibrium_constant()
        self.calc_Gibbs_from_sigma()

        return self.K, self.dG_from_sigma, self.sigA_corrected, self.correlation_time
