import os
import numpy as np
import scipy as sp
import pandas as pd
import lmfit
from lmfit import minimize, Parameters
import matplotlib.pyplot as plt
from pygments.unistring import Sm
import logging
import data_loader
from config import ModelConfig


class DataFitter():
    """A class to fit experimental NMR data to thermodynamic models."""
    def __init__(self, config=None, raw_df=None, output_dir=None):
        """
         Initializes the data_fitter class.

         Args:
             data_file (str): The path to the input data file.
             config (dict, optional): A dictionary of configuration options. Defaults to None.
             output_dir (str, optional): The path to the output directory. Defaults to None.
         """
        self.config: ModelConfig = config
        loader = data_loader.DataLoader(self.config["data_path"])
        self.df = loader.load_data()

        self.initial_Tm = self.config['Tm']
        self.output_dir = output_dir





        self.T = self.df.columns[3:].astype(float)

        n_colors = self.df.shape[0]
        base_colors = list(plt.cm.tab20.colors)  # 20 distinct colors
        self.colors = [base_colors[i % len(base_colors)] for i in range(n_colors)]

        import plot_style_publication


        self.k_B = 1.380649e-23 #J/K
        self.freq = config["frequency"]*1e6
        self.omega0 = 2.0 * np.pi * self.freq

        self.sigT = self.df.iloc[:-1,3:]

    def prepare_data(self):
        pass






        if self.config["sigB"] == "0":
            self.sigB = 0
        if self.config["sigB"] == "highest_T":
            self.sigB = 0
            raise NotImplementedError
        if self.config["sigB"] == "custom":
            self.sigB = self.df["sigB"]
        if self.config["sigB"] == "custom_temp":
            self.sigB = self.df[self.config["custom_temp"]] #TODO check with temperature and file

        # elif self.config["sigB"] == "fit":



    def which_stability_curve(self):
        """
        Selects the appropriate stability curve model based on the configuration.

        Returns:
            function: The selected stability curve function.
        """
        if self.config["Cp_model"] == True:
            if self.config["fit_SM"] == True:
                return self.stability_curve_SM
            else:
                logging.info("Ignore Sm values as they are not fit")
                return self.stability_curve_no_Sm
        else:  # Use van't Hoff equation
            logging.info("Ignore Cp values as they are not fit")
            if self.config["fit_SM"] == True:
                logging.info("Ignore Tm values as they are not fit")
                return self.vant_Hoff_Sm
            else:
                logging.info("Ignore Sm values as they are not fit")
                return self.vant_Hoff



    @staticmethod
    def delta_G(T, K):
        """
         Calculates the Gibbs free energy (ΔG) from the equilibrium constant (K).

         Args:
             T (pd.Series): A pandas Series of temperatures in Kelvin.
             K (pd.Series): A pandas Series of equilibrium constants.

         Returns:
             np.ndarray: An array of ΔG values in J/mol.
         """

        return -8.314 * T.to_numpy() * np.log(K.values.astype(float))

    def fit_stability_curve(self):
        """
        Fits the experimental data to the selected stability curve model.
        """
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        for pair in range(0,self.K.shape[0]):
            try:
                pair_name = self.df["Name"][self.K.index[pair]]
                experimentalData = np.log(self.K.iloc[pair].values.astype(float)) # self.delta_G(self.T, self.K.iloc[pair].values)
                valid_indices = ~np.isnan(experimentalData) & ~np.isinf(experimentalData)
                T_fit = self.T[valid_indices]
                experimentalData_fit = experimentalData[valid_indices]

                if len(experimentalData_fit) == 0:
                    logging.info(f"No data for {pair_name}, skipping.")
                    continue


                params = Parameters()
                params.add("Sm", value=10, vary=self.config["fit_SM"])
                params.add('Hm', value=1000,)
                params.add('Cp', value=1)

                if self.config["Tm"] is not None:
                    if self.config["fix_Tm"]:
                        params.add('Tm', value=self.initial_Tm, vary=False)
                        logging.info("Tm is fixed, ignore any values outputted for Tm")
                    else:
                        params.add('Tm', value=self.initial_Tm, vary=True, min=100, max=400)
                else:
                    params.add('Tm', value=300)

                stability_curve = self.which_stability_curve()



                def residual(params, T, data):
                    Sm = params["Sm"].value
                    Hm = params["Hm"].value
                    Cp = params["Cp"].value
                    Tm = params["Tm"].value

                    model = stability_curve(T, Sm, Hm, Cp, Tm)  # ln K

                    diff = data - model
                    return diff  # maybe use err


                mini = lmfit.Minimizer(residual, params, fcn_args=(T_fit, experimentalData_fit))
                out1 = mini.minimize(method=self.config["fit_method"], max_nfev=self.config["max_iterations"])


                logging.info(f"Results of stability fit for {pair_name}:")
                withResults = out1.params
                withResults.pretty_print()
                report = lmfit.fit_report(out1)
                logging.info("=== Fit Report ===\n%s", report)
                color = self.colors[pair]

                plt.plot(self.T, -8.314*self.T*np.log(self.K.iloc[pair]), ".", alpha=1, label=pair_name, color=color)
                plt.plot(self.T, -8.314*self.T*stability_curve(self.T, withResults["Sm"], withResults["Hm"],
                                                      withResults["Cp"], withResults["Tm"]), label="stability" + pair_name, color=color)
                plt.xlabel("Temperature (K)")
                plt.ylabel(r"$\Delta G [J/mol]$")
                plt.legend()

                plt.savefig(os.path.join(self.output_dir, f"plots/{pair_name}.pdf"), format="pdf")
                plt.show()



            except Exception as e:
                logging.error(f"Error fitting stability curve for {pair_name} with error {e}")




    def run(self):
        """
        Runs the entire data fitting process.
        """

        # correct sigA for correlation times at different temperatures
        logging.info("Choosing/interpolating correlation time ...")
        self.correct_tau_C()
        logging.info("Picking the correct sigma ...")
        self.sigma_strategy()
        logging.info("Calculating K analytically ...")
        self.calc_K()


        for pair in range(0,self.K.shape[0]):
            name = self.df["Name"][self.K.index[pair]]
            plt.plot(self.T, self.sigT.iloc[pair,:], label=name)
        plt.xlabel("Temperature (K)")
        plt.ylabel("sig(T)")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "sig_T.pdf"), format="pdf")
        plt.show()

        if self.config["ignore_lowest_T"] == True:
            self.T = self.T[1:]  # PFUSCH
        for pair in range(0, self.K.shape[0]):
            name = self.df["Name"][self.K.index[pair]]
            plt.plot(self.T, self.K.iloc[pair,:], alpha=0.5, label=name, marker=".")

        plt.xlabel("Temperature (K)")
        plt.ylabel("K")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "K_T.pdf"), format="pdf")
        plt.show()

        for pair in range(0, self.K.shape[0]):
            name = self.df["Name"][self.K.index[pair]]
            plt.plot(self.T, self.delta_G(self.T, self.K)[pair,:], alpha=0.5, label=name, marker=".")

        plt.xlabel("Temperature (K)")
        plt.ylabel("ΔG [J/mol]")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "delta_G_T.pdf"), format="pdf")
        plt.show()



        logging.info("Fitting stability curve ...")
        self.fit_stability_curve()









