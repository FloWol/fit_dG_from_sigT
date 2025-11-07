#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
from data_fitter import data_fitter

## your input goes here
model_config = {
    # General settings
    "frequency": 700,  # in MHz›
    "fit_method": "amgo", # leastsq or amgo or anything scipy supports (amgo is better, but takes longer, but better take amgo)

    # reference sigma settings
    "sigA": "lowest_T", # lowest_T, or custom for usage of sigA values in sigA column (not tested), // fit,or, numeric value to be coded
    "sigB": "0",  # 0, highest_T or fit or custom for usage of values from file
    "correct_for_correlation_time": False, # recommended: True corrects sigma for T-dependent global correlation time, False takes

    # Fitting equation settings
    "Cp_model": False, # True/False, False results in the Use of the van't Hoff equation
    # for van'T Hoff fitting fit_Sm = True is recomended, when using Cp_model = True, fit_Sm = False is recomended to prevent overfitting
    "fit_SM": True,  # True or False #
    "Tm": 300,  # optional, is just used as an initial guess for Tm in the fit, numeric value in K or None (sets value to 1)
    "fix_Tm": False,  # True of False, if True Tm won't be fit (to be used with setting Tm above)

    # test_features (will be removed in the future)
    "ignore_lowest_T": True,  # not recommended but often necessary

}
########### PUT THE PATH TO YOUR DATA FILE HERE ###################
data_path = "/Users/florianwolf/Downloads/sig_andrea.txt" #"/Users/florianwolf/Downloads/sigmaT_guneet_floformat.txt" #


def main():
    """Main entry point of the script."""
    logging.info("Starting the program...")

    try:
        # Your main logic here
        logging.debug("Inside try block. Do something meaningful.")
        fitter = data_fitter(data_path, config=model_config)
        fitter.run()


    except Exception as e:
        logging.exception("An unexpected error occurred: %s", e)
        sys.exit(1)

    logging.info("Program completed successfully.")


if __name__ == "__main__":
    main()
