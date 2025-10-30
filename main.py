#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is the main entry point for the NMR data fitting program.

It sets up the configuration for the data fitting, initializes the data_fitter class,
and runs the fitting process. It also handles the creation of a unique output directory
for each run, where it saves the configuration, logs, and plots.
"""

import argparse
import logging
import sys
import os
import json
from datetime import datetime
from data_fitter import data_fitter

## your input goes here
model_config = {
    # General settings
    "frequency": 700,  # in MHz
    "fit_method": "amgo", # leastsq or amgo or anything scipy supports (amgo is better, but takes longer, but better take amgo)
    "max_iterations": 1000,

    # reference sigma settings
    "sigA": "lowest_T", # lowest_T, or custom for usage of sigA values in sigA column (not tested), // fit,or, numeric value to be coded
    "sigB": "0",  # 0, highest_T or fit or custom for usage of values from file
    "correct_for_correlation_time": False, # recommended: True corrects sigma for T-dependent global correlation time, False takes

    # Fitting equation settings
    "Cp_model": True, # True/False, False results in the Use of the van't Hoff equation
    "fit_SM": False,  # True or False
    "Tm": 330,  # optional, is just used as an initial guess for Tm in the fit, numeric value in K or None (sets value to 1)
    "fix_Tm": False,  # True of False, if True Tm won't be fit (to be used with setting Tm above)

    # test_features (will be removed in the future)
    "ignore_lowest_T": True,  # not recommended but often necessary

}
########### PUT THE PATH TO YOUR DATA FILE HERE ###################
data_path = "/Users/florianwolf/Downloads/sig_andrea.txt" #"/Users/florianwolf/Downloads/sigmaT_guneet_floformat.txt" #


def main():
    """
    Main entry point of the script.

    This function sets up the output directory, configures logging,
    and runs the data fitting process.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    config_file = os.path.join(output_dir, "model_config.json")
    with open(config_file, "w") as f:
        json.dump(model_config, f, indent=4)

    log_file = os.path.join(output_dir, "fit_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("Starting the program...")

    try:
        # Your main logic here
        logging.debug("Inside try block. Do something meaningful.")
        fitter = data_fitter(data_path, config=model_config, output_dir=output_dir)
        fitter.run()


    except Exception as e:
        logging.exception("An unexpected error occurred: %s", e)
        sys.exit(1)

    logging.info("Program completed successfully.")


if __name__ == "__main__":
    main()
