#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
from data_fitter import data_fitter


## your input goes here
model_config = {
    "frequency": 700, #in MHz
    "sigA": "lowest_T",    # lowest_T, or custom for usage of sigA values in sigA column // fit,or, numeric value to be coded
    "sigB": "0",   # 0, highest_T or fit or custom for usage of values from file
    "fit_method": "amgo", # leastsq or amgo or anything scipy supports (amgo is better, but takes longer, but better take amgo)
    "fit_SM": False, # True or False
    "melting_temperature": 250, # is just used as an initial guess for Tm in the fit

    # test_features (will be removed in the future)
    "ignore_lowest_T": True, # not recommended but often necessary

}
########### PUT THE PATH TO YOUR DATA FILE HERE ###################
data_path = "/Users/florianwolf/Downloads/andrea_sigT.txt"

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