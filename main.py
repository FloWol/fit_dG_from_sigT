#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
from data_fitter import data_fitter


## your input goes here
model_config = {
    "frequency": 700, #in MHz
    "sigA": "lowest_T",    # lowest_T // fit, true //numeric value to be coded
    "sigB": "0",   # 0, highest_T or fit, true
    "fit_method": "leastsq", # leastsq or amgo


}

data_path = "/Users/florianwolf/sigT_andrea.txt"

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