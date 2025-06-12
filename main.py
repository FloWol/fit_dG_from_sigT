#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import sys


## your input goes here
model_config = {
    "frequency": 700, #in MHz
    "sigA": "lowest_T",    # lowest_T, fit, true //numeric value to be coded
    "sigB": "0",   # 0, highest_T or fit, true
    # For the model there is 0,1,2 with more fit parameters with increasing number
    "model": 1, #model 2 needs fixing


}

data_path = ""

def main():
    """Main entry point of the script."""
    logging.info("Starting the program...")

    try:
        # Your main logic here
        logging.debug("Inside try block. Do something meaningful.")



    except Exception as e:
        logging.exception("An unexpected error occurred: %s", e)
        sys.exit(1)

    logging.info("Program completed successfully.")

if __name__ == "__main__":
    main()