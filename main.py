
"""
This script is the main entry point for the NMR data fitting program.

It sets up the configuration for the data fitting, initializes the data_fitter class,
and runs the fitting process. It also handles the creation of a unique output directory
for each run, where it saves the configuration, logs, and plots.
"""
import shutil
import logging
import sys
import os
from datetime import datetime
from data_fitter import data_fitter
import yaml

def main():
    """
    Main entry point of the script.

    This function sets up the output directory, configures logging,
    and runs the data fitting process.
    """
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    output_dir = f"run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    with open("model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)

    # save copy of used yaml
    src = "model_config.yaml"
    dst = os.path.join(output_dir, "model_config_used.yaml")
    shutil.copy(src, dst)

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
        logging.debug("Inside try block.")
        fitter = data_fitter(model_config["data_path"], config=model_config, output_dir=output_dir)
        fitter.run()
    except ValueError as e:
        print("Caught exception:", e)
    except Exception as e:
        print("Something else went wrong:", e)


if __name__ == "__main__":
    main()
