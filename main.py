
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
import yaml

from data_loader import DataLoader
import config
import datafitter
import DataProcessor


def setup_logging(output_dir: str):
    """Configures the logging for the application."""
    log_file = os.path.join(output_dir, "fit_log.txt")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )





def main():
    """
    Main entry point of the script.

    This function sets up the output directory, configures logging,
    and runs the data fitting process.
    """
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    output_dir = f"run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)

    logging.info("Starting the CLAR_fit program...")
    try:
        with open("model_config.yaml", "r") as f:
            model_config = yaml.safe_load(f)

        # 1. Load and Validate Configuration
        src = "model_config.yaml"
        dst = os.path.join(output_dir, "model_config_used.yaml")
        shutil.copy(src, dst)

        # 2. Load Data
        logging.info(f"Loading data from {model_config.data_path}...")
        loader = DataLoader(model_config.data_path)
        raw_df = loader.load_data()

        # 3. Process Data
        logging.info("Processing data")
        processor = DataProcessor.DataProcessor(model_config, raw_df)
        K, dG, sigA_correcrted, correlation_times = processor.process_data()

        # 3. Perform Data Fitting
        logging.info("Initializing the data fitter...")
        fitter = datafitter.DataFitter(config, raw_df)
        fitter.prepare_data()

        logging.info("Fitting stability curves...")
        thermo_params, fitted_curves = fitter.fit_stability_curves()

        logging.info("Fitting complete.")
        logging.info(f"Thermodynamic parameters:\n{thermo_params}")

    except ValueError as e:
        print("Caught exception:", e)
    except NotImplementedError as e:
        print("This feature is yet to be implemented, please try something else", e)
    except Exception as e:
        print("Something else went wrong:", e)


if __name__ == "__main__":
    main()
