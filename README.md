# NMR Data Fitter

This program loads experimental NOESY NMR data and fits the equilibrium constant while correcting for the temperature dependence in the global correlation time. Fitted interactions are currently only between pairs of atoms.

## Features

*   **Handles Missing Data:** The script can fit data even if some temperature points are missing, without dropping the entire row.
*   **Unique Output Directories:** Each run generates a unique, timestamped directory (e.g., `run_2023-10-29_10-30-00`) to store all outputs. This prevents overwriting results from previous runs.
*   **Comprehensive Logging:** All console output is saved to a `fit_log.txt` file within the run's output directory.
*   **Reproducibility:** The configuration for each run is saved to `model_config.json` in the output directory.
*   **Robust Error Handling:** If there isn't enough data to fit the full thermodynamic model, the script will automatically switch to a simpler van't Hoff model and log a warning.
*   **SI Units:** The output of the fitted parameters includes SI units for clarity.
*   **Configurable Fitting:** The maximum number of iterations for the fitting procedure can be customized.

## How to Use

1.  **Configure the fit:** Open `main.py` and modify the `model_config` dictionary and the `data_path` variable.

    ```python
    ## your input goes here
    model_config = {
        # General settings
        "frequency": 700,  # in MHz
        "fit_method": "amgo", # leastsq or amgo or anything scipy supports
        "max_iterations": 1000, # Maximum number of fitting iterations

        # reference sigma settings
        "sigA": "lowest_T", # 'lowest_T', 'fit', or 'custom'
        "sigB": "0",  # '0', 'highest_T', 'fit', or 'custom'
        "correct_for_correlation_time": False, # Correct sigma for T-dependent global correlation time

        # Fitting equation settings
        "Cp_model": True, # If False, the van't Hoff equation is used
        "fit_SM": False,  # Fit for Sm
        "Tm": 330,  # Initial guess for Tm in Kelvin
        "fix_Tm": False,  # Fix Tm during the fit

        # test_features (will be removed in the future)
        "ignore_lowest_T": True,

    }
    ########### PUT THE PATH TO YOUR DATA FILE HERE ###################
    data_path = "path/to/your/data.txt"
    ```

2.  **Run the script:**
    ```bash
    python3 main.py
    ```

3.  **Check the output:** A new directory named `run_YYYY-MM-DD_HH-MM-SS` will be created. It will contain:
    *   `fit_log.txt`: The log file with all the output from the run.
    *   `model_config.json`: The configuration used for the run.
    *   PDF plots for each fitted pair and summary plots.
