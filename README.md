# CLAR_fit

## Description

CLAR_fit is a Python tool designed to analyze Nuclear Magnetic Resonance (NMR) data to determine thermodynamic parameters. It loads experimental data, fits it to various thermodynamic models, and corrects for the temperature dependence of the global correlation time.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/CLAR_fit.git
    cd CLAR_fit
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the program, execute the `main.py` script:

```bash
python main.py
```

The script will create a new directory named `run_<timestamp>` containing the output of the analysis, including plots and a log file.

## Configuration

The behavior of the fitting process is controlled by the `model_config.yaml` file. Here are the available options:

-   `frequency`: The NMR frequency in MHz.
-   `fit_method`: The fitting algorithm to use (e.g., `leastsq`, `amgo`).
-   `data_path`: The path to the input data file.
-   `sigA`: The method for determining `sigA` (`lowest_T`, `fit`, `custom`, `custom_temp`).
-   `sigB`: The method for determining `sigB` (`0`, `highest_T`, `fit`, `custom`, `custom_temp`).
-   `correct_for_correlation_time`: Whether to correct sigma for the temperature-dependent global correlation time.
-   `max_iterations`: The maximum number of iterations for the fitting process.
-   `Cp_model`: If `True`, the stability curve model including Cp is used. If `False`, the van't Hoff equation is used.
-   `fit_SM`: If `True`, Sm is included as a fitting parameter.
-   `Tm`: An initial guess for the melting temperature (Tm) in Kelvin.
-   `fix_Tm`: If `True`, Tm will be fixed to the initial guess and not be a fitting parameter.
-   `ignore_lowest_T`: If `True`, the lowest temperature data point is ignored in the fitting and plotting.
