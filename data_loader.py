import pandas as pd

class DataLoader:
    """
    Handles loading and initial parsing of the experimental data.
    """
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.df = None


    def load_data(self) -> pd.DataFrame:
        try:
            self.df = pd.read_csv(self.data_path, sep=None, engine="python")
        except:
            print("Input file is not  did not load properly, please check")
        return self.df
