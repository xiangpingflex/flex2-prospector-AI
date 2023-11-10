import pandas as pd


class DataHandler:
    def __init__(self, lead_info_path: str) -> None:
        self.lead_info_path = lead_info_path
        self.lead_info: pd.DataFrame = self.load_lead_info()

    def load_lead_info(self) -> pd.DataFrame:
        return pd.read_csv(self.lead_info_path)
