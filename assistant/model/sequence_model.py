import pickle

import pandas as pd
import datetime
import numpy as np
import math

from assistant.api.sagemaker_client import SagemakerClient


class SequenceModel:
    def __init__(
        self,
        region_name: str,
        endpoint_name: str,
        profile_name: str,
        category_map_path: str,
    ) -> None:
        self.region_name = region_name
        self.endpoint_name = endpoint_name
        self.category_map_path = category_map_path
        self.category_map = self.load_category_map(self.category_map_path)
        self.sagemaker_client = SagemakerClient(
            profile_name=profile_name,
            region_name=self.region_name,
            endpoint_name=self.endpoint_name,
        )

    def load_category_map(self, path: str):
        with open(path, "rb") as file:
            revert_class_map = pickle.load(file)
        return revert_class_map

    def input_data_convert(self, raw_input_data: pd.DataFrame) -> str:
        df = raw_input_data.copy()
        for col in df.columns:
            if col in self.category_map:
                df[col] = df[col].map(self.category_map[col])
        df["company_founded_years"] = (
            datetime.datetime.now().year - df["company_founded_year"]
        )
        df = df[
            [
                "sequence_name_cat",
                "contact_job_title_level",
                "contact_job_title_department",
                "company_protfolio_type",
                "company_protfolio_subtype",
                "company_segment",
                "company_state",
                "contact_state",
                "company_annual_revenue",
                "company_units",
                "company_founded_years",
            ]
        ]
        csv_record = df.to_csv(index=False, header=False)
        return csv_record

    def prob_to_score(self, prob, basePoint=600, PDO=100):
        y = np.log(prob / (1 - prob))
        if y == math.inf:
            y = 1000
        if y == -math.inf:
            y = -1000
        score = int(basePoint + PDO * y)
        return score

    def predict_prob(self, raw_input_data: pd.DataFrame) -> float:
        return self.sagemaker_client.predict_prob(
            self.input_data_convert(raw_input_data)
        )

    def predict_score(self, raw_input_data: pd.DataFrame) -> int:
        prob = self.predict_prob(raw_input_data)
        return self.prob_to_score(prob, basePoint=600, PDO=100)
