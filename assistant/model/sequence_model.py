import pickle
from collections import defaultdict

import pandas as pd
import datetime
import numpy as np
import math
from assistant.model.all_sequence_cat import ALL_SEQUENCE_CATEGORY
from assistant.api.sagemaker_client import SagemakerClient
import json


class SequenceModel:
    def __init__(
        self,
        region_name: str,
        endpoint_name: str,
        profile_name: str,
        category_encode_map_path: str,
        reverse_category_encode_map_path: str,
        email_template_path: str,
    ) -> None:
        self.category_encode_map = self.load_category_encode_map(
            category_encode_map_path
        )
        self.reverse_category_encode_map = self.load_reverse_category_encode_map(
            reverse_category_encode_map_path
        )
        self.email_template = self.load_email_template(email_template_path)
        self.sagemaker_client = SagemakerClient(
            profile_name=profile_name,
            region_name=region_name,
            endpoint_name=endpoint_name,
        )
        self.all_sequence_cat = ALL_SEQUENCE_CATEGORY

    def load_category_encode_map(self, path: str):
        with open(path, "rb") as file:
            category_encode_map = pickle.load(file)
        return category_encode_map

    def load_reverse_category_encode_map(self, path: str):
        with open(path, "rb") as file:
            reverse_category_encode_map = pickle.load(file)
        return reverse_category_encode_map

    def load_email_template(self, path):
        with open(path, "r") as file:
            data = json.load(file)
        return data

    def input_data_convert(self, raw_input_data: pd.DataFrame) -> str:
        df = raw_input_data.copy()
        for col in df.columns:
            if col in self.reverse_category_encode_map:
                df[col] = df[col].map(self.reverse_category_encode_map[col])
        df["company_founded_years"] = (
            datetime.datetime.now().year - df["company_founded_year"]
        ).astype(int)
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
        csv_record = df.to_csv(index=False, header=False).replace("\n", "")
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
        x = self.input_data_convert(raw_input_data)
        return self.sagemaker_client.predict_prob(
            self.input_data_convert(raw_input_data)
        )

    def predict_score(self, raw_input_data: pd.DataFrame) -> int:
        prob = self.predict_prob(raw_input_data)
        return self.prob_to_score(prob, basePoint=600, PDO=100)

    def get_top_sequence_category(
        self, top_n: int, raw_input_data: pd.DataFrame
    ) -> list:
        df_score = pd.DataFrame()
        for seq_cat in self.all_sequence_cat:
            new_df = raw_input_data.copy()
            new_df["sequence_name_cat"] = seq_cat
            new_df["score"] = self.predict_score(new_df)
            df_score = pd.concat([df_score, new_df])
        df_score.sort_values(by="score", ascending=False, inplace=True)
        return df_score[0:top_n]["sequence_name_cat"].values.tolist()

    def get_top_sequence_email_template(
        self, top_n: int, raw_input_data: pd.DataFrame
    ) -> dict:
        top_sequence_categories = self.get_top_sequence_category(top_n, raw_input_data)
        sequence_info = defaultdict(list)
        for seq_category in top_sequence_categories:
            for item in self.email_template:
                if item["sequence_cat"] == seq_category:
                    sequence_info[seq_category].append(item)
        sequence_info_sorted = {
            k: sorted(v, key=lambda x: x["step"]) for k, v in sequence_info.items()
        }
        return sequence_info_sorted
