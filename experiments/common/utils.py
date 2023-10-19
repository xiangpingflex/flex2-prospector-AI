import os

import snowflake.connector
import pandas as pd
from dotenv import load_dotenv

pd.set_option("mode.chained_assignment", None)


def snowflake_query(query_commands: str) -> pd.DataFrame:
    load_dotenv()
    snowflake_user = os.environ.get("SNOWFLAKE_USER")
    # Create Snowflake connection
    cnx = snowflake.connector.connect(
        user=snowflake_user,
        account="jma23782.us-east-1",
        authenticator="externalbrowser",
    )
    # Create cursor and execute query
    cursor = cnx.cursor()
    cursor.execute(query_commands)
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    cursor.close()
    cnx.close()
    return df


def snowflake_query_from_file(query_file_path: str) -> pd.DataFrame:
    with open(query_file_path, "r") as file:
        query_commands = file.read()
    return snowflake_query(query_commands)


def cat_encode(df, fillnan: bool = True) -> dict:
    from sklearn.preprocessing import LabelEncoder

    class_map = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if fillnan:
            df[col].fillna("MISSING", inplace=True)
        le = LabelEncoder()
        le = le.fit(df[col])
        df[col] = le.transform(df[col])
        df[col] = df[col].astype("category")
        class_map[col] = dict(zip(le.transform(le.classes_), le.classes_))
    return class_map


def chi2_test(X, y) -> pd.DataFrame:
    """
    :param X: df with all the categorical features (must be encoded)
    :param y: target variable
    :return:
    """
    from sklearn.feature_selection import chi2, SelectKBest, f_classif

    chi2_selector = SelectKBest(chi2, k="all")
    # chi_ret = chi2_selector.fit_transform(X, y) #transfer_back
    chi2_selector.fit(X, y)
    chi2_scores = pd.DataFrame(
        list(zip(X.columns, chi2_selector.scores_, chi2_selector.pvalues_)),
        columns=["ftr", "score", "pval"],
    )
    chi2_scores["p_test"] = chi2_scores["pval"].apply(
        lambda x: "SIG" if x <= 0.05 else "N-SIG"
    )
    chi2_scores.sort_values(by=["score"], inplace=True, ascending=False)
    return chi2_scores


def chi2_test_encode(
    df, categorical_cols, target_cols, fillnan: bool = True
) -> list[pd.DataFrame]:
    """

    :param df: df for checking
    :param categorical_cols:
    :param target_cols:
    :return:
    """
    df_cat = df[categorical_cols]
    cat_encode(df_cat, fillnan)
    return [chi2_test(df_cat, df[target_col]) for target_col in target_cols]


# def f_oneway_test(df, col_name, col_target):
#     import scipy.stats as stats
#     categorical_var = df[col_name]
#     numerical_target = df[col_target]
#     from scipy.stats import f_oneway
#     # Perform ANOVA
#     f_statistic, p_value = stats.f_oneway(
#         *[numerical_target[categorical_var == category] for category in categorical_var.unique()])
#
#     # Check the p-value
#     if p_value < 0.05:
#         print(
#             f"There is a significant correlation between the categorical variable {col_name.upper()} and the numerical target {col_target.upper()}")
#     else:
#         print(
#             f"There is no significant correlation between the categorical variable {col_name.upper()} and the numerical target {col_target.upper()}.")
#     return f_statistic, p_value
