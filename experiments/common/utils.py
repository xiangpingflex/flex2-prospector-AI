import snowflake.connector
import pandas as pd


def snowflake_query(query_commands: str,
                    snowflake_user: str) -> pd.DataFrame:
    # Create Snowflake connection
    cnx = snowflake.connector.connect(
        user=snowflake_user,
        account="jma23782.us-east-1",
        authenticator="externalbrowser"
    )
    # Create cursor and execute query
    cursor = cnx.cursor()
    cursor.execute(query_commands)
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    cursor.close()
    cnx.close()
    return df


def snowflake_query_from_file(query_file_path: str,
                              snowflake_user: str) -> pd.DataFrame:
    with open(query_file_path, 'r') as file:
        query_commands = file.read()
    return snowflake_query(query_commands, snowflake_user)
