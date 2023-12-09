import pandas as pd

import psycopg2
from sqlalchemy import create_engine, text

import torch

# try:
#     with engine.connect() as conn:
#         result = conn.execute(text(query))
# except Exception as e:
#     print(f"An error occurred: {e}")

def check_gpu():
    if torch.cuda.is_available():
        f"GPU is available. Detected {torch.cuda.device_count()} GPU(s)."
        return "cuda"
    else:
        "GPU is not available."
        return "cpu"

def get_data():
    pass

# Clunky single use function that moves the data provided from the excel file to my sql server hosted by my Kubernetes cluster
def write_to_sql(user, password, node_ip, port, db_name, table_name, excel_file):
    # Define the PostgreSQL URL
    # Replace NODE_IP, PORT, DB_NAME, USER, and PASSWORD appropriately
    postgres_url = f'postgresql://{user}:{password}@{node_ip}:{port}/{db_name}'

    # Create an engine
    engine = create_engine(postgres_url)

    # Read the Excel file
    df = pd.read_excel(excel_file)
    
    try:
        # Write records stored in a DataFrame to a SQL database
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Data pushed successfully to table '{table_name}' in database '{db_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
        

if __name__ == "__main__":
    pass
    # write_to_sql(user, password, node_ip, port, db_name, table_name, excel_file)