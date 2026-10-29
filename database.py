import pandas as pd
from sqlalchemy import create_engine, text

# Change these values for your local MySQL installation.
DB_URL = "mysql+pymysql://root:YOUR_PASSWORD@localhost/customer_churn"

def get_engine():
    return create_engine(DB_URL)

def load_csv_to_mysql(csv_path="data/customers.csv"):
    df = pd.read_csv(csv_path)
    engine = get_engine()
    df.to_sql("customers", engine, if_exists="replace", index=False)
    print("Loaded customers table.")

if __name__ == "__main__":
    load_csv_to_mysql()
