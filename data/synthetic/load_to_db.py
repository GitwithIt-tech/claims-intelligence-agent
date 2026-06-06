"""
data/synthetic/load_to_db.py
─────────────────────────────
Loads the generated CSVs into PostgreSQL.

Run AFTER:
  1. docker-compose up postgres redis -d
  2. python3 data/synthetic/generate_data.py

Then:
  python3 data/synthetic/load_to_db.py
"""

import sys
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.setting import db_settings

DATA_DIR = Path(__file__).parent


def load_table(engine, df: pd.DataFrame, table_name: str):
    print(f"  Loading {len(df):,} rows into '{table_name}'...")
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=500,
        method="multi",
    )
    print(f"  ✓ '{table_name}' loaded")


def main():
    print(f"Connecting to: {db_settings.HOST}:{db_settings.PORT}/{db_settings.DB}")
    engine = create_engine(db_settings.url)

    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("✓ Database connection successful\n")

    load_table(engine, pd.read_csv(DATA_DIR / "adjusters.csv"), "adjusters")
    load_table(engine, pd.read_csv(DATA_DIR / "policies.csv"),  "policies")
    load_table(engine, pd.read_csv(DATA_DIR / "claims.csv"),    "claims")

    print("\nVerification:")
    with engine.connect() as conn:
        for table in ["adjusters", "policies", "claims"]:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"  {table}: {count:,} rows")

    print("\n✓ All data loaded successfully")


if __name__ == "__main__":
    main()