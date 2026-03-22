"""
Pseudo-database engine backed by pandas/parquet files
We bother with Engine instead of simple "read_csv" to abstract the rest of the codebase from
the way of getting data. That way we can simple choose the build_eninge() once we move over to
production environment, instead of rewriting walls of code
"""

import os
import pandas as pd

class Engine:
    """Holds references to all data tables by name """

    def __init__(self, tables: dict[str, pd.DataFrame]):
        self.tables = tables

    def get_table(self, name: str) -> pd.DataFrame:
        return self.tables[name]
    
def build_engine(data_root: str) -> Engine:
    """Load all dataset files and return a ready Engine"""
    return Engine(tables={
        "customers": pd.read_csv(os.path.join(data_root, "customers.csv")),
        "receipts": pd.read_parquet(os.path.join(data_root, "receipts.parquet")),
        "campaigns": pd.read_csv(os.path.join(data_root, "campaigns.csv"))
    })
