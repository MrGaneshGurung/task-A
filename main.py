import pandas as pd
import threading
from collections import defaultdict
from typing import Dict, List, Any

# ======================== DataLoader Class ========================
class DataLoader:
    """Class to load data efficiently using buffered reading and error handling."""

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """
        Reads data from a CSV file in chunks to handle large files.
        Returns a combined DataFrame or an empty one in case of error.
        """
        try:
            with open(file_path, 'r') as file:
                data = pd.read_csv(file, chunksize=1000)
                df = pd.concat(data, ignore_index=True)
            return df
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()