import pandas as pd
import threading
from collections import defaultdict
from typing import Dict, List, Any

# ======================== DataLoader Class =======================
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
# ======================== Mapper Class ========================
class Mapper:
    """Mapper processes each data chunk and emits local count of flights per passenger."""

    def __init__(self):
        self.local_count = defaultdict(int)  # Dictionary to store intermediate counts

    def map(self, data_chunk: pd.DataFrame) -> None:
        """
        Iterates over each row to count flights per Passenger_ID.
        Ignores null values.
        """
        for _, row in data_chunk.iterrows():
            passenger_id = row.get('Passenger_ID')
            if pd.notnull(passenger_id):
                self.local_count[passenger_id] += 1

    def get_local_count(self) -> Dict[str, int]:
        """Returns the local (thread-specific) passenger flight counts."""
        return self.local_count
# ======================== Combiner Class ========================
class Combiner:
    """Combiner aggregates results from each Mapper before passing to Reducer."""

    @staticmethod
    def combine(data: Dict[str, int]) -> Dict[str, int]:
        """
        Merges partial results into a new combined dictionary.
        Helps reduce data size before final reduction.
        """
        combined_data = defaultdict(int)
        for passenger_id, count in data.items():
            combined_data[passenger_id] += count
        return combined_data

# ======================== Reducer Class ========================
class Reducer:
    """Reducer identifies the passenger(s) with the highest number of flights."""

    @staticmethod
    def reduce(combined_data: Dict[str, int]) -> (List[str], int):
        """
        Determines the max flight count and corresponding passenger(s).
        Returns both as a tuple.
        """
        if not combined_data:
            return [], 0
        max_flights = max(combined_data.values())
        top_passengers = [pid for pid, count in combined_data.items() if count == max_flights]
        return top_passengers, max_flights

# ======================== MapReduce Class ========================
class MapReduce:
    """
    Implements a multithreaded MapReduce framework.
    Executes Mapper, Combiner, and Reducer using parallelism.
    """

    def __init__(self, data: pd.DataFrame, num_threads: int = 4):
        self.data = data
        self.num_threads = num_threads
        self.result_dict = defaultdict(int)

    def execute(self) -> Dict[str, int]:
        """
        Splits data into chunks for parallel mapping.
        Aggregates all thread results into a global dictionary.
        """
        chunk_size = len(self.data) // self.num_threads
        threads = []

        for i in range(self.num_threads):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.num_threads - 1 else len(self.data)
            data_chunk = self.data.iloc[start_idx:end_idx]

            mapper = Mapper()
            thread = threading.Thread(target=mapper.map, args=(data_chunk,))
            threads.append((thread, mapper))
            thread.start()

        for thread, mapper in threads:
            thread.join()
            combined_data = Combiner.combine(mapper.get_local_count())
            for passenger_id, count in combined_data.items():
                self.result_dict[passenger_id] += count

        return self.result_dict

    def get_result(self) -> (List[str], int):
        """
        Applies Reducer on the final aggregated result to get top flyers.
        """
        return Reducer.reduce(self.result_dict)

# ======================== Driver Code ========================

# Step 1: Load data from CSV file using DataLoader
data_loader = DataLoader()
passenger_data = data_loader.load_data("AComp_Passenger_data_no_error.csv")

# Step 2: Assign proper column names and ensure data consistency
passenger_data.columns = ['Passenger_ID', 'Flight_Number', 'Departure', 'Arrival', 'Timestamp', 'Flight_Duration']
passenger_data['Passenger_ID'] = passenger_data['Passenger_ID'].astype(str)

# Step 3: Initialize and run MapReduce only if data is valid
if not passenger_data.empty:
    mapreduce = MapReduce(passenger_data, num_threads=4)
    mapreduce.execute()
    top_passengers, max_flights = mapreduce.get_result()

    # Step 4: Display result of passengers with maximum flights
    print("Passengers with the most flights:", top_passengers)
    print("Number of flights:", max_flights)
else:
    print("Data is empty or not loaded properly.")