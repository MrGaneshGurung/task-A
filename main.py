import pandas as pd
import threading
from collections import defaultdict
from typing import Dict, List, Any

# ======================== DataLoader Class =======================
class DataLoader:
    """
    Responsible for reading and loading CSV flight data into a pandas DataFrame.
    Uses buffered reading via chunks to support large datasets efficiently.
    """

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """
        Reads data from a CSV file using chunki9ng for memory efficiency.
        Returns a concatenated DataFrame or an empty one if an error occurs.
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
    """
    Processes a chunk of flight data and emits key-value pairs.
    In this case, each key is a 'Passenger_ID', and the value is a flight count (1).
    Each Mapper handles a seperate chunk of the full dataset in a seperate thread.
    """

    def __init__(self):
        self.local_count = defaultdict(int)  # Stores passenger-wise flight counts

    def map(self, data_chunk: pd.DataFrame) -> None:
        """
        Processes each row in the data chunk.
        Emits (Passenger_ID, 1) for each valid record and stores count in memory.
        """
        for _, row in data_chunk.iterrows():
            passenger_id = row.get('Passenger_ID')
            if pd.notnull(passenger_id):
                self.local_count[passenger_id] += 1

    def get_local_count(self) -> Dict[str, int]:
        """Returns the local dictionary of flight counts per passenger for this mapper."""
        return self.local_count
# ======================== Combiner Class ========================
class Combiner:
    """Aggregates intermediate key-value pairs (i.e., passenger flight counts) from a Mapper before passing them to the Reducer. This reduces the data size passed between Map and Reduce phases."""

    @staticmethod
    def combine(data: Dict[str, int]) -> Dict[str, int]:
        """
        Accepts a Mapper's output and merges it into a single combined dictionary.
        Used for local aggregation before the final reduction.
        """
        combined_data = defaultdict(int)
        for passenger_id, count in data.items():
            combined_data[passenger_id] += count
        return combined_data

# ======================== Reducer Class ========================
class Reducer:
    """Takes the combined results from all threads and finds the passenger(s) with the maximum number of flights. This is the final stage in the MapReduce pipeline."""

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
    Orchestrates the entire MapReduce-like pipeline.
    Splits the dataset, runs Mappers in parallel using threading.
    aggregates intermediate results via the combiner, and applies the Reducer.
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
            
# wait for all threads to finish and gather their results
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
