# data_loader.py
import pandas as pd
import time


class CSVDataLoader:
    def __init__(self, preprocessor=None, debug=False):
        """
        DataLoader initialization.
        :param preprocessor: An instance of Preprocessor class or similar, with a preprocess method. If None, preprocessing is skipped.
        """
        self.preprocessor = preprocessor

    def load_data(self, file_path, nrows=None):
        """
        Loads data from a CSV file.
        :param file_path: Path to the CSV file.
        :param nrows: Number of rows to read from the CSV file. If None, all rows are read.
        :return: A pandas DataFrame with the processed Articles.
        """
        # Load the CSV file into a DataFrame, limited to 'nrows' if provided
        data = pd.read_csv(file_path, nrows=nrows)
        print("Columns in loaded data:", data.columns)

        # Apply preprocessing if a preprocessor is provided
        if self.preprocessor is not None:
            data = self.preprocessor.preprocess(data)

        return data


# Testing the DataLoader class and measuring load time
if __name__ == "__main__":
    loader = CSVDataLoader()
    num_rows_to_load = 100  # Example number of rows to load

    start_time = time.time()
    data = loader.load_data("data/data.csv", nrows=num_rows_to_load)
    end_time = time.time()

    print(data.head())  # Display the first few rows of the DataFrame
    print(f"Time taken to load data: {end_time - start_time} seconds")
