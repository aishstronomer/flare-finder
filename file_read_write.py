import pandas as pd
from datetime import datetime
import glob
import os


class FileReadWrite:
    @staticmethod
    def get_df_from_goes_file(filepath):
        separator = ","
        tmp = []
        with open(filepath, "r") as infile:
            data = infile.readlines()
            for i in range(len(data)):
                if i >= 13:
                    #             print(i)
                    i_d = data[i]
                    #             print(i_d)
                    list_row = i_d.split()
                    #             print(list_row)
                    if list_row == []:
                        #                 print('it is an empty_list')
                        continue
                    #             print(i)
                    #             print(list_row)
                    if "+" in list_row:
                        #                 print('yes there is a +')
                        list_row = [i for i in list_row if i != "+"]
                    #                 print(list_row)
                    if len(list_row) == 9:
                        mylist = list_row
                    elif len(list_row) == 10:
                        mylist = list_row[:-1]
                    #                 print(mylist)
                    elif len(list_row) == 11:
                        mylist = list_row[:-2]
                    else:
                        continue
                    #             print(mylist)
                    #             i_d = separator.join(list_row)
                    #             print(i_d)
                    tmp.append(mylist)

        df = pd.DataFrame(
            tmp,
            columns=[
                "Event",
                "Begin",
                "Max",
                "End",
                "Obs",
                "Q",
                "Type",
                "Loc/Freq",
                "Particulars_a",
            ],
        )
        return df

    @staticmethod
    def get_df_from_from_date_range(path, start_date, end_date):
        # Initializing an empty array to store the Dataframes
        dfs = []

        file_extension = ".txt"

        # Convert start and end dates to datetime objects
        start_datetime = datetime.strptime(start_date, "%Y%m%d")
        end_datetime = datetime.strptime(end_date, "%Y%m%d")

        # Create a glob pattern to match text files within the specified directory
        pattern = os.path.join(path, f"*{file_extension}")

        # Use glob to get a list of matching file paths
        all_files = glob.glob(pattern)

        # Initialize an empty list to store file paths that meet the date condition
        filtered_file_paths = []

        # Iterate through the files and filter based on the date condition
        for file_path in all_files:
            # Extract the date from the file name
            file_date = os.path.basename(file_path)[:8]

            # Convert the date string to a datetime object
            file_datetime = datetime.strptime(file_date, "%Y%m%d")

            # Check if the file date falls within the specified date range
            if start_datetime <= file_datetime <= end_datetime:
                # If the condition is satisfied, add the file path to the list
                filtered_file_paths.append(file_path)

        # Iterate through the filtered file paths, read each file into a DataFrame, and add the 'Date' column
        for file_path in filtered_file_paths:
            # Extract the date from the file name
            file_date = os.path.basename(file_path)[:8]

            # Read the file into a DataFrame
            df = FileReadWrite.get_df_from_goes_file(
                file_path
            )  # Adjust file reading as needed

            # Add the "Date" column with the file date
            df["Date"] = file_date

            # Append the DataFrame to the list
            dfs.append(df)

        # Concatenate the list of DataFrames into a single DataFrame
        result_df = pd.concat(dfs, ignore_index=True)

        # Sort the DataFrame by the "Date" column in ascending order
        result_df = result_df.sort_values(by=["Date", "Event"])

        # Reset the index of the sorted DataFrame
        result_df = result_df.reset_index(drop=True)

        return result_df
