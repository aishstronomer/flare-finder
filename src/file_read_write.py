from datetime import datetime
import glob
import os
import pandas as pd
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class FileReadWrite:
    @staticmethod
    def get_df_from_goes_file(filepath):
        # TODO: let's doublecheck that we're not throwing out any rows when we're
        #   reading the files since the initial discard rowcount is hardcoded to 13

        separator = ","
        tmp = []
        with open(filepath, "r") as infile:
            data = infile.readlines()
            for i in range(len(data)):
                if i >= 13:
                    i_d = data[i]
                    list_row = i_d.split()
                    if list_row == []:
                        continue
                    if "+" in list_row:
                        list_row = [i for i in list_row if i != "+"]
                    if len(list_row) == 9:
                        mylist = list_row
                    elif len(list_row) == 10:
                        mylist = list_row[:-1]
                    elif len(list_row) == 11:
                        mylist = list_row[:-2]
                    else:
                        continue
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
    def get_goes_events_df_for_interval(path, start_date, end_date):
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

        # Add columns for start_time, end_time, max_time and drop (Date, Begin, End, Max)
        result_df["Begin"] = result_df["Begin"].str.replace("[a-zA-Z]", "")
        result_df["Max"] = result_df["Max"].str.replace("[a-zA-Z]", "")
        result_df["End"] = result_df["End"].str.replace("[a-zA-Z]", "")
        result_df["begin_datetime"] = pd.to_datetime(
            result_df["Date"] + " " + result_df["Begin"],
            format="%Y%m%d %H%M",
            errors="coerce",
        )
        result_df["max_datetime"] = pd.to_datetime(
            result_df["Date"] + " " + result_df["Max"],
            format="%Y%m%d %H%M",
            errors="coerce",
        )
        result_df["end_datetime"] = pd.to_datetime(
            result_df["Date"] + " " + result_df["End"],
            format="%Y%m%d %H%M",
            errors="coerce",
        )
        result_df = result_df.drop(columns=["Begin", "Max", "End", "Date"])
        result_df.loc[
            result_df["max_datetime"] < result_df["begin_datetime"], "max_datetime"
        ] = result_df["max_datetime"] + pd.DateOffset(days=1)
        result_df.loc[
            result_df["end_datetime"] < result_df["begin_datetime"], "end_datetime"
        ] = result_df["end_datetime"] + pd.DateOffset(days=1)

        # Sort the DataFrame by the "Date" column in ascending order
        result_df = result_df.sort_values(by=["begin_datetime", "Event"])

        # Reset the index of the sorted DataFrame
        result_df = result_df.reset_index(drop=True)

        return result_df
