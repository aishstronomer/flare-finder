import pandas as pd


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
