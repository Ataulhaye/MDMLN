import csv
from datetime import datetime

import openpyxl

from ExportEntity import ExportEntity


class ExportData:
    def create_and_write_CSV(data, sheet_name, title="results", delimeter=","):
        csv_data = ExportData.prepare_csv_data(data)
        dt = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        with open(f"{sheet_name}_{dt}.csv", "w", newline="") as file:
            for ed in csv_data:
                wr = csv.writer(file, delimiter=delimeter, quoting=csv.QUOTE_ALL)
                wr.writerow(ed)

    # for key, value in data.items():
    #    print(key)
    #   print(value)
    #  for i in value:
    #     print(i)
    def prepare_csv_data(data: list[ExportEntity]):
        csv_data = [["Classifier"]]
        for export_entity in data:
            col = f"{export_entity.sub_column_name}-{export_entity.column_name}"
            if col not in csv_data[0]:
                csv_data[0].append(col)
            row_col_index = ExportData.get_index(csv_data, export_entity.row_name)

            col_index = csv_data[0].index(col)

            if row_col_index is not None:
                row_index = row_col_index[0]
                row = csv_data[row_index]
                size = len(row) - 1
                if size < col_index:
                    for i in range(col_index - size):
                        row.append(None)

                if "Not" in export_entity.result[0]:
                    row[
                        col_index
                    ] = f"""{export_entity.result[0]} {export_entity.result[1]}
                    "p_value:"{export_entity.p_value}"""
                else:
                    row[
                        col_index
                    ] = f"{export_entity.result[0]} {export_entity.result[1]}"

                csv_data[row_index] = row

            else:
                insert_row = [None] * (col_index + 1)
                insert_row[0] = export_entity.row_name
                if "Not" in export_entity.result[0]:
                    insert_row[
                        col_index
                    ] = f"""{export_entity.result[0]} {export_entity.result[1]}
                    "p_value:"{export_entity.p_value}"""
                else:
                    insert_row[
                        col_index
                    ] = f"{export_entity.result[0]} {export_entity.result[1]}"

                csv_data.append(insert_row)
            print(export_entity)

        return csv_data

    # returns row, col index of a given value
    def get_index(matrix, v):
        for i, x in enumerate(matrix):
            if v in x:
                return (i, x.index(v))

    def create_and_write_datasheet(
        data,
        sheet_name,
        title="results",
    ):
        # to create a new blank Workbook object
        wb = openpyxl.Workbook()

        ws = wb.active

        ws.title = title

        # ws.insert_rows(1)

        for item in data:
            ws.append(item)

        dt = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        sheet_name = f"{sheet_name}_{dt}.xlsx"

        # Save File
        wb.save(sheet_name)

        """create_and_write_datasheet(
            [
                ("ID", "Name", "Email"),
                (1, "abc", "agmail.com"),
                (2, "def", "dgmail.com"),
                (3, "ghi", "ggmail.com"),
            ],
            "tstdata",
        )
        """
