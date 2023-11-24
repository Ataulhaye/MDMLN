import csv
from datetime import datetime
from string import ascii_uppercase

import numpy as np
import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

from ExelSettings import ExelSettings
from ExportEntity import ExportEntity


class ExportData:
    def create_and_write_CSV(self, data, sheet_name, title="results", delimeter=","):
        csv_data = self.prepare_data_matrix(data)
        file_name = self.get_file_name(extension=".csv", sheet_name=sheet_name)
        with open(file_name, "w", newline="") as file:
            for ed in csv_data:
                wr = csv.writer(file, delimiter=delimeter, quoting=csv.QUOTE_ALL)
                wr.writerow(ed)

    def create_and_write_datasheet(
        self, data, sheet_name, title="results", transpose=False
    ):
        matrix = self.prepare_data_matrix(data)

        if transpose is True:
            matrix = np.transpose(matrix)

        # to create a new blank Workbook object
        wb = openpyxl.Workbook()

        ws = wb.active

        ws.title = title

        sett = ExelSettings()

        col_widths = self.calculate_column_width(matrix)

        cell_position = self.calculate_cell_positions(matrix)

        for i, row in enumerate(matrix[0]):
            ws.column_dimensions[cell_position[0][i][0]].width = col_widths[i]

        for row in matrix:
            if type(row) != list:
                row = list(row)
            ws.append(row)

        self.set_font_significant_result(cell_position, matrix, ws)

        self.set_header_font(cell_position, matrix, ws, col_widths, sett)

        self.set_first_column_font(cell_position, matrix, ws, col_widths, sett)

        sheet_name = self.get_file_name(extension=".xlsx", sheet_name=sheet_name)

        # Save File
        wb.save(sheet_name)

    def set_first_column_font(
        self, cell_position, matrix, ws, col_widths, setting: ExelSettings
    ):
        for j, row in enumerate(matrix):
            cell = ws[cell_position[j][0]]
            cell.font = Font(name=setting.header_family, sz=setting.header_font)

        self.repair_first_column_width(cell_position, ws, col_widths, setting)

    def repair_first_column_width(
        self, cell_position, ws, col_widths, setting: ExelSettings
    ):
        ws.column_dimensions[cell_position[0][0][0]].width = col_widths[0] + (
            (setting.header_font - setting.default_font) * 2
        )

    def set_header_font(
        self, cell_position, matrix, ws, col_widths, setting: ExelSettings
    ):
        for i, cell in enumerate(matrix[0]):
            cell = ws[cell_position[0][i]]
            cell.font = Font(name=setting.header_family, sz=setting.header_font)
            self.repair_header_cell_width(cell_position, ws, col_widths, i, setting)

    def repair_header_cell_width(
        self, cell_position, ws, col_widths, i, setting: ExelSettings
    ):
        ws.column_dimensions[cell_position[0][i][0]].width = col_widths[i] + (
            (setting.header_font - setting.default_font)
        )

    def set_font_significant_result(self, cell_position, matrix, ws):
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if "Not" not in val and j != 0 and i != 0:
                    cell = ws[cell_position[i][j]]
                    cell.font = Font(bold=True)

    def calculate_cell_positions(self, matrix):
        cell_names = []
        for i, row in enumerate(matrix):
            row_names = []
            for j, val in enumerate(row):
                row_names.append(f"{ascii_uppercase[j]}{i+1}")
            cell_names.append(row_names)
        return cell_names

    def calculate_column_width(self, matrix):
        max_column_widths = [0] * len(matrix[0])
        for row in matrix:
            for i, cell in enumerate(row):
                cell_length = len(cell)
                if max_column_widths[i] < cell_length:
                    max_column_widths[i] = cell_length
        return max_column_widths

    def get_file_name(self, extension, sheet_name):
        dt = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        sheet_name = f"{sheet_name}_{dt}{extension}"
        return sheet_name

    # for key, value in data.items():
    #    print(key)
    #   print(value)
    #  for i in value:
    #     print(i)
    def prepare_data_matrix(self, data: list[ExportEntity]):
        matrix = [["Classifier"]]
        for export_entity in data:
            col = f"{export_entity.sub_column_name}-{export_entity.column_name}"
            if col not in matrix[0]:
                matrix[0].append(col)
            row_col_index = self.get_index(matrix, export_entity.row_name)

            col_index = matrix[0].index(col)

            if row_col_index is not None:
                row_index = row_col_index[0]
                row = matrix[row_index]
                size = len(row) - 1
                if size < col_index:
                    for i in range(col_index - size):
                        row.append(None)

                # if "Not" in export_entity.result[0]:
                # row[col_index] = f"{export_entity.result[0]} {export_entity.result[1]}   p_value: {export_entity.p_value}"
                # else:
                row[col_index] = f"{export_entity.result[0]} {export_entity.result[1]}"

                matrix[row_index] = row

            else:
                insert_row = [None] * (col_index + 1)
                insert_row[0] = export_entity.row_name
                # if "Not" in export_entity.result[0]:
                # insert_row[col_index] = f"{export_entity.result[0]} {export_entity.result[1]}   p_value: {export_entity.p_value}"
                # else:
                insert_row[
                    col_index
                ] = f"{export_entity.result[0]} {export_entity.result[1]}"

                matrix.append(insert_row)
            # print(export_entity)

        return matrix

    # returns row, col index of a given value
    def get_index(self, matrix, v):
        for i, x in enumerate(matrix):
            if v in x:
                return (i, x.index(v))

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
