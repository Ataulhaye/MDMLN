import csv
from datetime import datetime
from string import ascii_uppercase

import numpy as np
import openpyxl
from openpyxl.styles import Font

from ExelSettings import ExelSettings
from ExportEntity import ExportEntity


class ExportData:
    def create_and_write_csv(self, data, sheet_name, title="results", delimiter=","):
        csv_data = self.prepare_data_matrix(data)
        file_name = self.get_file_name(extension=".csv", sheet_name=sheet_name)
        with open(file_name, "w", newline="") as file:
            for ed in csv_data:
                wr = csv.writer(file, delimiter=delimiter, quoting=csv.QUOTE_ALL)
                wr.writerow(ed)

    def create_and_write_datasheet(
        self, data, sheet_name, title="results", transpose=False
    ):
        # matrix = data
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

        self.set_font_significant_result(cell_position, matrix, ws, transpose)

        self.set_header_font(cell_position, matrix, ws, col_widths, sett)

        self.set_first_column_font(cell_position, matrix, ws, col_widths, sett)

        if transpose:
            x = 2
            for row in matrix:
                ws.merge_cells(
                    start_row=x, start_column=1, end_row=(x + 1), end_column=1
                )
                if x % 2 == 0:
                    x = x + 2
        else:
            x = 2
            for row in matrix:
                ws.merge_cells(
                    start_row=1, start_column=x, end_row=1, end_column=(x + 1)
                )
                if x % 2 == 0:
                    x = x + 2
            # step incrment x to 2 to ..
            # ws.merge_cells(start_row=x, start_column=1, end_row=x, end_column=1)

        if transpose:
            self.set_2nd_column_font(
                cell_position, matrix, ws, col_widths, sett, transpose
            )
        else:
            self.set_2nd_row_font(
                cell_position, matrix, ws, col_widths, sett, transpose
            )

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

    def set_2nd_column_font(
        self, cell_position, matrix, ws, col_widths, setting: ExelSettings, transpose
    ):
        for j, row in enumerate(matrix):
            cell = ws[cell_position[j][1]]
            cell.font = Font(name=setting.header_family, sz=setting.header_font)

        self.repair_column_width(cell_position, ws, col_widths, setting, 1)

    def set_2nd_row_font(
        self, cell_position, matrix, ws, col_widths, setting: ExelSettings, transpose
    ):
        for j, row in enumerate(matrix[1]):
            cell = ws[cell_position[1][j]]
            cell.font = Font(name=setting.header_family, sz=setting.header_font)

        self.repair_row_width(cell_position, ws, col_widths, matrix)

    @staticmethod
    def repair_first_column_width(cell_position, ws, col_widths, setting: ExelSettings):
        ws.column_dimensions[cell_position[0][0][0]].width = col_widths[0] + (
            (setting.header_font - setting.default_font) * 2
        )

    @staticmethod
    def repair_column_width(
        cell_position, ws, col_widths, setting: ExelSettings, column: int
    ):
        ws.column_dimensions[cell_position[0][column][0]].width = col_widths[column] + (
            (setting.header_font - setting.default_font) * 2
        )

    @staticmethod
    def repair_row_width(cell_position, ws, col_widths, matrix):
        for i, row in enumerate(matrix[0]):
            if i > 0:
                ws.column_dimensions[cell_position[0][i][0]].width = col_widths[i]

    def set_header_font(
        self, cell_position, matrix, ws, col_widths, setting: ExelSettings
    ):
        for i, cell in enumerate(matrix[0]):
            cell = ws[cell_position[0][i]]
            cell.font = Font(name=setting.header_family, sz=setting.header_font)
            self.repair_header_cell_width(cell_position, ws, col_widths, i, setting)

    @staticmethod
    def repair_header_cell_width(
        cell_position, ws, col_widths, i, setting: ExelSettings
    ):
        ws.column_dimensions[cell_position[0][i][0]].width = col_widths[i] + (
            (setting.header_font - setting.default_font)
        )

    @staticmethod
    def set_font_significant_result(cell_position, matrix, ws, transpose):
        k = 0
        if transpose:
            k = 1
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if "Not" not in val and j > k and i > 0:
                    cell = ws[cell_position[i][j]]
                    cell.font = Font(bold=True)

    @staticmethod
    def calculate_cell_positions(matrix):
        cell_names = []
        for i, row in enumerate(matrix):
            row_names = []
            for j, val in enumerate(row):
                row_names.append(f"{ascii_uppercase[j]}{i+1}")
            cell_names.append(row_names)
        return cell_names

    @staticmethod
    def calculate_column_width(matrix):
        max_column_widths = [0] * len(matrix[0])
        for row in matrix:
            for i, cell in enumerate(row):
                cell_length = len(cell)
                if max_column_widths[i] < cell_length:
                    max_column_widths[i] = cell_length
        return max_column_widths

    @staticmethod
    def get_file_name(extension, sheet_name):
        dt = datetime.now().strftime("%d-%m-%Y_%H-%M-%S_%f")
        sheet_name = f"{sheet_name}_{dt}{extension}"
        return sheet_name

    def prepare_data_matrix(self, data: list[ExportEntity]):
        matrix = [["Strategy"], ["Classifier"]]

        for i, export_entity in enumerate(data):
            sub_col = f"{export_entity.sub_column_name}#{export_entity.column_name}"
            col = f"{export_entity.column_name}#{export_entity.sub_column_name}"
            if col not in matrix[1]:
                matrix[1].append(col)
            if sub_col not in matrix[0]:
                matrix[0].append(sub_col)
            row_col_index = self.get_index(matrix, export_entity.row_name)

            col_index = matrix[1].index(col)

            if row_col_index is not None:
                row_index = row_col_index[0]
                row = matrix[row_index]
                size = len(row) - 1
                if size < col_index:
                    for i in range(col_index - size):
                        row.append(None)

                row[col_index] = f"{export_entity.result[0]} {export_entity.result[1]}"

                matrix[row_index] = row

            else:
                insert_row = [None] * (col_index + 1)
                insert_row[0] = export_entity.row_name
                insert_row[
                    col_index
                ] = f"{export_entity.result[0]} {export_entity.result[1]}"

                matrix.append(insert_row)

        for i in range(2):
            for j, cell in enumerate(matrix[i]):
                matrix[i][j] = cell.split("#")[0]

        return matrix

    @staticmethod
    def init_matrix(rows, columns):
        matrix = []
        for r in range(rows):
            matrix.append([None] * columns)
        return matrix

    @staticmethod
    def matrix_dimensions(data: list[ExportEntity]):
        row_data, col_data = [], []
        for export_entity in data:
            col_data.append(
                f"{export_entity.sub_column_name}-{export_entity.column_name}"
            )
            row_data.append(export_entity.row_name)
        rows = len(set(row_data)) + 2
        columns = len(set(col_data)) + 1
        return rows, columns

    # returns row, col index of a given value
    @staticmethod
    def get_index(matrix, value):
        for i, row in enumerate(matrix):
            if value in row:
                return i, row.index(value)


d = [
    [
        "Strategy",
        "mean",
        "mean",
        "median",
        "median",
        "most_frequent",
        "most_frequent",
        "remove-voxels",
        "remove-voxels",
    ],
    [
        "Classifier",
        "subject_labels",
        "image_labels",
        "subject_labels",
        "image_labels",
        "subject_labels",
        "image_labels",
        "subject_labels",
        "image_labels",
    ],
    [
        "SVC",
        "Not significant: 34.81%",
        "Significant: 31.35%",
        "Not significant: 40.96%",
        "Significant: 32.69%",
        "Not significant: 43.46%",
        "Significant: 32.12%",
        "Not significant: 37.31%",
        "Significant: 31.92%",
    ],
    [
        "KNeighborsClassifier",
        "Significant: 37.31%",
        "Significant: 26.35%",
        "Not significant: 34.62%",
        "Not significant: 27.31%",
        "Not significant: 36.35%",
        "Significant: 28.65%",
        "Not significant: 39.42%",
        "Significant: 26.92%",
    ],
    [
        "LinearDiscriminantAnalysis",
        "Not significant: 35.58%",
        "Significant: 34.04%",
        "Significant: 40.38%",
        "Significant: 30.19%",
        "Significant: 56.35%",
        "Significant: 34.62%",
        "Not significant: 36.92%",
        "Significant: 32.50%",
    ],
]

ed = ExportData()
ed.create_and_write_datasheet(
    data=d, sheet_name="UnitTest", title="Test", transpose=True
)
ed.create_and_write_datasheet(
    data=d, sheet_name="UnitTest", title="Test", transpose=False
)
