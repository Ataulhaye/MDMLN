import openpyxl
from datetime import datetime


def create_and_write_CSV(
    data,
    sheet_name,
    title="results",
):
    for key, value in data.items():
        print(key)
        print(value)
        for i in value:
            print(i)


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
