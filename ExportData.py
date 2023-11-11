import openpyxl
from datetime import datetime


def create_and_write_datasheet(
    data,
    sheet_name,
    title="results",
):
    # to create a new blank Workbook object
    wb = openpyxl.Workbook()

    # Get workbook active sheet
    sheet = wb.active

    sheet.title = title

    # Adding Data to Sheet
    for item in data:
        sheet.append(item)

    dt = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    sheet_name = f"{sheet_name}_{dt}.xlsx"

    # Save File
    wb.save(sheet_name)


create_and_write_datasheet(
    [
        ("ID", "Name", "Email"),
        (1, "abc", "agmail.com"),
        (2, "def", "dgmail.com"),
        (3, "ghi", "ggmail.com"),
    ],
    "tstdata",
)
