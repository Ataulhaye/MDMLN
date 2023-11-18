class ExportEntity:
    def __init__(self, row_name, column_name, sub_column_name, p_value, result):
        self.row_name = row_name
        self.column_name = column_name
        self.sub_column_name = sub_column_name
        self.p_value = p_value
        self.result = result

    def __repr__(self) -> str:
        return f"Row:{self.row_name}, Column:{self.column_name}"
