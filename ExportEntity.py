class ExportEntity:
    def __init__(
        self,
        row_name,
        column_name,
        sub_column_name,
        p_value,
        result,
        standard_deviation,
        mean,
    ):
        self.row_name = row_name
        self.column_name = column_name
        self.sub_column_name = sub_column_name
        self.p_value = p_value
        self.result = result
        self.standard_deviation = standard_deviation
        self.mean = mean

    def __repr__(self) -> str:
        return f"Row:{self.row_name}, Column:{self.column_name}"
