import pickle
from pathlib import Path

import kaleido
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from Brain import Brain
from ExportData import ExportData
from Helper import Helper

# x = [i for i in range(7238)]


class VisualizeData:
    def plot_data(X, y, labels):
        fig = go.Figure()
        i = 0
        for row in X:
            py_list = list(row)
            fig.add_trace(go.Scatter(x=y, y=py_list, name=f"{labels[i]}"))
            i += 1
        # fig.update_traces(marker=dict(color="red"))
        fig.show()

    def plot_bar_graph(
        self,
        X,
        Y,
        title,
        hover_color="green",
        line_width=2,
        bar_color="green",
        opacity=1,
    ):
        x_label, x = X
        y_label, y = Y
        data_dic = {f"{x_label}": x, f"{y_label}": y}
        df = pd.DataFrame(data_dic)

        # fig = px.bar(df, x=f"{x_label}", y=f"{y_label}", color=f"{y_label}")
        fig = px.bar(df, x=f"{x_label}", y=f"{y_label}", title=title)
        fig.update_traces(
            marker_color=bar_color,
            marker_line_color=bar_color,
            marker_line_width=line_width,
            opacity=opacity,
        )
        # fig.update_layout(
        #   font_family="Courier New",
        #  font_color="blue",
        # title_font_family="Times New Roman",
        # title_font_color="red",
        # legend_title_font_color="green",
        # )
        fig.update_layout(
            xaxis_title=dict(text=x_label, font=dict(size=22, color="black")),
            yaxis_title=dict(text=y_label, font=dict(size=22, color="black")),
            title=dict(
                text=title,
                font=dict(family="Courier New, monospace", size=30, color="blue"),
                automargin=True,
                yref="paper",
            ),
        )

        html_name = ExportData.get_file_name(".html", title)
        directory_path = Helper.ensure_dir("Data_Graphs", "")
        html_path = Path(directory_path).joinpath(html_name)
        # fig.update_xaxes(title_font_family="Arial")
        fig.write_html(html_path)
        fig.show()

        # fig = px.bar(X, y=y)
        # fig.show()

    def visualize_nans(self, brain: Brain):
        ln = brain.lobe.name
        if "ALL" in ln:
            ln = "All lobes"

        nans_column_wise = brain.calculate_nans_voxel_wise(brain.voxels)
        print("---------------------------------------------------------")
        print(
            f"Indexes of {brain.lobe.name} where whole column is NAN: ",
            nans_column_wise.count(488),
        )
        total_nans = sum(nans_column_wise)
        print("Total NANs: ", total_nans)
        print(
            "Total NANs in all data: {:0.2f}%".format(
                ((total_nans / (brain.voxels.shape[0] * brain.voxels.shape[1])) * 100)
            )
        )
        print("-------------------------------------------------------------")
        columns = [i for i in range(brain.voxels.shape[1])]
        name = f"Voxelwise NaN values in the {ln}"

        self.plot_bar_graph(
            ("Dimension", columns),
            ("Trials", nans_column_wise),
            title=name,
            bar_color="#FFBF00",
        )

        nans_voxel_wise = brain.calculate_nans_trail_wise(brain.voxels)
        rows = [i for i in range(brain.voxels.shape[0])]
        name = f"Trialwise NaN values in the {ln}"
        self.plot_bar_graph(
            ("Trials", rows),
            ("NaN values each trail", nans_voxel_wise),
            bar_color="#FFBF00",
            title=name,
        )


"""
def plot_with_nan():
    fig = go.Figure()
i = 0
for row in STG_raw:
    py_list = list(row)
    fig.add_trace(
        go.Scatter(x=x, y=py_list, name=f"{subject_labels[i]}{image_labels[i]}")
    )
    i += 1
fig.show()

"""
