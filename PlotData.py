import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

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
        X, Y, title, hover_color="green", line_width=2, bar_color="green", opacity=1
    ):
        x_label, x = X
        y_label, y = Y
        data_dic = {f"{x_label}": x, f"{y_label}": y}
        df = pd.DataFrame(data_dic)

        # fig = px.bar(df, x="group", y="value", color="value")
        fig = px.bar(df, x=f"{x_label}", y=f"{y_label}", title=title)
        fig.update_traces(
            marker_color=hover_color,
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
            title=dict(
                text=title,
                font=dict(family="Courier New, monospace", size=30, color="blue"),
                automargin=True,
                yref="paper",
            )
        )
        # fig.update_xaxes(title_font_family="Arial")
        fig.show()

        # fig = px.bar(X, y=y)
        # fig.show()


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
