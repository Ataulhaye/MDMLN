import plotly.express as px
import plotly.graph_objs as go

# x = [i for i in range(7238)]


def plot_data(X, y, labels):
    fig = go.Figure()
    i = 0
    for row in X:
        py_list = list(row)
        fig.add_trace(go.Scatter(x=y, y=py_list, name=f"{labels[i]}"))
        i += 1
    # fig.update_traces(marker=dict(color="red"))
    fig.show()


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
