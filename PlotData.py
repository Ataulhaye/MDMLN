import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from nilearn import plotting

from Brain import Brain
from EvaluateTrainingModel import EvaluateTrainingModel
from ExportData import ExportData
from Helper import Helper


class Visualization:

    def plot_merged_bars(
        self,
        directory,
        all_data,
        lobe_name,
        N="N",
        D="D",
        S="S",
        legend_title="Mental Disorders",
        legend_text=[
            "Neurotypicals",
            "Depressives",
            "Schizophrenics",
        ],
        opt_info=None,
        legend_font=18,
    ):
        nested_dict = self.groupby_strategy(all_data)

        for strategy, clasiifiers in nested_dict.items():
            bar_dict = self.separate_results_by_patients(N, D, S, clasiifiers)

            models, bar_dictc = self.merge_results(N, D, S, strategy, bar_dict)

            self.plot_diagram(
                strategy=strategy,
                models=models,
                bar_dictc=bar_dictc,
                directory=directory,
                legend_title=legend_title,
                legend_text=legend_text,
                lobe_name=lobe_name,
                legend_font=legend_font,
                opt_info=opt_info,
            )

    plt.close("all")

    def plot_detailed_bars(
        self,
        directory,
        all_data,
        lobe_name,
        N="N",
        D="D",
        S="S",
        legend_text=["Neurotypicals", "Depressives", "Schizophrenics"],
        opt_info=None,
        legend_font=17,
    ):
        """
        This method plot the detailed graphs, with binary 6 combinations, std and significant
        """
        nested_dict = self.groupby_strategy(all_data)

        for strategy, clasiifiers in nested_dict.items():
            bar_dict = self.separate_results_by_patients(N, D, S, clasiifiers)

            models = []
            for label, y in bar_dict.items():
                for j, v in y.items():
                    if j not in models:
                        models.append(j)
                    for i in v:
                        i.column_name = i.column_name.split("_")[-1]

            data_stat = {
                f"{N}_AR-AU": {"data": [], "std": [], "result": []},
                f"{N}_AR-CR": {"data": [], "std": [], "result": []},
                f"{N}_AR-CU": {"data": [], "std": [], "result": []},
                f"{N}_AU-CR": {"data": [], "std": [], "result": []},
                f"{N}_AU-CU": {"data": [], "std": [], "result": []},
                f"{N}_CR-CU": {"data": [], "std": [], "result": []},
                f"{D}_AR-AU": {"data": [], "std": [], "result": []},
                f"{D}_AR-CR": {"data": [], "std": [], "result": []},
                f"{D}_AR-CU": {"data": [], "std": [], "result": []},
                f"{D}_AU-CR": {"data": [], "std": [], "result": []},
                f"{D}_AU-CU": {"data": [], "std": [], "result": []},
                f"{D}_CR-CU": {"data": [], "std": [], "result": []},
                f"{S}_AR-AU": {"data": [], "std": [], "result": []},
                f"{S}_AR-CR": {"data": [], "std": [], "result": []},
                f"{S}_AR-CU": {"data": [], "std": [], "result": []},
                f"{S}_AU-CR": {"data": [], "std": [], "result": []},
                f"{S}_AU-CU": {"data": [], "std": [], "result": []},
                f"{S}_CR-CU": {"data": [], "std": [], "result": []},
            }

            for patient, resu in bar_dict.items():
                for classi, res in resu.items():
                    for it in res:
                        k = f"{patient}_{it.column_name}"
                        data_stat[k]["std"].append(it.standard_deviation)
                        data_stat[k]["data"].append(it.mean)
                        data_stat[k]["result"].append(it.result[0])

            self.plot_diagram_per_strategy(
                strategy=strategy,
                models=models,
                bar_data=data_stat,
                directory=directory,
                legends=legend_text,
                lobe_name=lobe_name,
                legend_font=legend_font,
                opt_info=opt_info,
            )

    plt.close("all")

    def plot_detailed_bars_dynamic(
        self,
        directory,
        all_data,
        lobe_name,
        N="N",
        D="D",
        S="S",
        legend_text=["Neurotypicals", "Depressives", "Schizophrenics"],
        opt_info=None,
    ):
        """
        This method plot the detailed graphs, with binary 6 combinations, std and significant
        """
        nested_dict = self.groupby_strategy(all_data)

        for strategy, clasiifiers in nested_dict.items():
            bar_dict = self.separate_results_by_patients(N, D, S, clasiifiers)

            models = []
            data_stat = {}
            for label, y in bar_dict.items():
                for j, v in y.items():
                    if j not in models:
                        models.append(j)
                    for i in v:
                        cut = 2
                        if "ary" in i.column_name:
                            cut = 3
                        splits = i.column_name.split("_")
                        i.column_name = "_".join(splits[cut:])
                        data_stat[i.column_name] = {"data": [], "std": [], "result": []}

            for patient, resu in bar_dict.items():
                for classi, res in resu.items():
                    for it in res:
                        data_stat[it.column_name]["std"].append(it.standard_deviation)
                        data_stat[it.column_name]["data"].append(it.mean)
                        data_stat[it.column_name]["result"].append(it.result[0])

            self.plot_diagram_per_strategy(
                strategy=strategy,
                models=models,
                bar_data=data_stat,
                directory=directory,
                legends=legend_text,
                lobe_name=lobe_name,
                legend_font=18,
                opt_info=opt_info,
                separator="-",
            )

    plt.close("all")

    def plot_detailed_bars_data_labels(
        self, directory, lobe_name, all_data, opt_info=None
    ):
        """
        This method plot the detailed graphs, with Image labels and subject labels, std and significant
        """
        nested_dict = self.groupby_strategy(all_data)

        subject_l = "Subject"
        image_l = "Speech-Gesture"

        for strategy, clasiifiers in nested_dict.items():
            bar_dict = self.separate_results_by_labels(subject_l, image_l, clasiifiers)

            models = []
            for label, y in bar_dict.items():
                for j, v in y.items():
                    if j not in models:
                        models.append(j)

            data_stat = {
                f"{subject_l}": {"data": [], "std": [], "result": []},
                f"{image_l }": {"data": [], "std": [], "result": []},
            }

            for label, resu in bar_dict.items():
                for classi, res in resu.items():
                    for it in res:
                        data_stat[label]["std"].append(it.standard_deviation)
                        data_stat[label]["data"].append(it.mean)
                        data_stat[label]["result"].append(it.result[0])

            self.plot_diagram_per_strategy(
                strategy=strategy,
                models=models,
                bar_data=data_stat,
                directory=directory,
                legends=[
                    f"Subjects {chr(10)} (N, D, S)",
                    f"Speech-Gesture {chr(10)} Combinations {chr(10)}(AR, AU, CR, CU)",
                ],
                lobe_name=lobe_name,
                legend_font=17,
                legend_title="Conditions",
                opt_info=opt_info,
            )

    @staticmethod
    def plot_brain_image(smoothed_img, title_txt, directory, pdf, png, show=False):
        # rdm_typ = f"{self.rsa_config.related_unrelated_RDM=}".split("=")[0].split(".")[2]
        fig = plt.figure(figsize=(18, 7))
        # display, axes = plotting.plot_img_on_surf( smoothed_img,surf_mesh="fsaverage", views=["lateral", "medial"],hemispheres=["left", "right"],inflate=False,colorbar=True,bg_on_data=True,cmap="hsv_r")
        display = plotting.plot_glass_brain(
            smoothed_img,
            threshold=0,
            # title=title,
            display_mode="lzry",
            colorbar=True,
            figure=fig,
        )
        display.title(
            text=title_txt,
            # x=0.01,
            # y=0.99,
            x=0.02,
            y=0.02,
            size=20,
            # color="green",
            bgcolor=None,
            alpha=1,
            va="bottom",
        )
        # display = plotting.plot_stat_map(smoothed_img, threshold=0)
        # display.savefig("pretty_brain.png")
        # plotting.plot_glass_brain(smoothed_img, threshold=0)
        time.sleep(1)

        # pdf_name, png_name = ExportData.create_figure_names(title_txt.replace(" ", "_"))

        directory_path = Helper.ensure_dir("Searchlight_Graphs", directory)
        png_path = Path(directory_path).joinpath(png)
        pdf_path = Path(directory_path).joinpath(pdf)
        plt.savefig(pdf_path)
        time.sleep(1)
        plt.savefig(png_path, dpi=600)

        if show:
            plotting.show()
            display.close()
            plt.close()

    @staticmethod
    def plot_diagram_per_strategy(
        strategy,
        models,
        bar_data,
        directory,
        legends,
        lobe_name,
        legend_title="Mental Disorders",
        legend_font=18,
        opt_info=None,
        separator="_",
    ):
        bar_types_per_model = len(legends)
        barWidth = 0.5
        i = 0
        br_pre_pos = None
        all_br_positions = []
        color_len = int(len(bar_data) / bar_types_per_model)
        colors = ["tomato" for x in range(color_len)]
        colors.extend(["limegreen" for x in range(color_len)])
        colors.extend(["dodgerblue" for x in range(color_len)])

        bar_labels = [
            j.split(separator)[-1]
            for i in range(len(models))
            for j in list(bar_data.keys())
        ]

        br_position = None
        legend_bars = []
        plt.subplots(figsize=(30, 10))
        plt.rcParams.update({"legend.title_fontsize": 18})
        for key, br_data in bar_data.items():
            if i > 0:
                br_position = [x + barWidth for x in br_pre_pos]
                br_pre_pos = br_position
            else:
                nu = [0]
                # br_pre_pos = [0, int(len(bar_data) * barWidth) + 1]
                # br_position = [0, int(len(bar_data) * barWidth) + 1]
                k = 0
                for _ in range(len(br_data["data"]) - 1):
                    k = int(len(bar_data) * barWidth) + k + 1
                    nu.append(k)
                br_pre_pos = nu
                br_position = nu

            all_br_positions.extend(br_position)
            a = plt.bar(
                br_position,
                br_data["data"],
                color=colors[i],
                width=barWidth,
                edgecolor="grey",
                label=key,
            )
            for index, d in enumerate(br_data["data"]):
                txt_pos = 0.5 * d
                if "Speech-Gesture" in bar_labels[i]:
                    txt_pos = 0.6 * d
                plt.text(
                    br_position[index],
                    txt_pos,
                    bar_labels[i],
                    ha="center",
                    va="top",
                    color="white",
                    rotation="vertical",
                    fontsize=17,
                )
            for index, res in enumerate(br_data["result"]):
                if "Not" not in res:
                    plt.text(
                        br_position[index],
                        0,
                        "*",
                        ha="center",
                        va="baseline",
                        color="k",
                        fontsize=25,
                    )

            plt.errorbar(
                br_position, br_data["data"], yerr=br_data["std"], fmt="o", color="k"
            )
            if i % (int(len(bar_data) / bar_types_per_model)) == 0:
                legend_bars.append(a)
            i = i + 1

        lobe_n = lobe_name
        if "All" in lobe_n:
            lobe_n = "Whole Brain"
        else:
            lobe_n = f"{lobe_n} Lobe"

        sta_name = strategy

        if "mice" in sta_name:
            sta_name = "MICE Imputation"
        elif "mean" in sta_name:
            sta_name = "Mean Imputation"
        elif "remove" in sta_name:
            sta_name = "Voxel Deletion"

        title = f"{lobe_n} Analysis with {sta_name}"

        if opt_info is not None:
            title = f"{title} using {opt_info}"

        plt.xlabel(title, fontweight="bold", fontsize=24)
        plt.ylabel("Accuracy (in %)", fontweight="bold", fontsize=20)

        all_br_positions.sort()
        tick_pos = []
        bar_types_per_model = int((len(all_br_positions) / len(models)))
        end = 0
        start = 0
        while end < len(all_br_positions):
            end = end + bar_types_per_model
            seg = all_br_positions[start:end]
            tick_pos.append(seg[int(len(seg) / 2)] - barWidth / 2)
            start = end

        plt.xticks(tick_pos, models, fontsize=22)
        plt.yticks(fontsize=20)

        plt.legend(
            legend_bars,
            legends,
            fontsize=legend_font,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            title=legend_title,
        )
        # plt.legend(legend_bars, ["N", "D", "S"], fontsize=18, loc='upper left', bbox_to_anchor=(1, 1) ,title_fontsize=14,title="Mental Disorders")
        gname = f"{lobe_name}_{strategy}_{directory}"

        if opt_info is not None:
            gname = f"{opt_info}_{gname}"

        pdf_name, png_name = ExportData.create_figure_names(gname)

        directory_path = Helper.ensure_dir("Ml_Graphs", directory)
        png_path = Path(directory_path).joinpath(png_name)
        pdf_path = Path(directory_path).joinpath(pdf_name)
        plt.savefig(pdf_path)
        time.sleep(1)
        plt.savefig(png_path, dpi=600)

        # plt.show()
        plt.close()

    @staticmethod
    def plot_diagram(
        strategy,
        models,
        bar_dictc,
        directory,
        legend_title,
        legend_text,
        lobe_name,
        legend_font,
        opt_info=None,
    ):
        barWidth = 0.25
        i = 0
        br_pre = None
        colors = ["tomato", "limegreen", "dodgerblue"]
        br_p = None
        plt.subplots(figsize=(30, 10))
        plt.rcParams.update({"legend.title_fontsize": 18})
        legend_bars = []
        for key, br in bar_dictc.items():
            if i > 0:
                br_p = [x + barWidth for x in br_pre]
                br_pre = br_p
            else:
                br_pre = np.arange(len(br))
                br_p = np.arange(len(br))

            for j, bar_val in enumerate(br):
                a = plt.bar(
                    br_p[j],
                    br[j].mean,
                    color=colors[i],
                    width=barWidth,
                    edgecolor="grey",
                    label=key,
                )
                if "Not" not in br[j].result[0]:
                    plt.text(
                        br_p[j],
                        0,
                        "*",
                        ha="center",
                        va="baseline",
                        color="k",
                        fontsize=25,
                    )

                plt.errorbar(
                    br_p[j],
                    br[j].mean,
                    yerr=br[j].standard_deviation,
                    fmt="o",
                    color="k",
                )
                if j == 0:
                    legend_bars.append(a)

            i = i + 1

        lobe_n = lobe_name
        if "All" in lobe_n:
            lobe_n = "Whole Brain"
        else:
            lobe_n = f"{lobe_n} Lobe"

        sta_name = strategy
        if "mice" in sta_name:
            sta_name = "MICE Imputation"
        elif "mean" in sta_name:
            sta_name = "Mean Imputation"
        elif "remove" in sta_name:
            sta_name = "Voxel Deletion"

        name = f"{lobe_n} Analysis with {sta_name}"

        if opt_info is not None:
            name = f"{name} using {opt_info}"

        plt.xlabel(name, fontweight="bold", fontsize=24)
        plt.ylabel("Accuracy (in %)", fontweight="bold", fontsize=20)
        plt.yticks(fontsize=20)
        plt.xticks([r + barWidth for r in range(len(br_p))], models, fontsize=22)
        # l = plt.legend(fontsize=18,loc="upper left",bbox_to_anchor=(1, 1),title=legend_title,)
        plt.legend(
            legend_bars,
            legend_text,
            fontsize=legend_font,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            title=legend_title,
        )

        # for lidx, text in enumerate(legend_text):
        # l.get_texts()[lidx].set_text(text)

        gname = f"{lobe_name}_{strategy}_{directory}"

        if opt_info is not None:
            gname = f"{opt_info}_{gname}"

        gname = f"Merged_{gname}"

        pdf_name, png_name = ExportData.create_figure_names(gname)

        directory_path = Helper.ensure_dir("Ml_Graphs", directory)
        png_path = Path(directory_path).joinpath(png_name)
        pdf_path = Path(directory_path).joinpath(pdf_name)
        plt.savefig(pdf_path)
        time.sleep(1)
        plt.savefig(png_path, dpi=600)

        # plt.show()
        plt.close()

    @staticmethod
    def merge_results(N, D, S, strategy, bar_dict):
        models = []
        bar_dictc = {
            N: [],
            D: [],
            S: [],
        }

        for label, y in bar_dict.items():
            for classifier, v in y.items():
                means_per_classi = [x.mean for x in v]
                evaluation = EvaluateTrainingModel().evaluate_training_model_by_ttest(
                    classifier, 0.5, np.array(means_per_classi), label, strategy
                )
                if classifier not in models:
                    models.append(classifier)
                bar_dictc[label].append(evaluation)
                # bar_dictc[label].append(statistics.mean(means_per_classi))
        return models, bar_dictc

    @staticmethod
    def separate_results_by_labels(subject_l, image_l, clasiifiers):
        bar_dict = {
            subject_l: {},
            image_l: {},
        }
        for classifier, list_per_classifier in clasiifiers.items():
            for p in list_per_classifier:
                if "subject" in p.column_name:
                    if bar_dict[subject_l].get(classifier) is None:
                        bar_dict[subject_l][classifier] = [p]
                    else:
                        bar_dict[subject_l][classifier].append(p)
                if "image" in p.column_name:
                    if bar_dict[image_l].get(classifier) is None:
                        bar_dict[image_l][classifier] = [p]
                    else:
                        bar_dict[image_l][classifier].append(p)
        return bar_dict

    @staticmethod
    def separate_results_by_patients(N, D, S, clasiifiers):
        bar_dict = {
            N: {},
            D: {},
            S: {},
        }
        for classifier, list_per_classifier in clasiifiers.items():
            for p in list_per_classifier:
                if N in p.column_name:
                    if bar_dict[N].get(classifier) is None:
                        bar_dict[N][classifier] = [p]
                    else:
                        bar_dict[N][classifier].append(p)
                if D in p.column_name:
                    if bar_dict[D].get(classifier) is None:
                        bar_dict[D][classifier] = [p]
                    else:
                        bar_dict[D][classifier].append(p)
                if S in p.column_name:
                    if bar_dict[S].get(classifier) is None:
                        bar_dict[S][classifier] = [p]
                    else:
                        bar_dict[S][classifier].append(p)
        return bar_dict

    @staticmethod
    def groupby_strategy(all_export_data):
        nested_dict = {}
        for data in all_export_data:
            if nested_dict.get(data.sub_column_name) is None:
                nested_dict[data.sub_column_name] = [data]
            else:
                nested_dict[data.sub_column_name].append(data)

        for strategy in nested_dict:
            value = nested_dict.get(strategy)
            nested_dict[strategy] = {}
            for data in value:
                if nested_dict[strategy].get(data.row_name) is None:
                    nested_dict[strategy][data.row_name] = [data]
                else:
                    nested_dict[strategy][data.row_name].append(data)
        return nested_dict

    def plot_data(X, y, labels):
        fig = go.Figure()
        i = 0
        for row in X:
            py_list = list(row)
            fig.add_trace(go.Scatter(x=y, y=py_list, name=f"{labels[i]}"))
            i += 1
        # fig.update_traces(marker=dict(color="red"))
        fig.show()

    @staticmethod
    def plot_bar_graph(
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
            title=dict(
                text=title,
                font=dict(family="Courier New, monospace", size=40, color="black"),
                # automargin=True,
                xanchor="center",
                yanchor="bottom",
                yref="paper",
            ),
            title_x=0.5,
            xaxis=dict(tickfont=dict(size=25), title=x_label),
            yaxis=dict(tickfont=dict(size=25), title=y_label),
            xaxis_title=dict(text=x_label, font=dict(size=35)),
            yaxis_title=dict(text=y_label, font=dict(size=35)),
        )

        html_name = ExportData.get_graph_name(".html", title)
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
            ln = "All Lobes"

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
        name = f"Voxelwise NaN values Distribution in {ln}"

        self.plot_bar_graph(
            ("Data Dimension", columns),
            ("Data Trials & NaNs Count", nans_column_wise),
            title=name,
            bar_color="#FFBF00",
        )

        nans_voxel_wise = brain.calculate_nans_trail_wise(brain.voxels)
        rows = [i for i in range(brain.voxels.shape[0])]
        name = f"Trialwise NaN values Distribution in {ln}"
        self.plot_bar_graph(
            ("Data Trials", rows),
            ("Count of NaN Values", nans_voxel_wise),
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
