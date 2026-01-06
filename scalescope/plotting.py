"""Functions to generate comparison plots."""

from collections import namedtuple

import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pandas as pd
import seaborn as sns

from scalescope.labels import *  # noqa: F401,F403

UnitConversion = namedtuple("UnitConversion", ["conversion_function", "designation"])


def bytes2gb(x):
    return x / (1024 * 1024 * 1024)


def count2millions(x):
    return x / 1000000


data_in_gb = UnitConversion(conversion_function=bytes2gb, designation="GB")

data_in_millions = UnitConversion(conversion_function=count2millions, designation="in mil")


# Keep plotting-only helpers here
roi_marks = ["eff", "mem ineff", "comm_ineff", "both_ineff"]


def plot_node_comparison(
    df,
    tp_node_col="TP_Node",
    tp_xpu_count_col="Total_XPU_Count",
    log_scale=False,
    title=None,
    start_y_with_zero=False,
    reverse_x=False,
    x_label=None,
    y_column="Time_To_Train_Sec",
    y_label=None,
    x_tick_label_col=None,
    y_step=None,
    hlines=None,
    y_scaler=1.0 / (24 * 3600),
    show_relative=False,
    annotation_cols=None,
    annotation_labels=None,
    static_annotation_format_str=None,  # New: Format string for static annotations
    static_annotation_offset=(5, 5),  # New: (x, y) offset in points for static annotations
    static_annotation_fontsize=8,  # New: Fontsize for static annotations
):
    """
    Plots a comparison of a given metric for different nodes as a line graph
    with interactive tooltips and optional static annotations on the plot itself.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        tp_node_col (str): Column name identifying different nodes/lines.
        tp_xpu_count_col (str): Column name for the x-axis values (e.g., number of chips).
        log_scale (bool): Whether to use a logarithmic scale for the y-axis.
        title (str): Title of the plot.
        start_y_with_zero (bool): If True, y-axis starts at zero (if not log_scale).
        reverse_x (bool): Whether to reverse the x-axis.
        x_label (str): Label for the x-axis.
        y_column (str): Column name for the primary y-axis data.
        y_label (str): Label for the y-axis.
        x_tick_label_col (str): Optional column in df to use for x-tick labels.
        y_step (float): Specific step size for y-axis ticks (if not log_scale).
        hlines (list): List of dictionaries for horizontal lines.
        y_scaler (float): Multiplier for y_column values.
        show_relative (bool): If True, plots relative change.
        annotation_cols (list): List of column names from df to include in tooltips and static annotations.
        annotation_labels (list): Optional list of labels for annotation_cols.
        static_annotation_format_str (str): Python f-string or .format() style string to customize
                                            static annotations. Use {col_name} or {0}, {1} placeholders.
                                            Example: "BS:{Batch_Size}, Opt:{Optimizer}"
                                            If None, a default multiline format is used.
        static_annotation_offset (tuple): (x_offset, y_offset) in points for placing static text.
        static_annotation_fontsize (int): Font size for static annotations on the plot.
    """

    df = df.copy()
    df[tp_xpu_count_col] = pd.to_numeric(df[tp_xpu_count_col], errors="coerce")
    df[y_column] = pd.to_numeric(df[y_column], errors="coerce")
    if annotation_cols:
        for acol in annotation_cols:
            if acol not in df.columns:
                print(f"Warning: Annotation column '{acol}' not found in DataFrame.")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))  # Use subplots to get ax for annotations

    node_names = df[tp_node_col].unique()
    unique_x_values = sorted(df[tp_xpu_count_col].unique())

    x_positions = np.arange(len(unique_x_values))
    x_value_position_map = dict(zip(unique_x_values, x_positions))

    if reverse_x:
        unique_x_values = unique_x_values[::-1]
        x_value_position_map = dict(zip(unique_x_values, x_positions[::-1]))

    x_tick_labels_to_display = None
    if x_tick_label_col and x_tick_label_col in df.columns:
        x_tick_labels_to_display = []
        for x_val in unique_x_values:
            label_series = df[df[tp_xpu_count_col] == x_val][x_tick_label_col]
            x_tick_labels_to_display.append(
                label_series.iloc[0] if not label_series.empty else str(x_val)
            )
    else:
        x_tick_labels_to_display = [str(x_val) for x_val in unique_x_values]

    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "X", "P"]
    lines = []
    line_data_for_tooltips = {}  # Store data for tooltips

    plot_column_name = y_column
    effective_y_axis_label = y_label if y_label else "Time (days)"

    if show_relative:
        if len(node_names) == 0:
            print("Warning: Cannot show relative values with no nodes.")
            show_relative = False
        else:
            first_node_name = node_names[0]
            baseline_df = df[df[tp_node_col] == first_node_name].set_index(tp_xpu_count_col)
            relative_values_list = []
            for _, row in df.iterrows():
                current_x_val = row[tp_xpu_count_col]
                if current_x_val in baseline_df.index:
                    baseline_val = baseline_df.loc[current_x_val, y_column]
                    if pd.notna(baseline_val) and baseline_val != 0:
                        rel_change = ((row[y_column] - baseline_val) / baseline_val) * 100
                    elif pd.notna(baseline_val) and baseline_val == 0 and row[y_column] == 0:
                        rel_change = 0.0
                    else:
                        rel_change = np.nan
                else:
                    rel_change = np.nan
                relative_values_list.append(rel_change)
            df["relative_value"] = relative_values_list
            plot_column_name = "relative_value"
            effective_y_axis_label = y_label if y_label else "Relative Change (%)"
            applied_y_scaler = 1.0
    else:
        applied_y_scaler = y_scaler

    for i, node_name_val in enumerate(node_names):
        data_for_line = df[df[tp_node_col] == node_name_val].copy()
        data_for_line = data_for_line.sort_values(tp_xpu_count_col)
        data_for_line.dropna(subset=[plot_column_name, tp_xpu_count_col], inplace=True)
        if data_for_line.empty:
            continue

        current_x_plot_positions = [
            x_value_position_map[x_val] for x_val in data_for_line[tp_xpu_count_col]
        ]
        current_y_values = data_for_line[plot_column_name] * applied_y_scaler

        (line,) = ax.plot(
            current_x_plot_positions,
            current_y_values,
            marker=markers[i % len(markers)],
            label=node_name_val,
            linewidth=2,
            markersize=8,
            alpha=0.9,
        )  # Note: ax.plot returns a list of lines
        lines.append(line)
        line_data_for_tooltips[line] = data_for_line  # For tooltips, store original subset

        # Add static annotations
        if annotation_cols:
            for idx, row_data in data_for_line.iterrows():
                x_pos = x_value_position_map[row_data[tp_xpu_count_col]]
                y_pos = row_data[plot_column_name] * applied_y_scaler

                static_text_parts = []
                if static_annotation_format_str:
                    # Prepare data for formatting
                    format_data_dict = {col: row_data.get(col, "N/A") for col in annotation_cols}
                    try:
                        static_ann_text = static_annotation_format_str.format(**format_data_dict)
                    except KeyError as e:
                        print(
                            f"Warning: Key {e} not found for static_annotation_format_str. Check column names."
                        )
                        static_ann_text = "Format Error"
                    except Exception as e:
                        print(f"Error formatting static_annotation_format_str: {e}")
                        static_ann_text = "Format Error"

                else:  # Default multiline format
                    for k, ann_col in enumerate(annotation_cols):
                        ann_val = row_data.get(ann_col, "N/A")  # Use .get for safety
                        ann_lab = (
                            annotation_labels[k]
                            if annotation_labels and k < len(annotation_labels)
                            else ann_col.replace("_", " ").title()
                        )
                        if isinstance(ann_val, float):
                            static_text_parts.append(f"{ann_lab}: {ann_val:.2f}")
                        else:
                            static_text_parts.append(f"{ann_lab}: {str(ann_val)}")
                    static_ann_text = "\n".join(static_text_parts)

                if static_ann_text:  # Only annotate if text is generated
                    ax.annotate(
                        static_ann_text,
                        (x_pos, y_pos),
                        textcoords="offset points",
                        xytext=static_annotation_offset,
                        ha="left" if static_annotation_offset[0] >= 0 else "right",
                        va="bottom" if static_annotation_offset[1] >= 0 else "top",
                        fontsize=static_annotation_fontsize,
                        bbox=(
                            dict(boxstyle="round,pad=0.2", fc="ivory", alpha=0.6, ec="gray", lw=0.5)
                            if "\n" in static_ann_text
                            else None
                        ),  # Bbox for multiline
                    )
    # End static annotations

    if hlines:
        plot_xmin, plot_xmax = ax.get_xlim()
        for hline_props in hlines:
            value = hline_props.get("value")
            if value is not None:
                label = hline_props.get("label", f"y = {value:.2f}")
                color = hline_props.get("color", "red")
                style = hline_props.get("style", "--")
                ax.axhline(y=value, color=color, linestyle=style, linewidth=1.5)
                text_x_pos = plot_xmin - (plot_xmax - plot_xmin) * 0.02
                ax.text(text_x_pos, value, label, color=color, va="center", ha="right", fontsize=9)

    effective_x_axis_label = x_label if x_label else f'{tp_xpu_count_col.replace("_", " ").title()}'
    ax.set_xlabel(effective_x_axis_label, fontsize=14, labelpad=10)
    ax.set_ylabel(effective_y_axis_label, fontsize=14, labelpad=10)
    default_title = f"{effective_y_axis_label} vs. {effective_x_axis_label}"
    ax.set_title(title if title else default_title, fontsize=16, pad=20, loc="center")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_tick_labels_to_display, fontsize=12, rotation=0)

    if y_step is not None and not log_scale:
        current_ymin, current_ymax = ax.get_ylim()
        start_tick = (
            np.floor(current_ymin / y_step) * y_step
            if current_ymin > 0
            else np.ceil(current_ymin / y_step) * y_step
        )
        end_tick = (
            np.ceil(current_ymax / y_step) * y_step
            if current_ymax > 0
            else np.floor(current_ymax / y_step) * y_step
        )
        if start_tick > end_tick:
            start_tick = end_tick
        y_ticks_list = np.arange(start_tick, end_tick + y_step * 0.5, y_step)
        ax.set_yticks(y_ticks_list)
        ax.tick_params(axis="y", labelsize=12)
    else:
        ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="both", which="both", direction="in", length=6, width=1)

    if log_scale:
        ax.set_yscale("log")
    if start_y_with_zero and not log_scale:
        ax.set_ylim(bottom=0)
    ax.set_xlim(-0.5, len(unique_x_values) - 0.5)

    ax.grid(visible=True, which="major", color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    # Legend inside the plot
    ax.legend(
        title=tp_node_col.replace("_", " ").title(),
        title_fontsize=11,  # Slightly smaller for inside
        fontsize=9,  # Slightly smaller for inside
        loc="best",  # 'best' tries to find a good spot
        frameon=True,
        facecolor="white",
        edgecolor="gray",
        framealpha=0.7,  # Semi-transparent
    )

    if not lines:
        ax.text(
            0.5,
            0.5,
            "No data to plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        plt.tight_layout()
        plt.show()
        return

    cursor = mplcursors.cursor(lines, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")
    def on_add(sel):
        selected_line = sel.artist
        original_data_for_line = line_data_for_tooltips[selected_line]
        plot_x_coord = sel.target[0]

        line_x_plot_positions = [
            x_value_position_map[x_val] for x_val in original_data_for_line[tp_xpu_count_col]
        ]
        closest_idx_in_data = min(
            range(len(line_x_plot_positions)),
            key=lambda i: abs(line_x_plot_positions[i] - plot_x_coord),
        )
        selected_row = original_data_for_line.iloc[closest_idx_in_data]

        node_name_for_tooltip = selected_line.get_label()
        actual_x_value = selected_row[tp_xpu_count_col]
        plotted_y_value = sel.target[1]

        tooltip_text = f"Node: {node_name_for_tooltip}"
        tooltip_text += f"\n{effective_x_axis_label}: {actual_x_value}"
        y_value_suffix = "%" if show_relative else ""
        tooltip_text += f"\n{effective_y_axis_label}: {plotted_y_value:.2f}{y_value_suffix}"

        if annotation_cols:
            for i, ann_col_name in enumerate(annotation_cols):
                if ann_col_name in selected_row:
                    ann_value = selected_row[ann_col_name]
                    ann_label = (
                        annotation_labels[i]
                        if annotation_labels and i < len(annotation_labels)
                        else ann_col_name.replace("_", " ").title()
                    )
                    if isinstance(ann_value, float):
                        tooltip_text += f"\n{ann_label}: {ann_value:.2f}"
                    elif pd.isna(ann_value):
                        tooltip_text += f"\n{ann_label}: N/A"
                    else:
                        tooltip_text += f"\n{ann_label}: {str(ann_value)}"

        sel.annotation.set_text(tooltip_text)
        sel.annotation.get_bbox_patch().set(fc="lightyellow", alpha=0.9)
        sel.annotation.set_fontsize(9)

    plt.tight_layout()
    plt.show()
