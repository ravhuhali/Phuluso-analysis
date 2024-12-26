import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_weighted_age_distribution(df, age_column, weight_column, province_column):
    fig, ax = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
    axes = ax.flatten()
    percentiles = {}
    percentiles_list = []

    for axes, province in zip(axes, df[province_column].unique()):
        df_g = df[df[province_column] == province]  # Filter data for each province

        # Calculate the weighted average age
        weighted_avg_age = (df_g[age_column] * df_g[weight_column]).sum() / df_g[
            weight_column
        ].sum()

        _95th_percentile = np.percentile(df_g[age_column], 95, method="inverted_cdf")
        _25th_percentile = np.percentile(df_g[age_column], 25, method="inverted_cdf")

        # Plot the histogram with KDE and weighted average/percentiles
        sns.histplot(df_g, x=age_column, kde=True, ax=axes, weights=df_g[weight_column])
        axes.axvline(
            weighted_avg_age,
            linestyle=":",
            color="r",
            label=f"Mean: {round(weighted_avg_age)}",
        )
        axes.axvline(
            _95th_percentile,
            linestyle=":",
            color="g",
            label=f"95th percentile: {round(_95th_percentile)}",
        )
        axes.axvline(
            _25th_percentile,
            linestyle="-",
            color="b",
            label=f"25th percentile: {round(_25th_percentile)}",
        )

        # Store the 95th percentile for each province
        percentiles[province] = _95th_percentile
        percentiles_list.append(_95th_percentile)

        axes.set_title(f"{province} Age Distribution")
        axes.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

    province_max_percentile = max(percentiles_list)
    province_with_max_percentile = [
        key for key, value in percentiles.items() if value == province_max_percentile
    ][0]




def plot_age_distribution(df, age_column, weight_column, title):
    fig, ax = plt.subplots(figsize=(6, 3))

    # Calculate the weighted average age
    weighted_avg_age = (df[age_column] * df[weight_column]).sum() / df[
        weight_column
    ].sum()

    # Plot the histogram with KDE and weighted average age
    sns.histplot(data=df, x=age_column, kde=True, ax=ax, weights=df[weight_column])
    ax.axvline(
        weighted_avg_age,
        linestyle=":",
        color="r",
        label=f"Weighted Avg Age: {round(weighted_avg_age)}",
    )

    ax.set_title(f"{title} Distribution")
    ax.legend()

    # Calculate the age that owns the most houses (weighted by house_wgt)
    weighted_houses = df.groupby(age_column)[weight_column].sum().reset_index()
    highest_age = weighted_houses.loc[weighted_houses[weight_column].idxmax()]


def plot_weighted_avg_age_heatmap(df, geo_column, age_column, weight_column, province):

    grouped = (
        df.groupby([province, geo_column])
        .apply(
            lambda x: (x[age_column] * x[weight_column]).sum() / x[weight_column].sum()
        )
        .reset_index(name="weighted_avg_age")
    )

    heatmap_data = grouped.pivot(
        index=geo_column, columns=province, values="weighted_avg_age"
    )

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        cbar_kws={"label": "Weighted Avg Age"},
    )
    plt.title("Weighted Average Age of House Owners by Province and GeoType")
    plt.xlabel("Province")
    plt.ylabel("GeoType")
    plt.tight_layout()
    plt.show()


def plot_weighted_avg_age(
    df, geo_type_data, GeoType, age_column, weight_column, province
):

    grouped = (
        df.groupby([province, GeoType])
        .apply(
            lambda x: (x[age_column] * x[weight_column]).sum() / x[weight_column].sum()
        )
        .reset_index(name="weighted_avg_age")
    )

    return grouped


def plot_cell_phone_distribution(df, column_name):

    # Calculate value counts and percentages
    dfg = df[column_name].value_counts().reset_index()
    total = dfg["count"].sum()
    dfg["percentage"] = (dfg["count"] / total) * 100

    # Plotting the bar chart
    fig, ax = plt.subplots()
    sns.barplot(dfg, x=column_name, y="percentage", ax=ax)

    # Adding bar labels
    for container in ax.containers:
        ax.bar_label(container)

    ax.set_title("Cell Phone Distribution")

    plt.tight_layout()
    plt.show()


def percentage_phone(df, column_name, year):

    # Calculate value counts and percentages
    dfg = df[column_name].value_counts().reset_index()
    total = dfg["count"].sum()
    dfg["percentage"] = (dfg["count"] / total) * 100
    percent = dfg.loc[dfg["cell_phone"] == "Yes", "percentage"]
    xx = f"{percent}  %  in {year}  has cell phones"
    return xx


def plot_phone_ownership(df, column_name, province_col, weight_col, plot_name):

    fig, ax = plt.subplots(3, 3, figsize=(12, 9), sharex=False, sharey=True)
    axes = ax.flatten()

    percentiles = {}
    percentiles_list = []
    for axes, province in zip(axes, df[province_col].unique()):
        df_g = df[df[province_col] == province]

        # Calculate weighted counts and percentages
        weighted_counts = (
            df_g.groupby(column_name)
            .agg(weighted_count=(weight_col, "sum"))
            .reset_index()
        )
        total_weight = weighted_counts["weighted_count"].sum()
        weighted_counts["percentage"] = (
            weighted_counts["weighted_count"] / total_weight
        ) * 100
        weighted_counts[column_name] = weighted_counts[column_name].astype("category")

        # Plot the bar chart
        sns.barplot(
            data=weighted_counts, x=column_name, y="percentage", ax=axes, color="b"
        )

        # Store the maximum percentage for each province
        percentiles[province] = weighted_counts["percentage"].max()
        percentiles_list.append(weighted_counts["percentage"].max())

        # Add labels on top of the bars
        for container in axes.containers:
            axes.bar_label(container)

        # Set titles and axis labels
        axes.set_title(f"{province} {plot_name} ownership percentages")
        axes.set_yscale("log")  # Log scale for the Y-axis
        axes.set_ylabel("Percentage")
        # axes.tick_params(axis='x', rotation=90)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def overall_phone_ownwers(df, cell_owner, provinces, house_wgt, response_dict):

    for ii, ij in response_dict.items():
        percentiles = {}
        percentiles_list = []

        for province in df[provinces].unique():
            df_g = df[df[provinces] == province]

            weighted_counts = (
                df_g.groupby(cell_owner)
                .agg(weighted_count=(house_wgt, "sum"))
                .reset_index()
            )

            total_weight = weighted_counts["weighted_count"].sum()

            weighted_counts["percentage"] = (
                weighted_counts["weighted_count"] / total_weight
            ) * 100
            weighted_counts = weighted_counts[weighted_counts[cell_owner] == ii]

            percentiles[province] = weighted_counts["percentage"].max()
            percentiles_list.append(weighted_counts["percentage"].max())

        max_percentile_province = max(percentiles, key=percentiles.get)
        print(f"The province with the highest {ij} is {max_percentile_province}.")


def gemerate_cellphone_count_per_month(df, Q62cel, house_wgt, province):
    # Create an empty list to store all the data
    all_weighted_counts = []

    # Loop through each province
    for province in df[province].unique():
        df_g = df[df[province] == province]

        # Group by 'Q62cell' and calculate the weighted count
        weighted_counts = (
            df_g.groupby(Q62cel).agg(weighted_count=(house_wgt, "sum")).reset_index()
        )
        total_weight = weighted_counts["weighted_count"].sum()

        # Calculate the percentage
        weighted_counts["percentage"] = (
            weighted_counts["weighted_count"] / total_weight
        ) * 100
        weighted_counts[Q62cel] = weighted_counts[Q62cel].astype("category")

        # Add a column to identify the province
        weighted_counts[province] = province

        # Append the data to the list
        all_weighted_counts.append(weighted_counts)
    final_table = pd.concat(all_weighted_counts, ignore_index=True)
    final_table = final_table[final_table[Q62cel] == 1]
    return final_table[["province", "Q62cell", "weighted_count", "percentage"]]


def metro_df(df, province_col, Q62cell, GeoType):


    all_weighted_counts = []

    for province in df[province_col].unique():
        df_g = df[df[province_col] == province]
        weighted_counts = df_g.groupby(GeoType)[Q62cell].count().reset_index()
        total_weight = weighted_counts[Q62cell].sum()
        weighted_counts["percentage"] = (weighted_counts[Q62cell] / total_weight) * 100
        weighted_counts[Q62cell] = weighted_counts[Q62cell].astype("category")
        weighted_counts[province_col] = province
        all_weighted_counts.append(weighted_counts)

    final_table = pd.concat(all_weighted_counts, ignore_index=True)
    final_table = final_table.drop(columns=[Q62cell])

    return final_table


def phone_calculate_weighted_percentage(df, year_range, column, weights_col):

    df_list = []

    for year in year_range:
        df_year = df[df["year"] == year]

        weighted_counts = (
            df_year.groupby(column)[weights_col]
            .sum()
            .reset_index(name="weighted_count")
        )
        total_weighted = weighted_counts["weighted_count"].sum()
        weighted_counts["percentage"] = (
            weighted_counts["weighted_count"] / total_weighted
        ) * 100

        weighted_counts["year"] = year
        df_list.append(weighted_counts)
        result_df = pd.concat(df_list, ignore_index=True)

    return result_df
