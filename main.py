import os

import numpy as np
import pandas as pd
from pandas import DataFrame as DF
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

from icecream import ic


def plot_hist_graphs(heats_df: DF, cwd: str) -> None:

    exclude_above_quantile = 0.75 # write as fraction float e.g. 0.75

    drivers = list(heats_df['driver'].unique())  # debug- list the drivers

    quantiles = heats_df.groupby('driver')['time_seconds'].quantile(exclude_above_quantile)
    # filter outliers out, replace with nan
    heats_df['no_outlier_time_seconds'] = heats_df.apply(
        lambda row: row['time_seconds'] if row['time_seconds'] < quantiles.loc[row['driver']] else np.nan,
        axis=1
    )
    #ic(heats_df.head(10))

    sns.set_style("ticks")
    sns.set_context("talk")
    sns.set_palette('muted')

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 11), sharex=False)
    fig.subplots_adjust(hspace=0.4)

    g1 = sns.histplot(data=heats_df, x='time_seconds', hue='driver', multiple='stack',
                      ax=axs[0])
    g2 = sns.histplot(data=heats_df, x='no_outlier_time_seconds', hue='driver', multiple='stack',
                      ax=axs[1], binwidth=0.2)

    for ax in axs:
        ax.grid(True, which='both', axis='x', linewidth=0.5)
        ax.minorticks_on()

    axs[0].set_title("Histogram of all driver lap times")
    axs[1].set_title(f"Histogram of driver lap times, "
                     f"\nexcluding outliers above quantile {int(exclude_above_quantile*100)} of all times.")
    axs[1].legend_.remove()
    axs[0].set_xlabel("Time [s]")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_yticks(np.arange(0,15,2))

    plt.savefig(f'{cwd}/graphs/histogram_laptimes.png', dpi=150)
    plt.show()


    plt.close()
    plt.clf()


    return

def plot_basic_graphs(heats_df: DF, cwd: str) -> None:

    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette('muted')

    fastest_laps = heats_df.groupby('driver').min()
    fastest_laps = fastest_laps.sort_values(by='time_seconds')

    stdev_laps = heats_df.groupby('driver')["time_seconds"].agg('std')

    laps_data = fastest_laps.merge(DF(stdev_laps), on='driver', suffixes=['_min', '_stdev'])
    ic(laps_data)

    g = sns.catplot(kind='bar', data=laps_data, x='driver', y='time_seconds_min', errorbar=None)

    # Add error bars manually using the standard deviations
    ax = g.facet_axis(0, 0)  # Get the axis object
    ax.bar(laps_data.index, laps_data['time_seconds_min'], yerr=laps_data['time_seconds_stdev'], capsize=5)

    for ax in g.axes.flat:
        ax.grid(True, which='both', axis='y', linewidth=0.5)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(plt.NullLocator())

    for p in ax.patches:
        x = p.get_x() + p.get_width() / 2
        y = p.get_height() / 2
        ax.annotate(f'{p.get_height():.3f}', (x, y),
                    ha='center', va='center', xytext=(0, 0), textcoords='offset points', fontsize=10, color='white')

    plt.suptitle("Fastest lap times by each driver,\nerror bars show STDEV")
    plt.subplots_adjust(top=0.85, right=0.9, bottom=0.3)
    ticks = np.arange(0, 100, 5)
    plt.yticks(ticks)
    plt.ylim(0, 52)
    plt.xticks(rotation=30)
    plt.savefig(f'{cwd}/graphs/fastest_laps.png', dpi=150)
    plt.show()


    plt.close()
    plt.clf()




def plot_single_graphs(heats_df: DF, cwd: str) -> None:

    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette('muted')

    g = sns.catplot(data=heats_df, kind='point', x='lap', y='time_seconds',
                hue='driver', col='heat', legend_out=True, dodge=True, errorbar=None,
                alpha=0.75, markers='o')

    for ax in g.axes.flat:
        ax.grid(True, which='both', axis='y', linewidth=0.5)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(plt.NullLocator())

    ticks = np.arange(36, 62, 2)
    plt.yticks(ticks)
    plt.ylim(38, 54)
    plt.suptitle("Time against lap for different heats")
    plt.subplots_adjust(top=0.85, right=0.9)

    plt.savefig(f'{cwd}/graphs/laps_by_heat.png', dpi=150)
    plt.show()

    plt.close()
    plt.clf()

    return


def plot_cat_graphs(heats_df: DF, cwd: str) -> None:

    heats_df_min_time = heats_df.groupby(['heat', 'driver'], as_index=False)['time_seconds'].min()


    sns.set_style("ticks")
    sns.set_context("notebook")
    sns.set_palette('muted')

    sns.catplot(data=heats_df, kind='point', x='heat', y='time_seconds',
                hue='driver', legend_out=False, dodge=True, errorbar=('ci', 75),
                alpha=0.75, capsize=0.15)

    ticks = np.linspace(36, 62, 14)
    plt.yticks(list(ticks))
    plt.ylim(38, 54)
    plt.grid(True, which='both', axis='y')
    plt.title("Driver median lap time across different heats,\nerrorbars show IQR", pad=5)
    plt.subplots_adjust(top=0.85, right=0.9)

    plt.savefig(f'{cwd}/graphs/median_catplot.png', dpi=150)
    plt.show()


    plt.close()
    plt.clf()



    sns.catplot(data=heats_df_min_time, kind='point', x='heat', y='time_seconds',
                hue='driver', legend_out=False, dodge=True, errorbar=('ci', 75),
                alpha=0.75, capsize=0.15)


    ticks = np.arange(36, 62, 0.5)
    plt.yticks(list(ticks))
    plt.ylim(38, 44)
    plt.grid(True, which='both', axis='y')
    plt.title("Driver fastest lap time across different heats", pad=5)
    plt.subplots_adjust(top=0.9, right=0.9)
    plt.legend(loc='upper right')

    plt.savefig(f'{cwd}/graphs/fastest_catplot.png', dpi=150)
    plt.show()


    plt.close()
    plt.clf()

    return

def format_timedelta(time: str) -> Optional[pd.Timedelta]:
    """
    Format times as pd.timedelta.
    """

    if pd.isna(time) or time.lower()=='nan':
        return pd.NaT

    time_new = str(time)

    if time.count(':') == 1:
        time_new = f'00:{time}'
    elif time.count(':') == 0:
        time_new = f'00:00:{time}'

    return pd.to_timedelta(time_new)


def get_heat_from_csv(path: str, ftype: str = '.csv') -> DF:
    """
    Gets the data of the csv from path provided, and labels heat with number in filename, e.g. heat_1.csv = 1
    :param path:
    :return:
    """
    heat = path.rsplit('_')[-1].replace(ftype, '')
    df = pd.read_csv(path, dtype=str)
    df['heat'] = heat
    df['lap'] = df['lap'].astype(int)
    df['heat'] = df['heat'].astype(int)
    df = df.set_index(['heat', 'lap']) # multi index
    df.columns.name = 'driver' # give context to dataframe name values

    for col in df.columns:  # Assuming the first column is 'lap' or some other identifier
        df[col] = df[col].astype(str).apply(format_timedelta)

    return df



def main() -> None:
    """
    Main entry point.
    """

    cwd = fr'{os.getcwd()}'
    data_path = cwd + '/karting_times'
    file_type = '.csv'
    files = os.listdir(data_path)

    heats: List[DF] = []

    for file in files:
        # Check if .csv
        if file.endswith(file_type):
            heats.append(get_heat_from_csv(f'{data_path}/{file}', ftype=file_type))


    heats_df = pd.concat(heats)
    heats_df = heats_df.sort_values(by=['heat', 'lap'], ascending=[True, True])
    heats_df.to_csv(f"{cwd}/data/data_merged.csv")

    heats_reset = heats_df.reset_index()
    melted_heats_df = heats_reset.melt(id_vars=['heat', 'lap'], var_name='driver', value_name='time')
    melted_heats_df['time_seconds'] = melted_heats_df['time'].dt.total_seconds()
    melted_heats_df = melted_heats_df.dropna(subset=['time'])
    melted_heats_df.to_csv(f"{cwd}/data/data_melted.csv")

    plot_cat_graphs(melted_heats_df, cwd)
    plot_single_graphs(melted_heats_df, cwd)
    plot_basic_graphs(melted_heats_df, cwd)
    plot_hist_graphs(melted_heats_df, cwd)



if __name__ == '__main__':
    main()

