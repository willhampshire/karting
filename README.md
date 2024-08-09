# Karting Times EDA

## Overview

Exploratory data analysis of TeamSport go karting times from my 3 heats, using skills taught in DataCamp Associate Data Scientist track.
- Using `pandas` to merge, concat, melt, interact with .csv files, and other key `DataFrame` methods
- Clean and transform `DataFrames` using multi-indexes, suitible datatypes e.g. `pd.Timedelta`, and suitible values for no value - `pd.NaT`, `np.NaN`
- Filtering by applying lambda functions to `DataFrame`
- Type hinting using `Typing` library and module datatypes
- Visualisation using `seaborn` `catplot`, `histplot`, customising for maximum readability, saving at high dpi

## Installation

Install Python 3.11.
Pull the repository, and install the requried packages using `pip` or other:

```sh
pip install -r requirements.txt
```

## Project Structure

- `main.py`: The main script for running the analysis. It processes the data, applies transformations, and generates plots.
- `data/`: Directory where processed data files are saved.
- `graphs/`: Directory where generated plots are saved.

## Usage

1. **Check Data**: CSV files should already be present containing lap times in the `karting_times` directory. Ensure that each file follows the naming convention `heat_<number>.csv`.

2. **Run**: Execute `main.py`. This will process the data, apply transformations, and generate the visualizations.

    ```sh
    python main.py
    ```

3. **Check Results**: The processed data and generated plots will be saved in the `data/` and `graphs/` directories, respectively. The saved data can be used to do other external visualisation, such as in Tableau.

## Example

The below image is one of the generated visualisations, showing how each driver improved their fastest lap times across the heats.

![example_image](https://github.com/willhampshire/karting/blob/master/graphs/fastest_catplot.png)


