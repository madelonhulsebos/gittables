# GitTables
Code for extracting, parsing and annotating tables from GitTables (https://gittables.github.io).

## Background
The code in this repository resemble the procedures for:
- Extracting CSV files from GitHub based on query topics from WordNet.
- Parsing CSV files to Pandas tables.
- Annotating the tables with syntactic and semantic matching.
- Writing the table and annotation metadata to Parquet files.

These procedures were used to construct the GitTables 1.7M corpus of relational tables, which you can find [on Zenodo here](https://zenodo.org/record/4943312). The go-to for more information about the GitTables project or contact details, please visit [our website](https://gittables.github.io). The GitTables paper can be retrieved [here](https://arxiv.org/abs/2106.07258).

## Installation
Before running any of the code, you need to do three things:
- From the root directory, install the `gittables` package using `pip install .`.
- Install the dependencies in your environment with e.g. `pip` using `pip install -r requirements.txt`.
- Download the proper FastText model, and make sure you have the file named `cc.en.300.bin` in the `scripts/` directory.
- Add your personal GitHub username and token to the `settings.toml` file.

## Usage
The pipeline consists of two main stages, of which the main scripts are stored in `scripts/`.
1. Extract the CSV files from GitHub by running `python scripts/file_extraction.py`.
2. Annotate the retrieved tables by running `python scripts/table_annotation.py`.

This will yield a directory with CSV files and annotated tables per topic.
For each topic directory, you will find the raw CSV files and tables in `csv_files/` and `tables/` directories.

**Warning**: running the code as-is might be time consuming.

## Issues and contributions
Contributions to speed up the processes are appreciated.
If you run into issues or have question, please file them through GitHub [here](https://github.com/madelonhulsebos/gittables/issues).
