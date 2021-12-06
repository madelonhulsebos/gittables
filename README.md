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

The pipeline consists of two main stages, of which the main scripts are stored in `scripts/`, run these scripts from the root directory.
Log files of the extraction and annotation process are written to the `logs/` directory.

**Warning**: running the code as-is is time consuming.

### Extracting CSV files
The CSV files can be extracted by running `python scripts/file_extraction.py`.

This step will use the GitHub code search API and request module to extract CSV files based on topics from WordNet (WordNet will be downloaded automatically).

In each topic directory within the `table_collection` directory, you will find the raw CSV files and tables in `csv_files/`.

If you want to get tables for a custom list of topics, you can modify the `file_extraction.py` script by setting the `custom_topics` argument of the `set_topics` method to a specified list of query topics, e.g. ['apple', 'pie', 'nut'].
This list will then be used to build the table collection repository, instead of the topics from WordNet.

### Parsing and annotating tables
When the CSV files for (some of) the topics are extracted, these files can be parsed to a table and annotated with column semantics by running `python scripts/table_annotation.py`.

The ontologies used for the annotation are written to the `ontologies/` directory for future reference. The ontologies used for constructing GitTables 1.7M can be downloaded from our website https://gittables.github.io.


## Issues and contributions
Contributions to speed up the processes are appreciated.
If you run into issues or have question, please file them through GitHub [here](https://github.com/madelonhulsebos/gittables/issues).
