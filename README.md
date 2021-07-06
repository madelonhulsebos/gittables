# GitTables
Code for extracting, parsing and annotating tables from GitTables (https://gittables.github.io).

## Background
The code in this repository resemble the procedures for:
- Extracting CSV files from GitHub based on query topics from WordNet.
- Parsing CSV files to Pandas tables.
- Annotating the tables with syntactic and semantic matching.
- Writing the table and annotation metadata to Parquet files.

These procedures were used to construct the GitTables 1.7M corpus of relational tables, which you can find [on Zenodo here](https://zenodo.org/record/4943312). The go-to for more information about the GitTables project or contact details, please visit [our website](https://gittables.github.io). The GitTables paper can be retrieved [here](https://arxiv.org/abs/2106.07258).

## Usage
To run this code, you need two things:
- Download the proper FastText model, and make sure you have the file named `cc.en.300.bin` in the `scripts` directory.
- Add your personal GitHub username and token to the `settings.toml` file.
