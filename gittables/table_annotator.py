"""
This module facilitates the annotation of .
"""

import collections
import itertools
import json
import logging
import os
import re
import requests
import shutil
import time
import traceback
import typing

import fasttext
import numpy as np
import pandas as pd
import pickle
import pyarrow as pa
import pyarrow.parquet as pq

from scipy import spatial


class TopicTablesProcessor:
    """Process (parse, annotate and store) CSVs to tables for a topic."""
    
    def __init__(self, topic: str, logs_filepath: str, table_collection_dir: str, ontology_dir: str, ontology_date: str):
        
        np.seterr(all="raise")

        self.csv_dir = os.path.join(table_collection_dir, f"{topic}/csv_files")
        if not os.path.exists(self.csv_dir):
            raise ValueError(f"No CSVs found for topic {topic}.")

        self.table_dir = os.path.join(table_collection_dir, f"{topic}/tables_licensed")
        if not os.path.exists(self.table_dir):
            os.makedirs(self.table_dir)

        logging_filepath = f"{logs_filepath}/annotation_logfile.log"
        if not os.path.exists(logging_filepath):
            open(logging_filepath, "w+")

        logging.getLogger().handlers = []
        logging.basicConfig(filename=logging_filepath, filemode="a", level=logging.INFO)
        self._logger = logging.getLogger()

        self.ft_model = fasttext.load_model("../scripts/cc.en.300.bin")
        
        if not os.path.exists(ontology_dir):
            raise ValueError("Ontology directory was not detected.")

        self.schema_ontology_df, self.schema_ontology_dict = self._read_ontology("schema", ontology_dir, ontology_date)
        self.schema_embedded_types = [
            self.ft_model.get_sentence_vector(_type)
            for _type in self.schema_ontology_df["cleaned_label"].tolist()
        ]
        self.dbpedia_ontology_df, self.dbpedia_ontology_dict = self._read_ontology("dbpedia", ontology_dir, ontology_date)
        self.dbpedia_embedded_types = [
            self.ft_model.get_sentence_vector(_type)
            for _type in self.dbpedia_ontology_df["cleaned_label"].tolist()
        ]


    def process_topic_tables(self, filenames_to_url: typing.Dict, topic: str):
        topic_table_metadata = {}
        number_parsed_tables = 0
        number_processed_tables = 0
        table_id = 0
        row_counts = []
        column_counts = []
        dtypes_counter = collections.Counter([])
        dtypes_percentages = {}
        annotation_type_counts = {}
        annotation_type_similarities = {
            "schema_embedding": {},
            "dbpedia_embedding": {}           
        }
        domains = []
        domains_dict = {
            "schema_filter": [],
            "schema_embedding": [],
            "dbpedia_filter": [],
            "dbpedia_embedding": []
        }
        # number of annotated columns
        schema_embedding_nac = []
        schema_filter_nac = []
        dbpedia_embedding_nac = []
        dbpedia_filter_nac = []
        annotation_counts_schema_filter = collections.Counter([])
        annotation_counts_schema_embedding = collections.Counter([])
        annotation_counts_dbpedia_filter = collections.Counter([])
        annotation_counts_dbpedia_embedding = collections.Counter([])
        start = time.time()
        new_start = start
        for i, (filename, url) in enumerate(filenames_to_url.items()):
            try:
                table_license = self.get_table_license(url)
                if table_license == None:
                    continue
                table, table_metadata = self.parse_csv_file(filename, url)
                table_metadata = self.annotate_table(
                    list(table.columns),
                    table_metadata,
                    filename
                )
                number_parsed_tables += 1

                table_metadata = {
                    **table_metadata,
                    **table_license
                }
                
                table_id = self.write_table_to_parquet(
                    table,
                    table_metadata,
                    table_id,
                    # skip .csv part of filename
                    filename[:-4]
                )
                
                # Only report topic metadata of tables in the collection.
                row_counts.append(table.shape[0])
                column_counts.append(table.shape[1])
                dtypes_counter = dtypes_counter + collections.Counter(table_metadata["dtypes"].values())
                table_domains = table_metadata["table_domain"]
                for key in table_domains:
                    domains_dict[key].append(table_domains[key])

                for dtype in table_metadata["dtypes_percentages"]:
                    dtype_percentage = table_metadata["dtypes_percentages"][dtype]
                    if dtype in dtypes_percentages:
                        dtypes_percentages[dtype].append(dtype_percentage)
                    else:
                        dtypes_percentages[dtype] = [dtype_percentage]

                # number of annotated tables = number of non-zero annotated columns
                # percentage annotated columns = number_of_annotated_columns / column_counts
                schema_embedding_nac.append(len(table_metadata["schema_embedding_column_types"]))
                schema_filter_nac.append(len(table_metadata["schema_filter_column_types"]))
                dbpedia_embedding_nac.append(len(table_metadata["dbpedia_embedding_column_types"]))
                dbpedia_filter_nac.append(len(table_metadata["dbpedia_filter_column_types"]))
                
                type_counts, annotation_type_similarities = self._get_type_counts_and_similarities(
                    table_metadata,
                    annotation_type_similarities,
                    "schema_filter",
                )
                annotation_counts_schema_filter = annotation_counts_schema_filter + type_counts
            
                type_counts, annotation_type_similarities = self._get_type_counts_and_similarities(
                    table_metadata,
                    annotation_type_similarities,
                    "schema_embedding",
                )
                annotation_counts_schema_embedding = annotation_counts_schema_embedding + type_counts
                
                type_counts, annotation_type_similarities = self._get_type_counts_and_similarities(
                    table_metadata,
                    annotation_type_similarities,
                    "dbpedia_filter",
                )
                annotation_counts_dbpedia_filter = annotation_counts_dbpedia_filter + type_counts

                type_counts, annotation_type_similarities = self._get_type_counts_and_similarities(
                    table_metadata,
                    annotation_type_similarities,
                    "dbpedia_embedding",
                )
                annotation_counts_dbpedia_embedding = annotation_counts_dbpedia_embedding + type_counts
                
            except Exception as e:
                tb = traceback.format_exc()
                self._logger.error(f"On processing file {filename}, encountered error: {e}")
                self._logger.debug(f"With the following traceback: {tb}")
                continue
            finally:
                if i%1000 == 0:
                    runtime = np.round(time.time() - new_start, 4)
                    total_runtime = np.round(time.time() - start, 4)
                    self._logger.info(
                        f"""
                        Processed 1K files for topic {topic} in {runtime}s. In total {i} files in {total_runtime}s.
                        """
                    )
                    new_start = time.time()
    
        topic_table_metadata["number_CSVs"] = i+1 # Start counting at 1
        topic_table_metadata["number_parsed_tables"] = number_parsed_tables
        topic_table_metadata["number_procesed_tables"] = table_id
        topic_table_metadata["row_counts"] = row_counts
        topic_table_metadata["column_counts"] = column_counts
        topic_table_metadata["dtype_counts"] = dict(dtypes_counter)
        topic_table_metadata["dtypes_percentages"] = dtypes_percentages
        topic_table_metadata["table_domains"] = domains_dict
        topic_table_metadata["number_of_annotated_columns"] = {
            "schema_embedding": schema_embedding_nac,
            "schema_filter": schema_filter_nac,
            "dbpedia_embedding": dbpedia_embedding_nac,
            "dbpedia_filter": dbpedia_filter_nac
        }
        
        topic_table_metadata["annotation_type_counts"] = {
            "schema_filter": dict(annotation_counts_schema_filter),
            "schema_embedding": dict(annotation_counts_schema_embedding),
            "dbpedia_filter": dict(annotation_counts_dbpedia_filter),
            "dbpedia_embedding": dict(annotation_counts_dbpedia_embedding),
        }
        topic_table_metadata["annotation_type_similarities"] = annotation_type_similarities
        topic_table_metadata["annotation_runtime"] = time.time() - start
        
        return topic_table_metadata
    
    
    def parse_csv_file(self, filename: str, url: str):
        try:
            table = pd.read_csv(
                os.path.join(self.csv_dir, filename),
                error_bad_lines=False,
                warn_bad_lines=False,
                engine="python",
                sep=None
            )
            
            table_dtypes = table.dtypes.astype(str)
            table_shape = table.shape
            
            table_csv_url = url
            
            table_metadata = {
                "table_csv_url": table_csv_url,
                "dtypes": table_dtypes.to_dict(),
                "number_rows": table_shape[0],
                "number_columns": table_shape[1],
                "dtypes_percentages": self._calculate_dtype_percentages(table_dtypes),
            }
        except Exception as e:
            msg = f"CSV with filename {filename} could not be parsed."
            self._logger.error(msg)
            raise e

        return table, table_metadata


    def get_table_license(self, url: str):
        """Lookup of license associated with the repository of a CSV file.
        It will return:
        - License, if a 'named' license is found (e.g. all licenses except the 'Other' category).
        - None, if there is no license, an error is encountered, or the license was undetermined ('Other').
        """
        repository_url = url.split("blob")[0]
        owner = repository_url.split("/")[-3]
        repo = repository_url.split("/")[-2]
        try:
            response = requests.get(
                f"https://api.github.com/repos/{owner}/{repo}/license",
                headers={"accept": "application/vnd.github.v3+json"},
                auth=("madelonhulsebos", "6f986ef0a3f9154c8c6ebff9040dcf08522ac5b5")
            )
            if response.status_code == 200:
                table_license = response.json()["license"]
                if table_license["name"] == "Other":
                    table_license = None
                else:
                    table_license = {"license": table_license}
            elif response.status_code == 404:
                # In this case, the repository is not associated with a license.
                table_license = None
            elif response.status_code == 403:
                # In this case, we likely reached the API limit.
                waiting_time = float(response.headers["X-RateLimit-Reset"]) - time.time()
                if waiting_time < 0:
                    # We will not waiting for nothing, there was something else wrong.
                    table_license = None
                else:
                    msg = f"Reached limit on owner {owner}, repo {repo}, waiting for {waiting_time} s"
                    self._logger.info(msg)
                    time.sleep(waiting_time)
                    get_table_license(url)
            else:
                # In this case, we encountered another error.
                code = response.status_code
                msg = f"Ran into another issue, with status code {code}"
                self._logger.info(msg)
                table_license = None
        except Exception as e:
            msg = f"Ran into exception {e}"
            self._logger.error(msg)
            table_license = None

        return table_license


    def annotate_table(self, table_columns: typing.List, table_metadata: typing.Dict, filename: str):
        try:
            filter_annotation_dict = self.filter_annotate_table(table_columns)
            fasttext_annotation_dict = self.fasttext_annotate_table(table_columns)

            table_metadata = {
                **table_metadata,
                **filter_annotation_dict,
                **fasttext_annotation_dict,
            }

            table_metadata["table_domain"] = {}
            for annotation_type in ["schema_filter", "schema_embedding", "dbpedia_filter", "dbpedia_embedding"]:
                table_metadata["table_domain"][annotation_type] = self._classify_table_domain(
                    table_metadata
                    .copy()
                    [f"{annotation_type}_column_types"]
                )

        except Exception as e:
            msg = f"Table from filename {filename} could not be annotated."
            self._logger.error(msg)
            self._logger.error(table_metadata)
            raise e
        
        return table_metadata


    def write_table_to_parquet(
        self,
        table: pd.DataFrame,
        table_metadata: typing.Dict,
        table_id: int,
        filename: str,
        filepath: str = ""
    ):
        """
        Write table with enhanced metadata to parquet file.

        The pandas dataframe representing the table is first converted to an arrow Table.
        The URL from which the table was parsed with pandas is then attached to the metadata.
        Finally, the table with enhanced metadata is written to a parquet file.

        Parameters
        ----------
        table
            DataFrame containing parsed table.
        table_metadata
            Metadata holding information about annotations of table columns.
        table_id
            Unique identifier for the fully processed table.
        filename
            Name of the CSV file (w/o extension) used for the parquet filename.
        filepath
            Path to which the parquet file should be written.
            Defaults to empty string, hence saving file to root path.
        """
        try:
            
            table_metadata["table_id"] = table_id
            
            pa_table = pa.Table.from_pandas(table)
            enhanced_metadata = self._attach_metadata(
                pa_table.schema.metadata,
                table_metadata
            )
            table_with_enhanced_metadata = pa_table.replace_schema_metadata(enhanced_metadata)
            pq.write_table(
                table_with_enhanced_metadata,
                os.path.join(self.table_dir, f"{filename}.parquet")
            )
            
            table_id += 1
            
        except Exception as e:
            self._logger.error(
                f"Table from {filename} could not be stored."
            )
            raise e

        return table_id


    def filter_annotate_table(self, table_columns: typing.List) -> typing.Dict:

        cleaned_table_columns = [
            re.sub(r"[_-]"," ", " ".join(
                re.findall("[0-9,a-z,.,\"#!$%\^&\*;:{}=\-_`~()\n\t\d]+|[A-Z](?:[A-Z]*(?![a-z])|[a-z]*)", col)
            )).lower()
            for col in table_columns
        ]
        
        dbpedia_annotation_dict = self._filter_table_on_ontology(
            cleaned_table_columns,
            table_columns,
            self.dbpedia_ontology_dict,
            "dbpedia"
        )
        schema_annotation_dict = self._filter_table_on_ontology(
            cleaned_table_columns,
            table_columns,
            self.schema_ontology_dict,
            "schema"
        )
        
        return {**dbpedia_annotation_dict, **schema_annotation_dict}

    def _read_ontology(self, ontology_name: str, ontology_dir: str, date: str):

        filepath = f"{ontology_dir}{ontology_name}_{date}"
        ontology_df = pd.read_pickle(f"{filepath}.pkl")

        with open(f"{filepath}.pickle", "rb") as f:
            ontology_dict = pickle.load(f)

        return ontology_df, ontology_dict


    def _filter_table_on_ontology(
        self,
        cleaned_table_columns: typing.List,
        table_columns: typing.List,
        ontology_dict: typing.Dict,
        ontology_name: str
    ):
        types = list(ontology_dict.keys())
        
        annotated_columns_indices = [
            cleaned_table_columns.index(_type) for _type in types if _type in cleaned_table_columns
        ]

        annotated_columns = [cleaned_table_columns[index] for index in annotated_columns_indices]
        original_annotated_columns = [table_columns[index] for index in annotated_columns_indices]

        annotation_dict = {
            f"{ontology_name}_filter_column_types": {},
        }        
        
        if len(annotated_columns) != len(original_annotated_columns):
            self._logger.error(
                f"""
                Filter annotating ontology: {schema_name}.
                Annotated columns are: {annotated_columns}
                and original annotated columns: {original_annotated_columns}.
                """
            )
            return annotation_dict

        for i, col in enumerate(original_annotated_columns):
            annotation_dict[f"{ontology_name}_filter_column_types"][col] = (
                ontology_dict[annotated_columns[i]]
            )

        return annotation_dict

    
    
    def fasttext_annotate_table(self, table_columns: typing.List) -> typing.Dict:
        """Annotate table columns with FastText for dbpedia and schema types."""
        try:
            table_columns = [col for col in table_columns.copy() if not col.startswith("Unnamed:") and not any(char.isdigit() for char in col)]
            cleaned_table_columns = [
                re.sub(r"[_-]", " ", " ".join(
                    re.findall("[0-9,a-z,.,\"#!$%\^&\*;:{}=\-_`~()\n\t\d]+|[A-Z](?:[A-Z]*(?![a-z])|[a-z]*)", col)
                )).lower() for col in table_columns.copy()
            ]

            header_embeddings = [
                self.ft_model.get_sentence_vector(column.strip("\n"))
                for column in cleaned_table_columns
            ]
            schema_annotation_dict = self._map_embedded_header_to_ontology(
                header_embeddings,
                table_columns,
                self.schema_ontology_dict,
                self.schema_ontology_df["cleaned_label"].tolist(),
                self.schema_embedded_types,
                "schema"
            )
            dbpedia_annotation_dict = self._map_embedded_header_to_ontology(
                header_embeddings,
                table_columns,
                self.dbpedia_ontology_dict,
                self.dbpedia_ontology_df["cleaned_label"].tolist(),
                self.dbpedia_embedded_types,
                "dbpedia"
            )
            
            return {**dbpedia_annotation_dict, **schema_annotation_dict}
        except Exception as e:
            self._logger.error(f"Encountered error on fasttext embedding: {e}")
            # If an error is encountered, the annotations are empty.
            return {
                f"dbpedia_embedding_column_types": {},
                f"schema_embedding_column_types": {},
                f"dbpedia_embedding_similarities": {},
                f"schema_embedding_similarities": {},
            }

    def _get_type_counts_and_similarities(
        self,
        table_metadata: typing.Dict,
        annotation_type_similarities: typing.Dict,
        ontology_annotations_key: str,
    ):
        column_types = table_metadata[f"{ontology_annotations_key}_column_types"]
        annotation_type_counts = collections.Counter(
            [column_types[col]["cleaned_label"]
             for col in column_types]
        )
        
        # Filter methods do not have similarities
        if f"{ontology_annotations_key}_similarities" in table_metadata:
            annotation_similarities = table_metadata[f"{ontology_annotations_key}_similarities"]
            for col in annotation_similarities:
                _type = table_metadata[f"{ontology_annotations_key}_column_types"][col]["cleaned_label"]
                similarity = annotation_similarities[col]
                if _type in annotation_type_similarities[ontology_annotations_key]:
                    annotation_type_similarities[ontology_annotations_key][_type].append(similarity)
                else:
                    annotation_type_similarities[ontology_annotations_key][_type] = [similarity]
        
        return annotation_type_counts, annotation_type_similarities


    def _map_embedded_header_to_ontology(
        self,
        header_embeddings: typing.List,
        table_columns: typing.List,
        ontology_dict: typing.Dict,
        ontology_types: typing.List,
        ontology_embeddings: typing.List,
        ontology_name: str,
    ):
        """
        Extract similarities between embedded table header
        and embedded types from an ontology.
        
        Returns
        -------
        annotation_dict
            Dictionary holding the annotation for each original column.
            An example for a table annotated with the schema ontology:
            {
                "schema_embedding_column_types": {
                    "StreetName": {
                        <schema annotation dictionary with type info>
                    },
                    "FirstName": {
                        <schema annotation dictionary with type info>
                    }
                }
            }
        """
        similarities = pd.DataFrame(
            self._get_embedding_similarities(header_embeddings, ontology_embeddings)
        )
        similarities.index = table_columns
        similarities.columns = ontology_types

        # Cosine similarity should at least be "similarity_threshold" to be annotated
        # 0.56: mean - 1*std of highest similarities on query segmentation subset
        similarity_threshold = 0.56
        thresholded_similarities = similarities[similarities > similarity_threshold]

        # Create column mappings based on most similar type for column (if exceeding threshold)
        annotation_mapping = thresholded_similarities.idxmax(axis=1).to_dict()
        annotation_similarities = thresholded_similarities.max(axis=1).dropna().tolist()

        original_annotated_columns = [
            table_columns[i]
            for i, (key, value) in enumerate(annotation_mapping.items()) if isinstance(value, str)
        ]
        annotated_columns = [
            value for key, value in annotation_mapping.items() if isinstance(value, str)
        ]

        annotation_dict = {
            f"{ontology_name}_embedding_column_types": {},
            f"{ontology_name}_embedding_similarities": {},
        }

        for i, col in enumerate(original_annotated_columns):
            annotation_dict[f"{ontology_name}_embedding_column_types"][col] = (
                ontology_dict[annotated_columns[i]]
            )
            annotation_dict[f"{ontology_name}_embedding_similarities"][col] = (
                np.round(annotation_similarities[i], 4)
            )
        
        return annotation_dict


    def _get_embedding_similarities(self, header_embeddings: list, ontology_embeddings: list):
        """Preprocess and calculate similarity between two lists of header vectors."""

        cosine_distances = spatial.distance.cdist(header_embeddings, ontology_embeddings, "cosine")
        cosine_similarities = 1 - cosine_distances

        return cosine_similarities


    def _calculate_dtype_percentages(self, table_dtypes: pd.Series) -> typing.Dict:
        """Calculate the percentage of each dtype in the table.

        Parameters
        ----------
        table_dtypes
            data types of each column in the table (a pandas DataFrame).

        Returns
        -------
        dtypes_percentages
            dictionary with the percentage of dtypes for each dtype present in the table.
        """
        dtypes_counter = collections.Counter(
            table_dtypes.to_list()
        )
        dtypes_percentages = dict(
            [
                (dtype, dtypes_counter[dtype]/len(table_dtypes.to_list()))
                for dtype in dtypes_counter
            ]
        )

        return dtypes_percentages


    def _classify_table_domain(self, annotation_dict: typing.Dict) -> str:
        """
        Classify a table to a domain. 
        The domain label corresponds to the majority vote of the domains of the column annotations.
        At tie: take the first occurring domain of the table - assuming order of importance on table attributes.
        'None' values (types without domain) are filtered out.

        Parameters
        ----------
        annotation_dict
            Dictionary with the annotated type (and corresponding ontology data) for each column.

        Returns
        -------
        table_domain
            Domain for the table.
        """        
        column_domains = list(
            itertools.chain(
                *[value["domain"] for key, value in annotation_dict.items() if value["domain"]]
            )
        )
        
        if len(column_domains) > 0:
            column_domains = [domain for domain in column_domains if domain]
            table_domain = max(set(column_domains), key=column_domains.count)
        else:
            table_domain = None

        return table_domain


    def _attach_metadata(self, original_metadata: typing.Dict, table_metadata: typing.Dict):
        """
        Enhance original table metadata with GitTables metadata.
        
        This metadata consists of:
        - Annotations with dbpedia and schema ontology.
        - Table statistics.
        - URL that the original CSV file was retrieved from.
        
        Parameters
        ----------
        enhanced_metadata
            Original metadata enhanced with encoded GitTables metadata.
        """
        table_metadata = json.dumps(
            table_metadata
        )
        table_metadata_key = "gittables"
        enhanced_metadata = {
            table_metadata_key.encode(): table_metadata.encode(),
            **original_metadata
        }

        return enhanced_metadata
