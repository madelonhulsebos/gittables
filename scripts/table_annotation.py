import datetime
import json
import os
import random
import shutil
import typing

import numpy as np

from fasttext.util import util

from gittables import ontology
from gittables import table_annotator


def _sample_urls(urls: typing.List, max_sample_size: int):
    if len(urls) > max_sample_size:
        random.seed(RANDOM_STATE)
        sample_urls = random.sample(np.arange(0, len(urls)).tolist(), k=max_sample_size)

        table_url_sample = [urls[i] for i in sample_urls]
    else:
        table_url_sample = urls

    print(f"Number urls sampled: {len(table_url_sample)}")

    return table_url_sample


# TODO: this function is not available from the pypi version of fasttext.
# The package needs to be downloaded from the github repository.
util.download_model('en', if_exists='ignore')

RANDOM_STATE = 0

ontology_date = "20210528" #datetime.date.today().strftime("%Y%m%d")
ontology_dir = "ontologies/"
logs_dir = "logs/"
settings_filepath = "settings.toml"
table_collection_dir = "table_collection/"

if not os.path.exists(os.path.join(ontology_dir, f"schema_{ontology_date}.pickle")):
    ontology.build_schema_ontology(ontology_dir)
    ontology.build_dbpedia_ontology(ontology_dir)

with open(f"{table_collection_dir}github_topics.txt") as topic_file:
    topic_list = json.load(topic_file)

print(f"Annotating the following topics: {topic_list}")

for topic in topic_list:
    filenames_to_url = {}
    
    tables_dir = f"{table_collection_dir}{topic}/tables_licensed"
    if os.path.exists(tables_dir):
        continue
        #shutil.rmtree(tables_dir)
    if not os.path.exists(f"{table_collection_dir}{topic}/topic_csv_urls.txt"):
        continue
    with open(f"{table_collection_dir}{topic}/topic_csv_urls.txt") as f:
        urls = f.read().splitlines()
        for url in urls:
            filename = url.split("/")[-1]
            filename_wo_extension = filename[:-4]

            if filename in filenames_to_url:
                # The first one has no _number but is in count, hence -1, you would say but it was not implemented like that, so we start counting at 1.
                filename_count = len([key for key in filenames_to_url if key.startswith(filename_wo_extension)])
                filename = filename_wo_extension + "_" + str(filename_count) + ".csv"
            filenames_to_url[filename] = url
    with open(f"{table_collection_dir}{topic}/filenames_to_url.json", "w+") as f:
        json.dump(filenames_to_url, f)
    
    if len(filenames_to_url) == 0:
        # No csv files were extracted for this topic.
        continue

    topic_tables_annotator = table_annotator.TopicTablesProcessor(topic, logs_dir, table_collection_dir, ontology_dir, ontology_date)
    topic_tables_metadata = topic_tables_annotator.process_topic_tables(filenames_to_url, topic, settings_filepath)

    with open(f"{table_collection_dir}{topic}/tables_licensed_metadata.json", "w+") as f:
        json.dump(topic_tables_metadata, f)
