"""
This module facilitates the extraction of CSV files from GitHub.
All nouns from WordNet are used to build queries.
"""

import glob
import json
import logging
import os
import shutil
import time

import dynaconf
import nltk

# pylint: disable=wrong-import-position
nltk.download("wordnet")
from nltk.corpus import wordnet as wn
import numpy as np
import requests
from tqdm import tqdm


class GitHubFileExtractor:
    """File extractor class."""

    def __init__(self, settings_filepath: str, log_filepath: str, table_dir: str):

        github_username, github_token = self._get_github_settings(settings_filepath)

        self.session = requests.Session()
        self.session.auth = (github_username, github_token)
        self.allow_redirects = True

        self.table_dir = table_dir
        self.topics = []

        logging_filepath = f"{log_filepath}/extraction_logfile.log"
        if not os.path.exists(logging_filepath):
            open(logging_filepath, "w+")

        logging.basicConfig(filename=logging_filepath, filemode="a", level=logging.INFO)
        self._logger = logging.getLogger()

    def set_topics(self, custom_topics: list = None):
        """Get topics from WordNet to extract GitHub tables.
        CSV files will be searched on GitHub based on these topics.
        """
        topics_filepath = f"{self.table_dir}/github_topics.txt"
        if custom_topics is not None:
            topic_list = custom_topics
            self.topics = topic_list
            self._write_topics_to_file(topic_list, topics_filepath)
        else:
            if not os.path.exists(topics_filepath):
                synsets = pd.Series(list(wn.all_synsets("n"))).unique().tolist()

                for synset in synsets:
                    lemma = synset.lemma_names()[0]
                    lemma_clean = lemma.replace("_", " ").lower()
                    topic_list.append(lemma_clean)

                self.topics = topic_list
                self._write_topics_to_file(topic_list, topics_filepath)
            else:
                self._logger.info("Reading topics from existing topic list.")
                with open(topics_filepath, "r") as topic_file:
                    topic_list = json.load(topic_file)
                    self.topics = topic_list
                    topic_file.close()

        topics_metadata = (
            f"Number of topics to extract CSV files for is: {len(topic_list)}.\n"
        )
        self._write_github_metadata(topics_metadata, "a")

    def _write_topics_to_file(self, topic_list: list, filepath: str):
        """Write the topics used to query GitHub to a text file.

        Parameters
        ----------
        topic_list
            List of topics to save
        filepath
            Filepath to write the topics to.
        """
        with open(filepath, "w+") as topic_file:
            self.topics = topic_list
            json.dump(topic_list, topic_file)
            topic_file.close()

    def _write_github_metadata(self, metadata: str, write_mode: str):
        metadata_filepath = f"{self.table_dir}/github_metadata.txt"

        with open(metadata_filepath, write_mode) as f:
            f.write(metadata)
            f.close()

    def extract_github_files(self):
        """Extract CSV files from GitHub."""
        start = time.time()
        num_csvs = 0
        for num_topic, topic in enumerate(self.topics):
            try:
                topic_str = topic.replace(" ", "_")
                topic_dir = f"{self.table_dir}/{topic_str}"
                if os.path.exists(topic_dir):
                    shutil.rmtree(topic_dir)
                os.makedirs(topic_dir)

                self._logger.info(
                    "Extracting CSVs from #topic %s: %s", num_topic, topic
                )

                raw_urls, urls = self._get_raw_file_urls_from_response(topic, topic_dir)
                num_topic_csvs = self._write_url_contents_to_csv_files(
                    topic_dir, raw_urls, urls
                )

                num_csvs = num_csvs + num_topic_csvs

                self._logger.info(
                    "Extracted %s CSV files for #topic %s and %s CSV files in total.",
                    num_topic_csvs,
                    topic,
                    num_csvs,
                )

            except Exception as exception:
                self._logger.info(
                    "Error message for topic #%s was: %s", num_topic, exception
                )
                continue

        end = time.time()
        # pylint: disable=undefined-loop-variable
        self._logger.info(
            "Extracted CSV urls for %s topics in %s seconds", num_topic, end - start
        )

    def _get_github_settings(self, settings_filepath: str):
        """Read and return GitHub username and token.

        settings_filepath
            Filepath where github username and token can be found and loaded with dynaconf.
        """
        settings = dynaconf.Dynaconf(settings_files=[settings_filepath])

        if not (settings.exists("github_username") and settings.exists("github_token")):
            msg = "No username and token from GitHub were found."
            self._logger.info(msg)
            raise ValueError(msg)

        return settings.github_username, settings.github_token

    def _get_raw_file_urls_from_response(self, topic: str, topic_dir: str):
        """Get relevant items from response, specifically
        the total file count and raw url references.

        Parameters
        ----------
        topic
            Topic to query CSV files from GitHub for.
        topic_dir
            Directory of topic to write metadata to.
        """
        # Per search max. 1K results will be returned. Query should be segmented accordingly.
        # At most 100 per page can be retrieved per request.
        raw_urls = []
        urls = []

        query = f"https://api.github.com/search/code?q={topic}+in:file+extension:csv&per_page=100"
        response = self.session.get(query)
        if response.status_code == 200:

            response_limit = 1000
            url_count = response.json()["total_count"]

            topic_query_metadata = (
                f"URL count from original query of topic {topic} is {url_count}.\n"
            )
            self._write_github_metadata(topic_query_metadata, "a")

            if url_count > response_limit:
                raw_urls, urls = self._segment_query(topic, topic_dir, url_count, raw_urls, urls)
            else:
                raw_urls, urls = self._traverse_through_url_pages(
                    response, topic_dir, raw_urls, urls
                )
            num_raw_urls = len(raw_urls)
            raw_url_msg = f"Retrieved {num_raw_urls} raw urls for topic {topic}."
            self._logger.info(raw_url_msg)

        else:
            self._logger.info(response.json())
            self._log_response_and_wait(response)

        return raw_urls, urls

    def _generate_size_sequence(
        self, lower_quartile: float, upper_quartile: float, total_count: int
    ):
        """Generate a sequence of sizes based on prior statistics and the original total count.
        This sequence is expected to inform a query to return at most 1000 urls.

        Parameters
        ----------
        lower_quartile
            Lower quartile bound file size.
        upper_quartile
            Upper quartile bound file size.
        total_count
            Total number of items of the respective response.
        """
        step_size = np.max([(upper_quartile - lower_quartile) / (0.25 * total_count / 1000), 5])

        size_sequence = np.arange(
            start=lower_quartile, stop=upper_quartile, step=step_size, dtype="int32"
        )

        return size_sequence, step_size

    def _segment_query(self, topic: str, topic_dir: str, url_count: int, raw_urls: list, urls: list):
        """Segment the query into queries that are expected to yield less urls than the limit.

        topic
            Topic for the search query.
        url_count
            Original count of the urls pointing to CSV files.
        """
        # These size ranges were informed by 1000 CSV files from GitHub.
        split_1 = 256  # min
        split_2 = 2750  # 1st quartile
        split_3 = 4756  # 2nd quartile
        split_4 = 23459  # 3rd quartile
        split_5 = 53531  # max

        responses = []
        step_sizes = []
        url_counts = []
        number_queries = []
        for size_limits in [
            [split_1, split_2],
            [split_2, split_3],
            [split_3, split_4],
            [split_4, split_5],
        ]:
            size_sequence, step_size = self._generate_size_sequence(
                size_limits[0], size_limits[1], url_count
            )

            step_sizes.append(step_size)
            for i, lower_size_limit in enumerate(size_sequence):
                if i == (len(size_sequence) - 1):
                    # The end of the sequence was reached
                    continue
                upper_size_limit = size_sequence[i + 1]
                segmented_query = (
                    f"https://api.github.com/search/code?q='{topic}'+size:"
                    f"{lower_size_limit}..{upper_size_limit}+extension:csv"
                    "&per_page=100"
                )

                response = self.session.get(segmented_query)
                if response.status_code == 200:
                    # responses.append(response)
                    url_counts.append(response.json()["total_count"])
                    raw_urls, urls = self._traverse_through_url_pages(
                        response, topic_dir, raw_urls, urls
                    )
                else:
                    self._logger.info(response.json())
                    self._log_response_and_wait(response)

            number_queries.append(len(size_sequence))

        number_queries = np.sum(number_queries)
        mean_response_sizes = np.mean(url_counts)
        std_response_sizes = np.std(url_counts)

        self._logger.info(
            """
            The original query for topic %s of size %s is segmented into %s queries,
            with an average and std response size of %s and %s, and stepsizes of %s.
            """,
            topic,
            url_count,
            number_queries,
            mean_response_sizes,
            std_response_sizes,
            step_sizes,
        )

        return raw_urls, urls

    def _traverse_through_url_pages(
        self, response, topic_dir: str, raw_urls: list, urls: list
    ):
        """Traverse through response page by page to extract urls.

        response
            Response to traverse through, expected to have approx. 1K items.
        topic_dir
            Directory of the current topic in which metadata file should be placed.
        raw_urls
            List of urls pointing to raw content to extend.
        urls
            List of urls to extend.
        """
        with open(f"{topic_dir}/topic_query_urls.txt", "a") as topic_metadata_file:
            request_url = response.request.url
            topic_metadata_file.write(request_url + "\n")

        raw_urls, urls = self._add_urls_from_response(response, raw_urls, urls)
        while "next" in response.links.keys():
            try:
                old_response = response
                # The response captures only the 'last' link hence should be overwritten.
                response = self.session.get(
                    response.links["next"]["url"],
                )
                if response.status_code == 200:
                    raw_urls, urls = self._add_urls_from_response(
                        response, raw_urls, urls
                    )
                else:
                    self._log_response_and_wait(response)
                    # The old response contains the next pages.
                    response = old_response
            except Exception as exception:
                self._logger.error(
                    "Error message on extracting urls from response was: %s", exception
                )
                continue

        return raw_urls, urls

    def _add_urls_from_response(self, response, raw_urls, urls):
        """Extract urls (raw and plain) from response and add to existing lists.

        response
            Response to extract urls from.
        raw_urls
            List of raw content urls.
        urls
            List of urls.
        """
        items = response.json()["items"]
        raw_urls = raw_urls + [item["html_url"] + "?raw=true" for item in items]
        urls = urls + [item["html_url"] for item in items]

        return raw_urls, urls

    def _write_url_contents_to_csv_files(
        self, topic_dir: str, raw_urls: list, urls: list
    ):
        """Extract raw contents from GitHub URLs and write to CSV files.

        Parameters
        ----------
        topic_dir
            Topic directory in which the CSV files should be written,
            ollected in a csv_files directory.
        raw_urls
            URLs to raw content to write to CSV file.
        urls
            URLs to file on GitHub for later reference.
        """
        topic_tables_dir = f"{topic_dir}/csv_files"
        if not os.path.exists(topic_tables_dir):
            os.makedirs(topic_tables_dir)

        start = time.time()
        num_csvs = 0
        for num_raw_url, raw_url in enumerate(raw_urls):
            try:
                url = urls[num_raw_url]
                file_name = url.split("/")[-1]
                file_path = os.path.join(topic_tables_dir, file_name)

                if os.path.exists(file_path):
                    file_name_wo_extension = file_name.split(".csv")[0]
                    filename_count = len(
                        glob.glob1(
                            topic_tables_dir, f"{file_name_wo_extension}*.csv"
                        )
                    )
                    file_name = (
                        file_name_wo_extension + "_" + str(filename_count) + ".csv"
                    )
                    file_path = os.path.join(topic_tables_dir, file_name)

                response = self.session.get(raw_url)
                if response.status_code == 200:

                    raw_content = response.content

                    with open(file_path, "wb+") as content_file:
                        content_file.write(raw_content)
                        content_file.close()

                    with open(f"{topic_dir}/topic_csv_urls.txt", "a") as topic_metadata_file:
                        topic_metadata_file.write(url + "\n")

                    num_csvs += 1

                    if num_csvs % 2500 == 0:
                        end = time.time()
                        self._logger.info(
                            "Reading the content from %s urls took %s seconds.",
                            num_csvs,
                            end - start,
                        )

                else:
                    self._log_response_and_wait(response)
                    self._logger.info(response.json())
            except Exception as exception:
                self._logger.error(
                    "Error messsage for writing content of url #%s to file: %s",
                    num_raw_url,
                    exception,
                )
                continue

        return num_csvs

    def _log_response_and_wait(self, response):
        """Log the response status code and repeat the request after the set time.

        response
            response from GitHub API.
        """
        headers = response.headers
        if "Retry-After" in headers:
            wait_time = int(headers["Retry-After"])
        if "X-RateLimit-Reset" in headers:
            # Overwrite waiting time if the rate limit was hit.
            wait_time = int(headers["X-RateLimit-Reset"]) - time.time()
        else:
            wait_time = 60

        wait_time = max([0, wait_time])

        self._logger.error(
            "Received response code %s will wait %s s.", response.status_code, wait_time
        )

        time.sleep(wait_time)
