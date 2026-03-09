"""
This module facilitates the extraction of CSV files from GitHub.
All nouns from WordNet are used to build queries.

Modifications:
- Filters repositories by research-friendly licenses (MIT, Apache 2.0, CC-BY, etc.)
- Filters OUT CSV files that appear to be LLM/NLP datasets (tokens, n-grams, embeddings, etc.)
"""

import csv
import glob
import io
import json
import logging
import os
import shutil
import time
from typing import Dict, Optional

import nltk

# pylint: disable=wrong-import-position
nltk.download("wordnet")
from nltk.corpus import wordnet as wn
import numpy as np
import requests
from tqdm import tqdm

from gittables import utils


# ── License & column filter config ────────────────────────────────────────────

# SPDX identifiers accepted for research use.
# Extend or restrict this list to match your institution's policy.
RESEARCH_FRIENDLY_LICENSES = {
    "mit",
    "apache-2.0",
    "gpl-2.0",
    "gpl-3.0",
    "lgpl-2.1",
    "lgpl-3.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "isc",
    "mpl-2.0",
    "eupl-1.2",
    "agpl-3.0",
}

# Column name fragments that suggest an LLM / NLP dataset.
# A CSV whose header contains ANY of these substrings (case-insensitive) is skipped.
LLM_COLUMN_KEYWORDS = {
    "token",
    "ngram",
    "n_gram",
    "n-gram",
    "unigram",
    "bigram",
    "trigram",
    "embedding",
    "vector",
    "logit",
    "perplexity",
    "vocab",
    "vocabulary",
    "bpe",
    "subword",
    "wordpiece",
    "sentencepiece",
    "attention",
    "transformer",
    "bert",
    "gpt",
    "llm",
    "prompt",
    "completion",
    "fine_tun",      # covers fine_tune, fine_tuning
    "finetun",
    "pretrain",
    "pre_train",
    "language_model",
    "corpus_id",
    "doc_freq",
    "term_freq",
    "tf_idf",
    "tfidf",
    "word_freq",
    "pos_tag",
    "ner_tag",
    "lemma",
    "stem",
    "topic_id"
}


def _is_llm_csv(raw_content: bytes, sample_rows: int = 3) -> bool:
    """Return True if the CSV looks like an LLM / NLP dataset.

    Checks are performed on:
    1. Column headers — any LLM_COLUMN_KEYWORDS substring match triggers rejection.
    2. First `sample_rows` data rows — rejects if values look like space-separated
       token sequences (heuristic: avg words-per-cell > 6 in a text column).

    Parameters
    ----------
    raw_content
        Raw bytes of the downloaded CSV.
    sample_rows
        Number of data rows to inspect beyond the header.
    """
    try:
        text = raw_content.decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text))
        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if i > sample_rows:
                break

        if not rows:
            return False

        header = [col.strip().lower() for col in rows[0]]

        # 1. Header keyword check
        for col in header:
            for kw in LLM_COLUMN_KEYWORDS:
                if kw in col:
                    return True

        # 2. Heuristic: a column whose cells look like token sequences
        if len(rows) > 1:
            data_rows = rows[1:]
            for col_idx in range(len(header)):
                values = []
                for row in data_rows:
                    if col_idx < len(row):
                        values.append(row[col_idx].strip())
                if not values:
                    continue
                avg_words = np.mean([len(v.split()) for v in values if v])
                if avg_words > 6:
                    return True

    except Exception:
        # If we can't parse it, don't reject it on suspicion alone
        pass

    return False


# ── Main extractor ─────────────────────────────────────────────────────────────

class GitHubFileExtractor:
    """File extractor class.

    Two new keyword arguments compared to the original:
    - filter_license : bool  (default True)  — skip repos without a research-friendly license
    - filter_llm_cols: bool  (default True)  — skip CSVs that look like LLM/NLP datasets
    """

    def __init__(
        self,
        settings_filepath: str,
        log_filepath: str,
        table_dir: str,
        filter_license: bool = True,
        filter_llm_cols: bool = True,
    ):
        github_username, github_token = utils.get_github_settings(settings_filepath)

        self.session = requests.Session()
        self.session.auth = (github_username, github_token)
        self.allow_redirects = True

        self.table_dir = table_dir
        self.topics = []
        self.filter_license = filter_license
        self.filter_llm_cols = filter_llm_cols

        os.makedirs(log_filepath, exist_ok=True)

        logging_filepath = f"{log_filepath}/extraction_logfile.log"
        if not os.path.exists(logging_filepath):
            open(logging_filepath, "w+").close()

        logging.basicConfig(filename=logging_filepath, filemode="a", level=logging.INFO)
        self._logger = logging.getLogger()

    # ── License helpers ────────────────────────────────────────────────────────

    def _get_repo_license(self, repo_full_name: str) -> Optional[str]:
        """Return the SPDX license key for a repo, or None if unlicensed / unknown.

        Parameters
        ----------
        repo_full_name
            GitHub repo in ``owner/repo`` format.
        """
        url = f"https://api.github.com/repos/{repo_full_name}/license"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("license", {}).get("spdx_id", "").lower() or None
            # 404 → no license file in repo
        except Exception as exc:
            self._logger.warning("Could not fetch license for %s: %s", repo_full_name, exc)
        return None

    def _repo_has_research_license(self, repo_full_name: str) -> bool:
        """Return True only when the repo carries a research-friendly license."""
        spdx = self._get_repo_license(repo_full_name)
        if spdx is None:
            return False
        return spdx in RESEARCH_FRIENDLY_LICENSES

    # ── Topic setup (unchanged from original) ─────────────────────────────────

    def set_topics(self, custom_topics: list = None):
        """Get topics from WordNet to extract GitHub tables.
        CSV files will be searched on GitHub based on these topics.
        """
        os.makedirs(self.table_dir, exist_ok=True)

        topics_filepath = f"{self.table_dir}/github_topics.txt"
        topic_list = []

        if custom_topics is not None:
            topic_list = custom_topics
            self.topics = topic_list
            self._write_topics_to_file(topic_list, topics_filepath)
        else:
            if not os.path.exists(topics_filepath):
                synsets = list(set(list(wn.all_synsets("n"))))

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
        if not os.path.exists(filepath):
            open(filepath, "w+").close()
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
        # The GitTables token starts at end of topic list to avoid conflict
        for num_topic, topic in enumerate(self.topics[::-1]):
            try:
                topic_str = topic.replace(" ", "_")
                topic_dir = f"{self.table_dir}/{topic_str}"
                if os.path.exists(topic_dir):
                    shutil.rmtree(topic_dir)
                os.makedirs(topic_dir)

                self._logger.info(
                    "Extracting CSVs from #topic %s: %s", num_topic, topic
                )

                raw_urls, urls, repo_names = self._get_raw_file_urls_from_response(topic, topic_dir)
                num_topic_csvs = self._write_url_contents_to_csv_files(
                    topic_dir, raw_urls, urls, repo_names
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
        repo_names = []

        query = f"https://api.github.com/search/code?q={topic}+in:file+extension:csv&per_page=100"
        self._search_throttle()
        response = self.session.get(query)

        if response.status_code == 200:

            response_limit = 1000
            url_count = response.json()["total_count"]

            topic_query_metadata = (
                f"URL count from original query of topic {topic} is {url_count}.\n"
            )
            self._write_github_metadata(topic_query_metadata, "a")

            if url_count > response_limit:
                raw_urls, urls, repo_names = self._segment_query(
                    topic, topic_dir, url_count, raw_urls, urls, repo_names
                )
            else:
                raw_urls, urls, repo_names = self._traverse_through_url_pages(
                    response, topic_dir, raw_urls, urls, repo_names
                )
            num_raw_urls = len(raw_urls)
            raw_url_msg = f"Retrieved {num_raw_urls} raw urls for topic {topic}."
            self._logger.info(raw_url_msg)

        else:
            self._logger.info(response.json())
            self._log_response_and_wait(response)

        return raw_urls, urls, repo_names

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

    def _segment_query(self, topic, topic_dir, url_count, raw_urls, urls, repo_names):
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
                self._search_throttle()
                response = self.session.get(segmented_query)
                if response.status_code == 200:
                    url_counts.append(response.json()["total_count"])
                    raw_urls, urls, repo_names = self._traverse_through_url_pages(
                        response, topic_dir, raw_urls, urls, repo_names
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

        return raw_urls, urls, repo_names

    def _traverse_through_url_pages(
        self, response, topic_dir: str, raw_urls: list, urls: list, repo_names
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

        raw_urls, urls, repo_names = self._add_urls_from_response(response, raw_urls, urls, repo_names)
        while "next" in response.links:
            try:
                old_response = response
                self._search_throttle()
                # The response captures only the 'last' link hence should be overwritten.
                response = self.session.get(
                    response.links["next"]["url"],
                )
                if response.status_code == 200:
                    raw_urls, urls, repo_names = self._add_urls_from_response(
                        response, raw_urls, urls, repo_names
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

        return raw_urls, urls, repo_names

    @staticmethod
    def _to_raw_url(html_url: str) -> str:
        """Convert a GitHub blob URL to a raw.githubusercontent.com URL.

        Example:
          https://github.com/owner/repo/blob/abc123/path/file.csv
          → https://raw.githubusercontent.com/owner/repo/abc123/path/file.csv

        This avoids the ?raw=true redirect through the web UI, which is
        aggressively rate-limited for programmatic access.
        """
        return (
            html_url
            .replace("https://github.com/", "https://raw.githubusercontent.com/")
            .replace("/blob/", "/")
        )

    def _add_urls_from_response(self, response, raw_urls, urls, repo_names):
        """Extract urls (raw and plain) from response and add to existing lists.

        response
            Response to extract urls from.
        raw_urls
            List of raw content urls.
        urls
            List of urls.
        """
        items = response.json()["items"]
        raw_urls   += [self._to_raw_url(item["html_url"]) for item in items]
        urls       += [item["html_url"] for item in items]
        repo_names += [item["repository"]["full_name"] for item in items]
        return raw_urls, urls, repo_names

    def _write_url_contents_to_csv_files(
        self, topic_dir: str, raw_urls: list, urls: list, repo_names: list
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
        os.makedirs(topic_tables_dir, exist_ok=True)

        start = time.time()
        num_csvs = 0

        # Cache license lookups so each repo is only queried once per session.
        license_cache: Dict[str, bool] = {}

        skipped_license = 0
        skipped_llm = 0

        for num_raw_url, raw_url in enumerate(raw_urls):
            try:
                url = urls[num_raw_url]
                repo_full_name = repo_names[num_raw_url]

                # ── 1. License filter ──────────────────────────────────────
                if self.filter_license:
                    if repo_full_name not in license_cache:
                        license_cache[repo_full_name] = self._repo_has_research_license(
                            repo_full_name
                        )
                    if not license_cache[repo_full_name]:
                        skipped_license += 1
                        self._logger.debug(
                            "Skipped (license) repo %s", repo_full_name
                        )
                        continue

                # ── 2. Download with exponential backoff ───────────────────
                raw_content = self._fetch_with_retry(raw_url)
                if raw_content is None:
                    self._logger.warning("Giving up on %s after retries.", url)
                    continue

                # ── 3. LLM-column filter ───────────────────────────────────
                if self.filter_llm_cols and _is_llm_csv(raw_content):
                    skipped_llm += 1
                    self._logger.debug("Skipped (LLM cols) %s", url)
                    continue

                # ── 4. Persist ─────────────────────────────────────────────
                file_name = url.split("/")[-1]
                file_path = os.path.join(topic_tables_dir, file_name)

                if os.path.exists(file_path):
                    stem = file_name.split(".csv")[0]
                    count = len(glob.glob1(topic_tables_dir, f"{stem}*.csv"))
                    file_name = f"{stem}_{count}.csv"
                    file_path = os.path.join(topic_tables_dir, file_name)

                with open(file_path, "wb+") as content_file:
                    content_file.write(raw_content)

                with open(f"{topic_dir}/topic_csv_urls.txt", "a") as f:
                    f.write(url + "\n")

                num_csvs += 1

                if num_csvs % 2500 == 0:
                    elapsed = time.time() - start
                    self._logger.info(
                        "Saved %s CSVs in %.1f s (skipped: %s no-license, %s llm-cols).",
                        num_csvs, elapsed, skipped_license, skipped_llm,
                    )

            except Exception as exc:
                self._logger.error(
                    "Error writing content of url #%s: %s", num_raw_url, exc
                )
                continue

        self._logger.info(
            "Topic done — saved %s CSVs, skipped %s (license), %s (LLM columns).",
            num_csvs, skipped_license, skipped_llm,
        )
        return num_csvs

    def _fetch_with_retry(self, url: str, max_retries: int = 5) -> Optional[bytes]:
        """GET a URL and return the raw bytes, retrying on 429/403 with exponential backoff.

        Returns None if all retries are exhausted.

        Backoff schedule (seconds): 60, 120, 240, 480, 960
        This handles the headerless 429s GitHub's CDN sends for raw file downloads.
        """
        for attempt in range(max_retries):
            response = self.session.get(url)
            if response.status_code == 200:
                return response.content

            if response.status_code in (429, 403):
                backoff = 60 * (2 ** attempt)   # 60 → 120 → 240 → 480 → 960
                headers = response.headers

                # Prefer the reset-epoch header if present
                if "X-RateLimit-Reset" in headers:
                    try:
                        reset_at = float(headers["X-RateLimit-Reset"])
                        backoff  = max(backoff, reset_at - time.time() + 5)
                    except ValueError:
                        pass

                backoff = min(backoff, 1200.0)  # cap at 20 min
                self._logger.error(
                    "Download rate-limited (%s) on attempt %d/%d for %s — "
                    "backing off %.0f s.",
                    response.status_code, attempt + 1, max_retries, url, backoff,
                )
                time.sleep(backoff)
            else:
                # Non-retryable error (404, 500, …)
                self._logger.warning(
                    "Non-retryable status %s for %s.", response.status_code, url
                )
                return None

        self._logger.error("Exhausted %d retries for %s.", max_retries, url)
        return None
    # Enforce a minimum 6.5 s gap between search calls to stay under the ceiling
    # proactively, without relying solely on 429 reactions.
    _SEARCH_MIN_INTERVAL: float = 6.5   # seconds between consecutive search requests
    _last_search_ts: float = 0.0        # timestamp of the last search request

    def _search_throttle(self) -> None:
        """Block until it is safe to fire the next code-search request.

        Enforces a minimum inter-request gap so we stay under the 10 req/min
        ceiling without needing to react to 429s.
        """
        elapsed = time.time() - self._last_search_ts
        gap = self._SEARCH_MIN_INTERVAL - elapsed
        if gap > 0:
            self._logger.debug("Rate-limit throttle: sleeping %.2f s", gap)
            time.sleep(gap)
        self._last_search_ts = time.time()

    def _log_response_and_wait(self, response) -> None:
        """Log a non-200 response and sleep until it is safe to retry.

        Decision tree for wait time:
        1. ``Retry-After`` header      — seconds to wait (GitHub sets this on 429)
        2. ``X-RateLimit-Reset`` header — absolute UTC epoch when the window reopens;
                                          we sleep until that moment + 5 s buffer.
                                          Your 403 logs show this value is always present
                                          and correctly marks the next-minute boundary.
        3. Status-code fallback        — 403 → 65 s (just over one minute window),
                                          429 → 90 s,  anything else → 60 s.
        """
        headers = response.headers
        status  = response.status_code

        wait_time: float

        if "Retry-After" in headers:
            try:
                wait_time = float(headers["Retry-After"]) + 2
            except ValueError:
                wait_time = 65.0

        elif "X-RateLimit-Reset" in headers:
            try:
                reset_at  = float(headers["X-RateLimit-Reset"])
                # Sleep until the reset epoch, plus a 5-second safety buffer.
                wait_time = max(0.0, reset_at - time.time()) + 5
            except ValueError:
                wait_time = 65.0

        else:
            # No headers — GitHub CDN 429 on raw file downloads falls here.
            # Use exponential-backoff-style defaults per status code.
            if status == 403:
                wait_time = 65.0
            elif status == 429:
                wait_time = 90.0
            else:
                wait_time = 60.0

        # Hard clamp: never less than 10 s, never more than 20 min.
        wait_time = max(10.0, min(wait_time, 1200.0))

        self._logger.error(
            "Received response code %s — waiting %.0f s before retrying. "
            "(Retry-After=%s, X-RateLimit-Reset=%s)",
            status,
            wait_time,
            headers.get("Retry-After", "n/a"),
            headers.get("X-RateLimit-Reset", "n/a"),
        )
        time.sleep(wait_time)
        # Reset the search-throttle clock so the very next search request
        # is not additionally delayed after already waiting for the window.
        self._last_search_ts = 0.0
