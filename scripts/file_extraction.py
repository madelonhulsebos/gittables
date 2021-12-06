from gittables.file_extractor import GitHubFileExtractor

github_file_extractor = GitHubFileExtractor(
    "settings.toml", "logs", "table_collection"
)

github_file_extractor.set_topics()
github_file_extractor.extract_github_files()
