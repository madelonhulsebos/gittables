import dynaconf


def get_github_settings(settings_filepath: str):
    """Read and return GitHub username and token.

    settings_filepath
        Filepath where github username and token can be found and loaded with dynaconf.
    """
    settings = dynaconf.Dynaconf(settings_files=[settings_filepath])

    if not (settings.exists("github_username") and settings.exists("github_token")):
        msg = "No username and token from GitHub were found."
        raise ValueError(msg)

    return settings.github_username, settings.github_token