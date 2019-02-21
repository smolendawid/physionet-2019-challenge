import os


def project_root() -> str:
    """Returns path to root directory of a project"""
    return os.path.join(_file_directory_path(__file__), '..')


def _file_directory_path(file_path_name):
    return os.path.dirname(os.path.realpath(file_path_name))
