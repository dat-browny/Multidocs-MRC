from distutils.core import setup
from Cython.Build import cythonize
import os
import numpy


def check_skip(current_path):
    """
    Check if current file is in skip list.

    Args:
        current_path: path to file

    Returns: True or False

    """
    # List of files that you can not compile with cython. E.g. gpu_nms.c will be skipped
    skip_list = ["anchors", "bbox", "cpu_nms", "gpu_nms", "nms_kernel", "setup.py"]
    for check_key in skip_list:
        if current_path.find(check_key) != -1:
            return True
    return False


def get_list_of_files_python(dir_name):
    """
    Get list of python files

    Args:
        dir_name: name of directory

    Returns: list of all files inside dir_name

    """
    # create a list of file and sub directories
    # names in the given directory
    list_of_files = os.listdir(dir_name)
    all_files = list()
    # Iterate over all the entries
    for entry in list_of_files:
        # Create full path
        full_path = os.path.join(dir_name, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(full_path):
            all_files = all_files + get_list_of_files_python(full_path)
        else:
            if check_skip(full_path):
                continue
            if full_path.find(".py") != -1 and full_path.find(".pyc") == -1:
                all_files.append(full_path)
    return all_files


def get_list_of_files_except_so(dir_name, list_cant_remove):
    """
    Get list of all files except so file.

    Args:
        dir_name: name of directory

    Returns: list of all non-so files inside dir_name

    """
    # create a list of file and sub directories
    # names in the given directory
    list_of_file = os.listdir(dir_name)
    all_files = list()
    # Iterate over all the entries
    for entry in list_of_file:
        # Create full path3
        full_path = os.path.join(dir_name, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(full_path):
            all_files = all_files + get_list_of_files_except_so(
                full_path, list_cant_remove
            )
        else:
            if check_skip(full_path):
                continue
            flag_important = False
            for important_path in list_cant_remove:
                if important_path in full_path:
                    flag_important = True
                    break
            if flag_important:
                continue
            if full_path.find(".so") == -1 and full_path.find("weights") == -1:
                # print ("yes")
                os.remove(full_path)
                all_files.append(full_path)
    return all_files


# Note: ai_model_training_template is name of your module
# which is in the same directory with setup.py


allFiles = get_list_of_files_python("multi_document_mrc")
print(allFiles)
setup(
    ext_modules=cythonize(allFiles, compiler_directives={"language_level": "3"}),
    include_dirs=[numpy.get_include()],
)
list_cant_remove = [
    ".txt",
    ".jar",
    ".yml",
    ".sh",
    ".json",
    "gunicorn_conf.py",
    "start_batch_extractor.py",
]  # list files can't remove
allFilesDetele = get_list_of_files_except_so(
    "multi_document_mrc", list_cant_remove=list_cant_remove
)
print(allFilesDetele)
