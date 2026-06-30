# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import fnmatch
import glob
import logging
import os
from typing import Iterator, List, Optional, Tuple, Union

from meta_action.utils.type_check import (
    is_path_exists,
    is_path_exists_or_creatable,
    isfolder,
    isinteger,
    islist,
    islogical,
    isstring,
)

logger = logging.getLogger(__name__)


def safe_path(input_path: str, warning: bool = True, debug: bool = True) -> str:
    """Convert path to a valid OS format, e.g., empty string '' to '.',
    remove redundant '/' at the end from 'aa/' to 'aa'

    parameters:
        input_path:     a string

    outputs:
        safe_data:      a valid path in OS format
    """
    if debug and not isstring(input_path):
        raise TypeError(f"path is not a string: {input_path}")
    safe_data = copy.copy(input_path)
    safe_data = os.path.normpath(safe_data)
    return safe_data


def fileparts(input_path: str, warning: bool = True, debug: bool = True) -> Tuple[str, str, str]:
    """This function return a tuple, which contains (directory, filename, extension)
    if the file has multiple extension, only last one will be displayed

    parameters:
        input_path:     a string path

    outputs:
        directory:      the parent directory
        filename:       the file name without extension
        ext:            the extension
    """
    good_path = safe_path(input_path, debug=debug)
    if len(good_path) == 0:
        return ("", "", "")
    if good_path[-1] == "/":
        if len(good_path) > 1:
            return (good_path[:-1], "", "")  # ignore the final '/'
        else:
            return (good_path, "", "")  # ignore the final '/'

    directory = os.path.dirname(os.path.abspath(good_path))
    filename = os.path.splitext(os.path.basename(good_path))[0]
    ext = os.path.splitext(good_path)[1]
    return (directory, filename, ext)


def mkdir_if_missing(input_path: str, warning: bool = True, debug: bool = True) -> None:
    """Create a directory if not existing:
        1. if the input is a path of file, then create the parent directory
           of this file
        2. if the root directory does not exists for the input, then create
           all the root directories recursively until the parent directory of
           input exists

    parameters:
        input_path:     a string path
    """
    good_path = safe_path(input_path, warning=warning, debug=debug)
    if debug and not is_path_exists_or_creatable(good_path):
        raise ValueError(f"input path is not valid or creatable: {good_path}")
    dirname, _, _ = fileparts(good_path)
    if not is_path_exists(dirname):
        mkdir_if_missing(dirname)
    if isfolder(good_path) and not is_path_exists(good_path):
        os.mkdir(good_path)


def load_list_from_folder(
    folder_path: str,
    ext_filter: Optional[Union[str, List[str]]] = None,
    depth: Optional[int] = 1,
    recursive: bool = False,
    sort: bool = True,
    save_path: Optional[str] = None,
    debug: bool = True,
) -> Tuple[List[str], int]:
    """Load a list of files or folders from a system path

    parameters:
        folder_path:    root to search
        ext_filter:     a string to represent the extension of files interested
        depth:          maximum depth of folder to search.
                        when it's None, all levels of folders will be searched
        recursive:      False: only return current level
                        True: return all levels up to the input depth

    outputs:
        fulllist:       a list of elements
        num_elem:       number of the elements
    """
    folder_path = safe_path(folder_path)
    if debug and not isfolder(folder_path):
        raise ValueError(f"input folder path is not correct: {folder_path}")
    if not is_path_exists(folder_path):
        logger.warning("the input folder does not exist %s", folder_path)
        return [], 0
    if debug:
        if not islogical(recursive):
            raise TypeError(f"recursive should be a logical variable: {recursive}")
        if not (depth is None or (isinteger(depth) and depth >= 1)):
            raise ValueError(f"input depth is not correct {depth}")
        if not (
            ext_filter is None
            or (islist(ext_filter) and all(isstring(ext_tmp) for ext_tmp in ext_filter))
            or isstring(ext_filter)
        ):
            raise TypeError("extension filter is not correct")
    ext_filter_list: Optional[List[str]]
    if isstring(ext_filter):
        ext_filter_list = [str(ext_filter)]  # convert to a list
    elif ext_filter is None:
        ext_filter_list = None
    else:
        ext_filter_list = [str(ext_tmp) for ext_tmp in ext_filter]

    fulllist: List[str] = []
    if depth is None:  # find all files recursively
        recursive = True

        def _is_hidden_name(path_part: str) -> bool:
            return len(path_part) > 0 and path_part[0] == "."

        def _walk_items(top: str) -> Iterator[Tuple[str, List[str]]]:
            """Yield (path, direct-children-names) in glob2-compatible order."""
            try:
                names = os.listdir(top)
            except OSError:
                return
            items = list(names)
            yield top, items
            for name in items:
                new_path = os.path.join(top, name)
                # Keep glob2 behavior: recurse into non-symlinks only.
                if not os.path.islink(new_path):
                    yield from _walk_items(new_path)

        def _glob2_double_star_relative_names(base_dir: str, include_root: bool) -> List[str]:
            """Return relative names emitted by glob2 for pattern '**'."""
            names: List[str] = [""] if include_root else []
            for top, entries in _walk_items(base_dir):
                rel_top = top[len(base_dir) + 1 :]
                for entry in entries:
                    names.append(os.path.join(rel_top, entry))
            # Match glob2 hidden filtering semantics:
            # remove names starting with '.' at the beginning of the relative path.
            return [name for name in names if (not name) or (not _is_hidden_name(name))]

        if ext_filter_list is not None:
            dir_candidates_rel = _glob2_double_star_relative_names(folder_path, include_root=True)
            for ext_tmp in ext_filter_list:
                pattern = f"*{ext_tmp}"
                curlist = []
                for rel_dir in dir_candidates_rel:
                    abs_dir = folder_path if rel_dir == "" else os.path.join(folder_path, rel_dir)
                    try:
                        names = os.listdir(abs_dir)
                    except OSError:
                        continue
                    names = [name for name in names if not _is_hidden_name(name)]
                    for name in names:
                        if fnmatch.fnmatchcase(name, pattern):
                            curlist.append(os.path.join(abs_dir, name))
                if sort:
                    curlist = sorted(curlist)
                fulllist += curlist
        else:
            rel_names = _glob2_double_star_relative_names(folder_path, include_root=False)
            curlist = [os.path.join(folder_path, rel_name) for rel_name in rel_names]
            if sort:
                curlist = sorted(curlist)
            fulllist += curlist
    else:  # find files based on depth and recursive flag
        wildcard_prefix = "*"
        for index in range(depth - 1):
            wildcard_prefix = os.path.join(wildcard_prefix, "*")
        if ext_filter_list is not None:
            for ext_tmp in ext_filter_list:
                wildcard = f"{wildcard_prefix}{ext_tmp}"
                curlist = glob.glob(os.path.join(folder_path, wildcard))
                if sort:
                    curlist = sorted(curlist)
                fulllist += curlist

        else:
            wildcard = wildcard_prefix
            curlist = glob.glob(os.path.join(folder_path, wildcard))

            if sort:
                curlist = sorted(curlist)
            fulllist += curlist
        if recursive and depth > 1:
            newlist, _ = load_list_from_folder(
                folder_path=folder_path,
                ext_filter=ext_filter_list,
                depth=depth - 1,
                recursive=True,
            )
            fulllist += newlist

    fulllist = [os.path.normpath(path_tmp) for path_tmp in fulllist]
    num_elem = len(fulllist)

    # save list to a path
    if save_path is not None:
        save_path = safe_path(save_path)
        if debug and not is_path_exists_or_creatable(save_path):
            raise ValueError("the file cannot be created")
        with open(save_path, "w", encoding="utf-8") as file:
            for item in fulllist:
                file.write("%s\n" % item)

    return fulllist, num_elem
