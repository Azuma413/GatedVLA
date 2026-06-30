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

"""File-system helper utilities."""

import os
from typing import Tuple

from coc_labeling.utils.type_check import is_path_exists_or_creatable, isstring


def safe_path(input_path: str, debug: bool = True) -> str:
    """Normalize path string into canonical OS representation.

    Args:
        input_path: Path string to normalize.
        debug: Whether to validate argument type.
    """
    if debug and not isstring(input_path):
        raise TypeError(f"path is not a string: {input_path!r}")
    return os.path.normpath(input_path)


def fileparts(input_path: str, debug: bool = True) -> Tuple[str, str, str]:
    """Return `(directory, filename_without_ext, ext)` for a path."""
    good_path = safe_path(input_path, debug=debug)
    if len(good_path) == 0:
        return ("", "", "")
    if good_path[-1] == "/":
        if len(good_path) > 1:
            return (good_path[:-1], "", "")  # ignore the final '/'
        return (good_path, "", "")  # ignore the final '/'

    directory = os.path.dirname(os.path.abspath(good_path))
    filename = os.path.splitext(os.path.basename(good_path))[0]
    ext = os.path.splitext(good_path)[1]
    return (directory, filename, ext)


def mkdir_if_missing(input_path: str, debug: bool = True) -> None:
    """Create parent directory for file paths or directory path itself.

    Behavior:
    - If `input_path` explicitly looks like a directory path (trailing slash)
      or already exists as a directory, create that directory.
    - Otherwise, treat it as a file path and create its parent directory.
    """
    good_path = safe_path(input_path, debug=debug)
    if debug and not is_path_exists_or_creatable(good_path):
        raise ValueError(f"input path is not valid or creatable: {good_path}")

    looks_like_dir = input_path.endswith((os.sep, "/", "\\"))
    if looks_like_dir or os.path.isdir(good_path):
        os.makedirs(good_path, exist_ok=True)
        return

    parent_dir = os.path.dirname(good_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
