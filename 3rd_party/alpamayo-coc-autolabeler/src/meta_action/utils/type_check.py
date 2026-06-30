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


import os
from typing import Any

import numpy as np


def isstring(string_test: Any) -> bool:
    """Return whether the input is a string."""
    return isinstance(string_test, str)


def islist(list_test: Any) -> bool:
    """Return whether the input is a list."""
    return isinstance(list_test, list)


def islogical(logical_test: Any) -> bool:
    """Return whether the input is a boolean."""
    return isinstance(logical_test, bool)


def isnparray(nparray_test: Any) -> bool:
    """Return whether the input is a NumPy array."""
    return isinstance(nparray_test, np.ndarray)


def isinteger(integer_test: Any) -> bool:
    """Return whether the input behaves like an integer scalar."""
    if isnparray(integer_test):
        return False
    try:
        return isinstance(integer_test, int) or int(integer_test) == integer_test
    except (ValueError, TypeError, OverflowError):
        return False


def is_path_valid(pathname: Any) -> bool:
    """Return whether a path-like input is a non-empty valid string."""
    try:
        if not isstring(pathname) or not pathname:
            return False
    except TypeError:
        return False
    else:
        return True


def is_path_creatable(pathname: Any) -> bool:
    """Return whether the path can be created in an existing writable parent."""
    if not is_path_valid(pathname):
        return False
    current_path = os.path.normpath(pathname)
    current_path = os.path.dirname(os.path.abspath(current_path))

    # Recursively find the nearest existing parent directory.
    while not is_path_exists(current_path):
        pathname_new = os.path.dirname(os.path.abspath(current_path))
        if pathname_new == current_path:
            return False
        current_path = pathname_new
    return os.access(current_path, os.W_OK)


def is_path_exists(pathname: Any) -> bool:
    """Return whether the path exists on disk."""
    try:
        return is_path_valid(pathname) and os.path.exists(pathname)
    except OSError:
        return False


def is_path_exists_or_creatable(pathname: Any) -> bool:
    """Return whether the path exists or can be created."""
    try:
        return is_path_exists(pathname) or is_path_creatable(pathname)
    except OSError:
        return False


def isfolder(pathname: Any) -> bool:
    """Heuristically determine whether a path string represents a folder path.

    This helper treats paths without a final extension as folders.
    Example: `/tmp/adhoc_0.5x/abc` is treated as a folder while
    `/tmp/adhoc_0.5x` is treated as a file-like path.
    """
    if is_path_valid(pathname):
        normalized = os.path.normpath(pathname)
        if normalized == "./":
            return True
        name = os.path.splitext(os.path.basename(normalized))[0]
        ext = os.path.splitext(normalized)[1]
        return len(name) > 0 and len(ext) == 0
    return False
