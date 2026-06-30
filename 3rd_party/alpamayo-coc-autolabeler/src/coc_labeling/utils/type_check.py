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

"""Lightweight runtime type/path validation helpers."""

import os
from typing import Any

import numpy as np


def isstring(string_test: Any) -> bool:
    """Return whether input is a string."""
    return isinstance(string_test, str)


def islist(list_test: Any) -> bool:
    """Return whether input is a list."""
    return isinstance(list_test, list)


def islogical(logical_test: Any) -> bool:
    """Return whether input is a bool."""
    return isinstance(logical_test, bool)


def isnparray(nparray_test: Any) -> bool:
    """Return whether input is a numpy ndarray."""
    return isinstance(nparray_test, np.ndarray)


def isinteger(integer_test: Any) -> bool:
    """Return whether input can be treated as an integer scalar."""
    if isnparray(integer_test):
        return False
    try:
        return isinstance(integer_test, int) or int(integer_test) == integer_test
    except (TypeError, ValueError):
        return False


def is_path_valid(pathname: Any) -> bool:
    """Return whether pathname is a non-empty string."""
    try:
        if not isstring(pathname) or not pathname:
            return False
    except TypeError:
        return False
    return True


def is_path_creatable(pathname: Any) -> bool:
    """Return whether a path can be created under an existing writable parent."""
    if not is_path_valid(pathname):
        return False
    pathname = os.path.normpath(pathname)
    pathname = os.path.dirname(os.path.abspath(pathname))

    # recursively to find the previous level of parent folder existing
    while not is_path_exists(pathname):
        pathname_new = os.path.dirname(os.path.abspath(pathname))
        if pathname_new == pathname:
            return False
        pathname = pathname_new
    return os.access(pathname, os.W_OK)


def is_path_exists(pathname: Any) -> bool:
    """Return whether path exists on disk."""
    try:
        return is_path_valid(pathname) and os.path.exists(pathname)
    except OSError:
        return False


def is_path_exists_or_creatable(pathname: Any) -> bool:
    """Return whether path already exists or can be created."""
    try:
        return is_path_exists(pathname) or is_path_creatable(pathname)
    except OSError:
        return False


def isfolder(pathname: Any) -> bool:
    """Heuristically decide whether pathname points to a folder-like path."""
    if not is_path_valid(pathname):
        return False
    pathname = os.path.normpath(pathname)
    if pathname == "./":
        return True
    name = os.path.splitext(os.path.basename(pathname))[0]
    ext = os.path.splitext(pathname)[1]
    return len(name) > 0 and len(ext) == 0
