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

"""Shared OpenAI/Azure OpenAI client utilities."""

import os
from dataclasses import dataclass
from typing import Any, Literal

import requests
from openai import AzureOpenAI, OpenAI

NV_AZURE_BASE_URL = "https://prod.api.nvidia.com/llm/v1/azure/openai"
NV_OAUTH_TOKEN_URL = "https://prod.api.nvidia.com/oauth/api/v1/ssa/default/token"
NV_OAUTH_SCOPE = "azureopenai-readwrite"
NV_INFERENCE_URL = "https://inference-api.nvidia.com"


@dataclass(frozen=True)
class CloudOpenAIClient:
    """Selected OpenAI-compatible client and backend metadata."""

    client: Any
    backend: Literal["nvidia_azure", "nvidia_inference", "openai"]


def get_nv_oauth_token() -> str:
    """Fetch NVIDIA OAuth token used for Azure OpenAI-compatible endpoints."""
    client_id = os.environ.get("NVHOST_OAI_CLIENT_ID")
    client_secret = os.environ.get("NVHOST_OAI_CLIENT_SECRET")
    if client_id is None or client_secret is None:
        raise ValueError(
            "Please set NVHOST_OAI_CLIENT_ID and NVHOST_OAI_CLIENT_SECRET in your "
            "environment variables."
        )

    response = requests.post(
        NV_OAUTH_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": NV_OAUTH_SCOPE,
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(
            "Failed to obtain OAuth token. "
            f"Status code: {response.status_code}, Response: {response.text}"
        )
    return response.json()["access_token"]


def get_nv_azure_openai_client(api_key: str, api_version: str) -> AzureOpenAI:
    """Create AzureOpenAI client configured for NVIDIA-hosted endpoint."""
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=NV_AZURE_BASE_URL,
    )


def create_nv_azure_openai_client(api_version: str) -> AzureOpenAI:
    """Fetch token and build AzureOpenAI client in one call."""
    return get_nv_azure_openai_client(api_key=get_nv_oauth_token(), api_version=api_version)


def _has_nv_azure_credentials() -> bool:
    """Return whether complete NVIDIA-hosted Azure OpenAI credentials are configured."""
    return bool(
        os.environ.get("NVHOST_OAI_CLIENT_ID") and os.environ.get("NVHOST_OAI_CLIENT_SECRET")
    )


def create_standard_openai_client() -> OpenAI:
    """Create a standard OpenAI API client from public OpenAI environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("Please set OPENAI_API_KEY in your environment variables.")

    client_kwargs: dict[str, str] = {"api_key": api_key}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    organization = os.environ.get("OPENAI_ORG_ID") or os.environ.get("OPENAI_ORGANIZATION")
    if organization:
        client_kwargs["organization"] = organization
    project = os.environ.get("OPENAI_PROJECT")
    if project:
        client_kwargs["project"] = project
    return OpenAI(**client_kwargs)


def create_nv_inference_openai_client() -> OpenAI:
    """Create an OpenAI-compatible client for NVIDIA inference API keys."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    if api_key is None:
        raise ValueError("Please set NVIDIA_API_KEY in your environment variables.")

    base_url = os.environ.get("NV_INFERENCE_URL", NV_INFERENCE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def create_cloud_openai_client(api_version: str) -> CloudOpenAIClient:
    """Create the configured cloud OpenAI-compatible client.

    NVIDIA-hosted Azure OpenAI credentials take precedence to preserve existing
    internal behavior. Users can set ``NVIDIA_API_KEY`` for inference.nvidia.com,
    or ``OPENAI_API_KEY`` for the standard OpenAI API, with optional
    ``OPENAI_BASE_URL`` for compatible endpoints.
    """
    if _has_nv_azure_credentials():
        return CloudOpenAIClient(
            client=create_nv_azure_openai_client(api_version=api_version),
            backend="nvidia_azure",
        )
    if os.environ.get("NVIDIA_API_KEY"):
        return CloudOpenAIClient(
            client=create_nv_inference_openai_client(),
            backend="nvidia_inference",
        )
    if os.environ.get("OPENAI_API_KEY"):
        return CloudOpenAIClient(client=create_standard_openai_client(), backend="openai")
    raise ValueError(
        "Please configure cloud model credentials. Set either "
        "NVHOST_OAI_CLIENT_ID and NVHOST_OAI_CLIENT_SECRET for NVIDIA-hosted "
        "Azure OpenAI, NVIDIA_API_KEY for NVIDIA inference, or OPENAI_API_KEY "
        "for the standard OpenAI API."
    )
