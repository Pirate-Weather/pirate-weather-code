from __future__ import annotations

import os
import pickle
import shutil
from dataclasses import dataclass
from typing import Optional

import boto3
import s3fs
import zarr
from boto3.s3.transfer import TransferConfig

AWS_KEY = os.environ.get("AWS_KEY", "")
AWS_SECRET = os.environ.get("AWS_SECRET", "")
SAVE_TYPE = os.getenv("save_type", "S3")
S3_BUCKET = os.getenv("s3_bucket", "piratezarr2")
PW_API_KEY = os.environ.get("PW_API", "")


class S3ZipStore(zarr.storage.ZipStore):
    """Zarr ZipStore wrapper for reading objects via ``s3fs``."""

    def __init__(self, path: s3fs.S3File) -> None:
        super().__init__(path="", mode="r")
        self.path = path


def add_custom_header(request, **_kwargs) -> None:
    """Attach a Pirate Weather API key to ``s3fs`` HTTP requests."""

    request.headers["apikey"] = PW_API_KEY


@dataclass
class DownloadSpec:
    bucket: str
    object_key: str
    local_file: str
    local_store: str
    initial_download: bool = False


def download_if_newer(spec: DownloadSpec) -> None:
    """Download ``spec.object_key`` if the remote file is newer than ``spec.local_file``."""

    config = (
        TransferConfig(use_threads=True, max_bandwidth=None)
        if spec.initial_download
        else TransferConfig(use_threads=False, max_bandwidth=100000000)
    )

    if SAVE_TYPE == "S3":
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_KEY,
            aws_secret_access_key=AWS_SECRET,
        )
        s3_response = s3_client.head_object(Bucket=spec.bucket, Key=spec.object_key)
        s3_last_modified = s3_response["LastModified"].timestamp()
    else:
        s3_last_modified = os.path.getmtime(os.path.join(spec.bucket, spec.object_key))

    new_file = False
    mod_file = f"{spec.local_file}.modtime.pickle"
    if os.path.exists(mod_file):
        with open(mod_file, "rb") as fh:
            local_last = pickle.load(fh)
        if s3_last_modified > local_last:
            _download(config, spec)
            new_file = True
            with open(mod_file, "wb") as fh:
                pickle.dump(s3_last_modified, fh)
    else:
        _download(config, spec)
        with open(mod_file, "wb") as fh:
            pickle.dump(s3_last_modified, fh)
        new_file = True

    if new_file:
        with open(f"{spec.local_store}.lock", "w"):
            pass
        shutil.move(spec.local_file, f"{spec.local_store}_{s3_last_modified}")
        os.remove(f"{spec.local_store}.lock")


def _download(config: TransferConfig, spec: DownloadSpec) -> None:
    if SAVE_TYPE == "S3":
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_KEY,
            aws_secret_access_key=AWS_SECRET,
        )
        s3_client.download_file(spec.bucket, spec.object_key, spec.local_file, Config=config)
    else:
        shutil.copy(os.path.join(spec.bucket, spec.object_key), spec.local_file)
