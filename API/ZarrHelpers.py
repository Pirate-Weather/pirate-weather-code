# Helpers for working with Zarr data in the Pirate Weather API
# Alexander Rey, October 2025

import zarr
import s3fs
import time
import os
import random

from API.constants.api_const import (
    MAX_S3_RETRIES,
    S3_BASE_DELAY,
)

pw_api_key = os.environ.get("PW_API", "")


def _add_custom_header(request, **kwargs):
    request.headers["apikey"] = pw_api_key


class S3ZipStore(zarr.storage.ZipStore):
    def __init__(self, path: s3fs.S3File) -> None:
        super().__init__(path="", mode="r")
        self.path = path


def _retry_s3_operation(
    operation, max_retries=MAX_S3_RETRIES, base_delay=S3_BASE_DELAY
):
    """Retry S3 operations with exponential backoff for rate limiting."""
    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as e:
            # Check if it's a rate limiting error
            if "429" in str(e) or "Too Many Requests" in str(e):
                if attempt < max_retries - 1:
                    # Calculate delay with exponential backoff and jitter
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    print(
                        f"S3 rate limit hit (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)
                    continue
            # Re-raise the exception if it's not rate limiting or max retries reached
            raise e
    raise Exception(f"Failed after {max_retries} attempts")


def setup_testing_zipstore(
    s3: s3fs.S3FileSystem,
    s3_bucket: str,
    ingest_version: str,
    save_type: str,
    model_name: str,
) -> zarr.storage.BaseStore:
    """Sets up a zarr store from a zipped zarr file in S3 or locally.

    Parameters:
        - s3 (s3fs.S3FileSystem): An s3fs filesystem object.
        - s3_bucket (str): The S3 bucket name or local path.
        - ingest_version (str): The version string for the data.
        - save_type (str): The type of storage ("S3", "S3Zarr", or local path).
        - model_name (str): The name of the model.

    Returns:
        - zarr.storage.BaseStore: A zarr store object.
    """

    if save_type == "S3":
        try:
            f = _retry_s3_operation(
                lambda: s3.open(
                    "s3://ForecastTar_v2/"
                    + ingest_version
                    + "/"
                    + model_name
                    + ".zarr.zip"
                )
            )
            store = S3ZipStore(f)
        # Try an old ingest version for testing
        except FileNotFoundError:
            ingest_version = "v28"
            print("Using old ingest version: " + ingest_version)
            f = _retry_s3_operation(
                lambda: s3.open(
                    "s3://ForecastTar_v2/"
                    + ingest_version
                    + "/"
                    + model_name
                    + ".zarr.zip"
                )
            )
            store = S3ZipStore(f)
    elif save_type == "S3Zarr":
        try:
            f = _retry_s3_operation(
                lambda: s3.open(
                    "s3://"
                    + s3_bucket
                    + "/ForecastTar_v2/"
                    + ingest_version
                    + "/"
                    + model_name
                    + ".zarr.zip"
                )
            )
            store = S3ZipStore(f)
        except FileNotFoundError:
            ingest_version = "v28"
            print("Using old ingest version: " + ingest_version)
            f = _retry_s3_operation(
                lambda: s3.open(
                    "s3://"
                    + s3_bucket
                    + "/ForecastTar_v2/"
                    + ingest_version
                    + "/"
                    + model_name
                    + ".zarr.zip"
                )
            )
            store = S3ZipStore(f)

    else:
        f = s3_bucket + model_name + ".zarr.zip"
        store = zarr.storage.ZipStore(f, mode="r")

    return store
