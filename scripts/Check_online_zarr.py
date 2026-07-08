# %% Scratch notebook to view a zip zarr on S3

# # Open a zipped Zarr store from S3 using s3fs + AWS keys
#
# Expected object layout:
# s3://<bucket>/ForecastTar_v2/<ingest_version>/<model>.zarr.zip
#
# Example:
# s3://my-bucket/ForecastTar_v2/v30/GFS.zarr.zip

# Open a Pirate Weather zipped Zarr from S3 using repo helpers + .env

import os

import pandas as pd
import s3fs
import zarr
from dotenv import load_dotenv

from API.io.ZarrHelpers import setup_testing_zipstore

# %%
# ---- Load .env ----
#
# Looks for .env in the current working directory or parent directories.

load_dotenv()


# %%
# ---- Configuration ----

AWS_ACCESS_KEY_ID = os.environ["AWS_KEY"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET"]

S3_BUCKET = os.environ["s3_bucket"]

# Use .env value if supplied; otherwise fall back to repo constant.
INGEST_VERSION = "v30"

SAVE_TYPE = "S3Zarr"


# %%
# ---- Create S3 filesystem ----

s3 = s3fs.S3FileSystem(
    key=AWS_ACCESS_KEY_ID,
    secret=AWS_SECRET_ACCESS_KEY,
    version_aware=True,
)


# %%
# ---- Build zipped Zarr store using repo helper ----

store = setup_testing_zipstore(
    s3=s3,
    s3_bucket=S3_BUCKET,
    ingest_version=INGEST_VERSION,
    save_type=SAVE_TYPE,
    model_name="GFS",
)

store2 = setup_testing_zipstore(
    s3=s3,
    s3_bucket=S3_BUCKET,
    ingest_version=INGEST_VERSION,
    save_type=SAVE_TYPE,
    model_name="NBM",
)
# %%
# ---- Open Zarr ----

opened_zarr = zarr.open(store, mode="r")
opened_zarr2 = zarr.open(store2, mode="r")


# %% Test read
data = opened_zarr[1, 24 * 10 : 24 * 11, 500, 500]
print(data)

data2 = opened_zarr2[1, :, 500, 500]
print(data2)


# %% Print unix time as datetimes
unixTime = opened_zarr[0, 24 * 10 : 24 * 11, 500, 500]

print(pd.to_datetime(unixTime, unit="s"))
