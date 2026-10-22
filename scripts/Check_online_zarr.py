# %% Scratch notebook to view a sharded zarr store on S3

# # Open a Zarr store from S3 using s3fs + AWS keys
#
# Expected object layout:
# s3://<bucket>/ForecastTar_v2/<ingest_version>/<model>.zarr/
#
# Example:
# s3://my-bucket/ForecastTar_v2/v31/GFS.zarr/

# Open a Pirate Weather Zarr store from S3 using repo helpers + .env

import os

import pandas as pd
import s3fs
from dotenv import load_dotenv

from API.io.ZarrHelpers import open_testing_zarr

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
INGEST_VERSION = "v31"

SAVE_TYPE = "S3Zarr"


# %%
# ---- Create S3 filesystem ----

s3 = s3fs.S3FileSystem(
    key=AWS_ACCESS_KEY_ID,
    secret=AWS_SECRET_ACCESS_KEY,
    version_aware=True,
    asynchronous=True,
)


# %%
# ---- Open the Zarr store using the repo helper ----

opened_zarr = open_testing_zarr(
    s3=s3,
    s3_bucket=S3_BUCKET,
    ingest_version=INGEST_VERSION,
    save_type=SAVE_TYPE,
    model_name="SILAM",
)


# %% Test read
data = opened_zarr[1, 24 * 10 : 24 * 11, 500, 500]
print(data)


# %% Print unix time as datetimes
unixTime = opened_zarr[0, 24 * 10 : 24 * 11, 500, 500]
unixTime = opened_zarr[0, 24 * 0 : 24 * 1, 500, 500]

print(pd.to_datetime(unixTime, unit="s"))
