# %% Scratch notebook to view a published sharded zarr store on S3

# # Open the current version of a published store from S3 using s3fs + AWS keys
#
# Expected object layout (API.ingest_utils.publish_store):
# s3://<bucket>/ForecastTar_v2/<ingest_version>/<MODEL>.current.json
#     {"store": "<MODEL>/<UTC stamp>.zarr", "run_time": "..."}
# s3://<bucket>/ForecastTar_v2/<ingest_version>/<MODEL>/<UTC stamp>.zarr/
#
# Each store is one (variable, time, y, x) array; the variable names are the
# "variables" attribute and the valid times (Unix seconds) the "time" attribute.

import json
import os

import pandas as pd
import s3fs
import zarr
from dotenv import load_dotenv
from zarr.storage import FsspecStore

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

INGEST_VERSION = "v31"
MODEL_NAME = "SILAM"

STORE_ROOT = f"{S3_BUCKET}/ForecastTar_v2/{INGEST_VERSION}"


# %%
# ---- Resolve the current version through the pointer ----

s3 = s3fs.S3FileSystem(key=AWS_ACCESS_KEY_ID, secret=AWS_SECRET_ACCESS_KEY)
pointer = json.loads(s3.cat_file(f"{STORE_ROOT}/{MODEL_NAME}.current.json"))
print(pointer)


# %%
# ---- Open the store read-only ----

store = FsspecStore.from_url(
    f"s3://{STORE_ROOT}/{pointer['store']}",
    storage_options={"key": AWS_ACCESS_KEY_ID, "secret": AWS_SECRET_ACCESS_KEY},
    read_only=True,
)
opened_zarr = zarr.open_array(store, mode="r", zarr_format=3)
variables = opened_zarr.attrs["variables"]
print(opened_zarr.shape, variables)


# %% Test read (axis 0 is variables[i])
data = opened_zarr[0, 24 * 10 : 24 * 11, 500, 500]
print(variables[0], data)


# %% Print the valid times as datetimes
unixTime = opened_zarr.attrs["time"][24 * 0 : 24 * 1]

print(pd.to_datetime(unixTime, unit="s"))
