"""Convert the static ETOPO elevation grid to the sharded zarr v3 layout.

No ingest produces ETOPO_DA_C. The old store (unpacked from
ForecastTar_v2/v30/ETOPO_DA_C.zarr.zip) is zarr v2 with one file per 25x25 chunk:
374,546 objects that API.store_sync would have to fetch on every new node. This
keeps the values, chunks and codec, and groups the chunks into 243 shard files.

Upload the shards before zarr.json: store_sync copies this store once, and only
after it can see zarr.json, so a partial upload is never copied.

    python scripts/convert_etopo_to_zarr3.py <v2 directory> <output>/ETOPO_DA_C.zarr
"""

import argparse
import os

import numpy as np
import zarr

# Point reads through tensorstore, in the same run on a RAM disk (one read per
# API request):
#   chunk / shard   files   MiB    write   p50     warm p50
#   25 / 1000       243     271    99 s    218 us  175 us
#   10 / 1000       243     440    582 s   212 us  152 us
#   5 / 1000        243     1181   1968 s  194 us  154 us
# Smaller chunks save ~20 us per read, against 4.4x the size to copy to each node.
ETOPO_CHUNK = 25
ETOPO_SHARD = 1000
ETOPO_SHAPE = (10801, 21601)  # 1 arc-minute global grid (y, x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="Unpacked zarr v2 ETOPO_DA_C directory")
    parser.add_argument("output", help="New zarr v3 directory, ending ETOPO_DA_C.zarr")
    args = parser.parse_args()

    source = zarr.open(zarr.storage.LocalStore(args.source, read_only=True), mode="r")
    elevation = source[:]
    if elevation.shape != ETOPO_SHAPE:
        raise ValueError(f"Expected shape {ETOPO_SHAPE}, got {elevation.shape}")
    if np.isnan(elevation).any():
        raise ValueError("ETOPO source contains NaN")

    output = zarr.create_array(
        zarr.storage.LocalStore(args.output),
        shape=elevation.shape,
        dtype=elevation.dtype,
        chunks=(ETOPO_CHUNK, ETOPO_CHUNK),
        shards=(ETOPO_SHARD, ETOPO_SHARD),
        fill_value=source.fill_value,
        compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle="shuffle"),
    )
    output[:] = elevation

    # This store is copied once and never replaced, so check it before upload.
    reopened = zarr.open(zarr.storage.LocalStore(args.output, read_only=True), mode="r")
    if not np.array_equal(reopened[:], elevation):
        raise ValueError("Converted ETOPO differs from its source")

    file_count = sum(len(files) for _, _, files in os.walk(args.output))
    print(f"Wrote {args.output}: {file_count} files, values match the source")


if __name__ == "__main__":
    main()
