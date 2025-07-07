# Helper constants for dataset indexes

# Index positions for runtime values in various model arrays
HRRR_RUNTIME_INDEX = 36
HRRR_6H_RUNTIME_INDEX = 0
SUBH_RUNTIME_INDEX = 0
NBM_RUNTIME_INDEX = 36
NBM_FIRE_RUNTIME_INDEX = 30
GFS_RUNTIME_INDEX = 35
GEFS_RUNTIME_INDEX = 33

# S3 object keys and local destinations for forecast datasets
ZARR_DOWNLOADS = {
    "SubH": (
        "ForecastTar_v2/SubH.zarr.zip",
        "/tmp/SubH_TMP.zarr.zip",
        "/tmp/SubH.zarr.prod.zip",
    ),
    "HRRR_6H": (
        "ForecastTar_v2/HRRR_6H.zarr.zip",
        "/tmp/HRRR_6H_TMP.zarr.zip",
        "/tmp/HRRR_6H.zarr.prod.zip",
    ),
    "GFS": (
        "ForecastTar_v2/GFS.zarr.zip",
        "/tmp/GFS.zarr_TMP.zip",
        "/tmp/GFS.zarr.prod.zip",
    ),
    "NBM": (
        "ForecastTar_v2/NBM.zarr.zip",
        "/tmp/NBM.zarr_TMP.zip",
        "/tmp/NBM.zarr.prod.zip",
    ),
    "NBM_Fire": (
        "ForecastTar_v2/NBM_Fire.zarr.zip",
        "/tmp/NBM_Fire_TMP.zarr.zip",
        "/tmp/NBM_Fire.zarr.prod.zip",
    ),
    "GEFS": (
        "ForecastTar_v2/GEFS.zarr.zip",
        "/tmp/GEFS_TMP.zarr.zip",
        "/tmp/GEFS.zarr.prod.zip",
    ),
    "HRRR": (
        "ForecastTar_v2/HRRR.zarr.zip",
        "/tmp/HRRR_TMP.zarr.zip",
        "/tmp/HRRR.zarr.prod.zip",
    ),
    "NWS_Alerts": (
        "ForecastTar_v2/NWS_Alerts.zarr.zip",
        "/tmp/NWS_Alerts_TMP.zarr.zip",
        "/tmp/NWS_Alerts.zarr.prod.zip",
    ),
    "ETOPO": (
        "ForecastTar_v2/ETOPO_DA_C.zarr.zip",
        "/tmp/ETOPO_DA_C_TMP.zarr.zip",
        "/tmp/ETOPO_DA_C.zarr.prod.zip",
    ),
}
