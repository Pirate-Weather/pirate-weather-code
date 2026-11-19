"""Repo-local Herbie model template for ECCC RDAQA GRIB2 files."""

RDAQA_BASE_URL = "https://dd.weather.gc.ca"
RDAQA_PRODUCT = "10km"
RDAQA_LEVEL = "Sfc"
RDAQA_ANALYSIS = "final"

_POLLUTANTS = {"PM2.5", "PM10", "NO2", "O3", "SO2", "NO"}

# analysis -> (filename prefix, pollutants available for that analysis)
_ANALYSES = {
    "final": ("", _POLLUTANTS),
    "prelim": ("Prelim_", _POLLUTANTS),
    "fw": ("FW_", {"PM2.5", "PM10"}),
}

_LEVELS = {"Sfc"}


def build_rdaqa_url(
    valid_time, variable, analysis=RDAQA_ANALYSIS, level=RDAQA_LEVEL
) -> str:
    """Datamart URL of one RDAQA analysis hour (``valid_time`` in UTC).

    Raises ValueError for an analysis, variable or level MSC does not publish.
    """
    if analysis not in _ANALYSES:
        raise ValueError(
            f"Unsupported RDAQA analysis: {analysis} "
            f"(expected one of {sorted(_ANALYSES)})"
        )
    prefix, allowed = _ANALYSES[analysis]
    if variable not in allowed:
        raise ValueError(
            f"Unsupported RDAQA variable for '{analysis}' analysis: "
            f"{variable} (expected one of {sorted(allowed)})"
        )
    if level not in _LEVELS:
        raise ValueError(f"Unsupported RDAQA level: {level}")

    # MSC uses "RDAQA-" for prefixed (Prelim_/FW_) files and "RDAQA_" otherwise.
    sep = "-" if prefix else "_"
    filename = (
        f"{valid_time:%Y%m%dT%HZ}_MSC_RDAQA{sep}{prefix}{variable}"
        f"_{level}_RLatLon0.09_PT0H.grib2"
    )
    return (
        f"{RDAQA_BASE_URL}/{valid_time:%Y%m%d}/WXO-DD/model_rdaqa/"
        f"{RDAQA_PRODUCT}/{valid_time:%H}/{filename}"
    )


class rdaqa:
    """Herbie template for the Regional Deterministic Air Quality Analysis."""

    def template(self):
        if self.product is None:
            self.product = RDAQA_PRODUCT

        if self.product != RDAQA_PRODUCT:
            raise ValueError(f"product must be '{RDAQA_PRODUCT}'")

        if not hasattr(self, "variable"):
            raise ValueError("RDAQA requires a 'variable' argument")
        if not hasattr(self, "level"):
            self.level = RDAQA_LEVEL
        if not hasattr(self, "analysis"):
            self.analysis = RDAQA_ANALYSIS

        self.DESCRIPTION = (
            "Canada's Regional Deterministic Air Quality Analysis (RDAQA)"
        )
        self.DETAILS = {
            "Datamart product description": "https://eccc-msc.github.io/open-data/msc-data/nwp_rdaqa/readme_rdaqa_en/",
        }
        self.PRODUCTS = {RDAQA_PRODUCT: "regional 10 km air quality domain"}
        self.AVAILABLE_VARIABLES = sorted(_POLLUTANTS)
        self.AVAILABLE_LEVELS = sorted(_LEVELS)
        self.AVAILABLE_ANALYSES = sorted(_ANALYSES)
        self.SOURCES = {
            "msc": build_rdaqa_url(
                self.date, self.variable, analysis=self.analysis, level=self.level
            )
        }
        self.IDX_SUFFIX = [".idx", ".grib2.idx", ".grb2.idx"]
        self.LOCALFILE = f"{self.get_remoteFileName}"
