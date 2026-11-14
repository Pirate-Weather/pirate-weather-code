"""Repo-local Herbie model template for ECCC RDAQA GRIB2 files."""

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


class rdaqa:
    """Herbie template for the Regional Deterministic Air Quality Analysis."""

    def template(self):
        if self.product is None:
            self.product = "10km"

        if self.product != "10km":
            raise ValueError("product must be '10km'")

        if not hasattr(self, "variable"):
            raise ValueError("RDAQA requires a 'variable' argument")
        if not hasattr(self, "level"):
            self.level = RDAQA_LEVEL
        if not hasattr(self, "analysis"):
            self.analysis = RDAQA_ANALYSIS

        if self.analysis not in _ANALYSES:
            raise ValueError(
                f"Unsupported RDAQA analysis: {self.analysis} "
                f"(expected one of {sorted(_ANALYSES)})"
            )
        prefix, allowed = _ANALYSES[self.analysis]
        if self.variable not in allowed:
            raise ValueError(
                f"Unsupported RDAQA variable for '{self.analysis}' analysis: "
                f"{self.variable} (expected one of {sorted(allowed)})"
            )
        if self.level not in _LEVELS:
            raise ValueError(f"Unsupported RDAQA level: {self.level}")

        self.DESCRIPTION = (
            "Canada's Regional Deterministic Air Quality Analysis (RDAQA)"
        )
        self.DETAILS = {
            "Datamart product description": "https://eccc-msc.github.io/open-data/msc-data/nwp_rdaqa/readme_rdaqa_en/",
        }
        self.PRODUCTS = {"10km": "regional 10 km air quality domain"}
        self.AVAILABLE_VARIABLES = sorted(_POLLUTANTS)
        self.AVAILABLE_LEVELS = sorted(_LEVELS)
        self.AVAILABLE_ANALYSES = sorted(_ANALYSES)

        # MSC uses "RDAQA-" for prefixed (Prelim_/FW_) files and "RDAQA_" otherwise.
        sep = "-" if prefix else "_"
        filename = (
            f"{self.date:%Y%m%dT%HZ}_MSC_RDAQA{sep}{prefix}{self.variable}"
            f"_{self.level}_RLatLon0.09_PT0H.grib2"
        )
        self.SOURCES = {
            "msc": (
                f"https://dd.weather.gc.ca/{self.date:%Y%m%d}/WXO-DD/model_rdaqa/"
                f"{self.product}/{self.date:%H}/{filename}"
            )
        }
        self.IDX_SUFFIX = [".idx", ".grib2.idx", ".grb2.idx"]
        self.LOCALFILE = f"{self.get_remoteFileName}"
