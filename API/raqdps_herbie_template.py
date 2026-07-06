"""Repo-local Herbie model template for ECCC RAQDPS GRIB2 files."""

from API.raqdps_utils import RAQDPS_LEVEL, build_raqdps_url

_VARIABLES = {
    "PM2.5",
    "PM10",
    "NO2",
    "O3",
    "SO2",
    "NO",
    "PM2.5-WildfireSmokePlume",
    "PM10-WildfireSmokePlume",
}

_LEVELS = {"Sfc", "AGL-2m", "EAtm"}


class raqdps:
    """Herbie template for the Regional Air Quality Deterministic Prediction System."""

    def template(self):
        if self.product is None:
            self.product = "10km/grib2"

        product_aliases = {
            "10km": "10km/grib2",
            "10km/grib2": "10km/grib2",
        }
        self.product = product_aliases.get(self.product, self.product)

        if self.product != "10km/grib2":
            raise ValueError("product must be '10km/grib2' or '10km'")

        if not hasattr(self, "variable"):
            raise ValueError("RAQDPS requires a 'variable' argument")
        if not hasattr(self, "level"):
            self.level = RAQDPS_LEVEL

        if self.variable not in _VARIABLES:
            raise ValueError(f"Unsupported RAQDPS variable: {self.variable}")
        if self.level not in _LEVELS:
            raise ValueError(f"Unsupported RAQDPS level: {self.level}")

        self.DESCRIPTION = "Canada's Regional Air Quality Deterministic Prediction System (RAQDPS)"
        self.DETAILS = {
            "Datamart product description": "https://eccc-msc.github.io/open-data/msc-data/nwp_raqdps/readme_raqdps-datamart_en/",
        }
        self.PRODUCTS = {"10km/grib2": "regional 10 km air quality domain"}
        self.AVAILABLE_VARIABLES = sorted(_VARIABLES)
        self.AVAILABLE_LEVELS = sorted(_LEVELS)
        self.SOURCES = {
            "msc": build_raqdps_url(
                self.date,
                self.variable,
                self.fxx,
                level=self.level,
            )
        }
        self.IDX_SUFFIX = [".idx", ".grib2.idx", ".grb2.idx"]
        self.LOCALFILE = f"{self.get_remoteFileName}"
