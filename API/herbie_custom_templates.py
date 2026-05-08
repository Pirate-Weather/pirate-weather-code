"""Custom Herbie model templates for AWS-backed AI-GFS products.

This module overrides Herbie's built-in `aigfs` and `aigefs` templates so
the ingest scripts can read current NOAA EAGLE AWS objects first and then
fallback to NOMADS when needed.
"""


class aigfs:
    """NOAA AI Global Forecast System."""

    def template(self):
        self.DESCRIPTION = "NOAA AI Global Forecast System"
        self.DETAILS = {
            "nomads product description": "https://www.nco.ncep.noaa.gov/pmb/products/aigfs",
            "aws document": "https://registry.opendata.aws/noaa-nws-graphcastgfs-pds/",
        }
        self.PRODUCTS = {
            "sfc": "surface fields, 0.25 degree resolution",
            "pres": "pressure fields, 0.25 degree resolution",
        }

        post_root = (
            f"aigfs.{self.date:%Y%m%d/%H}/model/atmos/grib2/"
            f"aigfs.t{self.date:%H}z.{self.product}.f{self.fxx:03d}"
        )

        self.SOURCES = {
            "aws": f"https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/{post_root}.grib2",
            "nomads": f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/prod/{post_root}.grib2",
        }
        self.LOCALFILE = f"{self.get_remoteFileName}"
        self.IDX_SUFFIX = [".grib2.idx", ".idx"]


class aigefs:
    """AI Global Ensemble Forecast System (AIGEFS)."""

    def template(self):
        self.DESCRIPTION = "AI Global Ensemble Forecast System (AIGEFS)"
        self.DETAILS = {
            "NOMADS": "https://www.nco.ncep.noaa.gov/pmb/products/aigefs/",
            "aws document": "https://registry.opendata.aws/noaa-nws-graphcastgfs-pds/",
        }

        self.PRODUCTS = {
            "pres": "pressure fields, 0.25 degree resolution",
            "sfc": "surface fields, 0.25 degree resolution",
        }

        if self.product is None:
            self.product = list(self.PRODUCTS)[0]

        if self.member == 0:
            self.member = "mem000"
        elif self.member == "control":
            self.member = "mem000"
        elif isinstance(self.member, int):
            self.member = f"mem{self.member:03d}"

        filedir = f"aigefs.{self.date:%Y%m%d/%H}"
        if self.member in ["spr", "avg"]:
            filepaths = {
                "sfc": f"{filedir}/ensstat/products/atmos/grib2/aigefs.t{self.date:%H}z.sfc.{self.member}.f{self.fxx:03d}",
                "pres": f"{filedir}/ensstat/products/atmos/grib2/aigefs.t{self.date:%H}z.pres.{self.member}.f{self.fxx:03d}",
            }
        else:
            filepaths = {
                "sfc": f"{filedir}/{self.member}/model/atmos/grib2/aigefs.t{self.date:%H}z.sfc.f{self.fxx:03d}",
                "pres": f"{filedir}/{self.member}/model/atmos/grib2/aigefs.t{self.date:%H}z.pres.f{self.fxx:03d}",
            }

        valid_members = {
            "pres": [f"mem{i:03d}" for i in range(0, 31)] + ["control", "spr", "avg"],
            "sfc": [f"mem{i:03d}" for i in range(0, 31)] + ["control", "spr", "avg"],
        }

        filepath = filepaths.get(self.product)
        if filepath is None:
            raise ValueError(
                f"product={self.product} not recognized. Must be one of {self.PRODUCTS.keys()}"
            )

        allowed_members = valid_members.get(self.product)
        if allowed_members is not None and self.member not in allowed_members:
            raise ValueError(
                f"For AIGEFS product {self.product}, member must be one of {allowed_members}"
            )

        sources = {
            "nomads": f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigefs/prod/{filepath}.grib2"
        }

        # Current AWS AIGEFS objects are available for ensemble members mem000-mem030.
        if self.member.startswith("mem"):
            sources = {
                "aws": f"https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/EAGLE_ensemble/{filepath}.grib2",
                **sources,
            }

        self.SOURCES = sources
        self.IDX_SUFFIX = [".grib2.idx", ".idx"]
        self.LOCALFILE = (
            f"aigefs.t{self.date:%H}z.{self.product}.{self.member}.f{self.fxx:03d}"
        )


def register_aws_aigfs_aigefs_templates():
    """Override Herbie model templates so `model='aigfs'/'aigefs'` use AWS paths."""
    import herbie.models as model_templates

    model_templates.aigfs = aigfs
    model_templates.aigefs = aigefs
