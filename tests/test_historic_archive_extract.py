import shutil
import tarfile
from pathlib import Path

import pytest
import zarr

pytest.importorskip("herbie")

from API.ingest_utils import download_extract_historic_archive


class FakeS3:
    def exists(self, path):
        return Path(path).exists()

    def get_file(self, src, dst):
        shutil.copyfile(src, dst)

    def rm(self, path, recursive=False):
        target = Path(path)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _write_zarr(path, variables):
    root = zarr.open_group(str(path), mode="w")
    for variable in variables:
        arr = root.create_array(
            variable, shape=(1, 1, 1), chunks=(1, 1, 1), dtype="float32"
        )
        arr[:] = 1


def _write_archive(historic_path, final_zarr_name, member_name, variables):
    source_zarr = historic_path / member_name
    _write_zarr(source_zarr, variables)

    archive_path = historic_path / f"{final_zarr_name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source_zarr, arcname=member_name)

    shutil.rmtree(source_zarr)
    done_path = historic_path / f"{final_zarr_name}.done"
    done_path.touch()
    return archive_path, done_path


def test_download_extract_historic_archive_validates_expected_vars(tmp_path):
    historic_path = tmp_path / "historic"
    historic_path.mkdir()
    local_temp_dir = tmp_path / "downloads"
    final_zarr_name = "Model_Hist_v320260528T000000Z.zarr"
    _write_archive(
        historic_path,
        final_zarr_name,
        "Model_Hist.zarr",
        ("time", "TMP_2maboveground"),
    )

    extracted_path = download_extract_historic_archive(
        s3=FakeS3(),
        historic_path=str(historic_path),
        final_zarr_name=final_zarr_name,
        extracted_store_name="Model_Hist.zarr",
        local_temp_dir=str(local_temp_dir),
        expected_vars=("time", "TMP_2maboveground"),
    )

    assert extracted_path == str(local_temp_dir / final_zarr_name)
    assert Path(extracted_path).exists()


def test_download_extract_historic_archive_deletes_s3_archive_on_missing_vars(tmp_path):
    historic_path = tmp_path / "historic"
    historic_path.mkdir()
    local_temp_dir = tmp_path / "downloads"
    final_zarr_name = "Model_Hist_v320260528T000000Z.zarr"
    archive_path, done_path = _write_archive(
        historic_path,
        final_zarr_name,
        "Model_Hist.zarr",
        ("time",),
    )

    with pytest.raises(ValueError, match="Missing variables"):
        download_extract_historic_archive(
            s3=FakeS3(),
            historic_path=str(historic_path),
            final_zarr_name=final_zarr_name,
            extracted_store_name="Model_Hist.zarr",
            local_temp_dir=str(local_temp_dir),
            expected_vars=("time", "TMP_2maboveground"),
        )

    assert not archive_path.exists()
    assert not done_path.exists()
    assert not (local_temp_dir / final_zarr_name).exists()
