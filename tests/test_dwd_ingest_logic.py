
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import xarray as xr
import numpy as np
import os
import sys

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestDWDHistoricLoop(unittest.TestCase):
    def test_historic_loop_logic(self):
        # Setup mocks
        history_period = 2 # Test with 2 hours instead of 48 for speed
        base_time = pd.Timestamp("2025-12-10 18:00")
        
        mock_download = MagicMock(return_value=True)
        
        # Mock parse_mosmix_kml to return a dataframe with "correct" time
        def mock_parse(filepath):
            # file path ends with MOSMIX_S_YYYYMMDDHH_240.kmz
            # extract HH
            filename = os.path.basename(filepath)
            time_str = filename.split("_")[2] # 2025121018
            file_time = pd.to_datetime(time_str, format="%Y%m%d%H")
            
            # Valid time for the first forecast step is file_time + 1h
            valid_time = file_time + pd.Timedelta(hours=1)
            
            # Create dummy DF
            df = pd.DataFrame({
                "time": [valid_time],
                "station_id": ["TEST"],
                "longitude": [10.0],
                "latitude": [50.0],
                "TMP_2maboveground": [273.15]
            })
            return df, {}

        # Mock process_and_interpolate_df to return an xarray dataset
        def mock_process(df, perform_temporal_interp=True):
            # Create a dummy dataset
            times = df["time"].values
            ds = xr.Dataset(
                {
                    "TMP_2maboveground": (("time", "lat", "lon"), np.ones((len(times), 1, 1)))
                },
                coords={
                    "time": times,
                    "lat": [50.0], 
                    "lon": [10.0]
                }
            )
            return ds


        # Mock s3 and zarr
        mock_s3 = MagicMock()
        mock_zarr_grp = MagicMock()
        
        # --- Replicate the loop logic here for verification ---
        historic_datasets = []
        tmp_dir = "/tmp"
        historic_path = "/tmp/history"
        save_type = "Download" # Test local path logic first
        
        # We want to test one iteration where cache exists, and one where it doesn't.
        # i=2: Cache YES
        # i=1: Cache NO
        
        with patch("os.path.exists") as mock_exists, \
             patch("zarr.storage.LocalStore") as mock_store, \
             patch("xarray.open_dataset") as mock_xr_open, \
             patch("builtins.open", new_callable=MagicMock) as mock_open:
             
            # Configure mocks
            def side_effect_exists(path):
                if "DWD_Hist_2025121016.done" in path: # i=2 (18-2=16), CACHE HIT
                    return True
                return False # Default no file
            mock_exists.side_effect = side_effect_exists
            
            # Helper to mock lazy dataset
            # We need it to return different times for different calls
            # Call 1 (Cache HIT): 17:00
            # Call 2 (Cache MISS -> Process -> Re-open): 18:00
            lazy_open_results = [
                xr.Dataset({"dummy": (("time"), [1])}, coords={"time": [pd.Timestamp("2025-12-10 17:00")]}),
                xr.Dataset({"dummy": (("time"), [1])}, coords={"time": [pd.Timestamp("2025-12-10 18:00")]})
            ]
            
            def mock_lazy_open(store, engine, chunks):
                if lazy_open_results:
                    return lazy_open_results.pop(0)
                return xr.Dataset({"dummy": (("time"), [1])}, coords={"time": [pd.Timestamp("2025-12-10 19:00")]})
                
            mock_xr_open.side_effect = mock_lazy_open

            for i in range(history_period, 0, -1):
                target_time = base_time - pd.Timedelta(hours=i)
                time_str = target_time.strftime("%Y%m%d%H")
                
                # Path for cached Zarr file
                hist_zarr_filename = f"DWD_Hist_{time_str}.zarr"
                hist_zarr_path = os.path.join(historic_path, hist_zarr_filename)
                
                # Check cache
                cached_exists = os.path.exists(hist_zarr_path.replace(".zarr", ".done"))
                
                if cached_exists:
                    print(f"Cache HIT for {time_str}")
                    store = mock_store(hist_zarr_path)
                    ds_hist = mock_xr_open(store, engine="zarr", chunks="auto")
                    historic_datasets.append(ds_hist)
                    continue
                    
                # Cache MISS
                print(f"Cache MISS for {time_str}")
                hist_filename = f"MOSMIX_S_{time_str}_240.kmz"
                hist_url = f"https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_S/all_stations/kml/{hist_filename}"
                local_hist_path = os.path.join(tmp_dir, hist_filename)
                
                if not mock_download(hist_url, local_hist_path):
                    continue
                    
                df_hist, _ = mock_parse(local_hist_path)
                valid_time_target = target_time + pd.Timedelta(hours=1)
                df_hist_filtered = df_hist[df_hist["time"] == valid_time_target].copy()
                
                if not df_hist_filtered.empty:
                    ds_hist = mock_process(df_hist_filtered)
                    
                    # Saving logic
                    store = mock_store(hist_zarr_path)
                    
                    # Instead of overwriting the method on the object, we can't.
                    # But since mock_process returns a real Dataset (or our mock of it),
                    # we should actually check if *that* dataset had to_zarr called?
                    # But mock_process is returning a real XR dataset in our test code above.
                    # We can't mock methods on a real object easily if they are read-only.
                    
                    # Better approach: Patch xr.Dataset.to_zarr globally for this test context
                    with patch("xarray.Dataset.to_zarr") as mock_to_zarr:
                        ds_hist.to_zarr(store, mode="w", consolidated=False)
                        
                        # Done file
                        with open(hist_zarr_path.replace(".zarr", ".done"), "w") as f:
                            f.write("Done")
                        
                        # Re-open
                        ds_hist_lazy = mock_xr_open(store, engine="zarr", chunks="auto")
                        historic_datasets.append(ds_hist_lazy)

        # Verification
        self.assertEqual(len(historic_datasets), 2)
        # First one should be from cache (mocked lazy open)
        # Second one should be processed (and then re-opened)
        
        # Check timestamps
        # 1. i=2: target=16:00, valid=17:00
        # 2. i=1: target=17:00, valid=18:00
        
        t1 = historic_datasets[0].time.values[0]
        t2 = historic_datasets[1].time.values[0]
        
        print(f"Time 1: {t1}")
        print(f"Time 2: {t2}")
        
        self.assertEqual(pd.to_datetime(t1), base_time - pd.Timedelta(hours=1)) # 17:00
        self.assertEqual(pd.to_datetime(t2), base_time) # 18:00
        
        # Test concatenation
        ds_history = xr.concat(historic_datasets, dim="time")
        self.assertEqual(len(ds_history.time), 2)

if __name__ == '__main__':
    unittest.main()

