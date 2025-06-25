import xarray as xr
import pandas as pd
from pykml import parser
from lxml import etree
import zipfile
import os
import io
import requests
import shutil
from datetime import datetime, timedelta
import random
import numpy as np
import zarr  # noqa: F401


# Define KML and DWD namespaces for easier parsing
KML_NAMESPACE = "{http://www.opengis.net/kml/2.2}"
DWD_NAMESPACE = (
    "{https://opendata.dwd.de/weather/lib/pointforecast_dwd_extension_V1_0.xsd}"
)


def generate_dummy_mosmix_kml(
    filename="dummy_mosmix.kml", num_stations=3, num_forecast_steps=72
):
    """
    Generates a dummy KML file resembling the DWD MOSMIX structure based on user's example.
    Includes global ForecastTimeSteps and dwd:Forecast with elementName and space-separated values.
    Dynamically generates the ReferencedModel section.
    Adds R101 (precipitation probability) and WW (present weather) parameters.
    """

    # Define XML namespaces at the root level for proper KML and DWD structure
    nsmap = {
        None: "http://www.opengis.net/kml/2.2",
        "dwd": "https://opendata.dwd.de/weather/lib/pointforecast_dwd_extension_V1_0.xsd",
        "gx": "http://www.google.com/kml/ext/2.2",
        "xal": "urn:oasis:names:tc:ciq:xsdschema:xAL:2.0",
        "atom": "http://www.w3.org/2005/Atom",
    }
    root = etree.Element(KML_NAMESPACE + "kml", nsmap=nsmap)
    document = etree.SubElement(root, KML_NAMESPACE + "Document")

    # --- Document-level ExtendedData (for global metadata and time steps) ---
    doc_extended_data = etree.SubElement(document, KML_NAMESPACE + "ExtendedData")
    product_definition = etree.SubElement(
        doc_extended_data, DWD_NAMESPACE + "ProductDefinition"
    )

    etree.SubElement(
        product_definition, DWD_NAMESPACE + "Issuer"
    ).text = "Deutscher Wetterdienst"
    etree.SubElement(product_definition, DWD_NAMESPACE + "ProductID").text = "MOSMIX"
    etree.SubElement(
        product_definition, DWD_NAMESPACE + "GeneratingProcess"
    ).text = "DWD MOSMIX hourly, Version 1.0"
    issue_time_str = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
    etree.SubElement(
        product_definition, DWD_NAMESPACE + "IssueTime"
    ).text = issue_time_str

    # Dynamically generate Referenced Models
    referenced_model_elem = etree.SubElement(
        product_definition, DWD_NAMESPACE + "ReferencedModel"
    )

    # Example dynamic models
    model_configs = [
        {
            "name": "ICON",
            "referenceTime": (
                datetime.utcnow() - timedelta(hours=random.randint(0, 6))
            ).isoformat(timespec="milliseconds")
            + "Z",
        },
        {
            "name": "ECMWF/IFS",
            "referenceTime": (
                datetime.utcnow() - timedelta(hours=random.randint(0, 6))
            ).isoformat(timespec="milliseconds")
            + "Z",
        },
        {
            "name": "COSMO",
            "referenceTime": (
                datetime.utcnow() - timedelta(hours=random.randint(0, 6))
            ).isoformat(timespec="milliseconds")
            + "Z",
        },
    ]
    # Randomly select a subset of models (e.g., 1 to 3 models)
    selected_models = random.sample(
        model_configs, k=random.randint(1, len(model_configs))
    )

    for model_data in selected_models:
        etree.SubElement(
            referenced_model_elem,
            DWD_NAMESPACE + "Model",
            name=model_data["name"],
            referenceTime=model_data["referenceTime"],
        )

    # Global ForecastTimeSteps
    forecast_time_steps = etree.SubElement(
        product_definition, DWD_NAMESPACE + "ForecastTimeSteps"
    )
    base_time = datetime.utcnow().replace(
        minute=0, second=0, microsecond=0
    ) + timedelta(hours=1)  # Start from next hour
    for j in range(num_forecast_steps):
        etree.SubElement(forecast_time_steps, DWD_NAMESPACE + "TimeStep").text = (
            base_time + timedelta(hours=j)
        ).isoformat(timespec="milliseconds") + "Z"

    format_cfg = etree.SubElement(product_definition, DWD_NAMESPACE + "FormatCfg")
    etree.SubElement(format_cfg, DWD_NAMESPACE + "DefaultUndefSign").text = "-"

    # --- Placemark section (for station data) ---
    folder = etree.SubElement(document, KML_NAMESPACE + "Folder")
    name_folder = etree.SubElement(folder, KML_NAMESPACE + "name")
    name_folder.text = "Stations"

    for i in range(num_stations):
        placemark = etree.SubElement(folder, KML_NAMESPACE + "Placemark")

        # Station ID (e.g., "01025") from kml:name
        name_pm = etree.SubElement(placemark, KML_NAMESPACE + "name")
        name_pm.text = f"{10000 + i}"  # Dummy station ID

        # Station Name (e.g., "TROMSOE") from kml:description
        description_pm = etree.SubElement(placemark, KML_NAMESPACE + "description")
        description_pm.text = (
            f"Station Name {chr(65 + i)}"  # e.g., Station Name A, Station Name B
        )

        # Corrected structure for Point and coordinates
        point = etree.SubElement(placemark, KML_NAMESPACE + "Point")
        coordinates_elem = etree.SubElement(point, KML_NAMESPACE + "coordinates")
        coordinates_elem.text = (
            f"{10.0 + i * 0.5},{50.0 + i * 0.5},{(10.0 + i * 5.0):.1f}"
        )

        # ExtendedData for forecast parameters at this station
        extended_data_pm = etree.SubElement(placemark, KML_NAMESPACE + "ExtendedData")

        # Generate PPPP (Pressure)
        pppp_values = []
        for j in range(num_forecast_steps):
            pppp_values.append(f"{100000.0 + i * 100 + j * 10:.2f}")
        dwd_forecast_pppp = etree.SubElement(
            extended_data_pm, DWD_NAMESPACE + "Forecast", elementName="PPPP"
        )
        dwd_value_pppp = etree.SubElement(dwd_forecast_pppp, DWD_NAMESPACE + "value")
        dwd_value_pppp.text = " ".join(pppp_values)

        # Generate TX (Maximum Temperature - with some missing values '-')
        tx_values = []
        for j in range(num_forecast_steps):
            if j % 5 == 0 or j % 7 == 0:  # Introduce some missing values
                tx_values.append("-")
            else:
                tx_values.append(f"{273.15 + 15.0 + i * 0.5 + j * 0.2:.2f}")  # KELVIN
        dwd_forecast_tx = etree.SubElement(
            extended_data_pm, DWD_NAMESPACE + "Forecast", elementName="TX"
        )
        dwd_value_tx = etree.SubElement(dwd_forecast_tx, DWD_NAMESPACE + "value")
        dwd_value_tx.text = " ".join(tx_values)

        # Generate TTT (Temperature at 2m)
        ttt_values = []
        for j in range(num_forecast_steps):
            ttt_values.append(f"{273.15 + 10.0 + i * 0.5 + j * 0.1:.2f}")  # KELVIN
        dwd_forecast_ttt = etree.SubElement(
            extended_data_pm, DWD_NAMESPACE + "Forecast", elementName="TTT"
        )
        dwd_value_ttt = etree.SubElement(dwd_forecast_ttt, DWD_NAMESPACE + "value")
        dwd_value_ttt.text = " ".join(ttt_values)

        # Generate Td (Dew Point Temperature at 2m)
        td_values = []
        for j in range(num_forecast_steps):
            td_values.append(f"{273.15 + 5.0 + i * 0.5 + j * 0.05:.2f}")  # KELVIN
        dwd_forecast_td = etree.SubElement(
            extended_data_pm, DWD_NAMESPACE + "Forecast", elementName="Td"
        )
        dwd_value_td = etree.SubElement(dwd_forecast_td, DWD_NAMESPACE + "value")
        dwd_value_td.text = " ".join(td_values)

        # Generate Rh (Relative Humidity)
        rh_values = []
        for j in range(num_forecast_steps):
            rh_values.append(f"{60.0 + i * 0.5 + random.uniform(-5, 5):.2f}")  # %
        dwd_forecast_rh = etree.SubElement(
            extended_data_pm, DWD_NAMESPACE + "Forecast", elementName="Rh"
        )
        dwd_value_rh = etree.SubElement(dwd_forecast_rh, DWD_NAMESPACE + "value")
        dwd_value_rh.text = " ".join(rh_values)

        # Generate FF (Wind Speed at 10m)
        ff_values = []
        for j in range(num_forecast_steps):
            ff_values.append(f"{5.0 + i * 0.2 + random.uniform(0, 2):.2f}")  # m/s
        dwd_forecast_ff = etree.SubElement(
            extended_data_pm, DWD_NAMESPACE + "Forecast", elementName="FF"
        )
        dwd_value_ff = etree.SubElement(dwd_forecast_ff, DWD_NAMESPACE + "value")
        dwd_value_ff.text = " ".join(ff_values)

        # Generate DD (Wind Direction at 10m)
        dd_values = []
        for j in range(num_forecast_steps):
            dd_values.append(f"{random.randint(0, 359):.0f}")  # degrees
        dwd_forecast_dd = etree.SubElement(
            extended_data_pm, DWD_NAMESPACE + "Forecast", elementName="DD"
        )
        dwd_value_dd = etree.SubElement(dwd_forecast_dd, DWD_NAMESPACE + "value")
        dwd_value_dd.text = " ".join(dd_values)

        # Generate FX (Wind Gust at 10m) - now using FX1 for dummy generation
        fx_values = []
        for j in range(num_forecast_steps):
            fx_values.append(f"{7.0 + i * 0.3 + random.uniform(0, 3):.2f}")  # m/s
        # Use FX1 for dummy generation to match expectation
        dwd_forecast_fx = etree.SubElement(
            extended_data_pm, DWD_NAMESPACE + "Forecast", elementName="FX1"
        )
        dwd_value_fx = etree.SubElement(dwd_forecast_fx, DWD_NAMESPACE + "value")
        dwd_value_fx.text = " ".join(fx_values)

        # Generate RR1c (Precipitation Total 1h)
        rr1c_values = []
        for j in range(num_forecast_steps):
            if j % 10 == 0:  # Simulate some rain events
                rr1c_values.append(f"{random.uniform(0.1, 2.0):.2f}")  # mm/h (kg/m^2/h)
            else:
                rr1c_values.append("0.00")
        dwd_forecast_rr1c = etree.SubElement(
            extended_data_pm, DWD_NAMESPACE + "Forecast", elementName="RR1c"
        )
        dwd_value_rr1c = etree.SubElement(dwd_forecast_rr1c, DWD_NAMESPACE + "value")
        dwd_value_rr1c.text = " ".join(rr1c_values)

        # Generate VV (Visibility)
        vv_values = []
        for j in range(num_forecast_steps):
            vv_values.append(f"{10000 + random.randint(0, 6000):.0f}")  # meters
        dwd_forecast_vv = etree.SubElement(
            extended_data_pm, DWD_NAMESPACE + "Forecast", elementName="VV"
        )
        dwd_value_vv = etree.SubElement(dwd_forecast_vv, DWD_NAMESPACE + "value")
        dwd_value_vv.text = " ".join(vv_values)

        # Generate N (Cloud Cover)
        n_values = []
        for j in range(num_forecast_steps):
            n_values.append(f"{random.randint(0, 100):.0f}")  # %
        dwd_forecast_n = etree.SubElement(
            extended_data_pm, DWD_NAMESPACE + "Forecast", elementName="N"
        )
        dwd_value_n = etree.SubElement(dwd_forecast_n, DWD_NAMESPACE + "value")
        dwd_value_n.text = " ".join(n_values)

        # Generate R101 (Probability of precipitation > 0.1 mm during the last hour)
        r101_values = []
        for j in range(num_forecast_steps):
            if float(rr1c_values[j]) > 0.0:
                r101_values.append(
                    f"{random.uniform(50, 100):.2f}"
                )  # High probability if precip
            else:
                r101_values.append(
                    f"{random.uniform(0, 20):.2f}"
                )  # Low probability if no precip
        dwd_forecast_r101 = etree.SubElement(
            extended_data_pm, DWD_NAMESPACE + "Forecast", elementName="R101"
        )
        dwd_value_r101 = etree.SubElement(dwd_forecast_r101, DWD_NAMESPACE + "value")
        dwd_value_r101.text = " ".join(r101_values)

        # Generate WW (Present weather codes)
        # Simplified WMO-like codes for dummy data:
        # 00-49: No precipitation / non-precip phenomena (e.g., clear, clouds, fog)
        # 50-59: Drizzle
        # 60-69: Rain
        # 70-79: Snow
        # 80-82: Showers (rain)
        # 83-86: Showers (snow/sleet)
        # 90-99: Thunderstorms
        ww_values = []
        for j in range(num_forecast_steps):
            if float(rr1c_values[j]) > 0.0:  # If there's dummy precipitation
                if float(ttt_values[j]) - 273.15 < 0:  # Below freezing (snow/ice)
                    ww_values.append(
                        str(random.choice([71, 73, 83]))
                    )  # Light/Moderate Snow or Snow Showers
                else:  # Above freezing (rain/drizzle)
                    ww_values.append(
                        str(random.choice([51, 61, 80]))
                    )  # Light/Moderate Drizzle/Rain or Rain Showers
            else:  # No precipitation
                ww_values.append(
                    str(random.choice([0, 1, 2, 3, 4, 10, 20, 21]))
                )  # Clear, clouds, fog/mist
        dwd_forecast_ww = etree.SubElement(
            extended_data_pm, DWD_NAMESPACE + "Forecast", elementName="WW"
        )
        dwd_value_ww = etree.SubElement(dwd_forecast_ww, DWD_NAMESPACE + "value")
        dwd_value_ww.text = " ".join(ww_values)

    tree = etree.ElementTree(root)
    with open(filename, "wb") as f:
        tree.write(f, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print(f"Dummy KML file generated at: {filename}")


def download_file(url, destination_path):
    """
    Downloads a file from a given URL to a specified destination path.
    """
    print(f"Attempting to download file from: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        with open(destination_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File downloaded successfully to: {destination_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {e}")
        return False
    except IOError as e:
        print(f"Error saving file to {destination_path}: {e}")
        return False


def parse_mosmix_kml(kml_filepath):
    """
    Parses a DWD MOSMIX KML/KMZ file and extracts forecast data into a Pandas DataFrame.
    It reads global forecast time steps from kml:Document and then station-specific
    forecast parameters from each kml:Placemark.
    It also extracts global metadata like ProductDefinition and ReferencedModels.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The extracted forecast data.
            - dict: A dictionary of global metadata.
    """

    # Handle KMZ files (which are essentially zipped KML)
    if kml_filepath.lower().endswith(".kmz"):
        try:
            with zipfile.ZipFile(kml_filepath, "r") as kmz_file:
                kml_content = None
                for name in kmz_file.namelist():
                    if name.lower().endswith(".kml"):
                        kml_content = kmz_file.read(name)
                        break
                if kml_content is None:
                    print(f"Error: No KML file found inside {kml_filepath}")
                    return pd.DataFrame(), {}

                root = parser.parse(io.BytesIO(kml_content)).getroot()
        except zipfile.BadZipFile:
            print(f"Error: {kml_filepath} is not a valid KMZ file.")
            return pd.DataFrame(), {}
    else:
        # Parse directly from KML file
        try:
            with open(kml_filepath, "rb") as f:  # Open in binary mode for lxml
                root = parser.parse(f).getroot()
        except FileNotFoundError:
            print(f"Error: KML file not found at {kml_filepath}")
            return pd.DataFrame(), {}
        except etree.XMLSyntaxError as e:
            print(f"Error parsing XML in {kml_filepath}: {e}")
            return pd.DataFrame(), {}

    data_records = []
    global_metadata = {}

    # --- 1. Extract Global Metadata from Document level ---
    product_definition_elem = root.find(
        f"./{KML_NAMESPACE}Document/{KML_NAMESPACE}ExtendedData/{DWD_NAMESPACE}ProductDefinition"
    )
    if product_definition_elem is not None:
        global_metadata["Issuer"] = product_definition_elem.findtext(
            f"{DWD_NAMESPACE}Issuer"
        )
        global_metadata["ProductID"] = product_definition_elem.findtext(
            f"{DWD_NAMESPACE}ProductID"
        )
        global_metadata["GeneratingProcess"] = product_definition_elem.findtext(
            f"{DWD_NAMESPACE}GeneratingProcess"
        )

        issue_time_text = product_definition_elem.findtext(f"{DWD_NAMESPACE}IssueTime")
        if issue_time_text:
            try:
                global_metadata["IssueTime"] = pd.to_datetime(issue_time_text)
            except ValueError:
                global_metadata["IssueTime"] = pd.NaT

        # Extract Referenced Models
        referenced_models = []
        referenced_model_elem = product_definition_elem.find(
            f"{DWD_NAMESPACE}ReferencedModel"
        )
        if referenced_model_elem is not None:
            for model_elem in referenced_model_elem.findall(f"{DWD_NAMESPACE}Model"):
                # Correctly retrieve namespaced attribute 'dwd:name'
                model_name = model_elem.get(
                    f"{DWD_NAMESPACE}name"
                )  # Use namespaced attribute
                reference_time_str = model_elem.get("referenceTime")
                if model_name and reference_time_str:
                    try:
                        reference_time = pd.to_datetime(reference_time_str)
                        referenced_models.append(
                            {"name": model_name, "referenceTime": reference_time}
                        )
                    except ValueError:
                        referenced_models.append(
                            {"name": model_name, "referenceTime": pd.NaT}
                        )
        global_metadata["ReferencedModels"] = referenced_models

    # --- 2. Extract Global Forecast Time Steps from Document level ---
    global_forecast_times = []
    # Path: /kml:kml/kml:Document/kml:ExtendedData/dwd:ProductDefinition/dwd:ForecastTimeSteps/dwd:TimeStep
    time_step_elements = root.findall(
        f"./{KML_NAMESPACE}Document/{KML_NAMESPACE}ExtendedData/{DWD_NAMESPACE}ProductDefinition/{DWD_NAMESPACE}ForecastTimeSteps/{DWD_NAMESPACE}TimeStep"
    )

    if not time_step_elements:
        print(
            "Error: No global forecast time steps found in the KML Document. Cannot proceed with data extraction."
        )
        return pd.DataFrame(), global_metadata

    for ts_elem in time_step_elements:
        try:
            global_forecast_times.append(pd.to_datetime(ts_elem.text))
        except ValueError:
            global_forecast_times.append(pd.NaT)  # Handle invalid time strings

    # Filter out NaT values if any, and ensure unique sorted times
    global_forecast_times = (
        pd.Series(global_forecast_times)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not global_forecast_times:
        print(
            "Error: No valid global forecast time steps after parsing. Cannot proceed with data extraction."
        )
        return pd.DataFrame(), global_metadata

    print(f"Found {len(global_forecast_times)} global forecast time steps.")

    # --- 3. Iterate through Placemarks to extract station-specific data ---
    for placemark in root.findall(f".//{KML_NAMESPACE}Placemark"):
        station_id_elem = placemark.find(f"{KML_NAMESPACE}name")
        station_id = (
            station_id_elem.text.strip()
            if station_id_elem is not None
            else "Unknown ID"
        )

        station_name_elem = placemark.find(f"{KML_NAMESPACE}description")
        station_name = (
            station_name_elem.text.strip()
            if station_name_elem is not None
            else "Unknown Name"
        )

        coordinates_elem = placemark.find(
            f"{KML_NAMESPACE}Point/{KML_NAMESPACE}coordinates"
        )
        lon, lat, alt = None, None, None
        if coordinates_elem is not None and coordinates_elem.text:
            coords = coordinates_elem.text.strip().split(",")
            try:
                lon = float(coords[0])
                lat = float(coords[1])
                alt = float(coords[2]) if len(coords) > 2 else 0.0
            except ValueError:
                print(
                    f"Warning: Could not parse coordinates for station ID {station_id} ({station_name}): {coordinates_elem.text}"
                )

        extended_data_elem = placemark.find(f"{KML_NAMESPACE}ExtendedData")
        if extended_data_elem is None:
            print(
                f"Warning: No ExtendedData found for station ID {station_id} ({station_name}). Skipping."
            )
            continue

        station_forecast_data = {}  # To store {param_name: [value1, value2, ...]}

        # Parse dwd:Forecast elements within this Placemark's ExtendedData
        for forecast_elem in extended_data_elem.findall(f"{DWD_NAMESPACE}Forecast"):
            # Correctly retrieve namespaced attribute
            element_name = forecast_elem.get(f"{DWD_NAMESPACE}elementName")

            if element_name is None:
                continue  # Skip this element if its name cannot be determined

            value_elem = forecast_elem.find(f"{DWD_NAMESPACE}value")
            if value_elem is not None and value_elem.text:
                values_str = value_elem.text.strip().split()

                # Convert values, handling '-' for missing data
                param_values = []
                for val in values_str:
                    if val == "-":
                        param_values.append(float("nan"))  # Use NaN for missing values
                    else:
                        try:
                            # WW codes are integers, but parsing as float first handles NaN
                            if element_name == "WW":
                                param_values.append(float(val))
                            else:
                                param_values.append(float(val))
                        except ValueError:
                            param_values.append(
                                float("nan")
                            )  # Handle other non-numeric strings

                # Crucial check: Ensure the number of values matches global time steps
                if len(param_values) == len(global_forecast_times):
                    station_forecast_data[element_name] = param_values
                else:
                    print(
                        f"Warning: Mismatch in number of values for '{element_name}' ({len(param_values)}) and global times ({len(global_forecast_times)}) for station ID {station_id} ({station_name}). Skipping this parameter for this station."
                    )

        # Create a record for each (time, station) pair for this station
        for i, ftime in enumerate(global_forecast_times):
            record = {
                "station_id": station_id,
                "station_name": station_name,
                "longitude": lon,
                "latitude": lat,
                "altitude": alt,
                "time": ftime,
            }
            for param, values in station_forecast_data.items():
                record[param] = values[i]
            data_records.append(record)

    df = pd.DataFrame(data_records)

    # Ensure 'time' column is datetime and sort
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")  # Coerce errors to NaT
        df = df.dropna(subset=["time"])  # Drop rows where time parsing failed
        df = df.sort_values(by=["station_id", "time"]).reset_index(drop=True)

    return df, global_metadata


def convert_df_to_xarray(df, global_metadata=None):
    """
    COnverts a Pandas DataFrame to an xarray Dataset.
    Assumes 'station_id' and 'time' are dimensions.
    'station_name', 'longitude', 'latitude', 'altitude' become coordinates.
    Global metadata can be added as Dataset attributes.
    """
    attrs_to_add = None

    if df.empty:
        print("DataFrame is empty, cannot convert to xarray Dataset.")
        return xr.Dataset()

    # Convert 'time' column to timezone-naive UTC before setting index
    # This prevents the 'datetime64[ns, UTC]' error during Zarr serialization.
    if "time" in df.columns and pd.api.types.is_datetime64tz_dtype(df["time"]):
        df["time"] = df["time"].dt.tz_localize(
            None
        )  # Remove timezone information, assuming it's UTC

    # Identify unique station details to become coordinates
    unique_stations_df = df[
        ["station_id", "station_name", "longitude", "latitude", "altitude"]
    ].drop_duplicates(subset=["station_id"])
    unique_stations_df = unique_stations_df.set_index(
        "station_id"
    )  # Index by station_id for easy lookup

    # Columns that will become data variables (excluding dimensions and coordinates)
    data_vars_cols = [
        col
        for col in df.columns
        if col
        not in [
            "station_id",
            "time",
            "station_name",
            "longitude",
            "latitude",
            "altitude",
        ]
    ]

    if not data_vars_cols:
        print(
            "No forecast parameters found to create data variables in xarray Dataset. The DataFrame might only contain coordinate-like columns."
        )
        # Create an empty Dataset with only dimensions and coordinates if no data variables
        if not df.empty:
            ds = xr.Dataset(
                coords={
                    "station_id": df["station_id"].unique(),
                    "time": df["time"].unique(),
                }
            )
            if not unique_stations_df.empty:
                ds = ds.assign_coords(
                    station_name=(
                        "station_id",
                        unique_stations_df["station_name"]
                        .reindex(ds.station_id)
                        .values,
                    ),
                    longitude=(
                        "station_id",
                        unique_stations_df["longitude"].reindex(ds.station_id).values,
                    ),
                    latitude=(
                        "station_id",
                        unique_stations_df["latitude"].reindex(ds.station_id).values,
                    ),
                    altitude=(
                        "station_id",
                        unique_stations_df["altitude"].reindex(ds.station_id).values,
                    ),
                )
            # Add global metadata as attributes to the Dataset if provided
            if global_metadata:
                ds.attrs.update(attrs_to_add)
            return ds
        return xr.Dataset()

    # Set 'station_id' and 'time' as multi-index for to_xarray
    ds = df.set_index(["station_id", "time"])[data_vars_cols].to_xarray()

    # Assign static station properties as coordinates to the 'station_id' dimension
    if not unique_stations_df.empty:
        ds = ds.assign_coords(
            station_name=(
                "station_id",
                unique_stations_df["station_name"].reindex(ds.station_id).values,
            ),
            longitude=(
                "station_id",
                unique_stations_df["longitude"].reindex(ds.station_id).values,
            ),
            latitude=(
                "station_id",
                unique_stations_df["latitude"].reindex(ds.station_id).values,
            ),
            altitude=(
                "station_id",
                unique_stations_df["altitude"].reindex(ds.station_id).values,
            ),
        )

    # Add global metadata as attributes to the Dataset if provided
    if global_metadata:
        # Convert datetime objects in metadata to strings for Zarr compatibility, if needed
        attrs_to_add = {}
        for k, v in global_metadata.items():
            if isinstance(v, pd.Timestamp):
                attrs_to_add[k] = v.isoformat()
            elif isinstance(v, list) and all(
                isinstance(item, dict)
                and "referenceTime" in item
                and isinstance(item["referenceTime"], pd.Timestamp)
                for item in v
            ):
                # Convert list of model dicts to a more Zarr-friendly string representation
                attrs_to_add[k] = [
                    {
                        m_k: (m_v.isoformat() if isinstance(m_v, pd.Timestamp) else m_v)
                        for m_k, m_v in model.items()
                    }
                    for model in v
                ]
            else:
                attrs_to_add[k] = v
        ds.attrs.update(attrs_to_add)

    return ds


def save_to_zarr(xarray_dataset, zarr_path):
    """
    Saves an xarray Dataset to a Zarr store.
    """
    if xarray_dataset.dims:  # Check if dataset is not empty (has dimensions)
        try:
            # Removed 'overwrite=True' as 'mode='w'' handles overwriting
            xarray_dataset.to_zarr(zarr_path, mode="w", compute=True)
            print(f"Data successfully saved to Zarr at: {zarr_path}")
        except Exception as e:
            print(f"Error saving to Zarr: {e}")
    else:
        print("Xarray Dataset is empty (no dimensions found), nothing to save to Zarr.")


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on Earth using the Haversine formula.

    Args:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

    Returns:
        float: Distance in kilometers.
    """
    R = 6371  # Radius of Earth in kilometers
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


def _get_precip_type_from_ww(ww_code):
    """
    Determines precipitation type (rain, snow, ice, or none) from DWD WW code.
    This is a simplified mapping based on common WMO present weather codes.
    Source: General understanding of WMO codes (e.g., 50s-drizzle, 60s-rain, 70s-snow, 80s-showers, freezing rain is specific).

    Args:
        ww_code (int or float): The WW (present weather) code. Can be NaN if missing.

    Returns:
        str: "rain", "snow", "ice", or "none".
    """
    if np.isnan(ww_code):
        return "none"  # Cannot determine if code is missing

    ww_code = int(ww_code)  # Convert to integer for comparison

    # No precipitation / non-precipitation phenomena
    if 0 <= ww_code <= 49:
        return "none"
    # Drizzle
    elif 50 <= ww_code <= 59:
        return "rain"  # Drizzle is liquid
    # Rain
    elif 60 <= ww_code <= 69:
        if ww_code in [66, 67]:  # Freezing Rain
            return "ice"
        return "rain"
    # Snow, ice pellets, hail
    elif 70 <= ww_code <= 79:
        if ww_code in [79]:  # Ice pellets / hail
            return "ice"
        return "snow"
    # Showers (rain, snow, mixed)
    elif 80 <= ww_code <= 89:
        if ww_code in [83, 84, 85, 86]:  # Snow/sleet showers
            return "snow"
        if ww_code in [87, 88]:  # Hail showers
            return "ice"
        return "rain"  # Default for other showers (rain)
    # Thunderstorms
    elif 90 <= ww_code <= 99:
        # Thunderstorms can have various precip types, simplified here
        if ww_code in [95, 96, 97, 99]:  # Thunderstorm with hail, ice pellets
            return "ice"
        return "rain"  # Default to rain for other thunderstorms

    return "none"  # Fallback for unknown codes


def get_nearest_station_forecast(
    target_lat, target_lon, zarr_path, max_distance_km=None
):
    """
    Pulls forecast data from the nearest DWD MOSMIX station within a Zarr dataset
    for a given set of coordinates, with an optional maximum search distance.

    Args:
        target_lat (float): Latitude of the target location.
        target_lon (float): Longitude of the target location.
        zarr_path (str): Path to the Zarr dataset.
        max_distance_km (float, optional): Maximum distance in kilometers to search for a station.
                                            If None, no distance limit is applied.
                                            If a station is found further than this, returns empty data.

    Returns:
        tuple: A tuple containing:
            - list: A list of dictionaries, where each dictionary represents
                    forecast data for a time step at the nearest station.
                    Includes 'distance_km' to the nearest station.
                    Returns empty list if no station found within max_distance_km.
            - dict: Information about the nearest station (id, name, lat, lon, dist).
                    Returns empty dict if no station found within max_distance_km.
    """
    try:
        ds = xr.open_zarr(zarr_path)
    except Exception as e:
        print(f"Error loading Zarr dataset from {zarr_path}: {e}")
        return [], {}

    if (
        "station_id" not in ds.coords
        or "latitude" not in ds.coords
        or "longitude" not in ds.coords
    ):
        print("Error: Zarr dataset does not contain required station coordinates.")
        return [], {}

    # Get station IDs and their corresponding latitudes and longitudes directly from coordinates
    station_ids_values = ds.coords["station_id"].values
    station_latitudes = ds.coords["latitude"].values
    station_longitudes = ds.coords["longitude"].values

    # Calculate distances to all stations
    distances = []
    for i, station_id_val in enumerate(station_ids_values):
        station_lat = station_latitudes[i]
        station_lon = station_longitudes[i]
        dist = haversine_distance(target_lat, target_lon, station_lat, station_lon)
        distances.append(dist)

    # Find the nearest station
    if not distances:
        print("No station data available in the dataset.")
        return [], {}

    nearest_station_idx = np.argmin(distances)
    nearest_station_id = station_ids_values[nearest_station_idx]
    nearest_distance_km = distances[nearest_station_idx]

    # --- New logic for max_distance_km ---
    if max_distance_km is not None and nearest_distance_km > max_distance_km:
        print(
            f"Nearest station ({nearest_station_id}, {nearest_distance_km:.2f} km) is beyond the maximum allowed distance of {max_distance_km:.2f} km."
        )
        return [], {}  # Return empty if outside limit
    # --- End new logic ---

    nearest_station_data = ds.sel(station_id=nearest_station_id)

    # Prepare station information
    nearest_station_info = {
        "station_id": nearest_station_id,
        "station_name": nearest_station_data.coords["station_name"].item(),
        "latitude": nearest_station_data.coords["latitude"].item(),
        "longitude": nearest_station_data.coords["longitude"].item(),
        "distance_km": nearest_distance_km,
    }

    forecast_output = []

    # Iterate through each forecast time step for the nearest station
    for time_step in nearest_station_data["time"].values:
        single_time_forecast = nearest_station_data.sel(time=time_step)

        # Initialize output dict for this time step
        output_data = {
            # Convert to UNIX timestamp (integer seconds since epoch)
            "time": int(pd.Timestamp(time_step).timestamp()),
            "distance_km": nearest_distance_km,
        }

        # Mapping DWD parameters to requested output names and applying conversions
        # Units:
        # TTT, Td: Kelvin -> Celsius
        # PPPP: Pa -> hPa
        # FF, FX1: m/s -> km/h (FX1 for wind gust)
        # RR1c: mm/h (direct)
        # VV: meters -> km (capped at 16.09 km)
        # N, Rh, DD, R101, WW: direct (except WW is then mapped to type string)

        # --- Temperature (degrees C) ---
        if "TTT" in single_time_forecast.data_vars and not np.isnan(
            single_time_forecast["TTT"].item()
        ):
            output_data["temperature"] = (
                single_time_forecast["TTT"].item() - 273.15
            )  # Kelvin to Celsius

        # --- Dew Point (degrees C) ---
        if "Td" in single_time_forecast.data_vars and not np.isnan(
            single_time_forecast["Td"].item()
        ):
            output_data["dewPoint"] = (
                single_time_forecast["Td"].item() - 273.15
            )  # Kelvin to Celsius

        # --- Humidity (%) ---
        if "Rh" in single_time_forecast.data_vars and not np.isnan(
            single_time_forecast["Rh"].item()
        ):
            output_data["humidity"] = single_time_forecast["Rh"].item()

        # --- Pressure (hPa) ---
        if "PPPP" in single_time_forecast.data_vars and not np.isnan(
            single_time_forecast["PPPP"].item()
        ):
            output_data["pressure"] = (
                single_time_forecast["PPPP"].item() / 100
            )  # Pa to hPa

        # --- Wind Speed (km/h) ---
        if "FF" in single_time_forecast.data_vars and not np.isnan(
            single_time_forecast["FF"].item()
        ):
            output_data["windSpeed"] = (
                single_time_forecast["FF"].item() * 3.6
            )  # m/s to km/h

        # --- Wind Gust (km/h) ---
        if "FX1" in single_time_forecast.data_vars and not np.isnan(
            single_time_forecast["FX1"].item()
        ):  # Changed from 'FX' to 'FX1'
            output_data["windGust"] = (
                single_time_forecast["FX1"].item() * 3.6
            )  # m/s to km/h

        # --- Wind Bearing (degrees) ---
        if "DD" in single_time_forecast.data_vars and not np.isnan(
            single_time_forecast["DD"].item()
        ):
            output_data["windBearing"] = single_time_forecast["DD"].item()

        # --- Cloud Cover (%) ---
        if "N" in single_time_forecast.data_vars and not np.isnan(
            single_time_forecast["N"].item()
        ):
            output_data["cloudCover"] = single_time_forecast[
                "N"
            ].item()  # Assuming N is already in %

        # --- Visibility (km with a cap of 16.09 km) ---
        if "VV" in single_time_forecast.data_vars and not np.isnan(
            single_time_forecast["VV"].item()
        ):
            visibility_km = single_time_forecast["VV"].item() / 1000  # meters to km
            output_data["visibility"] = min(visibility_km, 16.09)  # Apply cap

        # --- Precipitation Intensity (mm/h) ---
        if "RR1c" in single_time_forecast.data_vars and not np.isnan(
            single_time_forecast["RR1c"].item()
        ):
            output_data["precipIntensity"] = single_time_forecast[
                "RR1c"
            ].item()  # mm/h (kg/m^2/h)

        # --- Precipitation Probability (%) (Using R101) ---
        if "R101" in single_time_forecast.data_vars and not np.isnan(
            single_time_forecast["R101"].item()
        ):
            output_data["precipProbability"] = single_time_forecast["R101"].item()  # %

        # --- Precipitation Accumulation (cm) (Current intensity converted to cm) ---
        if "precipIntensity" in output_data and not np.isnan(
            output_data["precipIntensity"]
        ):
            output_data["precipAccumulation"] = (
                output_data["precipIntensity"] / 10.0
            )  # mm/h to cm/h

        # --- Precipitation Type (rain, snow, ice or none) (Using WW) ---
        if "WW" in single_time_forecast.data_vars and not np.isnan(
            single_time_forecast["WW"].item()
        ):
            output_data["precipType"] = _get_precip_type_from_ww(
                single_time_forecast["WW"].item()
            )
        elif "precipIntensity" in output_data and output_data["precipIntensity"] > 0:
            # Fallback if WW is missing but precip exists: infer from temperature
            if "temperature" in output_data and not np.isnan(
                output_data["temperature"]
            ):
                output_data["precipType"] = (
                    "snow" if output_data["temperature"] < 0 else "rain"
                )
            else:
                output_data["precipType"] = (
                    "rain"  # Default if temperature also missing
                )
        else:
            output_data["precipType"] = "none"  # No precipitation or no info

        forecast_output.append(output_data)

    return forecast_output, nearest_station_info


# --- Example Usage ---
if __name__ == "__main__":

    # --- Configuration ---
    # Set to True to download and process the actual DWD file
    # Set to False to generate and use a dummy KML file for testing
    download_actual_data = True

    dwd_mosmix_url = "https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_S/all_stations/kml/MOSMIX_S_LATEST_240.kmz"
    downloaded_kmz_file = "MOSMIX_S_LATEST_240.kmz"
    dummy_kml_file = "dwd_mosmix_dummy.kml"
    zarr_output_path = "dwd_mosmix_forecast.zarr"

    kml_source_file = None  # Will store the path to the KML/KMZ file to be parsed

    # --- Step 1: Get the KML/KMZ source file (download or generate dummy) ---
    if download_actual_data:
        print("\n--- Attempting to download actual DWD MOSMIX data ---")
        if download_file(dwd_mosmix_url, downloaded_kmz_file):
            kml_source_file = downloaded_kmz_file
        else:
            print(
                "Failed to download actual data. Proceeding with dummy data for demonstration."
            )
            # For real data, 240 steps might mean 10 days of hourly data.
            # Dummy data to match this length:
            generate_dummy_mosmix_kml(
                dummy_kml_file, num_stations=5, num_forecast_steps=240
            )
            kml_source_file = dummy_kml_file
    else:
        print("\n--- Generating dummy KML file for testing ---")
        generate_dummy_mosmix_kml(
            dummy_kml_file, num_stations=5, num_forecast_steps=240
        )
        kml_source_file = dummy_kml_file

    if kml_source_file is None or not os.path.exists(kml_source_file):
        print("Error: No valid KML/KMZ source file available. Exiting.")
        exit()

    # --- Step 2: Read and parse the KML file into a Pandas DataFrame and extract global metadata ---
    print(f"\n--- Reading and Parsing KML from {kml_source_file} ---")
    df_data, global_metadata = parse_mosmix_kml(kml_source_file)

    if not df_data.empty:
        print("\nSuccessfully extracted data (first 5 rows of DataFrame):")
        print(df_data.head())
        print(f"\nDataFrame shape: {df_data.shape}")
        print(f"DataFrame columns: {df_data.columns.tolist()}")
        print("\nGlobal Metadata Extracted:")
        for key, value in global_metadata.items():
            print(f"  {key}: {value}")

        # --- Step 3: Convert Pandas DataFrame to xarray Dataset ---
        print("\n--- Converting DataFrame to xarray Dataset ---")
        ds = convert_df_to_xarray(df_data, global_metadata=global_metadata)

        if ds.dims:
            print("\nxarray Dataset created:")
            print(ds)
            print(f"\nDataset dimensions: {ds.dims}")
            print(f"Dataset coordinates: {ds.coords}")
            print(f"Dataset data variables: {list(ds.data_vars.keys())}")
            print("\nDataset Global Attributes (from metadata):")
            for attr_key, attr_value in ds.attrs.items():
                print(f"  {attr_key}: {attr_value}")

            # --- Step 4: Save xarray Dataset to Zarr ---
            print("\n--- Saving xarray Dataset to Zarr ---")
            save_to_zarr(ds, zarr_output_path)

            # Optional: Verify by loading the Zarr file
            print(f"\n--- Verifying Zarr file by loading from {zarr_output_path} ---")
            try:
                ds_from_zarr = xr.open_zarr(zarr_output_path)
                print("\nSuccessfully loaded Zarr file:")
                print(ds_from_zarr)
                # Access a specific data variable and its values
                if "PPPP" in ds_from_zarr.data_vars:
                    print("\nData check (first few PPPP values for first station):")
                    print(ds_from_zarr["PPPP"].isel(station_id=0).values[:5])
                if "TX" in ds_from_zarr.data_vars:
                    print("\nData check (first few TX values for first station):")
                    print(ds_from_zarr["TX"].isel(station_id=0).values[:5])
                print("\nLoaded Dataset Global Attributes:")
                for attr_key, attr_value in ds_from_zarr.attrs.items():
                    print(f"  {attr_key}: {attr_value}")
            except Exception as e:
                print(f"Error loading Zarr file: {e}")

            # --- Step 5: Get forecast for specific coordinates ---
            print("\n--- Getting forecast for specific coordinates ---")
            # Example 1: Coordinates in Germany (likely to find a station)
            target_latitude_germany = 50.1109  # Frankfurt am Main
            target_longitude_germany = 8.6821
            max_distance_germany = 10.0  # km

            print(
                f"\nAttempting to find station near ({target_latitude_germany}, {target_longitude_germany}) within {max_distance_germany} km:"
            )
            forecast_data_germany, station_info_germany = get_nearest_station_forecast(
                target_latitude_germany,
                target_longitude_germany,
                zarr_output_path,
                max_distance_km=max_distance_germany,
            )

            if forecast_data_germany:
                print(
                    f"\n--- Displaying Forecast for Coordinates ({target_latitude_germany}, {target_longitude_germany}) ---"
                )
                print(
                    f"Nearest Station Found: {station_info_germany['station_name']} (ID: {station_info_germany['station_id']})"
                )
                print(
                    f"Distance from Coordinates to Station: {station_info_germany['distance_km']:.2f} km"
                )

                print(
                    f"\nForecast details for {station_info_germany['station_name']} (first {min(5, len(forecast_data_germany))} time steps):"
                )
                for i, fcst in enumerate(
                    forecast_data_germany[:5]
                ):  # Still limiting to 5 for console readability
                    print(
                        f"\n--- Time Step: {datetime.fromtimestamp(fcst['time']).strftime('%Y-%m-%d %H:%M UTC')} ---"
                    )
                    # Only print if the key exists and value is not NaN
                    if "precipIntensity" in fcst and not np.isnan(
                        fcst["precipIntensity"]
                    ):
                        print(f"  precipIntensity: {fcst['precipIntensity']:.2f} mm/h")
                    if "precipProbability" in fcst and not np.isnan(
                        fcst["precipProbability"]
                    ):
                        print(f"  precipProbability: {fcst['precipProbability']:.2f}%")
                    if "precipAccumulation" in fcst and not np.isnan(
                        fcst["precipAccumulation"]
                    ):
                        print(
                            f"  precipAccumulation: {fcst['precipAccumulation']:.2f} cm"
                        )
                    if (
                        "precipType" in fcst
                    ):  # precipType should always be a string or "none"
                        print(f"  precipType: {fcst['precipType']}")
                    if "temperature" in fcst and not np.isnan(fcst["temperature"]):
                        print(f"  temperature: {fcst['temperature']:.2f}°C")
                    if "dewPoint" in fcst and not np.isnan(fcst["dewPoint"]):
                        print(f"  dewPoint: {fcst['dewPoint']:.2f}°C")
                    if "humidity" in fcst and not np.isnan(fcst["humidity"]):
                        print(f"  humidity: {fcst['humidity']:.2f}%")
                    if "pressure" in fcst and not np.isnan(fcst["pressure"]):
                        print(f"  pressure: {fcst['pressure']:.2f} hPa")
                    if "windSpeed" in fcst and not np.isnan(fcst["windSpeed"]):
                        print(f"  windSpeed: {fcst['windSpeed']:.2f} km/h")
                    if "windGust" in fcst and not np.isnan(fcst["windGust"]):
                        print(f"  windGust: {fcst['windGust']:.2f} km/h")
                    if "windBearing" in fcst and not np.isnan(fcst["windBearing"]):
                        print(f"  windBearing: {fcst['windBearing']:.0f}°")
                    if "cloudCover" in fcst and not np.isnan(fcst["cloudCover"]):
                        print(f"  cloudCover: {fcst['cloudCover']:.2f}%")
                    if "visibility" in fcst and not np.isnan(fcst["visibility"]):
                        print(f"  visibility: {fcst['visibility']:.2f} km")
                    print("-" * 30)  # Separator for each time step
            else:
                print(
                    f"No forecast data retrieved for ({target_latitude_germany}, {target_longitude_germany}) within {max_distance_germany} km. This might be expected if no station is close enough."
                )

            # Example 2: Coordinates at (0,0) with a small limit (likely to return nothing)
            target_latitude_equator = 0.0
            target_longitude_equator = 0.0
            max_distance_equator = 10.0  # km

            print(
                f"\nAttempting to find station near ({target_latitude_equator}, {target_longitude_equator}) within {max_distance_equator} km:"
            )
            forecast_data_equator, station_info_equator = get_nearest_station_forecast(
                target_latitude_equator,
                target_longitude_equator,
                zarr_output_path,
                max_distance_km=max_distance_equator,
            )

            if forecast_data_equator:
                print(
                    f"\n--- Displaying Forecast for Coordinates ({target_latitude_equator}, {target_longitude_equator}) ---"
                )
                print(
                    f"Nearest Station Found: {station_info_equator['station_name']} (ID: {station_info_equator['station_id']})"
                )
                print(
                    f"Distance from Coordinates to Station: {station_info_equator['distance_km']:.2f} km"
                )

                print(
                    f"\nForecast details for {station_info_equator['station_name']} (first {min(5, len(forecast_data_equator))} time steps):"
                )
                for i, fcst in enumerate(forecast_data_equator[:5]):
                    print(
                        f"\n--- Time Step: {datetime.fromtimestamp(fcst['time']).strftime('%Y-%m-%d %H:%M UTC')} ---"
                    )
                    # Only print if the key exists and value is not NaN
                    if "precipIntensity" in fcst and not np.isnan(
                        fcst["precipIntensity"]
                    ):
                        print(f"  precipIntensity: {fcst['precipIntensity']:.2f} mm/h")
                    if "precipProbability" in fcst and not np.isnan(
                        fcst["precipProbability"]
                    ):
                        print(f"  precipProbability: {fcst['precipProbability']:.2f}%")
                    if "precipAccumulation" in fcst and not np.isnan(
                        fcst["precipAccumulation"]
                    ):
                        print(
                            f"  precipAccumulation: {fcst['precipAccumulation']:.2f} cm"
                        )
                    if (
                        "precipType" in fcst
                    ):  # precipType should always be a string or "none"
                        print(f"  precipType: {fcst['precipType']}")
                    if "temperature" in fcst and not np.isnan(fcst["temperature"]):
                        print(f"  temperature: {fcst['temperature']:.2f}°C")
                    if "dewPoint" in fcst and not np.isnan(fcst["dewPoint"]):
                        print(f"  dewPoint: {fcst['dewPoint']:.2f}°C")
                    if "humidity" in fcst and not np.isnan(fcst["humidity"]):
                        print(f"  humidity: {fcst['humidity']:.2f}%")
                    if "pressure" in fcst and not np.isnan(fcst["pressure"]):
                        print(f"  pressure: {fcst['pressure']:.2f} hPa")
                    if "windSpeed" in fcst and not np.isnan(fcst["windSpeed"]):
                        print(f"  windSpeed: {fcst['windSpeed']:.2f} km/h")
                    if "windGust" in fcst and not np.isnan(fcst["windGust"]):
                        print(f"  windGust: {fcst['windGust']:.2f} km/h")
                    if "windBearing" in fcst and not np.isnan(fcst["windBearing"]):
                        print(f"  windBearing: {fcst['windBearing']:.0f}°")
                    if "cloudCover" in fcst and not np.isnan(fcst["cloudCover"]):
                        print(f"  cloudCover: {fcst['cloudCover']:.2f}%")
                    if "visibility" in fcst and not np.isnan(fcst["visibility"]):
                        print(f"  visibility: {fcst['visibility']:.2f} km")
                    print("-" * 30)  # Separator for each time step
            else:
                print(
                    f"No forecast data retrieved for ({target_latitude_equator}, {target_longitude_equator}) within {max_distance_equator} km. This might be expected if no station is close enough."
                )

        else:
            print("Xarray Dataset is empty. Check parsing and conversion steps.")
    else:
        print(
            f"Zarr output path '{zarr_output_path}' not found. Cannot perform forecast query."
        )

    # --- Cleanup ---
    if os.path.exists(dummy_kml_file):
        os.remove(dummy_kml_file)
        print(f"\nCleaned up dummy KML file: {dummy_kml_file}")

    if os.path.exists(downloaded_kmz_file):
        os.remove(downloaded_kmz_file)
        print(f"Cleaned up downloaded KMZ file: {downloaded_kmz_file}")

    if os.path.exists(zarr_output_path) and os.path.isdir(zarr_output_path):
        shutil.rmtree(zarr_output_path)
        print(f"Cleaned up Zarr directory: {zarr_output_path}")
