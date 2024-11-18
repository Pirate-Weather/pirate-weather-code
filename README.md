# Pirate Weather Code

This repository is the source for the code that runs the Pirate Weather API, a free, fast (~10 ms), Open-Source (AGPL), minimalist Weather API. If you're interested in using the API, the primary documentation on the weather variables and data usage is contained in the [main repository](https://github.com/Pirate-Weather/pirateweather), with this repository focused on the code that ingests the source weather data and produces the forecast!

## Status
While the production API service is stable and well tested, running the API on your own hardware is still very beta. There are a number of structural changes between the production service and this version, which could result in instabilities or data errors when running on your own hardware. This should improve in the coming months, and PRs are **eagerly welcomed** to speed up this process! Historic data is also not currently functioning, although should be possible in the future!

## Philosophy
Weather forecasts are largely produced by public agencies, and commercial weather APIs take this raw data, perform some secret black box processing, and then charge for it. While I have nothing against this, I thought there should be a service that: A) lighting fast, B) explains exactly what is happening to the data between the source and an API, and B) provide results as close to the raw data as possible. This means that Pirate Weather aims to do as little processing as practically required to go from raw model results to usable forecasts, providing an important data source for specialty applications where transparent data is vital. It also formats the required data using the same syntax and structure as the Dark Sky API response, which is important for some legacy applications.  

In implementing this, I wanted to make it as simple as possible while building off open tools. Python is used for all code, and a number of open source projects are extensively used, including:
- [Herbie](https://herbie.readthedocs.io/en/stable/ "Herbie")
- [Zarr](https://zarr.dev/ "Zarr")
- [Dask](https://www.dask.org/ "Dask")
- [WGRIB2](https://github.com/NOAA-EMC/wgrib2/releases "WGRIB2")
- [FastAPI](https://fastapi.tiangolo.com/ "FastAPI")

The goal with this philosophy is to avoid reinventing the wheel while making it easy for other people to use, contribute, and extend this project! It also means that if anyone has expertise in working with these tools, I'd love to hear your comments and suggestions.

## Overview
At a high level, a conceptual overview of this service is described on the [blog](http://pirateweather.net/en/latest/Blog/InfrastructureOverview/), with the key differences required for self-hosting being:
- Switch from Eventbridge to [Ofelia](https://github.com/mcuadros/ofelia) for scheduling.
- Docker container instead of response generation (front-end).
- Removal of the Kong/ AWS API gateway requirement

To run the service, clone this repository to a machine, edit the [docker compose file](https://github.com/Pirate-Weather/pirate-weather-code/blob/main/pirate-compose_oph) to match local paths and bring it up! The key variables to change in the compose script are the volumes, since by default they are mapped to the directory structure I use on AWS. In particular:
- `/mnt/nvme:/tmp:rw` is the path to where model data will be processed and stored;
- `/mnt/efs/scripts:/mnt:ro` is the path to where the python scripts are stored. 

Several environmental variables are also used to control the API, including:
- `wgrib2_path`: Where WGRIB2 is saved. Do not change;
- `forecast_process_path`: Where forecast data should be saved during processing;
- `hist_process_path`: Where short term (~36h) historic data should be saved during processing;
- `merge_process_dir`: Where merged datasets should be saved;
- `tmp_dir`: Where downloaded data should be saved;
- `save_type`: If data should be saved to S3 (`S3`) or locally (`Download`).
- `save_path`: Where the ingested data should be saved.

The initial `docker compose up` command will take some time (~1 hour), as the initial batch of models is downloaded, processed and saved. From there, [Ofelia](https://github.com/mcuadros/ofelia) is used to schedule additional data ingest runs, while the `api_server` container responds to requests (Ex. `curl 127.0.0.1:8083/forecast/abc123/45,-75`)!

Two docker containers are used for this service, and are both saved in the "[Docker](https://github.com/Pirate-Weather/pirate-weather-code/tree/main/Docker)" folder. `public.ecr.aws/j9v4j3c7/pirate-wgrib-python-arm` is the ingest container, and builds WGRIB2 from source in addition to adding several key python packages for processing. `public.ecr.aws/j9v4j3c7/pirate-alpine-zarr` is a smaller container for the API response, and only consists of FastAPI plus some additional dependencies. 

### Requirements
Running a weather API for the entire planet isn't a trivial task, and accordingly, this service requires a pretty hefty set of resources, although I hope to reduce this in the future! At least 32 GB of free (not total) memory are required for the ingest processing, as well as at least 200 GB of working disk space. The docker images are also ARM only at the moment, but there's no technical reason why they shouldn't generate on x64 machines as well. 


## Next steps
The immediate next steps for this project are getting everything setup to make it easy for people host this locally and contribute to the project! In priority order:


1. Significantly improving the documentation on what each script is doing, how it is doing it, and how to add additional sources. 
2. Code quality needs to be improved throughout, particularly in the [response script](https://github.com/Pirate-Weather/pirate-weather-code/blob/main/responseLocal.py). 
3. Tests need to be added, with the goal of checking:
	1. If this specific model run is ingested, what does the result look like?
	2. Given this set of processed model results, what does the response look like? 
4. Some additional work needs to be done on the docker setup, focusing on building the containers for x64 machines and activating Ofelia's logging tools! 
5. Get the historic data side of things sorted out- most of the data is already on AWS, so the basic building blocks are there.


## License
The code for the Pirate Weather API is licensed under the [AGPL](https://github.com/Pirate-Weather/pirate-weather-code#AGPL-3.0-1-ov-file) license. That means that you're free to use (for any purpose), remix, or build off anything in this repository, under the condition that the end product (including serving data in a network application) has the source code available and is licensed under the same (AGPL) terms. This setup seems to be pretty common with open-source weather APIs, and so is where I wanted to start. 

If you'd like to use Pirate Weather in another application that isn't AGPL compatible, reach out! Depending on the application, I'm more than happy to come up with a reasonable and custom solution: <mail@pirateweather.net>.