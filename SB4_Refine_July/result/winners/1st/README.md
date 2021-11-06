# Setup and Code running Guide

## Root folder structure and file descriptions
1. config.json: configuration file for robustness
2. main_config.json: configuration file for final results
3. robustness.py: script for running robustness check
4. final.py: script for creating final predictions from Oct-2019 to Mar-2020
5. final.csv: predictions from the model for Oct-2019 to Mar-2020
6. robustness: directory containing the results of robustness script.
7. data.xlsx: dataset provided by the topcoder. Renamed to make easier for command line args.


## Requirements
1. python: 3.7
2. Numpy
3. Pandas
4. statsmodels


## Guidelines for running the final and robustness scripts
1. final: 
	It accepts two args "configfile" and "filepath".
	configfile: path to the configuration file for final results. ./example main_config.json
	filepath: path to the dataset provided the topcoder. example ./data.xlsx
	run example: `python final.py --configfile=main_config.json --filepath=data.xlsx`
	output: results will be stored in the same directory with name `final.csv`.

1. robustness: 
	It accepts two args "configfile" and "filepath".
	configfile: path to the configuration file for final results. ./example main_config.json
	filepath: path to the dataset provided the topcoder. example ./data.xlsx
	run example: `python robustness.py --configfile=config.json --filepath=data.xlsx`
	output: results will be stored in the same directory with name `robustness/submission_{i}.csv` where `i` is the iteration number.




