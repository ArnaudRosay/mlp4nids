# mlp4nids
Multi-layer perceptron for network intrusion detection 

##### Coming Soon - All files will be available in October

## Objectives
This project demonstrates neural network performances for intrusion detection on a recent dataset.

## CICIDS2017 Dataset
Dataset can be retrieved from University of New Brunswick - Canadian Institure for Cybersecurity: 
[click here](https://www.unb.ca/cic/datasets/ids-2017.html) <br>
CSV files shall be located in ./cicids2017/csv_files/ (unless you change path in source files). <br>
For convenience, the 3 files required for training, cross-validation and test are provided in 
./cicids2017/datasets/ids2017-run-20190910154834/ but can be regenerated from the csv files using python programs.

## Files
- **extract_traffic_types.py**: read csv files and create one parquet files for each traffic type. In total 15 files 
are generated (1 benign and 14 attacks).
- **create_datasets.py**: read the 15 files generated in previous step and create parquet files for training, 
cross-validation and test in a new folder
- **deep_learning.py**: deep learning library.
- **mlp4nids.py**: main program to train neural network and get performances. Don't forget to update path to the 
generated parquet files.
- **requirements.txt**: list of python packages to reproduce the environment with all dependencies.

## Environment
This project has been developed using python 3.7 and TensorFlow-1.14.<br>
All packages required to run this project are listed in requirements.txt and shall be installed in your python
environment.

