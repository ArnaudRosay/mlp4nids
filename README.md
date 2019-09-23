# mlp4nids
Multi-layer perceptron for network intrusion detection 

COMING SOON

## CICIDS2017 Dataset
Dataset can be retrieved from University of New Brunswick - Canadian Institure for Cybersecurity: [click here](https://www.unb.ca/cic/datasets/ids-2017.html)

CSV files shall be located in ./cicids2017/csv_files/ (unless you change path in source files)

## Files
- extract_traffic_types.py: read csv files and create one parquet files for each traffic type. In total 15 files are generated (1 benign and 14 attacks)
- create_datasets.py: read the 15 files generated in previous step and create parquet files for training, cross-validation and test
- deep_learning.py: deep learning library
- mlp4nids.py: main program to train neural network and get performances

## Environment
This project has been developed using python 3.7 and TensorFlow-1.14.

