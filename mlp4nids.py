# -*- coding: utf-8 -*-
#

"""
Filename: mlp4nids.py
Date: Wed Sept 4 14:27:45 2019
Name: Arnaud Rosay
Description:
    -  Network intrusion detection using multi-layer perceptron (MLP) on
     UNB CIC IDS2017 dataset
"""

import deep_learning as dl
import numpy as np
import math
import pandas as pd
from datetime import datetime


desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)


def load_dataset(filename, file_format, n_feat):
    """
    Load extract of CIC2017 dataset from a pre-processed CSV file
    Included processing: normalization, randomization, split

    Parameter
    ---------
    filename: string
        name of the CSV file to be loaded
    file_format: string
            supported format: 'csv' and 'parquet'


    Return
    ------
    dataset: Dataset
        Returns a Dataset object
    """
    # Use size_percent to reduce dataset
    size_percent = 100
    # load dataset in a pandas DataFrame
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - Loading dataset from {}".format(t, filename))
    ds = dl.Dataset()
    ds.from_file(filename, file_format, size_percent)
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - dataset shape: {}".format(t, ds.data.shape))
    drop_list = []
    # Select labels from DataFrame
    ds.select_dfcol_as_label(' Label', False)
    if n_feat == 73:
        drop_list = [' Bwd PSH Flags', ' Bwd URG Flags',
                     'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk',
                     ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk',
                     ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
                     'Flow ID', ' Timestamp', ' Fwd Header Length.1',
                     ' Label']
        # convert IP address in numeric value
        change_ip = lambda x: sum([256**j*int(i) for j, i in enumerate(
            x.split('.')[::-1])])
        ds.data[' Source IP'] = ds.data[' Source IP'].apply(change_ip)
        ds.data[' Destination IP'] = ds.data[' Destination IP'].\
            apply(change_ip)
    elif n_feat == 70:
        drop_list = [' Bwd PSH Flags', ' Bwd URG Flags',
                     'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk',
                     ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk',
                     ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
                     'Flow ID', ' Timestamp', ' Fwd Header Length.1',
                     ' Source IP', ' Source Port', ' Destination IP',
                     ' Label']
    else:
        print("[ERROR] invalid number of features")
    ds.drop_dfcol(drop_list)
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - dataset shape after feature selection: {}".
          format(t, ds.data.shape))
    return ds


def load_files(n_feat):
    """
    Load training, cross-val and test sets

    Parameter
    ---------
    n_feat: int
        number of features to keep in datasets

    Returns
    -------
    ds_train: Dataset
        Dataset for training
    ds_cv: Dataset
        Dataset for cross-validation
    ds_test: Dataset
        Dataset for test
    """
    # set the path to generated datasets
    path = './cicids2017/datasets/ids2017-run-20190910154834/'
    filename_train = path + 'train_set.parquet'
    filename_cv = path + 'crossval_set.parquet'
    filename_test = path + 'test_set.parquet'
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - Loading data sets ...".format(t))
    # Load dataset in a pandas DataFrame
    ds_train = load_dataset(filename_train, 'parquet', n_feat)
    ds_cv = load_dataset(filename_cv, 'parquet', n_feat)
    ds_test = load_dataset(filename_test, 'parquet', n_feat)
    return ds_train, ds_cv, ds_test


def normalize(ds_train, ds_cv, ds_test):
    """
    Normalization of datasets

    Parameters
    ----------
    ds_train: Dataset
        Training set
    ds_cv: Dataset
        Cross-validation set
    ds_test: Dataset
        Test set

    Returns
    -------
    norm_train: Dataset
        Normalized training set
    norm_cv: Dataset
        Normalized cross-validation set
    norm_test: Dataset
        Normalized test set
    """
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - Normalizing training set ... ".format(t))
    normalizer = dl.Normalization()
    ds_train.data = normalizer.fit_and_transform(ds_train.data,
                                                 method='z_score_std',
                                                 per_col_scaler=True)
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - Normalizing crossval set ... ".format(t))
    ds_cv.data = normalizer.transform(ds_cv.data)
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - Normalizing test set ... ".format(t))
    ds_test.data = normalizer.transform(ds_test.data)
    return ds_train, ds_cv, ds_test


def print_metrics(ds, cost, cm):
    """
    Calculate and print some metrics on predictions

    Parameter
    ---------
    ds: Dataset
        Dataset
    cost: float
        overall cost
    cm: ndarray
        confusion matrix (row=true labels ; col=predicted labels)

    Return
    ------
    tp: int
        number of true positive
    fp: int
        number of false positive
    fn: int
        number of false negative
    tn: int
        number of true negative
    tnr: float
        true negative rate
    fpr: float
        false positive rate
    recall: float
        recall rate
    precision: float
        precision rate
    accuracy: float
        accuracy rate
    f1_score: float
        Harmonic mean of precision and recall
    MCC: float
        Matthews correlation coefficient
    """
    # create a simplified confusion matrix for multi-class classification
    simpl_cm = np.zeros(45)
    simpl_cm = simpl_cm.reshape((15, 3))
    simpl_cm[0][0] = cm[0][0]            # benign classified as benign
    simpl_cm[0][1] = 0                   # not applicable
    simpl_cm[0][2] = np.sum(cm[:1, 1:])  # benign classified as attack
    for i in range(1, cm.shape[0]):
        # Attack i classified as benign
        simpl_cm[i][0] = cm[i][0]
        # attack i correctly classified
        simpl_cm[i][1] = cm[i][i]
        # attack i classified as another attack
        simpl_cm[i][2] = np.sum(cm[i][1:i]) + np.sum(cm[i][i + 1:])
    simpl_cm = np.int_(simpl_cm)

    # create a confusion matrix for a binary classification (Normal vs Attack)
    binary_cm = np.zeros(4)
    binary_cm = binary_cm.reshape((2, 2))
    # True positive: attacks correctly detected
    # tp = np.sum(np.diagonal(cm[1:, 1:]))
    tp = np.sum(simpl_cm[1:, 1:])
    # True negative: normal traffic correctly detected
    tn = cm[0][0]
    # False positive: normal traffic detected as attacks
    fp = np.sum(simpl_cm[0, 1:])
    # False negative: attacks detected as normal traffic
    fn = np.sum(simpl_cm[1:, 0])
    binary_cm[0][0] = tp
    binary_cm[0][1] = fn
    binary_cm[1][0] = fp
    binary_cm[1][1] = tn
    binary_cm = np.int_(binary_cm)

    # calculate metrics
    accuracy = (tp+tn) / (tp+tn+fp+fn)
    precision = tp / (tp+fp)
    # recall = sensitivity
    recall = tp / (tp+fn)
    # f1-score is the harmonic mean of precision and recall.
    f1_score = (2*precision*recall) / (precision + recall)
    # True negative rate = specificity
    tnr = tn / (tn+fp)
    # False positive rate = fp / (tn+fp)
    fpr = 1 - tnr
    # Matthews correlation coefficient
    # MCC = (tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = 0
    mcc_exists = False
    if tp+fp != 0 and tp+fn != 0 and tn+fp != 0 and tn+fn != 0:
        mcc_exists = True
        mcc = np.float64(tp)*np.float64(tn)-np.float64(fp)*np.float64(fn)
        mcc = mcc / np.float64(np.sqrt(tp+fp))
        mcc = mcc / np.float64(math.sqrt(tp+fn))
        mcc = mcc / np.float64(math.sqrt(tn+fp))
        mcc = mcc / np.float64(math.sqrt(tn+fn))

    print(" ********** {}  **********".format(ds.data_file))
    print("Simplified confusion matrix")
    print("row=true labels (alphabet. order)")
    print("col=Benign | Correct Attack | Other Attack:")
    print(simpl_cm)
    print("Binary confusion matrix (row=true labels ; col=predicted labels)")
    print("  Attack  Benign")
    print(binary_cm)
    print("overall cost: {}".format(cost))
    print("true positive: {}".format(tp))
    print("false positive: {}".format(fp))
    print("false negative: {}".format(fn))
    print("true negative: {}".format(tn))
    print("true positive rate: {0:.6f}".format(tnr))
    print("false positive rate: {0:.6f}".format(fpr))
    print("recall: {0:.6f}".format(recall))
    print("precision: {0:.6f}".format(precision))
    print("accuracy: {0:.6f}".format(accuracy))
    print("f1_score: {0:.6f}".format(f1_score))
    if mcc_exists is True:
        print("MCC: {0:.6f}".format(mcc))

    return (tp, fp, fn, tn, tnr, fpr, recall, precision, accuracy, f1_score,
            mcc)


def create_nn(n_inputs, n_outputs, name, reuse_model):
    """
    Declare and initialize neural network. Initialize default values for
    hyper-parameters.

    Parameters
    ----------
    n_inputs: int
        number of inputs (features) for the neural network
    n_outputs: int
        number of outputs (classes) for the neural network
    name: str
        string containing neural network name
    reuse_model: bool
        True: initialization uses weights from a previous training
        False: weights will be randomly initialized by the deep_learning
        library

    Returns
    -------
    nn: NeuralNetwork
        Neural network that has been created and initialized
    """
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - Creating neural network ...".format(t))
    nn_desc = dl.NeuralNetworkDescriptor(name)
    l1_n_outputs = 256
    layer1 = dl.Layer(name='layer1',
                      n_inputs=n_inputs,
                      n_outputs=l1_n_outputs,
                      activation_name='selu')
    nn_desc.topology.add_layer(layer1)
    l1_drop = dl.Layer(name='l1_drop',
                       n_inputs=l1_n_outputs,
                       n_outputs=l1_n_outputs,
                       activation_name='dropout')
    nn_desc.topology.add_layer(l1_drop)
    l2_n_outputs = l1_n_outputs
    layer2 = dl.Layer(name='layer2',
                      n_inputs=l1_n_outputs,
                      n_outputs=l2_n_outputs,
                      activation_name='selu')
    nn_desc.topology.add_layer(layer2)
    l2_drop = dl.Layer(name='l2_drop',
                       n_inputs=l2_n_outputs,
                       n_outputs=l2_n_outputs,
                       activation_name='dropout')
    nn_desc.topology.add_layer(l2_drop)
    layer3 = dl.Layer(name='layer3',
                      n_inputs=l2_n_outputs,
                      n_outputs=n_outputs,
                      activation_name='softmax')
    nn_desc.topology.add_layer(layer3)
    nn_desc.hyper_param.optimizer = 'Adam'
    # params: learning_rate, Beta1, Beta2
    nn_desc.hyper_param.optim_params = [0.001, 0.9, 0.999]
    nn_desc.hyper_param.epochs = 10
    nn_desc.hyper_param.l2_reg = 0.0
    nn_desc.hyper_param.keep_prob = 0.5
    nn_desc.hyper_param.batch_size = 32
    nn_desc.hyper_param.batch_norm = False
    nn_desc.hyper_param.cost_print_period = 1
    nn_desc.hyper_param.eval_print_period = 1
    nn = dl.NeuralNetwork(nn_desc)
    nn.init(reuse_model)
    return nn


def set_nn_hyperparams(nn, hpp_name, hpp_value):
    """
    Set hyper-parameters
    Parameters
    ----------
    nn: NeuralNetwork
        neural network to configure
    hpp_name: str
        String containing the name hyper-parameter to configure
    hpp_value:
        Value (float or list) to set in selected hyper-parameter

    Returns
    -------
    None
    """
    if hpp_name == 'epochs':
        nn.nn_desc_list[0].hyper_param.epochs = hpp_value
    elif hpp_name == 'optim_params':
        nn.nn_desc_list[0].hyper_param.optim_params = hpp_value
    elif hpp_name == 'keep_prob':
        nn.nn_desc_list[0].hyper_param.keep_prob = hpp_value


def train_nn(nn, ds_train, ds_cv):
    """
    Create and train a simple MLP

    Parameters
    ----------
    nn: NeuralNetwork
        Neural network to train
    ds_train: pandas DataFrame
        Training set
    ds_cv: pandas DataFrame
        Cross-validation set

    Returns
    -------
    None
    """
    # training phase
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - Training neural network ...".format(t))
    (tr_loss, tr_accuracy, tr_cm,
     tr_fpr_vect, tr_tpr_vect, tr_roc_auc) = nn.fit(ds_train, ds_cv)

    # training results
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - Results ...".format(t))
    print("Training cost: {}".format(tr_loss))
    print("Training accuracy: {}".format(tr_accuracy))
    print("Training confusion matrix:")
    print(tr_cm)
    # get metrics for Crossval set
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - Calculating crossval metrics ...".format(t))
    (cv_loss_list, cv_acc_list, cv_cm_list, cv_pred_list, cv_cm,
     cv_fpr_vect, cv_tpr_vect, cv_roc_auc) = nn.get_metrics(ds_cv)
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - Results ...".format(t))
    print("Test cost: {}".format(cv_loss_list))
    print("Test accuracy: {}".format(cv_acc_list))
    print("Test confusion matrix:")
    print(cv_cm)


def get_performances(nn, ds_train, ds_cv, ds_test):
    """
    Create and train a simple MLP

    Parameters
    ----------
    nn: NeuralNetwork
        Neural network to train
    ds_train: pandas DataFrame
        Training set
    ds_cv: pandas DataFrame
        Cross-validation set
    ds_test: pandas DataFrame
        Test set
    Returns
    -------
    None
    """
    # get metrics for training set
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - Calculating training metrics ...".format(t))
    (train_loss_list, train_acc_list, train_cm_list, train_pred_list, train_cm,
     train_fpr_vect, train_tpr_vect, train_roc_auc) = nn.get_metrics(ds_train)
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - training results ...".format(t))
    print("\tTraining cost: {}".format(train_loss_list))
    print("\tTraining accuracy: {}".format(train_acc_list))
    print("\tTraining confusion matrix:")
    print(train_cm)
    # get metrics for crossval set
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - Calculating crossval metrics ...".format(t))
    (cv_loss_list, cv_acc_list, cv_cm_list, cv_pred_list, cv_cm,
     cv_fpr_vect, cv_tpr_vect, cv_roc_auc) = nn.get_metrics(ds_cv)
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - crossval results ...".format(t))
    print("Crossval cost: {}".format(cv_loss_list))
    print("Crossval accuracy: {}".format(cv_acc_list))
    print("Crossval confusion matrix:")
    print(cv_cm)
    # get metrics for test set
    t1 = datetime.now()
    t = t1.strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - Calculating test metrics ...".format(t))
    (test_loss_list, test_acc_list, test_cm_list, test_pred_list, test_cm,
     test_fpr_vect, test_tpr_vect, test_roc_auc) = nn.get_metrics(ds_test)
    t2 = datetime.now()
    t = t2.strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - test results ...".format(t))
    print("Test cost: {}".format(test_loss_list))
    print("Test accuracy: {}".format(test_acc_list))
    print("Test confusion matrix:")
    print(test_cm)
    print_metrics(ds_train, train_loss_list, train_cm)
    print_metrics(ds_cv, cv_loss_list, cv_cm)
    print_metrics(ds_test, test_loss_list, test_cm)


def main():
    """
    Main program

    Returns
    -------
    None

    """
    # set seed so that we can reproduce exact same settings
    np.random.seed(0)

    # initialize some variables
    n_features = 0
    c = ''
    reuse_model = True

    loop_exit = False
    while not loop_exit:
        print("Menu:")
        print("\t1: start NN training")
        print("\t2: get NN performances")
        c = input("Enter you choice: ")
        if c == '1':
            reuse_model = False
            loop_exit = True
        if c == '2':
            loop_exit = True
    loop_exit = False
    while not loop_exit:
        str_n_feat = input("Enter number of features to use (70 or 73): ")
        n_features = int(str_n_feat)
        if str_n_feat == '73' or str_n_feat == '70':
            loop_exit = True
    optimized_hpp = True
    k = input("Use optimized hyper-parameters [Y/n]: ")
    if k == 'n' or k == 'N':
        optimized_hpp = False

    # Load dataset in a pandas DataFrame
    ds_train, ds_cv, ds_test = load_files(n_features)
    n_classes = ds_train.n_classes
    # Normalize data
    ds_train, ds_cv, ds_test = normalize(ds_train, ds_cv, ds_test)

    # define number of epochs depending on number of features
    if n_features == 73:
        if optimized_hpp is True:
            n_epochs = 45
        else:
            n_epochs = 56
    else:
        if optimized_hpp is True:
            n_epochs = 715
        else:
            n_epochs = 749
    # define MLP name
    if optimized_hpp is True:
        name = 'MLP_' + str(n_features) + 'feat_' + str(n_epochs) + 'epochs'
    else:
        name = 'MLP_NoHPP_' + str(n_features) + 'feat_' \
               + str(n_epochs) + 'epochs'

    # create enural network with default values of hyper-parameters
    mlp = create_nn(n_features, n_classes, name, reuse_model)

    # set hyper-parameters
    if n_features == 73:
        if optimized_hpp is True:
            keep_prob = 0.8352118790504427
            optim_params = [0.0006368343670907567, 0.9155443486721901,
                            0.999]
        else:
            keep_prob = 0.5
            optim_params = [0.001, 0.9, 0.999]
        set_nn_hyperparams(mlp, 'epochs', n_epochs)
        set_nn_hyperparams(mlp, 'keep_prob', keep_prob)
        set_nn_hyperparams(mlp, 'optim_params', optim_params)
    else:
        if optimized_hpp is True:
            keep_prob = 0.833910013348719
            optim_params = [0.0006826106769894995, 0.9214915874130065,
                            0.999]
        else:
            keep_prob = 0.5
            optim_params = [0.001, 0.9, 0.999]
        set_nn_hyperparams(mlp, 'epochs', n_epochs)
        set_nn_hyperparams(mlp, 'keep_prob', keep_prob)
        set_nn_hyperparams(mlp, 'optim_params', optim_params)
    print("\tn_epochs: {}".format(n_epochs))
    print("\tkeep_prob: {}".format(keep_prob))
    print("\toptim_params: {}".format(optim_params))

    if c == '1':
        # train MLP
        train_nn(mlp, ds_train, ds_cv)
    else:
        # Get metrics from a trained model
        get_performances(mlp, ds_train, ds_cv, ds_test)

    # kill MLP
    mlp.close()


if __name__ == "__main__":
    main()
