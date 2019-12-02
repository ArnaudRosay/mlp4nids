# -*- coding: utf-8 -*-
#

"""
Filename: deeplearning.py
Date: Wed Oct 24 13:04:16 2018
Name: Arnaud Rosay
Description:
    -  Deep learning library
"""

import pandas as pd
import numpy as np
import tensorflow as tf
# from tensorflow.python.client import timeline
import math
import os
from datetime import datetime
import time
from sklearn.metrics import roc_curve, auc
from copy import deepcopy

# Enable Tensorboard (or not)
TENSORBOARD_ENABLED = True

# MLP training is faster without CUDA / GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# reduce debug info from Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# set seed so that we can reproduce exact same settings
tf.compat.v1.set_random_seed(0)


class Dataset(object):
    """
    Dataset class
    """

    def __init__(self):
        """
        Dataset creation

        Parameter
        ---------
        None

        Return
        ------
        Created object
        """
        self.data_file = ''
        self.data = pd.DataFrame()
        self.labels = pd.DataFrame()
        self.labels_onehot = pd.DataFrame()
        self.df = pd.DataFrame()
        self.n_samples = 0
        self.n_features = 0
        self.label_dict = {}
        self.inv_label_dict = {}
        self.n_classes = 0
        self.batch_size = 0
        self.n_batch = 0
        self.current_batch_idx = 0
        self.true_distribution = []

    def from_file(self, data_file, file_format, size_percent):
        """
        Load data file and if needed label file

        Parameters
        ----------
        data_file: string
            filename (incl. path) for data
        file_format: string
            supported format: 'csv' and 'parquet'
        size_percent: int
            percentage of the file to be loaded (from 1 to 100

        Returns
        -------
        None
        """
        self.data_file = data_file
        df = pd.DataFrame()
        if file_format == 'csv':
            df = self._csv2df(data_file)
        elif file_format == 'parquet':
            df = self._pq2df(data_file)
        else:
            print("[ERROR] Format {} not supported".format(format))
        max_idx = int(df.shape[0] * size_percent / 100)
        self.df = df[:max_idx][:]
        self.data = self.df
        self.df.name = data_file
        self.n_samples = np.shape(self.df)[0]
        self.n_features = np.shape(self.df)[1]

    @staticmethod
    def _csv2df(data_file):
        """
        Load CSV file in pandas DataFrame

        Parameter
        ---------
        self: Dataset

        Return
        ------
        df: DataFrame
            Loaded data Frame
        """
        df = pd.read_csv(data_file, encoding="ISO-8859-1", low_memory=False)
        return df

    @staticmethod
    def _pq2df(data_file):
        """
        Load parquet file in pandas DataFrame

        Parameter
        ---------
        self: Dataset

        Return
        ------
        df: DataFrame
            Loaded data Frame
        """
        df = pd.read_parquet(data_file)
        return df

    def drop_dfcol(self, drop_list):
        """
        Remove the columns from dataset DataFrame that are specified in
        drop_list and convert to numpy array loaded in data

        Parameters:
        -----------
        drop_list: list
            list of column names

        Return:
        -------
        None
        """
        self.data = self.df
        for lbl in drop_list:
            self.data = self.data.drop(lbl, axis=1)
        self.n_features = np.shape(self.data)[1]

    def select_dfcol_as_label(self, col_name, bin_class):
        """
        Select the column from dataset DataFrame and specified in col_name as
        label

        Parameters
        ----------
        col_name: string
            name of the column to select
        bin_class: bool
            convert multiclass label to enable binary classification

        Returns
        -------
        None

        """

        self.labels = self.df[col_name]
        if bin_class is True:
            self.label_dict = {'BENIGN': 0, 'Bot': 1, 'DDoS': 1,
                               'DoS GoldenEye': 1, 'DoS Hulk': 1,
                               'DoS Slowhttptest': 1, 'DoS slowloris': 1,
                               'FTP-Patator': 1, 'Heartbleed': 1,
                               'Infiltration': 1, 'PortScan': 1,
                               'SSH-Patator': 1,
                               'Web Attack \x96 Brute Force': 1,
                               'Web Attack \x96 Sql Injection': 1,
                               'Web Attack \x96 XSS': 1}

        else:
            self.label_dict = {label: idx for idx, label in enumerate(
                np.unique(self.df[col_name]))}
        self.inv_label_dict = {v: k for k, v in self.label_dict.items()}
        self.labels.replace(to_replace=self.label_dict, value=None,
                            inplace=True)
        if bin_class is True:
            self.label_dict = {'BENIGN': 0, 'Attack': 1}
        self.n_classes = np.size(self.labels.value_counts().index)
        self.labels_onehot = self.onehot_encode(self.labels, self.n_classes)
        # key_list = []
        # for key in dict.keys(self.label_dict):
        #     key_list.append(key)
        # self.df.reset_index(key_list)

        self.true_distribution = self._get_true_distribution()

    def onehot_encode(self, df, n_classes):
        """
        one-hot encoding

        Parameters:
        -----------
        df: DataFrame
            pandas DataFrame containing list of labels (numeric values) to be
            encoded

        Return:
        -------
        onehot_df: DataFrame
            m by n_classes matrix (m being the size of labels)
        """
        np_array = df.values
        m = np.size(np_array)
        n = n_classes
        onehot_matrix = np.zeros((m, n))
        for i in range(0, m):
            onehot_matrix[i][np_array[i]] = 1
        onehot_df = pd.DataFrame(np.int_(onehot_matrix),
                                 columns=self.label_dict)
        return onehot_df

    def set_batch_size(self, batch_size):
        """
        Function setting the batch size

        Parameters:
        -----------
        batch_size: integer
            value corresponding to the number of training examples contained
            in a batch

        Return:
        -------
        None

        """
        self.batch_size = batch_size
        self.n_batch = math.ceil(self.n_samples / batch_size)

    def get_next_batch(self, onehot=True):
        """
        Function returning the next batch
        When dataset has been fully fetched, a permutation is done
        By default, the labels will be one-hot encoded

        Parameters:
        -----------
        None

        Return:
        -------
        data_batch: array
            contains a subset of data corresponding to the next batch
        labels_batch: array
            contains a subset of labels corresponding to the next batch

        """
        if self.current_batch_idx == 0:
            self.permutation()
        next_beg = self.current_batch_idx * self.batch_size
        next_end = (self.current_batch_idx + 1) * self.batch_size
        if next_end > self.n_samples:
            next_end = self.n_samples
            self.current_batch_idx = 0
        data_batch = self.data.values[next_beg:next_end][:]
        if onehot is True:
            labels_batch = self.labels_onehot.values[next_beg:next_end][:]
        else:
            labels_batch = self.labels.values[next_beg:next_end][:]
        self.current_batch_idx += 1
        return data_batch, labels_batch

    def permutation(self):
        """
        Run permutations in the dataset to ensure that the different extracted
        sets will contain all type of labels

        Parameters
        ---------

        Return
        ------
        None

        """
        perm = np.random.permutation(self.n_samples)
        self.data = self.data.iloc[perm]
        self.labels = self.labels.iloc[perm]
        self.labels_onehot = self.labels_onehot.iloc[perm]

    def random_sampling(self, n_subset):
        """
        Split dataset in n_subset parts and create as many dataset objects
        containing samples of original dataset chosen randomly with replacement

        Parameters
        ----------
        n_subset: unsigned int
            number of sub-sampling datasets to generate

        Returns
        -------
        subset_list: list
            list of n_subset subsampling dataset objects
        """
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("[INFO] {} - Random sampling with replacement ...".format(t))
        subset_list = []
        training_set = self
        subset_size = math.ceil(training_set.n_samples / n_subset)
        # create subsets
        for i in range(n_subset):
            # run a permutation to mix all samples (sampling with replacement)
            self.permutation()
            #  always draw the first samples
            start_idx = 0
            stop_idx = subset_size
            subset = deepcopy(training_set)
            subset.data = subset.data[start_idx:stop_idx][:]
            subset.labels = subset.labels[start_idx:stop_idx][:]
            subset.labels_onehot = subset.labels_onehot[start_idx:stop_idx][:]
            subset.n_samples = stop_idx - start_idx
            subset.true_distribution = subset._get_true_distribution()
            subset.set_batch_size(training_set.batch_size)
            subset_list.append(subset)
            print("\tSubset shape {}".format(subset.data.shape))
        return subset_list

    def _get_true_distribution(self):
        true_distribution = np.array(self.labels.value_counts().sort_index())
        true_distribution = np.int_(true_distribution)
        return true_distribution


class Normalization(object):
    """ Normalization class """

    def __init__(self):
        """
        Normalization object  creator

        Parameters
        ----------

        Return
        ------
        Created object

        """
        self.dict_scalers = {}

    class Scaler(object):
        """ Scaler Class """

        def __init__(self,
                     min_val=None, max_val=None,
                     mean_val=None, std_val=None,
                     q1=None, q3=None,
                     method='min_max_scaling'):
            """
            Scaler object creation

            Parameters
            ----------
            min_val: float
                minimum value of the feature
            max_val: float
                maximum value of the feature
            mean_val: float
                mean value of the feature
            std_val: float
                standard deviation value of the feature
            q1: float
                First quartile value of the feature
            q3: float
                Third quartile value of the feature
            method: str
                supported methods are: 'min_max_scaling', 'z_score_std' or
                'robust_scaling'
            """
            self.min = min_val
            self.max = max_val
            self.mean = mean_val
            self.std = std_val
            self.q1 = q1
            self.q3 = q3
            self.method = method

    def fit(self, df, method='min_max_scaling', per_col_scaler=False):
        """
        Analyze data and create a dictionary of scalers

        Parameters
        ----------
        df: DataFrame
            pandas DataFrame containing data to normalize
        method: str
            supported methods are: 'min_max_scaling', 'z_score_std' or '
            robust_scaling'
        per_col_scaler: Boolean
            When True a different scaler is used for each column of the
            DataFrame
            When False, the same scaler is applied to all columns

        Returns
        -------
        None

        """
        # Does df contain multiple columns ?
        if df.size == len(df) or per_col_scaler is True:
            # df contains multiple columns
            lbl_list = df.columns.values
            for lbl in lbl_list:
                try:
                    min_val = float(np.amin(df[lbl]))
                    max_val = float(np.amax(df[lbl]))
                    mean_val = float(np.mean(df[lbl]))
                    std_val = float(np.std(df[lbl]))
                    # TODO Validate/Debug Robust Scaler
                    q1_val = float(np.percentile(df[lbl], 25))
                    q3_val = float(np.percentile(df[lbl], 75))
                except TypeError:
                    raise Exception("[ERROR] TypeError in normalization fit")
                scaler = self.Scaler(min_val=min_val, max_val=max_val,
                                     mean_val=mean_val, std_val=std_val,
                                     q1=q1_val, q3=q3_val,
                                     method=method)
                self.dict_scalers[lbl] = scaler
        else:
            # df contains one single column
            try:
                min_val = float(np.amin(df))
                max_val = float(np.amax(df))
                mean_val = float(np.mean(df))
                std_val = float(np.std(df))
                # TODO Validate/Debug Robust Scaler
                q1_val = float(np.percentile(df, 25))
                q3_val = float(np.percentile(df, 75))
            except TypeError:
                raise Exception("[ERROR] TypeError in normalization fit")
            scaler = self.Scaler(min_val=min_val, max_val=max_val,
                                 mean_val=mean_val, std_val=std_val,
                                 q1=q1_val, q3=q3_val,
                                 method=method)
            self.dict_scalers['OneForAll'] = scaler

    def transform(self, df):
        """
        Calculate normalized DataFrame

        Parameters
        ----------
        df: DataFrame
            pandas DataFrame containing data to normalize        df

        Returns
        -------
        normalized_df: DataFrame
            pandas DataFrame containing normalized values

        """
        # One scaler is required before calling transform
        if len(self.dict_scalers) == 0:
            raise Exception("[ERROR] normalization transform method called"
                            "prior to fit method")

        normalized_df = df.copy()
        if 'OneForAll' in self.dict_scalers:
            sclr = self.dict_scalers['OneForAll']
            if sclr.method == 'min_max_scaling':
                # TODO Saturation case of value >max or <min during fit
                total_range = sclr.max - sclr.min
                if total_range == 0:
                    total_range = 1
                    print("[Warning]: MinMax scaler with min=max")
                normalized_df = (df - sclr.min) / total_range
            elif sclr.method == 'z_score_std':
                normalized_df = (df - sclr.mean) / sclr.std
            elif sclr.method == 'robust_scaling':
                # TODO Validate/Debug Robust Scaler
                iqr = sclr.q3 - sclr.q1
                if iqr == 0:
                    iqr = 1
                    print("[Warning]: robust scaler with q1=q3")
                normalized_df = (df - sclr.q1) / iqr
            else:
                raise Exception("[ERROR] normalization method not "
                                "implemented yet")
        else:
            # Apply parameters to all columns
            lbl_list = df.columns.values
            for lbl in lbl_list:
                sclr = self.dict_scalers[lbl]
                if sclr.method == 'min_max_scaling':
                    # TODO Saturation case of value >max or <min during fit
                    total_range = sclr.max - sclr.min
                    if total_range == 0:
                        total_range = 1
                        print("[Warning]: scaler with min=max for feature: {}".
                              format(lbl))
                    normalized_df[lbl] = (df[lbl] - sclr.min) / total_range
                elif sclr.method == 'z_score_std':
                    normalized_df[lbl] = (df[lbl] - sclr.mean) / sclr.std
                elif sclr.method == 'robust_scaling':
                    # TODO Debug/Validate Robust Scaler
                    iqr = sclr.q3 - sclr.q1
                    if iqr == 0:
                        iqr = 1
                        print("[Warning]: scaler with q1=q3 for feature: {}".
                              format(lbl))
                    normalized_df[lbl] = (df - sclr.q1) / iqr
                else:
                    raise Exception("[ERROR] normalization method not "
                                    "implemented yet")
        return normalized_df

    def fit_and_transform(self, df, method='min_max_scaling',
                          per_col_scaler=False):
        """
        Analyze data, initialize parameters accordingly and then calculate
        normalized DataFrame

        Parameters
        ----------
        df: DataFrame
            pandas DataFrame containing data to normalize
        method: str
            supported methods are: 'min_max_scaling' or 'z_score_std'
        per_col_scaler: Boolean
            When True a different scaler is used for each column of the
            DataFrame
            When False, the same scaler is applied to all columns
        Returns
        -------
        normalized_df: DataFrame
            pandas DataFrame containing normalized values
        """
        self.fit(df, method, per_col_scaler)
        normalized_df = self.transform(df)
        return normalized_df


class Layer(object):
    """ Layer Class """

    def __init__(self,
                 name, n_inputs, n_outputs, activation_name,
                 dropout_keep_prob=1):
        """
        Layer creation

        Parameter
        ---------
        name: str
            layer name
        n_inputs: int
            number of inputs
        n_outputs: int
            number of outputs
        activation_name: string
            name of the activation function
            accepted string: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu',
                             'selu', 'softmax', 'dropout'
            Any other string will be replaced by tensorflow identity function
        dropout_keep_prob: float
            probability to keep a node during dropout
            accepted range: 0 < dropout_keep_prob <= 1
            default value = 1

        Return
        ------
        Created object

        """
        self.name = name
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dropout_keep_prob = dropout_keep_prob
        self.activation_name = activation_name
        self.tf_act_fn = self._activation_translation(activation_name)
        self.tf_biases = 0
        self.tf_weights = 0
        self.tf_keep_prob = 0
        self.tf_inputs = 0
        self.tf_outputs = 0
        self.tf_l2_loss = 0

    @staticmethod
    def _activation_translation(activation_name):
        """
        Translate activation name to tensorflow activation function.
        Supported activation functions: 'relu', 'tanh', 'sigmoid',
        'leaky_relu', 'elu', 'selu', 'softmax', 'dropout'

        Parameter
        ---------
        activation_name: string
            name of the activation function

        Return
        ------
        tf_activation: activation function
            tensorflow activation function
        """
        if activation_name == 'relu':
            tf_act = tf.nn.relu
        elif activation_name == 'leaky_relu':
            tf_act = tf.nn.leaky_relu
        elif activation_name == 'elu':
            tf_act = tf.nn.elu
        elif activation_name == 'selu':
            tf_act = tf.nn.selu
        elif activation_name == 'tanh':
            tf_act = tf.nn.tanh
        elif activation_name == 'sigmoid' or activation_name == 'swish':
            tf_act = tf.nn.sigmoid
        elif activation_name == 'softmax':
            tf_act = tf.nn.softmax
        elif activation_name == 'dropout':
            tf_act = tf.nn.dropout
        else:
            print("[INFO] activation function not implemented yet")
            tf_act = tf.identity
        return tf_act

    def tf_create(self, inputs, set_activation_function, keep_prob,
                  batch_norm, is_training):
        """
        Create a tensorflow layer

        Parameters
        ----------
        inputs: tensor
            tensor used as input of the layer
        set_activation_function: boolean
            True is activation function shall be set
            Note that for output layers it is more efficient not to set it
            (typical in case of sigmoid or softmax)
        keep_prob: tensor
            tensor used as a placeholder used in drop_out keep_prob
        batch_norm: bool
            Use batch normalization if True
        is_training: tensor
            tensor (placeholder) used to differentiate whether processing is
            executed in training phase or in inference phase

        Returns
        -------
        biases:  tensor
            bias values of the created layer
        weights: tensor
            weight values of the created layer
        outputs: tensor
            outputs of the created layer
        """
        with tf.name_scope(self.name):
            if self.activation_name != 'dropout':
                with tf.name_scope('weights'):
                    if self.activation_name == 'relu' or \
                            self.activation_name == 'elu' or \
                            self.activation_name == 'selu' or \
                            self.activation_name == 'leaky_relu':
                        std = tf.sqrt(2.0 / self.n_inputs)
                    elif self.activation_name == 'tanh':
                        std = tf.sqrt(1 / self.n_inputs)
                    else:
                        std = tf.sqrt(2 / (self.n_inputs + self.n_outputs))
                    shape = [self.n_inputs, self.n_outputs]
                    weights = tf.Variable(tf.random.truncated_normal(
                        shape=shape,
                        mean=0,
                        stddev=std),
                        name='weights')
                with tf.name_scope('biases'):
                    shape = [self.n_outputs]
                    biases = tf.Variable(tf.zeros(shape=shape),
                                         name='biases')
                with tf.name_scope('Wx_plus_b'):
                    op = tf.add(tf.matmul(inputs, weights, name='weights.x'),
                                biases,
                                name='plus_bias')
                    if batch_norm is True:
                        op = tf.layers.batch_normalization(
                            op, training=is_training, axis=1)
                    # tf.summary.histogram('pre_act', op)
                    if set_activation_function is True:
                        outputs = self.tf_act_fn(op, name=self.activation_name)
                        if self.activation_name == 'swish':
                            outputs = op * outputs
                    else:
                        outputs = op
                self.tf_biases = biases
                self.tf_weights = weights
                self.tf_l2_loss = tf.nn.l2_loss(self.tf_weights)
            else:
                # outputs = self.tf_act_fn(inputs, name=self.activation_name,
                #                          keep_prob=keep_prob)
                outputs = self.tf_act_fn(inputs, name=self.activation_name,
                                         rate=1 - keep_prob)
                self.tf_keep_prob = keep_prob
            self.tf_outputs = outputs


class Topology(object):
    """ Topology Class """

    def __init__(self):
        """
        Creator function
        """
        self.dict_topo = {}
        self.current = 0

    def add_layer(self, layer):
        """
        Add a layer in the topology description

        Parameters
        ----------
        layer: Layer
            Layer object containing parameters of the layer

        Returns
        -------
        None

        """
        idx = len(self.dict_topo)
        idx += 1
        self.dict_topo[idx] = layer

    def __iter__(self):
        """
        Iterator

        Returns
        -------
        self

        """
        return self

    def __next__(self):
        """
        Next function of Iterator

        Returns
        -------
        layer: Layer
            Next layer of topology

        """
        if self.current >= len(self.dict_topo):
            self.current = 0
            raise StopIteration
        self.current += 1
        return self.dict_topo[self.current]


class HyperParam(object):
    """ Hyperparameter Class """

    def __init__(self, optimizer='GradientDescent', *optim_params,
                 epochs=500, batch_size=32, l2_reg=0.0,
                 keep_prob=0.5, batch_norm=False, cost_print_period=0,
                 eval_print_period=0):
        self.optimizer = optimizer
        self.optim_params = optim_params
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.keep_prob = keep_prob
        self.batch_norm = batch_norm
        self.cost_print_period = cost_print_period
        self.eval_print_period = eval_print_period


class GraphDescriptor(object):
    """ TensorFlow graph elements for a neural network """

    def __init__(self):
        self.x = 0
        self.y = 0
        self.keep_prob = 0
        self.is_training = 0
        self.global_step = 0
        self.regularizer = 0
        self.logits = 0
        self.loss = 0
        self.accuracy = 0
        self.cm = 0
        self.prediction = 0
        self.auc_op = 0
        self.optimzr = 0
        self.saver = 0


class NeuralNetworkDescriptor(object):
    """
    Neural Network topology and hyper-parameter descriptions
    Contains also tensorflow descriptor that is automatically filled in during
    graph construction
    """

    def __init__(self, name):
        """
        Creator function
        """
        self.name = name
        self.topology = Topology()
        self.hyper_param = HyperParam()
        self.tf_desc = GraphDescriptor()


class NeuralNetworkEnsemble(object):
    """ Neural Network Class """

    def __init__(self, name, nn_descriptor_list, method='',
                 voting_method='soft'):
        """
        Creator function
        """
        self.name = name
        self.n_estimators = len(nn_descriptor_list)
        if self.n_estimators == 1 and method != '':
            print("[Warning] method should be an empty string for a single "
                  "neural network")
            self.method = ''
            self.voting_method = ''
        if self.n_estimators > 1 and method != 'bagging':
            raise Exception("[Error] Only bagging method is supported")
        else:
            self.method = method
            self.voting_method = voting_method
        self.nn_desc_list = nn_descriptor_list
        self.tf_session = tf.compat.v1.Session()
        # self.tf_epochs = tf.Variable(0, 'epochs')
        self.tf_voting_inputs = tf.identity
        self.tf_voting_outputs = tf.identity
        self.tf_merged = tf.identity
        self.tf_filewriter = tf.identity
        self.tf_train_writer = tf.identity
        self.tf_cv_writer = tf.identity

    def init(self, reuse_params=False):
        """

        Parameters
        ----------
        reuse_params: boolean
            Tensorflow model parameters are loaded from last saved checkpoint

        Returns
        -------
        None

        """
        self._build_graph()
        self._set_optimizer()
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        root_logdir = "tf_logs"
        logdir = "{}/run-{}/".format(root_logdir, now)
        self.tf_merged = tf.compat.v1.summary.merge_all()
        self.tf_train_writer = \
            tf.compat.v1.summary.FileWriter(logdir + '/train',
                                            graph=self.tf_session.graph)
        self.tf_cv_writer = tf.compat.v1.summary.FileWriter(logdir + '/test')
        init = tf.compat.v1.global_variables_initializer()
        self.tf_session.run(init)
        # load parameters if needed
        if reuse_params is True:
            print("[INFO] Loading TF parameters ...")
            self.tf_saver.restore(self.tf_session,
                                  './ckpt_dir/' + self.name + '.ckpt')

    def fit_singleNN(self, nn_desc, *ds):
        """
        Train neural network

        Parameters
        ----------
        nn_desc: NeuralNetworkDescriptor
            description of a single neural network
        ds: List[int]
            list of Dataset that can contain either one or two datasets
            first element is the training set
            second element is the development set

        Returns
        -------
        loss: float
            Cost value after training
        acc: float
            Accuracy value after training
        cm: ndarray
            confusion matrix
        fpr_vect: array
            vector of training false positive rate
        tpr_vect: array
            vector of training true positive rate
        roc_auc: float
            area under the curve for training

        """
        nn = nn_desc
        n_ds = len(ds)
        if n_ds == 2:
            ds_train = ds[0]
            ds_cv = ds[1]
            cv_eval = True
            train_loss = 0
            train_acc = 0
            steps = 0
        elif n_ds == 1:
            ds_train = ds[0]
            ds_cv = []
            cv_eval = False
            train_loss = 0
            train_acc = 0
            steps = 0
        else:
            raise Exception("[ERROR] Wrong parameters in NeuralNetwork.fit()")
        # Training cycle
        for epoch in range(1, nn.hyper_param.epochs + 1):
            avg_cost = 0.0
            n_batch = ds_train.n_batch
            t0 = time.time()
            # Loop over all batches
            for batch_idx in range(n_batch):
                batch_x, batch_y = ds_train.get_next_batch()
                feed_dict = {nn.tf_desc.x: batch_x, nn.tf_desc.y: batch_y,
                             nn.tf_desc.keep_prob: nn.hyper_param.keep_prob,
                             nn.tf_desc.is_training: True}
                # Run loss and optimizer operations
                # to get loss value and update mlp weights and biases
                # j, steps = self.tf_session.run([nn.tf_desc.loss,
                #                          nn.tf_desc.global_step],
                #                         feed_dict)
                # self.tf_session.run([nn.tf_desc.optimzr],feed_dict)
                j, steps, _ = self.tf_session.run([nn.tf_desc.loss,
                                                   nn.tf_desc.global_step,
                                                   nn.tf_desc.optimzr],
                                                  feed_dict)
                # Compute average cost
                avg_cost += j / n_batch
            t1 = time.time()
            # Display cost periodically
            t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if epoch % nn.hyper_param.cost_print_period == 0:
                print("{} - {} - ".format(t, nn_desc.name, ) +
                      "Epoch: {}/{} - ".format(epoch, nn.hyper_param.epochs) +
                      "{:0.3f}s - avg_cost = {:.9f}".format(t1 - t0,
                                                            avg_cost) +
                      " - Steps: {}".format(steps))
            # Display training set and dev set accuracies periodically
            if (epoch == 1) or \
                    (epoch % nn.hyper_param.eval_print_period == 0):
                # calculation on training set
                feed_dict = {nn.tf_desc.x: ds_train.data.values,
                             nn.tf_desc.y: ds_train.labels_onehot.values,
                             nn.tf_desc.keep_prob: 1.0,
                             nn.tf_desc.is_training: False}
                if TENSORBOARD_ENABLED is True:
                    train_acc, train_loss, summary = self.tf_session.run(
                        [nn.tf_desc.accuracy, nn.tf_desc.loss, self.tf_merged],
                        feed_dict)
                    self.tf_train_writer.add_summary(summary, epoch)
                else:
                    train_acc, train_loss = self.tf_session.run(
                        [nn.tf_desc.accuracy, nn.tf_desc.loss],
                        feed_dict)
                print("\ttrain_cost = {:.9f}".format(train_loss),
                      " train_acc =\t{:.9f}".format(train_acc))
                # self.tf_saver.save(self.tf_session, nn.name)
                # calculation on crossval set
                if cv_eval is True:
                    feed_dict = {nn.tf_desc.x: ds_cv.data.values,
                                 nn.tf_desc.y: ds_cv.labels_onehot.values,
                                 nn.tf_desc.keep_prob: 1.0,
                                 nn.tf_desc.is_training: False}
                    if TENSORBOARD_ENABLED is True:
                        cv_acc, cv_loss, summary = self.tf_session.run(
                          [nn.tf_desc.accuracy, nn.tf_desc.loss,
                           self.tf_merged],
                          feed_dict)
                        self.tf_cv_writer.add_summary(summary, epoch)
                    else:
                        cv_acc, cv_loss = self.tf_session.run(
                            [nn.tf_desc.accuracy, nn.tf_desc.loss],
                            feed_dict)
                    print("\tcv_cost =\t {:.9f}".format(cv_loss),
                          " cv_acc =\t\t{:.9f}".format(cv_acc))
        feed_dict = {nn.tf_desc.x: ds_train.data.values,
                     nn.tf_desc.y: ds_train.labels_onehot.values,
                     nn.tf_desc.keep_prob: 1.0,
                     nn.tf_desc.is_training: False}
        train_cm, train_pred = self.tf_session.run([nn.tf_desc.cm,
                                                    nn.tf_desc.prediction],
                                                   feed_dict)
        if ds_train.n_classes == 2:
            train_true_labels = ds_train.labels.values.reshape(
                (np.size(ds_train.labels.values), 1))
            train_attack_predictions = train_pred[:, 1]
            train_fpr_vect, train_tpr_vect, _ = \
                roc_curve(train_true_labels, train_attack_predictions)
            train_roc_auc = auc(train_fpr_vect, train_tpr_vect)
        else:
            train_fpr_vect = [0]
            train_tpr_vect = [0]
            train_roc_auc = 0
        save_path = self.tf_saver.save(self.tf_session,
                                       './ckpt_dir/' + nn.name + '.ckpt')
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("[INFO] {} - TF Model saved in path: {}".format(t, save_path))
        # self.tf_session.run(self.tf_epochs.assign(self.hyper_param.epochs))
        return (train_loss, train_acc, train_cm, train_fpr_vect,
                train_tpr_vect, train_roc_auc)

    def fit_sequential(self, *ds):
        """
        Train neural network

        Parameters
        ----------
        ds: List[int]
            list of Dataset that can contain either one or two dataset
            first element is the training set
            second element is the development set

        Returns
        -------
        loss: float
            Cost value after training
        acc: float
            Accuracy value after training
        cm: ndarray
            confusion matrix
        fpr_vect: array
            vector of training false positive rate
        tpr_vect: array
            vector of training true positive rate
        roc_auc: float
            area under the curve for training
        """
        if len(ds) == 2:
            training_set = ds[0]
            ds_cv = ds[1]
        elif len(ds) == 1:
            training_set = ds[0]
            ds_cv = []
        else:
            raise Exception("[ERROR] Wrong parameters in fit()")
        training_set.set_batch_size(self.nn_desc_list[
                                        0].hyper_param.batch_size)
        ds_cv.set_batch_size(self.nn_desc_list[0].hyper_param.batch_size)
        # create ensemble training sets
        if self.method == 'bagging':
            if self.n_estimators < 1:
                raise Exception("[ERROR] Wrong number of estimators (bagging)")
            ds_train_list = training_set.random_sampling(self.n_estimators)
        elif self.method == '':
            ds_train_list = training_set.random_sampling(1)
        else:
            raise Exception("[ERROR] Unsupported ensemble method")
        # Training cycle
        for idx in range(self.n_estimators):
            ds_train = ds_train_list[idx]
            nn = self.nn_desc_list[idx]
            t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("[INFO] {} - Training estimator #{} ...".format(t, idx))
            if len(ds) == 1:
                self.fit_singleNN(nn, ds_train)
            else:
                self.fit_singleNN(nn, ds_train, ds_cv)
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("[INFO] {} - Training Done...".format(t))

    def fit_parallel(self, *ds):
        """
        Train neural network

        Parameters
        ----------
        ds: List[int]
            list of Dataset that can contain either one or two dataset
            first element is the training set
            second element is the development set

        Returns
        -------
        loss: float
            Cost value after training
        acc: float
            Accuracy value after training
        cm: ndarray
            confusion matrix
        fpr_vect: array
            vector of training false positive rate
        tpr_vect: array
            vector of training true positive rate
        roc_auc: float
            area under the curve for training
        """
        if len(ds) == 2:
            training_set = ds[0]
            cv_eval = True
            ds_cv = ds[1]
        elif len(ds) == 1:
            training_set = ds[0]
            cv_eval = False
            ds_cv = []
        else:
            raise Exception("[ERROR] Wrong parameters in fit()")
        training_set.set_batch_size(self.nn_desc_list[
                                        0].hyper_param.batch_size)
        ds_cv.set_batch_size(self.nn_desc_list[0].hyper_param.batch_size)
        # create ensemble training sets
        if self.method == 'bagging':
            if self.n_estimators < 1:
                raise Exception("[ERROR] Wrong number of estimators (bagging)")
            ds_train_list = training_set.random_sampling(self.n_estimators)
        elif self.method == '':
            ds_train_list = training_set.random_sampling(1)
        else:
            raise Exception("[ERROR] Unsupported ensemble method")
        # Training cycle
        for epoch in range(1, self.nn_desc_list[0].hyper_param.epochs + 1):
            avg_cost_list = []
            for nn_idx in range(self.n_estimators):
                avg_cost_list.append(0)
            n_batch = math.ceil(ds_train_list[0].n_samples /
                                training_set.batch_size)
            t0 = time.time()
            # Loop over all batches
            session_outputs = ()
            for batch_idx in range(n_batch):
                # prepare values for placeholders
                feed_dict = {}
                session_params = []
                for nn_idx in range(self.n_estimators):
                    batch_x, batch_y = ds_train_list[nn_idx].get_next_batch()
                    feed_dict.update(
                        {self.nn_desc_list[nn_idx].tf_desc.x: batch_x,
                         self.nn_desc_list[nn_idx].tf_desc.y: batch_y,
                         self.nn_desc_list[nn_idx].tf_desc.keep_prob:
                             self.nn_desc_list[nn_idx].hyper_param.keep_prob,
                         self.nn_desc_list[nn_idx].tf_desc.is_training: True})
                    session_params.append(
                        self.nn_desc_list[nn_idx].tf_desc.loss)
                    session_params.append(
                        self.nn_desc_list[nn_idx].tf_desc.global_step)
                    session_params.append(
                        self.nn_desc_list[nn_idx].tf_desc.optimzr)
                # Run loss and optimizer operations
                session_outputs = self.tf_session.run(session_params,
                                                      feed_dict)
                n_outputs_per_nn = int(
                    len(session_outputs) / self.n_estimators)
                # Compute average cost
                for nn_idx in range(self.n_estimators):
                    avg_cost_list[nn_idx] += \
                        session_outputs[n_outputs_per_nn * nn_idx] / n_batch
            t1 = time.time()
            # Display cost periodically
            t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            nn = self.nn_desc_list[0]
            if epoch % nn.hyper_param.cost_print_period == 0:
                print("{} - ".format(t) +
                      "Epoch: {}/{} - ".format(epoch, nn.hyper_param.epochs) +
                      "{:0.3f}s - \t{} ".format(t1 - t0, nn.name) +
                      "avg_cost = {:.9f} - ".format(avg_cost_list[0]) +
                      "Steps: {}".format(session_outputs[1]))
                if self.n_estimators > 1:
                    for nn_idx in range(1, self.n_estimators):
                        print("\t\t\t\t\t\t\t\t\t\t\t\t{} avg_cost = {:.9f}".
                              format(self.nn_desc_list[nn_idx].name,
                                     avg_cost_list[nn_idx]))
            # Display training set and dev set accuracies periodically
            if (epoch == 1) or \
                    (epoch % nn.hyper_param.eval_print_period == 0):
                # calculation on training set
                feed_dict = {}
                session_params = []
                # session_outputs = ()
                for nn_idx in range(self.n_estimators):
                    nn = self.nn_desc_list[nn_idx]
                    train_set = ds_train_list[nn_idx]
                    feed_dict.update(
                        {nn.tf_desc.x: train_set.data.values,
                         nn.tf_desc.y: train_set.labels_onehot.values,
                         nn.tf_desc.keep_prob: 1.0,
                         nn.tf_desc.is_training: False
                         })
                    session_params.append(nn.tf_desc.accuracy)
                    session_params.append(nn.tf_desc.loss)
                # Run accuracy and loss operations
                session_outputs = self.tf_session.run(session_params,
                                                      feed_dict)
                n_outputs_per_nn = int(
                    len(session_outputs) / self.n_estimators)
                for nn_idx in range(self.n_estimators):
                    nn = self.nn_desc_list[nn_idx]
                    print("\t{} Training  Cost = {:.9f} - Acc = {:.9f}".
                          format(nn.name,
                                 session_outputs[
                                     n_outputs_per_nn * nn_idx + 1],
                                 session_outputs[n_outputs_per_nn * nn_idx]))

                self.tf_train_writer.add_summary(summary, epoch)
                # self.tf_saver.save(self.tf_session, nn.name)
                # calculation on dev set
                if cv_eval is True:
                    feed_dict = {}
                    session_params = []
                    # session_outputs = ()
                    for nn_idx in range(self.n_estimators):
                        nn = self.nn_desc_list[nn_idx]
                        feed_dict.update(
                            {nn.tf_desc.x: ds_cv.data.values,
                             nn.tf_desc.y: ds_cv.labels_onehot.values,
                             nn.tf_desc.keep_prob: 1.0,
                             nn.tf_desc.is_training: False})
                        session_params.append(nn.tf_desc.accuracy)
                        session_params.append(nn.tf_desc.loss)
                    # Run accuracy and loss operations
                    session_outputs = self.tf_session.run(session_params,
                                                          feed_dict)
                    n_outputs_per_nn = int(len(session_outputs) /
                                           self.n_estimators)
                    for nn_idx in range(self.n_estimators):
                        print("\t{} Cross-Val Cost = {:.9f} - Acc = {:.9f}".
                              format(self.nn_desc_list[nn_idx].name,
                                     session_outputs[n_outputs_per_nn * nn_idx
                                                     + 1],
                                     session_outputs[
                                         n_outputs_per_nn * nn_idx]))

        feed_dict = {}
        for nn_idx in range(self.n_estimators):
            nn = self.nn_desc_list[nn_idx]
            feed_dict.update(
                {nn.tf_desc.x: ds_train_list[nn_idx].data.values,
                 nn.tf_desc.y: ds_train_list[nn_idx].labels_onehot.values,
                 nn.tf_desc.keep_prob: 1.0,
                 nn.tf_desc.is_training: False})
        session_params = [self.tf_accuracy, self.tf_loss,
                          self.tf_cm, self.tf_voting_outputs]
        train_acc, train_loss, train_cm, train_pred = self.tf_session.run(
            session_params, feed_dict)
        if ds_train_list[0].n_classes == 2:
            train_true_labels = ds_train_list[0].labels.values.reshape(
                (np.size(ds_train_list[0].labels.values), 1))
            train_attack_predictions = train_pred[:, 1]
            train_fpr_vect, train_tpr_vect, _ = \
                roc_curve(train_true_labels, train_attack_predictions)
            train_roc_auc = auc(train_fpr_vect, train_tpr_vect)
        else:
            train_fpr_vect = [0]
            train_tpr_vect = [0]
            train_roc_auc = 0
        save_path = self.tf_saver.save(self.tf_session,
                                       './ckpt_dir/' + 'test_ensemble' +
                                       '.ckpt')
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("[INFO] {} - TF Model saved in path: {}".format(t, save_path))
        # self.tf_session.run(self.tf_epochs.assign(self.hyper_param.epochs))
        return (train_loss, train_acc, train_cm, train_fpr_vect,
                train_tpr_vect, train_roc_auc)

    def predict(self, data):
        """
        Predict labels from provided data

        Parameters
        ----------
        data: ndarray
            numpy array containing examples in rows and features in columns

        Returns
        -------
        predictions: ndarray
            numpy array providing labels (onehot-decoded)

        """
        feed_dict = {}
        for nn in self.nn_desc_list:
            feed_dict.update({nn.tf_desc.x: data,
                              nn.tf_desc.keep_prob: 1.0,
                              nn.tf_desc.is_training: False})
        voting_outputs = self.tf_session.run([self.tf_voting_outputs],
                                             feed_dict)
        return tf.argmax(voting_outputs, 1)

    def get_metrics(self, ds):
        """
        Calculate metrics: accuracy, loss and confusion matrix

        Parameters
        ----------
        ds: Dataset
            dataset containing data and labels for which metrics are calculated

        Returns
        -------
        loss_value_list: list
            loss values for the given data and labels for each estimator
        acc_value_list: list
            accuracy of the prediction for each estimator
        cm_list: list
            confusion matrix for each estimator
        pred_list: list
            predictions for each estimator
        cm: ndarray
            confusion matrix of the ensemble
        fpr_vect: array
            vector of training false positive rate
        tpr_vect: array
            vector of training true positive rate
        roc_auc: float
            area under the curve for training
        """
        # metrics on each estimator
        loss_value_list = []
        acc_value_list = []
        cm_list = []
        pred_list = []
        for nn in self.nn_desc_list:
            feed_dict = {nn.tf_desc.x: ds.data.values,
                         nn.tf_desc.y: ds.labels_onehot.values,
                         nn.tf_desc.keep_prob: 1.0,
                         nn.tf_desc.is_training: False}
            (loss_value, acc_value,
             cm, pred) = self.tf_session.run([nn.tf_desc.loss,
                                              nn.tf_desc.accuracy,
                                              nn.tf_desc.cm,
                                              nn.tf_desc.prediction],
                                             feed_dict)
            loss_value_list.append(loss_value)
            acc_value_list.append(acc_value)
            cm_list.append(cm)
            pred_list.append(pred)
        # metrics on the ensemble
        feed_dict = {}
        for nn in self.nn_desc_list:
            feed_dict.update({nn.tf_desc.x: ds.data.values,
                              nn.tf_desc.y: ds.labels_onehot.values,
                              nn.tf_desc.keep_prob: 1.0,
                              nn.tf_desc.is_training: False})
        voting_outputs, cm = self.tf_session.run([self.tf_voting_outputs,
                                                  self.tf_cm],
                                                 feed_dict)
        if ds.n_classes == 2:
            true_labels = ds.labels.values.reshape((np.size(ds.labels.values),
                                                    1))
            attack_predictions = voting_outputs[:, 1]
            fpr_vect, tpr_vect, _ = roc_curve(true_labels, attack_predictions)
            roc_auc = auc(fpr_vect, tpr_vect)
        else:
            fpr_vect = [0]
            tpr_vect = [0]
            roc_auc = 0
        return loss_value_list, acc_value_list, cm_list, pred_list, cm, \
            fpr_vect, tpr_vect, roc_auc

    def close(self):
        """
        Close Tensorflow session

        Returns
        -------
        None

        """
        self.tf_session.close()

    def _build_graph(self):
        """
        Build graph according to topology

        Returns
        -------

        """
        for i, nn in enumerate(self.nn_desc_list):
            with tf.compat.v1.variable_scope(nn.name):
                first_layer = nn.topology.dict_topo[1]
                n_layers = len(nn.topology.dict_topo)
                last_layer = nn.topology.dict_topo[n_layers]
                x = tf.compat.v1.placeholder('float',
                                             [None, first_layer.n_inputs],
                                             name='x')
                y = tf.compat.v1.placeholder('float',
                                             [None, last_layer.n_outputs],
                                             name='y')
                keep_prob = tf.compat.v1.placeholder('float',
                                                     shape=(),
                                                     name='keep_prob')
                is_training = tf.compat.v1.placeholder('bool', shape=(),
                                                       name="is_training")
                nn.tf_desc.global_step = tf.Variable(0, name='global_step',
                                                     trainable=False)

                regularizer = 0
                set_act_fn = True
                layer_inputs = x
                batch_norm = nn.hyper_param.batch_norm
                for idx, layer in nn.topology.dict_topo.items():
                    if idx == n_layers:
                        set_act_fn = False
                    layer.tf_create(layer_inputs, set_act_fn, keep_prob,
                                    batch_norm, is_training)
                    layer_inputs = layer.tf_outputs
                    if nn.hyper_param.l2_reg != 0:
                        # optimization: no processing is not needed
                        regularizer += layer.tf_l2_loss
                nn.tf_desc.regularizer = regularizer
                logits = last_layer.tf_outputs
                if last_layer.activation_name == 'sigmoid':
                    outputs = tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=logits, labels=y)
                    if nn.hyper_param.l2_reg != 0:
                        # optimization: no processing is not needed
                        outputs += nn.hyper_param.l2_reg * regularizer
                elif last_layer.activation_name == 'softmax':
                    outputs = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=logits, labels=y)
                    if nn.hyper_param.l2_reg != 0:
                        # optimization: no processing is not needed
                        outputs += nn.hyper_param.l2_reg * regularizer
                else:
                    outputs = logits
                nn.tf_desc.x = x
                nn.tf_desc.y = y
                nn.tf_desc.keep_prob = keep_prob
                nn.tf_desc.is_training = is_training
                nn.tf_desc.logits = logits
                with tf.name_scope('loss'):
                    nn.tf_desc.loss = tf.reduce_mean(outputs)
                nn.tf_desc.prediction = last_layer.tf_act_fn(nn.tf_desc.logits)
            with tf.compat.v1.variable_scope(nn.name + '_metrics'):
                # Operations for accuracy calculation
                correct_pred = tf.equal(tf.argmax(nn.tf_desc.prediction, 1),
                                        tf.argmax(y, 1))
                with tf.name_scope('acc'):
                    nn.tf_desc.accuracy = tf.reduce_mean(tf.cast(correct_pred,
                                                                 tf.float32))
                with tf.name_scope('cm'):
                    nn.tf_desc.cm = tf.math.confusion_matrix(
                        labels=tf.argmax(nn.tf_desc.y, 1),
                        predictions=tf.argmax(nn.tf_desc.prediction, 1),
                        num_classes=last_layer.n_outputs)
                with tf.name_scope('AUC'):
                    nn.tf_desc.auc_op = tf.compat.v1.metrics.auc(
                        labels=tf.argmax(nn.tf_desc.y, 1),
                        predictions=nn.tf_desc.prediction[:, 1],
                        num_thresholds=1000)
                tf.compat.v1.summary.scalar('accuracy', nn.tf_desc.accuracy)
                tf.compat.v1.summary.scalar('loss', nn.tf_desc.loss)
                self.tf_saver = tf.compat.v1.train.Saver()
        # graph elements for the voting classifier
        with tf.name_scope('voting_clf'):
            # operations for voting classifier
            if self.voting_method == 'soft':
                accumulated_votes = self.nn_desc_list[0].tf_desc.prediction
                if self.n_estimators != 1:
                    for i in range(self.n_estimators - 1):
                        accumulated_votes = tf.add(accumulated_votes,
                                                   self.nn_desc_list[i + 1].
                                                   tf_desc.prediction)
                    voting_outputs = tf.divide(accumulated_votes,
                                               self.n_estimators)
                    self.tf_voting_outputs = voting_outputs
                else:
                    self.tf_voting_outputs = accumulated_votes
            else:
                raise Exception("[ERROR] Only soft voting is implemented")
        # graph elements for the confusion matrix of the ensemble
        with tf.name_scope('global_metrics'):
            nn = self.nn_desc_list[0]
            n_layers = len(nn.topology.dict_topo)
            last_layer = nn.topology.dict_topo[n_layers]
            with tf.name_scope('cm'):
                self.tf_cm = tf.math.confusion_matrix(
                    labels=tf.argmax(nn.tf_desc.y, 1),
                    predictions=tf.argmax(self.tf_voting_outputs, 1),
                    num_classes=last_layer.n_outputs)
            # Operations for accuracy calculation
            correct_pred = tf.equal(tf.argmax(self.tf_voting_outputs, 1),
                                    tf.argmax(nn.tf_desc.y, 1))
            with tf.name_scope('acc'):
                self.tf_accuracy = tf.reduce_mean(tf.cast(correct_pred,
                                                          tf.float32))
            with tf.name_scope('loss'):
                loss_list = []
                for nn_idx in range(self.n_estimators):
                    loss_list.append(self.nn_desc_list[nn_idx].tf_desc.loss)
                self.tf_loss = tf.reduce_mean(loss_list)
            # self.tf_run_options =
            # tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # self.tf_run_metadata = tf.RunMetadata()

    def _set_optimizer(self):
        """
        Configure tensorflow optimizer

        Parameters
        ----------

        Returns
        -------

        """
        for i, nn in enumerate(self.nn_desc_list):
            optimzr = nn.hyper_param.optimizer
            params = nn.hyper_param.optim_params
            extra_ops = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_ops):
                if optimzr == 'GradientDescent' and len(params) == 1:
                    tf_optim = tf.train.GradientDescentOptimizer(params[0]). \
                        minimize(nn.tf_desc.loss,
                                 global_step=nn.tf_desc.global_step)
                elif optimzr == 'Momentum' and len(params) == 2:
                    tf_optim = tf.train.MomentumOptimizer(
                        learning_rate=params[0],
                        momentum=params[1]). \
                        minimize(nn.tf_desc.loss,
                                 global_step=nn.tf_desc.global_step)
                elif optimzr == 'Adam' and len(params) == 3:
                    tf_optim = tf.compat.v1.train.AdamOptimizer(
                        learning_rate=params[0],
                        beta1=params[1],
                        beta2=params[2],
                        name='Adam_' + str(i)). \
                        minimize(nn.tf_desc.loss,
                                 global_step=nn.tf_desc.global_step)
                elif optimzr == 'AdaDelta' and len(params) == 2:
                    tf_optim = tf.train.AdadeltaOptimizer(
                        learning_rate=params[0], rho=params[1]). \
                        minimize(nn.tf_desc.loss,
                                 global_step=nn.tf_desc.global_step)
                elif optimzr == 'RMSProp' and len(params) == 3:
                    tf_optim = tf.train.RMSPropOptimizer(
                        learning_rate=params[0], decay=params[1],
                        momentum=params[2]). \
                        minimize(nn.tf_desc.loss,
                                 global_step=nn.tf_desc.global_step)
                else:
                    raise Exception("[ERROR] issues in optimizer config")
            nn.tf_desc.optimzr = tf_optim

    def _voting_classifier(self, voting_inputs):
        """
        Hard/soft voting classifier

        Parameters
        ----------
        voting_inputs: list
            list of prediction from each estimator

        Returns
        -------
        voting_outputs: ndarray
            Results of the voting classifier
        """
        if self.n_estimators > 1:
            feed_dict = {self.tf_voting_inputs: voting_inputs}
            voting_outputs = self.tf_session.run([self.tf_voting_outputs],
                                                 feed_dict)
        else:
            voting_outputs = voting_inputs
        return voting_outputs


class NeuralNetwork(NeuralNetworkEnsemble):
    """ Neural Network Class """

    def __init__(self, nn_descriptor):
        """
        Creator function
        """
        name = nn_descriptor.name
        nn_descriptor_list = [nn_descriptor]
        NeuralNetworkEnsemble.__init__(self, name=name,
                                       nn_descriptor_list=nn_descriptor_list)

    def fit(self, *ds):
        """
        Train neural network

        Parameters
        ----------
        ds: List[int]
            list of Dataset that can contain either one or two dataset
            first element is the training set
            second element is the development set

        Returns
        -------
        loss: float
            Cost value after training
        acc: float
            Accuracy value after training
        cm: ndarray
            confusion matrix
        fpr_vect: array
            vector of training false positive rate
        tpr_vect: array
            vector of training true positive rate
        roc_auc: float
            area under the curve for training
        """
        if len(ds) == 2:
            ds_train = ds[0]
            ds_cv = ds[1]
        elif len(ds) == 1:
            ds_train = ds[0]
            ds_cv = []
        else:
            raise Exception("[ERROR] Wrong parameters in fit()")
        ds_train.set_batch_size(self.nn_desc_list[0].hyper_param.batch_size)
        ds_cv.set_batch_size(self.nn_desc_list[0].hyper_param.batch_size)
        # Training cycle
        nn = self.nn_desc_list[0]
        if len(ds) == 1:
            return NeuralNetworkEnsemble.fit_singleNN(self, nn, ds_train)
        else:
            return NeuralNetworkEnsemble.fit_singleNN(self, nn, ds_train,
                                                      ds_cv)
