import numpy as np
import pandas as pd
import os
from Constants import *
from sklearn.preprocessing import LabelEncoder
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    #from sklearn.cross_validation import StratifiedShuffleSplit
    # cross_validation -> now called: model_selection
    # https://stackoverflow.com/questions/30667525/importerror-no-module-named-sklearn-cross-validation
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[int(row), int(col)] = 1
    return out


class load_data():
    # data_train, data_test and le are public
    def __init__(self, x_train, y_train, x_test, y_test):
        self._load(x_train, y_train, x_test, y_test)
        
    def _load(self, x_train, y_train, x_test, y_test):
        print("Loading training data")
        train_data = self._make_dataset(x_train, y_train)
        print("Loading test data")
        test_data = self._make_dataset(x_test, y_test)        
        # need to reformat the train for validation split reasons in the batch_generator
        self.train = self._format_dataset(train_data)
        self.test = self._format_dataset(test_data)
        
        
    def _make_dataset(self, df, y=None):
        seq_length=SEQ_LENGTH
        # make dataset
        data = dict()

        for i, dat in enumerate(df.iterrows()):
            index, row = dat
            sample = dict()
            features = row.values
            sample[FEATS[0]] = features[:seq_length]
            sample[FEATS[1]] = features[seq_length:seq_length*2]
            sample[FEATS[2]] = features[seq_length*2:seq_length*3]
            sample[FEATS[3]] = features[seq_length*3:seq_length*4]
            sample[TEST_FEATS[0]] = features[seq_length*4]
            sample[TEST_FEATS[1]] = features[seq_length*4+1]
            sample[TEST_FEATS[2]] = features[seq_length*4+2]
            sample[TEST_FEATS[3]] = features[seq_length*4+3]
            sample['t'] = np.asarray(y[i], dtype='int32')
            data[index] = sample
            if i % 1000 == 0:
                print("\t%d of %d" % (i, len(df)))
        
        return data

    def _format_dataset(self, df):
        # making arrays with all data in, is nessesary when doing validation split
        data = dict()
        value = list(df.values())[0]
        feature_tot_shp = (len(df), SEQ_LENGTH)
        data[FEATS[0]] = np.zeros(feature_tot_shp, dtype='float32')
        data[FEATS[1]] = np.zeros(feature_tot_shp, dtype='float32')
        data[FEATS[2]] = np.zeros(feature_tot_shp, dtype='float32')
        data[FEATS[3]] = np.zeros(feature_tot_shp, dtype='float32')
        data[TEST_FEATS[0]] = np.zeros((len(df),), dtype='float32')
        data[TEST_FEATS[1]] = np.zeros((len(df),), dtype='float32')
        data[TEST_FEATS[2]] = np.zeros((len(df),), dtype='float32')
        data[TEST_FEATS[3]] = np.zeros((len(df),), dtype='float32')
        data['ts'] = np.zeros((len(df),), dtype='int32')
        data['ids'] = np.zeros((len(df),), dtype='int32')
        for i, pair in enumerate(df.items()):
            key, value = pair
            data[FEATS[0]][i] = value[FEATS[0]]
            data[FEATS[1]][i] = value[FEATS[1]]
            data[FEATS[2]][i] = value[FEATS[2]]
            data[FEATS[3]][i] = value[FEATS[3]]
            data[TEST_FEATS[0]][i] = value[TEST_FEATS[0]]
            data[TEST_FEATS[1]][i] = value[TEST_FEATS[1]]
            data[TEST_FEATS[2]][i] = value[TEST_FEATS[2]]
            data[TEST_FEATS[3]][i] = value[TEST_FEATS[3]]
            data['ts'][i] = value['t']
            data['ids'][i] = key
        
        return data

    
class batch_generator():
    def __init__(self, data, batch_size=BATCH_SIZE, num_classes=NUM_CLASSES,
                 num_iterations=5e3, num_features=SEQ_LENGTH, seed=42, val_size=0.1):
        self._train = data.train
        self._test = data.test
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._num_iterations = num_iterations
        self._num_features = num_features
        self._seed = seed
        self._val_size = val_size
        self._valid_split()
        
    def _valid_split(self):
        # Updated to use: model_selection
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self._val_size,
            random_state=self._seed
        ).split(
            np.zeros(self._train['ts'].shape),
            self._train['ts']
        )
        self._idcs_train, self._idcs_valid = next(iter(sss))
        
    def _shuffle_train(self):
        np.random.shuffle(self._idcs_train)

    def _batch_init(self, purpose):
        assert purpose in ['train', 'valid', 'test']
        batch_holder = dict()
        batch_holder[FEATS[0]] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        batch_holder[FEATS[1]] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        batch_holder[FEATS[2]] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        batch_holder[FEATS[3]] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        batch_holder[TEST_FEATS[0]] = np.zeros((self._batch_size,), dtype='float32')
        batch_holder[TEST_FEATS[1]] = np.zeros((self._batch_size,), dtype='float32')
        batch_holder[TEST_FEATS[2]] = np.zeros((self._batch_size,), dtype='float32')
        batch_holder[TEST_FEATS[3]] = np.zeros((self._batch_size,), dtype='float32')
        batch_holder['ts'] = np.zeros((self._batch_size, self._num_classes), dtype='float32')          
        batch_holder['ids'] = []
        return batch_holder

    def gen_valid(self):
        batch = self._batch_init(purpose='valid')
        i = 0
        for idx in self._idcs_valid:
            batch[FEATS[0]][i] = self._train[FEATS[0]][idx]
            batch[FEATS[1]][i] = self._train[FEATS[1]][idx]
            batch[FEATS[2]][i] = self._train[FEATS[2]][idx]
            batch[FEATS[3]][i] = self._train[FEATS[3]][idx]
            batch[TEST_FEATS[0]][i] = self._train[TEST_FEATS[0]][idx]
            batch[TEST_FEATS[1]][i] = self._train[TEST_FEATS[1]][idx]
            batch[TEST_FEATS[2]][i] = self._train[TEST_FEATS[2]][idx]
            batch[TEST_FEATS[3]][i] = self._train[TEST_FEATS[3]][idx]
            batch['ts'][i] = onehot(np.asarray([self._train['ts'][idx]], dtype='float32'), self._num_classes)
            i += 1
            if i >= self._batch_size:
                yield batch, i
                batch = self._batch_init(purpose='valid')
                i = 0
        if i != 0:
            batch['ts'] = onehot(np.asarray([self._train['ts'][idx]], dtype='float32'), self._num_classes)
            batch[FEATS[0]] = batch[FEATS[0]][:i]
            batch[FEATS[1]] = batch[FEATS[1]][:i]
            batch[FEATS[2]] = batch[FEATS[2]][:i]
            batch[FEATS[3]] = batch[FEATS[3]][:i]
            batch[TEST_FEATS[0]] = batch[TEST_FEATS[0]][:i]
            batch[TEST_FEATS[1]] = batch[TEST_FEATS[1]][:i]
            batch[TEST_FEATS[2]] = batch[TEST_FEATS[2]][:i]
            batch[TEST_FEATS[3]] = batch[TEST_FEATS[3]][:i]
            yield batch, i

    def gen_test(self):
        batch = self._batch_init(purpose='test')
        i = 0
        for idx in range(len(self._test['ids'])):
            batch[FEATS[0]][i] = self._test[FEATS[0]][idx]
            batch[FEATS[1]][i] = self._test[FEATS[1]][idx]
            batch[FEATS[2]][i] = self._test[FEATS[2]][idx]
            batch[FEATS[3]][i] = self._test[FEATS[3]][idx]
            batch[TEST_FEATS[0]][i] = self._test[TEST_FEATS[0]][idx]
            batch[TEST_FEATS[1]][i] = self._test[TEST_FEATS[1]][idx]
            batch[TEST_FEATS[2]][i] = self._test[TEST_FEATS[2]][idx]
            batch[TEST_FEATS[3]][i] = self._test[TEST_FEATS[3]][idx]
            batch['ts'][i] = onehot(np.asarray([self._test['ts'][idx]], dtype='float32'), self._num_classes)
            batch['ids'].append(self._test['ids'][idx])
            i += 1
            if i >= self._batch_size:
                yield batch, i
                batch = self._batch_init(purpose='test')
                i = 0
        if i != 0:
            yield batch, i     

    def gen_train(self):
        batch = self._batch_init(purpose='train')
        iteration = 0
        i = 0
        while iteration < self._num_iterations:
            # shuffling all batches
            self._shuffle_train()
            for idx in self._idcs_train:
                # extract data from dict
                batch[FEATS[0]][i] = self._train[FEATS[0]][idx]
                batch[FEATS[1]][i] = self._train[FEATS[1]][idx]
                batch[FEATS[2]][i] = self._train[FEATS[2]][idx]
                batch[FEATS[3]][i] = self._train[FEATS[3]][idx]
                batch[TEST_FEATS[0]][i] = self._train[TEST_FEATS[0]][idx]
                batch[TEST_FEATS[1]][i] = self._train[TEST_FEATS[1]][idx]
                batch[TEST_FEATS[2]][i] = self._train[TEST_FEATS[2]][idx]
                batch[TEST_FEATS[3]][i] = self._train[TEST_FEATS[3]][idx]
                batch['ts'][i] = onehot(np.asarray([self._train['ts'][idx]], dtype='float32'), self._num_classes)
                i += 1
                if i >= self._batch_size:
                    yield batch
                    batch = self._batch_init(purpose='train')
                    i = 0
                    iteration += 1
#                     if iteration >= self._num_iterations:
#                         break
                    
