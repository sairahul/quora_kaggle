
import numpy as np
import pandas as pd

from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop, SGD, Adam
from model import *

def read_pandas_df(model_name):
    df = pd.read_pickle(model_name)
    return df

def create_test_train_data(df):
    #shuffle df
    df = df.reindex(np.random.permutation(df.index))

    # set number of train and test instances
    num_train = int(df.shape[0] * 0.88)
    num_test = df.shape[0] - num_train
    print("Number of training pairs: %i"%(num_train))
    print("Number of testing pairs: %i"%(num_test))

    # init data data arrays
    X_train = np.zeros([num_train, 2, 300])
    X_test  = np.zeros([num_test, 2, 300])
    Y_train = np.zeros([num_train])
    Y_test = np.zeros([num_test])

    # format data
    b = [a[None,:] for a in list(df['q1_feats'].values)]
    q1_feats = np.concatenate(b, axis=0)

    b = [a[None,:] for a in list(df['q2_feats'].values)]
    q2_feats = np.concatenate(b, axis=0)

    # fill data arrays with features
    X_train[:,0,:] = q1_feats[:num_train]
    X_train[:,1,:] = q2_feats[:num_train]
    Y_train = df[:num_train]['is_duplicate'].values

    X_test[:,0,:] = q1_feats[num_train:]
    X_test[:,1,:] = q2_feats[num_train:]
    Y_test = df[num_train:]['is_duplicate'].values

    return X_train, Y_train, X_test, Y_test

#import tensorflow as tf
import keras.backend as K
def comp_accuracy(labels, predictions):
    '''
    Compute classification accuracy with a fixed threshold on distances.
    '''
    #return np.mean(np.equal(predictions.ravel() < 0.5, labels))
    pred = K.reshape(predictions, [-1])
    lab  = K.reshape(labels, [-1])
    lab_ = K.cast(lab, 'bool')
    cons = K.constant(0.5)

    pred_ = K.less(pred, cons)
    #pred_ = K.cast(pred_, 'bool')
    mean = K.mean(K.cast(K.equal(pred_, lab_), 'float32'))
    return mean

def main():

    #df = read_pandas_df('quora/data/spacy_en_tfidf.pkl')
    df = read_pandas_df('quora/data/spacy_enweb_tfidf.pkl')
    X_train, Y_train, X_test, Y_test = create_test_train_data(df)

    net = create_network(300)
    filepath="models/spacy_enweb_tfidf-{epoch:02d}-{val_comp_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]

    # train
    #optimizer = SGD(lr=1, momentum=0.8, nesterov=True, decay=0.004)
    optimizer = Adam(lr=0.001)
    net.compile(loss=contrastive_loss, optimizer=optimizer, metrics=[comp_accuracy])

    for epoch in range(50):
	#net.fit([X_train[:,0,:], X_train[:,1,:]], Y_train,
	#      validation_data=([X_test[:,0,:], X_test[:,1,:]], Y_test),
	#      batch_size=128, nb_epoch=1, shuffle=True, callbacks=callbacks_list)

	net.fit([X_train[:,0,:], X_train[:,1,:]], Y_train, validation_split=0.1,
	      batch_size=128, nb_epoch=epoch, shuffle=True, callbacks=callbacks_list)

	# compute final accuracy on training and test sets
	pred = net.predict([X_test[:,0,:], X_test[:,1,:]], batch_size=128)
	te_acc = compute_accuracy(pred, Y_test)

	#print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
	print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

if __name__=="__main__":
    main()


