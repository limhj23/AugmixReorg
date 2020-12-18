import sys
import tensorflow as tf
import multiprocessing
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import OrderedEnqueuer, Progbar
from preprocess.datagen import DataGenerator
from preprocess.dataset import get_imgset_lblset
import preprocess.models as models
import execute.utils as utils
from sklearn.model_selection import KFold
import numpy as np
import os
from tensorflow.keras.backend import clear_session
import shutil
import json

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def train_step():
    @tf.function
    def apply_grad(model, clean, aug1, aug2, labels, optim):
        with tf.GradientTape() as tape:
            # get predictions on clean imagesgi
            y_pred_clean = model(clean, training=True)
            
            # get predictions on augmented images
            y_pred_aug1 = model(aug1, training=True)
            y_pred_aug2 = model(aug2, training=True)

            # calculate loss
            loss_value = models.jsd_loss_fn(y_true = labels, 
                                y_pred_clean = y_pred_clean,
                                y_pred_aug1 = y_pred_aug1,
                                y_pred_aug2 = y_pred_aug2)
            
        grads = tape.gradient(loss_value, model.trainable_variables)
        optim.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value, y_pred_clean
    
    return apply_grad

@tf.function
def validate_step(model, images, labels):
    entropy = tf.keras.losses.CategoricalCrossentropy()
    y_pred = model(images, training=False)
    loss = entropy(labels, y_pred)
    return loss, y_pred

def count_class(y):
    num_classes = len(y[0])
    cnts = [0 for i in range(num_classes)]
    for onehot in y:
        onehot = onehot.tolist()
        ind = onehot.index(1.0)
        cnts[ind] += 1
    return cnts

def main(dataname):
    # metric to keep track of 
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()
    train_loss = tf.keras.metrics.Mean()
    test_loss = tf.keras.metrics.Mean()

    X, Y = get_imgset_lblset(dataname)
    np.random.seed(1204)
    np.random.shuffle(X)
    np.random.seed(1204)
    np.random.shuffle(Y)

    num_classes = len(Y[0])
    print("num_classes",num_classes)
    batch_size = 16

    scores = []
    kf = KFold(n_splits=3)
    kf.get_n_splits(X)
    for k, (train_index, test_index) in enumerate(kf.split(X)):
        print("# of train: ", len(train_index))
        print("# of test: ", len(test_index))
        x_train = X[train_index]
        y_train = Y[train_index]
        x_test = X[test_index]
        y_test = Y[test_index]
        
        print("y_train", count_class(y_train))
        print("y_test", count_class(y_test))

        test_indices = np.arange(len(x_test))

        # get the training data generator. We are not using validation generator because the 
        # data is already loaded in memory and we don't have to perform any extra operation 
        # apart from loading the validation images and validation labels.
        ds = DataGenerator(x_train, y_train, num_classes = num_classes, batch_size=batch_size)
        enqueuer = OrderedEnqueuer(ds, use_multiprocessing=False)
        enqueuer.start(workers=1)
        train_ds = enqueuer.get()

        plot_name = f"history_{dataname}_fold_{k}.png"
        history = utils.CTLHistory(filename=plot_name)

        pt = utils.ProgressTracker()

        nb_train_steps = int(np.ceil(len(x_train) / batch_size))
        nb_test_steps = int(np.ceil(len(x_test) / batch_size))

        starting_epoch = 0
        nb_epochs = 100
        
        save_dir_path = os.path.join("./checkpoints",f"{dataname}_fold_{k}")
        if os.path.exists(save_dir_path):
            shutil.rmtree(save_dir_path)

        total_steps = nb_train_steps * nb_epochs


        # get the optimizer
        # SGD with cosine lr is causing NaNs. Need to investigate more
        optim = optimizers.Adam(learning_rate=0.0001)
        model = models.get_resnet50(num_classes)

        checkpoint_prefix = os.path.join(save_dir_path,"ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optim, model=model)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, 
                                                        directory=save_dir_path,
                                                        max_to_keep=10)

        train_step_fn = train_step()
        for epoch in range(starting_epoch, nb_epochs):
            pbar = Progbar(target=nb_train_steps, interval=0.5, width=30)
            # Train for an epoch and keep track of 
            # loss and accracy for each batch.
            for bno, (images, labels) in enumerate(train_ds):
                if bno == nb_train_steps:
                    break

                # Get the batch data 
                clean, aug1, aug2 = images
                loss_value, y_pred_clean = train_step_fn(model, clean, aug1, aug2, labels, optim)

                # Record batch loss and batch accuracy
                train_loss(loss_value)
                train_accuracy(labels, y_pred_clean)
                pbar.update(bno+1)

            # Validate after each epoch
            for bno in range(nb_test_steps):
                # Get the indices for the current batch
                indices = test_indices[bno*batch_size:(bno + 1)*batch_size]

                # Get the data 
                images, labels = x_test[indices], y_test[indices]

                # Get the predicitions and loss for this batch
                loss_value, y_pred = validate_step(model, images, labels)

                # Record batch loss and accuracy
                test_loss(loss_value)
                test_accuracy(labels, y_pred)


            # get training and validataion stats
            # after one epoch is completed 
            loss = train_loss.result()
            acc =  train_accuracy.result()
            val_loss = test_loss.result()
            val_acc = test_accuracy.result()

            improved = pt.check_update(val_loss)
            # check if performance of model has imporved or not
            if improved:
                print("Saving model checkpoint.")
                checkpoint.save(checkpoint_prefix)
            
            history.update([loss, acc], [val_loss, val_acc])
            # print loss values and accuracy values for each epoch 
            # for both training as well as validation sets
            print(f"""Epoch: {epoch+1} 
                    train_loss: {loss:.6f}  train_acc: {acc*100:.2f}%  
                    test_loss:  {val_loss:.6f}  test_acc:  {val_acc*100:.2f}%\n""")

            history.plot_and_save(initial_epoch=starting_epoch)

            train_loss.reset_states() 
            train_accuracy.reset_states()
            test_loss.reset_states() 
            test_accuracy.reset_states()

        scores.append(pt.loss)
        clear_session()
    
    min_score = min(scores)
    with open(f"saved/{dataname}_result.json", "w+") as jf:
        myd = {
            'dataname': dataname,
            'score': float(min_score),
            'fold': scores.index(min_score)
        }
        json.dump(myd, jf)
        jf.close()

    print("scores : ", scores)