import os
import sys
import json
import tensorflow as tf
from tensorflow import keras as K
import numpy as np
import preprocess.models as models
from preprocess.dataset import get_imgset_lblset

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

MODEL_DIR = "checkpoints"
IMAGE_SIZE = (224, 224)
SAVED = ""

def load_train_result(dataname):
    if os.path.exists(os.path.join(SAVED, f"{dataname}_result.json")):
        with open(os.path.join(SAVED, f"{dataname}_result.json"), "r") as jf:
            train_result = json.load(jf)
            jf.close()
        
        max_fold = train_result['fold']
        num_classes = train_result['num_classes']
        print(f'Successfully read train result from fold {max_fold}!')
    else:
        # absolute path for model if dbg
        dataname = "fruit"
        max_fold = "0"
        num_classes = 2
    
    return (max_fold, num_classes)

def decode_pred(val):
    probs = val
    ind = np.argmax(probs)
    return ind, probs[ind]

def get_metrics(confmat):
    sum_axis0 = np.sum(confmat, axis=0)
    sum_axis1 = np.sum(confmat, axis=1)
    total_sum =  np.sum(confmat)
    diagonal = confmat.diagonal()
    tpsum = np.sum(diagonal)

    tpvec = diagonal
    fpvec = sum_axis0 - tpvec
    fnvec = sum_axis1 - tpvec
    tnvec = total_sum - tpvec - fpvec - fnvec
    tpfpfntn_vec = np.stack((tpvec,fpvec,fnvec,tnvec), axis=0)

    tpfpsum_vec = (tpvec+fpvec)
    tpfnsum_vec = (tpvec+fnvec)

    def zero_checking(vec):
        zero_where = np.where(vec==0)[0]
        if len(zero_where) > 0:
            idx = zero_where
            zero_flag = True
        else:
            idx = None
            zero_flag = False
        
        return zero_flag, idx

    precision_vec = np.true_divide(tpvec,tpfpsum_vec,where=(tpfpsum_vec!=0))
    zflag1, zidx1 = zero_checking(tpfpsum_vec)
    if zflag1:
        precision_vec[zidx1] = np.nan

    recall_vec = np.true_divide(tpvec,tpfnsum_vec,where=(tpfnsum_vec!=0))
    zflag2, zidx2 = zero_checking(tpfnsum_vec)
    if zflag2:
        recall_vec[zidx2] = np.nan
    
    if np.sum(np.isnan(recall_vec))>0:
        mean_recall = np.nanmean(recall_vec)
        recall_vec[np.isnan(recall_vec)] = 0.0
    else:
        mean_recall = np.sum(recall_vec)/len(recall_vec)
    
    if np.sum(np.isnan(precision_vec))>0:
        mean_prec = np.nanmean(precision_vec)
        precision_vec[np.isnan(precision_vec)] = 0.0
    else:
        mean_prec = np.sum(precision_vec)/len(precision_vec)
    
    acc = tpsum / total_sum

    return acc, mean_recall, mean_prec

def main(dataname, expr):
    global SAVED
    if expr:
        SAVED = "saved/exp"
    else:
        SAVED = "saved/default"

    (max_fold, num_classes) = load_train_result(dataname)
    # dataname = "inscape"
    # max_fold = 1
    # num_classes = 2
    save_path = os.path.join(MODEL_DIR, f"{dataname}_fold_{max_fold}")

    model = models.get_resnet50(num_classes)
    optim = K.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer= optim,
        loss= K.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0
        ),
        metrics=['accuracy']
    )
    checkpoint = tf.train.Checkpoint(optimizer=optim, model=model)
    latest = tf.train.latest_checkpoint(save_path)
    status = checkpoint.restore(latest).expect_partial()
    print("status", status)

    X, Y = get_imgset_lblset(dataname, mode = "test")
    
    preds = model.predict(X)
    len_preds = preds.shape[0]
    confmat = np.zeros(shape=(num_classes, num_classes), dtype=np.int32)
    for j, pred in enumerate(preds):
        if j > 0 and j % 50 == 0:
            print(f'{j}/{len_preds} is complete')
            sys.stdout.write("\033[F")
        pred_cls, prob = decode_pred(pred)
        true_cls = Y[j].tolist().index(1.0)
        np.add.at(confmat, (true_cls, pred_cls), 1)
    print("\nInference Complete")

    acc, mean_recall, mean_prec = get_metrics(confmat)
    print(f"confmat: \n {confmat}")
    print(f"acc: {acc}")
    print(f"mean_recall: {mean_recall}")
    print(f"mean_prec: {mean_prec}")
    with open(f"{SAVED}\\{dataname}_prediction_exp.txt", "w+") as txtf:
        txtf.write(f"confmat: \n {confmat}")
        txtf.write(f"\nacc: {acc}")
        txtf.write(f"\nmean_recall: {mean_recall}", )
        txtf.write(f"\nmean_prec: {mean_prec}", )
        txtf.close()