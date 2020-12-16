import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical

def get_mean_std(imageset):
    channel_wise_mean = []
    channel_wise_std = []
    for i in range(3):
        temp = []
        for img in imageset:
            ch = img[:,:,i]
            temp.append(ch)
        temp = np.asarray(temp)
        channel_wise_mean.append(np.mean(temp))
        channel_wise_std.append(np.std(temp))
    
    return channel_wise_mean, channel_wise_std

def normalize(image, xbar, sigma):
    """Normalize input image channel-wise to zero mean and unit variance."""
    image = image.transpose(2, 0, 1)  # Switch to channel-first
    mean, std = np.array(xbar), np.array(sigma)
    image = (image - mean[:, None, None]) / std[:, None, None]
    return image.transpose(1, 2, 0)

def make_imageset(pathset):
    resz = (224, 224)
    imageset = [np.array(Image.open(path).resize(resz, resample = Image.BILINEAR)) for path in pathset]
    return np.asarray(imageset)

def get_imgset_lblset(dataname = 'testimg'):
    cwd = "data/train"
    classes = os.listdir(os.path.join(cwd, dataname))
    num_classes = len(classes)
    clsidx = [i for i in range(num_classes)]
    labelcnt = [0 for i in range(num_classes)]
    pathset = []
    for i, cls in enumerate(classes):
        target_f = os.path.join(cwd, dataname, cls)
        imgs = os.listdir(target_f)
        for img in imgs:
            pathset.append(os.path.join(target_f, img))
            labelcnt[i] += 1
    
    imageset = make_imageset(pathset)
    xbar, sigma = get_mean_std(imageset)
    imageset_norm = np.asarray([normalize(image, xbar, sigma) for image in imageset])
    imageset_norm = imageset_norm.astype(np.float32) / 255.
    
    labelset = np.repeat(clsidx, labelcnt, axis = 0)
    labelset = np.expand_dims(labelset, axis = 1)
    labelset = to_categorical(labelset, num_classes=num_classes).astype(np.float32)
    
    return imageset_norm, labelset

