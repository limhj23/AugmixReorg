import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class CTLHistory:
    def __init__(self,
                 expr,
                 filename="history.png",
                 save_dir='plots',
                ):
        
        self.history = {'train_loss':[], 
                        "train_acc":[], 
                        "val_loss":[], 
                        "val_acc":[]}
        
        if expr:
            self.save_dir = os.path.join(save_dir, "exp")
        else:
            self.save_dir = os.path.join(save_dir, "default")
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.plot_name = os.path.join(self.save_dir, filename)
    
   
  
    def update(self, train_stats, val_stats):
        train_loss, train_acc = train_stats
        val_loss, val_acc = val_stats
        
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(np.round(train_acc*100))
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(np.round(val_acc*100))
        
        
    def plot_and_save(self, initial_epoch=0):
        train_loss = self.history['train_loss']
        train_acc = self.history['train_acc']
        val_loss = self.history['val_loss']
        val_acc = self.history['val_acc']
        
        epochs = [(i+initial_epoch) for i in range(len(train_loss))]
        
        f, ax = plt.subplots(1, 2, figsize=(15,8))
        ax[0].plot(epochs, train_loss)
        ax[0].plot(epochs, val_loss)
        ax[0].set_title('loss progression')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('loss values')
        ax[0].legend(['training', 'validation'])
        
        ax[1].plot(epochs, train_acc)
        ax[1].plot(epochs, val_acc)
        ax[1].set_title('accuracy progression')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend(['training', 'validation'])
        
        plt.savefig(self.plot_name)
        plt.close()

class ProgressTracker():
    def __init__(self):
        self.loss = None
        self.delta = 0.0
    
    def check_update(self, val_loss):
        improved = False
        if self.loss is None:
            self.loss = val_loss
        else:
            if self.loss - val_loss > self.delta:
                self.loss = val_loss
                self.delta = val_loss*0.01
                improved = True
        return improved

