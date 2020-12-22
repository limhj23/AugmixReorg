from execute import trainer, predictor
from tensorflow.keras.backend import clear_session
exprs = [True]
datanames = ['inscape','dworld','hanrim','implant','xray']
# datanames = []
for dataname in datanames:
    for expr in exprs:
        trainer.main(dataname, expr)
        predictor.main(dataname, expr)
        clear_session()