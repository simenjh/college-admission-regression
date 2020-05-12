from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np



def preprocess(dataset):
    X = dataset[:, 1:-1]
    y = dataset[:, -1].reshape(-1, 1)
    return X, y


def read_dataset(dataset_file):
    dataset = pd.read_csv(dataset_file)
    return dataset.values


def standardize(X1, *args):
    sc = StandardScaler()
    Xs_standard = sc.fit_transform(X1)
    if args != ():
        Xs_standard = [Xs_standard]
        Xs_standard.extend([sc.transform(Xi) for Xi in args])
    return Xs_standard
    

