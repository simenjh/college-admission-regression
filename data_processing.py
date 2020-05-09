from sklearn import preprocessing
import pandas as pd



def preprocess(dataset):
    X = dataset[:, 1:-1]
    y = dataset[:, -1].reshape(-1, 1)
    return preprocessing.scale(X), y


def read_dataset(dataset_file):
    dataset = pd.read_csv(dataset_file)
    return dataset.values




    
