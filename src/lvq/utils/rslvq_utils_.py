import pandas as pd
import numpy as np
from src.lvq.rslvq import RSLVQ


def normalize(data):
        data = (data - data.mean(axis = 0))/data.std(axis = 0)
        return data
def grid_search_rslvq(train_data, train_labels,test_data, test_labels, N):
    """grid_search over over possible hyperparameters to choose best ones"""
    
    accuracy = []
    for i in range(1, N+1):
        model = RSLVQ(i)
        prototypes, pro_lab = model.fit(train_data, train_labels)
        results = model.predict_all(test_data, prototypes, pro_lab )
        accuracy.append(np.mean(results == np.array(test_labels).flatten()))
    return np.argmax(accuracy) + 1, accuracy

def cross_validation(model, X_n,Y_n, n, show_plot = False, return_best_split = False):
    """general test of stability of the model, finds the best split and returns if command true"""
    arr = np.array(range(len(X_n)))


    cross_val_acc = []
    train_indices = []
    test_indices = []
    
    for i in range(n):
        np.random.shuffle(arr)
        split_point = int(len(arr) * (1 / n))
        train_idx, test_idx = arr[split_point:], arr[:split_point]
        train_indices.append(train_idx)
        test_indices.append(test_idx)
        train_set = X_n[train_idx]
        train_labels = Y_n[train_idx]
        test_set = X_n[test_idx]
        test_labels = Y_n[test_idx]
        t_new = np.array(test_labels).flatten()
        prototype_i,pro_lab = model.fit(train_set, train_labels) 
        if show_plot == True:
            import matplotlib.pyplot as plt
            plt.legend(list(range(1,n+1)))
     
        predict_i = model.predict_all(test_set)
        if len(predict_i) == 0:
             print('cross_validation for n-1')
             pass
        precision_i = np.mean(predict_i == t_new) 

        cross_val_acc.append(precision_i)
    best_acc = np.argmax(np.array(cross_val_acc))
    if return_best_split == True:
        train_set = X_n[train_indices[best_acc]]
        train_labels = Y_n[train_indices[best_acc]]
        test_set = X_n[test_indices[best_acc]]
        test_labels = Y_n[test_indices[best_acc]]
        return train_set, train_labels
         
    print(f'Accuracies: {cross_val_acc}, Mean: {np.array(cross_val_acc).mean()}, Variance: {np.array(cross_val_acc).var()}')