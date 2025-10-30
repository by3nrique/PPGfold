import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, BatchNormalization
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import KFold
from umap import UMAP

def customUMAP(S,Xtr,Ytr,Xval,Yval,p=0.05,n_neighbors=10,n_components=3,min_dist=0.1,metric='euclidean'):
    umap_instance = UMAP(n_neighbors=n_neighbors,min_dist=min_dist,n_components=n_components,metric=metric,random_state=42)

    if S==1:
        Ytr_umapS = Ytr.copy().astype(np.int8)
        p = 0.05 # Percentage of hidden labels
        num_to_modify = round(len(Ytr_umapS) * p)
        indices_to_modify = random.sample(range(len(Ytr_umapS)), num_to_modify)
        Ytr_umapS[indices_to_modify] = -1
        umap_instance.fit(Xtr,y=Ytr_umapS)

    if S==0:
        umap_instance.fit(Xtr)

    return umap_instance

def hypersearch_UMAP(S,Xtr,Ytr,Xval,Yval,n_components,n_neighbors,min_dist,p,metric):
    best_error = float('inf')
    best_model = None
    best_config = {}

    parameter_combinations = product(n_neighbors, min_dist)
    total_combinations = len(n_neighbors) * len(min_dist)

    for params in tqdm(parameter_combinations, total=total_combinations, desc='Parameter Combinations'):
        n_neighbors, min_dist = params

        print(
            f"Training for n_neighbors: {n_neighbors}, n_components: {n_components}, min_dist: {min_dist}")
        model = customUMAP(S,Xtr,Ytr,Xval,Yval,p=p,n_neighbors=n_neighbors,n_components=n_components,min_dist=min_dist,metric=metric)
        Hval = model.transform(Xval)

        #error = silhouette_score(Hval, Yval)
        if len(np.unique(Yval)) == 1:
            print(f"Only one class, skipping search")
            return model, {'n_neighbors': n_neighbors, 'n_components': n_components, 'min_dist': min_dist}
        davies_bouldin = davies_bouldin_score(Hval,Yval)
        #calinski_harabasz = calinski_harabasz_score(Yval, Hval)
        print(f"Davies-Bouldin Index: {davies_bouldin}")
        if davies_bouldin < best_error:
            best_error = davies_bouldin
            best_model = model
            best_config = {'n_neighbors': n_neighbors,
                           'n_components': n_components, 'min_dist': min_dist}

    print(
        f"Best Configuration: {best_config}, Error: {best_error}")
    return best_model, best_config

def hypersearch_UMAP_CV(S,Xtr,Ytr,Xval,Yval,n_components, n_neighbors, min_dist, p, n_splits=5,metric='euclidean'):
    best_error = float('inf')
    best_model = None
    best_config = {}

    parameter_combinations = list(product(n_neighbors, min_dist))
    total_combinations = len(parameter_combinations)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    errors = []

    X = np.concatenate((Xtr, Xval), axis=0)
    Y = np.concatenate((Ytr, Yval), axis=0)
    for params in tqdm(parameter_combinations, total=total_combinations, desc='Parameter Combinations'):
        n_neighbors, min_dist = params
        fold_errors = []
        fold_errors_train = []

        print(f"Training for n_neighbors: {n_neighbors}, n_components: {n_components}, min_dist: {min_dist}")
        for train_index, val_index in kf.split(X):
            KXtr, KXval = X[train_index], X[val_index]
            KYtr, KYval = Y[train_index], Y[val_index]


            model = customUMAP(S, KXtr, KYtr, KXval, KYval, p=p, n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist,metric=metric)
            Hval = model.transform(Xval)
            
            if len(np.unique(Yval)) == 1:
                print(f"Only one class, skipping search")
                return model, {'n_neighbors': n_neighbors, 'n_components': n_components, 'min_dist': min_dist},[]
            
            davies_bouldin = davies_bouldin_score(Hval, Yval)
            davies_bouldin_train = davies_bouldin_score(model.transform(Xtr), Ytr)
            fold_errors.append(davies_bouldin)
            fold_errors_train.append(davies_bouldin_train)
        
        avg_error = sum(fold_errors) / n_splits
        print(f"Avg Davies-Bouldin Index Val: {avg_error}")
        print(f"Avg Davies-Bouldin Index Train: {sum(fold_errors_train) / n_splits}")
        errors.append([avg_error, sum(fold_errors_train) / n_splits, n_neighbors, min_dist])
        if avg_error < best_error:
            best_error = avg_error
            best_model = model
            best_config = {'n_neighbors': n_neighbors, 'n_components': n_components, 'min_dist': min_dist}

    best_model = customUMAP(S, Xtr, Ytr, Xval, Yval, p=p, n_neighbors=best_config['n_neighbors'], n_components=best_config['n_components'], min_dist=best_config['min_dist'])
    print(f"Best Configuration: {best_config}, Error: {best_error}")
    return best_model, best_config,errors

def autoencoder_model(input_shape, architecture,activation):
    encoder_layers = [InputLayer(input_shape=(input_shape,))]
    encoder_layers.extend([Dense(units, activation=activation)
                          for units in architecture[0]])
    encoder_layers.extend([Dense(architecture[1], activation=activation)])
    encoder = Sequential(encoder_layers)

    decoder_layers = [Dense(units, activation=activation)
                      for units in architecture[2]]
    decoder_layers.append(Dense(input_shape, activation=None))
    decoder = Sequential(decoder_layers)

    autoencoder = Sequential([encoder, decoder])

    return autoencoder, encoder, decoder

def grid_search_UMAP(architecture_options, learning_rates,activation_functions, train_data,val_data,epochs=100,patience=10):
    # architecture_options = [[[], 2, []], [[], 3, []], [[], 5, []], [[], 10, []], [[], 20, []], [[10], 2, [10]], [[10], 3, [10]], [
    #    [10], 5, [10]], [[15], 10, [15]], [[30], 20, [30]], [[20], 2, [20]], [[20], 3, [20]], [[50], 2, [50]], [[50], 3, [50]]]
    # learning_rates = [0.001, 0.0001, 0.00001, 0.000001]

    best_validation_error = float('inf')
    best_model = None
    best_config = {}

    parameter_combinations = product(
        architecture_options, learning_rates,activation_functions)
    total_combinations = len(architecture_options) * \
        len(learning_rates)  * len(activation_functions)

    for params in tqdm(parameter_combinations, total=total_combinations, desc='Parameter Combinations'):
        architecture, lr, activation = params

        print(
            f"Training for architecture: {architecture}, lr: {lr}, activation: {activation}")
        history, model = train_and_evaluate_parametricUMAP(
                                                        train_data=train_data,val_data=val_data,architecture=architecture,activation=activation, lr=lr, epochs=epochs,patience=patience)
        validation_error = min(history['val_loss'])
        print(f"Validation Error: {validation_error}")
        if validation_error < best_validation_error:
            best_validation_error = validation_error
            best_model = model
            best_config = {'architecture': architecture,
                           'lr': lr, 'activation': activation}

    print(
        f"Best Configuration: {best_config}, Validation Error: {best_validation_error}")
    return best_model, best_config

def plot_autoencoder_representation(name,embedder,train_data, test_data, train_target, val_target):
    # Get the encoded representations for the training and validation data

    encoded_train_data = embedder.transform(train_data)
    encoded_test_data = embedder.transform(test_data)
    represented_train_data = embedder.inverse_transform(train_data)
    represented_test_data = embedder.inverse_transform(test_data)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_data[1, :], alpha=0.7, label='Original Train')
    plt.plot(represented_train_data[1, :],
             alpha=0.7, label='Representation Train')
    plt.title('Training Data Comparison')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_data[1, :], alpha=0.7, label='Original Validation')
    plt.plot(represented_test_data[1, :],
             alpha=0.7, label='Representation Validation')
    plt.title('Validation Data Comparison')
    plt.legend()

    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), subplot_kw={
                             'projection': '3d'} if encoded_train_data.shape[1] == 3 else {})

    # Plot for training data
    if encoded_train_data.shape[1] == 3:
        scatter1 = axes[0].scatter3D(encoded_train_data[:, 0], encoded_train_data[:, 1],
                                     encoded_train_data[:, 2], c=train_target, cmap='viridis')
        axes[0].set_title(f'Training Data 3D Representation')
        axes[0].set_xlabel('Dimension 1')
        axes[0].set_ylabel('Dimension 2')
        axes[0].set_zlabel('Dimension 3')
    else:
        scatter1 = axes[0].scatter(
            encoded_train_data[:, 0], encoded_train_data[:, 1], c=train_target, cmap='viridis')
        axes[0].set_title(f'Training Data 2D Representation')
        axes[0].set_xlabel('Dimension 1')
        axes[0].set_ylabel('Dimension 2')
    legend1 = axes[0].legend(*scatter1.legend_elements(), title='Video Class')

    # Plot for validation data
    if encoded_test_data.shape[1] == 3:
        scatter2 = axes[1].scatter3D(encoded_test_data[:, 0], encoded_test_data[:, 1],
                                     encoded_test_data[:, 2], c=val_target, cmap='viridis')
        axes[1].set_title(f'Test Data 3D Representation')
        axes[1].set_xlabel('Dimension 1')
        axes[1].set_ylabel('Dimension 2')
        axes[1].set_zlabel('Dimension 3')
    else:
        scatter2 = axes[1].scatter(
            encoded_test_data[:, 0], encoded_test_data[:, 1], c=val_target, cmap='viridis')
        axes[1].set_title(f'Test Data 2D Representation')
        axes[1].set_xlabel('Dimension 1')
        axes[1].set_ylabel('Dimension 2')
    legend2 = axes[1].legend(*scatter2.legend_elements(), title='Video Class')

    # Adjust the layout to make room for the suptitle
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(name)
    plt.show()
    return encoded_train_data, encoded_test_data


