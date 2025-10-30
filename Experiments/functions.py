import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import os
from umap import UMAP
from sklearn.metrics import davies_bouldin_score
from umap_code import hypersearch_UMAP_CV, hypersearch_UMAP
from autoencoder_code import train_and_evaluate_autoencoder, plot_autoencoder_representation, grid_search_CV, grid_search
from matplotlib import patches as mpatches
import subprocess as sp
from IPython import get_ipython
from IPython.display import display, HTML

def intialize_tf_session():
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

def mask_unused_gpus(leave_unmasked=1):
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"

    try:
        # Helper function to process command output
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        # Get free memory values for each GPU
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for x in memory_free_info]
        
        # Sort GPUs by available memory in descending order and filter based on threshold
        sorted_gpus = sorted([(i, mem) for i, mem in enumerate(memory_free_values) if mem > ACCEPTABLE_AVAILABLE_MEMORY],
                             key=lambda x: x[1], reverse=True)
        
        # Get the indices of the top GPUs to leave unmasked
        available_gpus = [gpu[0] for gpu in sorted_gpus[:leave_unmasked]]

        if len(available_gpus) < leave_unmasked:
            raise ValueError('Found only %d usable GPUs in the system' % len(available_gpus))

        # Set the CUDA_VISIBLE_DEVICES environment variable
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, available_gpus))
        print("Set the following GPU(s) to be visible:", os.environ["CUDA_VISIBLE_DEVICES"])
        intialize_tf_session()
        print("Tensorflow session initialized")
    except Exception as e:
        print('"nvidia-smi" is probably not installed. GPUs are not masked:', e)

def plot_hidden_representation(Htr, Hts, Ytr, Yts,title,labels,save=None):
    # Check if Hts_UmapS has three dimensions
    if Hts.shape[1] == 3:
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        scatter1 = ax1.scatter(Htr[:,0], Htr[:,1], Htr[:,2], c=Ytr, cmap='tab10',s=1)
        ax1.set_title('Training set')
        
        ax2 = fig.add_subplot(122, projection='3d')
        scatter2 = ax2.scatter(Hts[:,0], Hts[:,1], Hts[:,2], c=Yts, cmap='tab10',s=1)
        ax2.set_title('Test set')
    else:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        scatter = ax[0].scatter(Htr[:,0], Htr[:,1], c=Ytr, cmap='tab10',s=1)
        ax[0].set_title('Training set')
        
        ax[1].scatter(Hts[:,0], Hts[:,1], c=Yts, cmap='tab10',s=1)
        ax[1].set_title('Test set')

    colors = plt.cm.tab10(np.linspace(0,1,len(labels)))
    legend_patches = [mpatches.Patch(color=colors[i], label=f'{labels[i]}') for i in range(len(labels))]
    # Adjusting the legend to 3D plot scenario
    if Hts.shape[1] == 3:
        ax1.legend(handles=legend_patches, title="Signal")
    else:
        ax[0].legend(handles=legend_patches, title="Signal")

    fig.suptitle(f'{title}', fontsize=16)
    plt.show()

    if save != None:
        fig.savefig('./Experiments/'+save+title.replace(" ","")+'.png')

def train_umap(folder,S,Xtr, Ytr, Xval, Yval, Xts, Yts, experiment_name,labels, n_neighbors, min_dist, n_components, n_splits,force_train,metric='euclidean',plot=False):

    train_umap = True
    if os.path.isfile(f'{folder}{experiment_name}.pkl'):
        print(f'{experiment_name}.pkl file exists')
        train_umap = False
        # Load the model
        with open(f'{folder}{experiment_name}.pkl', 'rb') as f:
            umap = pickle.load(f)

    if train_umap or force_train:
        if n_splits ==1:
            best_UMAP, best_UMAP_config = hypersearch_UMAP(
                S, Xtr, Ytr, Xval, Yval, n_components, n_neighbors, min_dist, p=None, metric=metric)
            errors = []
        else:
            best_UMAP, best_UMAP_config, errors = hypersearch_UMAP_CV(
                S, Xtr, Ytr, Xval, Yval, n_components, n_neighbors, min_dist, p=None, n_splits=n_splits, metric=metric)
        # Save the model
        with open(f'{folder}{experiment_name}.pkl', 'wb') as f:
            pickle.dump([best_UMAP, best_UMAP_config, errors], f)
    else:
        best_UMAP = umap[0]
        best_UMAP_config = umap[1]
        print(best_UMAP_config)
        errors = umap[2]

    best_umap_instance = best_UMAP
    umap_instance = best_UMAP
    if errors:
        DBIs_UMAP = [x[0] for x in errors]

        # Get minimum index DBI
        min_index = np.argmin(DBIs_UMAP)
        print('Best Output:', errors[min_index])

    Htr_umap = umap_instance.transform(Xtr)
    Hval_umap = umap_instance.transform(Xval)
    Hts_umap = umap_instance.transform(Xts)

    # Compute Davies-Bouldin index
    if np.unique(Yts).size != 1 or np.unique(Yts)[0] != 1:
        print("David-Bouldin index for UMAP (Test):", davies_bouldin_score(Hts_umap, Yts))
    if plot:
        plot_hidden_representation(Htr_umap, Hts_umap, Ytr, Yts, 'UMAP Unsupervised', labels=labels, save=experiment_name)

    return best_umap_instance, best_UMAP_config, errors, Htr_umap, Hval_umap, Hts_umap
     
def train_AE(folder,S,Xtr,Otr, Ytr, Xval,Oval, Yval, Xts, Ots, Yts,labels, experiment_name, architecture_options, activation_functions, learning_rates, batch_sizes, n_splits,epochs, force_train,patience=100,plot=False):
    train_ae = True
    if os.path.isfile(f'{folder}{experiment_name}.pkl'):
        print(f'{experiment_name}.pkl file exists')
        train_ae = False
    
    if train_ae or force_train:
        if n_splits == 1:
            # Perform hyperparameter search without cross-validation
            best_autoencoder_model, best_autoencoder_config = grid_search(
                S=S,
                architecture_options=architecture_options,
                learning_rates=learning_rates,
                activation_functions=activation_functions,
                batch_sizes=batch_sizes,
                train_data=Xtr,
                train_target=Otr,
                val_data=Xval,
                val_target=Oval,
                epochs=epochs,
                patience=patience
            )
            errors = []
        else:
            # Perform cross-validation-based hyperparameter search
            best_autoencoder_model, best_autoencoder_config, errors = grid_search_CV(
                S=S,
                architecture_options=architecture_options,
                learning_rates=learning_rates,
                activation_functions=activation_functions,
                batch_sizes=batch_sizes,
                train_data=Xtr,
                train_target=Otr,
                val_data=Xval,
                val_target=Oval,
                n_splits=n_splits,
                epochs=epochs,
            )
        # Save the trained model
        with open(f'{folder}{experiment_name}.pkl', 'wb') as f:
            pickle.dump([best_autoencoder_model, best_autoencoder_config, errors], f)
    else:
        # Load the pre-trained model
        with open(f'{folder}{experiment_name}.pkl', 'rb') as f:
            best_autoencoder_model, best_autoencoder_config, errors = pickle.load(f)
        print(best_autoencoder_config)
    
    # Transform the data using the trained autoencoder
    Htr_AE = best_autoencoder_model.layers[0].predict(Xtr)
    Hval_AE = best_autoencoder_model.layers[0].predict(Xval)
    Hts_AE = best_autoencoder_model.layers[0].predict(Xts)

    # Evaluate the reconstruction loss or any other error metric if desired
    if errors:
        reconstruction_errors = [x[0] for x in errors]
        min_index = np.argmin(reconstruction_errors)
        print('Best Output:', errors[min_index])

    # Plot the hidden representations
    if plot:
        plot_autoencoder_representation(Htr_AE, Hts_AE, Ytr, Yts, '', labels=labels, save=experiment_name)
    return best_autoencoder_model, best_autoencoder_config, errors, Htr_AE, Hval_AE, Hts_AE


