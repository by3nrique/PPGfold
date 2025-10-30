import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold

def autoencoder_model(input_shape, architecture,activation='relu'):
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

def supervised_autoencoder_model(input_shape, architecture, n_classes=2, activation='relu'):
    encoder_layers = [InputLayer(input_shape=(input_shape,))]
    encoder_layers.extend([Dense(units, activation=activation)
                          for units in architecture[0]])
    encoder_layers.extend([Dense(architecture[1], activation=activation)])
    encoder = Sequential(encoder_layers)

    decoder_layers = [Dense(units, activation=activation)
                      for units in architecture[2]]
    decoder_layers.append(Dense(input_shape, activation=activation))
    if n_classes == 2:
        decoder_layers.append(Dense(1, activation="sigmoid"))
    else:
        decoder_layers.append(Dense(n_classes, activation="softmax"))

    decoder = Sequential(decoder_layers)

    autoencoder = Sequential([encoder, decoder])

    return autoencoder, encoder, decoder

def train_and_evaluate_autoencoder(S, train_data, train_target, val_data, val_target, architecture, lr=0.0001,activation='relu',epochs=200, batch_size=128, patience=10000, plot_history=False, verbose=0):

    input_shape = train_data.shape[1]

    if S == 0 or S == 'AE':
        autoencoder, _, _ = autoencoder_model(input_shape, architecture,activation=activation)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        autoencoder.compile(optimizer=optimizer, loss='mse', metrics=["mae"])
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True)

    elif S == 1 or S == 'FCNN':
        n_classes = train_target.shape[1] if len(train_target.shape) > 1 else 2
        autoencoder, _, _ = supervised_autoencoder_model(
            input_shape, architecture, n_classes=n_classes, activation=activation)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        if n_classes == 2:
            autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[
                                tf.metrics.Precision(), tf.metrics.Recall()])
        else:
            autoencoder.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[
                                tf.metrics.Precision(), tf.metrics.Recall()])
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True)



    history = autoencoder.fit(train_data, train_target, epochs=epochs, batch_size=batch_size,
                              validation_data=(val_data, val_target), callbacks=[early_stopping], verbose=verbose)

    if plot_history:
        plt.figure(figsize=(20, 6))  # Adjust the figure size as needed

        # First subplot for Loss
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')

        # Second subplot for Metrics
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
        for key in history.history.keys():
            # To avoid repeating loss plots, check if 'loss' is in the key
            if 'loss' not in key:  # This ensures only metrics are plotted here
                plt.plot(history.history[key], label=key)
        plt.title('Training and Validation Metrics')
        plt.ylabel('Metrics')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')

        plt.show()  # Show the figure with both subplots

    return history.history, autoencoder

def grid_search_NN(type, architecture_options, learning_rates,activation_functions,batch_sizes, train_data, train_target, val_data, val_target,epochs=500):
    # architecture_options = [[[], 2, []], [[], 3, []], [[], 5, []], [[], 10, []], [[], 20, []], [[10], 2, [10]], [[10], 3, [10]], [
    #    [10], 5, [10]], [[15], 10, [15]], [[30], 20, [30]], [[20], 2, [20]], [[20], 3, [20]], [[50], 2, [50]], [[50], 3, [50]]]
    # learning_rates = [0.001, 0.0001, 0.00001, 0.000001]
    # batch_sizes = [64, 128, 256, 512, 1024]
    # batch_sizes = [64]

    best_validation_error = float('inf')
    best_model = None
    best_config = {}

    parameter_combinations = product(
        architecture_options, learning_rates, batch_sizes,activation_functions)
    total_combinations = len(architecture_options) * \
        len(learning_rates) * len(batch_sizes) * len(activation_functions)

    for params in tqdm(parameter_combinations, total=total_combinations, desc='Parameter Combinations'):
        architecture, lr, batch_size,activation = params

        print(
            f"Training for architecture: {architecture}, lr: {lr}, batch_size: {batch_size}, activation: {activation} epochs: {epochs}")
        history, model = train_and_evaluate_autoencoder(S=type,
                                                        train_data=train_data, train_target=train_target, val_data=val_data, val_target=val_target, architecture=architecture, lr=lr,activation=activation,epochs=epochs, batch_size=batch_size)
        validation_error = min(history['val_loss'])
        print(f"Validation Error: {validation_error}")
        if validation_error < best_validation_error:
            best_validation_error = validation_error
            best_model = model
            best_config = {'architecture': architecture,
                           'lr': lr, 'batch_size': batch_size, 'activation':activation}

    print(
        f"Best Configuration: {best_config}, Validation Error: {best_validation_error}")
    return best_model, best_config

def grid_search_CV_NN_singlecore(type, architecture_options, learning_rates,activation_functions,batch_sizes, train_data, train_target, val_data, val_target,epochs=500,n_splits=5):
    best_validation_error = float('inf')
    best_model = None
    best_config = {}

    parameter_combinations = list(product(architecture_options, learning_rates, batch_sizes, activation_functions))
    total_combinations = len(parameter_combinations)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    data = np.concatenate((train_data, val_data), axis=0)
    target = np.concatenate((train_target, val_target), axis=0)


    errors = []

    for params in tqdm(parameter_combinations, total=total_combinations, desc='Parameter Combinations'):
        architecture, lr, batch_size, activation = params
        fold_errors = []
        fold_errors_train = []

        print(f"Training for architecture: {architecture}, lr: {lr}, batch_size: {batch_size}, activation: {activation}, epochs: {epochs}")
        for train_index, val_index in kf.split(data):
            train_data, val_data = data[train_index], data[val_index]
            train_target, val_target = target[train_index], target[val_index]

            history, model = train_and_evaluate_autoencoder(S=type,
                                                            train_data=train_data, train_target=train_target, val_data=val_data, val_target=val_target, architecture=architecture, lr=lr, activation=activation, epochs=epochs, batch_size=batch_size)
            validation_error = min(history['val_loss'])
            training_error = min(history['loss'])
            fold_errors.append(validation_error)
            fold_errors_train.append(training_error)
        
        avg_error = sum(fold_errors) / n_splits
        avg_error_train = sum(fold_errors_train) / n_splits
        print(f"Avg Validation Error: {avg_error}")
        errors.append([avg_error, avg_error_train, architecture, lr, batch_size, activation])

        if avg_error < best_validation_error:
            best_validation_error = avg_error
            best_model = model
            best_config = {'architecture': architecture, 'lr': lr, 'batch_size': batch_size, 'activation': activation}

    # Retrain the best model on the entire dataset
    best_model = train_and_evaluate_autoencoder(S=type,
                                                train_data=train_data, train_target=train_target, val_data=val_data, val_target=val_target, architecture=best_config['architecture'], lr=best_config['lr'], activation=best_config['activation'], epochs=epochs, batch_size=best_config['batch_size'])[1]
    print(f"Best Configuration: {best_config}, Validation Error: {best_validation_error}")
    return best_model, best_config, errors

def grid_search_CV_NN(type, architecture_options, learning_rates,activation_functions,batch_sizes, train_data, train_target, val_data, val_target,epochs=500,n_splits=5):
    return grid_search_CV_NN_singlecore(type, architecture_options, learning_rates,activation_functions,batch_sizes, train_data, train_target, val_data, val_target,epochs=epochs,n_splits=n_splits)

def plot_autoencoder_representation(name, autoencoder, train_data, test_data, train_target, val_target):
    # Get the encoded representations for the training and validation data

    encoder = autoencoder.layers[0]
    encoded_train_data = encoder.predict(train_data)
    encoded_test_data = encoder.predict(test_data)
    represented_train_data = autoencoder.predict(train_data)
    represented_test_data = autoencoder.predict(test_data)

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

def grid_search_CV(S, architecture_options, learning_rates,activation_functions,batch_sizes, train_data, train_target, val_data, val_target,epochs=500,n_splits=5):
    if S==0:
        return grid_search_CV_AE(architecture_options, learning_rates,activation_functions,batch_sizes, train_data, train_target, val_data, val_target,epochs,n_splits)
    elif S==1:
        return grid_search_CV_FCNN(architecture_options, learning_rates,activation_functions,batch_sizes, train_data, train_target, val_data, val_target,epochs,n_splits)

def grid_search(S, architecture_options, learning_rates,activation_functions,batch_sizes, train_data, train_target, val_data, val_target,epochs=500,patience=100):
    if S==0:
        return grid_search_AE(architecture_options, learning_rates,activation_functions,batch_sizes, train_data, train_target, val_data, val_target,epochs=epochs)
    elif S==1:
        return grid_search_FCNN(architecture_options, learning_rates,activation_functions,batch_sizes, train_data, train_target, val_data, val_target,epochs=epochs)

def grid_search_FCNN(architecture_options, learning_rates,activation_functions,batch_sizes, train_data, train_target, val_data, val_target,epochs=500):
    return grid_search_NN('FCNN', architecture_options, learning_rates,activation_functions,batch_sizes, train_data, train_target, val_data, val_target,epochs=epochs)

def grid_search_AE(architecture_options, learning_rates,activation_functions,batch_sizes, train_data, train_target, val_data, val_target,epochs=500):
    return grid_search_NN('AE', architecture_options, learning_rates,activation_functions,batch_sizes, train_data, train_target, val_data, val_target,epochs=epochs)

def grid_search_CV_FCNN(architecture_options, learning_rates,activation_functions,batch_sizes, train_data, train_target, val_data, val_target,epochs=500,n_splits=5):
    return grid_search_CV_NN('FCNN', architecture_options, learning_rates,activation_functions,batch_sizes, train_data, train_target, val_data, val_target,epochs=epochs,n_splits=n_splits)

def grid_search_CV_AE(architecture_options, learning_rates,activation_functions,batch_sizes, train_data, train_target, val_data, val_target,epochs=500,n_splits=5):
    return grid_search_CV_NN('AE', architecture_options, learning_rates,activation_functions,batch_sizes, train_data, train_target, val_data, val_target,epochs=epochs,n_splits=n_splits)