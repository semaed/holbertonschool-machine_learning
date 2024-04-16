import numpy as np
import GPyOpt
import GPy
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Load dataset
(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255
x_val = x_val.reshape(-1, 784).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_val = keras.utils.to_categorical(y_val, 10)

# Function to create and compile a model


def create_model(learning_rate, num_units, dropout_rate, l2_weight, batch_size):
    model = Sequential([
        Dense(int(num_units), activation='relu', input_shape=(
            784,), kernel_regularizer=l2(l2_weight)),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Objective function


def model_training(params):
    learning_rate, num_units, dropout_rate, l2_weight, batch_size = params[0]
    batch_size = int(batch_size)
    model = create_model(learning_rate, num_units,
                         dropout_rate, l2_weight, batch_size)

    checkpoint_filename = f"model_lr={learning_rate}_units={num_units}_dropout={dropout_rate}_l2={l2_weight}_batch={batch_size}.h5"
    checkpoint = ModelCheckpoint(
        checkpoint_filename, save_best_only=True, monitor='val_loss', mode='min')
    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(x_train, y_train, validation_data=(
        x_val, y_val), epochs=50, batch_size=batch_size, callbacks=[checkpoint, early_stop], verbose=0)
    validation_loss = np.min(history.history['val_loss'])
    return validation_loss


# Domain for the hyperparameters
domain = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.01)},
    {'name': 'num_units', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
    {'name': 'l2_weight', 'type': 'continuous', 'domain': (0.0001, 0.01)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128, 256)}
]

# Bayesian optimization
optimizer = GPyOpt.methods.BayesianOptimization(
    f=model_training, domain=domain, max_iter=30)

# Run optimization
optimizer.run_optimization()

# Plot convergence
optimizer.plot_convergence()

# Save report
with open('bayes_opt.txt', 'w') as f:
    f.write(f"Optimized Parameters: {optimizer.x_opt}\n")
    f.write(f"Minimum Validation Loss: {optimizer.fx_opt}\n")
