import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.activations import relu, sigmoid, tanh
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load and split the data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Define the model
def create_model(activation_func, optimizer):
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation=activation_func),
        Dropout(0.5),
        Dense(32, activation=activation_func),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Train and evaluate the models
activation_funcs = [relu, sigmoid, tanh]
optimizer_classes = [SGD, Adam, RMSprop]
results = []

for activation_func in activation_funcs:
    for optimizer_class in optimizer_classes:
        optimizer = optimizer_class(learning_rate=0.001)
        model = create_model(activation_func, optimizer)
        print(f"\nTraining model with activation function {activation_func.__name__} and optimizer {optimizer.__class__.__name__}...")
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        results.append((activation_func.__name__, optimizer.__class__.__name__, loss, accuracy))
        print(f"Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}")

print("\nSummary of Results:")
for activation, optimizer, loss, acc in results:
    print(f"Activation: {activation}, Optimizer: {optimizer}, Test Loss: {loss:.3f}, Test Accuracy: {acc:.3f}")