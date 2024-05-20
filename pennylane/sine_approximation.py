import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Define problem parameters
num_qubits = 1
num_layers = 9
num_params_per_layer = 1
total_num_params = num_layers * num_params_per_layer
# Initialize quantum device and optimizer
dev = qml.device("default.qubit", wires=num_qubits)

opt = qml.GradientDescentOptimizer(0.001)


def f(x):
    return np.sin(x)
    #  return np.sin(x) + 0.5*np.cos(2*x) + 0.25 * np.sin(3*x)


num_training_points = 30  # Increase the number of training points
training_inputs = np.linspace(0, 10, num_training_points)  # Use np.linspace for even distribution
training_data = [(x, f(x)) for x in training_inputs]

params = np.random.rand(total_num_params)


# Define the circuit with two RY gates
@qml.qnode(dev)
def circuit(params, x):
    # for i in range(num_layers):
    #     qml.RY(params[i] * x, wires=0)
    #     qml.RY(params[i + 1] + x, wires=0)
    qml.RY(params[0] * x, wires=0)
    qml.RY(params[1] * x, wires=0)
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=0)
    qml.RY(params[4] * x, wires=0)
    qml.RY(params[5], wires=0)
    qml.RY(params[6], wires=0)
    qml.RY(params[7], wires=0)
    qml.RY(params[8], wires=0)
    # qml.RY(params[2] / x, wires=0) geht nicht weil kein int

    return qml.expval(qml.PauliZ(wires=0))


# Define the mean squared error cost function
def cost(params, x, target):
    predicted_output = circuit(params, x)
    return ((predicted_output - target) ** 2) / 2


# Train the circuit using gradient descent
print("Training the circuit...")
print(params)
for iteration in range(200):
    for training_x, training_y in training_data:
        params = opt.step(cost, params, x=training_x, target=training_y)

    if iteration % 10 == 0:
        print(f"Iteration {iteration}:")
        for training_x, training_y in training_data:
            predicted_output = circuit(params, training_x)
            error = np.abs(predicted_output - training_y)
            print(
                f"Input: {training_x}, Expected: {training_y:.4f}, Predicted: {predicted_output:.4f}, Error: {error:.4f}")

# Evaluate the trained circuit and generate plot
print("Evaluating the trained circuit...")
print(params)
x_values = np.linspace(-3 * np.pi, 6 * np.pi, 100)  # Define range for plotting
actual_ouput = f(x_values)
predicted_outputs = [circuit(params, x) for x in x_values]
plt.ylim(-2, 2)
plt.grid(True)
plt.plot(x_values, actual_ouput, label="Actual f(x)")
plt.plot(x_values, predicted_outputs, label="Predicted Sin")
plt.legend()
plt.xlabel("x")
plt.ylabel("Sin(x)")
plt.title("Actual vs. Predicted Sin")
plt.show()
