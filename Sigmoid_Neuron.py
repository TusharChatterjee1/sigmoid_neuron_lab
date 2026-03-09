import csv
import random
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.widgets import Button


def load_csv(path):
    with open(path, "r", newline="") as file:
        file_reader = csv.reader(file)
        header = next(file_reader)
        rows = []
        for r in file_reader:
            if not r:
                continue
            row = []
            for value in r:
                number_value = float(value)
                row.append(number_value)
            rows.append(row)
    feature_data = []
    for r in rows:
        feature_data.append(r[:-1])
    label_data = []
    for r in rows:
        value = int(r[-1])
        label_data.append(value)
    feature_name = header[:-1]
    label_name = header[-1]
    return feature_data, label_data, feature_name, label_name

def get_data_list(data, indexes):
    lst = []
    for i in indexes:
        lst.append(data[i])
    return lst

def train_test_split(features, labels, test_ratio, seed_value):
    index = []
    length = len(features)
    for i in range(length):
        index.append(i)
    seed = random.Random(seed_value)
    seed.shuffle(index)
    cut = int(length * (1 - test_ratio))
    train_index = index[:cut]
    test_index = index[cut:]
    features_train = get_data_list(features, train_index)
    labels_train = get_data_list(labels, train_index)
    features_test = get_data_list(features, test_index)
    labels_test = get_data_list(labels, test_index)
    return features_train, labels_train, features_test, labels_test

def activation_function(z):
    prediction = 1/(1 + np.exp(-z))
    return prediction

def dot_product_one_vector(features_row, weights, bias):
    total = 0
    for i in range(len(features_row)):
        total += weights[i] * features_row[i]
    total += bias
    return total

def predict_one_vector(features_row, weights, bias):
    total = dot_product_one_vector(features_row, weights, bias)
    return activation_function(total)

def set_up_weights(num_features):
    weights = []
    for value in range(num_features):
        weights.append(0.0)
    return weights

def perceptron(path, learning_rate, epochs, label):
    x, y, feature_names, label_name = load_csv(path)
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_ratio=0.3, seed_value=75)

    #Beginning values for weights and bias
    num_features = len(x_train[0])
    weights = set_up_weights(num_features)
    bias = 0.0
    weight_history = []
    loss_history = []
    max_weight_change_history = []  

    for epoch in range(epochs):
        weightChangeList = []
        lossList = []
        for i in range(len(x_train)):
            weightChangeList.append([])
            features_row = x_train[i]
            true_label = y_train[i]
            pred_label = predict_one_vector(features_row, weights, bias)
            # Update each weight
            for j in range(num_features):
                change_in_weights = learning_rate * (pred_label - true_label) * features_row[j]
                weightChangeList[i].append(abs(change_in_weights))
                weights[j] = weights[j] - change_in_weights
                
            # Update bias
            change_in_bias = learning_rate * (pred_label - true_label)
            weightChangeList[i].append(abs(change_in_bias))
            bias -= change_in_bias

            weight_history.append((weights.copy(), bias))
        for k in range(len(x_train)):
            features_row = x_train[k]
            true_label = y_train[k]
            pred_label = predict_one_vector(features_row, weights, bias)
            if(true_label == 1):
                error = -1 * np.log(pred_label)
                lossList.append(error)
            elif(true_label == 0):
                error = -1 * np.log(1 - pred_label)
                lossList.append(error)

        # Compute epoch-level summaries for plotting
        epoch_bce = sum(lossList) / len(lossList)
        loss_history.append(epoch_bce)
        all_changes = []
        for change_row in weightChangeList:
            for change_value in change_row:
                all_changes.append(change_value)
        max_weight_change_history.append(max(all_changes))

    show_graph_menu(x_train, y_train, weight_history, loss_history, max_weight_change_history, num_features)

def update(frame, history, ax, line):
        weights, bias = history[frame]
        w1, w2 = weights

        if w2 != 0:
            x_vals = np.array(ax.get_xlim())
            y_vals = -(w1 * x_vals + bias) / w2
            line.set_data(x_vals, y_vals)
        elif w1 != 0:
            x_boundary = -bias / w1
            line.set_data([x_boundary, x_boundary], ax.get_ylim())
        else:
            print("No update")

        return line,

def animate_decision_boundary(X, y, history):
    features = np.array(X)
    labels = np.array(y)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot points
    label = 0
    is_Zero = []
    is_One = []
    for value in labels:
        if value == label:
            is_Zero.append(True)
            is_One.append(False)
        else:
            is_Zero.append(False)
            is_One.append(True)

    #plot the 0 points
    ax.scatter(
        features[is_Zero, 0],
        features[is_Zero, 1],
        c='red',
        label="Class 0",
        edgecolors='k'
    )
    #plot the 1 points
    ax.scatter(
        features[is_One, 0],
        features[is_One, 1],
        c='green',
        label="Class 1",
        edgecolors='k'
    )

    line, = ax.plot([], [], 'k--', lw=2)

    ax.set_xlim(features[:, 0].min() - 1, features[:, 0].max() + 1)
    ax.set_ylim(features[:, 1].min() - 1, features[:, 1].max() + 1)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Perceptron Decision Boundary (After Each Update)")
    ax.legend()

    ani = FuncAnimation(
        fig,
        update,
        frames=len(history),
        fargs=(history, ax, line),
        interval=1,
        repeat=False
    )

    plt.show()

def plot_loss_over_epochs(loss_history):
    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_history, label="BCE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.show()

def plot_weight_change_over_epochs(max_weight_change_history):
    epochs = range(1, len(max_weight_change_history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, max_weight_change_history, label="Max Weight Change")
    plt.xlabel("Epoch")
    plt.ylabel("Max Absolute Weight Change")
    plt.title("Weight Change Over Epochs")
    plt.legend()
    plt.show()

def show_graph_menu(x_train, y_train, weight_history, loss_history, max_weight_change_history, num_features):
    print("\n--- Graph Selection ---")
    print("1. Decision Boundary (animated)")
    print("2. Loss Over Epochs")
    print("3. Weight Change Over Epochs")
    print("4. Return to Main Menu")

    choice = input("\nEnter the number of the graph you want to see: ").strip()

    if choice == "1":
        if num_features == 2:
            animate_decision_boundary(x_train, y_train, weight_history)
        else:
            print("Decision boundary only available for 2 features.")
        show_graph_menu(x_train, y_train, weight_history, loss_history, max_weight_change_history, num_features)
    elif choice == "2":
        plot_loss_over_epochs(loss_history)
        show_graph_menu(x_train, y_train, weight_history, loss_history, max_weight_change_history, num_features)
    elif choice == "3":
        plot_weight_change_over_epochs(max_weight_change_history)
        show_graph_menu(x_train, y_train, weight_history, loss_history, max_weight_change_history, num_features)
    elif choice == "4":
        main()
    else:
        print("Invalid choice. Please enter 1, 2, 3, or 4.")
        show_graph_menu(x_train, y_train, weight_history, loss_history, max_weight_change_history, num_features)

def pickPath():
    print("Please select a dataset:")
    print("1. Dataset 1")
    print("2. Dataset 2")
    print("3. Dataset 3")
    print("4. Dataset 4")
    print("5. Dataset 5")
    print("6. Exit")
    choice = input("Enter the number of the dataset you want to use: ").strip()
    if choice == "1":
        return "dataset1.csv"
    elif choice == "2":
        return "dataset2.csv"
    elif choice == "3":
        return "dataset3.csv"
    elif choice == "4":
        return "dataset4.csv"
    elif choice == "5":
        return "dataset5.csv"
    elif choice == "6":
        print("Goodbye!")
        return "6"
    else:
        print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")
        return pickPath()

def main():
    path = pickPath()
    if(path != "6"):
        learning_rate = 0.01  # Update with your desired learning rate
        epochs = 1000 # Update with your desired number of epochs

        perceptron(path, learning_rate, epochs, label="label")
    
main()