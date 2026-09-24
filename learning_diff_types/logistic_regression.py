import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATA_DIR = "data"

BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

TEST_BATTERIES = ["B0007", "B0018"]

# Classification question:
# Will SOH fall below 90%?
SOH_THRESHOLD = 0.90


# --------------------------------------------------
# LOAD ONE BATTERY
# --------------------------------------------------

def load_battery(filename):

    path = os.path.join(DATA_DIR, filename + ".mat")

    mat = scipy.io.loadmat(path)

    battery = mat[filename][0, 0]

    cycles = battery["cycle"][0]

    return cycles


# --------------------------------------------------
# EXTRACT FEATURES
# --------------------------------------------------

def extract_features(cycles):

    features = []
    capacities = []

    for cycle in cycles:

        cycle_type = str(cycle["type"][0])

        if cycle_type != "discharge":
            continue

        data = cycle["data"][0, 0]

        voltage = np.asarray(
            data["Voltage_measured"][0]
        ).flatten()

        temperature = np.asarray(
            data["Temperature_measured"][0]
        ).flatten()

        time = np.asarray(
            data["Time"][0]
        ).flatten()

        capacity = float(
            np.asarray(data["Capacity"]).flatten()[0]
        )

        mean_voltage = np.mean(voltage)
        mean_temperature = np.mean(temperature)
        duration = time[-1] - time[0]

        features.append([
            mean_voltage,
            mean_temperature,
            duration
        ])

        capacities.append(capacity)

    return np.array(features), np.array(capacities)


# --------------------------------------------------
# BUILD DATASET
# --------------------------------------------------

X_train = []
y_train = []

X_test = []
y_test = []


for battery in BATTERIES:

    cycles = load_battery(battery)

    X, capacity = extract_features(cycles)

    # Calculate SOH
    soh = capacity / capacity[0]

    # Classification target
    # 0 = SOH >= threshold
    # 1 = SOH < threshold
    labels = (soh < SOH_THRESHOLD).astype(int)

    if battery in TEST_BATTERIES:

        X_test.append(X)
        y_test.append(labels)

    else:

        X_train.append(X)
        y_train.append(labels)


X_train = np.vstack(X_train)
y_train = np.concatenate(y_train)

X_test = np.vstack(X_test)
y_test = np.concatenate(y_test)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# TRAIN LOGISTIC REGRESSION
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\nRESULTS")
print("-------------------------")

print(f"Accuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# MODEL COEFFICIENTS
# --------------------------------------------------

feature_names = [
    "Mean Voltage",
    "Mean Temperature",
    "Discharge Duration"
]

print("\nMODEL COEFFICIENTS")
print("-------------------------")

for name, coefficient in zip(
    feature_names,
    model.coef_[0]
):

    print(
        f"{name:20s}: {coefficient:.6f}"
    )

print(
    f"\nIntercept: {model.intercept_[0]:.6f}"
)


# --------------------------------------------------
# PLOT PREDICTED PROBABILITY
# --------------------------------------------------

plt.figure(figsize=(9, 6))

start = 0

for battery in TEST_BATTERIES:

    cycles = load_battery(battery)

    X, capacity = extract_features(cycles)

    soh = capacity / capacity[0]

    n = len(soh)

    battery_prob = probabilities[start:start+n]

    cycle_numbers = np.arange(1, n + 1)

    plt.plot(
        cycle_numbers,
        battery_prob,
        label=f"{battery}"
    )

    start += n


plt.axhline(
    0.5,
    linestyle="--",
    label="Classification threshold"
)

plt.xlabel("Discharge Cycle")
plt.ylabel("Probability SOH < 90%")

plt.title(
    "Logistic Regression: Probability of SOH < 90%"
)

plt.legend()
plt.grid(True)

plt.show()