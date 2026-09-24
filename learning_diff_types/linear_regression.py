import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATA_DIR = "data"

BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

# Batteries used for testing
TEST_BATTERIES = ["B0007", "B0018"]


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
# EXTRACT DISCHARGE FEATURES
# --------------------------------------------------

def extract_features(cycles):

    features = []
    capacities = []

    for cycle in cycles:

        cycle_type = str(cycle["type"][0])

        # We only want discharge cycles
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

        # Basic features
        mean_voltage = np.mean(voltage)
        min_voltage = np.min(voltage)

        mean_temperature = np.mean(temperature)

        duration = time[-1] - time[0]

        # Capacity is NOT used as a model input
        features.append([
            mean_voltage,
            min_voltage,
            mean_temperature,
            duration
        ])

        # Capacity is kept only to calculate SOH
        capacities.append(capacity)

    return np.array(features), np.array(capacities)


# --------------------------------------------------
# BUILD DATASET
# --------------------------------------------------

X_train = []
y_train = []

X_test = []
y_test = []

battery_labels_test = []

for battery in BATTERIES:

    cycles = load_battery(battery)

    X, capacity = extract_features(cycles)

    # SOH
    initial_capacity = capacity[0]
    soh = capacity / initial_capacity

    if battery in TEST_BATTERIES:

        X_test.append(X)
        y_test.append(soh)

        battery_labels_test.extend(
            [battery] * len(soh)
        )

    else:

        X_train.append(X)
        y_train.append(soh)


X_train = np.vstack(X_train)
y_train = np.concatenate(y_train)

X_test = np.vstack(X_test)
y_test = np.concatenate(y_test)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# TRAIN LINEAR REGRESSION
# --------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# METRICS
# --------------------------------------------------

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

mae = mean_absolute_error(
    y_test, y_pred
)

percent_error = (
    np.abs(y_pred - y_test)
    / y_test
    * 100
)

within_5 = np.mean(percent_error <= 5) * 100
within_10 = np.mean(percent_error <= 10) * 100


print("\nRESULTS")
print("-------------------------")
print(f"RMSE:        {rmse:.4f}")
print(f"MAE:         {mae:.4f}")
print(f"Within 5%:   {within_5:.2f}%")
print(f"Within 10%:  {within_10:.2f}%")


# --------------------------------------------------
# MODEL COEFFICIENTS
# --------------------------------------------------

feature_names = [
    "Mean Voltage",
    "Minimum Voltage",
    "Mean Temperature",
    "Discharge Duration"
]

print("\nMODEL COEFFICIENTS")
print("-------------------------")

for name, coefficient in zip(
    feature_names,
    model.coef_
):

    print(
        f"{name:20s}: {coefficient:.6f}"
    )

print(
    f"\nIntercept: {model.intercept_:.6f}"
)


# --------------------------------------------------
# ACTUAL VS PREDICTED
# --------------------------------------------------

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual SOH")
plt.ylabel("Predicted SOH")

plt.title(
    "Linear Regression: Actual vs Predicted SOH"
)

plt.grid(True)

plt.show()


# --------------------------------------------------
# SOH OVER BATTERY LIFE
# --------------------------------------------------

plt.figure(figsize=(9, 6))

start = 0

for battery in TEST_BATTERIES:

    cycles = load_battery(battery)

    X, capacity = extract_features(cycles)

    soh = capacity / capacity[0]

    n = len(soh)

    battery_pred = y_pred[start:start+n]

    cycle_numbers = np.arange(1, n + 1)

    plt.plot(
        cycle_numbers,
        soh,
        label=f"{battery} Actual"
    )

    plt.plot(
        cycle_numbers,
        battery_pred,
        linestyle="--",
        label=f"{battery} Predicted"
    )

    start += n


plt.xlabel("Discharge Cycle")
plt.ylabel("SOH")

plt.title(
    "Linear Regression: SOH Over Battery Life"
)

plt.legend()
plt.grid(True)

plt.show()