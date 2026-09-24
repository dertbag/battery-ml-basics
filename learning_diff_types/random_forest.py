import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATA_DIR = "data"

BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

TEST_BATTERIES = ["B0007", "B0018"]


# --------------------------------------------------
# LOAD BATTERY
# --------------------------------------------------

def load_battery(filename):

    path = os.path.join(DATA_DIR, filename + ".mat")

    mat = scipy.io.loadmat(path)

    battery = mat[filename][0, 0]

    return battery["cycle"][0]


# --------------------------------------------------
# EXTRACT FEATURES
# --------------------------------------------------

def extract_features(cycles):

    features = []
    capacities = []

    for cycle in cycles:

        if str(cycle["type"][0]) != "discharge":
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

    soh = capacity / capacity[0]

    if battery in TEST_BATTERIES:

        X_test.append(X)
        y_test.append(soh)

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
# TRAIN RANDOM FOREST
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

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
    y_test,
    y_pred
)

percent_error = (
    np.abs(y_pred - y_test)
    / y_test
    * 100
)

within_5 = np.mean(
    percent_error <= 5
) * 100

within_10 = np.mean(
    percent_error <= 10
) * 100


print("\nRESULTS")
print("-------------------------")
print(f"RMSE:        {rmse:.4f}")
print(f"MAE:         {mae:.4f}")
print(f"Within 5%:   {within_5:.2f}%")
print(f"Within 10%:  {within_10:.2f}%")


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

feature_names = [
    "Mean Voltage",
    "Mean Temperature",
    "Discharge Duration"
]

print("\nFEATURE IMPORTANCE")
print("-------------------------")

for name, importance in zip(
    feature_names,
    model.feature_importances_
):

    print(
        f"{name:20s}: {importance:.4f}"
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
    "Random Forest: Actual vs Predicted SOH"
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
    "Random Forest: SOH Over Battery Life"
)

plt.legend()
plt.grid(True)

plt.show()