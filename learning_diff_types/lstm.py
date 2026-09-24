import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


DATA_DIR = "data"
BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
TEST_BATTERIES = ["B0007", "B0018"]

N_POINTS = 100


def load_battery(filename):
    path = os.path.join(DATA_DIR, filename + ".mat")
    mat = scipy.io.loadmat(path)
    battery = mat[filename][0, 0]
    return battery["cycle"][0]


def extract_curves(cycles):
    X = []
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

        # Resample each discharge to 100 points
        t_new = np.linspace(
            time[0],
            time[-1],
            N_POINTS
        )

        voltage_interp = np.interp(
            t_new,
            time,
            voltage
        )

        temperature_interp = np.interp(
            t_new,
            time,
            temperature
        )

        curve = np.column_stack([
            voltage_interp,
            temperature_interp
        ])

        X.append(curve)
        capacities.append(capacity)

    return np.array(X), np.array(capacities)


# --------------------------------------------------
# BUILD DATASET
# --------------------------------------------------

X_train = []
y_train = []

X_test = []
y_test = []

for battery in BATTERIES:

    cycles = load_battery(battery)
    X, capacity = extract_curves(cycles)

    soh = capacity / capacity[0]

    if battery in TEST_BATTERIES:
        X_test.append(X)
        y_test.append(soh)
    else:
        X_train.append(X)
        y_train.append(soh)


X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)

X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Input shape:", X_train.shape)


# --------------------------------------------------
# SCALE
# --------------------------------------------------

scaler = StandardScaler()

X_train_flat = X_train.reshape(-1, 2)
X_test_flat = X_test.reshape(-1, 2)

X_train_scaled = scaler.fit_transform(
    X_train_flat
).reshape(X_train.shape)

X_test_scaled = scaler.transform(
    X_test_flat
).reshape(X_test.shape)


# --------------------------------------------------
# LSTM
# --------------------------------------------------

model = Sequential([

    LSTM(
        64,
        input_shape=(N_POINTS, 2),
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(
        32,
        activation="relu"
    ),

    Dense(1)
])


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

y_pred = model.predict(
    X_test_scaled
).flatten()


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
plt.title("LSTM: Actual vs Predicted SOH")

plt.grid(True)
plt.show()


# --------------------------------------------------
# SOH OVER BATTERY LIFE
# --------------------------------------------------

plt.figure(figsize=(9, 6))

start = 0

for battery in TEST_BATTERIES:

    cycles = load_battery(battery)
    X, capacity = extract_curves(cycles)

    soh = capacity / capacity[0]

    n = len(soh)

    battery_pred = y_pred[
        start:start+n
    ]

    cycle_numbers = np.arange(
        1,
        n + 1
    )

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
plt.title("LSTM: SOH Over Battery Life")

plt.legend()
plt.grid(True)
plt.show()


# --------------------------------------------------
# TRAINING HISTORY
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("LSTM Training History")

plt.legend()
plt.grid(True)

plt.show()