import subprocess
import sys

scripts = [
    "linear_regression.py",
    "logistic_regression.py",
    "decision_tree.py",
    "random_forest.py",
    "gradient_boost.py",
    "xg_boost.py",
    "svr.py",
    "mlp.py",
    "cnn.py",
    "lstm.py",
    "tcn.py",
    "cnn_lstm.py",
    "tcm_lstm.py",
    "transformer.py",
]

for script in scripts:
    print("\n" + "=" * 60)
    print(f"RUNNING: {script}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, f"src/{script}"],
        text=True
    )

    if result.returncode != 0:
        print(f"\n❌ {script} FAILED")
        print("Stopping.")
        break

    print(f"\n✅ {script} finished")

print("\n" + "=" * 60)
print("ALL MODELS FINISHED")
print("=" * 60)