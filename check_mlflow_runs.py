# check_mlflow_runs.py
"""Quick diagnostic to verify MLflow logged your training run correctly."""

import mlflow


def check_experiment_runs() -> None:
    """Print all runs in the difficulty-adaptation experiment."""

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Get the experiment
    experiment = mlflow.get_experiment_by_name("difficulty-adaptation")

    if experiment is None:
        print("❌ Experiment 'difficulty-adaptation' not found!")
        print("   Did train.py set the experiment name correctly?")
        return

    print(f"✅ Experiment found: {experiment.name}")
    print(f"   Experiment ID : {experiment.experiment_id}")
    print(f"   Artifact root : {experiment.artifact_location}")
    print()

    # Fetch all runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )

    if runs.empty:
        print("⚠️  No runs found in this experiment yet.")
        print("   Check that train.py uses:")
        print("   mlflow.set_experiment('difficulty-adaptation')")
        print("   and that mlflow.start_run() wraps your training code.")
        return

    print(f"✅ Found {len(runs)} run(s):\n")

    for _, run in runs.iterrows():
        print(f"  Run ID   : {run['run_id']}")
        print(f"  Status   : {run['status']}")
        print(f"  Started  : {run['start_time']}")

        # Print all logged metrics
        metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
        print("  Metrics  :")
        for col in metric_cols:
            val = run[col]
            if val is not None and str(val) != "nan":
                print(f"    {col.replace('metrics.', '')}: {val:.4f}")

        # Print all logged params
        param_cols = [c for c in runs.columns if c.startswith("params.")]
        print("  Params   :")
        for col in param_cols:
            val = run[col]
            if val is not None and str(val) != "nan":
                print(f"    {col.replace('params.', '')}: {val}")

        print()


if __name__ == "__main__":
    check_experiment_runs()