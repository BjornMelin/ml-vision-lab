import streamlit as st
import mlflow
from pathlib import Path
import plotly.express as px
import pandas as pd


def load_experiment_metrics(experiment_name: str):
    """Load metrics from MLflow experiment.

    Args:
        experiment_name: Name of MLflow experiment
    """
    mlflow.set_tracking_uri("file://../../experiments/runs")
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        st.warning("No experiment found. Start training first!")
        return

    # Get all runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
    )

    if runs.empty:
        st.warning("No runs found in experiment.")
        return

    return runs


def plot_run_metrics(run_df: pd.DataFrame, metric_name: str):
    """Plot metrics for selected run.

    Args:
        run_df: DataFrame containing run metrics
        metric_name: Name of metric to plot
    """
    fig = px.line(
        run_df,
        x="step",
        y=metric_name,
        title=f"Training Progress - {metric_name}",
        labels={"step": "Step", metric_name: "Value"},
    )
    st.plotly_chart(fig)


def main():
    st.title("Food Classification Dashboard")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Training Monitor", "Model Predictions"])

    if page == "Training Monitor":
        st.header("Training Progress")

        # Load experiment data
        runs_df = load_experiment_metrics("food-classification")

        if runs_df is not None:
            # Select run
            run_id = st.selectbox(
                "Select Run",
                runs_df["run_id"].tolist(),
                format_func=lambda x: (
                    f"{runs_df[runs_df['run_id']==x]['start_time'].iloc[0]} "
                    f"({x[:8]})"
                ),
            )

            run = runs_df[runs_df["run_id"] == run_id].iloc[0]

            # Display run info
            st.subheader("Run Information")
            st.write(f"Start Time: {run['start_time']}")
            st.write(f"Status: {run['status']}")

            # Load metrics history
            client = mlflow.tracking.MlflowClient()
            metrics_hist = client.get_metric_history(run_id, "train/accuracy")

            if metrics_hist:
                metrics_df = pd.DataFrame(
                    [{"step": m.step, "train/accuracy": m.value} for m in metrics_hist]
                )

                # Plot metrics
                plot_run_metrics(metrics_df, "train/accuracy")

            # Display parameters
            st.subheader("Parameters")
            params = {k: v for k, v in run.items() if k.startswith("params.")}
            st.json(params)

    else:  # Model Predictions page
        st.header("Model Predictions")
        st.info("👈 Use the Predict page in the sidebar for image classification!")


if __name__ == "__main__":
    main()
