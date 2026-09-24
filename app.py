"""Streamlit entry point for the House Price Lab."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.house_price_lab.data import load_dataset, split_dataset
from src.house_price_lab.experiments import load_experiments, save_experiment
from src.house_price_lab.modeling import (
    MODEL_FACTORIES,
    load_model,
    save_model,
    train_and_evaluate,
)

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"

st.set_page_config(
    page_title="House Price Lab",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');
    :root { --ink: #17202a; --muted: #637083; --accent: #e76f51; --cream: #fff8ef; --line: #eadfd2; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
    .stApp { background: linear-gradient(135deg, #fff8ef 0%, #f7f3ee 48%, #eaf2f1 100%); }
    [data-testid="stSidebar"] { background: #17202a; }
    [data-testid="stSidebar"] * { color: #f8f1e8; }
    .hero { padding: 1.8rem 0 1rem; border-bottom: 1px solid var(--line); margin-bottom: 1.2rem; }
    .eyebrow { color: var(--accent); font-weight: 700; letter-spacing: .08em; text-transform: uppercase; font-size: .75rem; }
    .hero h1 { font-size: clamp(2.4rem, 5vw, 4.7rem); line-height: .98; margin: .35rem 0 .8rem; max-width: 780px; }
    .hero p { color: var(--muted); font-size: 1.05rem; max-width: 720px; }
    .metric-card { background: rgba(255,255,255,.75); border: 1px solid var(--line); padding: 1rem; border-radius: 8px; min-height: 105px; }
    .metric-label { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; }
    .metric-value { color: var(--ink); font-family: 'Space Grotesk'; font-size: 1.8rem; font-weight: 700; margin-top: .4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data

def dataset():
    return load_dataset()


@st.cache_data

def split_data():
    return split_dataset(dataset())


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def render_overview(data) -> None:
    st.subheader("A practical look at the data")
    columns = st.columns(4)
    with columns[0]:
        metric_card("Homes", f"{len(data.features):,}")
    with columns[1]:
        metric_card("Signals", str(len(data.feature_names)))
    with columns[2]:
        metric_card("Target unit", data.target_unit)
    with columns[3]:
        metric_card("Median target", f"${data.target.median() * 100:,.0f}k")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### What the model sees")
        st.dataframe(data.features.head(8), use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Target distribution")
        chart = px.histogram(
            data.target,
            nbins=45,
            labels={"value": "Median house value ($100k)"},
            color_discrete_sequence=["#e76f51"],
        )
        chart.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(chart, use_container_width=True)


def train_models(selected_models: list[str]) -> pd.DataFrame:
    x_train, x_test, y_train, y_test = split_data()
    rows = []
    for model_name in selected_models:
        model, evaluation, errors = train_and_evaluate(
            model_name, x_train, x_test, y_train, y_test
        )
        save_experiment(
            evaluation,
            settings={"test_size": 0.2, "random_state": 42},
        )
        rows.append(
            {
                "model": model_name,
                "MAE": evaluation.mae,
                "RMSE": evaluation.rmse,
                "R2": evaluation.r2,
                "pipeline": model,
                "errors": errors,
            }
        )
    results = pd.DataFrame(rows).sort_values("R2", ascending=False)
    if not results.empty:
        save_model(results.iloc[0]["pipeline"], MODEL_PATH)
        st.session_state["results"] = results
    return results


def render_training() -> None:
    st.subheader("Compare the candidates")
    selected = st.multiselect(
        "Models to train",
        list(MODEL_FACTORIES),
        default=list(MODEL_FACTORIES),
    )
    st.caption("Every run uses the same deterministic 80/20 split so comparisons stay honest.")
    if st.button("Run experiment", type="primary", disabled=not selected):
        with st.spinner("Training and evaluating models..."):
            results = train_models(selected)
        st.success(f"Saved {len(results)} experiment results.")

    results = st.session_state.get("results")
    if results is None:
        st.info("Choose at least one model and run an experiment to populate this view.")
        return
    display = results[["model", "MAE", "RMSE", "R2"]].copy()
    st.dataframe(
        display.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )
    chart = px.bar(
        display,
        x="model",
        y="R2",
        color="R2",
        color_continuous_scale=["#f4a261", "#2a9d8f"],
        labels={"R2": "R² score", "model": ""},
    )
    chart.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
    st.plotly_chart(chart, use_container_width=True)


def render_prediction(data) -> None:
    st.subheader("Estimate a property value")
    st.caption("Values use the same feature definitions as the California Housing dataset.")
    inputs = {}
    columns = st.columns(2)
    for index, feature in enumerate(data.feature_names):
        with columns[index % 2]:
            inputs[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=float(data.features[feature].median()),
                min_value=float(data.features[feature].min()),
                max_value=float(data.features[feature].max()),
                help=f"Typical range: {data.features[feature].min():.2f} to {data.features[feature].max():.2f}",
            )
    if st.button("Predict price", type="primary"):
        try:
            model = load_model(MODEL_PATH)
        except FileNotFoundError:
            st.warning("Run an experiment first so the lab has a trained model.")
            return
        prediction = float(model.predict(pd.DataFrame([inputs]))[0])
        st.success(f"Estimated median value: ${prediction * 100:,.0f}k")
        st.caption("This is a model estimate, not a property appraisal or financial advice.")


def render_explainability(data) -> None:
    st.subheader("Why does the model behave this way?")
    results = st.session_state.get("results")
    if results is None:
        st.info("Run an experiment first to unlock model explanations.")
        return
    best = results.iloc[0]
    model = best["pipeline"].named_steps["regressor"]
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        importance = abs(model.coef_)
    explanation = pd.DataFrame(
        {"feature": data.feature_names, "importance": importance}
    ).sort_values("importance", ascending=True)
    chart = px.bar(
        explanation,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#f4a261", "#264653"],
    )
    chart.update_layout(height=390, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
    st.plotly_chart(chart, use_container_width=True)
    st.caption("Importance shows model association, not causation.")


def render_errors() -> None:
    st.subheader("Where does the model miss?")
    results = st.session_state.get("results")
    if results is None:
        st.info("Run an experiment first to inspect errors.")
        return
    errors = results.iloc[0]["errors"].copy()
    left, right = st.columns(2)
    with left:
        chart = px.scatter(errors, x="actual", y="predicted", labels={"actual": "Actual", "predicted": "Predicted"})
        chart.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(chart, use_container_width=True)
    with right:
        chart = px.histogram(errors, x="residual", nbins=40, color_discrete_sequence=["#2a9d8f"], labels={"residual": "Residual"})
        chart.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(chart, use_container_width=True)
    st.markdown("#### Largest misses")
    st.dataframe(errors.assign(abs_error=errors.residual.abs()).sort_values("abs_error", ascending=False).head(10), use_container_width=True)


def render_history() -> None:
    records = load_experiments()
    st.subheader("Experiment journal")
    if not records:
        st.info("Your saved experiments will appear here.")
        return
    rows = []
    for record in records:
        evaluation = record["evaluation"]
        rows.append(
            {
                "saved": record["saved_at"][:19].replace("T", " "),
                "model": evaluation["model_name"],
                "MAE": evaluation["mae"],
                "RMSE": evaluation["rmse"],
                "R2": evaluation["r2"],
            }
        )
    st.dataframe(
        pd.DataFrame(rows).style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )


data = dataset()
st.markdown(
    '<div class="hero"><div class="eyebrow">House Price Lab / regression studio</div>'
    '<h1>Turn housing signals into a defensible estimate.</h1>'
    '<p>Compare models, inspect their mistakes, and understand which signals move a prediction. Built for experimentation, not blind certainty.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Lab controls")
    view = st.radio(
        "Navigate",
        ["Overview", "Train & compare", "Predict", "Explain", "Error analysis", "Experiment history"],
    )
    st.divider()
    st.caption("Dataset: California Housing")
    st.caption("Split: 80% train / 20% test")
    st.caption("Seed: 42")

if view == "Overview":
    render_overview(data)
elif view == "Train & compare":
    render_training()
elif view == "Predict":
    render_prediction(data)
elif view == "Explain":
    render_explainability(data)
elif view == "Error analysis":
    render_errors()
else:
    render_history()
