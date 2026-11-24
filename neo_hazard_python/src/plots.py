"""Figure builders: interactive Plotly figures for the app, static PNGs for the report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend: lets us save PNGs without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from src import RANDOM_STATE
from src.data import TARGET_COLUMN

CLASS_LABELS = ["Safe", "Hazardous"]
CLASS_COLORS = {"Hazardous": "#ef233c", "Safe": "#3a86ff"}  # red / blue
MODEL_COLORS = {"Random Forest": "#2a9d8f", "XGBoost": "#e76f51"}


def _marker_sizes(diameter: pd.Series, d_min: float, d_max: float) -> np.ndarray:
    """Map diameter to marker size (px) on a LOG scale with a minimum size.

    Diameters span 0.5 m to 7 km. A linear scale shrinks the many small (mostly
    safe) asteroids to invisible dots, so log10 is used and sizes run from 4 to 22 px.
    """
    lo, hi = np.log10(d_min), np.log10(d_max)
    scaled = (np.log10(diameter.clip(d_min, d_max)) - lo) / max(hi - lo, 1e-9)
    return 4 + 18 * scaled.to_numpy()


def risk_matrix_figure(df: pd.DataFrame, max_safe: int | None = None) -> go.Figure:
    """Velocity vs miss distance scatter; size = diameter (log scale), Red = Hazardous, Blue = Safe.

    Every object is plotted by default (WebGL keeps it fast). Safe points are drawn
    first and semi-transparent so the hazardous points on top stay readable.
    ``max_safe`` optionally samples the safe class; all hazardous points are always kept.
    """
    hazardous = df[df[TARGET_COLUMN]]
    safe = df[~df[TARGET_COLUMN]]
    if max_safe is not None and len(safe) > max_safe:
        safe = safe.sample(max_safe, random_state=RANDOM_STATE)
    d_min, d_max = df["estimated_diameter"].min(), df["estimated_diameter"].max()
    fig = go.Figure()
    for label, part, opacity in [("Safe", safe, 0.35), ("Hazardous", hazardous, 0.8)]:
        fig.add_trace(
            go.Scattergl(
                x=part["relative_velocity"],
                y=part["miss_distance"],
                mode="markers",
                name=f"{label} ({len(part):,})",
                marker={
                    "color": CLASS_COLORS[label],
                    "size": _marker_sizes(part["estimated_diameter"], d_min, d_max),
                    "opacity": opacity,
                    "line": {"width": 0},
                },
                customdata=part[["estimated_diameter", "absolute_magnitude"]].to_numpy(),
                hovertemplate=(
                    f"<b>{label}</b><br>Velocity: %{{x:.2f}} km/s<br>"
                    "Miss distance: %{y:,.0f} km<br>Diameter: %{customdata[0]:.3f} km<br>"
                    "Absolute magnitude H: %{customdata[1]:.2f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Risk Matrix: velocity vs miss distance (marker size = diameter, log scale)",
        xaxis_title="Relative velocity (km/s)",
        yaxis_title="Miss distance (km, log scale)",
        yaxis_type="log",
        legend_title_text="Hazard status (count)",
        height=550,
    )
    return fig


def confusion_matrix_figure(cm: np.ndarray, title: str, zmax: float) -> go.Figure:
    """Annotated confusion-matrix heatmap. Pass the same ``zmax`` to both models."""
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=[f"Predicted {c}" for c in CLASS_LABELS],
            y=[f"Actual {c}" for c in CLASS_LABELS],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 18},
            colorscale="Blues",
            zmin=0,
            zmax=zmax,
        )
    )
    fig.update_layout(title=title, yaxis_autorange="reversed", height=380)
    return fig


def roc_curve_figure(results: dict[str, dict[str, Any]]) -> go.Figure:
    """Overlapping ROC curves with AUC in the legend and a random-guess diagonal."""
    fig = go.Figure()
    for name, res in results.items():
        fig.add_trace(
            go.Scatter(
                x=res["fpr"], y=res["tpr"], mode="lines",
                name=f"{name} (AUC = {res['roc_auc']:.3f})",
                line={"color": MODEL_COLORS.get(name), "width": 2.5},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random guess (AUC = 0.5)",
            line={"color": "gray", "dash": "dash"},
        )
    )
    fig.update_layout(
        title="ROC Curves: Random Forest vs XGBoost",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate (recall)",
        height=480,
    )
    return fig


def feature_importance_figure(
    importances: dict[str, pd.DataFrame], show_rf: bool = False
) -> go.Figure:
    """Horizontal bar chart of XGBoost importances (largest on top), optionally with RF."""
    names = ["XGBoost", "Random Forest"] if show_rf else ["XGBoost"]
    order = importances["XGBoost"].sort_values("importance")["feature"].tolist()
    fig = go.Figure()
    for name in names:
        imp = importances[name].set_index("feature").loc[order]
        fig.add_trace(
            go.Bar(
                x=imp["importance"], y=order, orientation="h", name=name,
                marker_color=MODEL_COLORS[name],
            )
        )
    fig.update_layout(
        title="Feature Importance", xaxis_title="Importance", barmode="group", height=420
    )
    return fig


def threshold_figure(analysis: dict[str, pd.DataFrame]) -> go.Figure:
    """Recall and precision of each model as the decision threshold moves."""
    fig = go.Figure()
    for name, df in analysis.items():
        color = MODEL_COLORS[name]
        fig.add_trace(go.Scatter(x=df["threshold"], y=df["recall"], name=f"{name} recall",
                                 line={"color": color}))
        fig.add_trace(go.Scatter(x=df["threshold"], y=df["precision"], name=f"{name} precision",
                                 line={"color": color, "dash": "dot"}))
    fig.update_layout(
        title="Recall / precision vs decision threshold",
        xaxis_title="Threshold", yaxis_title="Score", height=420,
    )
    return fig


def save_static_figures(pipeline_output: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Export confusion matrices, ROC curves and feature importance as PNGs (matplotlib/seaborn)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = pipeline_output["results"]
    saved: list[Path] = []

    zmax = max(r["confusion_matrix"].max() for r in results.values())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, res) in zip(axes, results.items()):
        sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues", vmin=0,
                    vmax=zmax, xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax,
                    cbar=False, annot_kws={"size": 14})
        ax.set_title(f"{name}\nFN = {res['false_negatives']} "
                     f"(FNR = {res['false_negative_rate']:.1%})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    saved.append(out_dir / "confusion_matrices.png")
    fig.savefig(saved[-1], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, res in results.items():
        ax.plot(res["fpr"], res["tpr"], lw=2, color=MODEL_COLORS[name],
                label=f"{name} (AUC = {res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    saved.append(out_dir / "roc_curves.png")
    fig.savefig(saved[-1], dpi=150)
    plt.close(fig)

    imp = pd.concat(
        [df.assign(model=name) for name, df in pipeline_output["importances"].items()]
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    order = pipeline_output["importances"]["XGBoost"]["feature"].tolist()
    sns.barplot(data=imp, y="feature", x="importance", hue="model", order=order,
                palette=MODEL_COLORS, ax=ax)
    ax.set_title("Feature Importance")
    fig.tight_layout()
    saved.append(out_dir / "feature_importance.png")
    fig.savefig(saved[-1], dpi=150)
    plt.close(fig)
    return saved
