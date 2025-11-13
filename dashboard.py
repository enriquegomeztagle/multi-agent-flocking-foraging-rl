import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


st.set_page_config(
    page_title="Flocking & Foraging Dashboard", page_icon="🐦", layout="wide"
)

st.title("🐦 Multi-Agent Flocking & Foraging Dashboard")
st.markdown(
    "Visualización de resultados y métricas del sistema de aprendizaje por refuerzo multiagente"
)

st.sidebar.header("📁 Selección de Archivos")

results_dir = Path("results")
if not results_dir.exists():
    st.error("❌ Directorio 'results/' no encontrado")
    st.stop()

json_files = list(results_dir.glob("*.json"))
if not json_files:
    st.error("❌ No se encontraron archivos de resultados JSON")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Seleccionar archivo de resultados", options=[f.name for f in json_files], index=0
)


@st.cache_data
def load_results(file_path):
    """Load results from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


results = load_results(results_dir / selected_file)

if "summary" in results:
    file_type = "Baseline Boids"
    st.info(f"📊 Tipo: {file_type} | Episodios: {results.get('n_episodes', 0)}")
else:
    file_type = "RL Agent"
    config = results.get("configuration", "unknown")
    st.info(
        f"🤖 Tipo: {file_type} ({config}) | Episodios: {results.get('n_episodes', 0)}"
    )

st.header(f"📊 Análisis: {selected_file}")

summary = None
if "summary" in results:
    summary = results["summary"]
elif "statistics" in results and "all_episodes" in results:
    stats = results["statistics"]
    episodes = results["all_episodes"]

    gini_values = [ep.get("gini", 0) for ep in episodes]
    mean_gini = sum(gini_values) / len(gini_values) if gini_values else 0
    std_gini = (
        (sum((g - mean_gini) ** 2 for g in gini_values) / len(gini_values)) ** 0.5
        if len(gini_values) > 1
        else 0
    )

    summary = {
        "mean_efficiency_percent": stats.get("mean_efficiency", 0),
        "std_efficiency_percent": (
            stats.get("std", 0) / stats.get("mean", 1) * 100
            if stats.get("mean", 0) > 0
            else 0
        ),
        "mean_intake": stats.get("mean", 0),
        "std_intake": stats.get("std", 0),
        "mean_gini": mean_gini,
        "std_gini": std_gini,
    }

if summary:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Eficiencia Media", f"{summary.get('mean_efficiency_percent', 0):.2f}%"
        )
        st.caption(
            f"📊 Desv. estándar: ±{summary.get('std_efficiency_percent', 0):.2f}%"
        )

    with col2:
        mean_intake = summary.get("mean_intake", 0)
        theoretical_max = results.get("theoretical_max", 800)
        st.metric("Intake Medio", f"{mean_intake:.1f} / {theoretical_max}")
        st.caption(f"📊 Desv. estándar: ±{summary.get('std_intake', 0):.1f}")

    with col3:
        st.metric("Gini Medio", f"{summary.get('mean_gini', 0):.4f}")
        st.caption(f"📊 Desv. estándar: ±{summary.get('std_gini', 0):.4f}")

    with col4:
        if "mean_polarization" in summary:
            st.metric("Polarización", f"{summary.get('mean_polarization', 0):.4f}")
            st.caption(f"📊 Desv. estándar: ±{summary.get('std_polarization', 0):.4f}")
        else:
            episodes_count = len(
                results.get("episodes", results.get("all_episodes", []))
            )
            st.metric("Episodios", episodes_count)
            st.caption("📈 Total evaluado")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Distribuciones", "📉 Tendencias", "🎯 Comparación", "📋 Detalles"]
)

episodes_data = None
if "episodes" in results:
    episodes_data = results["episodes"]
elif "all_episodes" in results:
    episodes_data = results["all_episodes"]

with tab1:
    st.subheader("Distribución de Métricas")

    if episodes_data and len(episodes_data) > 0:
        episodes_df = pd.DataFrame(episodes_data)

        col1, col2 = st.columns(2)

        with col1:
            fig_eff = px.histogram(
                episodes_df,
                x="efficiency_percent",
                nbins=30,
                title="Distribución de Eficiencia",
                labels={"efficiency_percent": "Eficiencia (%)", "count": "Frecuencia"},
                color_discrete_sequence=["#1f77b4"],
            )
            fig_eff.add_vline(
                x=episodes_df["efficiency_percent"].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Media: {episodes_df['efficiency_percent'].mean():.2f}%",
            )
            st.plotly_chart(fig_eff, use_container_width=True)

        with col2:
            fig_gini = px.histogram(
                episodes_df,
                x="gini",
                nbins=30,
                title="Distribución de Gini (Fairness)",
                labels={"gini": "Coeficiente de Gini", "count": "Frecuencia"},
                color_discrete_sequence=["#ff7f0e"],
            )
            fig_gini.add_vline(
                x=episodes_df["gini"].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Media: {episodes_df['gini'].mean():.4f}",
            )
            st.plotly_chart(fig_gini, use_container_width=True)

        fig_intake = px.histogram(
            episodes_df,
            x="intake",
            nbins=30,
            title="Distribución de Intake",
            labels={"intake": "Intake Total", "count": "Frecuencia"},
            color_discrete_sequence=["#2ca02c"],
        )
        fig_intake.add_vline(
            x=episodes_df["intake"].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Media: {episodes_df['intake'].mean():.2f}",
        )
        st.plotly_chart(fig_intake, use_container_width=True)

with tab2:
    st.subheader("Tendencias por Episodio")

    if episodes_data and len(episodes_data) > 0:
        episodes_df = pd.DataFrame(episodes_data)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=episodes_df["episode"],
                y=episodes_df["efficiency_percent"],
                mode="lines+markers",
                name="Eficiencia",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=4),
            )
        )
        fig.add_hline(
            y=episodes_df["efficiency_percent"].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Media: {episodes_df['efficiency_percent'].mean():.2f}%",
        )
        fig.update_layout(
            title="Eficiencia por Episodio",
            xaxis_title="Episodio",
            yaxis_title="Eficiencia (%)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=episodes_df["episode"],
                y=episodes_df["gini"],
                mode="lines+markers",
                name="Gini",
                line=dict(color="#ff7f0e", width=2),
                marker=dict(size=4),
            )
        )
        fig2.add_hline(
            y=episodes_df["gini"].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Media: {episodes_df['gini'].mean():.4f}",
        )
        fig2.update_layout(
            title="Coeficiente de Gini por Episodio",
            xaxis_title="Episodio",
            yaxis_title="Gini",
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Comparación de Métricas")

    if episodes_data and len(episodes_data) > 0:
        episodes_df = pd.DataFrame(episodes_data)

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("Eficiencia", "Gini", "Intake"),
            horizontal_spacing=0.1,
        )

        fig.add_trace(
            go.Box(
                y=episodes_df["efficiency_percent"], name="Eficiencia", boxmean="sd"
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Box(y=episodes_df["gini"], name="Gini", boxmean="sd"), row=1, col=2
        )

        fig.add_trace(
            go.Box(y=episodes_df["intake"], name="Intake", boxmean="sd"), row=1, col=3
        )

        fig.update_layout(
            title="Distribución de Métricas (Box Plots)", height=400, showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Detalles por Episodio")

    if episodes_data and len(episodes_data) > 0:
        episodes_df = pd.DataFrame(episodes_data)

        display_cols = [
            "episode",
            "efficiency_percent",
            "intake",
            "gini",
            "reward",
            "steps",
        ]
        if "polarization" in episodes_df.columns:
            display_cols.append("polarization")
        if "mean_neighbor_distance" in episodes_df.columns:
            display_cols.append("mean_neighbor_distance")

        available_cols = [col for col in display_cols if col in episodes_df.columns]

        st.dataframe(
            episodes_df[available_cols].style.format(
                {
                    "efficiency_percent": "{:.2f}%",
                    "intake": "{:.2f}",
                    "gini": "{:.4f}",
                    "reward": "{:.2f}",
                    "polarization": "{:.4f}",
                    "mean_neighbor_distance": "{:.2f}",
                }
            ),
            use_container_width=True,
            height=400,
        )

        csv = episodes_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar CSV",
            data=csv,
            file_name=f"{selected_file.replace('.json', '')}.csv",
            mime="text/csv",
        )

# Footer
st.markdown("---")
st.markdown("**Multi-Agent Flocking & Foraging RL** | Dashboard generado con Streamlit")
