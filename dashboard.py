import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats


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

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Resumen Visual", "🧠 Learning Dynamics", "📈 Tendencias", "🎯 Mejores/Peores", "📋 Datos"]
)

episodes_data = None
if "episodes" in results:
    episodes_data = results["episodes"]
elif "all_episodes" in results:
    episodes_data = results["all_episodes"]

with tab1:
    st.subheader("Resumen de Métricas Principales")

    if episodes_data and len(episodes_data) > 0:
        episodes_df = pd.DataFrame(episodes_data)

        # Multi-metric overview with box plots
        available_metrics = []
        metric_names = []

        if "efficiency_percent" in episodes_df.columns:
            available_metrics.append("efficiency_percent")
            metric_names.append("Eficiencia (%)")
        if "gini" in episodes_df.columns:
            available_metrics.append("gini")
            metric_names.append("Gini")
        if "reward" in episodes_df.columns:
            available_metrics.append("reward")
            metric_names.append("Reward")
        if "polarization" in episodes_df.columns:
            available_metrics.append("polarization")
            metric_names.append("Polarización")

        if len(available_metrics) >= 2:
            fig = make_subplots(
                rows=1,
                cols=len(available_metrics),
                subplot_titles=metric_names,
                horizontal_spacing=0.08,
            )

            for idx, metric in enumerate(available_metrics, start=1):
                fig.add_trace(
                    go.Box(
                        y=episodes_df[metric],
                        name=metric_names[idx-1],
                        boxmean="sd",
                        marker_color=px.colors.qualitative.Plotly[idx-1]
                    ),
                    row=1,
                    col=idx,
                )

            fig.update_layout(
                title="Distribución de Métricas Clave",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # Performance tier breakdown
        col1, col2 = st.columns(2)

        with col1:
            if "performance_tiers" in results:
                tiers = results["performance_tiers"]

                # Handle both dict and int formats
                tier_data = []
                total_episodes = len(episodes_df)

                for k, v in tiers.items():
                    if isinstance(v, dict):
                        # Format: {"count": X, "percentage": Y}
                        count = v.get("count", 0)
                        percentage = v.get("percentage", 0)
                    else:
                        # Format: just an integer count
                        count = v
                        percentage = (count / total_episodes * 100) if total_episodes > 0 else 0

                    tier_data.append({
                        "Tier": k,
                        "Episodios": count,
                        "Porcentaje": percentage
                    })

                tier_df = pd.DataFrame(tier_data)

                fig_tier = px.bar(
                    tier_df,
                    x="Tier",
                    y="Episodios",
                    text="Porcentaje",
                    title="Distribución por Tier de Performance",
                    color="Tier",
                    color_discrete_map={
                        "excellent": "#2ca02c",
                        "good": "#17becf",
                        "fair": "#ff7f0e",
                        "poor": "#d62728"
                    }
                )
                fig_tier.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig_tier, use_container_width=True)
            else:
                # Create efficiency histogram with color-coded ranges
                fig_eff = go.Figure()

                bins = np.histogram_bin_edges(episodes_df["efficiency_percent"].to_numpy(), bins=30)
                hist, _ = np.histogram(episodes_df["efficiency_percent"].to_numpy(), bins=bins)

                colors = []
                for bin_center in (bins[:-1] + bins[1:]) / 2:
                    if bin_center >= 80:
                        colors.append('#2ca02c')  # green
                    elif bin_center >= 60:
                        colors.append('#17becf')  # cyan
                    elif bin_center >= 40:
                        colors.append('#ff7f0e')  # orange
                    else:
                        colors.append('#d62728')  # red

                fig_eff.add_trace(go.Bar(
                    x=(bins[:-1] + bins[1:]) / 2,
                    y=hist,
                    marker_color=colors,
                    name="Eficiencia"
                ))

                fig_eff.add_vline(
                    x=episodes_df["efficiency_percent"].mean(),
                    line_dash="dash",
                    line_color="black",
                    annotation_text=f"Media: {episodes_df['efficiency_percent'].mean():.2f}%",
                )

                fig_eff.update_layout(
                    title="Distribución de Eficiencia por Rango",
                    xaxis_title="Eficiencia (%)",
                    yaxis_title="Frecuencia",
                    showlegend=False
                )
                st.plotly_chart(fig_eff, use_container_width=True)

        with col2:
            # Scatter plot: Efficiency vs Gini
            fig_scatter = px.scatter(
                episodes_df,
                x="gini",
                y="efficiency_percent",
                color="reward" if "reward" in episodes_df.columns else None,
                size="intake" if "intake" in episodes_df.columns else None,
                title="Eficiencia vs Fairness (Gini)",
                labels={
                    "gini": "Coeficiente de Gini (Fairness)",
                    "efficiency_percent": "Eficiencia (%)",
                    "reward": "Reward",
                    "intake": "Intake"
                },
                hover_data=["episode"],
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Statistical summary
        st.subheader("Estadísticas Descriptivas")

        stats_cols = ["efficiency_percent", "gini", "intake", "reward"]
        available_stats_cols = [col for col in stats_cols if col in episodes_df.columns]

        stats_df = episodes_df[available_stats_cols].describe().T
        stats_df["cv"] = (stats_df["std"] / stats_df["mean"] * 100).round(2)
        stats_df.columns = ["Count", "Media", "Std", "Min", "25%", "50%", "75%", "Max", "CV (%)"]

        st.dataframe(
            stats_df.style.format({
                "Media": "{:.2f}",
                "Std": "{:.2f}",
                "Min": "{:.2f}",
                "25%": "{:.2f}",
                "50%": "{:.2f}",
                "75%": "{:.2f}",
                "Max": "{:.2f}",
                "CV (%)": "{:.2f}"
            }),
            width='stretch'
        )

with tab2:
    st.subheader("Dinámica de Aprendizaje y Consistencia")

    if episodes_data and len(episodes_data) > 0:
        episodes_df = pd.DataFrame(episodes_data)

        # Rolling statistics for convergence analysis
        st.markdown("### 📉 Convergencia y Estabilidad")

        windows = [5, 10, 20]
        available_windows = [w for w in windows if w < len(episodes_df)]

        if available_windows:
            col1, col2 = st.columns(2)

            with col1:
                # Rolling mean and std for efficiency
                fig_rolling = go.Figure()

                for window in available_windows:
                    rolling_mean = episodes_df["efficiency_percent"].rolling(window=window, min_periods=1).mean()
                    fig_rolling.add_trace(go.Scatter(
                        x=episodes_df["episode"],
                        y=rolling_mean,
                        mode="lines",
                        name=f"MA{window}",
                        line=dict(width=2)
                    ))

                fig_rolling.add_hline(
                    y=episodes_df["efficiency_percent"].mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Media Global"
                )

                fig_rolling.update_layout(
                    title="Media Móvil de Eficiencia (Convergencia)",
                    xaxis_title="Episodio",
                    yaxis_title="Eficiencia (%)",
                    height=400
                )
                st.plotly_chart(fig_rolling, use_container_width=True)

            with col2:
                # Rolling std for stability
                fig_std = go.Figure()

                for window in available_windows:
                    rolling_std = episodes_df["efficiency_percent"].rolling(window=window, min_periods=2).std()
                    fig_std.add_trace(go.Scatter(
                        x=episodes_df["episode"],
                        y=rolling_std,
                        mode="lines",
                        name=f"Std{window}",
                        line=dict(width=2)
                    ))

                fig_std.update_layout(
                    title="Volatilidad de la Política (Std Móvil)",
                    xaxis_title="Episodio",
                    yaxis_title="Desviación Estándar",
                    height=400
                )
                st.plotly_chart(fig_std, use_container_width=True)

        # Episode-to-episode improvement
        st.markdown("### 📈 Mejora Episode-to-Episode")

        episodes_df["eff_change"] = episodes_df["efficiency_percent"].diff()
        episodes_df["eff_pct_change"] = episodes_df["efficiency_percent"].pct_change() * 100

        col1, col2, col3 = st.columns(3)

        with col1:
            improvements = (episodes_df["eff_change"] > 0).sum()
            improvement_rate = (improvements / (len(episodes_df) - 1) * 100) if len(episodes_df) > 1 else 0
            st.metric(
                "Tasa de Mejora",
                f"{improvement_rate:.1f}%",
                f"{improvements} de {len(episodes_df)-1} episodios"
            )

        with col2:
            mean_improvement = episodes_df[episodes_df["eff_change"] > 0]["eff_change"].mean()
            st.metric(
                "Mejora Promedio",
                f"+{mean_improvement:.2f}%" if not pd.isna(mean_improvement) else "N/A",
                "cuando mejora"
            )

        with col3:
            mean_decline = episodes_df[episodes_df["eff_change"] < 0]["eff_change"].mean()
            st.metric(
                "Declive Promedio",
                f"{mean_decline:.2f}%" if not pd.isna(mean_decline) else "N/A",
                "cuando empeora"
            )

        # Episode-to-episode change distribution
        col1, col2 = st.columns(2)

        with col1:
            fig_change = px.histogram(
                episodes_df.dropna(subset=["eff_change"]),
                x="eff_change",
                nbins=30,
                title="Distribución de Cambios Episode-to-Episode",
                labels={"eff_change": "Cambio en Eficiencia (%)", "count": "Frecuencia"},
                color_discrete_sequence=["#636EFA"]
            )
            fig_change.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_change, use_container_width=True)

        with col2:
            # Cumulative performance
            episodes_df["cumulative_mean"] = episodes_df["efficiency_percent"].expanding().mean()

            fig_cumulative = go.Figure()
            fig_cumulative.add_trace(go.Scatter(
                x=episodes_df["episode"],
                y=episodes_df["cumulative_mean"],
                mode="lines",
                name="Media Acumulada",
                line=dict(color="#00CC96", width=2)
            ))

            fig_cumulative.update_layout(
                title="Media Acumulada (Convergencia Global)",
                xaxis_title="Episodio",
                yaxis_title="Eficiencia Media Acumulada (%)",
                height=400
            )
            st.plotly_chart(fig_cumulative, use_container_width=True)

        # Policy consistency: consecutive episode similarity
        st.markdown("### 🎯 Consistencia de la Política")

        # Calculate runs of similar performance
        threshold = 5  # episodes within 5% are considered "similar"
        episodes_df["similar_to_next"] = (episodes_df["eff_change"].abs() < threshold)

        col1, col2, col3 = st.columns(3)

        with col1:
            consistency_rate = episodes_df["similar_to_next"].sum() / len(episodes_df) * 100
            st.metric(
                "Tasa de Consistencia",
                f"{consistency_rate:.1f}%",
                "episodios dentro de ±5%"
            )

        with col2:
            # Coefficient of variation over quarters
            quarters = np.array_split(episodes_df["efficiency_percent"].to_numpy(), 4)
            cv_trend = [(q.std() / q.mean() * 100) for q in quarters if len(q) > 0 and q.mean() > 0]
            if len(cv_trend) >= 2:
                cv_change = cv_trend[-1] - cv_trend[0]
                st.metric(
                    "Cambio en CV",
                    f"{cv_change:+.1f}%",
                    "inicio vs fin" + (" ↑ más variable" if cv_change > 0 else " ↓ más estable")
                )

        with col3:
            # Max consecutive improvements
            improvements = (episodes_df["eff_change"] > 0).astype(int)
            max_streak = 0
            current_streak = 0
            for val in improvements:
                if val == 1:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
            st.metric(
                "Racha Máxima",
                f"{max_streak} episodios",
                "mejoras consecutivas"
            )

        # Performance quartile transitions
        episodes_df["quartile"] = pd.qcut(episodes_df["efficiency_percent"], q=4, duplicates='drop')
        episodes_df["quartile"] = episodes_df["quartile"].astype(str)

        fig_quartiles = go.Figure()
        fig_quartiles.add_trace(go.Scatter(
            x=episodes_df["episode"],
            y=episodes_df["efficiency_percent"],
            mode="markers",
            marker=dict(
                size=8,
                color=pd.Categorical(episodes_df["quartile"]).codes,
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(
                    title="Cuartil",
                    tickvals=[0, 1, 2, 3],
                    ticktext=["Q1", "Q2", "Q3", "Q4"]
                )
            ),
            text=episodes_df["quartile"],
            hovertemplate="<b>Episodio %{x}</b><br>Eficiencia: %{y:.2f}%<br>%{text}<extra></extra>"
        ))

        fig_quartiles.update_layout(
            title="Distribución de Desempeño por Cuartiles",
            xaxis_title="Episodio",
            yaxis_title="Eficiencia (%)",
            height=400
        )
        st.plotly_chart(fig_quartiles, use_container_width=True)

        # Success rate over time (if applicable)
        if "reward" in episodes_df.columns:
            st.markdown("### 🏆 Tasa de Éxito Temporal")

            # Define success threshold (e.g., top 50% of episodes)
            success_threshold = episodes_df["efficiency_percent"].median()
            episodes_df["success"] = (episodes_df["efficiency_percent"] >= success_threshold).astype(int)

            # Rolling success rate
            window = min(10, len(episodes_df) // 3)
            episodes_df["success_rate"] = episodes_df["success"].rolling(window=window, min_periods=1).mean() * 100

            fig_success = go.Figure()
            fig_success.add_trace(go.Scatter(
                x=episodes_df["episode"],
                y=episodes_df["success_rate"],
                mode="lines",
                name="Tasa de Éxito",
                line=dict(color="#AB63FA", width=3),
                fill="tozeroy"
            ))

            fig_success.add_hline(
                y=50,
                line_dash="dash",
                line_color="gray",
                annotation_text="50% (esperado)"
            )

            fig_success.update_layout(
                title=f"Tasa de Éxito Móvil (ventana={window}, umbral={success_threshold:.1f}%)",
                xaxis_title="Episodio",
                yaxis_title="Tasa de Éxito (%)",
                height=400
            )
            st.plotly_chart(fig_success, use_container_width=True)

with tab3:
    st.subheader("Tendencias Temporales")

    if episodes_data and len(episodes_data) > 0:
        episodes_df = pd.DataFrame(episodes_data)

        # Add moving averages
        window = min(10, len(episodes_df) // 5) if len(episodes_df) > 10 else 3
        episodes_df[f"efficiency_ma{window}"] = episodes_df["efficiency_percent"].rolling(window=window, min_periods=1).mean()
        if "reward" in episodes_df.columns:
            episodes_df[f"reward_ma{window}"] = episodes_df["reward"].rolling(window=window, min_periods=1).mean()

        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Eficiencia y Gini a lo Largo del Tiempo", "Reward y Pasos a lo Largo del Tiempo"),
            specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
            vertical_spacing=0.15
        )

        # Row 1: Efficiency and Gini
        fig.add_scatter(
            x=episodes_df["episode"],
            y=episodes_df["efficiency_percent"],
            mode="lines",
            name="Eficiencia",
            line=dict(color="#1f77b4", width=1),
            opacity=0.4,
            row=1, col=1, secondary_y=False
        )

        fig.add_scatter(
            x=episodes_df["episode"],
            y=episodes_df[f"efficiency_ma{window}"],
            mode="lines",
            name=f"Eficiencia (MA{window})",
            line=dict(color="#1f77b4", width=2),
            row=1, col=1, secondary_y=False
        )

        fig.add_scatter(
            x=episodes_df["episode"],
            y=episodes_df["gini"],
            mode="lines",
            name="Gini",
            line=dict(color="#ff7f0e", width=1.5),
            row=1, col=1, secondary_y=True
        )

        # Row 2: Reward and Steps
        if "reward" in episodes_df.columns:
            fig.add_scatter(
                x=episodes_df["episode"],
                y=episodes_df["reward"],
                mode="lines",
                name="Reward",
                line=dict(color="#2ca02c", width=1),
                opacity=0.4,
                row=2, col=1, secondary_y=False
            )

            fig.add_scatter(
                x=episodes_df["episode"],
                y=episodes_df[f"reward_ma{window}"],
                mode="lines",
                name=f"Reward (MA{window})",
                line=dict(color="#2ca02c", width=2),
                row=2, col=1, secondary_y=False
            )

        if "steps" in episodes_df.columns:
            fig.add_scatter(
                x=episodes_df["episode"],
                y=episodes_df["steps"],
                mode="lines",
                name="Steps",
                line=dict(color="#d62728", width=1.5),
                row=2, col=1, secondary_y=True
            )

        # Update axes
        fig.update_xaxes(title_text="Episodio", row=1, col=1)
        fig.update_xaxes(title_text="Episodio", row=2, col=1)
        fig.update_yaxes(title_text="Eficiencia (%)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Gini", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Reward", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Steps", row=2, col=1, secondary_y=True)

        fig.update_layout(height=700, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Trend analysis
        st.subheader("Análisis de Tendencia")

        col1, col2, col3 = st.columns(3)

        # Linear regression for efficiency trend
        x = np.arange(len(episodes_df))
        slope_eff, intercept_eff, r_value_eff, p_value_eff, std_err_eff = scipy_stats.linregress(x, episodes_df["efficiency_percent"].to_numpy())

        with col1:
            trend_direction = "📈 Ascendente" if slope_eff > 0.01 else "📉 Descendente" if slope_eff < -0.01 else "➡️ Estable"
            st.metric(
                "Tendencia de Eficiencia",
                trend_direction,
                f"{slope_eff:.4f} por episodio"
            )
            st.caption(f"R² = {r_value_eff**2:.3f}, p = {p_value_eff:.4f}")

        with col2:
            if "reward" in episodes_df.columns:
                slope_rew, intercept_rew, r_value_rew, p_value_rew, std_err_rew = scipy_stats.linregress(x, episodes_df["reward"].to_numpy())
                trend_direction = "📈 Ascendente" if slope_rew > 0.01 else "📉 Descendente" if slope_rew < -0.01 else "➡️ Estable"
                st.metric(
                    "Tendencia de Reward",
                    trend_direction,
                    f"{slope_rew:.4f} por episodio"
                )
                st.caption(f"R² = {r_value_rew**2:.3f}, p = {p_value_rew:.4f}")

        with col3:
            # Variability analysis
            first_half_eff = episodes_df["efficiency_percent"].iloc[:len(episodes_df)//2].std()
            second_half_eff = episodes_df["efficiency_percent"].iloc[len(episodes_df)//2:].std()
            stability_change = ((second_half_eff - first_half_eff) / first_half_eff * 100) if first_half_eff > 0 else 0

            st.metric(
                "Cambio en Estabilidad",
                f"{stability_change:+.1f}%",
                "Más variable" if stability_change > 0 else "Más estable"
            )
            st.caption(f"Std 1ª mitad: {first_half_eff:.2f}, 2ª mitad: {second_half_eff:.2f}")

with tab4:
    st.subheader("Análisis de Episodios Extremos")

    if episodes_data and len(episodes_data) > 0:
        episodes_df = pd.DataFrame(episodes_data)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🏆 Top 5 Mejores Episodios")
            top5 = episodes_df.nlargest(5, "efficiency_percent")

            display_cols = ["episode", "efficiency_percent", "intake", "gini", "reward"]
            available_display_cols = [col for col in display_cols if col in top5.columns]

            st.dataframe(
                top5[available_display_cols].style.background_gradient(
                    subset=["efficiency_percent"],
                    cmap="Greens"
                ).format({
                    "efficiency_percent": "{:.2f}%",
                    "intake": "{:.1f}",
                    "gini": "{:.4f}",
                    "reward": "{:.1f}"
                }),
                width='stretch',
                hide_index=True
            )

            # Characteristics of best episodes
            st.markdown("**Características promedio:**")
            best_stats = {
                "Eficiencia": f"{top5['efficiency_percent'].mean():.2f}%",
                "Gini": f"{top5['gini'].mean():.4f}",
                "Intake": f"{top5['intake'].mean():.1f}",
            }
            if "reward" in top5.columns:
                best_stats["Reward"] = f"{top5['reward'].mean():.1f}"
            if "steps" in top5.columns:
                best_stats["Steps"] = f"{top5['steps'].mean():.0f}"

            st.json(best_stats)

        with col2:
            st.markdown("### 📉 Bottom 5 Peores Episodios")
            bottom5 = episodes_df.nsmallest(5, "efficiency_percent")

            st.dataframe(
                bottom5[available_display_cols].style.background_gradient(
                    subset=["efficiency_percent"],
                    cmap="Reds"
                ).format({
                    "efficiency_percent": "{:.2f}%",
                    "intake": "{:.1f}",
                    "gini": "{:.4f}",
                    "reward": "{:.1f}"
                }),
                width='stretch',
                hide_index=True
            )

            # Characteristics of worst episodes
            st.markdown("**Características promedio:**")
            worst_stats = {
                "Eficiencia": f"{bottom5['efficiency_percent'].mean():.2f}%",
                "Gini": f"{bottom5['gini'].mean():.4f}",
                "Intake": f"{bottom5['intake'].mean():.1f}",
            }
            if "reward" in bottom5.columns:
                worst_stats["Reward"] = f"{bottom5['reward'].mean():.1f}"
            if "steps" in bottom5.columns:
                worst_stats["Steps"] = f"{bottom5['steps'].mean():.0f}"

            st.json(worst_stats)

        # Comparative analysis
        st.markdown("### 🔍 Diferencias Clave entre Mejores y Peores")

        comparison_data = []
        for metric in ["efficiency_percent", "gini", "intake", "reward"]:
            if metric in episodes_df.columns:
                best_val = top5[metric].mean()
                worst_val = bottom5[metric].mean()
                diff = best_val - worst_val
                diff_pct = (diff / worst_val * 100) if worst_val != 0 else 0

                comparison_data.append({
                    "Métrica": metric.replace("_", " ").title(),
                    "Top 5": f"{best_val:.2f}",
                    "Bottom 5": f"{worst_val:.2f}",
                    "Diferencia": f"{diff:+.2f}",
                    "Diferencia (%)": f"{diff_pct:+.1f}%"
                })

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, width='stretch', hide_index=True)

        # Outlier detection
        st.markdown("### 🎯 Detección de Outliers (Eficiencia)")

        Q1 = episodes_df["efficiency_percent"].quantile(0.25)
        Q3 = episodes_df["efficiency_percent"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = episodes_df[
            (episodes_df["efficiency_percent"] < lower_bound) |
            (episodes_df["efficiency_percent"] > upper_bound)
        ]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Outliers Detectados", len(outliers))
        with col2:
            st.metric("Rango Normal", f"{lower_bound:.1f}% - {upper_bound:.1f}%")
        with col3:
            st.metric("% Outliers", f"{len(outliers)/len(episodes_df)*100:.1f}%")

        if len(outliers) > 0:
            st.dataframe(
                outliers[available_display_cols].style.format({
                    "efficiency_percent": "{:.2f}%",
                    "intake": "{:.1f}",
                    "gini": "{:.4f}",
                    "reward": "{:.1f}"
                }),
                width='stretch',
                hide_index=True
            )

with tab5:
    st.subheader("Datos Detallados por Episodio")

    if episodes_data and len(episodes_data) > 0:
        episodes_df = pd.DataFrame(episodes_data)

        # Filter options
        col1, col2, col3 = st.columns(3)

        with col1:
            min_eff = st.slider(
                "Eficiencia mínima (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=5.0
            )

        with col2:
            sort_by = st.selectbox(
                "Ordenar por",
                options=["episode", "efficiency_percent", "gini", "intake", "reward"],
                index=1
            )

        with col3:
            sort_order = st.radio("Orden", ["Descendente", "Ascendente"], horizontal=True)

        # Apply filters
        filtered_df = episodes_df[episodes_df["efficiency_percent"] >= min_eff]

        # Apply sorting
        ascending = sort_order == "Ascendente"
        if sort_by in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

        st.info(f"Mostrando {len(filtered_df)} de {len(episodes_df)} episodios")

        display_cols = [
            "episode",
            "efficiency_percent",
            "intake",
            "gini",
            "reward",
            "steps",
        ]
        if "polarization" in filtered_df.columns:
            display_cols.append("polarization")
        if "mean_neighbor_distance" in filtered_df.columns:
            display_cols.append("mean_neighbor_distance")

        available_cols = [col for col in display_cols if col in filtered_df.columns]

        # Color code efficiency column
        def color_efficiency(val):
            if val >= 80:
                return 'background-color: #d4edda'
            elif val >= 60:
                return 'background-color: #d1ecf1'
            elif val >= 40:
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #f8d7da'

        format_dict = {
            "efficiency_percent": "{:.2f}%",
            "intake": "{:.2f}",
            "gini": "{:.4f}",
            "reward": "{:.2f}",
        }

        if "polarization" in filtered_df.columns:
            format_dict["polarization"] = "{:.4f}"
        if "mean_neighbor_distance" in filtered_df.columns:
            format_dict["mean_neighbor_distance"] = "{:.2f}"

        styled_df = filtered_df[available_cols].style.map(
            color_efficiency,
            subset=["efficiency_percent"]
        ).format(format_dict)

        st.dataframe(
            styled_df,
            width='stretch',
            height=450,
        )

        # Download options
        col1, col2 = st.columns(2)

        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Descargar datos filtrados (CSV)",
                data=csv,
                file_name=f"{selected_file.replace('.json', '')}_filtered.csv",
                mime="text/csv",
            )

        with col2:
            # Download full JSON
            full_json = json.dumps(results, indent=2)
            st.download_button(
                label="📥 Descargar JSON completo",
                data=full_json,
                file_name=selected_file,
                mime="application/json",
            )

# Footer
st.markdown("---")
st.markdown("**Multi-Agent Flocking & Foraging RL** | Dashboard generado con Streamlit")
