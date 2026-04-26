import json
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.graph_objects as go
except Exception:
    go = None


# ============================================================
# CONFIG / STYLE
# ============================================================

st.set_page_config(page_title="Laboratorio WLF", layout="wide")

st.markdown(
    """
<style>
div[data-testid="stHorizontalBlock"] { gap: 1rem; }
.lab-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 16px;
    padding: 15px 17px;
    min-height: 100px;
    background: rgba(255,255,255,0.025);
}
.lab-card-title {
    font-size: 0.90rem;
    opacity: 0.78;
    margin-bottom: 8px;
}
.lab-card-value {
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.1;
}
.small-note { opacity: 0.75; font-size: 0.90rem; }
.section-note {
    border-left: 4px solid rgba(128,128,128,0.45);
    padding: 8px 12px;
    margin: 8px 0 18px 0;
    opacity: 0.90;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SMALL HELPERS
# ============================================================


def card(title: str, value: str):
    st.markdown(
        f"""
        <div class="lab-card">
            <div class="lab-card-title">{title}</div>
            <div class="lab-card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_note(text: str):
    st.markdown(f"<div class='section-note'>{text}</div>", unsafe_allow_html=True)



def format_date_axis(ax, min_ticks: int = 5, max_ticks: int = 9, rotation: int = 25):
    """Keep date charts readable when many days/months are loaded."""
    locator = mdates.AutoDateLocator(minticks=min_ticks, maxticks=max_ticks)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", rotation=rotation)


def render_clean_daily_pnl_chart(daily_df: pd.DataFrame):
    """Interactive daily PnL chart with simple hover details."""
    if daily_df.empty:
        st.info("No hay datos diarios para mostrar.")
        return

    chart_df = daily_df.copy()
    chart_df["trade_day"] = pd.to_datetime(chart_df["trade_day"], errors="coerce")
    chart_df = chart_df.dropna(subset=["trade_day"]).sort_values("trade_day")

    if chart_df.empty:
        st.info("No hay fechas válidas para mostrar en el gráfico diario.")
        return

    chart_df["fecha"] = chart_df["trade_day"].dt.strftime("%Y-%m-%d")
    chart_df["resultado"] = np.where(chart_df["pnl_total"] >= 0, "Ganador", "Perdedor")

    hover_parts = [
        "<b>Día:</b> %{customdata[0]}",
        "<b>Resultado:</b> %{customdata[1]}",
        "<b>PnL:</b> $%{y:,.2f}",
    ]

    optional_cols = [
        ("operaciones", "Operaciones"),
        ("tasa_acierto", "Win rate"),
        ("peor_operacion", "Peor operación"),
        ("mejor_operacion", "Mejor operación"),
        ("reversiones_promedio", "Reversals prom."),
    ]

    custom_cols = ["fecha", "resultado"]
    for col, label in optional_cols:
        if col in chart_df.columns:
            custom_cols.append(col)
            idx = len(custom_cols) - 1
            if col == "tasa_acierto":
                hover_parts.append(f"<b>{label}:</b> %{{customdata[{idx}]:.1f}}%")
            elif col in ["peor_operacion", "mejor_operacion"]:
                hover_parts.append(f"<b>{label}:</b> $%{{customdata[{idx}]:,.2f}}")
            else:
                hover_parts.append(f"<b>{label}:</b> %{{customdata[{idx}]}}")

    hover_template = "<br>".join(hover_parts) + "<extra></extra>"

    if go is not None:
        bar_colors = np.where(chart_df["pnl_total"] >= 0, "#2E86C1", "#C0392B")
        fig = go.Figure()
        fig.add_bar(
            x=chart_df["trade_day"],
            y=chart_df["pnl_total"],
            customdata=chart_df[custom_cols].to_numpy(),
            hovertemplate=hover_template,
            marker_color=bar_colors,
            name="Resultado diario",
        )
        fig.add_hline(y=0, line_width=1)
        fig.update_layout(
            title="Resultado diario",
            xaxis_title="Fecha",
            yaxis_title="PnL",
            hovermode="x unified",
            height=420,
            margin=dict(l=40, r=25, t=55, b=40),
        )
        fig.update_xaxes(nticks=10, tickformat="%b %d")
        st.plotly_chart(fig, use_container_width=True)
        return

    # Fallback if Plotly is not available. This is static, but still keeps dates clean.
    st.warning("Plotly no está instalado en este entorno. El gráfico será estático. Para hover/interactividad agrega `plotly` en requirements.txt y reinicia la app.")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(chart_df["trade_day"], chart_df["pnl_total"], width=0.8)
    ax.axhline(0, linewidth=1)
    ax.set_title("Resultado diario")
    ax.set_ylabel("PnL")
    ax.set_xlabel("Fecha")
    format_date_axis(ax, min_ticks=5, max_ticks=10, rotation=25)
    fig.tight_layout()
    st.pyplot(fig)




def render_monthly_result_chart(month_df: pd.DataFrame):
    """Interactive month result chart used in Dashboard General."""
    if month_df.empty or "month" not in month_df.columns or "pnl_total" not in month_df.columns:
        st.info("No hay datos mensuales para mostrar.")
        return

    chart_df = month_df.copy().sort_values("month")
    chart_df["resultado"] = np.where(chart_df["pnl_total"] >= 0, "Mes ganador", "Mes perdedor")

    custom_cols = ["month", "resultado"]
    hover_lines = [
        "<b>Mes:</b> %{customdata[0]}",
        "<b>Resultado:</b> %{customdata[1]}",
        "<b>PnL:</b> $%{y:,.2f}",
    ]

    optional_cols = [
        ("operaciones", "Operaciones", "int"),
        ("profit_factor", "Profit factor", "num"),
        ("tasa_acierto", "Win rate", "pct"),
        ("pnl_promedio", "PnL promedio", "money"),
        ("peor_operacion", "Peor operación", "money"),
        ("mejor_operacion", "Mejor operación", "money"),
        ("drawdown_max", "Mayor drawdown", "money"),
        ("reversiones_promedio", "Reversals promedio", "num"),
        ("contratos_max", "Máx contratos", "num"),
    ]

    for col, label, kind in optional_cols:
        if col in chart_df.columns:
            custom_cols.append(col)
            idx = len(custom_cols) - 1
            if kind == "pct":
                hover_lines.append(f"<b>{label}:</b> %{{customdata[{idx}]:.1f}}%")
            elif kind == "money":
                hover_lines.append(f"<b>{label}:</b> $%{{customdata[{idx}]:,.2f}}")
            elif kind == "int":
                hover_lines.append(f"<b>{label}:</b> %{{customdata[{idx}]:.0f}}")
            else:
                hover_lines.append(f"<b>{label}:</b> %{{customdata[{idx}]:.2f}}")

    if go is not None:
        colors = np.where(chart_df["pnl_total"] >= 0, "#2E86C1", "#C0392B")
        fig = go.Figure()
        fig.add_bar(
            x=chart_df["month"],
            y=chart_df["pnl_total"],
            customdata=chart_df[custom_cols].to_numpy(),
            hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
            marker_color=colors,
            name="PnL mensual",
        )
        fig.add_hline(y=0, line_width=1)
        fig.update_layout(
            title="Resultado por mes",
            xaxis_title="Mes",
            yaxis_title="PnL",
            hovermode="closest",
            height=420,
            margin=dict(l=40, r=25, t=55, b=45),
        )
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, use_container_width=True)
        return

    st.warning("Plotly no está instalado. El gráfico será estático y no tendrá hover.")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(chart_df["month"].astype(str), chart_df["pnl_total"])
    ax.axhline(0, linewidth=1)
    ax.set_title("Resultado por mes")
    ax.set_xlabel("Mes")
    ax.set_ylabel("PnL")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    st.pyplot(fig)


def render_reversal_month_impact_chart(sim_month: pd.DataFrame, selected_reversal: int):
    """Interactive monthly comparison: real PnL vs selected reversal PnL."""
    needed = {"month", "pnl_real", "pnl_reversal_seleccionado"}
    if sim_month.empty or not needed.issubset(set(sim_month.columns)):
        return

    chart_df = sim_month.copy().sort_values("month")

    custom_cols = [
        "month",
        "diferencia_pnl",
        "profit_factor_real",
        "profit_factor_reversal_seleccionado",
        "max_drawdown_real",
        "max_drawdown_reversal_seleccionado",
        "operaciones_cortadas",
        "resultados_cambiados",
        "operaciones",
    ]
    custom_cols = [c for c in custom_cols if c in chart_df.columns]

    if go is not None:
        fig = go.Figure()
        fig.add_bar(
            x=chart_df["month"],
            y=chart_df["pnl_real"],
            name="Real",
            customdata=chart_df[custom_cols].to_numpy(),
            hovertemplate=(
                "<b>Mes:</b> %{customdata[0]}<br>"
                "<b>Tipo:</b> Real<br>"
                "<b>PnL real:</b> $%{y:,.2f}<br>"
                "<b>Diferencia si se corta:</b> $%{customdata[1]:,.2f}<br>"
                "<b>PF real:</b> %{customdata[2]:.2f}<br>"
                "<b>PF seleccionado:</b> %{customdata[3]:.2f}<br>"
                "<b>Caída máx real:</b> $%{customdata[4]:,.2f}<br>"
                "<b>Caída máx seleccionado:</b> $%{customdata[5]:,.2f}<br>"
                "<b>Ops cortadas:</b> %{customdata[6]:.0f}<br>"
                "<b>Resultados cambiados:</b> %{customdata[7]:.0f}<br>"
                "<b>Operaciones:</b> %{customdata[8]:.0f}"
                "<extra></extra>"
            ),
        )
        fig.add_bar(
            x=chart_df["month"],
            y=chart_df["pnl_reversal_seleccionado"],
            name=f"Reversal {selected_reversal}",
            customdata=chart_df[custom_cols].to_numpy(),
            hovertemplate=(
                "<b>Mes:</b> %{customdata[0]}<br>"
                f"<b>Tipo:</b> Reversal {selected_reversal}<br>"
                "<b>PnL con reversal seleccionado:</b> $%{y:,.2f}<br>"
                "<b>Diferencia vs real:</b> $%{customdata[1]:,.2f}<br>"
                "<b>PF real:</b> %{customdata[2]:.2f}<br>"
                "<b>PF seleccionado:</b> %{customdata[3]:.2f}<br>"
                "<b>Caída máx real:</b> $%{customdata[4]:,.2f}<br>"
                "<b>Caída máx seleccionado:</b> $%{customdata[5]:,.2f}<br>"
                "<b>Ops cortadas:</b> %{customdata[6]:.0f}<br>"
                "<b>Resultados cambiados:</b> %{customdata[7]:.0f}<br>"
                "<b>Operaciones:</b> %{customdata[8]:.0f}"
                "<extra></extra>"
            ),
        )
        fig.add_hline(y=0, line_width=1)
        fig.update_layout(
            title=f"Impacto mensual si el bot se detiene en reversal {selected_reversal}",
            xaxis_title="Mes",
            yaxis_title="PnL",
            barmode="group",
            hovermode="closest",
            height=430,
            margin=dict(l=40, r=25, t=55, b=45),
        )
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, use_container_width=True)
        return

    st.warning("Plotly no está instalado. El gráfico será estático y no tendrá hover.")
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(chart_df))
    width = 0.38
    ax.bar(x - width / 2, chart_df["pnl_real"], width, label="Real")
    ax.bar(x + width / 2, chart_df["pnl_reversal_seleccionado"], width, label=f"Reversal {selected_reversal}")
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(chart_df["month"].astype(str), rotation=25)
    ax.set_title(f"Impacto mensual si el bot se detiene en reversal {selected_reversal}")
    ax.set_xlabel("Mes")
    ax.set_ylabel("PnL")
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)


def show_help(title: str, description: str, questions: List[str]):
    with st.expander(f"¿Para qué sirve esta página? · {title}", expanded=False):
        st.markdown(description)
        st.markdown("**Decisión que deberías poder tomar:**")
        for q in questions:
            st.markdown(f"- {q}")


def show_conclusion(title: str, lines: List[str]):
    st.markdown(f"### Lectura rápida · {title}")
    if not lines:
        st.markdown("- Todavía no hay suficiente información para una conclusión clara.")
        return
    for line in lines:
        st.markdown(f"- {line}")


def fmt_money(x) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):,.2f}"


def fmt_pct(x) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):.1f}%"


def safe_div(a, b) -> float:
    if b is None or pd.isna(b) or float(b) == 0:
        return np.nan
    return float(a) / float(b)


def _to_datetime(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ============================================================
# LOAD / BUILD DATA
# ============================================================


def load_uploaded_jsonl_files(uploaded_files) -> List[dict]:
    records: List[dict] = []

    if not uploaded_files:
        st.sidebar.info("Carga uno o más archivos JSONL para empezar.")
        return records

    for uploaded_file in uploaded_files:
        try:
            content = uploaded_file.getvalue().decode("utf-8-sig", errors="ignore")
            lines = content.splitlines()
            valid_count = 0
            invalid_count = 0

            for i, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    obj["source_file"] = uploaded_file.name
                    records.append(obj)
                    valid_count += 1
                except json.JSONDecodeError as exc:
                    invalid_count += 1
                    if invalid_count <= 3:
                        st.sidebar.warning(f"JSON inválido en {uploaded_file.name}, línea {i}: {exc}")

            with st.sidebar.expander(f"Archivo · {uploaded_file.name}", expanded=False):
                st.write(f"Líneas crudas: {len(lines)}")
                st.write(f"Registros válidos: {valid_count}")
                st.write(f"Líneas inválidas: {invalid_count}")
        except Exception as exc:
            st.sidebar.error(f"Error al leer {uploaded_file.name}: {exc}")

    return records


def classify_session(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "Sin sesión"

    total_min = ts.hour * 60 + ts.minute

    if total_min >= 18 * 60 or total_min <= (3 * 60 + 29):
        return "Asia"
    if (3 * 60 + 30) <= total_min <= (9 * 60 + 29):
        return "Londres"
    if (9 * 60 + 30) <= total_min <= (10 * 60 + 30):
        return "NY Open"
    if (10 * 60 + 31) <= total_min <= (13 * 60 + 29):
        return "NY Midday"
    if (13 * 60 + 30) <= total_min <= (17 * 60):
        return "NY Late"
    return "Fuera de Sesión"


def build_dataframes(records: List[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not records:
        return pd.DataFrame(), pd.DataFrame()

    ops_rows = []
    legs_rows = []

    for rec in records:
        ops_rows.append(
            {
                "source_file": rec.get("source_file"),
                "operation_id": rec.get("operation_id"),
                "bot_name": rec.get("bot_name"),
                "bot_version": rec.get("bot_version"),
                "instrument": rec.get("instrument"),
                "date": rec.get("date"),
                "sequence_started_at": rec.get("sequence_started_at"),
                "sequence_ended_at": rec.get("sequence_ended_at"),
                "sequence_end_reason": rec.get("sequence_end_reason"),
                "base_price": rec.get("base_price"),
                "range_high": rec.get("range_high"),
                "range_low": rec.get("range_low"),
                "base_contracts": rec.get("base_contracts"),
                "reversal_count": rec.get("reversal_count"),
                "reversal_sizing_mode": rec.get("reversal_sizing_mode"),
                "max_reversals_allowed": rec.get("max_reversals_allowed"),
                "fixed_stop_ticks": rec.get("fixed_stop_ticks"),
                "fixed_target_ticks": rec.get("fixed_target_ticks"),
                "distance_points": rec.get("distance_points"),
                "sequence_net_pnl_currency": rec.get("sequence_net_pnl_currency"),
                "sequence_loss_currency": rec.get("sequence_loss_currency"),
                "sequence_execution_commission_currency": rec.get("sequence_execution_commission_currency"),
                "base_target_profit_currency": rec.get("base_target_profit_currency"),
                "operation_max_drawdown_currency": rec.get("operation_max_drawdown_currency"),
                "operation_max_runup_currency": rec.get("operation_max_runup_currency"),
            }
        )

        for leg in rec.get("legs", []):
            legs_rows.append(
                {
                    "source_file": rec.get("source_file"),
                    "operation_id": rec.get("operation_id"),
                    "date": rec.get("date"),
                    "sequence_started_at": rec.get("sequence_started_at"),
                    "leg_index": leg.get("leg_index"),
                    "leg_type": leg.get("leg_type"),
                    "reversal_number": leg.get("reversal_number"),
                    "direction": leg.get("direction"),
                    "signal_name": leg.get("signal_name"),
                    "entry_time": leg.get("entry_time"),
                    "entry_price_avg": leg.get("entry_price_avg"),
                    "entry_qty": leg.get("entry_qty"),
                    "initial_stop_price": leg.get("initial_stop_price"),
                    "initial_target_price": leg.get("initial_target_price"),
                    "exit_time": leg.get("exit_time"),
                    "exit_price_avg": leg.get("exit_price_avg"),
                    "exit_reason": leg.get("exit_reason"),
                    "exit_result_type": leg.get("exit_result_type"),
                    "realized_pnl_currency": leg.get("realized_pnl_currency"),
                    "sequence_loss_before_entry": leg.get("sequence_loss_before_entry"),
                    "smart_recovery_qty_computed": leg.get("smart_recovery_qty_computed"),
                    "auto_be_activated": leg.get("auto_be_activated"),
                    "trailing_activated": leg.get("trailing_activated"),
                    "cumulative_sequence_pnl_after_leg": leg.get("cumulative_sequence_pnl_after_leg"),
                    "operation_drawdown_after_leg": leg.get("operation_drawdown_after_leg"),
                    "operation_runup_after_leg": leg.get("operation_runup_after_leg"),
                }
            )

    ops_df = pd.DataFrame(ops_rows)
    legs_df = pd.DataFrame(legs_rows)

    ops_df = _to_datetime(ops_df, ["date", "sequence_started_at", "sequence_ended_at"])
    legs_df = _to_datetime(legs_df, ["date", "sequence_started_at", "entry_time", "exit_time"])

    ops_df = _to_numeric(
        ops_df,
        [
            "base_price", "range_high", "range_low", "base_contracts", "reversal_count",
            "max_reversals_allowed", "fixed_stop_ticks", "fixed_target_ticks", "distance_points",
            "sequence_net_pnl_currency", "sequence_loss_currency", "sequence_execution_commission_currency",
            "base_target_profit_currency", "operation_max_drawdown_currency", "operation_max_runup_currency",
        ],
    )
    legs_df = _to_numeric(
        legs_df,
        [
            "leg_index", "reversal_number", "entry_price_avg", "entry_qty", "initial_stop_price",
            "initial_target_price", "exit_price_avg", "realized_pnl_currency", "sequence_loss_before_entry",
            "smart_recovery_qty_computed", "cumulative_sequence_pnl_after_leg",
            "operation_drawdown_after_leg", "operation_runup_after_leg",
        ],
    )

    if ops_df.empty:
        return ops_df, legs_df

    ops_df = ops_df.sort_values("sequence_started_at").copy()
    ops_df["trade_day"] = ops_df["sequence_started_at"].dt.date
    ops_df["month"] = ops_df["sequence_started_at"].dt.to_period("M").astype(str)
    ops_df["hora_inicio"] = ops_df["sequence_started_at"].dt.hour
    ops_df["dia_semana"] = ops_df["sequence_started_at"].dt.day_name()
    ops_df["sesion"] = ops_df["sequence_started_at"].apply(classify_session)
    ops_df["es_ganadora"] = ops_df["sequence_net_pnl_currency"].fillna(0) > 0

    ops_df["config_key"] = (
        "BC=" + ops_df["base_contracts"].fillna(-1).astype(int).astype(str)
        + " | SL=" + ops_df["fixed_stop_ticks"].fillna(-1).astype(int).astype(str)
        + " | TP=" + ops_df["fixed_target_ticks"].fillna(-1).astype(int).astype(str)
        + " | REV=" + ops_df["max_reversals_allowed"].fillna(-1).astype(int).astype(str)
        + " | DIST=" + ops_df["distance_points"].fillna(-1).round(2).astype(str)
    )

    if not legs_df.empty:
        legs_df["month"] = legs_df["sequence_started_at"].dt.to_period("M").astype(str)
        max_qty = legs_df.groupby("operation_id", as_index=False).agg(max_contracts_used=("entry_qty", "max"))
        ops_df = ops_df.merge(max_qty, on="operation_id", how="left")

        legs_df = legs_df.merge(
            ops_df[["operation_id", "trade_day", "month", "config_key", "instrument", "sesion", "hora_inicio", "dia_semana"]],
            on="operation_id",
            how="left",
            suffixes=("", "_op"),
        )
    else:
        ops_df["max_contracts_used"] = np.nan

    return ops_df, legs_df


# ============================================================
# METRICS
# ============================================================


def aggregate_core(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(group_cols, dropna=False).agg(
        operaciones=("operation_id", "count"),
        pnl_total=("sequence_net_pnl_currency", "sum"),
        pnl_promedio=("sequence_net_pnl_currency", "mean"),
        pnl_mediano=("sequence_net_pnl_currency", "median"),
        tasa_acierto=("es_ganadora", "mean"),
        peor_operacion=("sequence_net_pnl_currency", "min"),
        mejor_operacion=("sequence_net_pnl_currency", "max"),
        drawdown_max=("operation_max_drawdown_currency", "max"),
        drawdown_promedio=("operation_max_drawdown_currency", "mean"),
        reversiones_promedio=("reversal_count", "mean"),
        contratos_max=("max_contracts_used", "max"),
    ).reset_index()
    grouped["tasa_acierto"] = grouped["tasa_acierto"] * 100
    return add_profit_factor(df, grouped, group_cols)


def add_profit_factor(source_df: pd.DataFrame, grouped: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if source_df.empty or grouped.empty:
        return grouped

    tmp = source_df.copy()
    tmp["gross_profit"] = tmp["sequence_net_pnl_currency"].where(tmp["sequence_net_pnl_currency"] > 0, 0)
    tmp["gross_loss"] = -tmp["sequence_net_pnl_currency"].where(tmp["sequence_net_pnl_currency"] < 0, 0)

    pf = tmp.groupby(group_cols, dropna=False).agg(
        gross_profit=("gross_profit", "sum"),
        gross_loss=("gross_loss", "sum"),
    ).reset_index()
    pf["profit_factor"] = pf.apply(lambda r: safe_div(r["gross_profit"], r["gross_loss"]), axis=1)
    return grouped.merge(pf[group_cols + ["profit_factor"]], on=group_cols, how="left")


def overview_metrics(ops_df: pd.DataFrame) -> Dict[str, float]:
    if ops_df.empty:
        return {}

    pnl = ops_df["sequence_net_pnl_currency"].fillna(0)
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    daily = ops_df.groupby("trade_day")["sequence_net_pnl_currency"].sum()

    return {
        "ops": len(ops_df),
        "days": daily.shape[0],
        "pnl": pnl.sum(),
        "win_rate": (pnl > 0).mean() * 100,
        "profit_factor": safe_div(winners.sum(), abs(losers.sum())),
        "avg_pnl": pnl.mean(),
        "best_op": pnl.max(),
        "worst_op": pnl.min(),
        "best_day": daily.max() if len(daily) else np.nan,
        "worst_day": daily.min() if len(daily) else np.nan,
        "positive_days_rate": (daily > 0).mean() * 100 if len(daily) else np.nan,
        "max_dd": ops_df["operation_max_drawdown_currency"].max(),
        "avg_rev": ops_df["reversal_count"].mean(),
        "max_contracts": ops_df["max_contracts_used"].max(),
    }


# ============================================================

def monthly_summary(ops_df: pd.DataFrame) -> pd.DataFrame:
    """Simple month-by-month health table used by Dashboard General."""
    if ops_df.empty or "month" not in ops_df.columns:
        return pd.DataFrame()
    grouped = aggregate_core(ops_df, ["month"])
    if grouped.empty:
        return grouped
    preferred_cols = [
        "month", "operaciones", "pnl_total", "profit_factor", "tasa_acierto",
        "pnl_promedio", "peor_operacion", "mejor_operacion", "drawdown_max",
        "reversiones_promedio", "contratos_max",
    ]
    cols = [c for c in preferred_cols if c in grouped.columns]
    return grouped[cols].sort_values("month")


def daily_summary(ops_df: pd.DataFrame) -> pd.DataFrame:
    """Simple day-by-day health table used by Dashboard General."""
    if ops_df.empty or "trade_day" not in ops_df.columns:
        return pd.DataFrame()
    grouped = aggregate_core(ops_df, ["trade_day"])
    if grouped.empty:
        return grouped
    if "month" in ops_df.columns:
        day_month = ops_df.groupby("trade_day", dropna=False).agg(month=("month", "first")).reset_index()
        grouped = grouped.merge(day_month, on="trade_day", how="left")
    preferred_cols = [
        "month", "trade_day", "operaciones", "pnl_total", "profit_factor",
        "tasa_acierto", "peor_operacion", "mejor_operacion", "drawdown_max",
        "reversiones_promedio", "contratos_max",
    ]
    cols = [c for c in preferred_cols if c in grouped.columns]
    return grouped[cols].sort_values("trade_day")


def _parse_hms_to_minutes(hms: str) -> int:
    """Convert HH:MM:SS or HH:MM to minutes after midnight."""
    try:
        parts = str(hms).strip().split(":")
        hour = int(parts[0]) if len(parts) > 0 else 0
        minute = int(parts[1]) if len(parts) > 1 else 0
        hour = max(0, min(23, hour))
        minute = max(0, min(59, minute))
        return hour * 60 + minute
    except Exception:
        return 18 * 60


def apply_operational_trade_day(
    df: pd.DataFrame,
    time_col: str,
    session_start_hms: str = "18:00:00",
    session_end_hms: str = "17:00:00",
) -> pd.DataFrame:
    """Assign bot trading day using the session start time.

    Example with 18:00 -> 17:00:
    - 2026-01-02 20:00 belongs to trading day 2026-01-02
    - 2026-01-03 02:00 belongs to trading day 2026-01-02
    """
    out = df.copy()
    if out.empty or time_col not in out.columns:
        if "trade_day" not in out.columns:
            out["trade_day"] = pd.NaT
        return out

    ts = pd.to_datetime(out[time_col], errors="coerce")
    start_min = _parse_hms_to_minutes(session_start_hms)
    minutes = ts.dt.hour.fillna(0).astype(int) * 60 + ts.dt.minute.fillna(0).astype(int)
    before_session_start = minutes < start_min
    adjusted = ts.where(~before_session_start, ts - pd.Timedelta(days=1))
    out["trade_day"] = adjusted.dt.date
    out["operational_trade_day"] = out["trade_day"]
    return out


def month_label_es(value) -> str:
    """Return compact Spanish month label like Ene 26."""
    if pd.isna(value):
        return "Sin mes"
    month_names = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
    try:
        if isinstance(value, pd.Period):
            year = value.year
            month = value.month
        else:
            s = str(value).strip()
            if len(s) >= 7 and s[4] == "-":
                year = int(s[:4])
                month = int(s[5:7])
            else:
                dt = pd.to_datetime(value, errors="coerce")
                if pd.isna(dt):
                    return s
                year = int(dt.year)
                month = int(dt.month)
        return f"{month_names.get(month, str(month))} {str(year)[-2:]}"
    except Exception:
        return str(value)
# SIMULATIONS
# ============================================================


def _prepare_leg_timeline(
    legs_df: pd.DataFrame,
    use_operational_day: bool = False,
    session_start_hms: str = "18:00:00",
    session_end_hms: str = "17:00:00",
) -> pd.DataFrame:
    """Return one row per real executed leg, ordered by the moment the PnL becomes known.

    For daily target/loss math we use legs, not operation totals.
    A leg is the real entry/exit unit. The realized PnL is only known at exit,
    so the simulator accumulates PnL by exit_time. If exit_time is missing,
    it falls back to entry_time and then sequence_started_at.
    """
    if legs_df.empty:
        return pd.DataFrame()

    legs = legs_df.copy()
    legs["event_time"] = legs["exit_time"]
    legs["event_time"] = legs["event_time"].fillna(legs["entry_time"])
    legs["event_time"] = legs["event_time"].fillna(legs["sequence_started_at"])

    if use_operational_day:
        legs = apply_operational_trade_day(legs, "event_time", session_start_hms, session_end_hms)
    elif "trade_day" not in legs.columns or legs["trade_day"].isna().all():
        legs["trade_day"] = legs["event_time"].dt.date

    legs["leg_pnl"] = pd.to_numeric(legs["realized_pnl_currency"], errors="coerce").fillna(0.0)
    legs["leg_sort"] = pd.to_numeric(legs["leg_index"], errors="coerce").fillna(0)

    return legs.sort_values(["trade_day", "event_time", "operation_id", "leg_sort"]).copy()


def _daily_metrics_from_results(df: pd.DataFrame, pnl_col: str, time_col: str = "trade_day") -> Dict[str, float]:
    """Core daily metrics from one PnL result per day."""
    if df.empty or pnl_col not in df.columns:
        return {}

    pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)
    dd = max_drawdown_from_pnl_sequence(df, pnl_col, time_col)

    return {
        "days": len(df),
        "total_pnl": pnl.sum(),
        "avg_day": pnl.mean(),
        "median_day": pnl.median(),
        "profit_factor": profit_factor_from_pnl(pnl),
        "winning_days_pct": (pnl > 0).mean() * 100,
        "losing_days_pct": (pnl < 0).mean() * 100,
        "flat_days_pct": (pnl == 0).mean() * 100,
        "best_day": pnl.max(),
        "worst_day": pnl.min(),
        "max_drawdown": dd.get("max_drawdown", np.nan),
        "ending_equity": dd.get("ending_equity", np.nan),
    }


def real_daily_from_legs(
    legs_df: pd.DataFrame,
    use_operational_day: bool = False,
    session_start_hms: str = "18:00:00",
    session_end_hms: str = "17:00:00",
) -> pd.DataFrame:
    """Real day result using the same source as the simulator: closed legs."""
    legs = _prepare_leg_timeline(legs_df, use_operational_day, session_start_hms, session_end_hms)
    if legs.empty:
        return pd.DataFrame()

    real = legs.groupby("trade_day", as_index=False).agg(
        real_day_pnl=("leg_pnl", "sum"),
        real_legs=("leg_pnl", "count"),
        real_operations_touched=("operation_id", "nunique"),
        first_time=("event_time", "min"),
        last_time=("event_time", "max"),
    )

    if "month" in legs.columns:
        months = legs.groupby("trade_day", as_index=False).agg(month=("month", "first"))
        real = real.merge(months, on="trade_day", how="left")
    else:
        real["month"] = ""

    return real.sort_values("trade_day")


def simulate_daily_stop(
    legs_df: pd.DataFrame,
    daily_target: float,
    daily_loss: float,
    flat_at_limits: bool = True,
    use_operational_day: bool = False,
    session_start_hms: str = "18:00:00",
    session_end_hms: str = "17:00:00",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Classic daily stop using real legs.

    The simulator consumes closed legs chronologically. After every closed leg, it checks
    whether the accumulated realized PnL touched the daily target/loss.

    If flat_at_limits=True, the result is capped exactly at +target / -loss.
    If flat_at_limits=False, the result uses the actual closed-leg amount, for example
    -615 when the configured max loss was -600.
    """
    rows = []
    legs = _prepare_leg_timeline(legs_df, use_operational_day, session_start_hms, session_end_hms)
    if legs.empty:
        return pd.DataFrame(), {}

    real_daily = real_daily_from_legs(legs_df, use_operational_day, session_start_hms, session_end_hms)
    real_lookup = real_daily.set_index("trade_day") if not real_daily.empty else pd.DataFrame()

    for trade_day, day_df in legs.groupby("trade_day", dropna=False):
        running = 0.0
        reason = "End of day"
        stopped_after_operation = None
        stopped_after_leg = None
        stopped_at_time = pd.NaT
        legs_used = 0
        touched_ops = set()
        raw_pnl_at_stop = 0.0

        for _, row in day_df.iterrows():
            running += float(row.get("leg_pnl", 0.0) or 0.0)
            raw_pnl_at_stop = running
            stopped_after_operation = row.get("operation_id")
            stopped_after_leg = row.get("leg_index")
            stopped_at_time = row.get("event_time")
            legs_used += 1
            if pd.notna(row.get("operation_id")):
                touched_ops.add(row.get("operation_id"))

            if running >= daily_target:
                reason = "Meta tocada"
                if flat_at_limits:
                    running = daily_target
                break
            if running <= -daily_loss:
                reason = "Pérdida tocada"
                if flat_at_limits:
                    running = -daily_loss
                break

        month_value = day_df["month"].dropna().iloc[0] if "month" in day_df.columns and not day_df["month"].dropna().empty else ""
        real_pnl = real_lookup.loc[trade_day, "real_day_pnl"] if not real_lookup.empty and trade_day in real_lookup.index else day_df["leg_pnl"].sum()
        real_legs = real_lookup.loc[trade_day, "real_legs"] if not real_lookup.empty and trade_day in real_lookup.index else len(day_df)
        real_ops = real_lookup.loc[trade_day, "real_operations_touched"] if not real_lookup.empty and trade_day in real_lookup.index else day_df["operation_id"].nunique()

        rows.append(
            {
                "month": month_value,
                "trade_day": trade_day,
                "real_day_pnl": float(real_pnl),
                "simulated_day_pnl": float(running),
                "difference": float(running) - float(real_pnl),
                "raw_pnl_at_stop": float(raw_pnl_at_stop),
                "result": "Win" if running > 0 else "Loss" if running < 0 else "Flat",
                "stop_reason": reason,
                "legs_used": legs_used,
                "operations_touched": len(touched_ops),
                "stopped_after_operation": stopped_after_operation,
                "stopped_after_leg": stopped_after_leg,
                "stopped_at_time": stopped_at_time,
                "real_legs": int(real_legs),
                "real_operations_touched": int(real_ops),
                "legs_skipped": int(real_legs) - int(legs_used),
                "operations_cut": int(real_ops) - len(touched_ops),
            }
        )

    sim = pd.DataFrame(rows).sort_values("trade_day")
    metrics = _daily_metrics_from_results(sim, "simulated_day_pnl", "trade_day")
    metrics.update(
        {
            "target_days_pct": (sim["stop_reason"] == "Meta tocada").mean() * 100,
            "loss_days_pct": (sim["stop_reason"] == "Pérdida tocada").mean() * 100,
            "open_days_pct": (sim["stop_reason"] == "End of day").mean() * 100,
            "avg_legs_used": sim["legs_used"].mean(),
            "days_changed": int((sim["real_day_pnl"].round(8) != sim["simulated_day_pnl"].round(8)).sum()),
            "legs_skipped": int(sim["legs_skipped"].clip(lower=0).sum()),
        }
    )
    return sim, metrics


def simulate_daily_sets(
    legs_df: pd.DataFrame,
    set_target: float,
    set_loss: float,
    use_operational_day: bool = False,
    session_start_hms: str = "18:00:00",
    session_end_hms: str = "17:00:00",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Daily set simulator using legs.

    A set consumes real legs in chronological order. When the running set PnL
    reaches target/loss, that set closes and the next leg starts a new set.
    """
    rows = []
    legs = _prepare_leg_timeline(legs_df, use_operational_day, session_start_hms, session_end_hms)
    if legs.empty:
        return pd.DataFrame(), {}

    for trade_day, day_df in legs.groupby("trade_day", dropna=False):
        set_number = 1
        running = 0.0
        raw_running = 0.0
        legs_used = 0
        touched_ops = set()
        set_start_operation = None
        set_start_leg = None
        set_start_time = pd.NaT

        for _, row in day_df.iterrows():
            if legs_used == 0:
                running = 0.0
                raw_running = 0.0
                touched_ops = set()
                set_start_operation = row.get("operation_id")
                set_start_leg = row.get("leg_index")
                set_start_time = row.get("event_time")

            leg_pnl = float(row.get("leg_pnl", 0.0) or 0.0)
            running += leg_pnl
            raw_running = running
            legs_used += 1
            if pd.notna(row.get("operation_id")):
                touched_ops.add(row.get("operation_id"))

            outcome = None
            result = None
            if running >= set_target:
                outcome = "Set target reached"
                result = set_target
            elif running <= -set_loss:
                outcome = "Set loss reached"
                result = -set_loss

            if outcome:
                month_value = day_df["month"].dropna().iloc[0] if "month" in day_df.columns and not day_df["month"].dropna().empty else ""
                rows.append(
                    {
                        "month": month_value,
                        "trade_day": trade_day,
                        "set_number": set_number,
                        "set_result": result,
                        "raw_pnl_at_set_close": raw_running,
                        "set_outcome": outcome,
                        "legs_used": legs_used,
                        "operations_touched": len(touched_ops),
                        "start_operation": set_start_operation,
                        "start_leg": set_start_leg,
                        "end_operation": row.get("operation_id"),
                        "end_leg": row.get("leg_index"),
                        "set_start_time": set_start_time,
                        "set_end_time": row.get("event_time"),
                    }
                )
                set_number += 1
                running = 0.0
                raw_running = 0.0
                legs_used = 0
                touched_ops = set()
                set_start_operation = None
                set_start_leg = None
                set_start_time = pd.NaT

        if legs_used > 0:
            month_value = day_df["month"].dropna().iloc[0] if "month" in day_df.columns and not day_df["month"].dropna().empty else ""
            last = day_df.iloc[-1]
            rows.append(
                {
                    "month": month_value,
                    "trade_day": trade_day,
                    "set_number": set_number,
                    "set_result": running,
                    "raw_pnl_at_set_close": raw_running,
                    "set_outcome": "End of day",
                    "legs_used": legs_used,
                    "operations_touched": len(touched_ops),
                    "start_operation": set_start_operation,
                    "start_leg": set_start_leg,
                    "end_operation": last.get("operation_id"),
                    "end_leg": last.get("leg_index"),
                    "set_start_time": set_start_time,
                    "set_end_time": last.get("event_time"),
                }
            )

    sets = pd.DataFrame(rows)
    metrics = {
        "sets": len(sets),
        "total_pnl": sets["set_result"].sum() if not sets.empty else np.nan,
        "avg_set": sets["set_result"].mean() if not sets.empty else np.nan,
        "target_sets_pct": (sets["set_outcome"] == "Set target reached").mean() * 100 if not sets.empty else np.nan,
        "loss_sets_pct": (sets["set_outcome"] == "Set loss reached").mean() * 100 if not sets.empty else np.nan,
        "open_sets_pct": (sets["set_outcome"] == "End of day").mean() * 100 if not sets.empty else np.nan,
        "avg_legs_per_set": sets["legs_used"].mean() if not sets.empty else np.nan,
        "avg_operations_per_set": sets["operations_touched"].mean() if not sets.empty else np.nan,
        "best_set": sets["set_result"].max() if not sets.empty else np.nan,
        "worst_set": sets["set_result"].min() if not sets.empty else np.nan,
    }
    return sets, metrics



def render_real_vs_sim_daily_cards(real_m: Dict[str, float], sim_m: Dict[str, float]):
    """Simple dashboard-style cards for daily simulator: real vs what-if."""
    st.markdown("### Comparación rápida · Real vs Simulado")
    st.caption("Real = lo que pasó con todas las piernas. Simulado = lo que habría pasado cortando el día en la meta/pérdida configurada.")

    row1 = st.columns(4)
    with row1[0]:
        card("PnL Real → Simulado", f"{fmt_money(real_m.get('total_pnl', np.nan))} → {fmt_money(sim_m.get('total_pnl', np.nan))}")
    with row1[1]:
        real_pf = real_m.get('profit_factor', np.nan)
        sim_pf = sim_m.get('profit_factor', np.nan)
        card("Profit Factor", f"{real_pf:.2f} → {sim_pf:.2f}" if pd.notna(real_pf) and pd.notna(sim_pf) else "-")
    with row1[2]:
        card("Días Ganadores", f"{fmt_pct(real_m.get('winning_days_pct', np.nan))} → {fmt_pct(sim_m.get('winning_days_pct', np.nan))}")
    with row1[3]:
        card("Días Perdidos", f"{fmt_pct(real_m.get('losing_days_pct', np.nan))} → {fmt_pct(sim_m.get('losing_days_pct', np.nan))}")

    row2 = st.columns(4)
    with row2[0]:
        card("Caída Máxima", f"{fmt_money(real_m.get('max_drawdown', np.nan))} → {fmt_money(sim_m.get('max_drawdown', np.nan))}")
    with row2[1]:
        card("Mejor Día", f"{fmt_money(real_m.get('best_day', np.nan))} → {fmt_money(sim_m.get('best_day', np.nan))}")
    with row2[2]:
        card("Peor Día", f"{fmt_money(real_m.get('worst_day', np.nan))} → {fmt_money(sim_m.get('worst_day', np.nan))}")
    with row2[3]:
        card("Días Cambiados", f"{int(sim_m.get('days_changed', 0))}")


def render_daily_sim_comparison_chart(daily_df: pd.DataFrame):
    """Interactive chart comparing real day result vs daily-stop simulation."""
    if daily_df.empty:
        return

    chart_df = daily_df.copy()
    chart_df["trade_day_dt"] = pd.to_datetime(chart_df["trade_day"], errors="coerce")
    chart_df = chart_df.dropna(subset=["trade_day_dt"]).sort_values("trade_day_dt")
    chart_df["fecha"] = chart_df["trade_day_dt"].dt.strftime("%Y-%m-%d")

    if chart_df.empty:
        return

    if go is not None:
        custom = chart_df[[
            "fecha", "real_day_pnl", "simulated_day_pnl", "difference", "stop_reason",
            "raw_pnl_at_stop", "legs_used", "real_legs", "legs_skipped",
            "stopped_after_operation", "stopped_after_leg",
        ]].to_numpy()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=chart_df["trade_day_dt"],
            y=chart_df["real_day_pnl"],
            name="Real",
            customdata=custom,
            hovertemplate=(
                "<b>Día:</b> %{customdata[0]}<br>"
                "<b>PnL Real:</b> $%{customdata[1]:,.2f}<br>"
                "<b>PnL Simulado:</b> $%{customdata[2]:,.2f}<br>"
                "<b>Diferencia:</b> $%{customdata[3]:,.2f}<br>"
                "<extra></extra>"
            ),
        ))
        fig.add_trace(go.Bar(
            x=chart_df["trade_day_dt"],
            y=chart_df["simulated_day_pnl"],
            name="Simulado con meta/loss",
            customdata=custom,
            hovertemplate=(
                "<b>Día:</b> %{customdata[0]}<br>"
                "<b>PnL Simulado:</b> $%{customdata[2]:,.2f}<br>"
                "<b>Motivo:</b> %{customdata[4]}<br>"
                "<b>PnL al tocar nivel:</b> $%{customdata[5]:,.2f}<br>"
                "<b>Piernas usadas:</b> %{customdata[6]} de %{customdata[7]}<br>"
                "<b>Piernas ignoradas:</b> %{customdata[8]}<br>"
                "<b>Última operación:</b> %{customdata[9]}<br>"
                "<b>Última pierna:</b> %{customdata[10]}<br>"
                "<extra></extra>"
            ),
        ))
        fig.add_hline(y=0, line_width=1)
        fig.update_layout(
            title="Resultado diario · Real vs Simulado",
            xaxis_title="Día",
            yaxis_title="PnL",
            barmode="group",
            hovermode="x unified",
            height=420,
            margin=dict(l=10, r=10, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(chart_df["trade_day_dt"], chart_df["real_day_pnl"], label="Real", alpha=0.65)
        ax.bar(chart_df["trade_day_dt"], chart_df["simulated_day_pnl"], label="Simulado", alpha=0.65)
        ax.axhline(0, linewidth=1)
        ax.set_title("Resultado diario · Real vs Simulado")
        ax.set_ylabel("PnL")
        format_date_axis(ax)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)


def monthly_daily_stop_summary(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()

    rows = []
    for month, month_df in daily_df.groupby("month", dropna=False):
        real_m = _daily_metrics_from_results(month_df, "real_day_pnl", "trade_day")
        sim_m = _daily_metrics_from_results(month_df, "simulated_day_pnl", "trade_day")
        rows.append({
            "month": month,
            "mes": month_label_es(month),
            "dias": len(month_df),
            "pnl_real": real_m.get("total_pnl", np.nan),
            "pnl_simulado": sim_m.get("total_pnl", np.nan),
            "diferencia": sim_m.get("total_pnl", np.nan) - real_m.get("total_pnl", np.nan),
            "pf_real": real_m.get("profit_factor", np.nan),
            "pf_simulado": sim_m.get("profit_factor", np.nan),
            "win_rate_real": real_m.get("winning_days_pct", np.nan),
            "win_rate_simulado": sim_m.get("winning_days_pct", np.nan),
            "dias_perdida_real": real_m.get("losing_days_pct", np.nan),
            "dias_perdida_simulado": sim_m.get("losing_days_pct", np.nan),
            "caida_max_real": real_m.get("max_drawdown", np.nan),
            "caida_max_simulada": sim_m.get("max_drawdown", np.nan),
            "dias_meta": (month_df["stop_reason"] == "Meta tocada").mean() * 100,
            "dias_loss": (month_df["stop_reason"] == "Pérdida tocada").mean() * 100,
            "dias_cambiados": int((month_df["real_day_pnl"].round(8) != month_df["simulated_day_pnl"].round(8)).sum()),
        })
    return pd.DataFrame(rows).sort_values("month")


def render_monthly_daily_stop_chart(monthly_df: pd.DataFrame):
    if monthly_df.empty:
        return
    if go is not None:
        plot_df = monthly_df.copy()
        if "mes" not in plot_df.columns:
            plot_df["mes"] = plot_df["month"].apply(month_label_es)
        custom = plot_df[[
            "mes", "pnl_real", "pnl_simulado", "diferencia", "pf_real", "pf_simulado",
            "win_rate_real", "win_rate_simulado", "caida_max_real", "caida_max_simulada",
            "dias_meta", "dias_loss", "dias_cambiados",
        ]].to_numpy()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=plot_df["mes"], y=plot_df["pnl_real"], name="Real", customdata=custom,
            hovertemplate=(
                "<b>Mes:</b> %{customdata[0]}<br>"
                "<b>PnL Real:</b> $%{customdata[1]:,.2f}<br>"
                "<b>PF Real:</b> %{customdata[4]:.2f}<br>"
                "<b>Días ganadores Real:</b> %{customdata[6]:.1f}%<br>"
                "<b>Caída máxima Real:</b> $%{customdata[8]:,.2f}<br>"
                "<extra></extra>"
            ),
        ))
        fig.add_trace(go.Bar(
            x=plot_df["mes"], y=plot_df["pnl_simulado"], name="Simulado", customdata=custom,
            hovertemplate=(
                "<b>Mes:</b> %{customdata[0]}<br>"
                "<b>PnL Simulado:</b> $%{customdata[2]:,.2f}<br>"
                "<b>Diferencia:</b> $%{customdata[3]:,.2f}<br>"
                "<b>PF Simulado:</b> %{customdata[5]:.2f}<br>"
                "<b>Días ganadores Simulado:</b> %{customdata[7]:.1f}%<br>"
                "<b>Caída máxima Simulada:</b> $%{customdata[9]:,.2f}<br>"
                "<b>Días meta:</b> %{customdata[10]:.1f}%<br>"
                "<b>Días pérdida:</b> %{customdata[11]:.1f}%<br>"
                "<b>Días cambiados:</b> %{customdata[12]}<br>"
                "<extra></extra>"
            ),
        ))
        fig.add_hline(y=0, line_width=1)
        fig.update_layout(
            title="Resultado por mes · Real vs Simulado",
            xaxis_title="Mes",
            yaxis_title="PnL",
            barmode="group",
            hovermode="x unified",
            height=420,
            margin=dict(l=10, r=10, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(monthly_df, use_container_width=True)


def calcular_resultado_simulado_por_cap_reversal(
    op_row: pd.Series,
    legs_df: pd.DataFrame,
    max_reversal_permitido: int,
) -> Dict[str, object]:
    """
    Simula el resultado de una operación si el bot NO pudiera pasar del reversal seleccionado.

    Regla importante:
    - Base entry = reversal_number 0.
    - Reversal 1 = reversal_number 1.
    - Reversal 2 = reversal_number 2.
    - Si seleccionas 3 y una operación real llegó a 4, significa que el reversal 3 perdió.
      Por eso el resultado simulado debe ser el PnL acumulado después de cerrar el reversal 3,
      no el resultado final real de la operación.
    """
    real_pnl = float(op_row.get("sequence_net_pnl_currency", 0) or 0)
    real_rev = int(op_row.get("reversal_count", 0) or 0)
    operation_id = str(op_row.get("operation_id", ""))

    real_end_time = op_row.get("sequence_ended_at", op_row.get("sequence_started_at", pd.NaT))

    # If the real operation did not exceed the selected max reversal, keep the real final result.
    if real_rev <= max_reversal_permitido:
        return {
            "reversal_count_simulado": real_rev,
            "sequence_net_pnl_simulado": real_pnl,
            "sequence_end_reason_simulado": op_row.get("sequence_end_reason", "Resultado real"),
            "simulated_stop_time": real_end_time,
            "cap_aplicado": False,
            "resultado_cambiado": False,
            "sim_detail": "No se corta: la operación real no pasó del reversal seleccionado.",
        }

    legs_op = legs_df[legs_df["operation_id"].astype(str) == operation_id].copy() if not legs_df.empty else pd.DataFrame()

    if legs_op.empty or "reversal_number" not in legs_op.columns:
        return {
            "reversal_count_simulado": max_reversal_permitido,
            "sequence_net_pnl_simulado": real_pnl,
            "sequence_end_reason_simulado": f"SIN_LEGS_PARA_SIMULAR_{max_reversal_permitido}",
            "simulated_stop_time": real_end_time,
            "cap_aplicado": True,
            "resultado_cambiado": False,
            "sim_detail": "Se debía cortar, pero no hay piernas suficientes; se mantiene el resultado real.",
        }

    legs_op = legs_op.copy()
    legs_op["reversal_number_int"] = pd.to_numeric(legs_op["reversal_number"], errors="coerce").fillna(-1).astype(int)
    legs_op["leg_index_num"] = pd.to_numeric(legs_op.get("leg_index", np.nan), errors="coerce")
    legs_op = legs_op.sort_values(["leg_index_num", "exit_time"], na_position="last")

    # The selected max reversal means: allow that reversal to close, then stop if the real trade continued deeper.
    target_leg = legs_op[legs_op["reversal_number_int"] == int(max_reversal_permitido)].copy()

    if target_leg.empty:
        return {
            "reversal_count_simulado": max_reversal_permitido,
            "sequence_net_pnl_simulado": real_pnl,
            "sequence_end_reason_simulado": f"SIN_PIERNA_REV_{max_reversal_permitido}",
            "simulated_stop_time": real_end_time,
            "cap_aplicado": True,
            "resultado_cambiado": False,
            "sim_detail": "Se debía cortar, pero no existe la pierna del reversal seleccionado; se mantiene el resultado real.",
        }

    last_allowed_leg = target_leg.sort_values(["leg_index_num", "exit_time"], na_position="last").iloc[-1]

    pnl_sim = last_allowed_leg.get("cumulative_sequence_pnl_after_leg", np.nan)
    if pd.isna(pnl_sim):
        # Fallback: sum realized PnL from the first leg up to the allowed leg.
        allowed_leg_index = last_allowed_leg.get("leg_index_num", np.nan)
        if pd.notna(allowed_leg_index) and "realized_pnl_currency" in legs_op.columns:
            pnl_sim = pd.to_numeric(
                legs_op.loc[legs_op["leg_index_num"] <= allowed_leg_index, "realized_pnl_currency"],
                errors="coerce",
            ).fillna(0).sum()
        else:
            pnl_sim = real_pnl

    stop_time = last_allowed_leg.get("exit_time", real_end_time)
    if pd.isna(stop_time):
        stop_time = real_end_time

    exit_reason = last_allowed_leg.get("exit_reason", "")
    if pd.isna(exit_reason) or str(exit_reason).strip() == "":
        exit_reason = f"CORTADO_DESPUES_REV_{max_reversal_permitido}"

    pnl_sim = float(pnl_sim)

    return {
        "reversal_count_simulado": max_reversal_permitido,
        "sequence_net_pnl_simulado": pnl_sim,
        "sequence_end_reason_simulado": str(exit_reason),
        "simulated_stop_time": stop_time,
        "cap_aplicado": True,
        "resultado_cambiado": abs(pnl_sim - real_pnl) > 0.000001,
        "sim_detail": (
            f"Cortada después de cerrar reversal {max_reversal_permitido}. "
            f"La operación real llegó a reversal {real_rev}."
        ),
    }


def aplicar_cap_reversal(ops_df: pd.DataFrame, legs_df: pd.DataFrame, max_reversal_permitido: int) -> pd.DataFrame:
    if ops_df.empty:
        return pd.DataFrame()

    base = ops_df.copy()
    resultados = base.apply(
        lambda row: calcular_resultado_simulado_por_cap_reversal(row, legs_df, max_reversal_permitido),
        axis=1,
    )
    resultados_df = pd.DataFrame(resultados.tolist(), index=base.index)
    out = pd.concat([base, resultados_df], axis=1)
    out["simulated_stop_time"] = pd.to_datetime(out.get("simulated_stop_time"), errors="coerce")
    return out


def profit_factor_from_pnl(pnl_series: pd.Series) -> float:
    pnl = pd.to_numeric(pnl_series, errors="coerce").dropna()
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    gross_profit = winners.sum()
    gross_loss = abs(losers.sum())
    return gross_profit / gross_loss if gross_loss > 0 else np.nan


def max_drawdown_from_pnl_sequence(df: pd.DataFrame, pnl_col: str, time_col: str) -> Dict[str, object]:
    """Calculates account max drawdown from a chronological sequence of PnL results."""
    if df.empty or pnl_col not in df.columns:
        return {
            "max_drawdown": np.nan,
            "peak_equity": np.nan,
            "trough_equity": np.nan,
            "ending_equity": np.nan,
            "peak_time": pd.NaT,
            "trough_time": pd.NaT,
        }

    work = df.copy()
    work[pnl_col] = pd.to_numeric(work[pnl_col], errors="coerce").fillna(0)
    if time_col in work.columns:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.sort_values(time_col, na_position="last")
    else:
        work = work.reset_index(drop=True)

    work["equity_tmp"] = work[pnl_col].cumsum()
    work["peak_tmp"] = work["equity_tmp"].cummax()
    work["drawdown_tmp"] = work["equity_tmp"] - work["peak_tmp"]

    if work.empty:
        return {"max_drawdown": np.nan}

    trough_idx = work["drawdown_tmp"].idxmin()
    max_dd = float(work.loc[trough_idx, "drawdown_tmp"])
    trough_equity = float(work.loc[trough_idx, "equity_tmp"])
    peak_equity = float(work.loc[trough_idx, "peak_tmp"])

    prior = work.loc[:trough_idx].copy()
    peak_rows = prior[prior["equity_tmp"] == peak_equity]
    peak_idx = peak_rows.index[-1] if not peak_rows.empty else trough_idx

    return {
        "max_drawdown": max_dd,
        "peak_equity": peak_equity,
        "trough_equity": trough_equity,
        "ending_equity": float(work["equity_tmp"].iloc[-1]),
        "peak_time": work.loc[peak_idx, time_col] if time_col in work.columns else pd.NaT,
        "trough_time": work.loc[trough_idx, time_col] if time_col in work.columns else pd.NaT,
    }


def simulated_reversal_metrics(ops_df: pd.DataFrame, legs_df: pd.DataFrame, max_reversal_permitido: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    sim_df = aplicar_cap_reversal(ops_df, legs_df, max_reversal_permitido)
    if sim_df.empty:
        return sim_df, {}

    real_pnl_col = "sequence_net_pnl_currency"
    sim_pnl_col = "sequence_net_pnl_simulado"

    real_time_col = "sequence_ended_at" if "sequence_ended_at" in ops_df.columns else "sequence_started_at"
    sim_time_col = "simulated_stop_time"

    real_dd = max_drawdown_from_pnl_sequence(ops_df, real_pnl_col, real_time_col)
    sim_dd = max_drawdown_from_pnl_sequence(sim_df, sim_pnl_col, sim_time_col)

    metrics = {
        "real_pnl": pd.to_numeric(ops_df[real_pnl_col], errors="coerce").fillna(0).sum(),
        "sim_pnl": pd.to_numeric(sim_df[sim_pnl_col], errors="coerce").fillna(0).sum(),
        "real_pf": profit_factor_from_pnl(ops_df[real_pnl_col]),
        "sim_pf": profit_factor_from_pnl(sim_df[sim_pnl_col]),
        "real_max_dd": real_dd.get("max_drawdown", np.nan),
        "sim_max_dd": sim_dd.get("max_drawdown", np.nan),
        "ops_cortadas": int(sim_df["cap_aplicado"].sum()) if "cap_aplicado" in sim_df.columns else 0,
        "ops_resultado_cambiado": int(sim_df["resultado_cambiado"].sum()) if "resultado_cambiado" in sim_df.columns else 0,
        "worst_real_op": pd.to_numeric(ops_df[real_pnl_col], errors="coerce").min(),
        "worst_sim_op": pd.to_numeric(sim_df[sim_pnl_col], errors="coerce").min(),
    }
    metrics["pnl_diff"] = metrics["sim_pnl"] - metrics["real_pnl"]
    metrics["pf_diff"] = metrics["sim_pf"] - metrics["real_pf"] if pd.notna(metrics["sim_pf"]) and pd.notna(metrics["real_pf"]) else np.nan
    metrics["dd_diff"] = metrics["sim_max_dd"] - metrics["real_max_dd"] if pd.notna(metrics["sim_max_dd"]) and pd.notna(metrics["real_max_dd"]) else np.nan
    return sim_df, metrics


def monthly_simulated_reversal_summary(sim_df: pd.DataFrame) -> pd.DataFrame:
    if sim_df.empty or "month" not in sim_df.columns:
        return pd.DataFrame()

    rows = []
    for month, g in sim_df.groupby("month"):
        real_time_col = "sequence_ended_at" if "sequence_ended_at" in g.columns else "sequence_started_at"
        real_dd = max_drawdown_from_pnl_sequence(g, "sequence_net_pnl_currency", real_time_col)
        sim_dd = max_drawdown_from_pnl_sequence(g, "sequence_net_pnl_simulado", "simulated_stop_time")
        real_pnl = pd.to_numeric(g["sequence_net_pnl_currency"], errors="coerce").fillna(0).sum()
        sim_pnl = pd.to_numeric(g["sequence_net_pnl_simulado"], errors="coerce").fillna(0).sum()
        rows.append({
            "month": month,
            "mes": month_label_es(month),
            "operaciones": len(g),
            "pnl_real": real_pnl,
            "pnl_reversal_seleccionado": sim_pnl,
            "diferencia_pnl": sim_pnl - real_pnl,
            "profit_factor_real": profit_factor_from_pnl(g["sequence_net_pnl_currency"]),
            "profit_factor_reversal_seleccionado": profit_factor_from_pnl(g["sequence_net_pnl_simulado"]),
            "max_drawdown_real": real_dd.get("max_drawdown", np.nan),
            "max_drawdown_reversal_seleccionado": sim_dd.get("max_drawdown", np.nan),
            "operaciones_cortadas": int(g["cap_aplicado"].sum()) if "cap_aplicado" in g.columns else 0,
            "resultados_cambiados": int(g["resultado_cambiado"].sum()) if "resultado_cambiado" in g.columns else 0,
        })
    return pd.DataFrame(rows).sort_values("month")


def build_consecutive_loss_streaks(df: pd.DataFrame, pnl_col: str, time_col: str, id_col: str, unit_label: str) -> pd.DataFrame:
    """
    Builds consecutive-loss streaks in chronological order.
    This is the dashboard MAX dropdown Rene asked for: max continuous losing sequences.
    """
    if df.empty or pnl_col not in df.columns or time_col not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work = work[pd.notna(work[pnl_col])].copy()
    if work.empty:
        return pd.DataFrame()

    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.sort_values(time_col)

    streaks = []
    active_rows = []
    streak_no = 0

    def close_streak(rows):
        nonlocal streak_no
        if not rows:
            return

        streak_no += 1
        block = pd.DataFrame(rows)
        ids = block[id_col].astype(str).dropna().tolist() if id_col in block.columns else []
        start_time = block[time_col].min()
        end_time = block[time_col].max()
        total_loss = float(block[pnl_col].sum())
        worst_item = float(block[pnl_col].min())
        month = str(start_time.to_period("M")) if pd.notna(start_time) else "-"
        trade_day = start_time.date() if pd.notna(start_time) else None

        streaks.append(
            {
                "streak_no": streak_no,
                "unit": unit_label,
                "month": month,
                "trade_day_start": trade_day,
                "start_time": start_time,
                "end_time": end_time,
                "losses_in_a_row": len(block),
                "total_streak_loss": total_loss,
                "worst_single_loss": worst_item,
                "ids": ", ".join(ids[:8]) + (" ..." if len(ids) > 8 else ""),
            }
        )

    for _, row in work.iterrows():
        pnl = row[pnl_col]
        if pd.notna(pnl) and float(pnl) < 0:
            active_rows.append(row.to_dict())
        else:
            close_streak(active_rows)
            active_rows = []

    close_streak(active_rows)

    if not streaks:
        return pd.DataFrame()

    out = pd.DataFrame(streaks)
    return out.sort_values(["losses_in_a_row", "total_streak_loss"], ascending=[False, True])


# ============================================================


def build_account_drawdown_from_legs(legs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Build account-style max drawdown from real closed legs.

    Max drawdown = biggest continuous drop in accumulated account PnL
    from a previous equity high to the next equity low. This is measured
    in dollars and is based on legs, not full operations.
    """
    legs = _prepare_leg_timeline(legs_df)
    if legs.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    legs = legs.copy().reset_index(drop=True)
    legs["equity"] = legs["leg_pnl"].cumsum()

    running_peak = []
    peak = 0.0
    for value in legs["equity"]:
        peak = max(peak, float(value))
        running_peak.append(peak)
    legs["peak_equity"] = running_peak
    legs["drawdown"] = legs["equity"] - legs["peak_equity"]

    max_drawdown = float(legs["drawdown"].min())
    ending_drawdown = float(legs["drawdown"].iloc[-1])

    periods = []
    peak = 0.0
    peak_time = pd.NaT
    in_dd = False
    trough = 0.0
    trough_time = pd.NaT
    start_time = pd.NaT
    touched_ops = set()
    legs_count = 0

    def close_period(recovered_time):
        nonlocal in_dd, trough, trough_time, start_time, touched_ops, legs_count, peak, peak_time
        if not in_dd:
            return
        periods.append({
            "peak_time": peak_time if pd.notna(peak_time) else start_time,
            "trough_time": trough_time,
            "recovered_time": recovered_time,
            "peak_equity": peak,
            "trough_equity": trough,
            "max_drawdown": trough - peak,
            "legs_in_drawdown": legs_count,
            "operations_touched": len(touched_ops),
            "recovered": pd.notna(recovered_time),
        })
        in_dd = False
        trough = peak
        trough_time = pd.NaT
        start_time = pd.NaT
        touched_ops = set()
        legs_count = 0

    for _, row in legs.iterrows():
        equity = float(row["equity"])
        event_time = row.get("event_time")
        op_id = row.get("operation_id")

        if equity > peak:
            close_period(event_time)
            peak = equity
            peak_time = event_time
            continue

        if equity < peak:
            if not in_dd:
                in_dd = True
                start_time = event_time
                trough = equity
                trough_time = event_time
                touched_ops = set()
                legs_count = 0
            legs_count += 1
            if pd.notna(op_id):
                touched_ops.add(op_id)
            if equity < trough:
                trough = equity
                trough_time = event_time
        elif equity >= peak and in_dd:
            close_period(event_time)

    close_period(pd.NaT)

    dd_periods = pd.DataFrame(periods)
    if not dd_periods.empty:
        dd_periods = dd_periods.sort_values("max_drawdown", ascending=True)

    metrics = {
        "max_drawdown": max_drawdown,
        "ending_drawdown": ending_drawdown,
        "ending_equity": float(legs["equity"].iloc[-1]),
        "peak_equity": float(legs["peak_equity"].max()),
        "drawdown_periods": len(dd_periods),
    }
    return legs, dd_periods, metrics
# FILTERS
# ============================================================


def apply_global_filters(ops_df: pd.DataFrame, legs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros globales")

    filtered_ops = ops_df.copy()
    filtered_legs = legs_df.copy()

    months = sorted(filtered_ops["month"].dropna().unique().tolist())
    selected_months = st.sidebar.multiselect("Meses", months, default=months)
    if selected_months:
        filtered_ops = filtered_ops[filtered_ops["month"].isin(selected_months)]
        if not filtered_legs.empty and "month" in filtered_legs.columns:
            filtered_legs = filtered_legs[filtered_legs["month"].isin(selected_months)]

    instruments = sorted(filtered_ops["instrument"].dropna().unique().tolist())
    selected_instruments = st.sidebar.multiselect("Instrumentos", instruments, default=instruments)
    if selected_instruments:
        filtered_ops = filtered_ops[filtered_ops["instrument"].isin(selected_instruments)]
        if not filtered_legs.empty and "instrument" in filtered_legs.columns:
            filtered_legs = filtered_legs[filtered_legs["instrument"].isin(selected_instruments)]

    configs = sorted(filtered_ops["config_key"].dropna().unique().tolist())
    selected_configs = st.sidebar.multiselect("Configuraciones", configs, default=configs)
    if selected_configs:
        filtered_ops = filtered_ops[filtered_ops["config_key"].isin(selected_configs)]
        if not filtered_legs.empty and "config_key" in filtered_legs.columns:
            filtered_legs = filtered_legs[filtered_legs["config_key"].isin(selected_configs)]

    if not filtered_ops.empty:
        min_date = filtered_ops["sequence_started_at"].min().date()
        max_date = filtered_ops["sequence_started_at"].max().date()
        date_range = st.sidebar.date_input("Rango de fechas", value=(min_date, max_date))
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_ops = filtered_ops[
                (filtered_ops["sequence_started_at"].dt.date >= start_date)
                & (filtered_ops["sequence_started_at"].dt.date <= end_date)
            ]
            if not filtered_legs.empty and "sequence_started_at" in filtered_legs.columns:
                filtered_legs = filtered_legs[
                    (filtered_legs["sequence_started_at"].dt.date >= start_date)
                    & (filtered_legs["sequence_started_at"].dt.date <= end_date)
                ]

    return filtered_ops, filtered_legs


# ============================================================
# PAGES
# ============================================================


def render_dashboard_general(ops_df: pd.DataFrame, legs_df: pd.DataFrame):
    st.header("Dashboard General")
    section_note("Esta página responde: ¿el bot está sano, estable y consistente por mes y por día?")
    show_help(
        "Dashboard General",
        "Vista principal del bot. Mira primero esta página antes de tocar settings.",
        [
            "¿El PnL total es positivo?",
            "¿El resultado depende de un solo mes o un solo día?",
            "¿La peor operación o peor día son demasiado grandes?",
            "¿El profit factor justifica seguir probando esta configuración?",
        ],
    )

    if ops_df.empty:
        st.warning("No hay operaciones con los filtros actuales.")
        return

    # ========================================================
    # ========================================================
    # ========================================================
    # MAIN HEALTH CARDS + ACCOUNT MAX DRAWDOWN
    # ========================================================
    st.subheader("Salud General")

    # Max Drawdown must be based on real closed legs, because those are the
    # actual entries/exits that move the account equity.
    equity_curve, dd_periods, dd_metrics = build_account_drawdown_from_legs(legs_df)

    m = overview_metrics(ops_df)
    c = st.columns(5)
    with c[0]: card("Operaciones", f"{m['ops']}")
    with c[1]: card("Días", f"{m['days']}")
    with c[2]: card("PnL Total", fmt_money(m["pnl"]))
    with c[3]: card("Profit Factor", "-" if pd.isna(m["profit_factor"]) else f"{m['profit_factor']:.2f}")
    with c[4]: card("Max Drawdown", fmt_money(dd_metrics.get("max_drawdown", np.nan)))

    c = st.columns(4)
    with c[0]: card("Win Rate", fmt_pct(m["win_rate"]))
    with c[1]: card("Días Positivos", fmt_pct(m["positive_days_rate"]))
    with c[2]: card("Peor Operación", fmt_money(m["worst_op"]))
    with c[3]: card("Peor Día", fmt_money(m["worst_day"]))

    st.markdown("### Caída máxima de la cuenta")
    section_note(
        "Esto muestra cuánto bajó la cuenta después de estar en su mejor punto. "
        "La línea azul es la cuenta actual. La línea naranja es el mejor punto alcanzado. "
        "La distancia entre ambas es la caída de la cuenta."
    )

    if equity_curve.empty:
        st.info("No hay piernas suficientes para calcular Max Drawdown.")
    else:
        c = st.columns(4)
        with c[0]: card("Max Drawdown", fmt_money(dd_metrics.get("max_drawdown", np.nan)))
        with c[1]: card("Mejor Punto", fmt_money(dd_metrics.get("peak_equity", np.nan)))
        with c[2]: card("Cuenta Final", fmt_money(dd_metrics.get("ending_equity", np.nan)))
        with c[3]: card("Caída Actual", fmt_money(dd_metrics.get("ending_drawdown", np.nan)))

        curve = equity_curve.copy()
        curve["event_time"] = pd.to_datetime(curve["event_time"], errors="coerce")
        curve = curve.dropna(subset=["event_time"]).sort_values("event_time")
        curve["fecha"] = curve["event_time"].dt.strftime("%Y-%m-%d %H:%M")
        curve["caida"] = curve["equity"] - curve["peak_equity"]

        if go is not None and not curve.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=curve["event_time"],
                y=curve["peak_equity"],
                mode="lines",
                name="Mejor punto alcanzado",
                customdata=curve[["fecha", "peak_equity"]].to_numpy(),
                hovertemplate=(
                    "<b>Fecha:</b> %{customdata[0]}<br>"
                    "<b>Mejor punto:</b> $%{customdata[1]:,.2f}"
                    "<extra></extra>"
                ),
            ))
            fig.add_trace(go.Scatter(
                x=curve["event_time"],
                y=curve["equity"],
                mode="lines",
                name="Cuenta actual",
                fill="tonexty",
                customdata=curve[["fecha", "equity", "peak_equity", "caida"]].to_numpy(),
                hovertemplate=(
                    "<b>Fecha:</b> %{customdata[0]}<br>"
                    "<b>Cuenta actual:</b> $%{customdata[1]:,.2f}<br>"
                    "<b>Mejor punto:</b> $%{customdata[2]:,.2f}<br>"
                    "<b>Caída desde mejor punto:</b> $%{customdata[3]:,.2f}"
                    "<extra></extra>"
                ),
            ))
            fig.update_layout(
                title="Cuenta actual vs mejor punto alcanzado",
                xaxis_title="Fecha",
                yaxis_title="Ganancia acumulada",
                hovermode="x unified",
                height=420,
                margin=dict(l=40, r=25, t=55, b=40),
            )
            fig.update_xaxes(nticks=10, tickformat="%b %d")
            st.plotly_chart(fig, use_container_width=True)
        else:
            if go is None:
                st.warning("Plotly no está instalado en este entorno. El gráfico será estático. Para hover/interactividad agrega `plotly` en requirements.txt y reinicia la app.")
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(curve["event_time"], curve["equity"], label="Cuenta actual")
            ax.plot(curve["event_time"], curve["peak_equity"], label="Mejor punto alcanzado")
            ax.fill_between(
                curve["event_time"],
                curve["equity"],
                curve["peak_equity"],
                where=curve["equity"] < curve["peak_equity"],
                alpha=0.15,
            )
            ax.set_title("Cuenta actual vs mejor punto alcanzado")
            ax.set_ylabel("Ganancia acumulada")
            ax.set_xlabel("Fecha")
            ax.legend(loc="best")
            format_date_axis(ax, min_ticks=5, max_ticks=9, rotation=25)
            fig.tight_layout()
            st.pyplot(fig)

        if not dd_periods.empty:
            st.markdown("**Peores momentos donde la cuenta cayó**")
            st.dataframe(
                dd_periods[[
                    "peak_time", "trough_time", "recovered_time", "peak_equity", "trough_equity",
                    "max_drawdown", "legs_in_drawdown", "operations_touched", "recovered",
                ]].head(10),
                use_container_width=True,
            )

    st.markdown("---")
    section_note(
        "El Dashboard queda solo para salud general y caída máxima de la cuenta. "
        "Los análisis por sesión, mes, día y hora están ahora organizados en Tiempo y Sesiones."
    )

    lines = []
    if m["pnl"] > 0:
        lines.append("El bot está positivo con los filtros actuales.")
    else:
        lines.append("El bot está negativo con los filtros actuales.")
    if not pd.isna(m["profit_factor"]):
        if m["profit_factor"] >= 1.5:
            lines.append("El profit factor está en una zona saludable.")
        elif m["profit_factor"] >= 1.0:
            lines.append("El profit factor es positivo, pero todavía necesita control de riesgo.")
        else:
            lines.append("El profit factor está débil; hay que revisar filtros o reversals.")
    if abs(m["worst_op"]) > max(abs(m["avg_pnl"]) * 10, 1):
        lines.append("La peor operación es grande comparada con el promedio; revisar Risk Killers y Motor de Reversiones.")
    show_conclusion("Dashboard General", lines)


def render_summary_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    title: str,
    x_title: str,
    sort_col: str | None = None,
    category_order: List | None = None,
):
    """Interactive summary bar used in Tiempo y Sesiones."""
    if df.empty or x_col not in df.columns or "pnl_total" not in df.columns:
        return

    chart_df = df.copy()

    if category_order is not None:
        order_map = {v: i for i, v in enumerate(category_order)}
        chart_df["_order"] = chart_df[x_col].map(order_map).fillna(999)
        chart_df = chart_df.sort_values("_order").drop(columns=["_order"])
    elif sort_col and sort_col in chart_df.columns:
        chart_df = chart_df.sort_values(sort_col)

    custom_cols = []
    hover_lines = [f"<b>{x_title}:</b> %{{x}}", "<b>PnL:</b> $%{y:,.2f}"]

    optional_cols = [
        ("operaciones", "Operaciones", "int"),
        ("tasa_acierto", "Win rate", "pct"),
        ("profit_factor", "Profit factor", "num"),
        ("pnl_promedio", "PnL promedio", "money"),
        ("peor_operacion", "Peor operación", "money"),
        ("mejor_operacion", "Mejor operación", "money"),
        ("drawdown_promedio", "Drawdown promedio", "money"),
        ("drawdown_max", "Mayor drawdown", "money"),
        ("reversiones_promedio", "Reversals promedio", "num"),
        ("contratos_max", "Máx contratos", "num"),
    ]

    for col, label, kind in optional_cols:
        if col in chart_df.columns:
            custom_cols.append(col)
            idx = len(custom_cols) - 1
            if kind == "pct":
                hover_lines.append(f"<b>{label}:</b> %{{customdata[{idx}]:.1f}}%")
            elif kind == "money":
                hover_lines.append(f"<b>{label}:</b> $%{{customdata[{idx}]:,.2f}}")
            elif kind == "int":
                hover_lines.append(f"<b>{label}:</b> %{{customdata[{idx}]:.0f}}")
            else:
                hover_lines.append(f"<b>{label}:</b> %{{customdata[{idx}]:.2f}}")

    hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

    if go is not None:
        fig = go.Figure()
        fig.add_bar(
            x=chart_df[x_col],
            y=chart_df["pnl_total"],
            customdata=chart_df[custom_cols].to_numpy() if custom_cols else None,
            hovertemplate=hovertemplate,
            name="PnL",
        )
        fig.add_hline(y=0, line_width=1)
        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title="PnL",
            hovermode="closest",
            height=420,
            margin=dict(l=40, r=25, t=55, b=55),
        )
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, use_container_width=True)
        return

    st.warning("Plotly no está instalado. El gráfico será estático y no tendrá hover.")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(chart_df[x_col].astype(str), chart_df["pnl_total"])
    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(x_title)
    ax.set_ylabel("PnL")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    st.pyplot(fig)


def render_month_grouped_bar(
    df: pd.DataFrame,
    x_col: str,
    title: str,
    x_title: str,
    category_order: List | None = None,
):
    """Interactive grouped bar by month for time/session stability."""
    if df.empty or "month" not in df.columns or x_col not in df.columns:
        return

    chart_df = df.copy()
    if category_order is not None:
        order_map = {v: i for i, v in enumerate(category_order)}
        chart_df["_order"] = chart_df[x_col].map(order_map).fillna(999)
        chart_df = chart_df.sort_values(["_order", "month"]).drop(columns=["_order"])
    else:
        chart_df = chart_df.sort_values([x_col, "month"])

    if go is None:
        st.warning("Plotly no está instalado. El gráfico por mes será tabla solamente.")
        return

    custom_cols = ["operaciones", "tasa_acierto", "profit_factor", "pnl_promedio", "peor_operacion", "mejor_operacion"]
    custom_cols = [c for c in custom_cols if c in chart_df.columns]

    fig = go.Figure()
    for month, month_df in chart_df.groupby("month"):
        fig.add_bar(
            x=month_df[x_col],
            y=month_df["pnl_total"],
            name=str(month),
            customdata=month_df[custom_cols].to_numpy() if custom_cols else None,
            hovertemplate=(
                f"<b>Mes:</b> {month}<br>"
                f"<b>{x_title}:</b> %{{x}}<br>"
                "<b>PnL:</b> $%{y:,.2f}<br>"
                "<b>Operaciones:</b> %{customdata[0]:.0f}<br>"
                "<b>Win rate:</b> %{customdata[1]:.1f}%<br>"
                "<b>Profit factor:</b> %{customdata[2]:.2f}<br>"
                "<b>PnL promedio:</b> $%{customdata[3]:,.2f}<br>"
                "<b>Peor operación:</b> $%{customdata[4]:,.2f}<br>"
                "<b>Mejor operación:</b> $%{customdata[5]:,.2f}"
                "<extra></extra>"
            ),
        )

    fig.add_hline(y=0, line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="PnL",
        barmode="group",
        hovermode="closest",
        height=430,
        margin=dict(l=40, r=25, t=55, b=55),
    )
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)


def render_tiempo_y_sesiones(ops_df: pd.DataFrame):
    st.header("Tiempo y Sesiones")
    section_note(
        "Esta página concentra todo lo relacionado con tiempo: sesión, mes, día y hora. "
        "La idea es ver primero el mapa grande y después bajar al detalle."
    )
    show_help(
        "Tiempo y Sesiones",
        "Análisis temporal del bot. Aquí buscamos qué momentos tienen edge y qué momentos conviene bloquear.",
        [
            "¿Qué sesión es más fuerte o más peligrosa?",
            "¿Qué mes está sosteniendo o dañando el resultado?",
            "¿Qué días concretos fueron los peores/mejores?",
            "¿Qué día de semana u hora debería evitarse?",
        ],
    )

    if ops_df.empty:
        st.warning("No hay operaciones con los filtros actuales.")
        return

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_labels = {
        "Monday": "Lunes",
        "Tuesday": "Martes",
        "Wednesday": "Miércoles",
        "Thursday": "Jueves",
        "Friday": "Viernes",
        "Saturday": "Sábado",
        "Sunday": "Domingo",
    }
    session_order = ["Asia", "Londres", "NY Open", "NY Midday", "NY Late", "Fuera de Sesión", "Sin sesión"]

    # ========================================================
    # 1. SESSION FIRST
    # ========================================================
    st.subheader("1. Sesión")
    section_note("Primero vemos el mapa grande: qué región/sesión aporta o quita más dinero.")

    by_session = aggregate_core(ops_df, ["sesion"])
    by_session["_order"] = by_session["sesion"].apply(lambda x: session_order.index(x) if x in session_order else 99)
    by_session = by_session.sort_values("_order").drop(columns=["_order"])
    render_summary_bar_chart(
        by_session,
        x_col="sesion",
        title="Resultado por sesión",
        x_title="Sesión",
        category_order=session_order,
    )
    with st.expander("Ver tabla por sesión", expanded=False):
        st.dataframe(by_session, use_container_width=True)

    if ops_df["month"].nunique() > 1:
        st.markdown("**Sesión por mes**")
        st.caption("Esto ayuda a ver si una sesión fue buena solo en un mes o si se mantiene fuerte en varios meses.")
        session_month = aggregate_core(ops_df, ["month", "sesion"])
        session_month["_order"] = session_month["sesion"].apply(lambda x: session_order.index(x) if x in session_order else 99)
        session_month = session_month.sort_values(["_order", "month"]).drop(columns=["_order"])
        render_month_grouped_bar(
            session_month,
            x_col="sesion",
            title="Resultado por sesión y mes",
            x_title="Sesión",
            category_order=session_order,
        )
        with st.expander("Ver tabla de sesión por mes", expanded=False):
            st.dataframe(session_month, use_container_width=True)

    # ========================================================
    # 2. MONTH
    # ========================================================
    st.subheader("2. Mes")
    section_note("Después vemos si el resultado es estable entre meses o si un solo mes está escondiendo el riesgo.")

    month_df = monthly_summary(ops_df)
    if not month_df.empty:
        render_monthly_result_chart(month_df)
        with st.expander("Ver tabla mensual detallada", expanded=False):
            st.dataframe(month_df, use_container_width=True)
    else:
        st.info("No hay datos mensuales para mostrar.")

    # ========================================================
    # 3. DAY
    # ========================================================
    st.subheader("3. Día")
    section_note("Aquí bajamos al día concreto para detectar días que hicieron daño o días que sostuvieron el resultado.")

    daily = daily_summary(ops_df)
    if not daily.empty:
        render_clean_daily_pnl_chart(daily)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Peores días**")
            st.dataframe(daily.sort_values("pnl_total", ascending=True).head(10), use_container_width=True)
        with c2:
            st.markdown("**Mejores días**")
            st.dataframe(daily.sort_values("pnl_total", ascending=False).head(10), use_container_width=True)

        with st.expander("Ver tabla diaria completa", expanded=False):
            st.dataframe(daily.sort_values("trade_day", ascending=False), use_container_width=True)
    else:
        st.info("No hay datos diarios para mostrar.")

    st.markdown("**Día de semana**")
    weekday = aggregate_core(ops_df, ["dia_semana"])
    weekday["dia_semana_label"] = weekday["dia_semana"].map(weekday_labels).fillna(weekday["dia_semana"].astype(str))
    weekday["_order"] = weekday["dia_semana"].apply(lambda x: weekday_order.index(x) if x in weekday_order else 99)
    weekday = weekday.sort_values("_order").drop(columns=["_order"])
    render_summary_bar_chart(
        weekday,
        x_col="dia_semana_label",
        title="Resultado por día de semana",
        x_title="Día",
        category_order=[weekday_labels[d] for d in weekday_order],
    )
    with st.expander("Ver tabla por día de semana", expanded=False):
        st.dataframe(weekday.drop(columns=["dia_semana_label"], errors="ignore"), use_container_width=True)

    if ops_df["month"].nunique() > 1:
        st.markdown("**Día de semana por mes**")
        weekday_month = aggregate_core(ops_df, ["month", "dia_semana"])
        weekday_month["dia_semana_label"] = weekday_month["dia_semana"].map(weekday_labels).fillna(weekday_month["dia_semana"].astype(str))
        weekday_month["_order"] = weekday_month["dia_semana"].apply(lambda x: weekday_order.index(x) if x in weekday_order else 99)
        weekday_month = weekday_month.sort_values(["_order", "month"]).drop(columns=["_order"])
        render_month_grouped_bar(
            weekday_month,
            x_col="dia_semana_label",
            title="Resultado por día de semana y mes",
            x_title="Día",
            category_order=[weekday_labels[d] for d in weekday_order],
        )
        with st.expander("Ver tabla de día de semana por mes", expanded=False):
            st.dataframe(weekday_month.drop(columns=["dia_semana_label"], errors="ignore"), use_container_width=True)

    # ========================================================
    # 4. HOUR LAST
    # ========================================================
    st.subheader("4. Hora")
    section_note("Por último miramos la hora. Este es el nivel más fino para decidir si bloquear ventanas específicas.")

    by_hour = aggregate_core(ops_df, ["hora_inicio"]).sort_values("hora_inicio")
    by_hour["hora_label"] = by_hour["hora_inicio"].apply(lambda h: f"{int(h):02d}:00" if pd.notna(h) else "Sin hora")
    render_summary_bar_chart(
        by_hour,
        x_col="hora_label",
        title="Resultado por hora",
        x_title="Hora",
        sort_col="hora_inicio",
    )
    with st.expander("Ver tabla por hora", expanded=False):
        st.dataframe(by_hour.drop(columns=["hora_label"], errors="ignore"), use_container_width=True)

    lines = []
    if not by_session.empty:
        best_s = by_session.sort_values("pnl_total", ascending=False).iloc[0]
        worst_s = by_session.sort_values("pnl_total", ascending=True).iloc[0]
        lines.append(f"Mejor sesión: {best_s['sesion']}.")
        lines.append(f"Sesión más débil: {worst_s['sesion']}.")
    if not month_df.empty:
        best_m = month_df.sort_values("pnl_total", ascending=False).iloc[0]
        worst_m = month_df.sort_values("pnl_total", ascending=True).iloc[0]
        lines.append(f"Mejor mes: {best_m['month']}.")
        lines.append(f"Mes más débil: {worst_m['month']}.")
    if not daily.empty:
        best_day = daily.sort_values("pnl_total", ascending=False).iloc[0]
        worst_day = daily.sort_values("pnl_total", ascending=True).iloc[0]
        lines.append(f"Mejor día: {best_day['trade_day']}.")
        lines.append(f"Peor día: {worst_day['trade_day']}.")
    if not weekday.empty:
        best_d = weekday.sort_values("pnl_total", ascending=False).iloc[0]
        worst_d = weekday.sort_values("pnl_total", ascending=True).iloc[0]
        lines.append(f"Mejor día de semana: {best_d['dia_semana_label']}.")
        lines.append(f"Día de semana más débil: {worst_d['dia_semana_label']}.")
    if not by_hour.empty:
        best_h = by_hour.sort_values("pnl_total", ascending=False).iloc[0]
        worst_h = by_hour.sort_values("pnl_total", ascending=True).iloc[0]
        lines.append(f"Mejor hora por PnL total: {best_h['hora_label']}.")
        lines.append(f"Peor hora por PnL total: {worst_h['hora_label']}.")

    show_conclusion("Tiempo y Sesiones", lines)


def render_motor_reversiones(ops_df: pd.DataFrame, legs_df: pd.DataFrame):
    st.header("Motor de Reversiones")
    section_note("Esta página responde: ¿cuántas reversiones ayudan y qué habría pasado si limitamos el máximo permitido?")
    show_help(
        "Motor de Reversiones",
        "Versión simple. Mantiene el dropdown importante de máximo reversal permitido, pero sin análisis confuso por pierna.",
        [
            "¿El PnL empeora cuando suben las reversiones?",
            "¿Qué cantidad de reversiones trae más drawdown?",
            "¿Qué pasa si permito máximo 0, 1, 2, 3 o 4 reversals?",
            "¿La reducción de riesgo vale más que el PnL que se pierde?",
        ],
    )

    if ops_df.empty:
        st.warning("No hay operaciones con los filtros actuales.")
        return

    rev = aggregate_core(ops_df, ["reversal_count"]).sort_values("reversal_count")
    st.subheader("1. Resultado real por cantidad de reversals")
    st.dataframe(rev, use_container_width=True)

    if len(rev) > 0:
        rev_chart = rev.copy()
        rev_chart["reversal_label"] = rev_chart["reversal_count"].fillna(0).astype(int).astype(str)
        render_summary_bar_chart(
            rev_chart,
            x_col="reversal_label",
            title="PnL total por cantidad de reversals",
            x_title="Reversals usados",
            category_order=rev_chart["reversal_label"].tolist(),
        )

    st.markdown("---")
    st.subheader("2. Simulación simple por máximo reversal permitido")
    section_note("Este es el dropdown importante: simula qué habría pasado si el bot no hubiera pasado de ese número máximo de reversals.")

    max_real_rev = int(ops_df["reversal_count"].fillna(0).max()) if "reversal_count" in ops_df.columns else 0
    max_reversal_permitido = st.selectbox(
        "Máximo reversal permitido",
        options=list(range(0, max_real_rev + 1)),
        index=max_real_rev,
        help="Ejemplo: si eliges 2, cualquier operación que llegó a reversal 3 o 4 se corta en el resultado acumulado después del reversal 2.",
    )

    sim_df, sim_metrics = simulated_reversal_metrics(ops_df, legs_df, max_reversal_permitido)

    c = st.columns(4)
    with c[0]: card("PnL Real", fmt_money(sim_metrics.get("real_pnl", np.nan)))
    with c[1]: card("PnL Reversal Seleccionado", fmt_money(sim_metrics.get("sim_pnl", np.nan)))
    with c[2]: card("Operaciones Cortadas", str(sim_metrics.get("ops_cortadas", 0)))
    with c[3]: card("Resultados Cambiados", str(sim_metrics.get("ops_resultado_cambiado", 0)))

    c = st.columns(4)
    with c[0]: card("Profit Factor Real", "-" if pd.isna(sim_metrics.get("real_pf", np.nan)) else f"{sim_metrics.get('real_pf'):.2f}")
    with c[1]: card("PF Reversal Seleccionado", "-" if pd.isna(sim_metrics.get("sim_pf", np.nan)) else f"{sim_metrics.get('sim_pf'):.2f}")
    with c[2]: card("Max DD Real", fmt_money(sim_metrics.get("real_max_dd", np.nan)))
    with c[3]: card("Max DD Seleccionado", fmt_money(sim_metrics.get("sim_max_dd", np.nan)))

    cap_summary = pd.DataFrame([
        {"métrica": "PnL total", "real": sim_metrics.get("real_pnl", np.nan), "reversal_seleccionado": sim_metrics.get("sim_pnl", np.nan), "diferencia": sim_metrics.get("pnl_diff", np.nan)},
        {"métrica": "Profit factor", "real": sim_metrics.get("real_pf", np.nan), "reversal_seleccionado": sim_metrics.get("sim_pf", np.nan), "diferencia": sim_metrics.get("pf_diff", np.nan)},
        {"métrica": "Max drawdown", "real": sim_metrics.get("real_max_dd", np.nan), "reversal_seleccionado": sim_metrics.get("sim_max_dd", np.nan), "diferencia": sim_metrics.get("dd_diff", np.nan)},
        {"métrica": "Peor operación", "real": sim_metrics.get("worst_real_op", np.nan), "reversal_seleccionado": sim_metrics.get("worst_sim_op", np.nan), "diferencia": sim_metrics.get("worst_sim_op", np.nan) - sim_metrics.get("worst_real_op", np.nan)},
        {"métrica": "Operaciones cortadas", "real": 0, "reversal_seleccionado": sim_metrics.get("ops_cortadas", 0), "diferencia": sim_metrics.get("ops_cortadas", 0)},
    ])
    st.dataframe(cap_summary, use_container_width=True)

    month_sim = monthly_simulated_reversal_summary(sim_df)
    if not month_sim.empty:
        st.markdown("**Impacto mensual del reversal seleccionado**")
        section_note(
            "Este gráfico compara el PnL real contra el PnL si el bot se hubiera detenido en el reversal seleccionado. "
            "Pasa el mouse sobre cada barra para ver Profit Factor, caída máxima, operaciones cortadas y resultados cambiados."
        )
        render_reversal_month_impact_chart(month_sim, max_reversal_permitido)
        with st.expander("Ver tabla mensual detallada", expanded=False):
            st.dataframe(month_sim, use_container_width=True)

    detail_cols = [
        "month", "trade_day", "operation_id", "sequence_started_at", "reversal_count", "reversal_count_simulado",
        "sequence_net_pnl_currency", "sequence_net_pnl_simulado", "sequence_end_reason", "sequence_end_reason_simulado",
        "operation_max_drawdown_currency", "base_contracts", "max_contracts_used", "sesion", "hora_inicio", "config_key",
    ]
    detail_cols = [c for c in detail_cols if c in sim_df.columns]

    st.markdown("**Operaciones donde el cap cambió el resultado**")
    changed = sim_df[sim_df["cap_aplicado"] == True].copy()
    if changed.empty:
        st.info("Con este máximo reversal permitido no se cortó ninguna operación.")
    else:
        st.dataframe(changed[detail_cols].sort_values("sequence_started_at", ascending=False), use_container_width=True)

    st.markdown("---")
    st.subheader("3. Peores operaciones con reversals")
    dangerous = ops_df.sort_values(["reversal_count", "operation_max_drawdown_currency"], ascending=[False, False]).head(25)
    cols = [
        "month", "trade_day", "operation_id", "sequence_started_at", "reversal_count",
        "sequence_net_pnl_currency", "operation_max_drawdown_currency", "sequence_end_reason",
        "base_contracts", "max_contracts_used", "sesion", "hora_inicio", "config_key",
    ]
    st.dataframe(dangerous[[c for c in cols if c in dangerous.columns]], use_container_width=True)

    lines = []
    if not rev.empty:
        best = rev.sort_values("pnl_promedio", ascending=False).iloc[0]
        worst = rev.sort_values("pnl_promedio", ascending=True).iloc[0]
        risky = rev.sort_values("drawdown_promedio", ascending=False).iloc[0]
        lines.append(f"Mejor PnL promedio real: {int(best['reversal_count']) if pd.notna(best['reversal_count']) else 0} reversal(es).")
        lines.append(f"Peor PnL promedio real: {int(worst['reversal_count']) if pd.notna(worst['reversal_count']) else 0} reversal(es).")
        lines.append(f"Mayor drawdown promedio real: {int(risky['reversal_count']) if pd.notna(risky['reversal_count']) else 0} reversal(es).")
    lines.append(
        f"Con reversal seleccionado = {max_reversal_permitido}, "
        f"el PnL cambia de {fmt_money(sim_metrics.get('real_pnl', np.nan))} "
        f"a {fmt_money(sim_metrics.get('sim_pnl', np.nan))}."
    )
    if sim_metrics.get("worst_sim_op", np.nan) > sim_metrics.get("worst_real_op", np.nan):
        lines.append("El reversal seleccionado mejora el peor caso; esto puede ayudar al control de riesgo.")
    elif sim_metrics.get("worst_sim_op", np.nan) < sim_metrics.get("worst_real_op", np.nan):
        lines.append("El reversal seleccionado empeora el peor caso en esta simulación; revisar antes de usarlo.")
    show_conclusion("Motor de Reversiones", lines)


def render_simulador_diario(ops_df: pd.DataFrame, legs_df: pd.DataFrame):
    st.header("Simulador Diario")
    section_note(
        "Esta página responde: ¿qué habría pasado si cambiamos la meta diaria y la pérdida diaria? "
        "La simulación usa piernas cerradas, porque cada pierna es una entrada/salida real."
    )
    show_help(
        "Simulador Diario",
        "Compara el resultado real contra un escenario simulado con otra meta diaria y otra pérdida máxima diaria.",
        [
            "¿El bot mejora si corto el día antes?",
            "¿La pérdida máxima diaria protege la cuenta o corta demasiadas recuperaciones?",
            "¿La meta diaria es realista mes por mes?",
            "¿Qué pasa con PnL, Profit Factor, días ganadores, días perdidos y caída máxima?",
        ],
    )

    if ops_df.empty or legs_df.empty:
        st.warning("No hay operaciones o piernas con los filtros actuales.")
        return

    st.subheader("1. Configuración de la simulación")
    c1, c2, c3 = st.columns([1, 1, 1.25])
    daily_target = c1.number_input("Meta diaria", min_value=1.0, value=600.0, step=50.0, key="daily_target")
    daily_loss = c2.number_input("Pérdida máxima diaria", min_value=1.0, value=600.0, step=50.0, key="daily_loss")
    flat_at_limits = c3.checkbox(
        "Flat exacto en meta/pérdida",
        value=True,
        help=(
            "ON: si la pérdida configurada es 600 y la pierna cierra en -615, el simulador usa -600. "
            "OFF: usa el resultado real de la pierna, por ejemplo -615."
        ),
    )

    st.markdown("**Sesión usada para separar los días**")
    s1, s2, s3 = st.columns([1, 1, 1.4])
    use_operational_day = s1.checkbox(
        "Usar día operativo del bot",
        value=True,
        help=(
            "ON: agrupa las piernas según la sesión intradía del bot, por ejemplo 18:00 a 17:00. "
            "Esto evita partir una misma sesión overnight en dos días calendario."
        ),
    )
    session_start_hms = s2.text_input("Inicio sesión", value="18:00:00", key="daily_session_start")
    session_end_hms = s3.text_input("Fin sesión", value="17:00:00", key="daily_session_end")

    st.caption(
        "Regla: después de cada pierna cerrada revisamos el PnL acumulado del día operativo. "
        "Si toca la meta o la pérdida, el día se detiene y las siguientes piernas de esa sesión se ignoran en la simulación."
    )

    daily_df, daily_m = simulate_daily_stop(
        legs_df,
        daily_target,
        daily_loss,
        flat_at_limits=flat_at_limits,
        use_operational_day=use_operational_day,
        session_start_hms=session_start_hms,
        session_end_hms=session_end_hms,
    )
    real_daily = real_daily_from_legs(legs_df, use_operational_day, session_start_hms, session_end_hms)
    real_m = _daily_metrics_from_results(real_daily, "real_day_pnl", "trade_day")

    if daily_df.empty or not daily_m:
        st.info("No hay suficientes piernas para simular.")
        return

    render_real_vs_sim_daily_cards(real_m, daily_m)

    st.subheader("2. Visual · Real vs Simulado")
    render_daily_sim_comparison_chart(daily_df)

    st.subheader("3. Resultado por mes")
    monthly_daily = monthly_daily_stop_summary(daily_df)
    render_monthly_daily_stop_chart(monthly_daily)

    with st.expander("Ver tabla mensual detallada", expanded=False):
        st.dataframe(monthly_daily, use_container_width=True)

    st.subheader("4. Días donde la simulación cambia el resultado")
    changed = daily_df.loc[daily_df["real_day_pnl"].round(8) != daily_df["simulated_day_pnl"].round(8)].copy()
    if changed.empty:
        st.info("Con esta configuración, ningún día cambió contra el resultado real.")
    else:
        display_cols = [
            "month", "trade_day", "real_day_pnl", "simulated_day_pnl", "difference",
            "stop_reason", "raw_pnl_at_stop", "legs_used", "real_legs", "legs_skipped",
            "operations_touched", "real_operations_touched", "stopped_after_operation",
            "stopped_after_leg", "stopped_at_time",
        ]
        st.dataframe(changed[[c for c in display_cols if c in changed.columns]].sort_values("trade_day"), use_container_width=True)

    st.subheader("5. Tabla completa por día")
    with st.expander("Ver todos los días", expanded=False):
        display_cols = [
            "month", "trade_day", "real_day_pnl", "simulated_day_pnl", "difference",
            "stop_reason", "raw_pnl_at_stop", "legs_used", "real_legs", "legs_skipped",
            "operations_touched", "real_operations_touched", "stopped_after_operation",
            "stopped_after_leg", "stopped_at_time",
        ]
        st.dataframe(daily_df[[c for c in display_cols if c in daily_df.columns]].sort_values("trade_day"), use_container_width=True)

    st.markdown("---")
    st.subheader("6. Sets diarios / avanzado")
    section_note(
        "Esto es opcional. A diferencia del stop diario clásico, aquí el día puede tener varios sets. "
        "Cuando un set toca target o loss, la siguiente pierna empieza otro set."
    )

    show_sets = st.checkbox("Mostrar simulador de sets diarios", value=False)
    if show_sets:
        c1, c2 = st.columns(2)
        set_target = c1.number_input("Target por set", min_value=1.0, value=600.0, step=50.0, key="set_target")
        set_loss = c2.number_input("Loss por set", min_value=1.0, value=300.0, step=50.0, key="set_loss")

        sets_df, sets_m = simulate_daily_sets(legs_df, set_target, set_loss, use_operational_day, session_start_hms, session_end_hms)
        c = st.columns(4)
        with c[0]: card("PnL Sets", fmt_money(sets_m.get("total_pnl", np.nan)))
        with c[1]: card("Sets Meta", fmt_pct(sets_m.get("target_sets_pct", np.nan)))
        with c[2]: card("Sets Pérdida", fmt_pct(sets_m.get("loss_sets_pct", np.nan)))
        with c[3]: card("Piernas/Set Prom", "-" if pd.isna(sets_m.get("avg_legs_per_set", np.nan)) else f"{sets_m['avg_legs_per_set']:.2f}")

        st.dataframe(sets_df.sort_values(["trade_day", "set_number"]), use_container_width=True)

        if not sets_df.empty:
            monthly_sets = sets_df.groupby("month", as_index=False).agg(
                sets=("set_number", "count"),
                pnl_sets=("set_result", "sum"),
                promedio_set=("set_result", "mean"),
                meta_pct=("set_outcome", lambda ss: (ss == "Set target reached").mean() * 100),
                perdida_pct=("set_outcome", lambda ss: (ss == "Set loss reached").mean() * 100),
                abierto_pct=("set_outcome", lambda ss: (ss == "End of day").mean() * 100),
                piernas_por_set=("legs_used", "mean"),
                operaciones_por_set=("operations_touched", "mean"),
            )
            st.markdown("**Resumen mensual · Sets diarios**")
            st.dataframe(monthly_sets, use_container_width=True)

    lines = []
    if daily_m:
        if daily_m.get("total_pnl", 0) > real_m.get("total_pnl", 0):
            lines.append("La simulación mejora el PnL total contra el resultado real.")
        else:
            lines.append("La simulación no mejora el PnL total contra el resultado real con estos valores.")

        if pd.notna(daily_m.get("max_drawdown", np.nan)) and pd.notna(real_m.get("max_drawdown", np.nan)):
            if daily_m["max_drawdown"] > real_m["max_drawdown"]:
                lines.append("La caída máxima mejora, porque la cuenta baja menos desde su mejor punto.")
            elif daily_m["max_drawdown"] < real_m["max_drawdown"]:
                lines.append("La caída máxima empeora con esta configuración; revisar antes de usarla.")

        lines.append(
            f"Con esta configuración, {int(daily_m.get('days_changed', 0))} día(s) cambian y se ignoran "
            f"{int(daily_m.get('legs_skipped', 0))} pierna(s) después de tocar meta/pérdida."
        )
    show_conclusion("Simulador Diario", lines)


def render_laboratorio_parametros(ops_df: pd.DataFrame):
    st.header("Laboratorio de Parámetros")
    section_note("Esta página responde: ¿qué setting tiene mejor balance entre PnL y riesgo?")
    show_help(
        "Laboratorio de Parámetros",
        "Aquí comparamos base contracts, stop, target, distance, max reversals y configuración completa.",
        [
            "¿Qué base contracts gana más sin disparar demasiado el peor caso?",
            "¿Qué stop/target parece más estable?",
            "¿Qué configuración funciona en varios meses?",
            "¿Subir contratos aumenta demasiado el drawdown o la peor operación?",
        ],
    )

    if ops_df.empty:
        st.warning("No hay operaciones con los filtros actuales.")
        return

    param = st.selectbox(
        "Comparar por",
        ["base_contracts", "fixed_stop_ticks", "fixed_target_ticks", "distance_points", "max_reversals_allowed", "config_key"],
    )

    grouped = aggregate_core(ops_df, [param]).sort_values("pnl_total", ascending=False)
    st.subheader("Comparación principal")
    st.dataframe(grouped, use_container_width=True)

    if len(grouped) > 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(grouped[param].astype(str), grouped["pnl_total"])
        ax.set_title(f"PnL por {param}")
        ax.set_xlabel(param)
        ax.set_ylabel("PnL")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    st.subheader("Riesgo por tamaño de contrato")
    section_note("Esto reemplaza la vieja sección confusa de exposición por pierna. Aquí vemos si más contratos mejoran el resultado o solo agrandan el riesgo.")
    contract_risk = aggregate_core(ops_df, ["base_contracts"]).sort_values("base_contracts")
    st.dataframe(contract_risk, use_container_width=True)

    if ops_df["month"].nunique() > 1:
        st.subheader("Estabilidad mensual por parámetro")
        monthly_param = aggregate_core(ops_df, ["month", param]).sort_values(["month", param])
        st.dataframe(monthly_param, use_container_width=True)

    lines = []
    if not grouped.empty:
        best_pnl = grouped.iloc[0]
        best_pf = grouped.sort_values("profit_factor", ascending=False).iloc[0]
        lines.append(f"Mejor PnL total: {param} = {best_pnl[param]}.")
        lines.append(f"Mejor profit factor: {param} = {best_pf[param]}.")
    if not contract_risk.empty:
        worst_risk = contract_risk.sort_values("peor_operacion", ascending=True).iloc[0]
        lines.append(f"Mayor peor caso por base contracts: {worst_risk['base_contracts']} contratos.")
    show_conclusion("Laboratorio de Parámetros", lines)


def render_risk_killers(ops_df: pd.DataFrame):
    st.header("Risk Killers")
    section_note("Esta página responde: ¿qué cosas están dañando el bot más rápido?")
    show_help(
        "Risk Killers",
        "Lista práctica de los riesgos que conviene revisar primero.",
        [
            "¿Cuáles son las peores operaciones?",
            "¿Qué días dañan el resultado?",
            "¿Hay ganadoras con drawdown demasiado feo?",
            "¿Qué horas o sesiones son más peligrosas?",
        ],
    )

    if ops_df.empty:
        st.warning("No hay operaciones con los filtros actuales.")
        return

    cols = [
        "month", "trade_day", "operation_id", "sequence_started_at", "sequence_net_pnl_currency",
        "operation_max_drawdown_currency", "reversal_count", "base_contracts", "max_contracts_used",
        "sesion", "hora_inicio", "sequence_end_reason", "config_key",
    ]
    cols = [c for c in cols if c in ops_df.columns]

    st.subheader("Peores operaciones")
    worst_ops = ops_df.sort_values("sequence_net_pnl_currency", ascending=True).head(25)
    st.dataframe(worst_ops[cols], use_container_width=True)

    st.subheader("Peores días")
    worst_days = ops_df.groupby(["month", "trade_day"], as_index=False).agg(
        pnl_dia=("sequence_net_pnl_currency", "sum"),
        operaciones=("operation_id", "count"),
        peor_operacion=("sequence_net_pnl_currency", "min"),
        drawdown_max=("operation_max_drawdown_currency", "max"),
        reversiones_promedio=("reversal_count", "mean"),
        contratos_max=("max_contracts_used", "max"),
    ).sort_values("pnl_dia", ascending=True).head(25)
    st.dataframe(worst_days, use_container_width=True)

    st.subheader("Ganadoras peligrosas")
    section_note("Operaciones que cerraron positivas, pero sufrieron mucho drawdown. Estas pueden verse bien en PnL, pero ser malas para operar en vivo.")
    winners_bad_dd = ops_df[ops_df["sequence_net_pnl_currency"] > 0].sort_values("operation_max_drawdown_currency", ascending=False).head(25)
    st.dataframe(winners_bad_dd[cols], use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Horas peligrosas")
        bad_hours = aggregate_core(ops_df, ["hora_inicio"]).sort_values("pnl_total", ascending=True).head(10)
        st.dataframe(bad_hours, use_container_width=True)
    with c2:
        st.subheader("Sesiones peligrosas")
        bad_sessions = aggregate_core(ops_df, ["sesion"]).sort_values("pnl_total", ascending=True)
        st.dataframe(bad_sessions, use_container_width=True)

    lines = []
    if not worst_ops.empty:
        row = worst_ops.iloc[0]
        lines.append(f"Peor operación: {row['operation_id']} con PnL {fmt_money(row['sequence_net_pnl_currency'])}.")
    if not worst_days.empty:
        row = worst_days.iloc[0]
        lines.append(f"Peor día: {row['trade_day']} con PnL {fmt_money(row['pnl_dia'])}.")
    if not bad_hours.empty:
        row = bad_hours.iloc[0]
        lines.append(f"Hora más débil por PnL: {int(row['hora_inicio'])}:00.")
    show_conclusion("Risk Killers", lines)


def render_explorador_operaciones(ops_df: pd.DataFrame, legs_df: pd.DataFrame):
    st.header("Explorador de Operaciones")
    section_note("Esta página responde: ¿qué pasó exactamente dentro de una operación?")
    show_help(
        "Explorador de Operaciones",
        "Detalle técnico por operación. Aquí sí vemos piernas, contratos, PnL acumulado y el resultado simulado con máximo reversal permitido.",
        [
            "¿Dónde entró y salió cada pierna?",
            "¿Cómo evolucionó el PnL acumulado?",
            "¿Cuántos contratos se usaron?",
            "¿Qué habría pasado con un máximo reversal diferente?",
        ],
    )

    if ops_df.empty:
        st.warning("No hay operaciones con los filtros actuales.")
        return

    filtered = ops_df.copy()

    st.subheader("Filtros de búsqueda")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        search_id = st.text_input("Buscar operation_id", "")
    with c2:
        only_losers = st.checkbox("Solo operaciones perdedoras", value=False)
    with c3:
        min_reversal = st.number_input("Reversal mínimo", min_value=0, value=0, step=1)
    with c4:
        max_real_rev = int(filtered["reversal_count"].fillna(0).max()) if "reversal_count" in filtered.columns else 0
        max_reversal_permitido = st.selectbox(
            "Máximo reversal permitido",
            options=list(range(0, max_real_rev + 1)),
            index=max_real_rev,
            key="explorer_max_reversal_permitido",
        )

    if search_id.strip():
        filtered = filtered[filtered["operation_id"].astype(str).str.contains(search_id.strip(), case=False, na=False)]
    if only_losers:
        filtered = filtered[filtered["sequence_net_pnl_currency"] < 0]
    filtered = filtered[filtered["reversal_count"].fillna(0) >= min_reversal]

    if filtered.empty:
        st.warning("No hay operaciones con esos filtros.")
        return

    filtered_sim = aplicar_cap_reversal(filtered, legs_df, max_reversal_permitido)

    c = st.columns(3)
    with c[0]: card("PnL Real Visible", fmt_money(filtered_sim["sequence_net_pnl_currency"].sum()))
    with c[1]: card("PnL Simulado Visible", fmt_money(filtered_sim["sequence_net_pnl_simulado"].sum()))
    with c[2]: card("Ops Cortadas", str(int(filtered_sim["cap_aplicado"].sum())))

    list_cols = [
        "month", "trade_day", "operation_id", "sequence_started_at", "sequence_net_pnl_currency",
        "sequence_net_pnl_simulado", "reversal_count", "reversal_count_simulado",
        "operation_max_drawdown_currency", "operation_max_runup_currency",
        "base_contracts", "max_contracts_used", "sesion", "hora_inicio", "sequence_end_reason", "sequence_end_reason_simulado", "config_key",
    ]
    list_cols = [c for c in list_cols if c in filtered_sim.columns]
    st.dataframe(filtered_sim[list_cols].sort_values("sequence_started_at", ascending=False), use_container_width=True)

    op_ids = filtered_sim.sort_values("sequence_started_at", ascending=False)["operation_id"].astype(str).tolist()
    selected = st.selectbox("Seleccionar operación", op_ids)
    if not selected:
        return

    op_row = filtered_sim[filtered_sim["operation_id"].astype(str) == selected]
    st.subheader("Resumen de la operación")
    st.dataframe(op_row, use_container_width=True)

    st.subheader("Piernas de la operación")
    if legs_df.empty:
        st.info("No hay piernas cargadas para esta operación.")
        return

    legs_op = legs_df[legs_df["operation_id"].astype(str) == selected].sort_values("leg_index")
    if legs_op.empty:
        st.info("No se encontraron piernas para esta operación.")
        return

    st.dataframe(legs_op, use_container_width=True)

    if "cumulative_sequence_pnl_after_leg" in legs_op.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(legs_op["leg_index"].astype(str), legs_op["cumulative_sequence_pnl_after_leg"], marker="o")
        ax.set_title("PnL Acumulado por Pierna")
        ax.set_xlabel("Pierna")
        ax.set_ylabel("PnL Acumulado")
        st.pyplot(fig)

    if "entry_qty" in legs_op.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(legs_op["leg_index"].astype(str), legs_op["entry_qty"])
        ax.set_title("Contratos por Pierna")
        ax.set_xlabel("Pierna")
        ax.set_ylabel("Contratos")
        st.pyplot(fig)

    row = op_row.iloc[0]
    lines = [
        f"PnL real: {fmt_money(row['sequence_net_pnl_currency'])}.",
        f"PnL simulado con cap {max_reversal_permitido}: {fmt_money(row['sequence_net_pnl_simulado'])}.",
        f"Reversals reales: {int(row['reversal_count']) if pd.notna(row['reversal_count']) else 0}.",
        f"Reversals simulados: {int(row['reversal_count_simulado']) if pd.notna(row['reversal_count_simulado']) else 0}.",
        f"Drawdown máximo: {fmt_money(row['operation_max_drawdown_currency'])}.",
        f"Máximo contratos usados: {fmt_money(row['max_contracts_used'])}.",
        f"Razón de cierre real: {row['sequence_end_reason']}.",
        f"Razón de cierre simulada: {row['sequence_end_reason_simulado']}.",
    ]
    show_conclusion("Operación seleccionada", lines)


# ============================================================
# NAVIGATION HELPERS
# ============================================================


def scroll_to_top_on_page_change(page: str):
    previous_page = st.session_state.get("_previous_main_page")
    page_changed = previous_page != page
    st.session_state["_previous_main_page"] = page

    if page_changed:
        components.html(
            """
            <script>
            function scrollParentToTop() {
                try {
                    const doc = window.parent.document;
                    const candidates = [
                        doc.querySelector('[data-testid="stAppViewContainer"]'),
                        doc.querySelector('section.main'),
                        doc.documentElement,
                        doc.body
                    ];

                    for (const el of candidates) {
                        if (el && typeof el.scrollTo === 'function') {
                            el.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
                        }
                    }

                    window.parent.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
                } catch (e) {
                    window.parent.scrollTo(0, 0);
                }
            }
            setTimeout(scrollParentToTop, 50);
            </script>
            """,
            height=0,
        )


# ============================================================
# MAIN
# ============================================================


def main():
    st.title("Laboratorio WLF")
    st.caption("Análisis simple para decidir qué mantener, qué bloquear y qué setting probar.")

    pages = [
        "Dashboard General",
        "Tiempo y Sesiones",
        "Motor de Reversiones",
        "Simulador Diario",
        "Laboratorio de Parámetros",
        "Risk Killers",
        "Explorador de Operaciones",
    ]

    uploaded_files = st.sidebar.file_uploader(
        "Cargar archivos JSONL mensuales",
        type=["jsonl"],
        accept_multiple_files=True,
    )

    # Navigation stays immediately below the JSON uploader.
    # This way you always choose the page first, before the long filters.
    st.sidebar.markdown("---")
    st.sidebar.subheader("Menú principal")
    page = st.sidebar.radio(
        "Selecciona una página",
        pages,
        index=0,
        key="main_page_selector",
    )

    scroll_to_top_on_page_change(page)

    records = load_uploaded_jsonl_files(uploaded_files)
    ops_df, legs_df = build_dataframes(records)

    if ops_df.empty:
        st.warning("Sube uno o más JSONL para iniciar el análisis.")
        return

    # Correct sidebar order:
    # 1) Upload JSONL files
    # 2) Main menu
    # 3) Global filters
    # 4) Loaded / filtered data metrics
    ops_filtered, legs_filtered = apply_global_filters(ops_df, legs_df)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Datos")
    c_loaded, c_filtered = st.sidebar.columns(2)
    with c_loaded:
        st.caption("Cargado")
        st.metric("Ops", len(ops_df))
        st.metric("Piernas", len(legs_df))
    with c_filtered:
        st.caption("Filtrado")
        st.metric("Ops", len(ops_filtered))
        st.metric("Piernas", len(legs_filtered))

    if page == "Dashboard General":
        render_dashboard_general(ops_filtered, legs_filtered)
    elif page == "Tiempo y Sesiones":
        render_tiempo_y_sesiones(ops_filtered)
    elif page == "Motor de Reversiones":
        render_motor_reversiones(ops_filtered, legs_filtered)
    elif page == "Simulador Diario":
        render_simulador_diario(ops_filtered, legs_filtered)
    elif page == "Laboratorio de Parámetros":
        render_laboratorio_parametros(ops_filtered)
    elif page == "Risk Killers":
        render_risk_killers(ops_filtered)
    elif page == "Explorador de Operaciones":
        render_explorador_operaciones(ops_filtered, legs_filtered)


if __name__ == "__main__":
    main()
