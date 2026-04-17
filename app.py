import json
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="Laboratorio lol_wlf", layout="wide")

st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] {
    gap: 1rem;
}
.lab-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 16px;
    padding: 16px 18px;
    min-height: 110px;
    background: rgba(255,255,255,0.02);
}
.lab-card-title {
    font-size: 0.95rem;
    opacity: 0.8;
    margin-bottom: 10px;
}
.lab-card-value {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.1;
}
</style>
""", unsafe_allow_html=True)


def load_uploaded_jsonl_files(uploaded_files) -> list[dict]:
    records: list[dict] = []

    if not uploaded_files:
        st.sidebar.info("Todavía no has cargado archivos.")
        return records

    for uploaded_file in uploaded_files:
        try:
            content = uploaded_file.getvalue().decode("utf-8-sig", errors="ignore")
            lines = content.splitlines()

            st.sidebar.write(f"Archivo: {uploaded_file.name}")
            st.sidebar.write(f"Líneas crudas: {len(lines)}")

            valid_count = 0
            invalid_count = 0

            for i, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                    records.append(obj)
                    valid_count += 1
                except json.JSONDecodeError as e:
                    invalid_count += 1
                    if invalid_count <= 3:
                        st.sidebar.write(f"Línea JSON inválida {i}: {line[:120]}")
                        st.sidebar.write(f"Error: {e}")

            st.sidebar.write(f"Registros válidos: {valid_count}")
            st.sidebar.write(f"Líneas inválidas: {invalid_count}")

        except Exception as e:
            st.sidebar.error(f"Error al leer {uploaded_file.name}: {e}")

    return records


def _clasificar_sesion(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "Sin sesión"

    h = ts.hour
    m = ts.minute
    total_min = h * 60 + m

    # Asia: 18:00–03:29
    if total_min >= 18 * 60 or total_min <= (3 * 60 + 29):
        return "Asia"
    # Londres: 03:30–09:29
    if (3 * 60 + 30) <= total_min <= (9 * 60 + 29):
        return "Londres"
    # NY Open: 09:30–10:30
    if (9 * 60 + 30) <= total_min <= (10 * 60 + 30):
        return "NY Open"
    # NY Midday: 10:31–13:29
    if (10 * 60 + 31) <= total_min <= (13 * 60 + 29):
        return "NY Midday"
    # NY Late: 13:30–17:00
    if (13 * 60 + 30) <= total_min <= (17 * 60):
        return "NY Late"

    return "Fuera de Sesión"


def build_dataframes(records: list[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not records:
        return pd.DataFrame(), pd.DataFrame()

    ops_rows = []
    legs_rows = []

    for rec in records:
        op_row = {
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
        ops_rows.append(op_row)

        for leg in rec.get("legs", []):
            legs_rows.append(
                {
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

    for col in ["date", "sequence_started_at", "sequence_ended_at"]:
        if col in ops_df.columns:
            ops_df[col] = pd.to_datetime(ops_df[col], errors="coerce")

    for col in ["date", "sequence_started_at", "entry_time", "exit_time"]:
        if col in legs_df.columns:
            legs_df[col] = pd.to_datetime(legs_df[col], errors="coerce")

    numeric_cols_ops = [
        "base_price",
        "range_high",
        "range_low",
        "base_contracts",
        "reversal_count",
        "max_reversals_allowed",
        "fixed_stop_ticks",
        "fixed_target_ticks",
        "distance_points",
        "sequence_net_pnl_currency",
        "sequence_loss_currency",
        "sequence_execution_commission_currency",
        "base_target_profit_currency",
        "operation_max_drawdown_currency",
        "operation_max_runup_currency",
    ]
    for col in numeric_cols_ops:
        if col in ops_df.columns:
            ops_df[col] = pd.to_numeric(ops_df[col], errors="coerce")

    numeric_cols_legs = [
        "leg_index",
        "reversal_number",
        "entry_price_avg",
        "entry_qty",
        "initial_stop_price",
        "initial_target_price",
        "exit_price_avg",
        "realized_pnl_currency",
        "sequence_loss_before_entry",
        "smart_recovery_qty_computed",
        "cumulative_sequence_pnl_after_leg",
        "operation_drawdown_after_leg",
        "operation_runup_after_leg",
    ]
    for col in numeric_cols_legs:
        if col in legs_df.columns:
            legs_df[col] = pd.to_numeric(legs_df[col], errors="coerce")

    ops_df["trade_day"] = ops_df["sequence_started_at"].dt.date
    ops_df["hora_inicio"] = ops_df["sequence_started_at"].dt.hour
    ops_df["minuto_inicio"] = ops_df["sequence_started_at"].dt.minute
    ops_df["dia_semana"] = ops_df["sequence_started_at"].dt.day_name()
    ops_df["sesion"] = ops_df["sequence_started_at"].apply(_clasificar_sesion)
    ops_df["es_ganadora"] = ops_df["sequence_net_pnl_currency"] > 0
    ops_df = ops_df.sort_values("sequence_started_at").copy()
    ops_df["numero_operacion_dia"] = ops_df.groupby("trade_day").cumcount() + 1

    return ops_df, legs_df


def compute_overview_metrics(ops_df: pd.DataFrame) -> dict:
    if ops_df.empty:
        return {}

    winners = ops_df.loc[ops_df["sequence_net_pnl_currency"] > 0, "sequence_net_pnl_currency"]
    losers = ops_df.loc[ops_df["sequence_net_pnl_currency"] < 0, "sequence_net_pnl_currency"]

    gross_profit = winners.sum()
    gross_loss = abs(losers.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    return {
        "total_operations": len(ops_df),
        "total_net_pnl": ops_df["sequence_net_pnl_currency"].sum(),
        "avg_pnl": ops_df["sequence_net_pnl_currency"].mean(),
        "median_pnl": ops_df["sequence_net_pnl_currency"].median(),
        "win_rate": (ops_df["sequence_net_pnl_currency"] > 0).mean() * 100,
        "profit_factor": profit_factor,
        "avg_winner": winners.mean() if not winners.empty else np.nan,
        "avg_loser": losers.mean() if not losers.empty else np.nan,
        "worst_operation": ops_df["sequence_net_pnl_currency"].min(),
        "best_operation": ops_df["sequence_net_pnl_currency"].max(),
        "max_operation_drawdown": ops_df["operation_max_drawdown_currency"].max(),
    }


def simulate_daily_plan(ops_df: pd.DataFrame, daily_target: float, daily_loss: float) -> Tuple[pd.DataFrame, dict]:
    if ops_df.empty:
        return pd.DataFrame(), {}

    sim_rows = []

    for trade_day, day_df in ops_df.sort_values("sequence_started_at").groupby("trade_day"):
        running = 0.0
        outcome = "ninguno"
        stop_after_operation = None

        for _, row in day_df.iterrows():
            running += float(row["sequence_net_pnl_currency"])
            stop_after_operation = row["operation_id"]

            if running >= daily_target:
                outcome = "meta_primero"
                running = daily_target
                break

            if running <= -daily_loss:
                outcome = "perdida_primero"
                running = -daily_loss
                break

        sim_rows.append(
            {
                "trade_day": trade_day,
                "resultado_diario_simulado": running,
                "resultado": outcome,
                "stop_after_operation": stop_after_operation,
            }
        )

    sim_df = pd.DataFrame(sim_rows)
    if sim_df.empty:
        return sim_df, {}

    metrics = {
        "days": len(sim_df),
        "target_first_rate": (sim_df["resultado"] == "meta_primero").mean() * 100,
        "loss_first_rate": (sim_df["resultado"] == "perdida_primero").mean() * 100,
        "neither_rate": (sim_df["resultado"] == "ninguno").mean() * 100,
        "avg_daily_result": sim_df["resultado_diario_simulado"].mean(),
        "median_daily_result": sim_df["resultado_diario_simulado"].median(),
        "total_simulated_pnl": sim_df["resultado_diario_simulado"].sum(),
        "best_day": sim_df["resultado_diario_simulado"].max(),
        "worst_day": sim_df["resultado_diario_simulado"].min(),
    }
    return sim_df, metrics


def max_streak(series: pd.Series, positive: bool = True) -> int:
    best = 0
    current = 0
    for value in series:
        cond = value > 0 if positive else value < 0
        if cond:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def render_overview(ops_df: pd.DataFrame):
    st.subheader("Salud del Bot")
    metrics = compute_overview_metrics(ops_df)
    if not metrics:
        st.info("No se encontraron datos de operaciones.")
        return

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

    row1 = st.columns(4)
    with row1[0]:
        card("Operaciones", f"{metrics['total_operations']}")
    with row1[1]:
        card("PnL Neto Total", f"{metrics['total_net_pnl']:.2f}")
    with row1[2]:
        card("Tasa de Acierto %", f"{metrics['win_rate']:.1f}")
    with row1[3]:
        pf = "-" if pd.isna(metrics["profit_factor"]) else f"{metrics['profit_factor']:.2f}"
        card("Profit Factor", pf)

    row2 = st.columns(4)
    with row2[0]:
        card("PnL Promedio", f"{metrics['avg_pnl']:.2f}")
    with row2[1]:
        card("PnL Mediano", f"{metrics['median_pnl']:.2f}")
    with row2[2]:
        card("Mejor Op", f"{metrics['best_operation']:.2f}")
    with row2[3]:
        card("Peor Op", f"{metrics['worst_operation']:.2f}")

    daily = (
        ops_df.groupby("trade_day", as_index=False)["sequence_net_pnl_currency"]
        .sum()
        .sort_values("trade_day")
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily["trade_day"].astype(str), daily["sequence_net_pnl_currency"])
    ax.set_title("PnL Neto Diario")
    ax.set_ylabel("PnL")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)


def render_reversal_engine(ops_df: pd.DataFrame):
    st.subheader("Motor de Reversiones")
    if ops_df.empty:
        st.info("No se encontraron datos de operaciones.")
        return

    grouped = ops_df.groupby("reversal_count").agg(
        operaciones=("operation_id", "count"),
        pnl_total=("sequence_net_pnl_currency", "sum"),
        pnl_promedio=("sequence_net_pnl_currency", "mean"),
        tasa_acierto=("es_ganadora", "mean"),
        drawdown_promedio=("operation_max_drawdown_currency", "mean"),
        perdida_promedio_antes_recuperacion=("sequence_loss_currency", "mean"),
    ).reset_index()
    grouped["tasa_acierto"] = grouped["tasa_acierto"] * 100
    grouped = grouped.rename(columns={"reversal_count": "reversiones"})

    st.dataframe(grouped, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(grouped["reversiones"].astype(str), grouped["pnl_total"])
    ax.set_title("PnL Total por Cantidad de Reversiones")
    ax.set_xlabel("Cantidad de Reversiones")
    ax.set_ylabel("PnL Total")
    st.pyplot(fig)


def render_daily_goal_simulator(ops_df: pd.DataFrame):
    st.subheader("Simulador de Meta Diaria")
    if ops_df.empty:
        st.info("No se encontraron datos de operaciones.")
        return

    c1, c2 = st.columns(2)
    daily_target = c1.number_input("Meta Diaria", min_value=1.0, value=600.0, step=50.0)
    daily_loss = c2.number_input("Pérdida Máxima Diaria", min_value=1.0, value=300.0, step=50.0)

    sim_df, metrics = simulate_daily_plan(ops_df, daily_target=daily_target, daily_loss=daily_loss)
    if not metrics:
        st.info("No hay suficientes datos para simular.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Meta Primero %", f"{metrics['target_first_rate']:.1f}")
    c2.metric("Pérdida Primero %", f"{metrics['loss_first_rate']:.1f}")
    c3.metric("Ninguno %", f"{metrics['neither_rate']:.1f}")
    c4.metric("Resultado Diario Promedio", f"{metrics['avg_daily_result']:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("PnL Simulado Total", f"{metrics['total_simulated_pnl']:.2f}")
    c6.metric("Resultado Diario Mediano", f"{metrics['median_daily_result']:.2f}")
    c7.metric("Mejor Día", f"{metrics['best_day']:.2f}")
    c8.metric("Peor Día", f"{metrics['worst_day']:.2f}")

    if not sim_df.empty:
        ordered = sim_df.sort_values("trade_day")
        win_streak = max_streak(ordered["resultado_diario_simulado"], positive=True)
        loss_streak = max_streak(ordered["resultado_diario_simulado"], positive=False)
        st.write(f"Racha máxima de días ganadores: **{win_streak}**")
        st.write(f"Racha máxima de días perdedores: **{loss_streak}**")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ordered["trade_day"].astype(str), ordered["resultado_diario_simulado"])
        ax.set_title("Resultados Diarios Simulados")
        ax.set_ylabel("PnL")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

        st.dataframe(ordered, use_container_width=True)


def render_time_edge(ops_df: pd.DataFrame):
    st.subheader("Ventaja Temporal")
    if ops_df.empty:
        st.info("No se encontraron datos de operaciones.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**PnL por día de la semana**")
        weekday = ops_df.groupby("dia_semana", as_index=False).agg(
            pnl_total=("sequence_net_pnl_currency", "sum"),
            operaciones=("operation_id", "count"),
            tasa_acierto=("es_ganadora", "mean"),
        )
        weekday["tasa_acierto"] = weekday["tasa_acierto"] * 100
        st.dataframe(weekday, use_container_width=True)

    with col2:
        st.markdown("**PnL por hora**")
        by_hour = ops_df.groupby("hora_inicio", as_index=False).agg(
            pnl_total=("sequence_net_pnl_currency", "sum"),
            operaciones=("operation_id", "count"),
            reversiones_promedio=("reversal_count", "mean"),
            drawdown_promedio=("operation_max_drawdown_currency", "mean"),
        )
        st.dataframe(by_hour, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(by_hour["hora_inicio"].astype(str), by_hour["pnl_total"])
    ax.set_title("PnL Total por Hora")
    ax.set_xlabel("Hora de Inicio")
    ax.set_ylabel("PnL")
    st.pyplot(fig)


def render_session_intelligence(ops_df: pd.DataFrame):
    st.subheader("Inteligencia por Sesión")
    if ops_df.empty:
        st.info("No se encontraron datos de operaciones.")
        return

    session_df = ops_df.groupby("sesion", as_index=False).agg(
        operaciones=("operation_id", "count"),
        pnl_total=("sequence_net_pnl_currency", "sum"),
        pnl_promedio=("sequence_net_pnl_currency", "mean"),
        tasa_acierto=("es_ganadora", "mean"),
        reversiones_promedio=("reversal_count", "mean"),
        drawdown_promedio=("operation_max_drawdown_currency", "mean"),
    )
    session_df["tasa_acierto"] = session_df["tasa_acierto"] * 100

    if len(session_df) == 1:
    st.info("Por ahora solo hay una sesión en los datos cargados. Cuando cargues más meses o más horarios, esta vista se volverá mucho más útil.")
    st.dataframe(session_df, use_container_width=True)
    return

    st.dataframe(session_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(session_df["sesion"], session_df["pnl_total"])
    ax.set_title("PnL Total por Sesión")
    ax.set_xlabel("Sesión")
    ax.set_ylabel("PnL")
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)


def render_crazy_mode_timing(ops_df: pd.DataFrame):
    st.subheader("Timing de Crazy Mode")
    if ops_df.empty:
        st.info("No se encontraron datos de operaciones.")
        return

    timing_df = ops_df.groupby("numero_operacion_dia", as_index=False).agg(
        operaciones=("operation_id", "count"),
        pnl_total=("sequence_net_pnl_currency", "sum"),
        pnl_promedio=("sequence_net_pnl_currency", "mean"),
        tasa_acierto=("es_ganadora", "mean"),
        reversiones_promedio=("reversal_count", "mean"),
        drawdown_promedio=("operation_max_drawdown_currency", "mean"),
    )
    timing_df["tasa_acierto"] = timing_df["tasa_acierto"] * 100

    st.dataframe(timing_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(timing_df["numero_operacion_dia"].astype(str), timing_df["pnl_total"])
    ax.set_title("PnL Total por Número de Operación del Día")
    ax.set_xlabel("Número de Operación del Día")
    ax.set_ylabel("PnL")
    st.pyplot(fig)


def render_parameter_lab(ops_df: pd.DataFrame):
    st.subheader("Laboratorio de Parámetros")
    if ops_df.empty:
        st.info("No se encontraron datos de operaciones.")
        return

    param = st.selectbox(
        "Selecciona un parámetro",
        ["fixed_stop_ticks", "fixed_target_ticks", "distance_points", "max_reversals_allowed"],
    )

    grouped = ops_df.groupby(param, as_index=False).agg(
        operaciones=("operation_id", "count"),
        pnl_total=("sequence_net_pnl_currency", "sum"),
        pnl_promedio=("sequence_net_pnl_currency", "mean"),
        tasa_acierto=("es_ganadora", "mean"),
        drawdown_promedio=("operation_max_drawdown_currency", "mean"),
    )
    grouped["tasa_acierto"] = grouped["tasa_acierto"] * 100

    st.dataframe(grouped, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(grouped[param].astype(str), grouped["pnl_total"])
    ax.set_title(f"PnL Total por {param}")
    ax.set_xlabel(param)
    ax.set_ylabel("PnL")
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)


def render_exit_engine(ops_df: pd.DataFrame, legs_df: pd.DataFrame):
    st.subheader("Motor de Salidas")
    if legs_df.empty:
        st.info("No se encontraron datos de piernas.")
        return

    exit_df = legs_df.groupby("exit_result_type", as_index=False).agg(
        piernas=("operation_id", "count"),
        pnl_total=("realized_pnl_currency", "sum"),
        pnl_promedio=("realized_pnl_currency", "mean"),
    )
    st.dataframe(exit_df, use_container_width=True)

    be_stats = legs_df.groupby("auto_be_activated", as_index=False).agg(
        piernas=("operation_id", "count"),
        pnl_promedio=("realized_pnl_currency", "mean"),
    )
    tr_stats = legs_df.groupby("trailing_activated", as_index=False).agg(
        piernas=("operation_id", "count"),
        pnl_promedio=("realized_pnl_currency", "mean"),
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Impacto de Auto BE**")
        st.dataframe(be_stats, use_container_width=True)
    with c2:
        st.markdown("**Impacto del Trailing**")
        st.dataframe(tr_stats, use_container_width=True)


def render_operation_explorer(ops_df: pd.DataFrame, legs_df: pd.DataFrame):
    st.subheader("Explorador de Operaciones")
    if ops_df.empty:
        st.info("No se encontraron datos de operaciones.")
        return

    display_cols = [
        "operation_id",
        "sequence_started_at",
        "sequence_end_reason",
        "reversal_count",
        "sequence_net_pnl_currency",
        "sequence_loss_currency",
        "operation_max_drawdown_currency",
        "operation_max_runup_currency",
        "fixed_stop_ticks",
        "fixed_target_ticks",
        "distance_points",
        "sesion",
        "numero_operacion_dia",
    ]
    st.dataframe(ops_df[display_cols].sort_values("sequence_started_at", ascending=False), use_container_width=True)

    op_ids = ops_df["operation_id"].dropna().tolist()
    selected_op = st.selectbox("Inspeccionar operación", op_ids)
    if selected_op:
        op_row = ops_df.loc[ops_df["operation_id"] == selected_op]
        st.write(op_row)
        st.dataframe(
            legs_df.loc[legs_df["operation_id"] == selected_op].sort_values("leg_index"),
            use_container_width=True,
        )


def main():
    st.title("Laboratorio Python de lol_wlf")
    st.caption("Analítica orientada a decisiones para tu bot")

    uploaded_files = st.sidebar.file_uploader(
        "Cargar archivos JSONL mensuales",
        type=["jsonl"],
        accept_multiple_files=True,
    )

    records = load_uploaded_jsonl_files(uploaded_files)
    ops_df, legs_df = build_dataframes(records)

    if ops_df.empty:
        st.warning("No se cargaron registros JSONL. Sube uno o más archivos JSONL mensuales, por ejemplo desde 2026-01.jsonl hasta 2026-06.jsonl.")
        return

    st.sidebar.metric("Archivos cargados", f"{len(uploaded_files)}")
    st.sidebar.metric("Operaciones cargadas", f"{len(ops_df)}")
    st.sidebar.metric("Piernas cargadas", f"{len(legs_df)}")

    page = st.sidebar.radio(
        "Página",
        [
            "Resumen",
            "Motor de Reversiones",
            "Simulador de Meta Diaria",
            "Ventaja Temporal",
            "Inteligencia por Sesión",
            "Timing de Crazy Mode",
            "Laboratorio de Parámetros",
            "Motor de Salidas",
            "Explorador de Operaciones",
        ],
    )

    if page == "Resumen":
        render_overview(ops_df)
    elif page == "Motor de Reversiones":
        render_reversal_engine(ops_df)
    elif page == "Simulador de Meta Diaria":
        render_daily_goal_simulator(ops_df)
    elif page == "Ventaja Temporal":
        render_time_edge(ops_df)
    elif page == "Inteligencia por Sesión":
        render_session_intelligence(ops_df)
    elif page == "Timing de Crazy Mode":
        render_crazy_mode_timing(ops_df)
    elif page == "Laboratorio de Parámetros":
        render_parameter_lab(ops_df)
    elif page == "Motor de Salidas":
        render_exit_engine(ops_df, legs_df)
    elif page == "Explorador de Operaciones":
        render_operation_explorer(ops_df, legs_df)


if __name__ == "__main__":
    main()
