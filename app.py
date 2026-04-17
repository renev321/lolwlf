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


def mostrar_bloque_ayuda(titulo: str, descripcion: str, preguntas: list[str]):
    with st.expander(f"¿Para qué sirve esta página? · {titulo}", expanded=False):
        st.markdown(descripcion)
        st.markdown("**Qué deberías mirar aquí:**")
        for q in preguntas:
            st.markdown(f"- {q}")


def mostrar_bloque_conclusion(titulo: str, lineas: list[str]):
    st.markdown(f"**Lectura rápida · {titulo}**")
    for linea in lineas:
        st.markdown(f"- {linea}")


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

def calcular_pnl_simulado_por_cap_reversal(
    op_row: pd.Series,
    legs_df: pd.DataFrame,
    max_reversal_permitido: int,
) -> float:
    operation_id = op_row["operation_id"]
    reversal_count_real = int(op_row["reversal_count"]) if pd.notna(op_row["reversal_count"]) else 0

    # Si la operación real ya está dentro del cap, usamos el pnl real
    if reversal_count_real <= max_reversal_permitido:
        return float(op_row["sequence_net_pnl_currency"])

    legs_op = legs_df.loc[legs_df["operation_id"] == operation_id].copy()
    if legs_op.empty:
        return float(op_row["sequence_net_pnl_currency"])

    legs_op = legs_op.sort_values("leg_index")

    # Base leg = reversal_number 0
    target_reversal = max_reversal_permitido

    leg_target = legs_op.loc[legs_op["reversal_number"] == target_reversal].copy()

    if leg_target.empty:
        return float(op_row["sequence_net_pnl_currency"])

    leg_target = leg_target.sort_values("leg_index").iloc[-1]
    pnl_simulado = leg_target["cumulative_sequence_pnl_after_leg"]

    if pd.isna(pnl_simulado):
        return float(op_row["sequence_net_pnl_currency"])

    return float(pnl_simulado)


def render_overview(ops_df: pd.DataFrame):
    st.subheader("Salud del Bot")
    mostrar_bloque_ayuda(
        "Salud del Bot",
        "Esta página te da una vista general del rendimiento del bot. Sirve para validar si el sistema tiene sentido antes de entrar en detalles más finos como sesiones, reversals o timing.",
        [
            "¿El bot gana dinero de forma general?",
            "¿El profit factor y la tasa de acierto son razonables?",
            "¿La mejor operación y la peor operación están demasiado separadas?",
            "¿Hay estabilidad o el resultado depende de pocos días grandes?",
        ],
    )

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

    lineas = []
    if metrics["total_net_pnl"] > 0:
        lineas.append("El bot está positivo en el rango de datos cargado.")
    else:
        lineas.append("El bot todavía no está positivo en el rango de datos cargado.")

    if not pd.isna(metrics["profit_factor"]):
        if metrics["profit_factor"] >= 1.5:
            lineas.append("El profit factor es saludable y sugiere una ventaja interesante.")
        elif metrics["profit_factor"] >= 1.0:
            lineas.append("El profit factor es aceptable, pero todavía puede requerir filtrado o ajuste.")
        else:
            lineas.append("El profit factor está débil; conviene revisar sesiones, timing y reversals.")

    mostrar_bloque_conclusion("Salud del Bot", lineas)


def render_reversal_engine(ops_df: pd.DataFrame):
    st.subheader("Motor de Reversiones")
    mostrar_bloque_ayuda(
        "Motor de Reversiones",
        "Esta página sirve para entender si el sistema realmente mejora o empeora cuando profundiza en más reversals. Aquí debes decidir si conviene permitir 0, 1, 2, 3 o más reversiones.",
        [
            "¿Las operaciones sin reversals ya son suficientemente buenas?",
            "¿A partir de qué número de reversals cae el PnL promedio?",
            "¿Qué profundidad de reversal trae demasiado drawdown?",
            "¿Cuál es la profundidad máxima sensata para el bot?",
        ],
    )

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

    lineas = []
    if not grouped.empty:
        mejor_fila = grouped.sort_values("pnl_promedio", ascending=False).iloc[0]
        peor_fila = grouped.sort_values("pnl_promedio", ascending=True).iloc[0]
        lineas.append(f"La mejor profundidad por PnL promedio es {mejor_fila['reversiones']} reversión(es).")
        lineas.append(f"La peor profundidad por PnL promedio es {peor_fila['reversiones']} reversión(es).")
        if peor_fila["reversiones"] != mejor_fila["reversiones"]:
            lineas.append("Esto te puede ayudar a decidir si conviene cortar el bot antes de profundidades débiles.")

    mostrar_bloque_conclusion("Motor de Reversiones", lineas)


def render_daily_goal_simulator(ops_df: pd.DataFrame):
    st.subheader("Simulador de Meta Diaria")
    mostrar_bloque_ayuda(
        "Simulador de Meta Diaria",
        "Esta página simula cómo se habría comportado el bot si hubieras detenido el día al alcanzar una meta diaria o una pérdida máxima diaria. Sirve para elegir un plan diario realista.",
        [
            "¿Qué tan seguido se alcanza la meta antes que la pérdida?",
            "¿Una meta diaria alta reduce demasiado la consistencia?",
            "¿Una pérdida máxima demasiado amplia empeora el perfil diario?",
            "¿Qué combinación de meta/pérdida parece más equilibrada?",
        ],
    )

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

        lineas = []
        if metrics["target_first_rate"] > metrics["loss_first_rate"]:
            lineas.append("La meta diaria se alcanza antes que la pérdida en una proporción favorable.")
        else:
            lineas.append("La pérdida diaria está saltando demasiado pronto frente a la meta configurada.")

        if metrics["avg_daily_result"] > 0:
            lineas.append("El resultado diario promedio de esta simulación es positivo.")
        else:
            lineas.append("El resultado diario promedio de esta simulación todavía no es positivo.")

        mostrar_bloque_conclusion("Simulador de Meta Diaria", lineas)


def render_time_edge(ops_df: pd.DataFrame):
    st.subheader("Ventaja Temporal")
    mostrar_bloque_ayuda(
        "Ventaja Temporal",
        "Esta página ayuda a detectar en qué horas y días el bot tiene mejor o peor comportamiento. Es clave para decidir si conviene bloquear ciertas franjas horarias.",
        [
            "¿Qué horas generan más PnL?",
            "¿Qué horas exigen más reversals o más drawdown?",
            "¿Hay días de la semana consistentemente débiles?",
            "¿Conviene reducir o cortar el bot en ciertos horarios?",
        ],
    )

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

    if len(by_hour) <= 1:
        st.info("Todavía no hay suficiente variedad horaria para sacar conclusiones fuertes.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(by_hour["hora_inicio"].astype(str), by_hour["pnl_total"])
    ax.set_title("PnL Total por Hora")
    ax.set_xlabel("Hora de Inicio")
    ax.set_ylabel("PnL")
    st.pyplot(fig)

    lineas = []
    if not by_hour.empty:
        mejor_hora = by_hour.sort_values("pnl_total", ascending=False).iloc[0]
        peor_hora = by_hour.sort_values("pnl_total", ascending=True).iloc[0]
        lineas.append(f"La mejor hora cargada por PnL total es {int(mejor_hora['hora_inicio'])}:00.")
        lineas.append(f"La peor hora cargada por PnL total es {int(peor_hora['hora_inicio'])}:00.")

    mostrar_bloque_conclusion("Ventaja Temporal", lineas)


def render_session_intelligence(ops_df: pd.DataFrame):
    st.subheader("Inteligencia por Sesión")
    mostrar_bloque_ayuda(
        "Inteligencia por Sesión",
        "Esta página compara el rendimiento entre Asia, Londres, NY Open, NY Midday y NY Late. Sirve para decidir qué sesión conviene priorizar, reducir o eliminar.",
        [
            "¿Qué sesión aporta más PnL total y promedio?",
            "¿Qué sesión exige más reversals?",
            "¿Qué sesión trae más drawdown?",
            "¿Hay una sesión que conviene bloquear completamente?",
        ],
    )

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

        lineas = []
        mejor_sesion = session_df.iloc[0]
        lineas.append(f"Por ahora solo hay datos de la sesión {mejor_sesion['sesion']}.")
        lineas.append("Cuando cargues más días podrás comparar mejor qué sesión conviene mantener o bloquear.")
        mostrar_bloque_conclusion("Inteligencia por Sesión", lineas)
        return

    st.dataframe(session_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(session_df["sesion"], session_df["pnl_total"])
    ax.set_title("PnL Total por Sesión")
    ax.set_xlabel("Sesión")
    ax.set_ylabel("PnL")
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)

    lineas = []
    if not session_df.empty:
        mejor_sesion = session_df.sort_values("pnl_total", ascending=False).iloc[0]
        peor_sesion = session_df.sort_values("pnl_total", ascending=True).iloc[0]
        lineas.append(f"La sesión más fuerte por PnL total es {mejor_sesion['sesion']}.")
        lineas.append(f"La sesión más débil por PnL total es {peor_sesion['sesion']}.")

    mostrar_bloque_conclusion("Inteligencia por Sesión", lineas)


def render_crazy_mode_timing(ops_df: pd.DataFrame):
    st.subheader("Timing de Crazy Mode")
    mostrar_bloque_ayuda(
        "Timing de Crazy Mode",
        "Esta página analiza qué tan buenas o malas son la primera, segunda, tercera y demás operaciones del día. Sirve para decidir hasta qué número de operación conviene dejar correr el bot.",
        [
            "¿La operación #1 del día es la mejor?",
            "¿La operación #3 o #4 ya pierde edge?",
            "¿A partir de qué número sube demasiado el drawdown?",
            "¿Conviene limitar Crazy Mode a pocas operaciones por día?",
        ],
    )

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

    if len(timing_df) <= 1:
        st.info("Todavía no hay suficiente variedad en número de operaciones por día para sacar conclusiones fuertes.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(timing_df["numero_operacion_dia"].astype(str), timing_df["pnl_total"])
    ax.set_title("PnL Total por Número de Operación del Día")
    ax.set_xlabel("Número de Operación del Día")
    ax.set_ylabel("PnL")
    st.pyplot(fig)

    lineas = []
    if not timing_df.empty:
        mejor_num = timing_df.sort_values("pnl_promedio", ascending=False).iloc[0]
        peor_num = timing_df.sort_values("pnl_promedio", ascending=True).iloc[0]
        lineas.append(f"La mejor operación del día por PnL promedio es la #{int(mejor_num['numero_operacion_dia'])}.")
        lineas.append(f"La peor operación del día por PnL promedio es la #{int(peor_num['numero_operacion_dia'])}.")

    mostrar_bloque_conclusion("Timing de Crazy Mode", lineas)


def render_parameter_lab(ops_df: pd.DataFrame):
    st.subheader("Laboratorio de Parámetros")
    mostrar_bloque_ayuda(
        "Laboratorio de Parámetros",
        "Esta página sirve para comparar cómo rinden distintas configuraciones del bot. Es muy útil cuando ya tengas más meses o más pruebas con diferentes settings.",
        [
            "¿Qué stop está dando mejor equilibrio entre PnL y drawdown?",
            "¿Qué target está funcionando mejor?",
            "¿DistancePoints más alto o más bajo mejora el resultado?",
            "¿Más MaxReversals realmente ayuda o solo agranda el riesgo?",
        ],
    )

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

    if len(grouped) <= 1:
        st.info("Todavía no hay suficiente variedad de parámetros en los datos cargados para comparar.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(grouped[param].astype(str), grouped["pnl_total"])
    ax.set_title(f"PnL Total por {param}")
    ax.set_xlabel(param)
    ax.set_ylabel("PnL")
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)

    lineas = []
    if not grouped.empty:
        mejor_param = grouped.sort_values("pnl_promedio", ascending=False).iloc[0]
        lineas.append(f"El mejor valor cargado de {param} por PnL promedio es {mejor_param[param]}.")

    mostrar_bloque_conclusion("Laboratorio de Parámetros", lineas)


def render_exit_engine(ops_df: pd.DataFrame, legs_df: pd.DataFrame):
    st.subheader("Motor de Salidas")
    mostrar_bloque_ayuda(
        "Motor de Salidas",
        "Esta página sirve para entender cómo termina realmente cada pierna y si Auto BE o Trailing están ayudando o estorbando. Ojo: una cosa es el tipo de salida final y otra distinta es si el management fue activado.",
        [
            "¿Las piernas terminan más por TP, SL, BE o TS?",
            "¿Cuando Auto BE se activa, mejora o empeora el resultado promedio?",
            "¿Trailing protege beneficios o corta demasiado pronto?",
            "¿Conviene retrasar o suavizar el management?",
        ],
    )

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

    be_stats["auto_be_activated"] = be_stats["auto_be_activated"].map({False: "No", True: "Sí", 0: "No", 1: "Sí"})
    tr_stats["trailing_activated"] = tr_stats["trailing_activated"].map({False: "No", True: "Sí", 0: "No", 1: "Sí"})

    be_stats = be_stats.rename(columns={
        "auto_be_activated": "Auto BE activado",
        "piernas": "Piernas",
        "pnl_promedio": "PnL Promedio",
    })
    tr_stats = tr_stats.rename(columns={
        "trailing_activated": "Trailing activado",
        "piernas": "Piernas",
        "pnl_promedio": "PnL Promedio",
    })

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Impacto de Auto BE**")
        st.dataframe(be_stats, use_container_width=True)
    with c2:
        st.markdown("**Impacto del Trailing**")
        st.dataframe(tr_stats, use_container_width=True)

    lineas = []
    if not exit_df.empty:
        top_exit = exit_df.sort_values("piernas", ascending=False).iloc[0]
        lineas.append(f"El tipo de salida más frecuente es {top_exit['exit_result_type']}.")

    mostrar_bloque_conclusion("Motor de Salidas", lineas)


def render_operation_explorer(ops_df: pd.DataFrame, legs_df: pd.DataFrame):
    st.subheader("Explorador de Operaciones")
    if ops_df.empty:
        st.info("No se encontraron datos de operaciones.")
        return

    filtro_df = ops_df.copy()

    st.markdown("### Filtros")

    c1, c2 = st.columns(2)

    with c1:
        min_date = filtro_df["sequence_started_at"].min()
        max_date = filtro_df["sequence_started_at"].max()

        fecha_inicio = st.date_input(
            "Fecha inicial",
            value=min_date.date() if pd.notna(min_date) else None,
        )

        sesiones_disponibles = sorted([s for s in filtro_df["sesion"].dropna().unique().tolist()])
        sesiones_sel = st.multiselect(
            "Sesiones",
            options=sesiones_disponibles,
            default=sesiones_disponibles,
        )

        razones_disponibles = sorted([r for r in filtro_df["sequence_end_reason"].dropna().unique().tolist()])
        razones_sel = st.multiselect(
            "Razón de cierre",
            options=razones_disponibles,
            default=razones_disponibles,
        )

    with c2:
        fecha_fin = st.date_input(
            "Fecha final",
            value=max_date.date() if pd.notna(max_date) else None,
        )

        max_reversal_disponible = int(filtro_df["reversal_count"].fillna(0).max())

        max_reversal_permitido = st.selectbox(
            "Máximo reversal permitido",
            options=list(range(0, max_reversal_disponible + 1)),
            index=max_reversal_disponible,
        )

        numeros_op_disponibles = sorted(
            [int(x) for x in filtro_df["numero_operacion_dia"].dropna().unique().tolist()]
        )
        numeros_op_sel = st.multiselect(
            "Número de operación del día",
            options=numeros_op_disponibles,
            default=numeros_op_disponibles,
        )

    texto_busqueda = st.text_input("Buscar por operation_id", "")

    if fecha_inicio:
        filtro_df = filtro_df[filtro_df["sequence_started_at"].dt.date >= fecha_inicio]

    if fecha_fin:
        filtro_df = filtro_df[filtro_df["sequence_started_at"].dt.date <= fecha_fin]

    if sesiones_sel:
        filtro_df = filtro_df[filtro_df["sesion"].isin(sesiones_sel)]

    if razones_sel:
        filtro_df = filtro_df[filtro_df["sequence_end_reason"].isin(razones_sel)]

    if numeros_op_sel:
        filtro_df = filtro_df[filtro_df["numero_operacion_dia"].isin(numeros_op_sel)]

    if texto_busqueda.strip():
        filtro_df = filtro_df[
            filtro_df["operation_id"].astype(str).str.contains(texto_busqueda.strip(), case=False, na=False)
        ]

    # Nuevo cálculo: pnl simulado según cap de reversal
    filtro_df = filtro_df.copy()
    filtro_df["pnl_simulado_cap_reversal"] = filtro_df.apply(
        lambda row: calcular_pnl_simulado_por_cap_reversal(
            row,
            legs_df,
            max_reversal_permitido,
        ),
        axis=1,
    )

    # Visible solo hasta el cap seleccionado
    filtro_df_visible = filtro_df[
        filtro_df["reversal_count"].fillna(0) <= max_reversal_permitido
    ].copy()

    st.markdown("### Resultado filtrado")

    c1, c2, c3 = st.columns(3)
    c1.metric("Operaciones visibles", f"{len(filtro_df_visible)}")
    c2.metric(
        "PnL Real Visible",
        f"{filtro_df_visible['sequence_net_pnl_currency'].sum():.2f}" if not filtro_df_visible.empty else "0.00",
    )
    c3.metric(
        "PnL Simulado Cap",
        f"{filtro_df['pnl_simulado_cap_reversal'].sum():.2f}" if not filtro_df.empty else "0.00",
    )

    if filtro_df.empty:
        st.warning("No hay operaciones que cumplan con los filtros actuales.")
        return

    display_cols = [
        "operation_id",
        "sequence_started_at",
        "sequence_end_reason",
        "reversal_count",
        "sequence_net_pnl_currency",
        "pnl_simulado_cap_reversal",
        "sequence_loss_currency",
        "operation_max_drawdown_currency",
        "operation_max_runup_currency",
        "fixed_stop_ticks",
        "fixed_target_ticks",
        "distance_points",
        "sesion",
        "numero_operacion_dia",
    ]

    st.dataframe(
        filtro_df[display_cols].sort_values("sequence_started_at", ascending=False),
        use_container_width=True,
    )

    op_ids = (
        filtro_df.sort_values("sequence_started_at", ascending=False)["operation_id"]
        .dropna()
        .astype(str)
        .tolist()
    )

    selected_op = st.selectbox("Inspeccionar operación", op_ids)

    if selected_op:
        op_row = filtro_df.loc[filtro_df["operation_id"] == selected_op]
        st.markdown("### Resumen de la operación")
        st.dataframe(op_row, use_container_width=True)

        st.markdown("### Piernas de la operación")
        legs_op = legs_df.loc[legs_df["operation_id"] == selected_op].sort_values("leg_index")
        st.dataframe(legs_op, use_container_width=True)

        if not legs_op.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(
                legs_op["leg_index"].astype(str),
                legs_op["cumulative_sequence_pnl_after_leg"],
                marker="o",
            )
            ax.set_title("Evolución del PnL Acumulado por Pierna")
            ax.set_xlabel("Pierna")
            ax.set_ylabel("PnL Acumulado")
            st.pyplot(fig)

        fila = op_row.iloc[0]
        st.markdown("### Lectura rápida")
        st.markdown(
            f"- Esta operación terminó con **{int(fila['reversal_count']) if pd.notna(fila['reversal_count']) else 0}** reversal(es)."
        )
        st.markdown(
            f"- El **PnL real** de la operación fue **{fila['sequence_net_pnl_currency']:.2f}**."
        )
        st.markdown(
            f"- El **PnL simulado** con máximo reversal permitido = **{max_reversal_permitido}** fue **{fila['pnl_simulado_cap_reversal']:.2f}**."
        )
        st.markdown(
            f"- La sesión clasificada para esta operación es **{fila['sesion']}**."
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
