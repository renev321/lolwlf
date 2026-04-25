import json
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG / STYLE
# ============================================================

st.set_page_config(page_title="Laboratorio WLF", layout="wide")

st.markdown(
    """
<style>
div[data-testid="stHorizontalBlock"] { gap: 1rem; }
.lab-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 16px;
    padding: 16px 18px;
    min-height: 105px;
    background: rgba(255,255,255,0.02);
}
.lab-card-title {
    font-size: 0.90rem;
    opacity: 0.80;
    margin-bottom: 8px;
}
.lab-card-value {
    font-size: 1.95rem;
    font-weight: 700;
    line-height: 1.1;
}
.small-note {
    opacity: 0.75;
    font-size: 0.90rem;
}
.good { color: #2ecc71; font-weight: 700; }
.bad { color: #ff7675; font-weight: 700; }
.warn { color: #f1c40f; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
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


def mostrar_bloque_ayuda(titulo: str, descripcion: str, preguntas: List[str]):
    with st.expander(f"¿Para qué sirve esta página? · {titulo}", expanded=False):
        st.markdown(descripcion)
        st.markdown("**Qué deberías mirar aquí:**")
        for q in preguntas:
            st.markdown(f"- {q}")


def mostrar_conclusion(titulo: str, lineas: List[str]):
    st.markdown(f"### Lectura rápida · {titulo}")
    for linea in lineas:
        st.markdown(f"- {linea}")


def safe_div(a: float, b: float) -> float:
    if b is None or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def fmt_money(x) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):,.2f}"


def fmt_pct(x) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.1f}%"


# ============================================================
# LOAD / TRANSFORM DATA
# ============================================================


def load_uploaded_jsonl_files(uploaded_files) -> List[dict]:
    records: List[dict] = []

    if not uploaded_files:
        st.sidebar.info("Todavía no has cargado archivos JSONL.")
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
                except json.JSONDecodeError as e:
                    invalid_count += 1
                    if invalid_count <= 3:
                        st.sidebar.warning(f"JSON inválido en {uploaded_file.name}, línea {i}: {str(e)}")

            with st.sidebar.expander(f"Archivo · {uploaded_file.name}", expanded=False):
                st.write(f"Líneas crudas: {len(lines)}")
                st.write(f"Registros válidos: {valid_count}")
                st.write(f"Líneas inválidas: {invalid_count}")

        except Exception as e:
            st.sidebar.error(f"Error al leer {uploaded_file.name}: {e}")

    return records


def _clasificar_sesion(ts: pd.Timestamp) -> str:
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


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _to_datetime(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def build_dataframes(records: List[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not records:
        return pd.DataFrame(), pd.DataFrame()

    ops_rows = []
    legs_rows = []

    for rec in records:
        op_row = {
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
        ops_rows.append(op_row)

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
        ],
    )

    legs_df = _to_numeric(
        legs_df,
        [
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
        ],
    )

    if ops_df.empty:
        return ops_df, legs_df

    ops_df = ops_df.sort_values("sequence_started_at").copy()
    ops_df["trade_day"] = ops_df["sequence_started_at"].dt.date
    ops_df["month"] = ops_df["sequence_started_at"].dt.to_period("M").astype(str)
    ops_df["hora_inicio"] = ops_df["sequence_started_at"].dt.hour
    ops_df["minuto_inicio"] = ops_df["sequence_started_at"].dt.minute
    ops_df["dia_semana"] = ops_df["sequence_started_at"].dt.day_name()
    ops_df["sesion"] = ops_df["sequence_started_at"].apply(_clasificar_sesion)
    ops_df["es_ganadora"] = ops_df["sequence_net_pnl_currency"] > 0
    ops_df["numero_operacion_dia"] = ops_df.groupby("trade_day").cumcount() + 1

    # Full config key. This is important when several months/settings are loaded together.
    ops_df["config_key"] = (
        "BC=" + ops_df["base_contracts"].fillna(-1).astype(int).astype(str)
        + " | SL=" + ops_df["fixed_stop_ticks"].fillna(-1).astype(int).astype(str)
        + " | TP=" + ops_df["fixed_target_ticks"].fillna(-1).astype(int).astype(str)
        + " | REV=" + ops_df["max_reversals_allowed"].fillna(-1).astype(int).astype(str)
        + " | DIST=" + ops_df["distance_points"].round(2).astype(str)
    )

    if not legs_df.empty:
        legs_df["month"] = legs_df["sequence_started_at"].dt.to_period("M").astype(str)
        legs_df = legs_df.merge(
            ops_df[["operation_id", "trade_day", "config_key", "instrument", "sesion", "hora_inicio", "dia_semana"]],
            on="operation_id",
            how="left",
            suffixes=("", "_op"),
        )

    return ops_df, legs_df


# ============================================================
# METRICS / SIMULATIONS
# ============================================================


def compute_overview_metrics(ops_df: pd.DataFrame) -> Dict[str, float]:
    if ops_df.empty:
        return {}

    pnl = ops_df["sequence_net_pnl_currency"].fillna(0)
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]

    gross_profit = winners.sum()
    gross_loss = abs(losers.sum())
    profit_factor = safe_div(gross_profit, gross_loss)

    daily = ops_df.groupby("trade_day")["sequence_net_pnl_currency"].sum()

    return {
        "total_operations": len(ops_df),
        "days": daily.shape[0],
        "total_net_pnl": pnl.sum(),
        "avg_pnl": pnl.mean(),
        "median_pnl": pnl.median(),
        "win_rate": (pnl > 0).mean() * 100,
        "profit_factor": profit_factor,
        "avg_winner": winners.mean() if not winners.empty else np.nan,
        "avg_loser": losers.mean() if not losers.empty else np.nan,
        "worst_operation": pnl.min(),
        "best_operation": pnl.max(),
        "positive_days_rate": (daily > 0).mean() * 100 if len(daily) else np.nan,
        "best_day": daily.max() if len(daily) else np.nan,
        "worst_day": daily.min() if len(daily) else np.nan,
        "max_operation_drawdown": ops_df["operation_max_drawdown_currency"].max(),
        "avg_reversals": ops_df["reversal_count"].mean(),
    }


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
    ).reset_index()

    grouped["tasa_acierto"] = grouped["tasa_acierto"] * 100
    return grouped


def add_profit_factor_by_group(df: pd.DataFrame, grouped: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty or grouped.empty:
        return grouped

    tmp = df.copy()
    tmp["gross_profit"] = tmp["sequence_net_pnl_currency"].where(tmp["sequence_net_pnl_currency"] > 0, 0)
    tmp["gross_loss"] = -tmp["sequence_net_pnl_currency"].where(tmp["sequence_net_pnl_currency"] < 0, 0)
    pf = tmp.groupby(group_cols, dropna=False).agg(
        gross_profit=("gross_profit", "sum"),
        gross_loss=("gross_loss", "sum"),
    ).reset_index()
    pf["profit_factor"] = pf.apply(lambda r: safe_div(r["gross_profit"], r["gross_loss"]), axis=1)
    return grouped.merge(pf[group_cols + ["profit_factor"]], on=group_cols, how="left")


def simulate_daily_stop(ops_df: pd.DataFrame, daily_target: float, daily_loss: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if ops_df.empty:
        return pd.DataFrame(), {}

    sim_rows = []
    for trade_day, day_df in ops_df.sort_values("sequence_started_at").groupby("trade_day"):
        running = 0.0
        outcome = "ninguno"
        stop_after_operation = None
        month = day_df["month"].iloc[0]

        for _, row in day_df.iterrows():
            running += float(row["sequence_net_pnl_currency"] or 0)
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
                "month": month,
                "trade_day": trade_day,
                "resultado_diario_simulado": running,
                "resultado": outcome,
                "stop_after_operation": stop_after_operation,
            }
        )

    sim_df = pd.DataFrame(sim_rows)
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


def simulate_daily_sets(ops_df: pd.DataFrame, set_target: float, set_loss: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    User logic:
    - Operations are consumed chronologically.
    - A set starts from the next available operation.
    - Keep adding operation PnL until target or loss is reached.
    - When a set closes, the next operation starts a new set.
    """
    if ops_df.empty:
        return pd.DataFrame(), {}

    set_rows = []

    for trade_day, day_df in ops_df.sort_values("sequence_started_at").groupby("trade_day"):
        set_number = 1
        running = 0.0
        ops_in_set = 0
        set_start_time = None
        set_start_op = None
        month = day_df["month"].iloc[0]

        for _, row in day_df.iterrows():
            if ops_in_set == 0:
                set_start_time = row["sequence_started_at"]
                set_start_op = row["operation_id"]
                running = 0.0

            running += float(row["sequence_net_pnl_currency"] or 0)
            ops_in_set += 1

            outcome = None
            final_result = None

            if running >= set_target:
                outcome = "set_win"
                final_result = set_target
            elif running <= -set_loss:
                outcome = "set_loss"
                final_result = -set_loss

            if outcome:
                set_rows.append(
                    {
                        "month": month,
                        "trade_day": trade_day,
                        "set_number": set_number,
                        "set_start_time": set_start_time,
                        "set_end_time": row["sequence_started_at"],
                        "start_operation_id": set_start_op,
                        "end_operation_id": row["operation_id"],
                        "operations_used": ops_in_set,
                        "raw_running_pnl_at_trigger": running,
                        "set_result": final_result,
                        "outcome": outcome,
                    }
                )
                set_number += 1
                running = 0.0
                ops_in_set = 0
                set_start_time = None
                set_start_op = None

        if ops_in_set > 0:
            set_rows.append(
                {
                    "month": month,
                    "trade_day": trade_day,
                    "set_number": set_number,
                    "set_start_time": set_start_time,
                    "set_end_time": day_df.iloc[-1]["sequence_started_at"],
                    "start_operation_id": set_start_op,
                    "end_operation_id": day_df.iloc[-1]["operation_id"],
                    "operations_used": ops_in_set,
                    "raw_running_pnl_at_trigger": running,
                    "set_result": running,
                    "outcome": "open_end_day",
                }
            )

    sets_df = pd.DataFrame(set_rows)
    if sets_df.empty:
        return sets_df, {}

    metrics = {
        "sets": len(sets_df),
        "set_win_rate": (sets_df["outcome"] == "set_win").mean() * 100,
        "set_loss_rate": (sets_df["outcome"] == "set_loss").mean() * 100,
        "open_end_rate": (sets_df["outcome"] == "open_end_day").mean() * 100,
        "total_set_pnl": sets_df["set_result"].sum(),
        "avg_set_pnl": sets_df["set_result"].mean(),
        "median_set_pnl": sets_df["set_result"].median(),
        "avg_operations_per_set": sets_df["operations_used"].mean(),
        "best_set": sets_df["set_result"].max(),
        "worst_set": sets_df["set_result"].min(),
    }
    return sets_df, metrics


def calcular_resultado_simulado_por_cap_reversal(op_row: pd.Series, legs_df: pd.DataFrame, max_reversal_permitido: int) -> Dict[str, object]:
    operation_id = op_row["operation_id"]
    reversal_count_real = int(op_row["reversal_count"]) if pd.notna(op_row["reversal_count"]) else 0

    if reversal_count_real <= max_reversal_permitido:
        return {
            "reversal_count_simulado": reversal_count_real,
            "sequence_end_reason_simulado": op_row.get("sequence_end_reason"),
            "sequence_net_pnl_simulado": float(op_row.get("sequence_net_pnl_currency") or 0),
        }

    legs_op = legs_df.loc[legs_df["operation_id"] == operation_id].copy()
    if legs_op.empty:
        return {
            "reversal_count_simulado": max_reversal_permitido,
            "sequence_end_reason_simulado": f"CAP_{max_reversal_permitido}",
            "sequence_net_pnl_simulado": float(op_row.get("sequence_net_pnl_currency") or 0),
        }

    legs_op = legs_op.sort_values("leg_index")
    leg_target = legs_op.loc[legs_op["reversal_number"] == max_reversal_permitido].copy()

    if leg_target.empty:
        return {
            "reversal_count_simulado": max_reversal_permitido,
            "sequence_end_reason_simulado": f"CAP_{max_reversal_permitido}",
            "sequence_net_pnl_simulado": float(op_row.get("sequence_net_pnl_currency") or 0),
        }

    leg_target = leg_target.sort_values("leg_index").iloc[-1]
    pnl_simulado = leg_target.get("cumulative_sequence_pnl_after_leg")
    if pd.isna(pnl_simulado):
        pnl_simulado = op_row.get("sequence_net_pnl_currency") or 0

    exit_reason = leg_target.get("exit_reason")
    if pd.isna(exit_reason) or str(exit_reason).strip() == "":
        exit_reason = f"CAP_{max_reversal_permitido}"

    return {
        "reversal_count_simulado": max_reversal_permitido,
        "sequence_end_reason_simulado": str(exit_reason),
        "sequence_net_pnl_simulado": float(pnl_simulado),
    }


def apply_reversal_cap(ops_df: pd.DataFrame, legs_df: pd.DataFrame, max_reversal_permitido: int) -> pd.DataFrame:
    if ops_df.empty:
        return ops_df.copy()
    results = ops_df.apply(
        lambda row: calcular_resultado_simulado_por_cap_reversal(row, legs_df, max_reversal_permitido),
        axis=1,
    )
    results_df = pd.DataFrame(results.tolist(), index=ops_df.index)
    return pd.concat([ops_df.copy(), results_df], axis=1)


# ============================================================
# GLOBAL FILTERS
# ============================================================


def apply_global_filters(ops_df: pd.DataFrame, legs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros Globales")

    filtered_ops = ops_df.copy()

    months = sorted(filtered_ops["month"].dropna().unique().tolist())
    selected_months = st.sidebar.multiselect("Meses", months, default=months)
    if selected_months:
        filtered_ops = filtered_ops[filtered_ops["month"].isin(selected_months)]

    instruments = sorted(filtered_ops["instrument"].dropna().unique().tolist())
    selected_instruments = st.sidebar.multiselect("Instrumentos", instruments, default=instruments)
    if selected_instruments:
        filtered_ops = filtered_ops[filtered_ops["instrument"].isin(selected_instruments)]

    configs = sorted(filtered_ops["config_key"].dropna().unique().tolist())
    selected_configs = st.sidebar.multiselect("Configs", configs, default=configs)
    if selected_configs:
        filtered_ops = filtered_ops[filtered_ops["config_key"].isin(selected_configs)]

    if not filtered_ops.empty:
        min_dt = filtered_ops["sequence_started_at"].min().date()
        max_dt = filtered_ops["sequence_started_at"].max().date()
        c1, c2 = st.sidebar.columns(2)
        start_date = c1.date_input("Desde", value=min_dt)
        end_date = c2.date_input("Hasta", value=max_dt)
        filtered_ops = filtered_ops[
            (filtered_ops["sequence_started_at"].dt.date >= start_date)
            & (filtered_ops["sequence_started_at"].dt.date <= end_date)
        ]

    valid_op_ids = set(filtered_ops["operation_id"].dropna().astype(str).tolist())
    filtered_legs = legs_df.copy()
    if not filtered_legs.empty:
        filtered_legs = filtered_legs[filtered_legs["operation_id"].astype(str).isin(valid_op_ids)]

    st.sidebar.markdown("---")
    st.sidebar.metric("Operaciones visibles", f"{len(filtered_ops)}")
    st.sidebar.metric("Piernas visibles", f"{len(filtered_legs)}")

    return filtered_ops, filtered_legs


# ============================================================
# PAGES
# ============================================================


def render_dashboard_general(ops_df: pd.DataFrame):
    st.header("Dashboard General")
    mostrar_bloque_ayuda(
        "Dashboard General",
        "Esta página resume la salud global del bot, pero también la separa por mes y por día para evitar conclusiones falsas cuando cargas varios meses.",
        [
            "¿El bot está positivo de forma global?",
            "¿Todos los meses ayudan o un mes está escondiendo problemas?",
            "¿Los días malos son demasiado grandes?",
            "¿El resultado depende de una o dos operaciones gigantes?",
        ],
    )

    if ops_df.empty:
        st.warning("No hay operaciones con los filtros actuales.")
        return

    metrics = compute_overview_metrics(ops_df)

    row1 = st.columns(4)
    with row1[0]: card("Operaciones", f"{metrics['total_operations']}")
    with row1[1]: card("Días", f"{metrics['days']}")
    with row1[2]: card("PnL Neto", fmt_money(metrics["total_net_pnl"]))
    with row1[3]: card("Profit Factor", "-" if pd.isna(metrics["profit_factor"]) else f"{metrics['profit_factor']:.2f}")

    row2 = st.columns(4)
    with row2[0]: card("Win Rate", fmt_pct(metrics["win_rate"]))
    with row2[1]: card("Días Positivos", fmt_pct(metrics["positive_days_rate"]))
    with row2[2]: card("Mejor Día", fmt_money(metrics["best_day"]))
    with row2[3]: card("Peor Día", fmt_money(metrics["worst_day"]))

    row3 = st.columns(4)
    with row3[0]: card("Mejor Op", fmt_money(metrics["best_operation"]))
    with row3[1]: card("Peor Op", fmt_money(metrics["worst_operation"]))
    with row3[2]: card("Max DD Op", fmt_money(metrics["max_operation_drawdown"]))
    with row3[3]: card("Rev Promedio", f"{metrics['avg_reversals']:.2f}")

    st.markdown("---")
    st.subheader("Comparación mensual")
    monthly = aggregate_core(ops_df, ["month"])
    monthly = add_profit_factor_by_group(ops_df, monthly, ["month"])
    st.dataframe(monthly.sort_values("month"), use_container_width=True)

    if len(monthly) > 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(monthly["month"].astype(str), monthly["pnl_total"])
        ax.set_title("PnL por Mes")
        ax.set_xlabel("Mes")
        ax.set_ylabel("PnL")
        ax.tick_params(axis="x", rotation=25)
        st.pyplot(fig)

    st.subheader("Análisis diario")
    daily = ops_df.groupby(["month", "trade_day"], as_index=False).agg(
        operaciones=("operation_id", "count"),
        pnl_diario=("sequence_net_pnl_currency", "sum"),
        peor_operacion=("sequence_net_pnl_currency", "min"),
        mejor_operacion=("sequence_net_pnl_currency", "max"),
        drawdown_max=("operation_max_drawdown_currency", "max"),
        reversiones_promedio=("reversal_count", "mean"),
    ).sort_values("trade_day")

    st.dataframe(daily, use_container_width=True)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily["trade_day"].astype(str), daily["pnl_diario"], marker="o")
    ax.set_title("PnL Diario")
    ax.set_xlabel("Día")
    ax.set_ylabel("PnL")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

    st.subheader("Mejores y peores días")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Peores días**")
        st.dataframe(daily.sort_values("pnl_diario").head(10), use_container_width=True)
    with c2:
        st.markdown("**Mejores días**")
        st.dataframe(daily.sort_values("pnl_diario", ascending=False).head(10), use_container_width=True)

    lineas = []
    if metrics["total_net_pnl"] > 0:
        lineas.append("El bot está positivo en el rango filtrado.")
    else:
        lineas.append("El bot está negativo en el rango filtrado.")

    if pd.notna(metrics["profit_factor"]):
        if metrics["profit_factor"] >= 1.5:
            lineas.append("El profit factor se ve saludable.")
        elif metrics["profit_factor"] >= 1.0:
            lineas.append("El profit factor está positivo, pero todavía necesita control de riesgo.")
        else:
            lineas.append("El profit factor está débil; conviene revisar filtros, reversals y horarios.")

    if pd.notna(metrics["worst_day"]) and abs(metrics["worst_day"]) > abs(metrics["best_day"]) * 0.8:
        lineas.append("El peor día es grande comparado con el mejor día; hay que revisar los días peligrosos.")

    mostrar_conclusion("Dashboard General", lineas)


def render_tiempo_y_sesiones(ops_df: pd.DataFrame):
    st.header("Tiempo y Sesiones")
    mostrar_bloque_ayuda(
        "Tiempo y Sesiones",
        "Aquí mantenemos lo útil del análisis viejo: días de semana, horas y sesiones. Esta página responde en qué ventanas el bot tiene edge real.",
        [
            "¿Qué día de la semana aporta más o menos PnL?",
            "¿Qué hora destruye el resultado?",
            "¿Qué sesión exige más reversals o drawdown?",
            "¿Hay horarios que conviene bloquear?",
        ],
    )

    if ops_df.empty:
        st.warning("No hay operaciones con los filtros actuales.")
        return

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    st.subheader("Por día de la semana")
    weekday = aggregate_core(ops_df, ["dia_semana"])
    weekday["weekday_order"] = weekday["dia_semana"].apply(lambda x: weekday_order.index(x) if x in weekday_order else 99)
    weekday = weekday.sort_values("weekday_order").drop(columns=["weekday_order"])
    st.dataframe(weekday, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(weekday["dia_semana"].astype(str), weekday["pnl_total"])
    ax.set_title("PnL por Día de la Semana")
    ax.set_xlabel("Día")
    ax.set_ylabel("PnL")
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)

    st.subheader("Por hora")
    by_hour = aggregate_core(ops_df, ["hora_inicio"])
    by_hour = by_hour.sort_values("hora_inicio")
    st.dataframe(by_hour, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(by_hour["hora_inicio"].astype(str), by_hour["pnl_total"])
    ax.set_title("PnL por Hora de Inicio")
    ax.set_xlabel("Hora")
    ax.set_ylabel("PnL")
    st.pyplot(fig)

    st.subheader("Por sesión")
    session_df = aggregate_core(ops_df, ["sesion"])
    st.dataframe(session_df.sort_values("pnl_total", ascending=False), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(session_df["sesion"].astype(str), session_df["pnl_total"])
    ax.set_title("PnL por Sesión")
    ax.set_xlabel("Sesión")
    ax.set_ylabel("PnL")
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)

    if ops_df["month"].nunique() > 1:
        st.subheader("Sesión por mes")
        session_month = aggregate_core(ops_df, ["month", "sesion"])
        st.dataframe(session_month.sort_values(["month", "pnl_total"], ascending=[True, False]), use_container_width=True)

        st.subheader("Hora por mes")
        hour_month = aggregate_core(ops_df, ["month", "hora_inicio"])
        st.dataframe(hour_month.sort_values(["month", "hora_inicio"]), use_container_width=True)

    st.subheader("Ventanas peligrosas sugeridas")
    danger_hour = by_hour[(by_hour["pnl_total"] < 0) | (by_hour["drawdown_promedio"] > by_hour["drawdown_promedio"].median())]
    st.dataframe(danger_hour.sort_values("pnl_total"), use_container_width=True)

    lineas = []
    if not by_hour.empty:
        best_hour = by_hour.sort_values("pnl_total", ascending=False).iloc[0]
        worst_hour = by_hour.sort_values("pnl_total", ascending=True).iloc[0]
        lineas.append(f"Mejor hora por PnL total: {int(best_hour['hora_inicio'])}:00.")
        lineas.append(f"Peor hora por PnL total: {int(worst_hour['hora_inicio'])}:00.")
    if not session_df.empty:
        best_session = session_df.sort_values("pnl_total", ascending=False).iloc[0]
        worst_session = session_df.sort_values("pnl_total", ascending=True).iloc[0]
        lineas.append(f"Mejor sesión por PnL total: {best_session['sesion']}.")
        lineas.append(f"Peor sesión por PnL total: {worst_session['sesion']}.")
    mostrar_conclusion("Tiempo y Sesiones", lineas)


def render_motor_reversiones(ops_df: pd.DataFrame, legs_df: pd.DataFrame):
    st.header("Motor de Reversiones")
    mostrar_bloque_ayuda(
        "Motor de Reversiones",
        "Esta página junta el viejo Motor de Reversiones y Reversal Quality. Aquí decidimos cuántas reversiones valen la pena y cuándo se vuelven peligrosas.",
        [
            "¿Qué profundidad de reversal aporta PnL?",
            "¿Dónde sube demasiado el drawdown?",
            "¿Qué pasa si capamos el bot en 0, 1, 2, 3 o 4 reversals?",
            "¿Las reversiones realmente recuperan o solo agrandan el riesgo?",
        ],
    )

    if ops_df.empty:
        st.warning("No hay operaciones con los filtros actuales.")
        return

    st.subheader("Resumen por cantidad real de reversals")
    rev = aggregate_core(ops_df, ["reversal_count"])
    rev = add_profit_factor_by_group(ops_df, rev, ["reversal_count"])
    st.dataframe(rev.sort_values("reversal_count"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(rev["reversal_count"].fillna(0).astype(int).astype(str), rev["pnl_total"])
    ax.set_title("PnL por Cantidad de Reversals")
    ax.set_xlabel("Reversals")
    ax.set_ylabel("PnL")
    st.pyplot(fig)

    if not legs_df.empty:
        st.subheader("Calidad por pierna / reversal number")
        leg_quality = legs_df.groupby("reversal_number", dropna=False).agg(
            piernas=("operation_id", "count"),
            pnl_total=("realized_pnl_currency", "sum"),
            pnl_promedio=("realized_pnl_currency", "mean"),
            qty_promedio=("entry_qty", "mean"),
            qty_max=("entry_qty", "max"),
            dd_promedio=("operation_drawdown_after_leg", "mean"),
            runup_promedio=("operation_runup_after_leg", "mean"),
        ).reset_index().sort_values("reversal_number")
        st.dataframe(leg_quality, use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(leg_quality["reversal_number"].fillna(0).astype(int).astype(str), leg_quality["pnl_promedio"])
        ax.set_title("PnL Promedio por Pierna/Reversal")
        ax.set_xlabel("Reversal Number")
        ax.set_ylabel("PnL Promedio")
        st.pyplot(fig)

    st.subheader("Simulación de cap de reversals")
    max_real_rev = int(ops_df["reversal_count"].fillna(0).max())
    cap_options = list(range(0, max_real_rev + 1))
    if len(cap_options) == 0:
        cap_options = [0]

    cap_rows = []
    for cap in cap_options:
        capped = apply_reversal_cap(ops_df, legs_df, cap)
        pnl_col = capped["sequence_net_pnl_simulado"].fillna(0)
        winners = pnl_col[pnl_col > 0]
        losers = pnl_col[pnl_col < 0]
        cap_rows.append(
            {
                "max_reversal_cap": cap,
                "operaciones": len(capped),
                "pnl_simulado": pnl_col.sum(),
                "pnl_promedio": pnl_col.mean(),
                "win_rate": (pnl_col > 0).mean() * 100,
                "profit_factor": safe_div(winners.sum(), abs(losers.sum())),
                "peor_operacion_simulada": pnl_col.min(),
                "mejor_operacion_simulada": pnl_col.max(),
            }
        )
    cap_df = pd.DataFrame(cap_rows)
    st.dataframe(cap_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(cap_df["max_reversal_cap"].astype(str), cap_df["pnl_simulado"])
    ax.set_title("PnL Simulado por Cap de Reversals")
    ax.set_xlabel("Cap")
    ax.set_ylabel("PnL Simulado")
    st.pyplot(fig)

    st.subheader("Operaciones peligrosas por reversals")
    dangerous = ops_df.sort_values(["reversal_count", "operation_max_drawdown_currency"], ascending=[False, False]).head(20)
    st.dataframe(
        dangerous[
            [
                "month", "trade_day", "operation_id", "sequence_started_at", "reversal_count",
                "sequence_net_pnl_currency", "operation_max_drawdown_currency", "sequence_end_reason",
                "base_contracts", "config_key", "sesion", "hora_inicio",
            ]
        ],
        use_container_width=True,
    )

    lineas = []
    if not rev.empty:
        best_rev = rev.sort_values("pnl_promedio", ascending=False).iloc[0]
        worst_rev = rev.sort_values("pnl_promedio", ascending=True).iloc[0]
        lineas.append(f"Mejor profundidad por PnL promedio: {int(best_rev['reversal_count']) if pd.notna(best_rev['reversal_count']) else 0} reversal(es).")
        lineas.append(f"Peor profundidad por PnL promedio: {int(worst_rev['reversal_count']) if pd.notna(worst_rev['reversal_count']) else 0} reversal(es).")
    if not cap_df.empty:
        best_cap = cap_df.sort_values("pnl_simulado", ascending=False).iloc[0]
        lineas.append(f"Mejor cap simulado por PnL total: {int(best_cap['max_reversal_cap'])} reversal(es).")
    mostrar_conclusion("Motor de Reversiones", lineas)


def render_simulador_diario(ops_df: pd.DataFrame):
    st.header("Simulador Diario")
    mostrar_bloque_ayuda(
        "Simulador Diario",
        "Esta página junta la meta diaria clásica y la simulación por sets. La de sets sigue tu lógica: consumir operaciones en orden hasta target/loss y luego empezar otro set con la siguiente operación.",
        [
            "¿Qué meta diaria se alcanza antes que la pérdida?",
            "¿Qué target/loss por set funciona mejor?",
            "¿Cuántas operaciones consume cada set?",
            "¿La lógica sobrevive mes por mes?",
        ],
    )

    if ops_df.empty:
        st.warning("No hay operaciones con los filtros actuales.")
        return

    st.subheader("A. Stop diario clásico")
    c1, c2 = st.columns(2)
    daily_target = c1.number_input("Meta Diaria", min_value=1.0, value=600.0, step=50.0)
    daily_loss = c2.number_input("Pérdida Máxima Diaria", min_value=1.0, value=300.0, step=50.0)

    sim_df, metrics = simulate_daily_stop(ops_df, daily_target, daily_loss)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Meta Primero %", fmt_pct(metrics.get("target_first_rate", np.nan)))
    c2.metric("Pérdida Primero %", fmt_pct(metrics.get("loss_first_rate", np.nan)))
    c3.metric("Ninguno %", fmt_pct(metrics.get("neither_rate", np.nan)))
    c4.metric("PnL Simulado", fmt_money(metrics.get("total_simulated_pnl", np.nan)))

    if not sim_df.empty:
        st.dataframe(sim_df.sort_values("trade_day"), use_container_width=True)
        monthly_stop = sim_df.groupby("month", as_index=False).agg(
            dias=("trade_day", "count"),
            pnl_simulado=("resultado_diario_simulado", "sum"),
            promedio_diario=("resultado_diario_simulado", "mean"),
            meta_primero_pct=("resultado", lambda s: (s == "meta_primero").mean() * 100),
            perdida_primero_pct=("resultado", lambda s: (s == "perdida_primero").mean() * 100),
        )
        st.markdown("**Breakdown mensual del stop diario**")
        st.dataframe(monthly_stop, use_container_width=True)

    st.markdown("---")
    st.subheader("B. Simulador de sets diarios")
    c1, c2 = st.columns(2)
    set_target = c1.number_input("Target por Set", min_value=1.0, value=600.0, step=50.0)
    set_loss = c2.number_input("Loss por Set", min_value=1.0, value=300.0, step=50.0)

    sets_df, set_metrics = simulate_daily_sets(ops_df, set_target, set_loss)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sets", f"{set_metrics.get('sets', 0)}")
    c2.metric("Set Win %", fmt_pct(set_metrics.get("set_win_rate", np.nan)))
    c3.metric("Set Loss %", fmt_pct(set_metrics.get("set_loss_rate", np.nan)))
    c4.metric("PnL Sets", fmt_money(set_metrics.get("total_set_pnl", np.nan)))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Promedio Set", fmt_money(set_metrics.get("avg_set_pnl", np.nan)))
    c6.metric("Ops/Set Prom", f"{set_metrics.get('avg_operations_per_set', np.nan):.2f}" if set_metrics else "-")
    c7.metric("Mejor Set", fmt_money(set_metrics.get("best_set", np.nan)))
    c8.metric("Peor Set", fmt_money(set_metrics.get("worst_set", np.nan)))

    if not sets_df.empty:
        st.dataframe(sets_df.sort_values(["trade_day", "set_number"]), use_container_width=True)
        monthly_sets = sets_df.groupby("month", as_index=False).agg(
            sets=("set_number", "count"),
            pnl_sets=("set_result", "sum"),
            promedio_set=("set_result", "mean"),
            set_win_pct=("outcome", lambda s: (s == "set_win").mean() * 100),
            set_loss_pct=("outcome", lambda s: (s == "set_loss").mean() * 100),
            ops_por_set=("operations_used", "mean"),
        )
        st.markdown("**Breakdown mensual de sets**")
        st.dataframe(monthly_sets, use_container_width=True)

    lineas = []
    if metrics:
        if metrics["target_first_rate"] > metrics["loss_first_rate"]:
            lineas.append("En el stop diario clásico, la meta se alcanza antes que la pérdida con mayor frecuencia.")
        else:
            lineas.append("En el stop diario clásico, la pérdida aparece demasiado fuerte frente a la meta.")
    if set_metrics:
        if set_metrics["set_win_rate"] > set_metrics["set_loss_rate"]:
            lineas.append("En la simulación por sets, los sets ganadores superan a los sets perdedores.")
        else:
            lineas.append("En la simulación por sets, el ratio todavía no se ve cómodo.")
    mostrar_conclusion("Simulador Diario", lineas)


def render_laboratorio_parametros(ops_df: pd.DataFrame):
    st.header("Laboratorio de Parámetros")
    mostrar_bloque_ayuda(
        "Laboratorio de Parámetros",
        "Aquí se comparan settings reales: base contracts, SL, TP, distance, reversals y config completa. También sirve para comparar 2, 3, 5, 7, 10 contratos.",
        [
            "¿Qué base contracts tiene mejor balance entre PnL y riesgo?",
            "¿Qué config completa es más estable por mes?",
            "¿Subir contratos mejora el PnL o agranda demasiado el peor caso?",
            "¿Qué setting funciona en varios meses y no solo en uno?",
        ],
    )

    if ops_df.empty:
        st.warning("No hay operaciones con los filtros actuales.")
        return

    param_options = [
        "base_contracts",
        "fixed_stop_ticks",
        "fixed_target_ticks",
        "distance_points",
        "max_reversals_allowed",
        "config_key",
    ]
    param = st.selectbox("Comparar por", param_options)

    grouped = aggregate_core(ops_df, [param])
    grouped = add_profit_factor_by_group(ops_df, grouped, [param])
    grouped = grouped.sort_values("pnl_total", ascending=False)
    st.dataframe(grouped, use_container_width=True)

    if len(grouped) > 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(grouped[param].astype(str), grouped["pnl_total"])
        ax.set_title(f"PnL por {param}")
        ax.set_xlabel(param)
        ax.set_ylabel("PnL")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    st.subheader("Estabilidad mensual por parámetro")
    if ops_df["month"].nunique() > 1:
        monthly_param = aggregate_core(ops_df, ["month", param])
        monthly_param = add_profit_factor_by_group(ops_df, monthly_param, ["month", param])
        st.dataframe(monthly_param.sort_values([param, "month"]), use_container_width=True)
    else:
        st.info("Cuando cargues más de un mes, aquí se verá si el setting funciona de forma estable o solo fue suerte de un mes.")

    st.subheader("Contract Exposure")
    contract_df = aggregate_core(ops_df, ["base_contracts"])
    contract_df = add_profit_factor_by_group(ops_df, contract_df, ["base_contracts"])
    st.dataframe(contract_df.sort_values("base_contracts"), use_container_width=True)

    lineas = []
    if not grouped.empty:
        best = grouped.sort_values("pnl_total", ascending=False).iloc[0]
        worst_risk = grouped.sort_values("peor_operacion", ascending=True).iloc[0]
        lineas.append(f"Mejor valor por PnL total para {param}: {best[param]}.")
        lineas.append(f"Valor con peor operación más agresiva para {param}: {worst_risk[param]}.")
    mostrar_conclusion("Laboratorio de Parámetros", lineas)


def render_risk_killers(ops_df: pd.DataFrame, legs_df: pd.DataFrame):
    st.header("Risk Killers")
    mostrar_bloque_ayuda(
        "Risk Killers",
        "Esta página busca lo que más está dañando el bot. La idea no es ver todo, sino encontrar qué bloquear o limitar primero.",
        [
            "¿Cuáles son las peores operaciones?",
            "¿Qué días tuvieron más daño?",
            "¿Qué horas/sesiones son peligrosas?",
            "¿Qué operaciones terminaron positivas pero con drawdown feo?",
        ],
    )

    if ops_df.empty:
        st.warning("No hay operaciones con los filtros actuales.")
        return

    st.subheader("Peores operaciones")
    worst_ops = ops_df.sort_values("sequence_net_pnl_currency").head(20)
    st.dataframe(
        worst_ops[
            [
                "month", "trade_day", "operation_id", "sequence_started_at", "sequence_net_pnl_currency",
                "operation_max_drawdown_currency", "reversal_count", "base_contracts", "sequence_end_reason",
                "sesion", "hora_inicio", "config_key",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Días más peligrosos")
    daily = ops_df.groupby(["month", "trade_day"], as_index=False).agg(
        operaciones=("operation_id", "count"),
        pnl_diario=("sequence_net_pnl_currency", "sum"),
        peor_operacion=("sequence_net_pnl_currency", "min"),
        drawdown_max=("operation_max_drawdown_currency", "max"),
        reversiones_promedio=("reversal_count", "mean"),
        reversiones_max=("reversal_count", "max"),
    )
    st.dataframe(daily.sort_values(["pnl_diario", "drawdown_max"], ascending=[True, False]).head(20), use_container_width=True)

    st.subheader("Ganadoras peligrosas")
    dd_threshold = st.number_input("Drawdown mínimo para marcar ganadora peligrosa", min_value=0.0, value=300.0, step=50.0)
    risky_winners = ops_df[(ops_df["sequence_net_pnl_currency"] > 0) & (ops_df["operation_max_drawdown_currency"] >= dd_threshold)]
    st.dataframe(
        risky_winners.sort_values("operation_max_drawdown_currency", ascending=False)[
            [
                "month", "trade_day", "operation_id", "sequence_started_at", "sequence_net_pnl_currency",
                "operation_max_drawdown_currency", "reversal_count", "base_contracts", "sesion", "hora_inicio", "config_key",
            ]
        ].head(30),
        use_container_width=True,
    )

    st.subheader("Peores horas y sesiones")
    c1, c2 = st.columns(2)
    with c1:
        by_hour = aggregate_core(ops_df, ["hora_inicio"])
        st.dataframe(by_hour.sort_values("pnl_total").head(10), use_container_width=True)
    with c2:
        by_session = aggregate_core(ops_df, ["sesion"])
        st.dataframe(by_session.sort_values("pnl_total").head(10), use_container_width=True)

    if not legs_df.empty:
        st.subheader("Mayor exposición de contratos por pierna")
        exposure = legs_df.sort_values("entry_qty", ascending=False).head(30)
        st.dataframe(
            exposure[
                [
                    "month", "trade_day", "operation_id", "leg_index", "reversal_number", "direction",
                    "entry_qty", "realized_pnl_currency", "cumulative_sequence_pnl_after_leg", "exit_reason",
                    "sesion", "hora_inicio", "config_key",
                ]
            ],
            use_container_width=True,
        )

    lineas = []
    if not worst_ops.empty:
        w = worst_ops.iloc[0]
        lineas.append(f"Peor operación: {w['operation_id']} con PnL {fmt_money(w['sequence_net_pnl_currency'])}.")
    if not daily.empty:
        d = daily.sort_values("pnl_diario").iloc[0]
        lineas.append(f"Peor día: {d['trade_day']} con PnL {fmt_money(d['pnl_diario'])}.")
    mostrar_conclusion("Risk Killers", lineas)


def render_explorador_operaciones(ops_df: pd.DataFrame, legs_df: pd.DataFrame):
    st.header("Explorador de Operaciones")
    mostrar_bloque_ayuda(
        "Explorador de Operaciones",
        "Aquí se inspecciona una operación individual con todas sus piernas, PnL acumulado, contratos y simulación por cap de reversals.",
        [
            "¿Cómo evolucionó el PnL pierna por pierna?",
            "¿Cuántos contratos usó cada reversal?",
            "¿Qué habría pasado con un cap menor?",
            "¿La operación fue sana o una recuperación peligrosa?",
        ],
    )

    if ops_df.empty:
        st.warning("No hay operaciones con los filtros actuales.")
        return

    filtro_df = ops_df.copy()

    st.subheader("Filtros internos")
    c1, c2, c3 = st.columns(3)

    with c1:
        sesiones = sorted(filtro_df["sesion"].dropna().unique().tolist())
        sesiones_sel = st.multiselect("Sesiones", sesiones, default=sesiones)
    with c2:
        razones = sorted(filtro_df["sequence_end_reason"].dropna().unique().tolist())
        razones_sel = st.multiselect("Razón de cierre", razones, default=razones)
    with c3:
        max_rev_available = int(filtro_df["reversal_count"].fillna(0).max())
        max_reversal_permitido = st.selectbox("Máximo reversal simulado", list(range(0, max_rev_available + 1)), index=max_rev_available)

    text_search = st.text_input("Buscar operation_id", "")

    if sesiones_sel:
        filtro_df = filtro_df[filtro_df["sesion"].isin(sesiones_sel)]
    if razones_sel:
        filtro_df = filtro_df[filtro_df["sequence_end_reason"].isin(razones_sel)]
    if text_search.strip():
        filtro_df = filtro_df[filtro_df["operation_id"].astype(str).str.contains(text_search.strip(), case=False, na=False)]

    if filtro_df.empty:
        st.warning("No hay operaciones con esos filtros.")
        return

    filtro_df = apply_reversal_cap(filtro_df, legs_df, max_reversal_permitido)

    c1, c2, c3 = st.columns(3)
    c1.metric("Operaciones visibles", f"{len(filtro_df)}")
    c2.metric("PnL Real", fmt_money(filtro_df["sequence_net_pnl_currency"].sum()))
    c3.metric("PnL Simulado Cap", fmt_money(filtro_df["sequence_net_pnl_simulado"].sum()))

    display_cols = [
        "month", "trade_day", "operation_id", "sequence_started_at", "sequence_end_reason", "sequence_end_reason_simulado",
        "reversal_count", "reversal_count_simulado", "sequence_net_pnl_currency", "sequence_net_pnl_simulado",
        "operation_max_drawdown_currency", "operation_max_runup_currency", "base_contracts", "sesion", "hora_inicio", "config_key",
    ]
    st.dataframe(filtro_df[display_cols].sort_values("sequence_started_at", ascending=False), use_container_width=True)

    op_ids = filtro_df.sort_values("sequence_started_at", ascending=False)["operation_id"].dropna().astype(str).tolist()
    selected_op = st.selectbox("Inspeccionar operación", op_ids)

    if not selected_op:
        return

    op_row = filtro_df.loc[filtro_df["operation_id"].astype(str) == str(selected_op)]
    if op_row.empty:
        return

    st.subheader("Resumen de operación")
    st.dataframe(op_row, use_container_width=True)

    st.subheader("Piernas")
    legs_op = legs_df.loc[legs_df["operation_id"].astype(str) == str(selected_op)].sort_values("leg_index")
    st.dataframe(legs_op, use_container_width=True)

    if not legs_op.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(legs_op["leg_index"].astype(str), legs_op["cumulative_sequence_pnl_after_leg"], marker="o")
        ax.set_title("PnL acumulado por pierna")
        ax.set_xlabel("Pierna")
        ax.set_ylabel("PnL acumulado")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(legs_op["leg_index"].astype(str), legs_op["entry_qty"])
        ax.set_title("Contratos por pierna")
        ax.set_xlabel("Pierna")
        ax.set_ylabel("Contratos")
        st.pyplot(fig)

    fila = op_row.iloc[0]
    st.subheader("Lectura rápida")
    st.markdown(f"- Reversals reales: **{int(fila['reversal_count']) if pd.notna(fila['reversal_count']) else 0}**")
    st.markdown(f"- Reversals simulados con cap {max_reversal_permitido}: **{int(fila['reversal_count_simulado']) if pd.notna(fila['reversal_count_simulado']) else 0}**")
    st.markdown(f"- PnL real: **{fmt_money(fila['sequence_net_pnl_currency'])}**")
    st.markdown(f"- PnL simulado: **{fmt_money(fila['sequence_net_pnl_simulado'])}**")
    st.markdown(f"- Drawdown operación: **{fmt_money(fila['operation_max_drawdown_currency'])}**")
    st.markdown(f"- Sesión: **{fila['sesion']}** | Hora: **{fila['hora_inicio']}:00**")


# ============================================================
# MAIN
# ============================================================


def main():
    st.title("Laboratorio WLF")
    st.caption("Lab limpio: global, mensual, diario, tiempo/sesiones, reversals, simulación, parámetros, riesgo y operación detallada.")

    uploaded_files = st.sidebar.file_uploader(
        "Cargar archivos JSONL mensuales",
        type=["jsonl"],
        accept_multiple_files=True,
    )

    records = load_uploaded_jsonl_files(uploaded_files)
    ops_df, legs_df = build_dataframes(records)

    if ops_df.empty:
        st.warning("No se cargaron registros JSONL. Sube uno o más archivos JSONL.")
        return

    ops_filtered, legs_filtered = apply_global_filters(ops_df, legs_df)

    page = st.sidebar.radio(
        "Página",
        [
            "Dashboard General",
            "Tiempo y Sesiones",
            "Motor de Reversiones",
            "Simulador Diario",
            "Laboratorio de Parámetros",
            "Risk Killers",
            "Explorador de Operaciones",
        ],
    )

    if page == "Dashboard General":
        render_dashboard_general(ops_filtered)
    elif page == "Tiempo y Sesiones":
        render_tiempo_y_sesiones(ops_filtered)
    elif page == "Motor de Reversiones":
        render_motor_reversiones(ops_filtered, legs_filtered)
    elif page == "Simulador Diario":
        render_simulador_diario(ops_filtered)
    elif page == "Laboratorio de Parámetros":
        render_laboratorio_parametros(ops_filtered)
    elif page == "Risk Killers":
        render_risk_killers(ops_filtered, legs_filtered)
    elif page == "Explorador de Operaciones":
        render_explorador_operaciones(ops_filtered, legs_filtered)


if __name__ == "__main__":
    main()
