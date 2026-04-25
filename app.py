import json
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


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
# SIMULATIONS
# ============================================================


def _prepare_leg_timeline(legs_df: pd.DataFrame) -> pd.DataFrame:
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

    if "trade_day" not in legs.columns or legs["trade_day"].isna().all():
        legs["trade_day"] = legs["event_time"].dt.date

    legs["leg_pnl"] = pd.to_numeric(legs["realized_pnl_currency"], errors="coerce").fillna(0.0)
    legs["leg_sort"] = pd.to_numeric(legs["leg_index"], errors="coerce").fillna(0)

    return legs.sort_values(["trade_day", "event_time", "operation_id", "leg_sort"]).copy()


def simulate_daily_stop(legs_df: pd.DataFrame, daily_target: float, daily_loss: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Classic daily stop using real legs.

    The simulator consumes legs chronologically. After every closed leg, it checks
    whether the accumulated realized PnL reached the target or loss.
    """
    rows = []
    legs = _prepare_leg_timeline(legs_df)
    if legs.empty:
        return pd.DataFrame(), {}

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
                reason = "Target reached"
                running = daily_target
                break
            if running <= -daily_loss:
                reason = "Loss reached"
                running = -daily_loss
                break

        month_value = day_df["month"].dropna().iloc[0] if "month" in day_df.columns and not day_df["month"].dropna().empty else ""
        rows.append(
            {
                "month": month_value,
                "trade_day": trade_day,
                "simulated_day_pnl": running,
                "raw_pnl_at_stop": raw_pnl_at_stop,
                "result": "Win" if running > 0 else "Loss" if running < 0 else "Flat",
                "stop_reason": reason,
                "legs_used": legs_used,
                "operations_touched": len(touched_ops),
                "stopped_after_operation": stopped_after_operation,
                "stopped_after_leg": stopped_after_leg,
                "stopped_at_time": stopped_at_time,
                "real_day_pnl_legs": day_df["leg_pnl"].sum(),
                "real_legs": len(day_df),
                "real_operations_touched": day_df["operation_id"].nunique(),
            }
        )

    sim = pd.DataFrame(rows)
    metrics = {
        "days": len(sim),
        "total_pnl": sim["simulated_day_pnl"].sum(),
        "avg_day": sim["simulated_day_pnl"].mean(),
        "target_days_pct": (sim["stop_reason"] == "Target reached").mean() * 100,
        "loss_days_pct": (sim["stop_reason"] == "Loss reached").mean() * 100,
        "open_days_pct": (sim["stop_reason"] == "End of day").mean() * 100,
        "winning_days_pct": (sim["simulated_day_pnl"] > 0).mean() * 100,
        "best_day": sim["simulated_day_pnl"].max(),
        "worst_day": sim["simulated_day_pnl"].min(),
        "avg_legs_used": sim["legs_used"].mean(),
    }
    return sim, metrics


def simulate_daily_sets(legs_df: pd.DataFrame, set_target: float, set_loss: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Daily set simulator using legs.

    A set consumes real legs in chronological order. When the running set PnL
    reaches target/loss, that set closes and the next leg starts a new set.
    """
    rows = []
    legs = _prepare_leg_timeline(legs_df)
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



def calcular_resultado_simulado_por_cap_reversal(
    op_row: pd.Series,
    legs_df: pd.DataFrame,
    max_reversal_permitido: int,
) -> Dict[str, object]:
    """Simula el resultado si la operación se hubiera cortado al máximo reversal permitido."""
    real_pnl = float(op_row.get("sequence_net_pnl_currency", 0) or 0)
    real_rev = int(op_row.get("reversal_count", 0) or 0)

    if real_rev <= max_reversal_permitido:
        return {
            "reversal_count_simulado": real_rev,
            "sequence_net_pnl_simulado": real_pnl,
            "sequence_end_reason_simulado": op_row.get("sequence_end_reason", "Real result"),
            "cap_aplicado": False,
        }

    operation_id = str(op_row.get("operation_id", ""))
    legs_op = legs_df[legs_df["operation_id"].astype(str) == operation_id].copy() if not legs_df.empty else pd.DataFrame()

    if legs_op.empty or "reversal_number" not in legs_op.columns:
        return {
            "reversal_count_simulado": max_reversal_permitido,
            "sequence_net_pnl_simulado": real_pnl,
            "sequence_end_reason_simulado": f"CAP_{max_reversal_permitido}_SIN_LEGS",
            "cap_aplicado": True,
        }

    legs_op = legs_op.sort_values("leg_index")
    target_leg = legs_op[legs_op["reversal_number"].fillna(-1).astype(int) == int(max_reversal_permitido)].copy()

    if target_leg.empty:
        return {
            "reversal_count_simulado": max_reversal_permitido,
            "sequence_net_pnl_simulado": real_pnl,
            "sequence_end_reason_simulado": f"CAP_{max_reversal_permitido}_SIN_PIERNA",
            "cap_aplicado": True,
        }

    last_leg = target_leg.sort_values("leg_index").iloc[-1]
    pnl_sim = last_leg.get("cumulative_sequence_pnl_after_leg", np.nan)
    if pd.isna(pnl_sim):
        pnl_sim = real_pnl

    exit_reason = last_leg.get("exit_reason", "")
    if pd.isna(exit_reason) or str(exit_reason).strip() == "":
        exit_reason = f"CAP_{max_reversal_permitido}"

    return {
        "reversal_count_simulado": max_reversal_permitido,
        "sequence_net_pnl_simulado": float(pnl_sim),
        "sequence_end_reason_simulado": str(exit_reason),
        "cap_aplicado": True,
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
    return pd.concat([base, resultados_df], axis=1)



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

    st.sidebar.metric("Operaciones filtradas", len(filtered_ops))
    st.sidebar.metric("Piernas filtradas", len(filtered_legs))

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
    # MAX LOST CONTINUE - visible at the TOP of the dashboard
    # ========================================================
    st.subheader("Max Lost Continue")
    section_note(
        "Este control está arriba porque es clave para riesgo: muestra las peores rachas de pérdidas consecutivas. "
        "Por defecto usa piernas reales, porque cada pierna es una entrada/salida real."
    )

    c1, c2 = st.columns(2)
    loss_streak_mode = c1.selectbox(
        "Analizar pérdida continua por",
        options=["Piernas reales", "Operaciones completas"],
        index=0,
        key="dashboard_loss_streak_mode",
    )
    max_loss_streak_rows = c2.selectbox(
        "Max dropdown · rachas a mostrar",
        options=[5, 10, 15, 25, 50, 100],
        index=1,
        key="dashboard_max_loss_streak_rows",
        help="Controla cuántas rachas de pérdida continua quieres ver en la tabla.",
    )

    if loss_streak_mode == "Piernas reales":
        streak_df = build_consecutive_loss_streaks(
            legs_df,
            pnl_col="realized_pnl_currency",
            time_col="exit_time",
            id_col="operation_id",
            unit_label="Pierna",
        )
    else:
        streak_df = build_consecutive_loss_streaks(
            ops_df,
            pnl_col="sequence_net_pnl_currency",
            time_col="sequence_started_at",
            id_col="operation_id",
            unit_label="Operación",
        )

    if streak_df.empty:
        st.info("No se encontraron rachas de pérdidas con los filtros actuales.")
    else:
        c = st.columns(4)
        with c[0]: card("Máx pérdidas seguidas", f"{int(streak_df['losses_in_a_row'].max())}")
        with c[1]: card("Peor racha", fmt_money(streak_df["total_streak_loss"].min()))
        with c[2]: card("Rachas encontradas", f"{len(streak_df)}")
        with c[3]: card("Peor pérdida individual", fmt_money(streak_df["worst_single_loss"].min()))

        show_cols = [
            "month", "trade_day_start", "start_time", "end_time", "unit",
            "losses_in_a_row", "total_streak_loss", "worst_single_loss", "ids",
        ]
        st.dataframe(
            streak_df[show_cols].head(int(max_loss_streak_rows)),
            use_container_width=True,
        )

    st.markdown("---")

    # ========================================================
    # MAIN HEALTH CARDS
    # ========================================================
    st.subheader("Salud General")
    m = overview_metrics(ops_df)
    c = st.columns(4)
    with c[0]: card("Operaciones", f"{m['ops']}")
    with c[1]: card("Días", f"{m['days']}")
    with c[2]: card("PnL Total", fmt_money(m["pnl"]))
    with c[3]: card("Profit Factor", "-" if pd.isna(m["profit_factor"]) else f"{m['profit_factor']:.2f}")

    c = st.columns(4)
    with c[0]: card("Win Rate", fmt_pct(m["win_rate"]))
    with c[1]: card("Días Positivos", fmt_pct(m["positive_days_rate"]))
    with c[2]: card("Peor Operación", fmt_money(m["worst_op"]))
    with c[3]: card("Peor Día", fmt_money(m["worst_day"]))

    st.markdown("---")
    st.subheader("Simulación rápida por máximo reversal permitido")
    section_note(
        "Este control cambia el máximo reversal permitido y recalcula el PnL como si el bot se hubiera detenido en ese reversal. "
        "El cálculo usa las piernas de la operación para tomar el PnL acumulado hasta el reversal seleccionado."
    )

    max_real_rev = int(ops_df["reversal_count"].fillna(0).max()) if "reversal_count" in ops_df.columns and not ops_df.empty else 0
    dashboard_max_reversal = st.selectbox(
        "Máximo reversal permitido",
        options=list(range(0, max_real_rev + 1)),
        index=max_real_rev,
        key="dashboard_max_reversal_permitido",
        help="Ejemplo: si eliges 2, las operaciones que llegaron a reversal 3 o más se simulan cortadas después del reversal 2.",
    )

    dashboard_sim = aplicar_cap_reversal(ops_df, legs_df, dashboard_max_reversal)
    real_pnl = ops_df["sequence_net_pnl_currency"].sum()
    sim_pnl = dashboard_sim["sequence_net_pnl_simulado"].sum()
    pnl_diff = sim_pnl - real_pnl
    affected_ops = int(dashboard_sim["cap_aplicado"].sum()) if "cap_aplicado" in dashboard_sim.columns else 0
    worst_sim = dashboard_sim["sequence_net_pnl_simulado"].min() if "sequence_net_pnl_simulado" in dashboard_sim.columns else np.nan

    c = st.columns(4)
    with c[0]: card("PnL Real", fmt_money(real_pnl))
    with c[1]: card("PnL con Max Reversal", fmt_money(sim_pnl))
    with c[2]: card("Diferencia", fmt_money(pnl_diff))
    with c[3]: card("Ops Afectadas", str(affected_ops))

    c = st.columns(4)
    with c[0]: card("Peor Op Real", fmt_money(m["worst_op"]))
    with c[1]: card("Peor Op Simulada", fmt_money(worst_sim))
    with c[2]: card("Max Reversal Real", str(max_real_rev))
    with c[3]: card("Max Reversal Usado", str(dashboard_max_reversal))

    sim_month = dashboard_sim.groupby("month", as_index=False).agg(
        pnl_real=("sequence_net_pnl_currency", "sum"),
        pnl_simulado=("sequence_net_pnl_simulado", "sum"),
        operaciones=("operation_id", "count"),
        operaciones_afectadas=("cap_aplicado", "sum"),
    ) if "month" in dashboard_sim.columns and not dashboard_sim.empty else pd.DataFrame()
    if not sim_month.empty:
        sim_month["diferencia"] = sim_month["pnl_simulado"] - sim_month["pnl_real"]
        st.markdown("**Impacto mensual del máximo reversal permitido**")
        st.dataframe(sim_month.sort_values("month"), use_container_width=True)

    changed_dashboard = dashboard_sim[dashboard_sim["cap_aplicado"] == True].copy() if "cap_aplicado" in dashboard_sim.columns else pd.DataFrame()
    if not changed_dashboard.empty:
        changed_dashboard["diferencia"] = changed_dashboard["sequence_net_pnl_simulado"] - changed_dashboard["sequence_net_pnl_currency"]
        show_cols = [
            "month", "trade_day", "operation_id", "sequence_started_at",
            "reversal_count", "reversal_count_simulado",
            "sequence_net_pnl_currency", "sequence_net_pnl_simulado", "diferencia",
            "operation_max_drawdown_currency", "base_contracts", "sesion", "hora_inicio",
        ]
        st.markdown("**Operaciones cambiadas por el máximo reversal permitido**")
        st.dataframe(changed_dashboard[show_cols].sort_values("diferencia"), use_container_width=True)

    st.markdown("---")
    st.subheader("Resultado por Mes")
    month_df = monthly_summary(ops_df)
    st.dataframe(month_df, use_container_width=True)

    if not month_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(month_df["month"], month_df["pnl_total"])
        ax.set_title("PnL por Mes")
        ax.set_xlabel("Mes")
        ax.set_ylabel("PnL")
        ax.tick_params(axis="x", rotation=25)
        st.pyplot(fig)

    st.subheader("Resultado por Día")
    daily = daily_summary(ops_df)
    st.dataframe(daily.sort_values("trade_day", ascending=False), use_container_width=True)

    if not daily.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily["trade_day"].astype(str), daily["pnl_total"], marker="o")
        ax.set_title("PnL Diario")
        ax.set_ylabel("PnL")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    st.subheader("Mejores y Peores Días")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Peores días**")
        st.dataframe(daily.sort_values("pnl_total", ascending=True).head(10), use_container_width=True)
    with c2:
        st.markdown("**Mejores días**")
        st.dataframe(daily.sort_values("pnl_total", ascending=False).head(10), use_container_width=True)

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

def render_tiempo_y_sesiones(ops_df: pd.DataFrame):
    st.header("Tiempo y Sesiones")
    section_note("Esta página responde: ¿cuándo debería operar el bot y qué horarios/sesiones conviene bloquear?")
    show_help(
        "Tiempo y Sesiones",
        "Análisis por día de semana, hora y sesión. Aquí buscamos edge temporal.",
        [
            "¿Qué horas generan PnL positivo?",
            "¿Qué sesiones son peligrosas?",
            "¿Hay días de semana que dañan el resultado?",
            "¿El mismo horario falla en varios meses?",
        ],
    )

    if ops_df.empty:
        st.warning("No hay operaciones con los filtros actuales.")
        return

    st.subheader("Día de semana")
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = aggregate_core(ops_df, ["dia_semana"])
    weekday["_order"] = weekday["dia_semana"].apply(lambda x: weekday_order.index(x) if x in weekday_order else 99)
    weekday = weekday.sort_values("_order").drop(columns=["_order"])
    st.dataframe(weekday, use_container_width=True)

    st.subheader("Hora")
    by_hour = aggregate_core(ops_df, ["hora_inicio"]).sort_values("hora_inicio")
    st.dataframe(by_hour, use_container_width=True)
    if len(by_hour) > 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(by_hour["hora_inicio"].astype(str), by_hour["pnl_total"])
        ax.set_title("PnL por Hora")
        ax.set_xlabel("Hora")
        ax.set_ylabel("PnL")
        st.pyplot(fig)

    st.subheader("Sesión")
    by_session = aggregate_core(ops_df, ["sesion"]).sort_values("pnl_total", ascending=False)
    st.dataframe(by_session, use_container_width=True)

    if ops_df["month"].nunique() > 1:
        st.subheader("Sesión por mes")
        session_month = aggregate_core(ops_df, ["month", "sesion"]).sort_values(["month", "pnl_total"], ascending=[True, False])
        st.dataframe(session_month, use_container_width=True)

    lines = []
    if not by_hour.empty:
        best_h = by_hour.sort_values("pnl_total", ascending=False).iloc[0]
        worst_h = by_hour.sort_values("pnl_total", ascending=True).iloc[0]
        lines.append(f"Mejor hora por PnL total: {int(best_h['hora_inicio'])}:00.")
        lines.append(f"Peor hora por PnL total: {int(worst_h['hora_inicio'])}:00.")
    if not by_session.empty:
        best_s = by_session.iloc[0]
        worst_s = by_session.sort_values("pnl_total", ascending=True).iloc[0]
        lines.append(f"Mejor sesión: {best_s['sesion']}.")
        lines.append(f"Sesión más débil: {worst_s['sesion']}.")
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
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(rev["reversal_count"].fillna(0).astype(int).astype(str), rev["pnl_total"])
        ax.set_title("PnL Total por Cantidad de Reversals")
        ax.set_xlabel("Reversals usados")
        ax.set_ylabel("PnL")
        st.pyplot(fig)

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

    sim_df = aplicar_cap_reversal(ops_df, legs_df, max_reversal_permitido)

    real_pnl = ops_df["sequence_net_pnl_currency"].sum()
    sim_pnl = sim_df["sequence_net_pnl_simulado"].sum()
    ops_afectadas = int(sim_df["cap_aplicado"].sum()) if "cap_aplicado" in sim_df.columns else 0
    peor_real = ops_df["sequence_net_pnl_currency"].min()
    peor_sim = sim_df["sequence_net_pnl_simulado"].min()

    c = st.columns(4)
    with c[0]: card("PnL Real", fmt_money(real_pnl))
    with c[1]: card("PnL Simulado", fmt_money(sim_pnl))
    with c[2]: card("Operaciones Cortadas", str(ops_afectadas))
    with c[3]: card("Peor Op Simulada", fmt_money(peor_sim))

    cap_summary = pd.DataFrame([
        {"métrica": "PnL total", "real": real_pnl, "simulado": sim_pnl, "diferencia": sim_pnl - real_pnl},
        {"métrica": "Peor operación", "real": peor_real, "simulado": peor_sim, "diferencia": peor_sim - peor_real},
        {"métrica": "Operaciones afectadas", "real": 0, "simulado": ops_afectadas, "diferencia": ops_afectadas},
    ])
    st.dataframe(cap_summary, use_container_width=True)

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
    lines.append(f"Con cap = {max_reversal_permitido}, el PnL cambia de {fmt_money(real_pnl)} a {fmt_money(sim_pnl)}.")
    if peor_sim > peor_real:
        lines.append("El cap mejora el peor caso; esto puede ser bueno para control de riesgo.")
    elif peor_sim < peor_real:
        lines.append("El cap empeora el peor caso en esta simulación; revisar antes de usarlo.")
    show_conclusion("Motor de Reversiones", lines)


def render_simulador_diario(ops_df: pd.DataFrame, legs_df: pd.DataFrame):
    st.header("Simulador Diario")
    section_note("Esta página responde: ¿qué pasa si aplico una meta diaria y una pérdida diaria usando las piernas reales del bot, no solo el resultado final de la operación?")
    show_help(
        "Simulador Diario",
        "Separado en dos bloques: stop diario clásico y sets diarios. Ambos consumen las piernas en orden cronológico, porque cada pierna es una entrada/salida real.",
        [
            "¿Cuántos días llegan a la meta antes que a la pérdida?",
            "¿Cuántos días terminan abiertos sin tocar meta/loss?",
            "¿La lógica de sets funciona mejor que parar el día completo?",
            "¿La simulación sobrevive mes por mes?",
        ],
    )

    if ops_df.empty or legs_df.empty:
        st.warning("No hay operaciones o piernas con los filtros actuales.")
        return

    st.subheader("1. Stop diario clásico")
    section_note("El día se detiene completo cuando el PnL acumulado de las piernas cerradas toca la meta o la pérdida. Si no toca ninguna, queda con el resultado acumulado hasta el final del día.")
    c1, c2 = st.columns(2)
    daily_target = c1.number_input("Meta diaria", min_value=1.0, value=600.0, step=50.0, key="daily_target")
    daily_loss = c2.number_input("Pérdida máxima diaria", min_value=1.0, value=300.0, step=50.0, key="daily_loss")

    daily_df, daily_m = simulate_daily_stop(legs_df, daily_target, daily_loss)
    c = st.columns(4)
    with c[0]: card("PnL Simulado", fmt_money(daily_m.get("total_pnl", np.nan)))
    with c[1]: card("Días Meta", fmt_pct(daily_m.get("target_days_pct", np.nan)))
    with c[2]: card("Días Pérdida", fmt_pct(daily_m.get("loss_days_pct", np.nan)))
    with c[3]: card("Días Abiertos", fmt_pct(daily_m.get("open_days_pct", np.nan)))

    st.dataframe(daily_df.sort_values("trade_day"), use_container_width=True)

    monthly_daily = daily_df.groupby("month", as_index=False).agg(
        dias=("trade_day", "count"),
        pnl_simulado=("simulated_day_pnl", "sum"),
        promedio_dia=("simulated_day_pnl", "mean"),
        meta_pct=("stop_reason", lambda s: (s == "Target reached").mean() * 100),
        perdida_pct=("stop_reason", lambda s: (s == "Loss reached").mean() * 100),
        abierto_pct=("stop_reason", lambda s: (s == "End of day").mean() * 100),
    )
    st.markdown("**Resumen mensual · Stop diario**")
    st.dataframe(monthly_daily, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Sets diarios")
    section_note("El día se divide en sets. Cada set consume piernas reales en orden hasta llegar al target o loss. Después, la siguiente pierna empieza un nuevo set.")
    c1, c2 = st.columns(2)
    set_target = c1.number_input("Target por set", min_value=1.0, value=600.0, step=50.0, key="set_target")
    set_loss = c2.number_input("Loss por set", min_value=1.0, value=300.0, step=50.0, key="set_loss")

    sets_df, sets_m = simulate_daily_sets(legs_df, set_target, set_loss)
    c = st.columns(4)
    with c[0]: card("PnL Sets", fmt_money(sets_m.get("total_pnl", np.nan)))
    with c[1]: card("Sets Meta", fmt_pct(sets_m.get("target_sets_pct", np.nan)))
    with c[2]: card("Sets Pérdida", fmt_pct(sets_m.get("loss_sets_pct", np.nan)))
    with c[3]: card("Piernas/Set Prom", "-" if pd.isna(sets_m.get("avg_legs_per_set", np.nan)) else f"{sets_m['avg_legs_per_set']:.2f}")

    st.dataframe(sets_df.sort_values(["trade_day", "set_number"]), use_container_width=True)

    monthly_sets = sets_df.groupby("month", as_index=False).agg(
        sets=("set_number", "count"),
        pnl_sets=("set_result", "sum"),
        promedio_set=("set_result", "mean"),
        meta_pct=("set_outcome", lambda s: (s == "Set target reached").mean() * 100),
        perdida_pct=("set_outcome", lambda s: (s == "Set loss reached").mean() * 100),
        abierto_pct=("set_outcome", lambda s: (s == "End of day").mean() * 100),
        piernas_por_set=("legs_used", "mean"),
        operaciones_por_set=("operations_touched", "mean"),
    )
    st.markdown("**Resumen mensual · Sets diarios**")
    st.dataframe(monthly_sets, use_container_width=True)

    lines = []
    if daily_m:
        lines.append("Stop diario: la meta se alcanza más que la pérdida." if daily_m["target_days_pct"] > daily_m["loss_days_pct"] else "Stop diario: la pérdida aparece demasiado cerca de la meta.")
    if sets_m:
        lines.append("Sets: hay más sets que alcanzan target que sets que alcanzan loss." if sets_m["target_sets_pct"] > sets_m["loss_sets_pct"] else "Sets: el balance target/loss todavía no se ve claro.")
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
# MAIN
# ============================================================


def main():
    st.title("Laboratorio WLF")
    st.caption("Análisis simple para decidir qué mantener, qué bloquear y qué setting probar.")

    uploaded_files = st.sidebar.file_uploader(
        "Cargar archivos JSONL mensuales",
        type=["jsonl"],
        accept_multiple_files=True,
    )

    records = load_uploaded_jsonl_files(uploaded_files)
    ops_df, legs_df = build_dataframes(records)

    if ops_df.empty:
        st.warning("Sube uno o más JSONL para iniciar el análisis.")
        return

    st.sidebar.metric("Operaciones cargadas", len(ops_df))
    st.sidebar.metric("Piernas cargadas", len(legs_df))

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
