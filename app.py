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


def simulate_daily_stop(ops_df: pd.DataFrame, daily_target: float, daily_loss: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    if ops_df.empty:
        return pd.DataFrame(), {}

    for trade_day, day_df in ops_df.sort_values("sequence_started_at").groupby("trade_day"):
        running = 0.0
        reason = "End of day"
        stopped_after = None
        ops_used = 0

        for _, row in day_df.iterrows():
            running += float(row["sequence_net_pnl_currency"] or 0)
            stopped_after = row["operation_id"]
            ops_used += 1

            if running >= daily_target:
                running = daily_target
                reason = "Target reached"
                break
            if running <= -daily_loss:
                running = -daily_loss
                reason = "Loss reached"
                break

        rows.append(
            {
                "month": day_df["month"].iloc[0],
                "trade_day": trade_day,
                "simulated_day_pnl": running,
                "result": "Win" if running > 0 else "Loss" if running < 0 else "Flat",
                "stop_reason": reason,
                "operations_used": ops_used,
                "stopped_after_operation": stopped_after,
                "real_day_pnl": day_df["sequence_net_pnl_currency"].sum(),
                "real_operations": len(day_df),
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
    }
    return sim, metrics


def simulate_daily_sets(ops_df: pd.DataFrame, set_target: float, set_loss: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    if ops_df.empty:
        return pd.DataFrame(), {}

    for trade_day, day_df in ops_df.sort_values("sequence_started_at").groupby("trade_day"):
        set_number = 1
        running = 0.0
        ops_used = 0
        set_start_op = None
        set_start_time = None

        for _, row in day_df.iterrows():
            if ops_used == 0:
                running = 0.0
                set_start_op = row["operation_id"]
                set_start_time = row["sequence_started_at"]

            running += float(row["sequence_net_pnl_currency"] or 0)
            ops_used += 1

            outcome = None
            result = None
            if running >= set_target:
                outcome = "Set target reached"
                result = set_target
            elif running <= -set_loss:
                outcome = "Set loss reached"
                result = -set_loss

            if outcome:
                rows.append(
                    {
                        "month": day_df["month"].iloc[0],
                        "trade_day": trade_day,
                        "set_number": set_number,
                        "set_result": result,
                        "set_outcome": outcome,
                        "operations_used": ops_used,
                        "start_operation": set_start_op,
                        "end_operation": row["operation_id"],
                        "set_start_time": set_start_time,
                        "set_end_time": row["sequence_started_at"],
                    }
                )
                set_number += 1
                running = 0.0
                ops_used = 0
                set_start_op = None
                set_start_time = None

        if ops_used > 0:
            rows.append(
                {
                    "month": day_df["month"].iloc[0],
                    "trade_day": trade_day,
                    "set_number": set_number,
                    "set_result": running,
                    "set_outcome": "End of day",
                    "operations_used": ops_used,
                    "start_operation": set_start_op,
                    "end_operation": day_df.iloc[-1]["operation_id"],
                    "set_start_time": set_start_time,
                    "set_end_time": day_df.iloc[-1]["sequence_started_at"],
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
        "avg_ops_per_set": sets["operations_used"].mean() if not sets.empty else np.nan,
        "best_set": sets["set_result"].max() if not sets.empty else np.nan,
        "worst_set": sets["set_result"].min() if not sets.empty else np.nan,
    }
    return sets, metrics


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


def render_dashboard_general(ops_df: pd.DataFrame):
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

    m = overview_metrics(ops_df)
    c = st.columns(4)
    c[0].markdown(card("Operaciones", f"{m['ops']}"), unsafe_allow_html=True) if False else None
    with c[0]: card("Operaciones", f"{m['ops']}")
    with c[1]: card("Días", f"{m['days']}")
    with c[2]: card("PnL Total", fmt_money(m["pnl"]))
    with c[3]: card("Profit Factor", "-" if pd.isna(m["profit_factor"]) else f"{m['profit_factor']:.2f}")

    c = st.columns(4)
    with c[0]: card("Win Rate", fmt_pct(m["win_rate"]))
    with c[1]: card("Días Positivos", fmt_pct(m["positive_days_rate"]))
    with c[2]: card("Peor Operación", fmt_money(m["worst_op"]))
    with c[3]: card("Peor Día", fmt_money(m["worst_day"]))

    st.subheader("Resultado mensual")
    monthly = aggregate_core(ops_df, ["month"]).sort_values("month")
    st.dataframe(monthly, use_container_width=True)

    if len(monthly) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(monthly["month"].astype(str), monthly["pnl_total"])
        ax.set_title("PnL por Mes")
        ax.set_xlabel("Mes")
        ax.set_ylabel("PnL")
        st.pyplot(fig)

    st.subheader("Resultado diario")
    daily = ops_df.groupby(["month", "trade_day"], as_index=False).agg(
        pnl_diario=("sequence_net_pnl_currency", "sum"),
        operaciones=("operation_id", "count"),
        peor_operacion=("sequence_net_pnl_currency", "min"),
        mejor_operacion=("sequence_net_pnl_currency", "max"),
        drawdown_max=("operation_max_drawdown_currency", "max"),
        reversiones_promedio=("reversal_count", "mean"),
    ).sort_values("trade_day")
    st.dataframe(daily, use_container_width=True)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily["trade_day"].astype(str), daily["pnl_diario"], marker="o")
    ax.set_title("PnL Diario")
    ax.set_ylabel("PnL")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

    lines = []
    lines.append("El bot está positivo en el rango filtrado." if m["pnl"] > 0 else "El bot está negativo en el rango filtrado.")
    if not pd.isna(m["profit_factor"]):
        if m["profit_factor"] >= 1.40:
            lines.append("El profit factor se ve saludable para seguir validando.")
        elif m["profit_factor"] >= 1.0:
            lines.append("El profit factor está apenas aceptable; conviene revisar riesgos y horarios.")
        else:
            lines.append("El profit factor está débil; no confiaría todavía en esta configuración.")
    if abs(m["worst_op"]) > max(abs(m["avg_pnl"]) * 10, 1):
        lines.append("La peor operación parece grande comparada con el promedio; revisa Risk Killers.")
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


def render_motor_reversiones(ops_df: pd.DataFrame):
    st.header("Motor de Reversiones")
    section_note("Esta página responde: ¿cuántas reversiones ayudan y cuándo empiezan a ser peligrosas?")
    show_help(
        "Motor de Reversiones",
        "Versión simple. Sin calidad por pierna, sin cap simulation y sin exposición por pierna.",
        [
            "¿El PnL empeora cuando suben las reversiones?",
            "¿Qué cantidad de reversiones trae más drawdown?",
            "¿La peor operación aparece con muchas reversiones?",
            "¿Conviene limitar el bot antes de cierta profundidad?",
        ],
    )

    if ops_df.empty:
        st.warning("No hay operaciones con los filtros actuales.")
        return

    rev = aggregate_core(ops_df, ["reversal_count"]).sort_values("reversal_count")
    st.subheader("Resultado por cantidad de reversals")
    st.dataframe(rev, use_container_width=True)

    if len(rev) > 0:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(rev["reversal_count"].fillna(0).astype(int).astype(str), rev["pnl_total"])
        ax.set_title("PnL Total por Cantidad de Reversals")
        ax.set_xlabel("Reversals")
        ax.set_ylabel("PnL")
        st.pyplot(fig)

    st.subheader("Peores operaciones con reversals")
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
        lines.append(f"Mejor PnL promedio: {int(best['reversal_count']) if pd.notna(best['reversal_count']) else 0} reversal(es).")
        lines.append(f"Peor PnL promedio: {int(worst['reversal_count']) if pd.notna(worst['reversal_count']) else 0} reversal(es).")
        lines.append(f"Mayor drawdown promedio: {int(risky['reversal_count']) if pd.notna(risky['reversal_count']) else 0} reversal(es).")
    show_conclusion("Motor de Reversiones", lines)


def render_simulador_diario(ops_df: pd.DataFrame):
    st.header("Simulador Diario")
    section_note("Esta página responde: ¿qué pasa si aplico una meta diaria y una pérdida diaria de forma clara?")
    show_help(
        "Simulador Diario",
        "Separado en dos bloques: stop diario clásico y sets diarios. Cada uno tiene su propia tabla clara.",
        [
            "¿Cuántos días llegan a la meta antes que a la pérdida?",
            "¿Cuántos días terminan abiertos sin tocar meta/loss?",
            "¿La lógica de sets funciona mejor que parar el día completo?",
            "¿La simulación sobrevive mes por mes?",
        ],
    )

    if ops_df.empty:
        st.warning("No hay operaciones con los filtros actuales.")
        return

    st.subheader("1. Stop diario clásico")
    section_note("El día se detiene completo cuando toca la meta o la pérdida. Si no toca ninguna, queda con el resultado acumulado hasta el final del día.")
    c1, c2 = st.columns(2)
    daily_target = c1.number_input("Meta diaria", min_value=1.0, value=600.0, step=50.0, key="daily_target")
    daily_loss = c2.number_input("Pérdida máxima diaria", min_value=1.0, value=300.0, step=50.0, key="daily_loss")

    daily_df, daily_m = simulate_daily_stop(ops_df, daily_target, daily_loss)
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
    section_note("El día se divide en sets. Cada set consume operaciones en orden hasta llegar al target o loss. Después, la siguiente operación empieza un nuevo set.")
    c1, c2 = st.columns(2)
    set_target = c1.number_input("Target por set", min_value=1.0, value=600.0, step=50.0, key="set_target")
    set_loss = c2.number_input("Loss por set", min_value=1.0, value=300.0, step=50.0, key="set_loss")

    sets_df, sets_m = simulate_daily_sets(ops_df, set_target, set_loss)
    c = st.columns(4)
    with c[0]: card("PnL Sets", fmt_money(sets_m.get("total_pnl", np.nan)))
    with c[1]: card("Sets Meta", fmt_pct(sets_m.get("target_sets_pct", np.nan)))
    with c[2]: card("Sets Pérdida", fmt_pct(sets_m.get("loss_sets_pct", np.nan)))
    with c[3]: card("Ops/Set Prom", "-" if pd.isna(sets_m.get("avg_ops_per_set", np.nan)) else f"{sets_m['avg_ops_per_set']:.2f}")

    st.dataframe(sets_df.sort_values(["trade_day", "set_number"]), use_container_width=True)

    monthly_sets = sets_df.groupby("month", as_index=False).agg(
        sets=("set_number", "count"),
        pnl_sets=("set_result", "sum"),
        promedio_set=("set_result", "mean"),
        meta_pct=("set_outcome", lambda s: (s == "Set target reached").mean() * 100),
        perdida_pct=("set_outcome", lambda s: (s == "Set loss reached").mean() * 100),
        abierto_pct=("set_outcome", lambda s: (s == "End of day").mean() * 100),
        ops_por_set=("operations_used", "mean"),
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
        "Detalle técnico por operación. Aquí sí vemos piernas, contratos, PnL acumulado y motivos de salida.",
        [
            "¿Dónde entró y salió cada pierna?",
            "¿Cómo evolucionó el PnL acumulado?",
            "¿Cuántos contratos se usaron?",
            "¿La recuperación fue sana o demasiado peligrosa?",
        ],
    )

    if ops_df.empty:
        st.warning("No hay operaciones con los filtros actuales.")
        return

    filtered = ops_df.copy()

    st.subheader("Filtros de búsqueda")
    c1, c2, c3 = st.columns(3)
    with c1:
        search_id = st.text_input("Buscar operation_id", "")
    with c2:
        only_losers = st.checkbox("Solo operaciones perdedoras", value=False)
    with c3:
        min_reversal = st.number_input("Reversal mínimo", min_value=0, value=0, step=1)

    if search_id.strip():
        filtered = filtered[filtered["operation_id"].astype(str).str.contains(search_id.strip(), case=False, na=False)]
    if only_losers:
        filtered = filtered[filtered["sequence_net_pnl_currency"] < 0]
    filtered = filtered[filtered["reversal_count"].fillna(0) >= min_reversal]

    if filtered.empty:
        st.warning("No hay operaciones con esos filtros.")
        return

    list_cols = [
        "month", "trade_day", "operation_id", "sequence_started_at", "sequence_net_pnl_currency",
        "operation_max_drawdown_currency", "operation_max_runup_currency", "reversal_count",
        "base_contracts", "max_contracts_used", "sesion", "hora_inicio", "sequence_end_reason", "config_key",
    ]
    list_cols = [c for c in list_cols if c in filtered.columns]
    st.dataframe(filtered[list_cols].sort_values("sequence_started_at", ascending=False), use_container_width=True)

    op_ids = filtered.sort_values("sequence_started_at", ascending=False)["operation_id"].astype(str).tolist()
    selected = st.selectbox("Seleccionar operación", op_ids)
    if not selected:
        return

    op_row = filtered[filtered["operation_id"].astype(str) == selected]
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
        f"PnL final: {fmt_money(row['sequence_net_pnl_currency'])}.",
        f"Reversals reales: {int(row['reversal_count']) if pd.notna(row['reversal_count']) else 0}.",
        f"Drawdown máximo: {fmt_money(row['operation_max_drawdown_currency'])}.",
        f"Máximo contratos usados: {fmt_money(row['max_contracts_used'])}.",
        f"Razón de cierre: {row['sequence_end_reason']}.",
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
        render_dashboard_general(ops_filtered)
    elif page == "Tiempo y Sesiones":
        render_tiempo_y_sesiones(ops_filtered)
    elif page == "Motor de Reversiones":
        render_motor_reversiones(ops_filtered)
    elif page == "Simulador Diario":
        render_simulador_diario(ops_filtered)
    elif page == "Laboratorio de Parámetros":
        render_laboratorio_parametros(ops_filtered)
    elif page == "Risk Killers":
        render_risk_killers(ops_filtered)
    elif page == "Explorador de Operaciones":
        render_explorador_operaciones(ops_filtered, legs_filtered)


if __name__ == "__main__":
    main()
