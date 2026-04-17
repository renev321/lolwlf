import json
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="lol_wlf Lab", layout="wide")


def load_uploaded_jsonl_files(uploaded_files) -> list[dict]:
    records: list[dict] = []

    if not uploaded_files:
        st.sidebar.info("No files uploaded yet.")
        return records

    for uploaded_file in uploaded_files:
        try:
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            lines = content.splitlines()

            st.sidebar.write(f"File: {uploaded_file.name}")
            st.sidebar.write(f"Raw lines: {len(lines)}")

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
                        st.sidebar.write(f"Invalid JSON line {i}: {line[:120]}")
                        st.sidebar.write(f"Error: {e}")

            st.sidebar.write(f"Valid records: {valid_count}")
            st.sidebar.write(f"Invalid lines: {invalid_count}")

        except Exception as e:
            st.sidebar.error(f"Failed reading {uploaded_file.name}: {e}")

    return records


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
    ops_df["start_hour"] = ops_df["sequence_started_at"].dt.hour
    ops_df["start_minute"] = ops_df["sequence_started_at"].dt.minute
    ops_df["is_winner"] = ops_df["sequence_net_pnl_currency"] > 0

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
        outcome = "neither"
        stop_after_operation = None

        for _, row in day_df.iterrows():
            running += float(row["sequence_net_pnl_currency"])
            stop_after_operation = row["operation_id"]

            if running >= daily_target:
                outcome = "target_first"
                running = daily_target
                break

            if running <= -daily_loss:
                outcome = "loss_first"
                running = -daily_loss
                break

        sim_rows.append(
            {
                "trade_day": trade_day,
                "simulated_day_result": running,
                "outcome": outcome,
                "stop_after_operation": stop_after_operation,
            }
        )

    sim_df = pd.DataFrame(sim_rows)
    if sim_df.empty:
        return sim_df, {}

    metrics = {
        "days": len(sim_df),
        "target_first_rate": (sim_df["outcome"] == "target_first").mean() * 100,
        "loss_first_rate": (sim_df["outcome"] == "loss_first").mean() * 100,
        "neither_rate": (sim_df["outcome"] == "neither").mean() * 100,
        "avg_daily_result": sim_df["simulated_day_result"].mean(),
        "median_daily_result": sim_df["simulated_day_result"].median(),
        "total_simulated_pnl": sim_df["simulated_day_result"].sum(),
        "best_day": sim_df["simulated_day_result"].max(),
        "worst_day": sim_df["simulated_day_result"].min(),
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
    st.subheader("Bot Health")
    metrics = compute_overview_metrics(ops_df)
    if not metrics:
        st.info("No operation data found.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Operations", f"{metrics['total_operations']}")
    c2.metric("Total Net PnL", f"{metrics['total_net_pnl']:.2f}")
    c3.metric("Win Rate %", f"{metrics['win_rate']:.1f}")
    c4.metric("Profit Factor", "-" if pd.isna(metrics['profit_factor']) else f"{metrics['profit_factor']:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg PnL", f"{metrics['avg_pnl']:.2f}")
    c6.metric("Median PnL", f"{metrics['median_pnl']:.2f}")
    c7.metric("Best Op", f"{metrics['best_operation']:.2f}")
    c8.metric("Worst Op", f"{metrics['worst_operation']:.2f}")

    daily = (
        ops_df.groupby("trade_day", as_index=False)["sequence_net_pnl_currency"]
        .sum()
        .sort_values("trade_day")
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily["trade_day"].astype(str), daily["sequence_net_pnl_currency"])
    ax.set_title("Daily Net PnL")
    ax.set_ylabel("PnL")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)


def render_reversal_engine(ops_df: pd.DataFrame):
    st.subheader("Reversal Engine")
    if ops_df.empty:
        st.info("No operation data found.")
        return

    grouped = ops_df.groupby("reversal_count").agg(
        operations=("operation_id", "count"),
        total_pnl=("sequence_net_pnl_currency", "sum"),
        avg_pnl=("sequence_net_pnl_currency", "mean"),
        win_rate=("is_winner", "mean"),
        avg_drawdown=("operation_max_drawdown_currency", "mean"),
        avg_loss_before_recovery=("sequence_loss_currency", "mean"),
    ).reset_index()
    grouped["win_rate"] = grouped["win_rate"] * 100

    st.dataframe(grouped, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(grouped["reversal_count"].astype(str), grouped["total_pnl"])
    ax.set_title("Total PnL by Reversal Count")
    ax.set_xlabel("Reversal Count")
    ax.set_ylabel("Total PnL")
    st.pyplot(fig)


def render_daily_goal_simulator(ops_df: pd.DataFrame):
    st.subheader("Daily Goal Simulator")
    if ops_df.empty:
        st.info("No operation data found.")
        return

    c1, c2 = st.columns(2)
    daily_target = c1.number_input("Daily Target", min_value=1.0, value=600.0, step=50.0)
    daily_loss = c2.number_input("Daily Max Loss", min_value=1.0, value=300.0, step=50.0)

    sim_df, metrics = simulate_daily_plan(ops_df, daily_target=daily_target, daily_loss=daily_loss)
    if not metrics:
        st.info("Not enough data to simulate.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target First %", f"{metrics['target_first_rate']:.1f}")
    c2.metric("Loss First %", f"{metrics['loss_first_rate']:.1f}")
    c3.metric("Neither %", f"{metrics['neither_rate']:.1f}")
    c4.metric("Avg Daily Result", f"{metrics['avg_daily_result']:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Simulated PnL", f"{metrics['total_simulated_pnl']:.2f}")
    c6.metric("Median Daily Result", f"{metrics['median_daily_result']:.2f}")
    c7.metric("Best Day", f"{metrics['best_day']:.2f}")
    c8.metric("Worst Day", f"{metrics['worst_day']:.2f}")

    if not sim_df.empty:
        ordered = sim_df.sort_values("trade_day")
        win_streak = max_streak(ordered["simulated_day_result"], positive=True)
        loss_streak = max_streak(ordered["simulated_day_result"], positive=False)
        st.write(f"Max winning streak of days: **{win_streak}**")
        st.write(f"Max losing streak of days: **{loss_streak}**")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ordered["trade_day"].astype(str), ordered["simulated_day_result"])
        ax.set_title("Simulated Daily Results")
        ax.set_ylabel("PnL")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

        st.dataframe(ordered, use_container_width=True)


def render_operation_explorer(ops_df: pd.DataFrame, legs_df: pd.DataFrame):
    st.subheader("Operation Explorer")
    if ops_df.empty:
        st.info("No operation data found.")
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
    ]
    st.dataframe(ops_df[display_cols].sort_values("sequence_started_at", ascending=False), use_container_width=True)

    op_ids = ops_df["operation_id"].dropna().tolist()
    selected_op = st.selectbox("Inspect operation", op_ids)
    if selected_op:
        op_row = ops_df.loc[ops_df["operation_id"] == selected_op]
        st.write(op_row)
        st.dataframe(
            legs_df.loc[legs_df["operation_id"] == selected_op].sort_values("leg_index"),
            use_container_width=True,
        )


def main():
    st.title("lol_wlf Python Lab")
    st.caption("Decision-oriented analytics for your bot")

    uploaded_files = st.sidebar.file_uploader(
        "Load monthly JSONL files",
        type=["jsonl"],
        accept_multiple_files=True,
    )

    records = load_uploaded_jsonl_files(uploaded_files)
    ops_df, legs_df = build_dataframes(records)

    if ops_df.empty:
        st.warning("No JSONL records loaded. Upload one or more monthly JSONL files, for example 2026-01.jsonl through 2026-06.jsonl.")
        return

    st.sidebar.metric("Loaded files", f"{len(uploaded_files)}")
    st.sidebar.metric("Loaded operations", f"{len(ops_df)}")
    st.sidebar.metric("Loaded legs", f"{len(legs_df)}")

    page = st.sidebar.radio(
        "Page",
        [
            "Overview",
            "Reversal Engine",
            "Daily Goal Simulator",
            "Operation Explorer",
        ],
    )

    if page == "Overview":
        render_overview(ops_df)
    elif page == "Reversal Engine":
        render_reversal_engine(ops_df)
    elif page == "Daily Goal Simulator":
        render_daily_goal_simulator(ops_df)
    elif page == "Operation Explorer":
        render_operation_explorer(ops_df, legs_df)


if __name__ == "__main__":
    main()
