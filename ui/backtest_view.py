"""Backtest results: equity curve + trades table + metrics."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_backtest_results(backtest_result: dict) -> None:
    """Render backtest equity curve, metrics, and trade table."""
    if not backtest_result or not backtest_result.get("trades"):
        st.info("No trades generated for the selected period.")
        return

    metrics = backtest_result["metrics"]
    trades = backtest_result["trades"]
    equity_curve = backtest_result["equity_curve"]

    # --- Summary metrics ---
    st.subheader("Backtest Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Trades", metrics["n_trades"])
    col2.metric("Win Rate", f"{metrics['win_rate']:.1%}")
    col3.metric("Avg P&L", f"${metrics['avg_pnl']:,.0f}")
    col4.metric("Sharpe", f"{metrics['sharpe']:.2f}")
    col5.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")

    col6, col7 = st.columns(2)
    col6.metric("Total P&L", f"${metrics['total_pnl']:,.0f}")
    col7.metric("Sortino", f"{metrics['sortino']:.2f}")

    # --- Equity Curve ---
    if not equity_curve.empty:
        st.subheader("Equity Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_curve["date"],
            y=equity_curve["equity"],
            mode="lines",
            name="Portfolio Value",
            line=dict(color="#26a69a", width=2),
            fill="tozeroy",
            fillcolor="rgba(38,166,154,0.1)",
        ))
        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=40, r=40, t=30, b=30),
            yaxis_title="Portfolio Value ($)",
            xaxis_title="Date",
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Exit reason breakdown ---
    if trades:
        reasons = {}
        for t in trades:
            r = t.get("exit_reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
        reason_cols = st.columns(len(reasons))
        for col, (reason, count) in zip(reason_cols, reasons.items()):
            label = {"stop_loss": "Stop Loss", "profit_target": "Profit Target",
                     "signal_reversal": "Signal Reversal", "max_hold": "Max Hold"}.get(reason, reason)
            col.metric(label, count)

    # --- Trades table ---
    st.subheader("Trade Log")
    df_trades = pd.DataFrame(trades)
    if not df_trades.empty:
        display_cols = ["entry_date", "exit_date", "direction", "option_type",
                        "entry_spot", "exit_spot", "days_held", "exit_reason",
                        "entry_price", "exit_price", "pnl", "pnl_pct", "signal_score"]
        available_cols = [c for c in display_cols if c in df_trades.columns]
        df_display = df_trades[available_cols].copy()

        for col in ["entry_spot", "exit_spot", "entry_price", "exit_price"]:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"${x:.2f}")
        if "pnl" in df_display.columns:
            df_display["pnl"] = df_display["pnl"].apply(lambda x: f"${x:+,.0f}")
        if "pnl_pct" in df_display.columns:
            df_display["pnl_pct"] = df_display["pnl_pct"].apply(lambda x: f"{x:+.1%}")
        if "signal_score" in df_display.columns:
            df_display["signal_score"] = df_display["signal_score"].apply(lambda x: f"{x:.2f}")

        st.dataframe(df_display, use_container_width=True, height=300)
