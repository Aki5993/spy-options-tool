"""SPY Options Signal Tool — Streamlit dashboard."""
import os
import sys
import time
import datetime

import streamlit as st
import pandas as pd
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPY Options Signal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ───────────────────────────────────────────────────────────────────
from ui.sidebar import render_sidebar
from ui.chart import build_main_chart, build_hourly_trendline_chart
from ui.backtest_view import render_backtest_results
from data.market_data import (fetch_spy_history, fetch_vix_history,
                               fetch_realtime_spy, fetch_current_vix,
                               fetch_spy_hourly)
from data.sentiment import (fetch_fear_greed, fetch_fear_greed_history,
                             fetch_stocktwits_sentiment, fetch_news_sentiment)
from data.events import days_to_next_fomc, days_to_next_opex, is_opex_week
from data.options_flow import fetch_pcr_current
from features.technical import add_all_indicators, get_hourly_trendline_signal
from features.sentiment_features import merge_sentiment_into_df
from features.event_features import build_event_feature_df
from models.signal_generator import compute_signal, generate_signals_series
from models.options_strategy import recommend_strategy
from models.trainer import load_model, train_model, get_feature_importance
from backtest.engine import run_backtest


# ── Cached data loaders ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Loading SPY history…")
def load_spy(_years: int = 20):
    return fetch_spy_history(_years)


@st.cache_data(ttl=3600, show_spinner="Loading VIX history…")
def load_vix(_years: int = 20):
    return fetch_vix_history(_years)


@st.cache_data(ttl=3600 * 6, show_spinner="Loading Fear & Greed…")
def load_fg_history():
    return fetch_fear_greed_history(days=365 * 5)


@st.cache_data(ttl=300, show_spinner="Loading hourly data…")
def load_hourly():
    return fetch_spy_hourly(days=59)


@st.cache_resource(show_spinner="Loading ML model…")
def load_ml_model():
    return load_model()


# ── Build full feature dataframe ──────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Computing features…")
def build_feature_df(spy_df_json: str, vix_df_json: str, fg_df_json: str) -> pd.DataFrame:
    spy = pd.read_json(spy_df_json)
    spy.index = pd.to_datetime(spy.index)
    vix = pd.read_json(vix_df_json)
    vix.index = pd.to_datetime(vix.index)
    fg = pd.read_json(fg_df_json) if fg_df_json else pd.DataFrame()
    if not fg.empty:
        fg.index = pd.to_datetime(fg.index)

    df = add_all_indicators(spy)
    df = merge_sentiment_into_df(df, fg)
    event_df = build_event_feature_df(df, vix)
    df = df.join(event_df, how="left", rsuffix="_ev")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cfg = render_sidebar()

    if cfg["polygon_key"]:
        os.environ["POLYGON_API_KEY"] = cfg["polygon_key"]
    if cfg["alpha_key"]:
        os.environ["ALPHA_VANTAGE_KEY"] = cfg["alpha_key"]
    if cfg["fred_key"]:
        os.environ["FRED_API_KEY"] = cfg["fred_key"]

    # ── Top info bar ──────────────────────────────────────────────────────────
    spy_quote = fetch_realtime_spy()
    vix_now = fetch_current_vix()
    fg_now = fetch_fear_greed()
    pcr_now = fetch_pcr_current()
    dtf = days_to_next_fomc()
    dtop = days_to_next_opex()
    opex_wk = is_opex_week()

    top_cols = st.columns(7)
    price = spy_quote.get("price")
    change = spy_quote.get("change_pct")
    top_cols[0].metric("SPY", f"${price:.2f}" if price else "N/A",
                        f"{change:+.2f}%" if change else None)
    top_cols[1].metric("VIX", f"{vix_now:.2f}" if vix_now else "N/A")
    fg_val = fg_now.get("value", 50)
    fg_label = fg_now.get("classification", "Neutral")
    top_cols[2].metric("Fear & Greed", f"{fg_val} — {fg_label}")
    pcr_val = pcr_now.get("pcr", 1.0)
    pcr_label = pcr_now.get("label", "Neutral")
    top_cols[3].metric("Put/Call Ratio", f"{pcr_val:.2f} — {pcr_label}")
    top_cols[4].metric("Days to FOMC", dtf)
    top_cols[5].metric("Days to OPEX", dtop, "OPEX week!" if opex_wk else None)
    st.markdown("---")

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Fetching market data…"):
        spy_df = load_spy(20)
        vix_df = load_vix(20)
        fg_history = load_fg_history()
        hourly_df = load_hourly()

    # ── Hourly trendline signal ───────────────────────────────────────────────
    tl_signal = get_hourly_trendline_signal(hourly_df)

    # ── Build full feature DataFrame ──────────────────────────────────────────
    spy_json = spy_df.to_json(date_format="iso")
    vix_json = vix_df.to_json(date_format="iso")
    fg_json = fg_history.to_json(date_format="iso") if not fg_history.empty else ""
    full_df = build_feature_df(spy_json, vix_json, fg_json)

    # Inject real-time features into the last row for current signal
    full_df["pcr"] = 1.0               # default for historical rows
    full_df["tl_break_up"] = 0
    full_df["tl_break_down"] = 0
    if not full_df.empty:
        full_df.iloc[-1, full_df.columns.get_loc("pcr")] = pcr_val
        full_df.iloc[-1, full_df.columns.get_loc("tl_break_up")] = int(tl_signal["tl_break_up"])
        full_df.iloc[-1, full_df.columns.get_loc("tl_break_down")] = int(tl_signal["tl_break_down"])

    # Slice to selected date range
    start = pd.Timestamp(cfg["start_date"])
    end = pd.Timestamp(cfg["end_date"])
    view_df = full_df.loc[start:end].copy()

    # ── Generate signals ──────────────────────────────────────────────────────
    ml_model = load_ml_model() if cfg["use_ml"] else None
    view_df = generate_signals_series(
        view_df, ml_model, min_confidence=cfg["min_confidence"]
    )

    # ── Current signal card ───────────────────────────────────────────────────
    st.subheader("Current Signal")
    if not view_df.empty:
        latest = view_df.iloc[-1]
        current_signal = compute_signal(latest, ml_model)

        st_info = fetch_stocktwits_sentiment()
        news_info = fetch_news_sentiment()
        current_iv = vix_now / 100 * 1.1 * 100 if vix_now else 20.0

        strategy = recommend_strategy(current_signal, current_iv=current_iv, spot_price=price)

        sig_cols = st.columns(5)
        dir_color = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(
            current_signal["direction"], "⚪")
        sig_cols[0].metric("Direction", f"{dir_color} {current_signal['direction'].title()}")
        sig_cols[1].metric("Signal Score", f"{current_signal['score']:.2f}")
        sig_cols[2].metric("Confidence", f"{current_signal['confidence']}%")
        trend_icon = "✅" if current_signal.get("trend_aligned") else "⚠️"
        sig_cols[3].metric("Trend Aligned", f"{trend_icon} {'Yes' if current_signal.get('trend_aligned') else 'No'}")
        if current_signal["ml_prob"] is not None:
            sig_cols[4].metric("ML Prob (bull)", f"{current_signal['ml_prob']:.2f}")
        else:
            sig_cols[4].metric("Rule Score", f"{current_signal['rule_score']:.2f}")

        # Trendline signal banner
        if tl_signal["tl_break_up"]:
            st.success(
                f"Hourly trendline BREAK UP detected in last 24 bars  |  "
                f"Resistance: ${tl_signal['resistance']:.2f}" if tl_signal['resistance'] else
                "Hourly trendline BREAK UP detected"
            )
        elif tl_signal["tl_break_down"]:
            st.error(
                f"Hourly trendline BREAK DOWN detected in last 24 bars  |  "
                f"Support: ${tl_signal['support']:.2f}" if tl_signal['support'] else
                "Hourly trendline BREAK DOWN detected"
            )

        st.markdown("---")
        rec_cols = st.columns(3)
        rec_cols[0].info(f"**Strategy:** {strategy['strategy']}")
        rec_cols[1].info(f"**Expiry:** ~{strategy['expiry_days']} days"
                          if strategy['expiry_days'] else "**Expiry:** N/A")
        rec_cols[2].info(f"**Strike:** {strategy['strike_offset']}"
                          if strategy['strike_offset'] else "**Strike:** N/A")

        with st.expander("Signal rationale"):
            for r in strategy["rationale"]:
                st.write(f"• {r}")
            if current_signal["reduce_conf"]:
                st.warning("FOMC proximity — confidence reduced, spreads preferred.")
            if not current_signal.get("trend_aligned"):
                st.warning("Signal is COUNTER-TREND (against SMA50 direction) — lower conviction.")

        # Sentiment row
        sent_cols = st.columns(4)
        sent_cols[0].metric("StockTwits Bull %", f"{st_info.get('bull_ratio', 0.5):.1%}")
        sent_cols[1].metric("News Sentiment", news_info.get("label", "Neutral"))
        sent_cols[2].metric("News Score", f"{news_info.get('score', 0.0):+.3f}")
        sent_cols[3].metric("PCR", f"{pcr_val:.2f}", pcr_label)

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.subheader("SPY Chart")
    chart_tabs = st.tabs(["Price Chart", "Hourly + Trendlines", "MACD", "Backtest"])

    with chart_tabs[0]:
        fig = build_main_chart(view_df, show_bb=cfg["show_bb"], show_signals=cfg["show_signals"])
        st.plotly_chart(fig, use_container_width=True)

    with chart_tabs[1]:
        if not hourly_df.empty and "resistance" in tl_signal["hourly_df"].columns:
            fig_h = build_hourly_trendline_chart(tl_signal["hourly_df"])
            st.plotly_chart(fig_h, use_container_width=True)
            tl_cols = st.columns(2)
            if tl_signal["resistance"]:
                tl_cols[0].metric("Resistance", f"${tl_signal['resistance']:.2f}")
            if tl_signal["support"]:
                tl_cols[1].metric("Support", f"${tl_signal['support']:.2f}")
        else:
            st.info("Hourly data not available yet.")

    with chart_tabs[2]:
        if "MACD" in view_df.columns:
            import plotly.graph_objects as go
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=view_df.index, y=view_df["MACD"],
                                           name="MACD", line=dict(color="#26a69a")))
            fig_macd.add_trace(go.Scatter(x=view_df.index, y=view_df["MACD_signal"],
                                           name="Signal", line=dict(color="#ef5350")))
            fig_macd.add_trace(go.Bar(x=view_df.index, y=view_df["MACD_hist"],
                                       name="Histogram",
                                       marker_color=["#26a69a" if v >= 0 else "#ef5350"
                                                     for v in view_df["MACD_hist"].fillna(0)]))
            fig_macd.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_macd, use_container_width=True)

    with chart_tabs[3]:
        run_bt = st.button("Run Backtest", type="primary")
        if run_bt:
            with st.spinner("Running backtest…"):
                bt_result = run_backtest(
                    view_df,
                    initial_capital=cfg["bt_capital"],
                    max_hold_days=cfg["bt_max_hold"],
                    option_expiry_days=cfg["bt_expiry"],
                    min_hold_days=cfg["bt_min_hold"],
                    stop_loss_pct=cfg["bt_stop_loss"],
                    profit_target_pct=cfg["bt_profit_target"],
                    min_confidence=cfg["min_confidence"],
                    contract_size=cfg["bt_contracts"],
                )
            st.session_state["bt_result"] = bt_result

        if "bt_result" in st.session_state:
            render_backtest_results(st.session_state["bt_result"])

    # ── Model training ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Model Training")
    train_cols = st.columns(2)
    with train_cols[0]:
        if st.button("Train / Retrain XGBoost Model"):
            with st.spinner("Training model on historical data…"):
                try:
                    train_model(full_df)
                    st.success("Model trained and saved!")
                    st.cache_resource.clear()
                except Exception as e:
                    st.error(f"Training failed: {e}")

    with train_cols[1]:
        current_model = load_ml_model()
        if current_model is not None:
            imp_df = get_feature_importance(current_model)
            if not imp_df.empty:
                st.write("**Top Feature Importances**")
                st.dataframe(imp_df.head(10), use_container_width=True)
        else:
            st.info("No trained model found. Click 'Train' to build one.")

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if cfg["auto_refresh"]:
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
