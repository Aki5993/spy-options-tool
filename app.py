"""SPY Options Signal Tool — Streamlit dashboard."""
import os
import sys
import time
import datetime

import streamlit as st
import pandas as pd
import numpy as np

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="SPY Options Signal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.sidebar import render_sidebar
from ui.chart import (build_main_chart, build_hourly_trendline_chart,
                      build_macro_chart, build_breadth_chart, add_flag_markers)
from ui.backtest_view import render_backtest_results
from data.market_data import (fetch_spy_history, fetch_vix_history,
                               fetch_realtime_spy, fetch_current_vix,
                               fetch_spy_hourly)
from data.sentiment import (fetch_fear_greed, fetch_fear_greed_history,
                             fetch_stocktwits_sentiment, fetch_news_sentiment)
from data.events import days_to_next_fomc, days_to_next_opex, is_opex_week
from data.options_flow import fetch_pcr_current
from data.macro_data import (fetch_macro_history, fetch_consumer_sentiment,
                              fetch_ism_pmi, get_economic_calendar,
                              get_current_macro_snapshot)
from features.technical import add_all_indicators, get_hourly_trendline_signal
from features.sentiment_features import merge_sentiment_into_df
from features.event_features import build_event_feature_df
from features.breadth import build_full_breadth_df, get_breadth_snapshot
from models.signal_generator import compute_signal, generate_signals_series
from models.options_strategy import recommend_strategy
from models.trainer import load_model, train_model, get_feature_importance
from backtest.engine import run_backtest


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600,   show_spinner="Loading SPY history…")
def load_spy(_y=20):      return fetch_spy_history(_y)

@st.cache_data(ttl=3600,   show_spinner="Loading VIX history…")
def load_vix(_y=20):      return fetch_vix_history(_y)

@st.cache_data(ttl=3600*6, show_spinner="Loading Fear & Greed…")
def load_fg():            return fetch_fear_greed_history(days=365*5)

@st.cache_data(ttl=300,    show_spinner="Loading hourly data…")
def load_hourly():        return fetch_spy_hourly(days=59)

@st.cache_data(ttl=3600*4, show_spinner="Loading macro data…")
def load_macro(_y=5):     return fetch_macro_history(_y)

@st.cache_data(ttl=3600*24,show_spinner="Loading consumer sentiment…")
def load_cons_sent():     return fetch_consumer_sentiment()

@st.cache_data(ttl=3600*6, show_spinner="Loading breadth data…")
def load_breadth(spy_json: str):
    spy = pd.read_json(spy_json); spy.index = pd.to_datetime(spy.index)
    return build_full_breadth_df(spy, years=5)

@st.cache_resource(show_spinner="Loading ML model…")
def load_ml_model():      return load_model()


# ── Feature assembly ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Computing features…")
def build_feature_df(spy_json, vix_json, fg_json,
                     macro_json, sent_json, breadth_json) -> pd.DataFrame:
    spy = pd.read_json(spy_json);   spy.index   = pd.to_datetime(spy.index)
    vix = pd.read_json(vix_json);   vix.index   = pd.to_datetime(vix.index)
    fg  = pd.read_json(fg_json)  if fg_json  else pd.DataFrame()
    if not fg.empty:
        fg.index = pd.to_datetime(fg.index)

    macro = pd.read_json(macro_json) if macro_json else pd.DataFrame()
    if not macro.empty:
        macro.index = pd.to_datetime(macro.index)

    cons = pd.read_json(sent_json) if sent_json else pd.DataFrame()
    if not cons.empty:
        cons.index = pd.to_datetime(cons.index)

    breadth = pd.read_json(breadth_json) if breadth_json else pd.DataFrame()
    if not breadth.empty:
        breadth.index = pd.to_datetime(breadth.index)

    df = add_all_indicators(spy)
    df = merge_sentiment_into_df(df, fg)
    df = df.join(build_event_feature_df(df, vix), how="left", rsuffix="_ev")

    # Macro features
    if not macro.empty:
        macro_cols = [c for c in macro.columns if c in
                      ["DXY","DXY_roc5","DXY_roc20",
                       "Oil","Oil_roc5","Oil_roc20",
                       "Yield10Y","Yield10Y_roc5","Yield10Y_roc20"]]
        df = df.join(macro[macro_cols].reindex(df.index, method="ffill"), how="left")

    # Consumer sentiment (monthly → forward-fill to daily)
    if not cons.empty and "consumer_sentiment" in cons.columns:
        cs = cons["consumer_sentiment"].reindex(df.index, method="ffill")
        df["consumer_sentiment"] = cs

    # Breadth features
    if not breadth.empty:
        breadth_cols = [c for c in breadth.columns if c in
                        ["breadth_ratio","breadth_ema10","pct_above_sma50",
                         "n_new_highs_52wk","n_new_lows_52wk",
                         "zweig_thrust","hindenburg_omen"]]
        df = df.join(breadth[breadth_cols].reindex(df.index, method="ffill"), how="left")

    # Fill missing macro/breadth cols with neutral defaults
    defaults = {
        "pcr": 1.0, "tl_break_up": 0, "tl_break_down": 0,
        "bull_flag": 0, "bear_flag": 0,
        "DXY_roc5": 0, "Oil_roc5": 0, "Yield10Y_roc5": 0,
        "Yield10Y": 4.0, "consumer_sentiment": 80,
        "breadth_ratio": 0.5, "breadth_ema10": 0.5,
        "pct_above_sma50": 0.5, "n_new_highs_52wk": 0, "n_new_lows_52wk": 0,
        "zweig_thrust": 0, "hindenburg_omen": 0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cfg = render_sidebar()

    for env_key, cfg_key in [("POLYGON_API_KEY","polygon_key"),
                              ("ALPHA_VANTAGE_KEY","alpha_key"),
                              ("FRED_API_KEY","fred_key")]:
        if cfg.get(cfg_key):
            os.environ[env_key] = cfg[cfg_key]

    # ── Real-time top bar ─────────────────────────────────────────────────────
    spy_quote  = fetch_realtime_spy()
    vix_now    = fetch_current_vix()
    fg_now     = fetch_fear_greed()
    pcr_now    = fetch_pcr_current()
    macro_snap = get_current_macro_snapshot()
    dtf        = days_to_next_fomc()
    dtop       = days_to_next_opex()
    opex_wk    = is_opex_week()

    price  = spy_quote.get("price")
    change = spy_quote.get("change_pct")
    fg_val = fg_now.get("value", 50)
    pcr_val= pcr_now.get("pcr", 1.0)

    row1 = st.columns(5)
    row1[0].metric("SPY",  f"${price:.2f}" if price else "N/A",
                   f"{change:+.2f}%" if change else None)
    row1[1].metric("VIX",  f"{vix_now:.2f}" if vix_now else "N/A")
    row1[2].metric("Fear & Greed", f"{fg_val} — {fg_now.get('classification','?')}")
    row1[3].metric("Put/Call Ratio",
                   f"{pcr_val:.2f} — {pcr_now.get('label','?')}")
    row1[4].metric("Days to FOMC", dtf,
                   f"OPEX in {dtop}d {'(this week!)' if opex_wk else ''}")

    row2 = st.columns(5)
    dxy = macro_snap.get("DXY"); dxy_r = macro_snap.get("DXY_roc5")
    oil = macro_snap.get("Oil"); oil_r = macro_snap.get("Oil_roc5")
    y10 = macro_snap.get("Yield10Y"); y10_r = macro_snap.get("Yield10Y_roc5")
    row2[0].metric("DXY",
                   f"{dxy:.2f}"  if dxy else "N/A",
                   f"{dxy_r:+.1%}" if dxy_r else None)
    row2[1].metric("WTI Oil",
                   f"${oil:.2f}" if oil else "N/A",
                   f"{oil_r:+.1%}" if oil_r else None)
    row2[2].metric("10Y Yield",
                   f"{y10:.2f}%" if y10 else "N/A",
                   f"{y10_r:+.3f}pt" if y10_r else None)

    ism = fetch_ism_pmi()
    row2[3].metric("ISM Mfg PMI",
                   f"{ism['mfg_pmi']:.1f}" if ism['mfg_pmi'] else "N/A",
                   ism['mfg_label'])
    row2[4].metric("ISM Svc PMI",
                   f"{ism['services_pmi']:.1f}" if ism['services_pmi'] else "N/A",
                   ism['services_label'])
    st.markdown("---")

    # ── Data loading ─────────────────────────────────────────────────────────
    spy_df    = load_spy(20)
    vix_df    = load_vix(20)
    fg_hist   = load_fg()
    hourly_df = load_hourly()
    macro_df  = load_macro(5)
    cons_df   = load_cons_sent()
    breadth_df= load_breadth(spy_df.to_json(date_format="iso"))

    tl_signal = get_hourly_trendline_signal(hourly_df)
    breadth_snap = get_breadth_snapshot(breadth_df)

    full_df = build_feature_df(
        spy_df.to_json(date_format="iso"),
        vix_df.to_json(date_format="iso"),
        fg_hist.to_json(date_format="iso") if not fg_hist.empty else "",
        macro_df.to_json(date_format="iso") if not macro_df.empty else "",
        cons_df.to_json(date_format="iso")  if not cons_df.empty  else "",
        breadth_df.to_json(date_format="iso") if not breadth_df.empty else "",
    )

    # Inject real-time values into the latest row
    if not full_df.empty:
        last = full_df.index[-1]
        full_df.loc[last, "pcr"]             = pcr_val
        full_df.loc[last, "tl_break_up"]     = int(tl_signal["tl_break_up"])
        full_df.loc[last, "tl_break_down"]   = int(tl_signal["tl_break_down"])
        if breadth_snap:
            full_df.loc[last, "zweig_thrust"]    = int(breadth_snap.get("zweig_active", False))
            full_df.loc[last, "hindenburg_omen"] = int(breadth_snap.get("hindenburg_active", False))

    start   = pd.Timestamp(cfg["start_date"])
    end     = pd.Timestamp(cfg["end_date"])
    view_df = full_df.loc[start:end].copy()

    ml_model = load_ml_model() if cfg["use_ml"] else None
    view_df  = generate_signals_series(view_df, ml_model,
                                        min_confidence=cfg["min_confidence"])

    # ── Current signal card ───────────────────────────────────────────────────
    st.subheader("Current Signal")
    if not view_df.empty:
        latest         = view_df.iloc[-1]
        current_signal = compute_signal(latest, ml_model)
        current_iv     = vix_now / 100 * 1.1 * 100 if vix_now else 20.0
        strategy       = recommend_strategy(current_signal,
                                            current_iv=current_iv, spot_price=price)
        st_info  = fetch_stocktwits_sentiment()
        news_info= fetch_news_sentiment()

        sig_cols = st.columns(5)
        dir_icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(
            current_signal["direction"], "⚪")
        sig_cols[0].metric("Direction",
                           f"{dir_icon} {current_signal['direction'].title()}")
        sig_cols[1].metric("Signal Score", f"{current_signal['score']:.2f}")
        sig_cols[2].metric("Confidence",   f"{current_signal['confidence']}%")
        trend_icon = "✅" if current_signal.get("trend_aligned") else "⚠️"
        sig_cols[3].metric("Trend Aligned",
                           f"{trend_icon} {'Yes' if current_signal.get('trend_aligned') else 'No'}")
        sig_cols[4].metric("ML Prob (bull)" if current_signal["ml_prob"] else "Rule Score",
                           f"{current_signal['ml_prob']:.2f}"
                           if current_signal["ml_prob"] else
                           f"{current_signal['rule_score']:.2f}")

        # Banners
        if tl_signal["tl_break_up"]:
            st.success(f"Hourly trendline BREAK UP — Resistance: "
                       f"${tl_signal['resistance']:.2f}" if tl_signal['resistance']
                       else "Hourly trendline BREAK UP detected")
        elif tl_signal["tl_break_down"]:
            st.error(f"Hourly trendline BREAK DOWN — Support: "
                     f"${tl_signal['support']:.2f}" if tl_signal['support']
                     else "Hourly trendline BREAK DOWN detected")

        if int(latest.get("bull_flag", 0)):
            st.success("Bull Flag BREAK detected on daily chart — bullish continuation")
        elif int(latest.get("bear_flag", 0)):
            st.error("Bear Flag BREAK detected on daily chart — bearish continuation")

        if breadth_snap.get("zweig_active"):
            st.success("⭐ Zweig Breadth Thrust ACTIVE — rare, historically very bullish")
        if breadth_snap.get("hindenburg_active"):
            st.warning("⚠️ Hindenburg Omen ACTIVE — elevated distribution risk")

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
                st.warning("FOMC proximity — confidence reduced.")
            if not current_signal.get("trend_aligned"):
                st.warning("Counter-trend signal (against SMA50).")

        sent_cols = st.columns(4)
        sent_cols[0].metric("StockTwits Bull %",
                            f"{st_info.get('bull_ratio',0.5):.1%}")
        sent_cols[1].metric("News Sentiment", news_info.get("label","Neutral"))
        sent_cols[2].metric("Breadth EMA10",
                            f"{breadth_snap.get('breadth_ema10',0.5):.2f}" if breadth_snap else "N/A")
        sent_cols[3].metric("% Sectors > SMA50",
                            f"{breadth_snap.get('pct_above_sma50',0.5):.0%}" if breadth_snap else "N/A")

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.subheader("SPY Chart")
    chart_tabs = st.tabs([
        "Price Chart", "Hourly + Trendlines", "MACD",
        "Macro", "Breadth", "Events", "Backtest",
    ])

    with chart_tabs[0]:
        fig = build_main_chart(view_df, show_bb=cfg["show_bb"],
                               show_signals=cfg["show_signals"])
        fig = add_flag_markers(fig, view_df)
        st.plotly_chart(fig, use_container_width=True)

    with chart_tabs[1]:
        if not hourly_df.empty and "resistance" in tl_signal["hourly_df"].columns:
            st.plotly_chart(build_hourly_trendline_chart(tl_signal["hourly_df"]),
                            use_container_width=True)
            tl_cols = st.columns(2)
            if tl_signal["resistance"]:
                tl_cols[0].metric("Resistance", f"${tl_signal['resistance']:.2f}")
            if tl_signal["support"]:
                tl_cols[1].metric("Support", f"${tl_signal['support']:.2f}")
        else:
            st.info("Hourly data not available.")

    with chart_tabs[2]:
        if "MACD" in view_df.columns:
            import plotly.graph_objects as _go
            fig_m = _go.Figure()
            fig_m.add_trace(_go.Scatter(x=view_df.index, y=view_df["MACD"],
                                        name="MACD", line=dict(color="#26a69a")))
            fig_m.add_trace(_go.Scatter(x=view_df.index, y=view_df["MACD_signal"],
                                        name="Signal", line=dict(color="#ef5350")))
            fig_m.add_trace(_go.Bar(
                x=view_df.index, y=view_df["MACD_hist"], name="Histogram",
                marker_color=["#26a69a" if v >= 0 else "#ef5350"
                              for v in view_df["MACD_hist"].fillna(0)]))
            fig_m.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_m, use_container_width=True)

    with chart_tabs[3]:
        if not macro_df.empty:
            st.plotly_chart(build_macro_chart(
                spy_df.loc[start:end], macro_df, lookback_days=180),
                use_container_width=True)
            # Current macro readings table
            macro_tbl_cols = ["DXY","DXY_roc5","Oil","Oil_roc5",
                               "Yield10Y","Yield10Y_roc5"]
            if not macro_df.empty:
                avail = [c for c in macro_tbl_cols if c in macro_df.columns]
                last_macro = macro_df[avail].tail(1).T.rename(columns={macro_df.index[-1]: "Latest"})
                last_macro["Latest"] = last_macro["Latest"].apply(
                    lambda x: f"{x:.4f}" if abs(x) < 1 else f"{x:.2f}"
                )
                st.dataframe(last_macro, use_container_width=True)
        else:
            st.info("Macro data unavailable. No API key required for DXY/Oil/Yield (uses yfinance).")

    with chart_tabs[4]:
        if not breadth_df.empty:
            b_cols = st.columns(4)
            b_cols[0].metric("Breadth EMA10",
                             f"{breadth_snap.get('breadth_ema10',0):.2f}")
            b_cols[1].metric("% > SMA50",
                             f"{breadth_snap.get('pct_above_sma50',0):.0%}")
            b_cols[2].metric("52wk Highs (sectors)",
                             breadth_snap.get("n_new_highs", 0))
            b_cols[3].metric("52wk Lows (sectors)",
                             breadth_snap.get("n_new_lows", 0))

            z_date = breadth_snap.get("last_zweig_date")
            h_date = breadth_snap.get("last_ho_date")
            ind_cols = st.columns(2)
            ind_cols[0].metric(
                "Zweig Breadth Thrust",
                "🔥 ACTIVE" if breadth_snap.get("zweig_active") else
                f"Last: {z_date}" if z_date else "No signal",
            )
            ind_cols[1].metric(
                "Hindenburg Omen",
                "⚠️ ACTIVE" if breadth_snap.get("hindenburg_active") else
                f"Last: {h_date}" if h_date else "Clear",
            )
            st.plotly_chart(build_breadth_chart(breadth_df), use_container_width=True)
        else:
            st.info("Breadth data unavailable — sector ETFs could not be fetched.")

    with chart_tabs[5]:
        st.subheader("Economic Calendar")
        events = get_economic_calendar(days_ahead=45)
        if events:
            ev_df = pd.DataFrame(events)
            ev_df["date"] = ev_df["date"].astype(str)
            ev_df = ev_df.rename(columns={
                "date": "Date", "event": "Event",
                "importance": "Importance", "note": "Note",
                "days_away": "Days Away",
            })
            # Colour-code by importance
            def _style(row):
                color = {"HIGH": "#ef5350", "MEDIUM": "#FFB300", "LOW": "#26a69a"}.get(
                    row["Importance"], "")
                return [f"color: {color}"] * len(row)
            st.dataframe(
                ev_df[["Date","Days Away","Event","Importance","Note"]].style.apply(_style, axis=1),
                use_container_width=True, height=500,
            )
        else:
            st.info("No upcoming events in the next 45 days.")

    with chart_tabs[6]:
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
            with st.spinner("Training on 20yr data…"):
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
                st.dataframe(imp_df.head(12), use_container_width=True)
        else:
            st.info("No trained model. Click 'Train' to build one.")

    if cfg["auto_refresh"]:
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
