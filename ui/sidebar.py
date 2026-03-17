"""Streamlit sidebar controls."""
import datetime
import streamlit as st


def render_sidebar() -> dict:
    """Render sidebar and return config dict with user selections."""
    st.sidebar.title("SPY Options Signal Tool")
    st.sidebar.markdown("---")

    # Date range
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=365)
    start_date = st.sidebar.date_input("Start date", value=default_start,
                                        min_value=datetime.date(2004, 1, 1),
                                        max_value=today)
    end_date = st.sidebar.date_input("End date", value=today,
                                      min_value=start_date,
                                      max_value=today)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Chart Settings")
    show_bb = st.sidebar.checkbox("Bollinger Bands", value=True)
    show_signals = st.sidebar.checkbox("Signal Markers", value=True)
    show_sma = st.sidebar.checkbox("SMAs (20/50/200)", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Signal Settings")
    use_ml = st.sidebar.checkbox("Use ML Model", value=True,
                                   help="If unchecked, uses rule-based signals only")
    signal_threshold = st.sidebar.slider(
        "Bull/Bear threshold offset", 0.0, 0.2, 0.15, 0.01,
        help="Added to 0.5 to get bull threshold, subtracted for bear"
    )
    min_confidence = st.sidebar.slider(
        "Min signal confidence (%)", 0, 60, 20, 5,
        help="Only enter trades when signal confidence exceeds this value"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Backtest Settings")
    bt_capital = st.sidebar.number_input("Initial capital ($)", value=100_000,
                                          min_value=1_000, step=10_000)
    bt_max_hold = st.sidebar.slider("Max hold days", 5, 45, 20)
    bt_expiry = st.sidebar.slider(
        "Option expiry days (at purchase)", 21, 90, 45, 7,
        help="Buy options with this many days to expiry; hold ≤ max hold days"
    )
    bt_min_hold = st.sidebar.slider(
        "Min hold days", 1, 10, 3,
        help="Minimum days before a signal reversal triggers exit"
    )
    bt_stop_loss = st.sidebar.slider(
        "Stop loss (% of premium)", 10, 80, 55, 5,
        help="Close trade if option loses this % of entry premium"
    )
    bt_profit_target = st.sidebar.slider(
        "Profit target (% of premium)", 20, 200, 60, 10,
        help="Close trade if option gains this % of entry premium"
    )
    bt_contracts = st.sidebar.number_input("Contracts per trade", value=1,
                                            min_value=1, max_value=100)

    st.sidebar.markdown("---")
    st.sidebar.subheader("API Keys")
    st.sidebar.caption("Keys are read from .env file. Override here for this session.")
    polygon_key = st.sidebar.text_input("Polygon API Key", type="password", value="")
    alpha_key = st.sidebar.text_input("Alpha Vantage Key", type="password", value="")
    fred_key = st.sidebar.text_input("FRED API Key", type="password", value="")

    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (1 min)", value=False)

    return {
        "start_date": start_date,
        "end_date": end_date,
        "show_bb": show_bb,
        "show_signals": show_signals,
        "show_sma": show_sma,
        "use_ml": use_ml,
        "signal_threshold": signal_threshold,
        "min_confidence": min_confidence,
        "bt_capital": bt_capital,
        "bt_max_hold": bt_max_hold,
        "bt_min_hold": bt_min_hold,
        "bt_stop_loss": bt_stop_loss / 100,        # convert % → fraction
        "bt_profit_target": bt_profit_target / 100,
        "bt_expiry": bt_expiry,
        "bt_contracts": bt_contracts,
        "polygon_key": polygon_key,
        "alpha_key": alpha_key,
        "fred_key": fred_key,
        "auto_refresh": auto_refresh,
    }
