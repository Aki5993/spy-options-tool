"""Plotly candlestick chart with signal overlays, BB, VIX, and Sentiment subplots."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_main_chart(df: pd.DataFrame,
                     show_bb: bool = True,
                     show_signals: bool = True) -> go.Figure:
    """
    Build a 4-row subplot figure:
      Row 1: Candlestick + BB + signal markers
      Row 2: Volume bars
      Row 3: VIX line
      Row 4: RSI
    """
    rows = 4
    row_heights = [0.5, 0.15, 0.15, 0.20]
    subplot_titles = ["SPY Price", "Volume", "VIX", "RSI"]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
    )

    # Row 1: Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="SPY", increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    if show_bb and "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"],
            name="BB Upper", line=dict(color="rgba(100,100,255,0.4)", width=1),
            showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"],
            name="BB Lower", line=dict(color="rgba(100,100,255,0.4)", width=1),
            fill="tonexty", fillcolor="rgba(100,100,255,0.05)",
            showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_mid"],
            name="BB Mid", line=dict(color="rgba(100,100,255,0.6)", width=1, dash="dot"),
            showlegend=False,
        ), row=1, col=1)

    if show_signals and "signal_direction" in df.columns:
        bull_mask = df["signal_direction"] == "bullish"
        bear_mask = df["signal_direction"] == "bearish"

        if bull_mask.any():
            fig.add_trace(go.Scatter(
                x=df.index[bull_mask],
                y=df["Low"][bull_mask] * 0.995,
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#26a69a"),
                name="Buy Signal",
            ), row=1, col=1)

        if bear_mask.any():
            fig.add_trace(go.Scatter(
                x=df.index[bear_mask],
                y=df["High"][bear_mask] * 1.005,
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#ef5350"),
                name="Sell Signal",
            ), row=1, col=1)

    # SMA overlays
    for sma, color in [("SMA_20", "#FFD700"), ("SMA_50", "#FF8C00"), ("SMA_200", "#9400D3")]:
        if sma in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[sma],
                name=sma, line=dict(color=color, width=1),
            ), row=1, col=1)

    # Row 2: Volume
    if "Volume" in df.columns:
        colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            marker_color=colors, name="Volume", showlegend=False,
        ), row=2, col=1)

    # Row 3: VIX
    if "vix_level" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["vix_level"],
            name="VIX", line=dict(color="#FF6B6B", width=1.5),
        ), row=3, col=1)
        # VIX regime lines
        fig.add_hline(y=15, line_dash="dot", line_color="green", opacity=0.5, row=3, col=1)
        fig.add_hline(y=25, line_dash="dot", line_color="red", opacity=0.5, row=3, col=1)

    # Row 4: RSI
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"],
            name="RSI", line=dict(color="#7EC8E3", width=1.5),
        ), row=4, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", opacity=0.5, row=4, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", opacity=0.5, row=4, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=4, col=1)

    fig.update_layout(
        height=850,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(size=11),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="VIX", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1, range=[0, 100])

    return fig


def build_hourly_trendline_chart(hourly_df: pd.DataFrame) -> go.Figure:
    """
    Build a Plotly candlestick chart of hourly SPY data with:
      - Projected resistance trendline (red dashed)
      - Projected support trendline (green dashed)
      - Break-up markers (green triangles)
      - Break-down markers (red triangles)
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03,
                        subplot_titles=["SPY Hourly + Trendlines", "Volume"])

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=hourly_df.index,
        open=hourly_df["Open"], high=hourly_df["High"],
        low=hourly_df["Low"], close=hourly_df["Close"],
        name="SPY 1H",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Resistance trendline
    if "resistance" in hourly_df.columns:
        res = hourly_df["resistance"].replace(0, np.nan)
        fig.add_trace(go.Scatter(
            x=hourly_df.index, y=res,
            name="Resistance", mode="lines",
            line=dict(color="rgba(239,83,80,0.7)", width=1.5, dash="dash"),
        ), row=1, col=1)

    # Support trendline
    if "support" in hourly_df.columns:
        sup = hourly_df["support"].replace(0, np.nan)
        fig.add_trace(go.Scatter(
            x=hourly_df.index, y=sup,
            name="Support", mode="lines",
            line=dict(color="rgba(38,166,154,0.7)", width=1.5, dash="dash"),
        ), row=1, col=1)

    # Break-up markers
    if "tl_break_up" in hourly_df.columns:
        break_up = hourly_df[hourly_df["tl_break_up"] == 1]
        if not break_up.empty:
            fig.add_trace(go.Scatter(
                x=break_up.index,
                y=break_up["Low"] * 0.998,
                mode="markers",
                marker=dict(symbol="triangle-up", size=14, color="#26a69a",
                            line=dict(color="white", width=1)),
                name="TL Break Up",
            ), row=1, col=1)

    # Break-down markers
    if "tl_break_down" in hourly_df.columns:
        break_down = hourly_df[hourly_df["tl_break_down"] == 1]
        if not break_down.empty:
            fig.add_trace(go.Scatter(
                x=break_down.index,
                y=break_down["High"] * 1.002,
                mode="markers",
                marker=dict(symbol="triangle-down", size=14, color="#ef5350",
                            line=dict(color="white", width=1)),
                name="TL Break Down",
            ), row=1, col=1)

    # Volume
    if "Volume" in hourly_df.columns:
        colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(hourly_df["Close"], hourly_df["Open"])]
        fig.add_trace(go.Bar(
            x=hourly_df.index, y=hourly_df["Volume"],
            marker_color=colors, name="Volume", showlegend=False,
        ), row=2, col=1)

    fig.update_layout(
        height=600,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(size=11),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)

    return fig


def add_flag_markers(fig: go.Figure, df: pd.DataFrame,
                     row: int = 1, col: int = 1) -> go.Figure:
    """
    Overlay bull/bear flag break markers on an existing figure row.
    Call after build_main_chart to add flag annotations.
    """
    if "bull_flag" in df.columns:
        flags = df[df["bull_flag"] == 1]
        if not flags.empty:
            fig.add_trace(go.Scatter(
                x=flags.index,
                y=flags["Low"] * 0.993,
                mode="markers+text",
                marker=dict(symbol="triangle-up", size=13,
                            color="#00E5FF", line=dict(color="white", width=1)),
                text="F",
                textposition="bottom center",
                textfont=dict(size=8, color="#00E5FF"),
                name="Bull Flag Break",
            ), row=row, col=col)

    if "bear_flag" in df.columns:
        flags = df[df["bear_flag"] == 1]
        if not flags.empty:
            fig.add_trace(go.Scatter(
                x=flags.index,
                y=flags["High"] * 1.007,
                mode="markers+text",
                marker=dict(symbol="triangle-down", size=13,
                            color="#FF6B35", line=dict(color="white", width=1)),
                text="F",
                textposition="top center",
                textfont=dict(size=8, color="#FF6B35"),
                name="Bear Flag Break",
            ), row=row, col=col)

    return fig


def build_macro_chart(spy_df: pd.DataFrame, macro_df: pd.DataFrame,
                      lookback_days: int = 180) -> go.Figure:
    """
    4-row subplot showing SPY vs macro indicators (last lookback_days days):
      Row 1: SPY close (normalized to 100)
      Row 2: DXY
      Row 3: WTI Oil
      Row 4: 10-year Treasury yield
    """
    # Slice to lookback window
    spy   = spy_df.tail(lookback_days)
    macro = macro_df.reindex(spy.index, method="ffill")

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.40, 0.20, 0.20, 0.20],
        vertical_spacing=0.03,
        subplot_titles=["SPY (normalized)", "DXY — US Dollar Index",
                        "WTI Crude Oil ($)", "10-Year Treasury Yield (%)"],
    )

    # SPY normalized
    spy_norm = spy["Close"] / spy["Close"].iloc[0] * 100
    fig.add_trace(go.Scatter(x=spy.index, y=spy_norm, name="SPY",
                             line=dict(color="#26a69a", width=2)), row=1, col=1)

    # DXY
    if "DXY" in macro.columns:
        fig.add_trace(go.Scatter(x=macro.index, y=macro["DXY"], name="DXY",
                                 line=dict(color="#FFD700", width=1.5)), row=2, col=1)

    # Oil
    if "Oil" in macro.columns:
        fig.add_trace(go.Scatter(x=macro.index, y=macro["Oil"], name="WTI Oil",
                                 line=dict(color="#FF6B35", width=1.5),
                                 fill="tozeroy",
                                 fillcolor="rgba(255,107,53,0.08)"), row=3, col=1)

    # 10yr yield
    if "Yield10Y" in macro.columns:
        yield_vals = macro["Yield10Y"]
        fig.add_trace(go.Scatter(x=macro.index, y=yield_vals, name="10Y Yield",
                                 line=dict(color="#C77DFF", width=1.5)), row=4, col=1)
        fig.add_hline(y=4.5, line_dash="dot", line_color="red", opacity=0.5, row=4, col=1)

    fig.update_layout(
        height=700, template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=30),
        font=dict(size=11),
    )
    fig.update_yaxes(title_text="SPY", row=1, col=1)
    fig.update_yaxes(title_text="DXY", row=2, col=1)
    fig.update_yaxes(title_text="Oil $", row=3, col=1)
    fig.update_yaxes(title_text="Yield %", row=4, col=1)

    return fig


def build_breadth_chart(breadth_df: pd.DataFrame,
                         lookback_days: int = 252) -> go.Figure:
    """
    3-row breadth chart:
      Row 1: % sectors above SMA50 + Zweig/Hindenburg event markers
      Row 2: Breadth ratio + 10-day EMA
      Row 3: 52-week new highs vs new lows (sector ETFs)
    """
    df = breadth_df.tail(lookback_days)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.40, 0.30, 0.30],
        vertical_spacing=0.04,
        subplot_titles=["% Sectors Above SMA50", "Daily Breadth Ratio + EMA10",
                        "52-Week New Highs vs New Lows (Sectors)"],
    )

    # Row 1: % above SMA50
    if "pct_above_sma50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["pct_above_sma50"] * 100,
            name="% > SMA50", fill="tozeroy",
            line=dict(color="#26a69a"), fillcolor="rgba(38,166,154,0.15)",
        ), row=1, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1)

    # Zweig thrust events
    if "zweig_thrust" in df.columns:
        z = df[df["zweig_thrust"] == 1]
        if not z.empty:
            fig.add_trace(go.Scatter(
                x=z.index, y=[55] * len(z),
                mode="markers", marker=dict(symbol="star", size=14, color="#FFD700"),
                name="Zweig Thrust",
            ), row=1, col=1)

    # Hindenburg Omen events
    if "hindenburg_omen" in df.columns:
        h = df[df["hindenburg_omen"] == 1]
        if not h.empty:
            fig.add_trace(go.Scatter(
                x=h.index, y=[45] * len(h),
                mode="markers", marker=dict(symbol="x", size=10, color="#FF4444"),
                name="Hindenburg Omen",
            ), row=1, col=1)

    # Row 2: breadth ratio + EMA
    if "breadth_ratio" in df.columns:
        fig.add_trace(go.Bar(
            x=df.index, y=df["breadth_ratio"],
            name="Daily Breadth",
            marker_color=df["breadth_ratio"].apply(
                lambda x: "rgba(38,166,154,0.5)" if x >= 0.5 else "rgba(239,83,80,0.5)"
            ),
        ), row=2, col=1)
    if "breadth_ema10" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["breadth_ema10"],
            name="EMA10", line=dict(color="#FFD700", width=2),
        ), row=2, col=1)
        fig.add_hline(y=0.615, line_dash="dot", line_color="green",
                      opacity=0.6, row=2, col=1)
        fig.add_hline(y=0.400, line_dash="dot", line_color="red",
                      opacity=0.6, row=2, col=1)

    # Row 3: new highs vs lows
    if "n_new_highs_52wk" in df.columns:
        fig.add_trace(go.Bar(
            x=df.index, y=df["n_new_highs_52wk"],
            name="New Highs", marker_color="rgba(38,166,154,0.7)",
        ), row=3, col=1)
    if "n_new_lows_52wk" in df.columns:
        fig.add_trace(go.Bar(
            x=df.index, y=-df["n_new_lows_52wk"],
            name="New Lows", marker_color="rgba(239,83,80,0.7)",
        ), row=3, col=1)

    fig.update_layout(
        height=650, template="plotly_dark", barmode="overlay",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=30), font=dict(size=11),
    )
    return fig
