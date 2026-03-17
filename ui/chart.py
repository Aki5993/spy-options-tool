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
