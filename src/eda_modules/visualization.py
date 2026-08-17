from __future__ import annotations

from typing import Any, Optional

from .constants import COLORWAY, TARGET_WER


def style_plotly_figure(
    fig: Any,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    show_target_wer: bool = False,
    y_is_wer: bool = False,
    legend_orientation: str = "h",
) -> Any:
    """Apply a consistent visual style and optional target-WER guide line."""
    fig.update_layout(
        template="plotly_white",
        colorway=COLORWAY,
        legend=dict(
            orientation=legend_orientation,
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.7)",
        ),
        margin=dict(l=60, r=30, t=80, b=60),
    )
    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)
    if show_target_wer and y_is_wer:
        fig.add_hline(
            y=TARGET_WER,
            line_dash="dash",
            line_color="firebrick",
            annotation_text=f"Ciljni WER = {TARGET_WER:.2f}",
            annotation_position="top left",
        )
    return fig
