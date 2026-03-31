from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TARGET_WER: float = 0.40
COLORWAY: List[str] = ["#1D3557", "#457B9D", "#2A9D8F", "#E9C46A", "#E76F51", "#A8DADC"]


def configure_plotting() -> None:
    """Configure deterministic plotting style for notebook visual consistency."""
    sns.set_theme(style="whitegrid", context="paper", palette=COLORWAY)
    plt.rcParams["figure.figsize"] = (13, 5)
    plt.rcParams["axes.titleweight"] = "bold"
    pd.set_option("display.max_colwidth", 200)
