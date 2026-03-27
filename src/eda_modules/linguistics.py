from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

from .visualization import style_plotly_figure


def load_text_corpus(text_dirs: List[Path], max_files: int | None = None) -> pd.DataFrame:
    """Load transcript corpus and compute base textual dimensions.

    When ``max_files`` is None, all available transcript files are loaded.
    """
    rows: List[Dict[str, Any]] = []
    for text_dir in text_dirs:
        if not text_dir.exists():
            continue
        txt_paths = list(text_dir.rglob("*.txt"))
        if max_files is not None:
            txt_paths = txt_paths[:max_files]
        for path in txt_paths:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not text:
                continue
            words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
            rows.append({"path": str(path), "text": text, "n_words": len(words), "n_chars": len(text)})
    return pd.DataFrame(rows)


def run_text_analysis(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Run lexical statistics, n-grams and script-mix diagnostics."""
    text_dirs = [
        ctx["ROOT"] / "data" / "aligned_raw_v1_improved" / "text",
        ctx["RUN_V1_DIR"] / "aligned_train" / "text",
        ctx["RUN_V1_DIR"] / "aligned_holdout" / "text",
    ]
    text_df = load_text_corpus(text_dirs, max_files=None)

    all_tokens: List[str] = []
    for text in text_df.get("text", pd.Series(dtype=str)).tolist():
        all_tokens.extend(re.findall(r"\w+", str(text).lower(), flags=re.UNICODE))

    if all_tokens:
        word_df = pd.DataFrame(Counter(all_tokens).most_common(30), columns=["word", "count"])
        fig = px.bar(word_df.sort_values("count"), x="count", y="word", orientation="h", title="Top 30 najčešćih reči")
        style_plotly_figure(fig, x_title="Frekvencija [count]", y_title="Reč")
        fig.show()

        def top_ngrams(tokens: List[str], n: int, top_k: int = 30) -> pd.DataFrame:
            grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            counts = Counter(grams).most_common(top_k)
            return pd.DataFrame({"ngram": [" ".join(g) for g, _ in counts], "count": [c for _, c in counts]})

        bigram_df = top_ngrams(all_tokens, 2)
        trigram_df = top_ngrams(all_tokens, 3)

        fig = px.bar(bigram_df.sort_values("count"), x="count", y="ngram", orientation="h", title="Top 30 bigrama")
        style_plotly_figure(fig, x_title="Frekvencija [count]", y_title="Bigram")
        fig.show()

        fig = px.bar(trigram_df.sort_values("count"), x="count", y="ngram", orientation="h", title="Top 30 trigrama")
        style_plotly_figure(fig, x_title="Frekvencija [count]", y_title="Trigram")
        fig.show()

    if not text_df.empty:
        plt.figure(figsize=(14, 6))
        sns.histplot(text_df["n_words"], bins=40, kde=True, color="#16A085")
        plt.title("Distribucija broja reči po transkriptu")
        plt.xlabel("Broj reči")
        plt.show()

        plt.figure(figsize=(14, 6))
        sns.histplot(text_df["n_chars"], bins=40, kde=True, color="#8E44AD")
        plt.title("Distribucija dužine transkripta po broju karaktera")
        plt.xlabel("Broj karaktera")
        plt.show()

        def script_ratio(value: str) -> Tuple[float, float]:
            cyr = len(re.findall(r"[\u0400-\u04FF]", value))
            lat = len(re.findall(r"[A-Za-zČĆŽŠĐčćžšđ]", value))
            total = max(cyr + lat, 1)
            return cyr / total, lat / total

        ratios = text_df["text"].astype(str).apply(script_ratio)
        text_df["cyr_ratio"] = [ratio[0] for ratio in ratios]
        text_df["lat_ratio"] = [ratio[1] for ratio in ratios]
        text_df["digit_ratio"] = text_df["text"].astype(str).str.count(r"\d") / text_df["n_chars"].clip(lower=1)
        text_df["special_ratio"] = text_df["text"].astype(str).str.count(r"[^\w\s]") / text_df["n_chars"].clip(lower=1)
        display(text_df[["cyr_ratio", "lat_ratio", "digit_ratio", "special_ratio"]].describe().T)

    return {"text_df": text_df, "all_tokens": all_tokens}
