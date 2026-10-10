from __future__ import annotations

import pandas as pd

from cvn_rl.metrics import summarize_table


def build_summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    return summarize_table(metrics)
