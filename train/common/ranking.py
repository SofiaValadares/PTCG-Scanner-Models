from __future__ import annotations

from pathlib import Path

import pandas as pd

MAP_TOL = 0.002
OCR_EXACT_TOL = 1.0


def fmt(value, nd: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "—"
    if x != x:
        return "—"
    return f"{x:.{nd}f}"


def pick_yolo_winner(ok: pd.DataFrame, map_tol: float = MAP_TOL) -> pd.Series:
    """Highest val mAP50-95. Speed only breaks a tie at 4 decimal places.

    `val_infer_ms` jitters between val() runs. A 0.002 mAP window used to flip
    YOLOv26 (0.9938) vs YOLOv8 (0.9937) when latency noise changed.
    """
    del map_tol
    work = ok.copy()
    work["_map4"] = work["val_map50_95"].astype(float).round(4)
    best = float(work["_map4"].max())
    near = work[work["_map4"] >= best].copy()
    if "val_infer_ms" in near.columns and near["val_infer_ms"].notna().any():
        near = near.sort_values(["val_infer_ms", "val_map50_95"], ascending=[True, False])
    else:
        near = near.sort_values("val_map50_95", ascending=False)
    return near.iloc[0]


def mark_yolo_winner(ranking: pd.DataFrame, map_tol: float = MAP_TOL) -> tuple[pd.DataFrame, pd.Series]:
    work = ranking.copy()
    if "winner" in work.columns:
        work = work.drop(columns=["winner"])
    ok = work.dropna(subset=["val_map50_95"]) if "val_map50_95" in work.columns else work.iloc[0:0]
    if ok.empty:
        raise RuntimeError("Nenhum candidato treinou com sucesso (sem val_map50_95).")
    winner = pick_yolo_winner(ok, map_tol=map_tol)
    work.insert(0, "winner", work["id"].astype(str).eq(str(winner["id"])).map({True: "←", False: ""}))
    sort_cols = [c for c in ("winner", "val_map50_95") if c in work.columns]
    work = work.sort_values(
        by=sort_cols,
        ascending=[False, False][: len(sort_cols)],
        na_position="last",
    ).reset_index(drop=True)
    return work, winner


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    view = df.copy()
    if columns:
        cols = [c for c in columns if c in view.columns]
        view = view[cols]
    if view.empty:
        return "_sem dados_"
    headers = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in view.iterrows():
        cells = []
        for col in view.columns:
            val = row[col]
            if pd.isna(val):
                cells.append("—")
            elif isinstance(val, float):
                cells.append(fmt(val, 4 if abs(val) < 10 else 2))
            else:
                cells.append(str(val).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_ranking(path: Path, ranking: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(path, index=False, encoding="utf-8")
