from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from train.catalog.synthetic import TEXT_SETS, expand_decision
from train.common.env import inspect_environment
from train.common.paths import find_repo, pretrained_dir
from train.common.ranking import markdown_table

DECISION_COLUMNS = [
    "ocr_name", "ocr_number", "ocr_collection",
    "gt_name", "gt_number", "gt_set", "gt_raw",
    "catalog_number", "catalog_set_id",
    "catalog_name_en", "catalog_name_pt",
    "catalog_set_name_en", "catalog_set_name_pt",
    "in_catalog", "origem",
]

FEATURE_COLUMNS = ["name_score", "name_exact", "number_exact", "number_near", "total_exact", "set_hit"]
WEIGHT_ORDER = ["name_w", "name_exact", "number_w", "number_near", "total_w", "set_w"]
# Uma grade só. O zero desliga o campo; os outros valores espalham do baixo ao alto e guardam os pesos que já empataram no topo.
GRID_AXES = {
    "name_w": [0, 100, 250, 400, 700, 1000],
    "name_exact": [0, 10, 40, 100],
    "number_w": [0, 40, 110, 250, 500],
    "number_near": [0, 20, 40, 80],
    "total_w": [0, 30, 120, 250],
    "set_w": [0, 40, 150, 400, 800],
}
LOCAL_SPAN = 20
LOCAL_STEP = 10
OLD_WEIGHTS = {
    "name_w": 1000.0,
    "name_exact": 40.0,
    "number_w": 50.0,
    "number_near": 15.0,
    "total_w": 25.0,
    "set_w": 10.0,
}


def number_key(text: str) -> str:
    raw = re.sub(r"\s+", "", str(text or "")).upper()
    match = re.search(r"([A-Z]*)0*(\d+)[/\-]0*(\d+)", raw)
    if not match:
        return ""
    return f"{match.group(1)}{int(match.group(2))}/{int(match.group(3))}"


def set_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(text or "")).upper()


def _api_path(repo: Path) -> Path:
    return repo.parent / "API-PTCG Scanner"


def _import_catalog(repo: Path):
    api = _api_path(repo)
    for folder in (api, api / "src"):
        text = str(folder)
        if text not in sys.path:
            sys.path.insert(0, text)
    from ptcg_api.catalog import CURRENT_WEIGHTS, load_catalogs, match_features

    return CURRENT_WEIGHTS, load_catalogs, match_features


def build_decision(repo: Path, eval_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(eval_path, dtype=str, keep_default_na=False)
    catalog = pd.read_csv(repo / "data" / "catalog" / "cards.csv", dtype=str, keep_default_na=False)
    truth = raw[(raw["gt_number"].str.strip() != "") & (raw["gt_set"].str.strip() != "")].copy()
    truth["number_key"] = truth["gt_number"].map(number_key)
    truth["set_key"] = truth["gt_set"].map(set_key)
    catalog["number_key"] = catalog["number"].map(number_key)
    catalog["set_key"] = catalog["set_id"].map(set_key)
    lookup = catalog.drop_duplicates(["number_key", "set_key"])[
        ["number_key", "set_key", "number", "set_id", "name_en", "name_pt", "set_name_en", "set_name_pt"]
    ].rename(
        columns={
            "number": "catalog_number",
            "set_id": "catalog_set_id",
            "name_en": "catalog_name_en",
            "name_pt": "catalog_name_pt",
            "set_name_en": "catalog_set_name_en",
            "set_name_pt": "catalog_set_name_pt",
        }
    )
    merged = truth.merge(lookup, on=["number_key", "set_key"], how="left")
    frame = pd.DataFrame(
        {
            "ocr_name": merged["ocr_name"],
            "ocr_number": merged["ocr_number"],
            "ocr_collection": merged["ocr_collection"],
            "gt_name": merged["gt_name"],
            "gt_number": merged["gt_number"],
            "gt_set": merged["gt_set"],
            "gt_raw": merged["gt_raw"],
            "catalog_number": merged["catalog_number"],
            "catalog_set_id": merged["catalog_set_id"],
            "catalog_name_en": merged["catalog_name_en"],
            "catalog_name_pt": merged["catalog_name_pt"],
            "catalog_set_name_en": merged["catalog_set_name_en"],
            "catalog_set_name_pt": merged["catalog_set_name_pt"],
            "in_catalog": merged["catalog_set_id"].fillna("").str.strip().ne("").astype(int),
            "origem": "real",
        }
    )
    return frame.fillna("")


def public_decision(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in DECISION_COLUMNS if column in frame.columns]
    return frame.loc[:, columns].copy()


def with_internal_id(frame: pd.DataFrame) -> pd.DataFrame:
    public = public_decision(frame).reset_index(drop=True)
    public.insert(0, "sample_id", [f"{index:05d}" for index in range(1, len(public) + 1)])
    return public


def _candidates_match(candidates: pd.DataFrame, usable: pd.DataFrame) -> bool:
    if candidates.empty or "sample_id" not in candidates.columns:
        return False
    expected = set(usable["sample_id"].astype(str))
    found = set(candidates["sample_id"].astype(str).unique())
    numbered = int(usable["gt_number_key"].ne("").sum())
    return bool(found) and found <= expected and len(found) >= numbered * 0.8


def build_candidates(rows: pd.DataFrame, repo: Path, match_features, load_catalogs) -> pd.DataFrame:
    catalog = load_catalogs(repo / "data" / "catalog")
    keep = ["number", "set_id", *FEATURE_COLUMNS]
    frames = []
    total = len(rows)
    for index, row in enumerate(rows.itertuples(index=False), start=1):
        features = match_features(row.ocr_name, row.ocr_number, row.ocr_collection, catalog)
        if features.empty or "name_score" not in features.columns:
            continue
        part = features.loc[:, keep].copy()
        part.insert(0, "sample_id", row.sample_id)
        frames.append(part)
        if index % 400 == 0 or index == total:
            print(f"  candidatos {index}/{total}", flush=True)
    if not frames:
        return pd.DataFrame(columns=["sample_id", *keep])
    return pd.concat(frames, ignore_index=True)


def _groups(candidates: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    grouped = []
    for sample_id, part in candidates.groupby("sample_id", sort=False):
        grouped.append(
            (
                str(sample_id),
                part[FEATURE_COLUMNS].to_numpy(np.float32),
                part["number"].astype(str).map(number_key).to_numpy(),
                part["set_id"].astype(str).map(set_key).to_numpy(),
            )
        )
    return grouped


def score_many(weights: np.ndarray, grouped, truth: dict) -> tuple[np.ndarray, np.ndarray]:
    count = len(weights)
    hits_real = np.zeros(count, dtype=np.int32)
    hits_synthetic = np.zeros(count, dtype=np.int32)
    numbers_of = truth["number"]
    sets_of = truth["set"]
    origins = truth["origem"]
    transposed = np.ascontiguousarray(weights.T)
    total = len(grouped)
    for index, (sample_id, matrix, numbers, sets) in enumerate(grouped, start=1):
        origin = origins.get(sample_id)
        if origin is None or len(matrix) == 0:
            continue
        target = hits_real if origin == "real" else hits_synthetic
        wanted_number = numbers_of[sample_id]
        wanted_set = sets_of[sample_id]
        chunk = max(1, 32_000_000 // max(len(matrix), 1))
        for start in range(0, count, chunk):
            stop = min(count, start + chunk)
            chosen = (matrix @ transposed[:, start:stop]).argmax(axis=0)
            ok = (numbers[chosen] == wanted_number) & (sets[chosen] == wanted_set) & (numbers[chosen] != "")
            target[start:stop] += ok
        if index == total or index % 1000 == 0:
            print(f"    cartas {index}/{total}", flush=True)
    return hits_real, hits_synthetic


def _weight_frame(weights: np.ndarray, hits_real: np.ndarray, hits_synthetic: np.ndarray, label: str, n_real: int, n_synthetic: int) -> pd.DataFrame:
    n_all = n_real + n_synthetic
    data = {name: weights[:, index].astype(int) for index, name in enumerate(WEIGHT_ORDER)}
    data.update(
        {
            "label": label,
            "hits_real": hits_real,
            "hits_synthetic": hits_synthetic,
            "hit_real": hits_real / n_real if n_real else 0.0,
            "hit_synthetic": hits_synthetic / n_synthetic if n_synthetic else 0.0,
            "hit": (hits_real + hits_synthetic) / n_all if n_all else 0.0,
        }
    )
    return pd.DataFrame(data)


def local_axes(center: pd.Series | dict) -> dict[str, list[int]]:
    axes = {}
    for name in WEIGHT_ORDER:
        base = int(center[name])
        axes[name] = list(range(max(0, base - LOCAL_SPAN), base + LOCAL_SPAN + 1, LOCAL_STEP))
    return axes


def _combine(axes: dict[str, list[int]]) -> np.ndarray:
    lists = [axes[name] for name in WEIGHT_ORDER]
    return np.asarray(list(itertools.product(*lists)), dtype=np.float32)


def pool_counts(grouped, truth: dict) -> tuple[int, int]:
    real = synthetic = 0
    numbers_of = truth["number"]
    sets_of = truth["set"]
    origins = truth["origem"]
    for sample_id, matrix, numbers, sets in grouped:
        origin = origins.get(sample_id)
        if origin is None or len(matrix) == 0:
            continue
        match = (numbers == numbers_of[sample_id]) & (sets == sets_of[sample_id]) & (numbers != "")
        if not bool(np.any(match)):
            continue
        if origin == "real":
            real += 1
        else:
            synthetic += 1
    return real, synthetic


def _pct(hits: int, total: int) -> str:
    if not total:
        return "0/0"
    return f"{hits}/{total} ({hits / total * 100:.2f}%)"


def _weights_text(row: pd.Series) -> str:
    return ", ".join(f"{name} {row[name]:.0f}" for name in WEIGHT_ORDER)


AXIS_LABELS = {
    "name_w": "nome",
    "name_exact": "nome exato",
    "number_w": "número",
    "number_near": "número perto",
    "total_w": "total",
    "set_w": "coleção",
}
MARKS = {
    "melhor": ("eleito", "#1f4e79"),
    "atual": ("pipeline", "#b85c38"),
    "nome_alto": ("nome alto", "#6b4c9a"),
    "melhor_sintetica": ("melhor sintética", "#2f6f4e"),
}


def save_catalog_figures(
    docs: Path,
    ranking: pd.DataFrame,
    *,
    pool_real: float | None = None,
    pool_synthetic: float | None = None,
) -> list[tuple[str, str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = docs / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    published: list[tuple[str, str, str]] = []

    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.8), sharey=True)
    curves = []
    for ax, name in zip(axes.ravel(), WEIGHT_ORDER):
        grouped = (
            ranking.groupby(name, as_index=False)
            .agg(real=("hit_real", "max"), synthetic=("hit_synthetic", "max"))
            .sort_values(name)
        )
        curves.append(grouped)
        ax.plot(grouped[name], grouped["real"] * 100, marker="o", color="#1f4e79", label="reais")
        ax.plot(grouped[name], grouped["synthetic"] * 100, marker="o", color="#b85c38", label="sintéticas")
        ax.set_title(AXIS_LABELS[name])
        ax.set_xlabel("peso")
        ax.grid(True, axis="y", alpha=0.3)
    low = min(float(part[["real", "synthetic"]].min().min()) for part in curves) * 100
    high = max(float(part[["real", "synthetic"]].max().max()) for part in curves) * 100
    for ax in axes.ravel():
        ax.set_ylim(max(0, low - 4), min(100, high + 2))
    axes[0, 0].set_ylabel("melhor acerto (%)")
    axes[1, 0].set_ylabel("melhor acerto (%)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / "acerto_por_peso.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    published.append(("Acerto por peso", "Melhor acerto em cada valor de peso. Os outros pesos variam juntos.", "figures/acerto_por_peso.png"))

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.scatter(
        ranking["hit_real"] * 100,
        ranking["hit_synthetic"] * 100,
        s=12,
        alpha=0.35,
        color="#8aa0b8",
        linewidths=0,
    )
    for label, (caption, color) in MARKS.items():
        part = ranking[ranking["label"] == label]
        if part.empty:
            continue
        row = part.iloc[0]
        ax.scatter([row["hit_real"] * 100], [row["hit_synthetic"] * 100], s=46, color=color, label=caption, zorder=3)
    ax.set_xlabel("acerto nas reais (%)")
    ax.set_ylabel("acerto nas sintéticas (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / "reais_vs_sinteticas.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    published.append(("Reais e sintéticas", "Cada ponto é uma combinação de pesos.", "figures/reais_vs_sinteticas.png"))

    present = [label for label in MARKS if (ranking["label"] == label).any()]
    if present:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        x = np.arange(len(present))
        real_bars = [float(ranking.loc[ranking["label"] == label, "hit_real"].iloc[0]) * 100 for label in present]
        syn_bars = [float(ranking.loc[ranking["label"] == label, "hit_synthetic"].iloc[0]) * 100 for label in present]
        ax.bar(x - 0.18, real_bars, width=0.36, color="#1f4e79", label="reais")
        ax.bar(x + 0.18, syn_bars, width=0.36, color="#b85c38", label="sintéticas")
        if pool_real is not None:
            ax.axhline(pool_real * 100, color="#1f4e79", linestyle="--", linewidth=1, label="teto reais")
        if pool_synthetic is not None:
            ax.axhline(pool_synthetic * 100, color="#b85c38", linestyle="--", linewidth=1, label="teto sintéticas")
        ax.set_xticks(x)
        ax.set_xticklabels([MARKS[label][0] for label in present])
        ax.set_ylabel("acerto (%)")
        ax.set_ylim(70, 100)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path = fig_dir / "veredito.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        published.append(("Veredito", "Pesos de referência. A linha tracejada é o teto: a carta certa está entre as candidatas.", "figures/veredito.png"))
    return published


def write_readme(
    repo: Path,
    docs: Path,
    env: pd.DataFrame,
    ranking: pd.DataFrame,
    *,
    stats: dict,
    eval_path: Path,
) -> Path:
    docs.mkdir(parents=True, exist_ok=True)
    labeled = {"atual", "nome_alto", "melhor", "melhor_sintetica"}
    show = pd.concat(
        [
            ranking[ranking["label"].isin(labeled)],
            ranking.head(8),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=WEIGHT_ORDER + ["label"])
    show = show.sort_values(["hit_real", "hit_synthetic", "number_w", "set_w"], ascending=False)
    table = show.copy()
    table["reais"] = (table["hit_real"] * 100).map(lambda value: f"{value:.2f}%")
    table["sinteticas"] = (table["hit_synthetic"] * 100).map(lambda value: f"{value:.2f}%")
    table["todas"] = (table["hit"] * 100).map(lambda value: f"{value:.2f}%")
    elected = ranking.iloc[0]
    current = ranking[ranking["label"] == "atual"].iloc[0]
    old = ranking[ranking["label"] == "nome_alto"].iloc[0]
    synthetic_best = ranking.sort_values(
        ["hit_synthetic", "hit_real", "number_w", "set_w"], ascending=False
    ).iloc[0]
    same_winner = all(elected[name] == synthetic_best[name] for name in WEIGHT_ORDER)
    n_real = stats["n_real"]
    n_synthetic = stats["n_synthetic"]
    n_all = n_real + n_synthetic
    lines = [
        "# Catálogo — pesos da decisão",
        "",
        "Relatório gerado por `python -m train catalog`. Não treina rede: compara pesos da escolha da carta no catálogo em cima das leituras já gravadas.",
        "",
        "O peso eleito é o de `pipeline.py` e o da API do site (`ptcg_api/catalog.py`).",
        "",
        "## Ambiente",
        "",
        "| Item | Valor |",
        "| --- | --- |",
    ]
    for _, row in env.iterrows():
        lines.append(f"| {row['Item']} | {row['Valor']} |")
    lines.extend(
        [
            "",
            "## Protocolo",
            "",
            f"- Fonte das leituras reais: `{eval_path.relative_to(repo).as_posix()}` · split `identify/test`.",
            "- Cada linha junta o texto lido (`ocr_name`, `ocr_number`, `ocr_collection`) à carta do catálogo com o mesmo número e a mesma coleção.",
            "- Acerto: o primeiro lugar tem o mesmo número e a mesma coleção da anotação. Nome em português ou inglês não entra.",
            "- Sem número de colecionador na leitura, a carta conta como erro.",
            "- O arquivo não guarda `sample_id` nem o nome da foto.",
            "- A linha sintética é uma carta do catálogo que não estava na leitura. O erro copia o padrão real: nome quase igual, número exato na maior parte, e o resto sem número ou com o total ou o colecionador deslocado.",
            "- `ocr_collection` sintético fica vazio antes de SVI. De Scarlet & Violet em diante o código é texto, e a leitura repete o OCR: vazio, o código, o código com PT ou EN, ou uma letra trocada.",
            "- A grade cruza todos os eixos, do zero até um peso alto: "
            + "; ".join(f"{name} ({', '.join(str(value) for value in GRID_AXES[name])})" for name in WEIGHT_ORDER)
            + ".",
            "- `atual` é o peso da pipeline e do site. `nome_alto` é a conta antiga, com nome até 1000.",
            "- O veredito elege o maior acerto nas leituras reais. Empate prefere o maior peso de número e, depois, o de coleção. As sintéticas entram no relatório ao lado, e não decidem o peso.",
            f"- Depois da grade, cada peso do eleito anda de {LOCAL_STEP} em {LOCAL_STEP}, de -{LOCAL_SPAN} a +{LOCAL_SPAN}.",
            "",
            "## O arquivo",
            "",
            "| | Reais | Sintéticas | Todas |",
            "| --- | --- | --- | --- |",
            f"| Cartas | {stats['rows_real']} | {stats['rows_synthetic']} | {stats['rows_real'] + stats['rows_synthetic']} |",
            f"| No `cards.csv` | {stats['catalog_real']} | {stats['catalog_synthetic']} | {stats['catalog_real'] + stats['catalog_synthetic']} |",
            f"| De SVI em diante | {stats['svi_real']} | {stats['svi_synthetic']} | {stats['svi_real'] + stats['svi_synthetic']} |",
            f"| Com coleção lida | {stats['col_real']} | {stats['col_synthetic']} | {stats['col_real'] + stats['col_synthetic']} |",
            f"| Com número lido | {stats['num_real']} | {stats['num_synthetic']} | {stats['num_real'] + stats['num_synthetic']} |",
            f"| Carta certa entre os candidatos | {_pct(stats['pool_real'], n_real)} | {_pct(stats['pool_synthetic'], n_synthetic)} | {_pct(stats['pool_real'] + stats['pool_synthetic'], n_all)} |",
            "",
            "A última linha é o teto desta conta. A soma linear não passa disso: se a carta certa não está no grupo, nenhum peso a escolhe.",
            "",
            "## Veredito",
            "",
            f"O peso eleito pelas fotos reais é {_weights_text(elected)}. Ele acerta {_pct(int(elected['hits_real']), n_real)} das leituras reais e {_pct(int(elected['hits_synthetic']), n_synthetic)} das sintéticas. Nas duas juntas, {_pct(int(elected['hits_real']) + int(elected['hits_synthetic']), n_all)}.",
            "",
            "Nota igual fica com o colecionador mais perto do número lido. A grade acima não conta esse desempate.",
            "",
            f"O peso que está na pipeline acerta {_pct(int(current['hits_real']), n_real)} das reais e {_pct(int(current['hits_synthetic']), n_synthetic)} das sintéticas. A conta antiga, com o nome até 1000, acerta {_pct(int(old['hits_real']), n_real)} das reais e {_pct(int(old['hits_synthetic']), n_synthetic)} das sintéticas.",
            "",
        ]
    )
    if same_winner:
        lines.append("O mesmo peso também é o melhor nas linhas sintéticas.")
    else:
        lines.append(
            f"Nas sintéticas o melhor seria {_weights_text(synthetic_best)}, com {_pct(int(synthetic_best['hits_synthetic']), n_synthetic)} nelas e {_pct(int(synthetic_best['hits_real']), n_real)} nas reais. A eleição fica com as fotos, porque as sintéticas repetem um padrão de erro e têm mais cartas de SVI em diante."
        )
    figures = save_catalog_figures(
        docs,
        ranking,
        pool_real=(stats["pool_real"] / n_real) if n_real else None,
        pool_synthetic=(stats["pool_synthetic"] / n_synthetic) if n_synthetic else None,
    )
    if figures:
        lines.extend(["", "## Gráficos", ""])
        for alt, caption, relative in figures:
            lines.append(f"![{alt}]({relative})")
            lines.append("")
            lines.append(caption)
            lines.append("")
    lines.extend(
        [
            "## Resultado",
            "",
            markdown_table(table, ["label", "reais", "sinteticas", "todas", *WEIGHT_ORDER]),
            "",
            "Tabela completa: [`ranking.csv`](ranking.csv).",
            "",
        ]
    )
    path = docs / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compara pesos da decisão do catálogo.")
    parser.add_argument("--rebuild", action="store_true", help="Refaz decision.csv e os candidatos a partir do pipeline_eval.csv.")
    args = parser.parse_args(argv)

    repo = find_repo()
    docs = repo / "train" / "catalog"
    cache = repo / "runs" / "catalog"
    cache.mkdir(parents=True, exist_ok=True)
    eval_path = repo / "output" / "tables" / "pipeline_eval.csv"
    decision_path = docs / "decision.csv"
    candidate_path = cache / "candidates.csv"
    legacy_candidates = docs / "candidates.csv"
    if not eval_path.is_file():
        print(f"Falta {eval_path}. Rode python pipeline.py no split identify antes.")
        return 1

    current, load_catalogs, match_features = _import_catalog(repo)
    if args.rebuild or not decision_path.is_file():
        print("Montando decision.csv ...", flush=True)
        catalog = pd.read_csv(repo / "data" / "catalog" / "cards.csv", dtype=str, keep_default_na=False)
        decision = expand_decision(build_decision(repo, eval_path), catalog)
        public_decision(decision).to_csv(decision_path, index=False, encoding="utf-8")
    else:
        decision = pd.read_csv(decision_path, dtype=str, keep_default_na=False)
    decision = with_internal_id(decision)
    usable = decision[decision["in_catalog"].astype(int) == 1].copy()
    usable["gt_number_key"] = usable["catalog_number"].map(number_key)
    usable["gt_set_key"] = usable["catalog_set_id"].map(set_key)
    print(f"{len(decision)} cartas | {len(usable)} no catálogo", flush=True)

    if legacy_candidates.is_file() and not candidate_path.is_file() and not args.rebuild:
        legacy_candidates.replace(candidate_path)
    if candidate_path.is_file() and not args.rebuild:
        cached = pd.read_csv(candidate_path, dtype={"sample_id": str})
        cache_ok = _candidates_match(cached, usable)
    else:
        cached = None
        cache_ok = False
    if args.rebuild or not cache_ok:
        print("Montando candidatos (a leitura de OCR não roda de novo) ...", flush=True)
        candidates = build_candidates(usable, repo, match_features, load_catalogs)
        candidates.to_csv(candidate_path, index=False)
    else:
        print(f"Candidatos em cache: {candidate_path}", flush=True)
        candidates = cached if cached is not None else pd.read_csv(candidate_path)
    grouped = _groups(candidates)
    usable_ids = usable.copy()
    usable_ids["sample_id"] = usable_ids["sample_id"].astype(str)
    truth = {
        "number": dict(zip(usable_ids["sample_id"], usable_ids["gt_number_key"])),
        "set": dict(zip(usable_ids["sample_id"], usable_ids["gt_set_key"])),
        "origem": dict(zip(usable_ids["sample_id"], usable_ids["origem"])),
    }
    n_real = int((usable_ids["origem"] == "real").sum())
    n_synthetic = int((usable_ids["origem"] == "sintetica").sum())
    pool_real, pool_synthetic = pool_counts(grouped, truth)
    print(f"{len(candidates)} candidatos | {len(grouped)} cartas com número", flush=True)

    references = np.asarray(
        [
            [current[name] for name in WEIGHT_ORDER],
            [OLD_WEIGHTS[name] for name in WEIGHT_ORDER],
        ],
        dtype=np.float32,
    )
    grid = _combine(GRID_AXES)
    packed = np.vstack([references, grid])
    print(f"  grade: {len(grid)} combinações", flush=True)
    packed_real, packed_synthetic = score_many(packed, grouped, truth)
    atual = _weight_frame(packed[0:1], packed_real[0:1], packed_synthetic[0:1], "atual", n_real, n_synthetic)
    nome_alto = _weight_frame(packed[1:2], packed_real[1:2], packed_synthetic[1:2], "nome_alto", n_real, n_synthetic)
    grade = _weight_frame(packed[2:], packed_real[2:], packed_synthetic[2:], "grade", n_real, n_synthetic)
    seeded = pd.concat([atual, nome_alto, grade], ignore_index=True)
    center = seeded.sort_values(["hit_real", "number_w", "set_w"], ascending=False).iloc[0]
    near = _combine(local_axes(center))
    print(f"  variação local: {len(near)} combinações", flush=True)
    near_real, near_synthetic = score_many(near, grouped, truth)
    local = _weight_frame(near, near_real, near_synthetic, "local", n_real, n_synthetic)
    ranking = pd.concat([seeded, local], ignore_index=True)
    ranking = ranking.drop_duplicates(subset=WEIGHT_ORDER, keep="first")
    ranking = ranking.sort_values(
        ["hit_real", "number_w", "set_w", "hit_synthetic"], ascending=False
    ).reset_index(drop=True)
    if ranking.iloc[0]["label"] in {"grade", "local"}:
        ranking.loc[0, "label"] = "melhor"
    synthetic_order = ranking.sort_values(
        ["hit_synthetic", "hit_real", "number_w", "set_w"], ascending=False
    )
    if synthetic_order.iloc[0]["label"] in {"grade", "local"}:
        ranking.loc[synthetic_order.index[0], "label"] = "melhor_sintetica"
    ranking_path = docs / "ranking.csv"
    ranking.to_csv(ranking_path, index=False, encoding="utf-8")

    def _slice(frame: pd.DataFrame) -> tuple[int, int, int, int, int]:
        catalog_rows = frame["in_catalog"].astype(int) == 1
        text_era = frame["catalog_set_id"].isin(TEXT_SETS)
        return (
            len(frame),
            int(catalog_rows.sum()),
            int((catalog_rows & text_era).sum()),
            int((catalog_rows & frame["ocr_collection"].str.strip().ne("")).sum()),
            int((catalog_rows & frame["ocr_number"].map(number_key).ne("")).sum()),
        )

    real_rows = decision[decision["origem"] == "real"]
    synthetic_rows = decision[decision["origem"] == "sintetica"]
    rows_real, catalog_real, svi_real, col_real, num_real = _slice(real_rows)
    rows_synthetic, catalog_synthetic, svi_synthetic, col_synthetic, num_synthetic = _slice(synthetic_rows)
    env, _device = inspect_environment(cache, pretrained_dir(repo))
    readme = write_readme(
        repo,
        docs,
        env,
        ranking,
        stats={
            "rows_real": rows_real,
            "rows_synthetic": rows_synthetic,
            "catalog_real": catalog_real,
            "catalog_synthetic": catalog_synthetic,
            "svi_real": svi_real,
            "svi_synthetic": svi_synthetic,
            "col_real": col_real,
            "col_synthetic": col_synthetic,
            "num_real": num_real,
            "num_synthetic": num_synthetic,
            "pool_real": pool_real,
            "pool_synthetic": pool_synthetic,
            "n_real": n_real,
            "n_synthetic": n_synthetic,
        },
        eval_path=eval_path,
    )
    best = ranking.iloc[0]
    print(
        f"Melhor nas reais: {best['hit_real'] * 100:.2f}%  nas sinteticas: {best['hit_synthetic'] * 100:.2f}%  "
        f"{ {name: best[name] for name in WEIGHT_ORDER} }"
    )
    print(f"README: {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
