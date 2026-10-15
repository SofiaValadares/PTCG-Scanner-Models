from __future__ import annotations

from pathlib import Path

import pandas as pd

from train.common.env import read_environment
from train.common.figures import collect_run_images, publish_images
from train.common.ranking import fmt, markdown_table
from train.common.yolo import SweepConfig


def _rel(repo: Path, path: Path) -> str:
    try:
        return path.relative_to(repo).as_posix()
    except ValueError:
        return path.as_posix()


def _val_columns(ranking: pd.DataFrame) -> list[str]:
    meta = [c for c in ("winner", "family", "model", "task", "imgsz", "batch", "epochs", "hours") if c in ranking.columns]
    return meta + [c for c in ranking.columns if str(c).startswith("val_")]


def _test_columns(ranking: pd.DataFrame) -> list[str]:
    meta = [c for c in ("winner", "family") if c in ranking.columns]
    return meta + [c for c in ranking.columns if str(c).startswith("test_")]


def _recipe_lines(cfg: SweepConfig) -> list[str]:
    lines = [
        f"- Épocas (teto): `{cfg.epochs}` · early stop `patience={cfg.patience}` · seed `{cfg.seed}`",
        f"- Otimizador: `{cfg.optim}`",
        f"- Loss: `{cfg.loss}`",
        f"- Augmentation: `{cfg.augment}`",
        f"- Critério do vencedor: maior **val mAP@0.50:0.95** (4 casas). Só se empatar nessa precisão vai para o menor `val_infer_ms` (o tempo de inferência varia entre `val()`).",
    ]
    lines.extend(f"- {note}" for note in cfg.notes)
    return lines


def write_yolo_readme(cfg: SweepConfig, ranking: pd.DataFrame, winner: pd.Series) -> Path:
    docs = cfg.docs_dir
    docs.mkdir(parents=True, exist_ok=True)
    env = read_environment(cfg.sweep_dir / "environment.json")
    figure_index: dict[str, list[tuple[str, Path]]] = {}
    for cand in cfg.candidates:
        cand_dir = cfg.sweep_dir / cand["id"]
        dest = docs / "figures" / cand["id"]
        items = collect_run_images(cand_dir, include_confusion=cfg.include_confusion)
        if not items:
            figure_index[cand["id"]] = []
            continue
        published = publish_images(items, dest)
        figure_index[cand["id"]] = published

    ranking_copy = docs / "ranking.csv"
    ranking.to_csv(ranking_copy, index=False, encoding="utf-8")

    title = cfg.name
    kind_label = "detector de carta" if cfg.kind == "detector" else "detector de ROI"
    winner_id = str(winner.get("id"))
    winner_family = str(winner.get("family"))
    lines = [
        f"# {title} — varredura de arquiteturas",
        "",
        f"Relatório gerado por `python -m train {cfg.kind}`. Este arquivo é o artefato **versionado no GitHub**: tabelas de todas as famílias, gráficos comprimidos de todas elas, e uma seção do modelo eleito. O PDF opcional (`report.pdf`) serve para o TCC e **não entra no git**.",
        "",
        f"Papel: **{kind_label}**. Runs (gitignored): `{_rel(cfg.repo, cfg.sweep_dir)}`. Peso de produção: `models/{cfg.winner_weight_name or (cfg.name + '.pt')}`."
        "",
        "## Ambiente",
        "",
    ]
    if env:
        lines.append("| Item | Valor |")
        lines.append("| --- | --- |")
        for key, value in env.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
    else:
        lines.append("_Rode o treino para gravar `environment.json`._")
        lines.append("")

    lines.extend(["## Receita", ""])
    lines.extend(_recipe_lines(cfg))
    lines.extend(["", "## Candidatos", "", "| Família | Checkpoint | Tarefa | imgsz | batch |", "| --- | --- | --- | --- | --- |"])
    for cand in cfg.candidates:
        task = cand.get("task", "obb")
        extra = f" + `{cand['init_from']}`" if cand.get("init_from") else ""
        lines.append(
            f"| {cand['family']} | `{cand['model']}`{extra} | {task} | {cand['imgsz']} | {cand['batch']} |"
        )

    lines.extend(
        [
            "",
            "## Ranking — validação (todas as famílias)",
            "",
            markdown_table(ranking, _val_columns(ranking)),
            "",
            "## Ranking — teste (todas as famílias)",
            "",
            markdown_table(ranking, _test_columns(ranking)),
            "",
            "## Modelo eleito",
            "",
            f"**{winner_family}** (`{winner_id}`, checkpoint `{winner.get('model')}`).",
            "",
            f"- val mAP50-95 = **{fmt(winner.get('val_map50_95'))}** · mAP75 = {fmt(winner.get('val_map75'))} · F1 = {fmt(winner.get('val_f1'), 3)} · {fmt(winner.get('val_infer_ms'), 1)} ms",
        ]
    )
    if "test_map50_95" in winner.index and pd.notna(winner.get("test_map50_95")):
        lines.append(
            f"- test mAP50-95 = **{fmt(winner.get('test_map50_95'))}** · mAP75 = {fmt(winner.get('test_map75'))} · F1 = {fmt(winner.get('test_f1'), 3)}"
        )
    lines.append(
        f"- Peso copiado para `models/{cfg.winner_weight_name or (cfg.name + '.pt')}`."
    )
    if str(winner.get("task") or "obb") == "detect":
        lines.append(
            "- Nota: RT-DETR treina caixa alinhada (AABB). O cropper espera OBB; se este for o vencedor, o warp usa o retângulo como quadrilátero."
        )
    lines.extend(
        [
            "",
            "A eleição usa só o val. O test confirma; não escolhe o modelo.",
            "",
            "## Gráficos de todas as famílias",
            "",
            "Cada família tem curvas de treino, PR/F1 e lotes de validação. As imagens abaixo foram redimensionadas (lado máximo 900 px) e as fotos JPEG comprimidas para o GitHub não repetir o problema dos notebooks pesados.",
            "",
        ]
    )
    for cand in cfg.candidates:
        mark = " ← eleito" if cand["id"] == winner_id else ""
        lines.append(f"### {cand['family']}{mark}")
        lines.append("")
        published = figure_index.get(cand["id"], [])
        if not published:
            lines.append(
                f"_Sem figuras em `{_rel(cfg.repo, cfg.sweep_dir / cand['id'])}`. "
                "Rode `python -m train "
                f"{cfg.kind} --report-only` com os `best.pt` locais para gerar os plots._"
            )
            lines.append("")
            continue
        for title, path in published:
            rel = path.relative_to(docs).as_posix()
            lines.append(f"**{title}**")
            lines.append("")
            lines.append(f"![{title} — {cand['family']}]({rel})")
            lines.append("")

    readme = docs / "README.md"
    readme.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"README: {readme}")
    return readme


def write_yolo_pdf(cfg: SweepConfig, ranking: pd.DataFrame, winner: pd.Series) -> Path:
    import fitz

    docs = cfg.docs_dir
    dest = docs / "report.pdf"
    env = read_environment(cfg.sweep_dir / "environment.json")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 48
    width = 595 - 80

    def new_page():
        nonlocal page, y
        page = doc.new_page(width=595, height=842)
        y = 48

    def ensure(space: float = 24):
        nonlocal y
        if y + space > 800:
            new_page()

    def text(msg: str, size: float = 10, bold: bool = False):
        nonlocal y
        font = "helv" if not bold else "hebo"
        ensure(size + 6)
        page.insert_text((40, y), msg[:140], fontsize=size, fontname=font)
        y += size + 6

    def para(msg: str, size: float = 10):
        nonlocal y
        font = "helv"
        for raw in msg.split("\n"):
            line = raw or " "
            while line:
                chunk = line[:110]
                ensure(size + 4)
                page.insert_text((40, y), chunk, fontsize=size, fontname=font)
                y += size + 4
                line = line[110:]

    text(f"{cfg.name} — relatório do sweep", size=16, bold=True)
    text(f"Modelo eleito: {winner.get('family')}  ({winner.get('id')})", size=12, bold=True)
    para(
        f"val mAP50-95={fmt(winner.get('val_map50_95'))}  "
        f"test mAP50-95={fmt(winner.get('test_map50_95'))}  "
        f"{fmt(winner.get('val_infer_ms'), 1)} ms"
    )
    y += 6
    text("Ambiente", size=13, bold=True)
    for key, value in env.items():
        para(f"{key}: {value}", size=9)
    y += 6
    text("Ranking val", size=13, bold=True)
    cols = [c for c in ("winner", "family", "epochs", "val_map50_95", "val_f1", "val_infer_ms") if c in ranking.columns]
    para("  ".join(cols), size=8)
    for _, row in ranking.iterrows():
        para("  ".join(str(row.get(c, ""))[:16] for c in cols), size=8)
    y += 6
    text("Ranking test", size=13, bold=True)
    tcols = [c for c in ("winner", "family", "test_map50_95", "test_f1", "test_infer_ms") if c in ranking.columns]
    para("  ".join(tcols), size=8)
    for _, row in ranking.iterrows():
        para("  ".join(str(row.get(c, ""))[:16] for c in tcols), size=8)

    for cand in cfg.candidates:
        fig_dir = docs / "figures" / cand["id"]
        images = sorted(fig_dir.glob("*")) if fig_dir.is_dir() else []
        if not images:
            continue
        new_page()
        mark = " (eleito)" if str(cand["id"]) == str(winner.get("id")) else ""
        text(f"{cand['family']}{mark}", size=14, bold=True)
        for img_path in images:
            ensure(280)
            if y > 520:
                new_page()
                text(f"{cand['family']}{mark} (cont.)", size=12, bold=True)
            rect = fitz.Rect(40, y, 40 + width, y + 260)
            try:
                page.insert_image(rect, filename=str(img_path), keep_proportion=True)
            except Exception:
                para(f"[falha ao inserir {img_path.name}]")
                continue
            y += 270
            para(img_path.name, size=8)

    dest.parent.mkdir(parents=True, exist_ok=True)
    doc.save(dest, deflate=True, garbage=4, clean=True)
    doc.close()
    size_mb = dest.stat().st_size / 1024**2
    print(f"PDF: {dest}  ({size_mb:.2f} MB)")
    return dest


def write_ocr_readme(
    *,
    repo: Path,
    docs_dir: Path,
    summary: pd.DataFrame,
    by_tipo: pd.DataFrame,
    winner: str,
    env: dict[str, str],
    figures: list[tuple[str, Path]],
    n_readable: int,
    n_illegible: int,
    notes: list[str],
) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ocr-cmp-v1 — comparação de motores de OCR",
        "",
        "Relatório gerado por `python -m train ocr`. **Não treina** um reconhecedor: compara EasyOCR, RapidOCR (Paddle ONNX) e Tesseract nas caixas anotadas de `data/ocr/`. O detector de ROI não entra neste laço.",
        "",
        "Ranking: maior **exact_clean** nas caixas legíveis; empate < 1 p.p. desempata no CER; depois latência.",
        "",
        "## Ambiente",
        "",
    ]
    if env:
        lines.append("| Item | Valor |")
        lines.append("| --- | --- |")
        for key, value in env.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
    lines.extend(["## Protocolo", ""])
    lines.extend(f"- {note}" for note in notes)
    lines.extend(
        [
            f"- Caixas legíveis no ranking: **{n_readable}**. Ilegíveis (fora do ranking): {n_illegible}.",
            "",
            "## Ranking (todas as engines)",
            "",
            markdown_table(summary),
            "",
            f"## Motor eleito",
            "",
            f"**{winner}** entra em `src/ocr/read_card.py` e no pipeline. EasyOCR e Tesseract ficam como baseline deste *benchmark*.",
            "",
            "## Acerto por tipo de faixa",
            "",
            markdown_table(by_tipo),
            "",
            "## Figuras",
            "",
        ]
    )
    for title, path in figures:
        rel = path.relative_to(docs_dir).as_posix()
        lines.append(f"**{title}**")
        lines.append("")
        lines.append(f"![{title}]({rel})")
        lines.append("")
    readme = docs_dir / "README.md"
    readme.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"README: {readme}")
    return readme


def write_ocr_pdf(docs_dir: Path, title: str, figures: list[tuple[str, Path]], body: list[str]) -> Path:
    import fitz

    dest = docs_dir / "report.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 48

    def new_page():
        nonlocal page, y
        page = doc.new_page(width=595, height=842)
        y = 48

    def ensure(space: float = 24):
        nonlocal y
        if y + space > 800:
            new_page()

    def para(msg: str, size: float = 10, bold: bool = False):
        nonlocal y
        font = "hebo" if bold else "helv"
        line = msg
        while line:
            chunk = line[:110]
            ensure(size + 4)
            page.insert_text((40, y), chunk, fontsize=size, fontname=font)
            y += size + 4
            line = line[110:]

    para(title, size=16, bold=True)
    for line in body:
        para(line, size=9)
    for caption, img_path in figures:
        ensure(280)
        if y > 520:
            new_page()
        para(caption, size=11, bold=True)
        rect = fitz.Rect(40, y, 555, y + 240)
        try:
            page.insert_image(rect, filename=str(img_path), keep_proportion=True)
            y += 250
        except Exception:
            para(f"[falha {img_path.name}]")
    dest.parent.mkdir(parents=True, exist_ok=True)
    doc.save(dest, deflate=True, garbage=4, clean=True)
    doc.close()
    print(f"PDF: {dest}  ({dest.stat().st_size / 1024**2:.2f} MB)")
    return dest
