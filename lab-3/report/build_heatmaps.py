from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "images"
CORPUS = ROOT / "тексты из условия lab-3"

MODELS = {
    "gigachat": "GigaChat",
    "qwen": "Qwen3.7-Plus",
    "korolev-ai": "Korolev AI",
    "deepseek": "DeepSeek",
}

REPRESENTATIONS = {
    "count_bow": "Мешок слов",
    "tfidf_lemma": "TF–IDF по леммам",
    "bge_m3": "BGE-M3",
}

# Each model uses its original names for the same output measures.
METRICS = (
    ("cosine", "Косинусная близость", "signed", ("cosine", "cosine_similarity")),
    ("pearson", "Корреляция Пирсона", "signed", ("pearson", "pearson_correlation")),
    ("spearman", "Корреляция Спирмена", "signed", ("spearman", "spearman_correlation")),
    (
        "weighted_jaccard",
        "Взвешенный коэффициент Жаккара",
        "similarity",
        ("weighted_jaccard", "weighted_jaccard_similarity"),
    ),
    (
        "weighted_dice",
        "Взвешенный коэффициент Дайса",
        "similarity",
        ("weighted_dice", "weighted_dice_similarity"),
    ),
    ("euclidean", "Евклидово расстояние", "distance", ("euclidean", "euclidean_distance")),
    ("manhattan", "Манхэттенское расстояние", "distance", ("manhattan", "manhattan_distance")),
    ("chebyshev", "Чебышёвское расстояние", "distance", ("chebyshev", "chebyshev_distance")),
    (
        "euclidean_closeness",
        "Близость по евклидову расстоянию",
        "similarity",
        ("euclidean_closeness", "euclidean_close", "euclidean_proximity", "inv_euclidean"),
    ),
    (
        "manhattan_closeness",
        "Близость по манхэттенскому расстоянию",
        "similarity",
        ("manhattan_closeness", "manhattan_close", "manhattan_proximity", "inv_manhattan"),
    ),
    (
        "chebyshev_closeness",
        "Близость по чебышёвскому расстоянию",
        "similarity",
        ("chebyshev_closeness", "chebyshev_close", "chebyshev_proximity", "inv_chebyshev"),
    ),
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 11,
        "axes.titlesize": 11.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def corpus_order():
    documents = sorted(
        path.relative_to(CORPUS).as_posix()
        for path in CORPUS.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".html", ".docx", ".odt"}
    )
    if not documents:
        raise ValueError(f"The corpus contains no supported documents: {CORPUS}")
    groups = []
    for index, document in enumerate(documents, start=1):
        topic = document.split("/", 1)[0]
        if not groups or groups[-1][0] != topic:
            groups.append([topic, index, index])
        else:
            groups[-1][2] = index
    return documents, groups


def load_matrices(documents):
    matrices = {}
    expected = set(documents)
    count = len(documents)
    for representation in REPRESENTATIONS:
        for metric, _, kind, aliases in METRICS:
            for model in MODELS:
                directory = ROOT / "models" / model / "final" / "output" / "matrices"
                candidates = [directory / f"{representation}__{name}.csv" for name in aliases]
                matches = [path for path in candidates if path.is_file()]
                if len(matches) != 1:
                    raise ValueError(f"Expected one source matrix for {model}/{representation}/{metric}: {matches}")
                path = matches[0]
                frame = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
                frame.index = frame.index.astype(str).str.replace("\\", "/", regex=False)
                frame.columns = frame.columns.astype(str).str.replace("\\", "/", regex=False)
                if (
                    frame.shape != (count, count)
                    or not frame.index.is_unique
                    or not frame.columns.is_unique
                    or set(frame.index) != expected
                    or set(frame.columns) != expected
                ):
                    raise ValueError(f"Document labels or matrix dimensions differ from the corpus: {path}")
                matrix = frame.loc[documents, documents].to_numpy(dtype=float)
                if not np.isfinite(matrix).all():
                    raise ValueError(f"A source matrix contains non-finite numbers: {path}")
                lower = -1.0 if kind == "signed" else 0.0
                if matrix.min() < lower - 1e-6:
                    raise ValueError(f"A source matrix contains values below {lower}: {path}")
                if kind != "distance" and matrix.max() > 1.0 + 1e-6:
                    raise ValueError(f"A source matrix contains values above one: {path}")
                matrices[representation, metric, model] = matrix

    for representation in REPRESENTATIONS:
        for metric in ("euclidean", "manhattan", "chebyshev"):
            for model in MODELS:
                distance = matrices[representation, metric, model]
                closeness = matrices[representation, f"{metric}_closeness", model]
                if not np.allclose(closeness, 1.0 / (1.0 + distance), rtol=2e-5, atol=1e-6):
                    raise ValueError(f"The distance and its closeness output disagree: {model}/{representation}/{metric}")
    return matrices


def value_label(value):
    return f"{value:.4g}".replace(".", ",")


def draw_matrix(ax, matrix, groups, *, vmin, vmax, cmap):
    count = len(matrix)
    edges = np.arange(count + 1) + 0.5
    image = ax.pcolormesh(
        edges,
        edges,
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        edgecolors="none",
        antialiased=False,
        rasterized=True,
    )
    ax.set_xlim(0.5, count + 0.5)
    ax.set_ylim(count + 0.5, 0.5)
    ax.set_aspect("equal")
    ticks = sorted({1, *range(10, count + 1, 10), count})
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Номер документа")
    ax.set_ylabel("Номер документа")
    ax.tick_params(length=3, pad=2)
    for _, _, last in groups[:-1]:
        ax.axvline(last + 0.5, color="white", linewidth=0.65, linestyle="--", alpha=0.9)
        ax.axhline(last + 0.5, color="white", linewidth=0.65, linestyle="--", alpha=0.9)
    return image


def comparison_figure(representation, metric, title, kind, matrices, groups):
    panels = [matrices[representation, metric, model] for model in MODELS]
    vmin = -1.0 if kind == "signed" else 0.0
    vmax = max(float(matrix.max()) for matrix in panels) if kind == "distance" else 1.0
    if vmax == vmin:
        vmax = vmin + 1.0
    cmap = {"similarity": "viridis", "signed": "coolwarm", "distance": "magma_r"}[kind]

    fig, axes = plt.subplots(2, 2, figsize=(6.45, 7.1), layout="constrained")
    for letter, ax, matrix, model_name in zip(("а", "б", "в", "г"), axes.ravel(), panels, MODELS.values()):
        image = draw_matrix(ax, matrix, groups, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(
            f"{letter}) {model_name}\nДиапазон {value_label(matrix.min())}…{value_label(matrix.max())}",
            pad=5,
        )
    fig.suptitle(f"{REPRESENTATIONS[representation]}\n{title}", fontsize=12.5)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.025, shrink=0.78)
    colorbar.set_ticks(np.linspace(vmin, vmax, 5))
    colorbar.set_ticklabels([value_label(value) for value in np.linspace(vmin, vmax, 5)])
    colorbar.set_label("Расстояние" if kind == "distance" else "Значение меры", fontsize=11)
    colorbar.ax.tick_params(labelsize=11)
    return fig


def main():
    documents, groups = corpus_order()
    matrices = load_matrices(documents)
    OUT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="heatmaps-", dir=OUT) as temporary_directory:
        staging = Path(temporary_directory)
        for representation in REPRESENTATIONS:
            for metric, title, kind, _ in METRICS:
                fig = comparison_figure(representation, metric, title, kind, matrices, groups)
                fig.savefig(staging / f"comparison_{representation}_{metric}.pdf", dpi=300)
                if (representation, metric) == ("bge_m3", "cosine"):
                    fig.savefig(staging / "comparison_bge_cosine.png", dpi=240)
                plt.close(fig)

        # Preserve the four single-model assets used by previous report revisions.
        for model, model_name in MODELS.items():
            matrix = matrices["bge_m3", "cosine", model]
            fig, ax = plt.subplots(figsize=(5.2, 4.8), layout="constrained")
            image = draw_matrix(ax, matrix, groups, vmin=-1.0, vmax=1.0, cmap="coolwarm")
            ax.set_title(f"{model_name}\nBGE-M3, косинусная близость", fontsize=12)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label("Косинусная близость")
            name = "korolev" if model == "korolev-ai" else model
            fig.savefig(staging / f"{name}_bge_cosine.png", dpi=220)
            plt.close(fig)

        for asset in staging.iterdir():
            asset.replace(OUT / asset.name)

    print(f"Validated {len(matrices)} source matrices for {len(documents)} documents.")
    print(f"Created {len(REPRESENTATIONS) * len(METRICS)} comparison PDFs and preserved five PNG assets in {OUT}.")
    for topic, first, last in groups:
        print(f"{first}–{last}: {topic}")


if __name__ == "__main__":
    main()
