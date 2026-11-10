#!/usr/bin/env python
"""
Analise comparativa das replicatas: prior puro vs agente pos-transfer-learning.

Consome a saida de `run_replicates.sh` e produz as tabelas e figuras da
secao de resultados: funil medio com desvio-padrao, metricas de espaco
quimico e os testes de hipotese correspondentes.

DESENHO E O QUE OS TESTES SIGNIFICAM
  Cada replicata e uma amostragem independente (semente distinta) do mesmo
  modelo. A unidade experimental e a REPLICATA, nao a molecula: uma metrica
  por replicata, N replicatas por grupo. Comparar moleculas individuais
  entre os dois conjuntos infla o n artificialmente (pseudo-replicacao) e
  produz p-valores sem sentido - 10.000 moleculas de uma mesma execucao nao
  sao 10.000 observacoes independentes do modelo.

  Os grupos NAO sao pareados: prior e agente sao modelos diferentes, e a
  semente r nao cria correspondencia entre a replicata r de um e do outro.
  Por isso Mann-Whitney independente, nao o teste de postos sinalizados.

  Com n=10 contra n=10, o menor p bilateral possivel em Mann-Whitney e
  1.08e-5. Separacao completa entre os grupos satura o teste nesse valor;
  isso e o piso do metodo, nao uma medida da forca do efeito. Para o
  tamanho do efeito leia o delta de Cliff (-1 a 1) e a diferenca de medias.

DESCRITORES: EM QUE ETAPA DO FUNIL
  As medias de descritores sao calculadas sobre o conjunto UNICO E VALIDO
  (etapa 2), antes dos filtros de novidade, SA score e drug-likeness. Medir
  depois deles compararia conjuntos selecionados por criterios que atingem
  os dois modelos de forma desigual - os filtros removem 51% do agente e
  15% do prior - e a diferenca observada seria efeito do filtro, nao do
  transfer learning.

Uso:
    docker compose run --rm tools python compare_replicates.py
    docker compose run --rm tools python compare_replicates.py --root results/replicates
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats

RDLogger.DisableLog("rdApp.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ----------------------------------------------------------------- SA score
def _load_sascorer():
    from rdkit.Chem import RDConfig

    contrib = Path(RDConfig.RDContribDir) / "SA_Score"
    sys.path.append(str(contrib))
    import sascorer  # type: ignore

    return sascorer


SASCORER = _load_sascorer()
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

LABELS = ("prior", "agent")
PRETTY = {"prior": "Prior (ChEMBL)", "agent": "Agente (pos-TL)"}

# --------------------------------------------------------------------- cores
# Paleta categorica validada (lightness, croma, separacao para daltonismo
# protan/tritan e contraste contra a superficie clara). Nao trocar por
# conveniencia: a ordem dos slots e o mecanismo de seguranca para CVD.
COLOR = {"prior": "#2a78d6", "agent": "#eb6834"}
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#b4b3ad"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": MUTED,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "axes.grid": False,
        "figure.dpi": 150,
    }
)


def style_axes(ax, grid_axis: str | None = "y") -> None:
    """Grid e eixos recessivos: os dados ficam acima da moldura."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(MUTED)
    if grid_axis:
        ax.grid(axis=grid_axis, color=MUTED, linewidth=0.5, alpha=0.45)
        ax.set_axisbelow(True)


# ------------------------------------------------------------ nomes do funil
# Lidos por POSICAO, nao pelo texto: os rotulos do filter_generated.py
# carregam os limiares numericos e mudam se o limiar mudar.
FUNNEL_STAGES = [
    "Amostradas",
    "Validas",
    "Unicas",
    "Fora do BraCoLi",
    "Novas",
    "Sintetizaveis",
    "Drug-like",
]


def read_funnel(path: Path) -> dict[str, int]:
    df = pd.read_csv(path)
    if len(df) != len(FUNNEL_STAGES):
        sys.exit(f"{path}: esperava {len(FUNNEL_STAGES)} etapas, achei {len(df)}")
    return dict(zip(FUNNEL_STAGES, df["N"].astype(int).tolist()))


# --------------------------------------------------------------- descritores
def describe_set(smiles: list[str], subsample: int, seed: int) -> dict[str, float]:
    """Descritores agregados de um conjunto de moleculas."""
    rows = []
    scaffolds = set()
    fps = []
    rng = np.random.default_rng(seed)
    # Subamostra apenas para a diversidade interna, que e O(n^2).
    idx_internal = set(
        rng.choice(len(smiles), size=min(subsample, len(smiles)), replace=False).tolist()
    )

    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        rows.append(
            {
                "MW": Descriptors.MolWt(mol),
                "LogP": Crippen.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol),
                "HBD": Descriptors.NumHDonors(mol),
                "HBA": Descriptors.NumHAcceptors(mol),
                "RotB": Descriptors.NumRotatableBonds(mol),
                "Rings": Descriptors.RingCount(mol),
                "QED": Descriptors.qed(mol),
                "SA_score": SASCORER.calculateScore(mol),
                "HeavyAtoms": mol.GetNumHeavyAtoms(),
            }
        )
        try:
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
        except Exception:
            pass
        if i in idx_internal:
            fps.append(MORGAN.GetFingerprint(mol))

    df = pd.DataFrame(rows)
    out = {f"mean_{c}": float(df[c].mean()) for c in df.columns}
    out["scaffold_diversity"] = len(scaffolds) / max(len(df), 1)
    out["n_scaffolds"] = float(len(scaffolds))

    # Diversidade interna: Tanimoto medio entre todos os pares da subamostra.
    # Valor BAIXO = conjunto diverso. Reportado como similaridade para nao
    # inverter o sinal na leitura.
    sims = []
    for i in range(1, len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
    out["internal_similarity"] = float(np.mean(sims)) if sims else float("nan")
    return out, df


def replicate_metrics(
    rep_dir: Path, label: str, subsample: int, cache: bool
) -> tuple[dict[str, float], pd.DataFrame]:
    """Metricas de uma replicata/modelo. Descritores sobre o conjunto unico."""
    filt = rep_dir / "filtered"
    funnel = read_funnel(filt / f"{label}_funnel.csv")

    n = funnel["Amostradas"]
    valid = funnel["Validas"]
    unique = funnel["Unicas"]

    m: dict[str, float] = {
        "n_sampled": float(n),
        "validity_pct": 100 * valid / n,
        "uniqueness_pct": 100 * unique / n,
        # Duplicacao interna sobre as VALIDAS: diversidade do gerador,
        # independente da taxa de validade.
        "duplication_pct": 100 * (1 - unique / max(valid, 1)),
        "not_in_bracoli_pct": 100 * funnel["Fora do BraCoLi"] / n,
        "novel_pct": 100 * funnel["Novas"] / n,
        "synthesizable_pct": 100 * funnel["Sintetizaveis"] / n,
        "druglike_pct": 100 * funnel["Drug-like"] / n,
    }

    # Etapa 2 (unicas) + similaridade maxima ao BraCoLi, ja calculada
    # pelo filter_generated.py.
    uniq_csv = filt / f"{label}_01_unique_with_similarity.csv"
    uniq = pd.read_csv(uniq_csv)
    smiles = uniq["SMILES"].astype(str).tolist()
    tani = uniq["max_tanimoto_bracoli"].to_numpy(dtype=float)

    m["mean_max_tanimoto_bracoli"] = float(np.mean(tani))
    m["median_max_tanimoto_bracoli"] = float(np.median(tani))
    m["near_copy_pct"] = float(100 * np.mean(tani > 0.60))

    # O calculo de descritores + diversidade interna domina o tempo desta
    # analise; cacheado por replicata para que reexecutar so as figuras
    # nao custe tudo de novo.
    cache_path = rep_dir / f"descriptors_{label}.csv"
    meta_path = rep_dir / f"descriptors_{label}.meta.json"
    if cache and cache_path.exists() and meta_path.exists():
        df = pd.read_csv(cache_path)
        agg = {f"mean_{c}": float(df[c].mean()) for c in df.columns}
        agg.update(json.loads(meta_path.read_text()))
    else:
        agg, df = describe_set(smiles, subsample=subsample, seed=12345)
        df.to_csv(cache_path, index=False)
        meta_path.write_text(
            json.dumps(
                {
                    k: agg[k]
                    for k in ("scaffold_diversity", "n_scaffolds", "internal_similarity")
                },
                indent=2,
            )
        )
    m.update(agg)
    m.update({f"funnel_{k}": float(v) for k, v in funnel.items()})
    return m, pd.DataFrame({"max_tanimoto_bracoli": tani})


# ------------------------------------------------------------------- estatist
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a>b) - P(a<b). Robusto, sem suposicao de normalidade."""
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    d = (a.mean() - b.mean()) / sp
    return d * (1 - 3 / (4 * (na + nb) - 9))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni: controla o erro familiar sem supor independencia."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def interpret_delta(d: float) -> str:
    a = abs(d)
    if a < 0.147:
        return "desprezivel"
    if a < 0.33:
        return "pequeno"
    if a < 0.474:
        return "medio"
    return "grande"


# --------------------------------------------------------------------- figuras
def fig_funnel(long: pd.DataFrame, nrep: int, outdir: Path) -> None:
    """Funil medio: retencao por etapa, barras agrupadas com DP."""
    stages = FUNNEL_STAGES[1:]  # "Amostradas" e sempre 100%
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    h = 0.36
    gap = 0.04  # separacao entre barras adjacentes
    ypos = np.arange(len(stages))[::-1]

    for k, label in enumerate(LABELS):
        sub = long.loc[long["label"] == label]
        means, sds = [], []
        for st in stages:
            pct = 100 * sub[f"funnel_{st}"] / sub["n_sampled"]
            means.append(pct.mean())
            sds.append(pct.std(ddof=1))
        off = (h / 2 + gap / 2) * (1 if k == 0 else -1)
        ax.barh(
            ypos + off,
            means,
            height=h,
            color=COLOR[label],
            label=PRETTY[label],
            xerr=sds,
            error_kw={"ecolor": INK_2, "elinewidth": 0.9, "capsize": 2.5},
        )
        for y, mu, sd in zip(ypos + off, means, sds):
            ax.text(
                mu + sd + 1.5,
                y,
                f"{mu:.1f}",
                va="center",
                ha="left",
                fontsize=7.5,
                color=INK_2,
            )

    ax.set_yticks(ypos)
    ax.set_yticklabels(stages)
    n_sampled = int(long["n_sampled"].iloc[0])
    ax.set_xlabel(f"Retencao sobre as {n_sampled:,} amostradas (%)".replace(",", "."))
    ax.set_xlim(0, 112)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_title(f"Funil de filtragem: media de {nrep} replicatas (barras = DP)")
    style_axes(ax, grid_axis="x")
    # Legenda abaixo do eixo: no canto ela disputa espaco com os rotulos
    # diretos das barras mais longas.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"funil_medio.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


DESC_PANELS = [
    ("validity_pct", "Validade quimica (%)"),
    ("duplication_pct", "Duplicacao interna (%)"),
    ("druglike_pct", "Rendimento final drug-like (%)"),
    ("mean_max_tanimoto_bracoli", "Tanimoto max. vs BraCoLi"),
    ("near_copy_pct", "Quase-copias do BraCoLi (%)"),
    ("internal_similarity", "Similaridade interna media"),
    ("scaffold_diversity", "Scaffolds unicos / molecula"),
    ("mean_MW", "Massa molecular media (Da)"),
    ("mean_LogP", "LogP medio"),
    ("mean_TPSA", "TPSA media (A2)"),
    ("mean_SA_score", "SA score medio"),
    ("mean_QED", "QED medio"),
]


def fig_metrics(long: pd.DataFrame, nrep: int, outdir: Path) -> None:
    """Uma replicata = um ponto. Com n=10 os pontos vao a vista, nao um boxplot."""
    ncol = 4
    nrow = int(np.ceil(len(DESC_PANELS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.2, 2.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(7)

    for ax, (metric, title) in zip(axes, DESC_PANELS):
        for k, label in enumerate(LABELS):
            vals = long.loc[long["label"] == label, metric].to_numpy(dtype=float)
            x = k + rng.uniform(-0.09, 0.09, size=len(vals))
            ax.scatter(
                x,
                vals,
                s=26,
                color=COLOR[label],
                edgecolor=SURFACE,
                linewidth=0.8,  # anel de 2px: separa pontos sobrepostos
                zorder=3,
                label=PRETTY[label] if ax is axes[0] else None,
            )
            mu, sd = vals.mean(), vals.std(ddof=1)
            ax.errorbar(
                k,
                mu,
                yerr=sd,
                fmt="_",
                markersize=22,
                color=INK,
                elinewidth=1.1,
                capsize=4,
                zorder=4,
            )
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Prior", "Agente"])
        ax.set_title(title, fontsize=8.5)
        style_axes(ax, grid_axis="y")

    for ax in axes[len(DESC_PANELS) :]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(
        f"Metricas por replicata (traco preto = media +/- DP, n={nrep})",
        fontsize=10.5,
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"metricas_por_replicata.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_tanimoto(
    pooled: dict[str, np.ndarray], threshold: float, nrep: int, outdir: Path
) -> None:
    """Distribuicao da similaridade maxima ao BraCoLi, replicatas agrupadas."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(0, 1, 61)
    for label in LABELS:
        v = pooled[label]
        dens, edges = np.histogram(v, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, dens, color=COLOR[label], linewidth=2.0, label=PRETTY[label])
        ax.fill_between(centers, dens, color=COLOR[label], alpha=0.12)

    ax.axvline(threshold, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)))
    # A esquerda da linha e abaixo do topo: a direita e em cima o rotulo
    # bate na legenda.
    ax.text(
        threshold - 0.015,
        ax.get_ylim()[1] * 0.62,
        f"limiar de novidade ({threshold:.2f})",
        fontsize=7.5,
        color=INK_2,
        va="center",
        ha="right",
    )
    ax.set_xlabel("Tanimoto ECFP4 maximo contra o BraCoLi")
    ax.set_ylabel("Densidade")
    ax.set_xlim(0, 1)
    ax.set_title(f"Proximidade ao BraCoLi das moleculas unicas ({nrep} replicatas)")
    style_axes(ax, grid_axis="y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"similaridade_bracoli.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ main
METRIC_PT = {
    "validity_pct": "Validade quimica (% das amostradas)",
    "uniqueness_pct": "Unicas (% das amostradas)",
    "duplication_pct": "Duplicacao interna (% das validas)",
    "not_in_bracoli_pct": "Fora do BraCoLi (% das amostradas)",
    "novel_pct": "Novas, Tanimoto <= 0.6 (% das amostradas)",
    "synthesizable_pct": "Sintetizaveis, SA <= 4.5 (% das amostradas)",
    "druglike_pct": "Drug-like, rendimento final (% das amostradas)",
    "mean_max_tanimoto_bracoli": "Tanimoto maximo medio vs BraCoLi",
    "median_max_tanimoto_bracoli": "Tanimoto maximo mediano vs BraCoLi",
    "near_copy_pct": "Quase-copias do BraCoLi (% das unicas)",
    "internal_similarity": "Similaridade interna media (Tanimoto)",
    "scaffold_diversity": "Scaffolds de Murcko unicos / molecula",
    "n_scaffolds": "Numero de scaffolds de Murcko unicos",
    "mean_MW": "Massa molecular media (Da)",
    "mean_LogP": "LogP medio",
    "mean_TPSA": "TPSA media (A2)",
    "mean_HBD": "Doadores de H medios",
    "mean_HBA": "Aceptores de H medios",
    "mean_RotB": "Ligacoes rotaveis medias",
    "mean_Rings": "Numero medio de aneis",
    "mean_QED": "QED medio",
    "mean_SA_score": "SA score medio",
    "mean_HeavyAtoms": "Atomos pesados medios",
}

TEST_METRICS = list(METRIC_PT.keys())


def main() -> None:
    ap = argparse.ArgumentParser(description="Comparacao prior vs agente com replicatas")
    ap.add_argument("--root", type=Path, default=Path("results/replicates"))
    ap.add_argument("--threshold", type=float, default=0.60,
                    help="limiar de novidade usado na filtragem (so para a figura)")
    ap.add_argument("--subsample", type=int, default=1000,
                    help="moleculas por replicata na diversidade interna (O(n^2))")
    ap.add_argument("--no-cache", action="store_true",
                    help="recalcula descritores mesmo com cache em disco")
    args = ap.parse_args()

    rep_dirs = sorted(d for d in args.root.glob("r[0-9]*") if d.is_dir())
    if not rep_dirs:
        sys.exit(f"Nenhuma replicata em {args.root}. Rode antes: bash run_replicates.sh")

    outdir = args.root
    figdir = outdir / "figuras"
    figdir.mkdir(parents=True, exist_ok=True)

    rows = []
    pooled: dict[str, list[np.ndarray]] = {l: [] for l in LABELS}

    for d in rep_dirs:
        for label in LABELS:
            if not (d / "filtered" / f"{label}_funnel.csv").exists():
                print(f"  aviso: {d.name}/{label} incompleta, ignorada")
                continue
            print(f"  {d.name}/{label} ...", flush=True)
            m, tani = replicate_metrics(d, label, args.subsample, not args.no_cache)
            m["replicate"] = d.name
            m["label"] = label
            rows.append(m)
            pooled[label].append(tani["max_tanimoto_bracoli"].to_numpy())
            # Metricas da replicata gravadas junto dos dados dela.
            (d / f"metrics_{label}.json").write_text(
                json.dumps({k: v for k, v in m.items()}, indent=2, default=str)
            )

    long = pd.DataFrame(rows)
    front = ["replicate", "label"]
    long = long[front + [c for c in long.columns if c not in front]]
    long.to_csv(outdir / "metricas_por_replicata.csv", index=False)

    counts = long["label"].value_counts().to_dict()
    n_prior, n_agent = counts.get("prior", 0), counts.get("agent", 0)
    if n_prior < 3 or n_agent < 3:
        sys.exit("Menos de 3 replicatas por grupo: sem base para comparar.")

    # ---------------------------------------------------------------- testes
    available = [m for m in TEST_METRICS if m in long.columns]
    recs = []
    for metric in available:
        a = long.loc[long["label"] == "prior", metric].to_numpy(dtype=float)
        b = long.loc[long["label"] == "agent", metric].to_numpy(dtype=float)
        if np.allclose(a, a[0]) and np.allclose(b, b[0]) and np.allclose(a[0], b[0]):
            u_p = t_p = 1.0
            u_stat = float("nan")
            method = "degenerado"
        else:
            # method="auto" do SciPy so usa a permutacao exata com n < 8; com
            # n=10 ele cai na aproximacao normal, que superestima o p por uma
            # ordem de grandeza nesta faixa (1.7e-4 contra o exato 1.08e-5).
            # Com metricas continuas nao ha empates, entao o exato se aplica.
            has_ties = len(np.unique(np.concatenate([a, b]))) < len(a) + len(b)
            method = "asymptotic" if has_ties else "exact"
            u_stat, u_p = stats.mannwhitneyu(
                a, b, alternative="two-sided", method=method
            )
            t_p = stats.ttest_ind(a, b, equal_var=False).pvalue
        recs.append(
            {
                "metrica": METRIC_PT[metric],
                "chave": metric,
                "prior_media": a.mean(),
                "prior_dp": a.std(ddof=1),
                "prior_cv_pct": 100 * a.std(ddof=1) / abs(a.mean()) if a.mean() else np.nan,
                "agente_media": b.mean(),
                "agente_dp": b.std(ddof=1),
                "agente_cv_pct": 100 * b.std(ddof=1) / abs(b.mean()) if b.mean() else np.nan,
                "diferenca": b.mean() - a.mean(),
                "U": u_stat,
                "p_mannwhitney": u_p,
                "metodo_mw": method,
                "p_welch": t_p,
                "cliffs_delta": cliffs_delta(b, a),
                "hedges_g": hedges_g(b, a),
            }
        )

    comp = pd.DataFrame(recs)
    comp["p_holm"] = holm(comp["p_mannwhitney"].tolist())
    comp["efeito"] = comp["cliffs_delta"].map(interpret_delta)
    comp["significativo_holm_0.05"] = comp["p_holm"] < 0.05
    comp = comp[
        [
            "metrica", "chave",
            "prior_media", "prior_dp", "prior_cv_pct",
            "agente_media", "agente_dp", "agente_cv_pct",
            "diferenca", "U", "metodo_mw", "p_mannwhitney", "p_holm", "p_welch",
            "cliffs_delta", "efeito", "hedges_g", "significativo_holm_0.05",
        ]
    ]
    comp.to_csv(outdir / "comparacao_prior_vs_agente.csv", index=False)

    # Funil medio lado a lado: a primeira tabela da secao de resultados.
    funnel_rows = []
    for st in FUNNEL_STAGES:
        r = {"Etapa": st}
        for label in LABELS:
            sub = long.loc[long["label"] == label]
            pct = 100 * sub[f"funnel_{st}"] / sub["n_sampled"]
            r[f"{label}_N_media"] = sub[f"funnel_{st}"].mean()
            r[f"{label}_N_dp"] = sub[f"funnel_{st}"].std(ddof=1)
            r[f"{label}_pct_media"] = pct.mean()
            r[f"{label}_pct_dp"] = pct.std(ddof=1)
        funnel_rows.append(r)
    funnel_df = pd.DataFrame(funnel_rows).round(2)
    funnel_df.to_csv(outdir / "funil_medio.csv", index=False)

    # ---------------------------------------------------------------- figuras
    print("\n  gerando figuras ...", flush=True)
    fig_funnel(long, n_prior, figdir)
    fig_metrics(long, n_prior, figdir)
    fig_tanimoto(
        {l: np.concatenate(pooled[l]) for l in LABELS}, args.threshold, n_prior, figdir
    )

    # ---------------------------------------------------------------- relatorio
    min_p = float(comp["p_mannwhitney"].min())
    # Separacao completa se le no delta de Cliff, nao no p: metricas com
    # empates caem na aproximacao assintotica e nao atingem o piso exato,
    # embora os grupos tambem nao se sobreponham.
    separated = comp.loc[comp["cliffs_delta"].abs() >= 1.0, "metrica"].tolist()

    def fmt_p(p: float) -> str:
        return f"{p:.1e}" if p < 1e-3 else f"{p:.4f}"

    md = [
        "# Prior vs agente pos-transfer-learning: comparacao com replicatas",
        "",
        "## Desenho",
        "",
        f"- Replicatas por modelo: **{n_prior}** (prior) e **{n_agent}** (agente)",
        f"- SMILES por replicata: **{int(long['n_sampled'].iloc[0]):,}**".replace(",", "."),
        "- Variou-se **apenas a semente de amostragem** (1..N). O checkpoint do",
        "  agente e o arquivo do prior sao fixos, logo a variancia medida e a de",
        "  amostragem do gerador, nao a de treino.",
        "- Unidade experimental: a **replicata**. Uma metrica por replicata, nao",
        "  uma por molecula - 10.000 moleculas de uma execucao nao sao 10.000",
        "  observacoes independentes do modelo.",
        "- Teste: Mann-Whitney bilateral para amostras independentes, **permutacao",
        "  exata** (o `method=\"auto\"` do SciPy cairia na aproximacao normal com",
        "  n=10, que superestima o p em cerca de uma ordem de grandeza nesta",
        "  faixa). Correcao de Holm-Bonferroni sobre a familia de metricas. Welch",
        "  reportado como referencia parametrica.",
        "- Descritores calculados sobre o conjunto **unico e valido** (etapa 2),",
        "  antes dos filtros de novidade, SA score e drug-likeness.",
        "",
        f"Com n={n_prior} contra n={n_agent}, o menor p bilateral alcancavel pela",
        f"permutacao exata e **{min_p:.2e}**, e varias metricas saturam esse piso.",
        "Saturar o piso nao mede a forca do efeito: significa apenas que o teste",
        "chegou ao limite do que n=10 por grupo permite resolver.",
        "",
        f"As **{len(separated)} de {len(comp)}** metricas abaixo tem separacao completa",
        "(delta de Cliff = +/-1): nenhuma replicata de um modelo se sobrepoe a",
        "nenhuma do outro. Leia a diferenca de medias para o tamanho do efeito.",
        "",
    ]
    md += [f"- {m}" for m in separated] or ["- (nenhuma)"]
    md += ["", "## Funil medio (% das amostradas)", ""]

    md += [
        "| Etapa | Prior | Agente |",
        "|---|---|---|",
    ]
    for _, r in funnel_df.iterrows():
        md.append(
            f"| {r['Etapa']} | {r['prior_pct_media']:.2f} +/- {r['prior_pct_dp']:.2f} "
            f"| {r['agent_pct_media']:.2f} +/- {r['agent_pct_dp']:.2f} |"
        )

    md += [
        "",
        "## Comparacao metrica a metrica",
        "",
        "Media +/- DP entre replicatas. `delta` = delta de Cliff do agente contra o",
        "prior: +1 significa que toda replicata do agente supera toda do prior, -1 o",
        "contrario, 0 nenhuma tendencia.",
        "",
        "| Metrica | Prior | Agente | Diferenca | p (Holm) | delta | Efeito |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in comp.iterrows():
        md.append(
            f"| {r['metrica']} | {r['prior_media']:.3f} +/- {r['prior_dp']:.3f} "
            f"| {r['agente_media']:.3f} +/- {r['agente_dp']:.3f} "
            f"| {r['diferenca']:+.3f} | {fmt_p(r['p_holm'])} "
            f"| {r['cliffs_delta']:+.2f} | {r['efeito']} |"
        )

    ns = comp.loc[~comp["significativo_holm_0.05"], "metrica"].tolist()
    md += [
        "",
        "## Metricas sem diferenca detectavel (Holm, alfa = 0,05)",
        "",
    ]
    md += [f"- {m}" for m in ns] or ["- (nenhuma: todas as metricas diferem)"]
    md += [
        "",
        "Ausencia de significancia aqui significa que **estas replicatas nao",
        "distinguem os modelos nesta metrica**, nao que os modelos sejam iguais.",
        f"Com n={n_prior} por grupo a potencia so alcanca efeitos grandes.",
        "",
        "## Arquivos",
        "",
        "| Arquivo | Conteudo |",
        "|---|---|",
        "| `metricas_por_replicata.csv` | uma linha por replicata e modelo, todas as metricas |",
        "| `comparacao_prior_vs_agente.csv` | tabela de testes completa, com Welch e Hedges g |",
        "| `funil_medio.csv` | funil medio em N absoluto e em percentual, com DP |",
        "| `figuras/funil_medio.{png,pdf}` | funil medio, barras agrupadas com DP |",
        "| `figuras/metricas_por_replicata.{png,pdf}` | uma replicata por ponto, media +/- DP |",
        "| `figuras/similaridade_bracoli.{png,pdf}` | distribuicao do Tanimoto maximo ao BraCoLi |",
        "| `rNN/` | dados brutos de cada replicata: SMILES, funil, descritores, metricas |",
        "",
    ]
    (outdir / "RELATORIO.md").write_text("\n".join(md), encoding="utf-8")

    # ------------------------------------------------------------------ stdout
    pd.set_option("display.width", 200)
    print(f"\n=== Funil medio ({n_prior} replicatas por modelo) ===")
    print(
        funnel_df[
            ["Etapa", "prior_pct_media", "prior_pct_dp", "agent_pct_media", "agent_pct_dp"]
        ].to_string(index=False)
    )
    print("\n=== Comparacao (agente - prior) ===")
    print(
        comp[["metrica", "prior_media", "agente_media", "diferenca", "p_holm", "cliffs_delta", "efeito"]]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nRelatorio: {outdir / 'RELATORIO.md'}")
    print(f"Figuras:   {figdir}")


if __name__ == "__main__":
    main()
