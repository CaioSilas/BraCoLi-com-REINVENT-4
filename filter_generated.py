#!/usr/bin/env python
"""
Pos-processamento das moleculas geradas pelo REINVENT 4.

Cascata de filtros (etapas 2 a 5 do pipeline acordado):

  1. Validade quimica + padronizacao + canonicalizacao
  2. Deduplicacao interna (InChIKey)
  3. NOVIDADE: remove moleculas identicas ao BraCoLi e as que sao
     quase-copias (Tanimoto ECFP4 acima do limiar)
  4. SINTETIZABILIDADE: SA score (Ertl & Schuffenhauer)
  5. DRUG-LIKENESS: regras de Lipinski + TPSA <= 140 A^2
     (mesmos criterios de Campera, 2026, para comparabilidade)

Cada etapa grava um CSV e um contador, de modo que o funil completo
possa ser reportado como tabela no artigo.

Uso:
    python filter_generated.py results/sampled_agent.csv \\
        --reference data/bracoli_clean.smi \\
        --label agent \\
        --outdir results/filtered

    python filter_generated.py results/sampled_prior.csv \\
        --reference data/bracoli_clean.smi \\
        --label prior \\
        --outdir results/filtered
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, inchi
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------- SA score
# sascorer vive no diretorio Contrib do RDKit e nao e importavel direto.
def _load_sascorer():
    try:
        from rdkit.Chem import RDConfig
    except ImportError:
        sys.exit("RDKit nao encontrado.")
    contrib = Path(RDConfig.RDContribDir) / "SA_Score"
    sys.path.append(str(contrib))
    try:
        import sascorer  # type: ignore
    except ImportError:
        sys.exit(
            f"sascorer nao encontrado em {contrib}.\n"
            "Instale o RDKit completo (conda install -c conda-forge rdkit)."
        )
    return sascorer


SASCORER = _load_sascorer()

# ECFP4 = Morgan radius 2, 2048 bits (mesmo usado na analise do BraCoLi)
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

ALLOWED_ATOMS = {"C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


# ---------------------------------------------------------------- etapa 1
def standardize(smiles: str, tools: dict) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = tools["lfc"].choose(mol)
        mol = tools["uncharger"].uncharge(mol)
        mol = tools["tautomer"].Canonicalize(mol)
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ALLOWED_ATOMS:
            return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def build_standardizers() -> dict:
    return {
        "lfc": rdMolStandardize.LargestFragmentChooser(),
        "uncharger": rdMolStandardize.Uncharger(),
        "tautomer": rdMolStandardize.TautomerEnumerator(),
    }


# ---------------------------------------------------------------- descritores
def describe(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)

    # Lipinski: tolera no maximo uma violacao, criterio usual.
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

    return {
        "MW": round(mw, 2),
        "LogP": round(logp, 2),
        "HBD": hbd,
        "HBA": hba,
        "TPSA": round(tpsa, 2),
        "RotB": Descriptors.NumRotatableBonds(mol),
        "Rings": Descriptors.RingCount(mol),
        "QED": round(Descriptors.qed(mol), 3),
        "SA_score": round(SASCORER.calculateScore(mol), 3),
        "lipinski_violations": violations,
    }


# ---------------------------------------------------------------- novidade
def max_similarity(smiles_list: list[str], ref_smiles: list[str]) -> np.ndarray:
    """Tanimoto ECFP4 maximo de cada molecula contra o conjunto de referencia."""
    ref_fps = []
    for s in ref_smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            ref_fps.append(MORGAN.GetFingerprint(m))
    if not ref_fps:
        sys.exit("Conjunto de referencia vazio ou invalido.")

    out = np.zeros(len(smiles_list))
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            out[i] = 1.0
            continue
        fp = MORGAN.GetFingerprint(m)
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        out[i] = max(sims)
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Filtragem das moleculas geradas")
    ap.add_argument("input", type=Path, help="CSV de sampling do REINVENT")
    ap.add_argument("--reference", type=Path, required=True,
                    help=".smi do BraCoLi curado (data/bracoli_clean.smi)")
    ap.add_argument("--label", required=True, help="Rotulo do conjunto: agent | prior")
    ap.add_argument("--outdir", type=Path, default=Path("results/filtered"))
    ap.add_argument("--smiles-col", default="SMILES")
    ap.add_argument("--max-similarity", type=float, default=0.60,
                    help="Tanimoto maximo permitido vs BraCoLi (default: 0.60)")
    ap.add_argument("--max-sa", type=float, default=4.5,
                    help="SA score maximo aceito (default: 4.5)")
    ap.add_argument("--max-tpsa", type=float, default=140.0)
    ap.add_argument("--max-lipinski-violations", type=int, default=1)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    funnel: list[tuple[str, int]] = []

    # ---- leitura
    df = pd.read_csv(args.input)
    col = args.smiles_col
    if col not in df.columns:
        candidates = [c for c in df.columns if c.strip().lower() == "smiles"]
        if not candidates:
            sys.exit(f"Coluna de SMILES nao encontrada. Colunas: {list(df.columns)}")
        col = candidates[0]

    raw = df[col].dropna().astype(str).tolist()
    funnel.append(("Amostradas", len(raw)))

    # ---- etapa 1: validade e padronizacao
    tools = build_standardizers()
    std = [standardize(s, tools) for s in raw]
    valid = [s for s in std if s is not None]
    funnel.append(("Validas apos padronizacao", len(valid)))

    # ---- etapa 2: deduplicacao
    seen: dict[str, str] = {}
    for s in valid:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        key = inchi.MolToInchiKey(m)
        if key:
            seen.setdefault(key, s)
    unique = sorted(seen.values())
    funnel.append(("Unicas", len(unique)))

    work = pd.DataFrame({"SMILES": unique})

    # ---- etapa 3: novidade vs BraCoLi
    ref = [l.split()[0] for l in args.reference.read_text().splitlines() if l.strip()]
    ref_canon = set(ref)
    work["in_bracoli"] = work["SMILES"].isin(ref_canon)
    work = work[~work["in_bracoli"]].drop(columns=["in_bracoli"]).reset_index(drop=True)
    funnel.append(("Nao presentes no BraCoLi", len(work)))

    work["max_tanimoto_bracoli"] = max_similarity(work["SMILES"].tolist(), ref)
    work.to_csv(args.outdir / f"{args.label}_01_unique_with_similarity.csv", index=False)

    novel = work[work["max_tanimoto_bracoli"] <= args.max_similarity].reset_index(drop=True)
    funnel.append((f"Novas (Tanimoto <= {args.max_similarity})", len(novel)))

    # ---- descritores (necessarios para as etapas 4 e 5)
    desc_rows = []
    keep_idx = []
    for i, s in enumerate(novel["SMILES"]):
        d = describe(s)
        if d is not None:
            desc_rows.append(d)
            keep_idx.append(i)
    novel = novel.iloc[keep_idx].reset_index(drop=True)
    novel = pd.concat([novel, pd.DataFrame(desc_rows)], axis=1)
    novel.to_csv(args.outdir / f"{args.label}_02_novel_described.csv", index=False)

    # ---- etapa 4: sintetizabilidade
    synth = novel[novel["SA_score"] <= args.max_sa].reset_index(drop=True)
    funnel.append((f"Sintetizaveis (SA <= {args.max_sa})", len(synth)))
    synth.to_csv(args.outdir / f"{args.label}_03_synthesizable.csv", index=False)

    # ---- etapa 5: drug-likeness
    druglike = synth[
        (synth["TPSA"] <= args.max_tpsa)
        & (synth["lipinski_violations"] <= args.max_lipinski_violations)
    ].reset_index(drop=True)
    funnel.append((f"Drug-like (TPSA <= {args.max_tpsa}, Lipinski)", len(druglike)))

    final_path = args.outdir / f"{args.label}_04_druglike.csv"
    druglike.to_csv(final_path, index=False)

    # Arquivo .smi pronto para o calculo de fingerprints PaDEL na etapa seguinte.
    smi_path = args.outdir / f"{args.label}_04_druglike.smi"
    with smi_path.open("w", encoding="utf-8") as fh:
        for i, s in enumerate(druglike["SMILES"], start=1):
            fh.write(f"{s}\t{args.label}_{i:05d}\n")

    # ---- funil
    funnel_df = pd.DataFrame(funnel, columns=["Etapa", "N"])
    funnel_df["% da amostragem"] = (funnel_df["N"] / max(funnel[0][1], 1) * 100).round(2)
    funnel_path = args.outdir / f"{args.label}_funnel.csv"
    funnel_df.to_csv(funnel_path, index=False)

    print(f"\n=== Funil de filtragem [{args.label}] ===")
    print(funnel_df.to_string(index=False))
    print(f"\nFinal: {final_path}")
    print(f"SMI:   {smi_path}")


if __name__ == "__main__":
    main()
