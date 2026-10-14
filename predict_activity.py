#!/usr/bin/env python
"""
Etapas 6 e 7: dominio de aplicabilidade + predicao de atividade antibacteriana
com os modelos de florestas aleatorias de Campera (2026).

ORDEM INEGOCIAVEL: dominio de aplicabilidade PRIMEIRO, predicao DEPOIS.

Motivo: um Random Forest nao extrapola e nao sinaliza que esta extrapolando.
Para uma molecula fora do espaco quimico de treino ele devolve uma
probabilidade aparentemente confiante, produzida por folhas ajustadas a
outra regiao do espaco. Como as moleculas aqui foram geradas por um modelo
generativo, elas tendem justamente a cair fora desse espaco. Predicao fora
do dominio nao e "molecula inativa" nem "molecula ativa": e predicao sem
validade, e deve ser descartada, nao reinterpretada.

Criterio de dominio adotado (reportavel no artigo):
  (a) similaridade: Tanimoto ECFP4 maximo contra o conjunto de TREINO dos
      modelos >= limiar (default 0.30); e
  (b) envelope de propriedades: dentro do intervalo observado no treino para
      os 6 descritores usados na PCA da dissertacao (MW, TPSA, HBD, HBA,
      RotB, Rings), com margem configuravel.

Consenso de atividade: media das probabilidades dos tres modelos, com o
mesmo limiar da dissertacao para a cepa correspondente.

PRE-REQUISITOS
  - Os tres modelos .pkl cedidos pela autora, treinados com os
    hiperparametros da Tabela 3 da dissertacao.
  - Os fingerprints PaDEL correspondentes, calculados com o MESMO software
    e versao usados no treino (padelpy / PaDEL-Descriptor).
  - O conjunto de treino dela em .smi, para definir o dominio.

Uso:
    python predict_activity.py \\
        --smiles results/filtered/agent_04_druglike.smi \\
        --train-smiles data/mrsa_train.smi \\
        --models models_mrsa/model_config.json \\
        --consensus-threshold 0.72 \\
        --out results/agent_mrsa_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Os 6 descritores da PCA de espaco quimico da dissertacao (Figura 22).
PCA_DESCRIPTORS = ["MW", "TPSA", "HBD", "HBA", "RotB", "Rings"]

# Fingerprints PaDEL por cepa, conforme Tabela 3.
PADEL_BY_STRAIN = {
    "MRSA": ["KlekotaRothCount", "AtomPairs2DCount", "SubstructureCount"],
    "MSSA": ["KlekotaRothCount", "SubstructureCount", "PubChem"],
    "ECOLI": ["KlekotaRothCount", "AtomPairs2DCount", "SubstructureCount"],
}

# Limiares de consenso da dissertacao.
CONSENSUS_BY_STRAIN = {"MRSA": 0.72, "MSSA": 0.50, "ECOLI": 0.50}


# ---------------------------------------------------------------- utilidades
def read_smi(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(line.split()[0])
    return out


def descriptor_frame(smiles: list[str]) -> pd.DataFrame:
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            rows.append({d: np.nan for d in PCA_DESCRIPTORS})
            continue
        rows.append({
            "MW": Descriptors.MolWt(m),
            "TPSA": Descriptors.TPSA(m),
            "HBD": Descriptors.NumHDonors(m),
            "HBA": Descriptors.NumHAcceptors(m),
            "RotB": Descriptors.NumRotatableBonds(m),
            "Rings": Descriptors.RingCount(m),
        })
    return pd.DataFrame(rows)


def max_tanimoto(query: list[str], reference: list[str]) -> np.ndarray:
    ref_fps = []
    for s in reference:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            ref_fps.append(MORGAN.GetFingerprint(m))
    if not ref_fps:
        sys.exit("Conjunto de treino vazio ou invalido.")

    out = np.zeros(len(query))
    for i, s in enumerate(query):
        m = Chem.MolFromSmiles(s)
        if m is None:
            out[i] = 0.0
            continue
        sims = DataStructs.BulkTanimotoSimilarity(MORGAN.GetFingerprint(m), ref_fps)
        out[i] = max(sims)
    return out


# ---------------------------------------------------------------- dominio
def applicability_domain(
    query_smiles: list[str],
    train_smiles: list[str],
    sim_threshold: float,
    margin: float,
) -> pd.DataFrame:
    """Retorna DataFrame com as colunas de diagnostico do dominio."""
    sim = max_tanimoto(query_smiles, train_smiles)

    q_desc = descriptor_frame(query_smiles)
    t_desc = descriptor_frame(train_smiles)

    lo = t_desc.quantile(0.01)
    hi = t_desc.quantile(0.99)
    span = (hi - lo).replace(0, 1.0)
    lo_m = lo - margin * span
    hi_m = hi + margin * span

    inside = pd.Series(True, index=q_desc.index)
    for d in PCA_DESCRIPTORS:
        inside &= q_desc[d].between(lo_m[d], hi_m[d])

    out = q_desc.copy()
    out["max_tanimoto_train"] = np.round(sim, 4)
    out["within_similarity"] = sim >= sim_threshold
    out["within_property_envelope"] = inside.values
    out["in_domain"] = out["within_similarity"] & out["within_property_envelope"]
    return out


# ---------------------------------------------------------------- fingerprints
def compute_padel(smiles: list[str], fp_names: list[str], workdir: Path) -> dict[str, pd.DataFrame]:
    """
    Calcula os fingerprints PaDEL via padelpy.

    IMPORTANTE: use a mesma versao de PaDEL-Descriptor empregada no treino
    dos modelos. Versoes diferentes podem alterar a ordem ou a definicao de
    bits, o que invalida silenciosamente a predicao.
    """
    try:
        from padelpy import padeldescriptor
    except ImportError:
        sys.exit("padelpy nao instalado. Rode: pip install padelpy")

    workdir.mkdir(parents=True, exist_ok=True)
    smi_path = workdir / "input.smi"
    with smi_path.open("w", encoding="utf-8") as fh:
        for i, s in enumerate(smiles, start=1):
            fh.write(f"{s}\tmol_{i:05d}\n")

    frames: dict[str, pd.DataFrame] = {}
    for name in fp_names:
        xml = workdir / f"{name}.xml"
        if not xml.exists():
            sys.exit(
                f"Arquivo de descritor PaDEL ausente: {xml}\n"
                "Baixe os XML de fingerprints do PaDEL-Descriptor e coloque em "
                f"{workdir}/ (ex.: KlekotaRothCount.xml)."
            )
        out_csv = workdir / f"{name}.csv"
        padeldescriptor(
            mol_dir=str(smi_path),
            d_file=str(out_csv),
            descriptortypes=str(xml),
            detectaromaticity=True,
            standardizenitro=True,
            standardizetautomers=True,
            removesalt=True,
            fingerprints=True,
            retainorder=True,
            threads=-1,
        )
        df = pd.read_csv(out_csv)
        frames[name] = df.drop(columns=[c for c in df.columns if c.lower() == "name"])
    return frames


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Dominio de aplicabilidade + predicao de atividade")
    ap.add_argument("--smiles", type=Path, required=True,
                    help=".smi das moleculas filtradas (saida de filter_generated.py)")
    ap.add_argument("--train-smiles", type=Path, required=True,
                    help=".smi do conjunto de TREINO dos modelos (ChEMBL curado)")
    ap.add_argument("--models", type=Path, required=True,
                    help="JSON mapeando nome do fingerprint -> caminho do .pkl")
    ap.add_argument("--strain", default="MRSA", choices=sorted(PADEL_BY_STRAIN))
    ap.add_argument("--consensus-threshold", type=float, default=None,
                    help="Sobrescreve o limiar padrao da cepa")
    ap.add_argument("--similarity-threshold", type=float, default=0.30,
                    help="Tanimoto minimo vs treino para considerar no dominio")
    ap.add_argument("--envelope-margin", type=float, default=0.10,
                    help="Margem relativa no envelope de propriedades")
    ap.add_argument("--padel-dir", type=Path, default=Path("padel"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    threshold = args.consensus_threshold
    if threshold is None:
        threshold = CONSENSUS_BY_STRAIN[args.strain]

    query = read_smi(args.smiles)
    train = read_smi(args.train_smiles)
    print(f"Moleculas a avaliar: {len(query)}")
    print(f"Conjunto de treino:  {len(train)}")

    # ---- etapa 6: dominio de aplicabilidade
    dom = applicability_domain(
        query, train, args.similarity_threshold, args.envelope_margin
    )
    n_in = int(dom["in_domain"].sum())
    print(f"\nDentro do dominio: {n_in} / {len(query)} "
          f"({n_in / max(len(query), 1) * 100:.1f}%)")
    print(f"  falham por similaridade:  {int((~dom['within_similarity']).sum())}")
    print(f"  falham por propriedades:  {int((~dom['within_property_envelope']).sum())}")

    if n_in == 0:
        print("\nNenhuma molecula dentro do dominio. Os modelos nao sao aplicaveis\n"
              "a este conjunto gerado. Isso e um resultado, nao uma falha:\n"
              "reporte-o e nao force a predicao.")
        dom.insert(0, "SMILES", query)
        dom.to_csv(args.out, index=False)
        return

    in_idx = dom.index[dom["in_domain"]].tolist()
    in_smiles = [query[i] for i in in_idx]

    # ---- etapa 7: predicao de atividade
    fp_names = PADEL_BY_STRAIN[args.strain]
    model_map = json.loads(args.models.read_text())
    missing = [n for n in fp_names if n not in model_map]
    if missing:
        sys.exit(f"Modelos ausentes no JSON para: {missing}")

    fps = compute_padel(in_smiles, fp_names, args.padel_dir)

    probs = {}
    for name in fp_names:
        with open(model_map[name], "rb") as fh:
            clf = pickle.load(fh)
        X = fps[name]

        # Alinhamento estrito de colunas com o que o modelo viu no treino.
        expected = getattr(clf, "feature_names_in_", None)
        if expected is not None:
            faltando = [c for c in expected if c not in X.columns]
            if faltando:
                sys.exit(
                    f"[{name}] {len(faltando)} colunas esperadas pelo modelo nao "
                    f"foram geradas pelo PaDEL. Verifique a versao do PaDEL."
                )
            X = X[list(expected)]

        probs[name] = clf.predict_proba(X)[:, 1]
        print(f"  {name}: ativas preditas = {int((probs[name] >= 0.5).sum())}")

    pred = pd.DataFrame({"SMILES": in_smiles})
    for name in fp_names:
        pred[f"prob_{name}"] = np.round(probs[name], 4)
    pred["prob_mean"] = np.round(
        np.mean([probs[n] for n in fp_names], axis=0), 4
    )
    pred["active_all_models"] = np.all(
        [probs[n] >= 0.5 for n in fp_names], axis=0
    )
    pred["consensus_hit"] = pred["prob_mean"] >= threshold

    # Reanexa o diagnostico de dominio.
    dom_in = dom.loc[in_idx].reset_index(drop=True)
    pred = pd.concat([pred, dom_in.drop(columns=PCA_DESCRIPTORS)], axis=1)

    pred = pred.sort_values("prob_mean", ascending=False).reset_index(drop=True)
    pred.to_csv(args.out, index=False)

    n_hit = int(pred["consensus_hit"].sum())
    print(f"\nConsenso (media >= {threshold}): {n_hit} moleculas")
    print(f"Saida: {args.out}")

    # Registro completo, incluindo as fora do dominio, para rastreabilidade.
    full = dom.copy()
    full.insert(0, "SMILES", query)
    full_path = args.out.with_name(args.out.stem + "_domain_full.csv")
    full.to_csv(full_path, index=False)
    print(f"Diagnostico de dominio (todas): {full_path}")


if __name__ == "__main__":
    main()
