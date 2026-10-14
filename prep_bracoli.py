#!/usr/bin/env python
"""
Preparacao do conjunto BraCoLi para transfer learning no REINVENT 4.

Aplica a mesma curadoria quimica usada por Campera (2026), de modo que o
conjunto de TL seja compativel com o conjunto de triagem da dissertacao:
  - remocao de sais / maior fragmento
  - remocao de hidrogenios explicitos e de atomos metalicos
  - neutralizacao de cargas
  - tautomero canonico
  - restricao a C, N, O, F, P, S, Cl, Br, I
  - canonicalizacao SMILES e remocao de duplicatas por InChIKey
  - filtro de tokens compativeis com o prior Reinvent (sem estereoquimica)
  - split treino/validacao para monitorar overfitting durante o TL

Formatos de entrada aceitos:
  .sdf   estruturas reais (PREFERIDO - nao depende de SMILES transcrito)
  .mol2  estruturas reais
  .xlsx  planilha, com coluna de SMILES
  .csv   texto delimitado, com coluna de SMILES
  .smi   uma estrutura por linha

Uso (dentro do contentor, a partir de /work):
    python prep_bracoli.py BraCoLi_v1/BraCoLi_v1.sdf
    python prep_bracoli.py BraCoLi_v1/BraCoLi_v1_databse.xlsx --smiles-col SMILES
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

ALLOWED_ATOMS = {"C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}

# Faixa de tamanho aceita pelo prior Reinvent (RNN treinado em ChEMBL drug-like).
MIN_HEAVY_ATOMS = 6
MAX_HEAVY_ATOMS = 70

# O prior Reinvent gera no maximo 256 tokens. Um SMILES mais longo que isso
# seria truncado durante a amostragem e produziria estruturas invalidas, entao
# nao faz sentido ensinar o modelo a gerar algo que ele nao consegue terminar.
MAX_SMILES_CHARS = 250


def build_standardizers():
    """Instancia os objetos de padronizacao uma unica vez (sao caros)."""
    return {
        "lfc": rdMolStandardize.LargestFragmentChooser(),
        "uncharger": rdMolStandardize.Uncharger(),
        "tautomer": rdMolStandardize.TautomerEnumerator(),
    }


def standardize_mol(mol, tools: dict) -> str | None:
    """Padroniza um objeto Mol e devolve o SMILES canonico, ou None se descartado."""
    if mol is None:
        return None

    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = tools["lfc"].choose(mol)          # remove contraions / sais
        mol = tools["uncharger"].uncharge(mol)  # neutraliza cargas
        mol = tools["tautomer"].Canonicalize(mol)
    except Exception:
        return None

    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # Apenas organicas, conforme o criterio da dissertacao.
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ALLOWED_ATOMS:
            return None

    if not (MIN_HEAVY_ATOMS <= mol.GetNumHeavyAtoms() <= MAX_HEAVY_ATOMS):
        return None

    # O prior Reinvent nao modela estereoquimica: remover evita tokens
    # desconhecidos (que derrubam o treino) e duplicatas que diferem apenas
    # por configuracao.
    Chem.RemoveStereochemistry(mol)

    smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

    if len(smi) > MAX_SMILES_CHARS:
        return None

    return smi


def standardize(smiles: str, tools: dict) -> str | None:
    """Versao do padronizador que parte de um SMILES."""
    return standardize_mol(Chem.MolFromSmiles(smiles), tools)


# --------------------------------------------------------------- leitura
def read_structures(path: Path, smiles_col: str | None):
    """
    Devolve (modo, itens).

    modo == "mol"    -> itens sao objetos RDKit Mol
    modo == "smiles" -> itens sao strings SMILES
    """
    suffix = path.suffix.lower()

    # --- SDF: estruturas reais, caminho preferido
    if suffix in {".sdf", ".sd"}:
        supplier = Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True)
        return "mol", list(supplier)

    # --- MOL2: o RDKit le um registro por vez
    if suffix == ".mol2":
        mols = []
        block: list[str] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("@<TRIPOS>MOLECULE") and block:
                mols.append(Chem.MolFromMol2Block("\n".join(block), sanitize=True))
                block = []
            block.append(line)
        if block:
            mols.append(Chem.MolFromMol2Block("\n".join(block), sanitize=True))
        return "mol", mols

    # --- XLSX: planilha
    if suffix in {".xlsx", ".xlsm"}:
        try:
            import pandas as pd
        except ImportError:
            sys.exit("pandas e openpyxl sao necessarios para ler .xlsx")
        df = pd.read_excel(path)
        col = _pick_smiles_column(list(df.columns), smiles_col, path)
        return "smiles", df[col].dropna().astype(str).tolist()

    # --- SMI / TXT sem coluna indicada: primeira palavra de cada linha
    if suffix in {".smi", ".txt"} and smiles_col is None:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "smiles", [l.split()[0] for l in lines if l.strip()]

    # --- CSV / outros textos delimitados
    import csv

    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = list(csv.DictReader(text))
    if not rows:
        return "smiles", []
    col = _pick_smiles_column(list(rows[0].keys()), smiles_col, path)
    return "smiles", [r[col] for r in rows if r.get(col)]


def _pick_smiles_column(columns: list, requested: str | None, path: Path) -> str:
    """Escolhe a coluna de SMILES, com busca tolerante a maiusculas."""
    if requested and requested in columns:
        return requested

    for c in columns:
        if isinstance(c, str) and c.strip().lower() in {"smiles", "canonical_smiles", "smile"}:
            return c

    sys.exit(
        f"Coluna de SMILES nao encontrada em {path.name}.\n"
        f"Colunas disponiveis: {columns}\n"
        f"Informe com --smiles-col."
    )


# --------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Curadoria do BraCoLi para TL no REINVENT 4")
    ap.add_argument("input", type=Path, help="Arquivo .sdf, .mol2, .xlsx, .csv ou .smi")
    ap.add_argument("--smiles-col", default=None, help="Nome da coluna de SMILES")
    ap.add_argument("--val-frac", type=float, default=0.10,
                    help="Fracao destinada a validacao (default: 0.10)")
    ap.add_argument("--seed", type=int, default=5,
                    help="Semente aleatoria (default: 5, mesma da dissertacao)")
    ap.add_argument("--outdir", type=Path, default=Path("data"))
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"Arquivo nao encontrado: {args.input}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    mode, items = read_structures(args.input, args.smiles_col)
    print(f"Formato detectado:           {mode} ({args.input.suffix})")
    print(f"Entradas lidas:              {len(items)}")

    tools = build_standardizers()

    seen: dict[str, str] = {}   # InChIKey -> SMILES canonico
    n_invalid = 0

    for item in items:
        std = standardize_mol(item, tools) if mode == "mol" else standardize(item, tools)

        if std is None:
            n_invalid += 1
            continue

        mol = Chem.MolFromSmiles(std)
        if mol is None:
            n_invalid += 1
            continue

        key = inchi.MolToInchiKey(mol)
        if not key:
            n_invalid += 1
            continue

        seen.setdefault(key, std)

    clean = sorted(seen.values())
    print(f"Descartadas na padronizacao: {n_invalid}")
    print(f"Unicas apos deduplicacao:    {len(clean)}")

    if not clean:
        sys.exit("Nenhuma molecula sobreviveu a curadoria. Verifique o arquivo de entrada.")

    if len(clean) < 100:
        print("\nAVISO: conjunto muito pequeno para transfer learning estavel.")

    # Diagnostico de comprimento: SMILES longos demais sao truncados pelo
    # limite de 256 tokens do prior durante a amostragem.
    lengths = sorted(len(s) for s in clean)
    print(f"\nComprimento do SMILES: mediana {lengths[len(lengths) // 2]}, "
          f"maximo {lengths[-1]} (limite pratico {MAX_SMILES_CHARS})")

    rng = random.Random(args.seed)
    rng.shuffle(clean)

    n_val = max(1, int(round(len(clean) * args.val_frac)))
    val, train = clean[:n_val], clean[n_val:]

    train_path = args.outdir / "bracoli_train.smi"
    val_path = args.outdir / "bracoli_valid.smi"
    all_path = args.outdir / "bracoli_clean.smi"

    train_path.write_text("\n".join(train) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val) + "\n", encoding="utf-8")
    all_path.write_text("\n".join(sorted(clean)) + "\n", encoding="utf-8")

    print(f"\nTreino:    {len(train):5d}  ->  {train_path}")
    print(f"Validacao: {len(val):5d}  ->  {val_path}")
    print(f"Referencia (novidade/Tanimoto): {all_path}")


if __name__ == "__main__":
    main()
