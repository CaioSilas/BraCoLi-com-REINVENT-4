#!/usr/bin/env python
"""
Le os arquivos de evento do TensorBoard gerados pelo transfer learning e
imprime a perda por epoca, identificando o checkpoint a ser usado.

Motivacao: escolher a epoca "no olho" pela interface do TensorBoard e
impreciso, e a curva de validacao com randomize_all_smiles ativo tem minimo
raso - o proprio codigo do REINVENT comenta isso ("if true much shallower
minimum"). Com os numeros na tela a escolha fica objetiva, e o valor vai
para a secao de metodos do artigo.

Roda no contentor do REINVENT, que ja tem o tensorboard instalado:

    docker compose run --rm reinvent python read_tl_curve.py

Opcional, para outro diretorio de log:

    docker compose run --rm reinvent python read_tl_curve.py --logdir tb_tl_bracoli
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit(
        "tensorboard nao encontrado. Rode este script no contentor do REINVENT:\n"
        "    docker compose run --rm reinvent python read_tl_curve.py"
    )


def find_run_dirs(logdir: Path) -> list[Path]:
    """Todo diretorio que contenha ao menos um arquivo de evento."""
    runs = set()
    for path in logdir.rglob("events.out.tfevents.*"):
        runs.add(path.parent)
    return sorted(runs)


def load_run(run_dir: Path) -> dict[str, dict[int, float]]:
    """Devolve {tag: {step: valor}} para os escalares do diretorio."""
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()

    out: dict[str, dict[int, float]] = {}
    for tag in acc.Tags().get("scalars", []):
        out[tag] = {e.step: e.value for e in acc.Scalars(tag)}
    return out


def classify(label: str) -> str | None:
    """Mapeia o nome do run/tag para treino, validacao ou amostragem."""
    low = label.lower()
    if "valid" in low:
        return "validacao"
    if "train" in low:
        return "treino"
    if "sample" in low:
        return "amostragem"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Perda por epoca do transfer learning")
    ap.add_argument("--logdir", type=Path, default=Path("tb_tl_bracoli"))
    args = ap.parse_args()

    if not args.logdir.exists():
        sys.exit(f"Diretorio nao encontrado: {args.logdir}")

    runs = find_run_dirs(args.logdir)
    if not runs:
        sys.exit(f"Nenhum arquivo de evento em {args.logdir}")

    # series[categoria][epoca] = valor
    series: dict[str, dict[int, float]] = {}

    for run_dir in runs:
        rel = run_dir.relative_to(args.logdir)
        scalars = load_run(run_dir)

        for tag, steps in scalars.items():
            # A categoria pode estar no nome do diretorio ou no nome da tag.
            cat = classify(str(rel)) or classify(tag)
            if cat is None:
                continue
            series.setdefault(cat, {}).update(steps)

    if "validacao" not in series:
        print("AVISO: nenhuma serie de validacao encontrada.")
        print("Verifique se validation_smiles_file estava definido no TOML.\n")

    epochs = sorted({e for s in series.values() for e in s})
    if not epochs:
        sys.exit("Nenhum escalar reconhecido nos arquivos de evento.")

    cols = [c for c in ("treino", "validacao", "amostragem") if c in series]

    # ---- tabela
    header = f"{'epoca':>6}" + "".join(f"{c:>14}" for c in cols)
    print(header)
    print("-" * len(header))

    for ep in epochs:
        row = f"{ep:>6}"
        for c in cols:
            v = series[c].get(ep)
            row += f"{v:>14.4f}" if v is not None else f"{'-':>14}"
        print(row)

    # ---- diagnostico
    #
    # Comparar apenas "ultima epoca" com "minimo global" e enganoso: a curva
    # de validacao oscila, e o minimo global costuma cair numa epoca
    # afortunada. Se a oscilacao for maior que a diferenca, o minimo nao e
    # informativo e escolhe-se ruido.
    #
    # O criterio aqui compara a TENDENCIA (media dos ultimos 5 contra a dos
    # 5 anteriores) com o RUIDO (desvio-padrao da cauda), e observa a
    # distancia entre treino e validacao, que e o sinal honesto de
    # sobreajuste.
    if "validacao" not in series or not series["validacao"]:
        return

    val = series["validacao"]
    eps = sorted(val)
    vals = [val[e] for e in eps]

    best_ep = min(val, key=val.get)
    last_ep = eps[-1]

    print()
    print(f"Minimo global:  epoca {best_ep}  (loss {val[best_ep]:.4f})")
    print(f"Ultima epoca:   epoca {last_ep}  (loss {val[last_ep]:.4f})")

    if len(vals) < 10:
        print("\nPoucas epocas para diagnostico de tendencia. Treine mais.")
        return

    tail = vals[-10:]
    media = sum(tail) / len(tail)
    ruido = (sum((v - media) ** 2 for v in tail) / len(tail)) ** 0.5

    recentes = sum(vals[-5:]) / 5
    anteriores = sum(vals[-10:-5]) / 5
    melhora = anteriores - recentes   # positivo = ainda descendo

    print()
    print(f"Ruido da cauda (desvio-padrao das ultimas 10):  {ruido:.4f}")
    print(f"Tendencia (media 5 anteriores - media 5 ultimas): {melhora:+.4f}")

    # Distancia treino-validacao: o indicador honesto de sobreajuste.
    folga = None
    if "treino" in series and last_ep in series["treino"]:
        folga = val[last_ep] - series["treino"][last_ep]
        print(f"Distancia treino-validacao na ultima epoca:    {folga:+.4f}")

    print()

    if melhora > ruido:
        print("VEREDITO: a validacao AINDA ESTA DESCENDO.")
        print("  A melhora recente supera a oscilacao da curva. O treino nao")
        print("  convergiu e o agente esta subtreinado. Aumente num_epochs em")
        print("  tl_bracoli.toml e rode o transfer learning de novo.")
        print()
        print("  Nao escolha checkpoint por enquanto: o minimo global aqui")
        print("  cai dentro do ruido e nao e informativo.")
    elif folga is not None and folga > 3 * ruido and melhora < -ruido:
        print("VEREDITO: SOBREAJUSTE.")
        print("  A validacao subiu acima do ruido e se descolou do treino.")
        print(f'  Use: model_file = "models/bracoli_agent.chkpt.{best_ep}.chkpt"')
    else:
        print("VEREDITO: PLATO.")
        print("  A validacao parou de melhorar mas nao se descolou do treino.")
        print("  Este e o ponto de parada. Entre os checkpoints do plato a")
        print("  diferenca e ruido, entao o minimo global serve como criterio")
        print("  objetivo e reportavel.")
        print()
        print(f'  Use: model_file = "models/bracoli_agent.chkpt.{best_ep}.chkpt"')
        print()
        print("  Confira o nome exato do arquivo:")
        print("      docker compose run --rm reinvent ls models")


if __name__ == "__main__":
    main()
