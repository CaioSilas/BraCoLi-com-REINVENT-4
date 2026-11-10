#!/usr/bin/env bash
# ===========================================================================
# Replicatas de amostragem: prior puro vs agente pos-transfer-learning
#
# DESENHO EXPERIMENTAL
#   Varia apenas a semente de amostragem. O checkpoint do agente e fixo
#   (escolhido pela curva de validacao no TensorBoard) e o prior e o mesmo
#   arquivo em todas as replicatas. Portanto a variancia medida aqui e a
#   variancia de AMOSTRAGEM do gerador, nao a de treino.
#
#   Isso e o controle minimo para afirmar que a diferenca entre os dois
#   funis nao e ruido: se o intervalo de uma metrica no prior nao encosta
#   no do agente ao longo de N replicatas independentes, a diferenca e do
#   modelo. Variancia de treino exigiria refazer o TL por replicata, com
#   criterio automatico de parada - e outro experimento.
#
#   As sementes sao 1..N e ficam registradas nos .toml e .json de cada
#   replicata, de modo que qualquer uma possa ser reproduzida isolada.
#
# SAIDA (uma pasta por replicata, nada e sobrescrito entre elas)
#   results/replicates/rNN/sampled_{prior,agent}.csv   moleculas geradas
#   results/replicates/rNN/sampling_{prior,agent}.log  log do REINVENT
#   results/replicates/rNN/sampling_{prior,agent}.json config efetiva
#   results/replicates/rNN/filtered/                   funil completo
#   results/replicates/configs/                        .toml de cada run
#
# EXECUTAR (Git Bash, a partir da pasta Artigo/)
#   bash run_replicates.sh
#
# E retomavel: replicatas que ja terminaram sao puladas. Para refazer uma,
# apague a pasta rNN correspondente.
#
# Parametros por variavel de ambiente:
#   NREP=10 NSMILES=10000 AGENT_CKPT=models/bracoli_agent.chkpt.85.chkpt \
#       bash run_replicates.sh
# ===========================================================================

set -euo pipefail

NREP="${NREP:-10}"
NSMILES="${NSMILES:-10000}"
AGENT_CKPT="${AGENT_CKPT:-models/bracoli_agent.chkpt.85.chkpt}"
PRIOR_MODEL="${PRIOR_MODEL:-.reinvent}"
REFERENCE="${REFERENCE:-data/bracoli_clean.smi}"
DEVICE="${DEVICE:-cpu}"
MAX_SIMILARITY="${MAX_SIMILARITY:-0.60}"
MAX_SA="${MAX_SA:-4.5}"

ROOT="results/replicates"
CONFIGS="${ROOT}/configs"

# --no-TTY: sem isso o docker compose injeta codigos de terminal no log.
REINVENT="docker compose run --rm --no-TTY reinvent"
TOOLS="docker compose run --rm --no-TTY tools"

mkdir -p "${CONFIGS}"

if [[ ! -f "${AGENT_CKPT}" ]]; then
    echo "ERRO: checkpoint do agente nao encontrado: ${AGENT_CKPT}" >&2
    echo "Escolha a epoca pela curva de validacao e passe AGENT_CKPT=..." >&2
    exit 1
fi
if [[ ! -f "${REFERENCE}" ]]; then
    echo "ERRO: referencia do BraCoLi nao encontrada: ${REFERENCE}" >&2
    echo "Rode antes: docker compose run --rm tools python prep_bracoli.py ..." >&2
    exit 1
fi

echo "==========================================================="
echo " Replicatas de amostragem"
echo "   replicatas por modelo : ${NREP}"
echo "   SMILES por replicata  : ${NSMILES}"
echo "   prior                 : ${PRIOR_MODEL}"
echo "   agente                : ${AGENT_CKPT}"
echo "   sementes              : 1..${NREP}"
echo "==========================================================="

t0=$(date +%s)

for ((r = 1; r <= NREP; r++)); do
    rep=$(printf "r%02d" "${r}")
    dir="${ROOT}/${rep}"
    mkdir -p "${dir}/filtered"

    for label in prior agent; do
        if [[ "${label}" == "prior" ]]; then
            model="${PRIOR_MODEL}"
        else
            model="${AGENT_CKPT}"
        fi

        funnel="${dir}/filtered/${label}_funnel.csv"
        if [[ -f "${funnel}" ]]; then
            echo "[${rep}/${label}] ja concluida, pulando"
            continue
        fi

        toml="${CONFIGS}/sampling_${label}_${rep}.toml"
        csv="${dir}/sampled_${label}.csv"

        # A semente vai no .toml (nao apenas na CLI) para que o arquivo
        # sozinho reproduza a replicata.
        cat > "${toml}" <<EOF
# Gerado por run_replicates.sh - replicata ${rep}, modelo ${label}
run_type = "sampling"
device = "${DEVICE}"
seed = ${r}
json_out_config = "${dir}/sampling_${label}.json"

[parameters]
model_file = "${model}"
output_file = "${csv}"
num_smiles = ${NSMILES}
unique_molecules = false
randomize_smiles = false
EOF

        echo
        echo "--- [${rep}/${label}] amostragem (semente ${r})"
        ${REINVENT} reinvent \
            -l "${dir}/sampling_${label}.log" \
            -s "${r}" \
            "${toml}"

        if [[ ! -f "${csv}" ]]; then
            echo "ERRO: amostragem nao produziu ${csv}" >&2
            exit 1
        fi

        echo "--- [${rep}/${label}] filtragem"
        ${TOOLS} python filter_generated.py "${csv}" \
            --reference "${REFERENCE}" \
            --label "${label}" \
            --outdir "${dir}/filtered" \
            --max-similarity "${MAX_SIMILARITY}" \
            --max-sa "${MAX_SA}"
    done

    el=$(( $(date +%s) - t0 ))
    echo
    echo "=== ${rep} concluida (${el}s acumulados) ==="
done

echo
echo "==========================================================="
echo " ${NREP} replicatas concluidas em $(( $(date +%s) - t0 ))s"
echo
echo " Analise comparativa:"
echo "   docker compose run --rm tools python compare_replicates.py"
echo "==========================================================="
