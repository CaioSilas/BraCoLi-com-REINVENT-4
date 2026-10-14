# ---------------------------------------------------------------------------
# ALTERNATIVA em conda-forge para a imagem de pos-processamento.
#
# Use apenas se a imagem padrao (docker/tools.Dockerfile, baseada em pip)
# nao servir - por exemplo, se as rodas do RDKit no PyPI nao trouxerem o
# diretorio Contrib de onde vem o sascorer.
#
# Para usar esta aqui, troque em docker-compose.yml:
#     dockerfile: docker/tools-conda.Dockerfile
#
# ---------------------------------------------------------------------------
# POR QUE A VERSAO ANTERIOR FALHAVA
#
#   error    libmamba Cannot find a valid extracted directory cache for
#                     'typing_extensions-4.16.0-pyhcf101f3_0.conda'
#   critical libmamba Package cache error.
#
# Nao era conflito de dependencias: o solver resolvia em menos de um segundo
# e instalava numpy 1.26.4 com scikit-learn 1.3.2 sem problema.
#
# O erro vinha de `mamba install -n base python=3.11`. A imagem miniforge3
# traz Python 3.14 no ambiente `base`, que e onde o proprio mamba roda.
# Pedir Python 3.11 ali obriga a transacao a religar toda a stack do conda
# (pydantic, requests, typing_extensions, libmambapy...) para a nova versao.
# No meio dessa troca o mamba perde o acesso ao proprio cache de pacotes e
# aborta - ele desmonta a si mesmo enquanto trabalha.
#
# CORRECAO: criar um ambiente SEPARADO com `mamba create -n chem` e colocar
# esse ambiente no PATH. O `base` fica intacto, com o Python que veio na
# imagem, e o gerenciador continua funcional do inicio ao fim.
# ---------------------------------------------------------------------------

FROM condaforge/miniforge3:latest

ARG PYTHON_VERSION=3.11

# O log de build provou que scikit-learn 1.3.2 convive com o RDKit atual.
# Pode fixar a versao da Gabriella aqui, ex.: "scikit-learn=1.3.2"
ARG SKLEARN_SPEC="scikit-learn"

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# Ambiente proprio, nunca o base. openjdk: o PaDEL-Descriptor e Java.
RUN mamba create -y -n chem -c conda-forge \
        python=${PYTHON_VERSION} \
        rdkit \
        pandas \
        numpy \
        scipy \
        openpyxl \
        matplotlib-base \
        openjdk \
        "${SKLEARN_SPEC}" \
    && mamba clean -afy

# Coloca o ambiente novo no PATH. A partir daqui, `python`, `pip` e `java`
# resolvem para /opt/conda/envs/chem/bin, sem precisar de `conda activate`
# (que nao funciona de forma confiavel em RUN/CMD de Dockerfile).
ENV PATH=/opt/conda/envs/chem/bin:${PATH} \
    CONDA_DEFAULT_ENV=chem

# padelpy nao esta no conda-forge; vem do PyPI, para dentro do env `chem`.
RUN pip install --no-cache-dir padelpy

# Verificacoes em tempo de build: falhar aqui e melhor do que descobrir
# no meio da filtragem, depois de horas de treino.
RUN python -c "\
import sys, pathlib; \
from rdkit.Chem import RDConfig; \
p = pathlib.Path(RDConfig.RDContribDir) / 'SA_Score'; \
sys.path.append(str(p)); \
import sascorer; \
from rdkit import Chem; \
print('sascorer OK:', round(sascorer.calculateScore(Chem.MolFromSmiles('CCO')), 3))"

RUN java -version

RUN python -c "\
import sklearn, numpy, rdkit, pandas; \
print('sklearn', sklearn.__version__); \
print('numpy  ', numpy.__version__); \
print('rdkit  ', rdkit.__version__); \
print('pandas ', pandas.__version__)"

WORKDIR /work
