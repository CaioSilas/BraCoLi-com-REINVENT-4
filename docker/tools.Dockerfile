# ---------------------------------------------------------------------------
# Imagem 2 de 2: pos-processamento
#   prep_bracoli.py / filter_generated.py / predict_activity.py
#
# Base pip em vez de conda.
#
# A versao anterior (conda-forge) falhava com:
#   error    libmamba Cannot find a valid extracted directory cache for
#                     'typing_extensions-4.16.0-pyhcf101f3_0.conda'
#   critical libmamba Package cache error.
#
# NAO era conflito de dependencias: o solver resolvia em 0,4 s e instalava
# numpy 1.26.4 com scikit-learn 1.3.2 sem reclamar. A causa era
# `mamba install -n base python=3.11`: a imagem miniforge3 traz Python 3.14
# no ambiente `base`, que e onde o proprio mamba roda. Forcar 3.11 ali obriga
# a transacao a religar toda a stack do conda (pydantic, requests,
# typing_extensions, libmambapy...), e no meio da troca o gerenciador perde
# acesso ao proprio cache e aborta.
#
# Com pip o problema nao existe: nao ha gerenciador de pacotes se
# reconstruindo durante a instalacao. As rodas oficiais do RDKit no PyPI
# tambem trazem o diretorio Contrib, de onde vem o sascorer (SA score).
#
# Se precisar de conda, use docker/tools-conda.Dockerfile, que ja foi
# corrigido para criar um ambiente separado em vez de mexer no `base`.
# ---------------------------------------------------------------------------

FROM python:3.11-slim-bookworm

# ATENCAO - compatibilidade dos modelos da Gabriella:
# um RandomForest salvo com pickle so carrega de forma confiavel na mesma
# versao menor de scikit-learn em que foi treinado. Versao diferente pode
# falhar ao desempacotar ou, pior, carregar e produzir predicoes erradas
# em silencio.
#
# O padrao abaixo e SEM PIN. Assim que ela informar a versao, troque por
# exemplo para:
#     SKLEARN_SPEC="scikit-learn==1.3.2"
# e reconstrua com:
#     docker compose build tools
#
# O log de build do conda confirmou que 1.3.2 convive bem com o RDKit atual
# (ele resolveu para numpy 1.26.4), entao fixar essa versao e seguro.
# O pip baixa a combinacao correta sozinho, sem pin manual de numpy.
ARG SKLEARN_SPEC="scikit-learn"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# default-jre-headless: o PaDEL-Descriptor e um programa Java; sem JRE o
# padelpy falha em tempo de execucao, no meio da etapa 7.
# libxrender1/libxext6: bibliotecas de sistema que as rodas do RDKit esperam.
RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jre-headless \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Etapa separada do scikit-learn, de proposito: se o pin de versao for
# impossivel, o build falha AQUI, com mensagem clara, e nao no meio de uma
# instalacao de dez pacotes.
RUN pip install --no-cache-dir \
        rdkit \
        pandas \
        scipy \
        openpyxl \
        matplotlib \
        padelpy

RUN pip install --no-cache-dir "${SKLEARN_SPEC}"

# Verificacoes em tempo de build. Falhar aqui e muito melhor do que descobrir
# na etapa 4 do pipeline, depois de horas de treino.
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
