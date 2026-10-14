# ---------------------------------------------------------------------------
# Imagem 1 de 2: REINVENT 4 (transfer learning + sampling)
#
# Mantida separada da imagem de pos-processamento de proposito: o REINVENT
# fixa suas dependencias com um lock file, e instalar RDKit/PaDEL/scikit-learn
# no mesmo ambiente costuma quebrar essa resolucao.
#
# Versao atual exige Python >= 3.11 (pyproject.toml: requires-python = ">=3.11").
# ---------------------------------------------------------------------------

FROM python:3.12-slim-bookworm

# Tag do repositorio. Fixada para reprodutibilidade: um artigo precisa dizer
# qual versao gerou os resultados. Troque so se tiver motivo.
ARG REINVENT_REF=v4.8

# Alvo do PyTorch: "cpu", "cu126", "rocm6.4", "xpu".
# Para GPU NVIDIA, reconstrua com --build-arg TORCH_TARGET=cu126
ARG TORCH_TARGET=cpu

# Dependencias opcionais do REINVENT:
#   none = so o nucleo (suficiente para TL + sampling)
#   all  = inclui chemprop, openeye etc. (imagem muito maior)
ARG REINVENT_EXTRAS=none

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# git para o clone; build-essential porque algumas dependencias compilam.
#
# As bibliotecas libX* nao sao opcionais, apesar de nao haver interface
# grafica aqui. O REINVENT importa rdkit.Chem.Draw logo no inicio
# (runmodes/utils/image.py, para as grades de moleculas do TensorBoard), e
# esse modulo e ligado ao libXrender. Sem elas o programa nem chega a ler
# o arquivo de configuracao:
#     ImportError: libXrender.so.1: cannot open shared object file
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libxrender1 \
        libxext6 \
        libsm6 \
        libx11-6 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${REINVENT_REF} \
        https://github.com/MolecularAI/REINVENT4.git /opt/REINVENT4

WORKDIR /opt/REINVENT4

# install.py monta a linha de comando do pip com a URL correta do PyTorch
# conforme o tipo de processador.
RUN python install.py ${TORCH_TARGET} -d ${REINVENT_EXTRAS}

# scipy NAO esta declarado no pyproject.toml do REINVENT 4.8 - nem nas
# dependencias obrigatorias, nem em nenhum extra. Mas
# reinvent/runmodes/utils/plot.py faz, na linha 3:
#     from scipy.stats import gaussian_kde
# no topo do modulo, sem protecao. Esse modulo e carregado na importacao de
# runmodes, antes de qualquer arquivo de configuracao ser lido, entao o
# programa morre na partida:
#     ModuleNotFoundError: No module named 'scipy'
#
# E um descuido do empacotamento: matplotlib, logo abaixo no mesmo arquivo,
# esta corretamente protegido pelo decorador @extra_dependencies e importado
# so quando usado. O scipy ficou de fora.
#
# Instalar "-d all" NAO resolve, porque scipy nao pertence a nenhum extra.
RUN pip install "scipy>=1.11,<2"

# Verificacao de partida: se faltar mais alguma dependencia nao declarada,
# a imagem falha aqui, no build, e nao no inicio de um treino de horas.
RUN reinvent --help > /dev/null && echo "CLI do REINVENT responde"

# O registro de priors (reinvent/utils/prior_registry.py) le esta variavel.
# Com ela, os TOML podem referenciar ".reinvent" em vez de um caminho,
# e os arquivos .prior ficam montados de fora da imagem.
ENV REINVENT_PRIOR_BASE=/work/priors

# Componentes de scoring da pasta contrib/ (so usados em staged learning).
ENV PYTHONPATH=/opt/REINVENT4/contrib

# Diretorio de trabalho e o projeto montado, nao o repositorio.
# Os caminhos relativos dos seus TOML resolvem a partir daqui.
WORKDIR /work

# Sem CMD: os comandos vem via `docker compose run`.
