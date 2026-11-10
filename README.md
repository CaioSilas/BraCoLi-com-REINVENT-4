# Expansão do espaço químico do BraCoLi por *transfer learning* com REINVENT 4

Geração *de novo* de moléculas a partir de um prior treinado em ChEMBL,
redirecionado por *transfer learning* para a química da Brazilian Compound
Library (BraCoLi), com avaliação crítica da transferibilidade de modelos QSAR
publicados às moléculas geradas.

---

## Objetivo

O BraCoLi reúne 1176 compostos sintetizados e caracterizados por grupos de
pesquisa brasileiros. É uma biblioteca pequena para os padrões de triagem
virtual. A pergunta deste trabalho é se um modelo generativo pode **expandir**
esse espaço químico: produzir moléculas inéditas que preservem as
características estruturais da coleção original sem serem cópias dela.

O método é o *transfer learning* do REINVENT 4: parte-se de um prior
(RNN/GRU treinada em ~1-2 milhões de SMILES do ChEMBL) e continua-se o
treinamento sobre o BraCoLi, deslocando a distribuição de probabilidade do
gerador para a região de interesse.

A segunda pergunta, derivada da primeira, é metodológica: **modelos QSAR
treinados em bibliotecas de compostos reais transferem para moléculas
geradas?** Modelos de florestas aleatórias não extrapolam e não sinalizam
quando estão extrapolando. Avaliar isso explicitamente, com critério de
domínio de aplicabilidade declarado, é parte do resultado.

---

## Pipeline

| # | Etapa | Script / configuração | Produto |
|---|---|---|---|
| 1 | Curadoria do BraCoLi | `prep_bracoli.py` | `data/bracoli_{train,valid,clean}.smi` |
| 2 | **Baseline**: amostragem do prior puro | `sampling_baseline.toml` | `results/sampled_prior.csv` |
| 3 | Transfer learning | `tl_bracoli.toml` | `models/bracoli_agent.chkpt.N.chkpt` |
| 4 | Amostragem do agente | `sampling_agent.toml` | `results/sampled_agent.csv` |
| 5 | Filtragem em cascata | `filter_generated.py` | `results/filtered/*_funnel.csv` |
| 6 | Domínio de aplicabilidade | `predict_activity.py` | diagnóstico por molécula |
| 7 | Predição de atividade | `predict_activity.py` | consenso dos três modelos |
| 8 | **Replicatas e comparação** | `run_replicates.sh`, `compare_replicates.py` | `results/replicates/` |

A etapa 2 não é opcional. Toda alegação de "expansão do espaço químico" é uma
comparação, e sem um conjunto gerado pelo mesmo gerador, no mesmo N, sem
exposição ao BraCoLi, não há como separar o efeito do *transfer learning* do
que o prior já fazia sozinho. Os dois funis lado a lado são a primeira tabela
de resultados.

### Cascata de filtragem (etapa 5)

1. Validade química, padronização e canonicalização
2. Deduplicação por InChIKey
3. **Novidade**: remoção de moléculas idênticas ao BraCoLi e de quase-cópias
   (Tanimoto ECFP4 acima do limiar)
4. **Sintetizabilidade**: SA score (Ertl & Schuffenhauer)
5. ***Drug-likeness***: Lipinski + TPSA ≤ 140 Å²

A sintetizabilidade vem antes da atividade de propósito: filtra-se o que não
se consegue fazer antes de gastar tempo predizendo o que seria bom se
existisse.

### Replicatas e teste de hipótese (etapa 8)

Uma única amostragem de cada modelo não sustenta a comparação: não se sabe
quanto da diferença entre os dois funis é o *transfer learning* e quanto é
ruído da amostragem multinomial. A etapa 8 repete a amostragem **N = 10 vezes
por modelo**, com sementes 1 a 10, e trata cada replicata como uma observação.

**O que varia.** Apenas a semente de amostragem. O *checkpoint* do agente e o
arquivo do prior são fixos, logo a variância medida é a de **amostragem do
gerador**, não a de treino. Medir a variância de treino exigiria refazer o TL
por replicata com um critério automático de parada — é outro experimento, e
está declarado como tal.

**Unidade experimental: a replicata, não a molécula.** Uma métrica por
replicata, dez por grupo. Comparar as 10.000 moléculas de uma execução contra
as 10.000 de outra seria pseudo-replicação: moléculas de uma mesma amostragem
não são observações independentes do modelo, e o p-valor resultante não teria
significado.

**Teste.** Mann-Whitney bilateral para amostras independentes — os grupos não
são pareados, já que a semente *r* não cria correspondência entre modelos
distintos —, com correção de Holm-Bonferroni sobre a família de métricas.
Tamanho de efeito pelo delta de Cliff. Com n = 10 contra n = 10 o menor p
bilateral alcançável é 1,08 × 10⁻⁵: métricas que saturam esse piso indicam
separação completa entre os grupos, e nesses casos o p não mede a força do
efeito.

**Em que etapa do funil os descritores são medidos.** Sobre o conjunto único e
válido (etapa 2), antes dos filtros de novidade, SA score e *drug-likeness*.
Medir depois deles compararia conjuntos selecionados por critérios que atingem
os dois modelos de forma desigual, e a diferença observada seria efeito do
filtro, não do *transfer learning*.

---

## Status

| Etapa | Situação |
|---|---|
| 1. Curadoria | **Concluída** — 982 moléculas de 1176 |
| 2. Baseline | **Concluída** |
| 3. Transfer learning | **Concluída** — 100 épocas; checkpoint 85 escolhido pela curva de validação |
| 4. Amostragem do agente | **Concluída** |
| 5. Filtragem | **Concluída** |
| 6-7. Domínio e atividade | Implementadas, **bloqueadas** por material de terceiros |
| 8. Replicatas e comparação | **Concluída** — 10 replicatas por modelo |

**Resultado da curadoria (etapa 1):**

| | |
|---|---|
| Estruturas lidas do SDF | 1176 |
| Descartadas na padronização | 28 |
| Duplicatas removidas | 166 |
| **Retidas** | **982** |
| Treino / validação | 884 / 98 |
| Comprimento do SMILES | mediana 40, máximo 124 |

O bloqueio das etapas 6-7 depende de material cedido pela autora da
dissertação de referência: conjunto de treino dos modelos, os três `.pkl` de
MRSA, os XML de *fingerprint* do PaDEL, e as versões exatas de scikit-learn e
PaDEL-Descriptor usadas no treino. **As etapas 1 a 5 e a análise comparativa
de espaço químico independem disso** e constituem o núcleo da contribuição.

---

## Execução

### Pré-requisitos

- Docker Desktop
- ~10 GB livres (imagens + priors)
- GPU NVIDIA é opcional; o padrão roda em CPU

### Montagem

Este repositório **não contém** priors, dados do BraCoLi nem modelos de
terceiros (ver `.gitignore`). Antes de rodar:

**1. Dados do BraCoLi.** Baixe de
<https://www.farmacia.ufmg.br/qf/downloads/> e coloque o `BraCoLi_v1.sdf` em
`BraCoLi_v1/`.

**2. Prior do REINVENT.** O `reinvent.prior` (treinado em ChEMBL) foi removido
tanto da árvore de trabalho do repositório quanto da versão atual do registro
no Zenodo. Ele permanece no histórico do git, na tag `v4.5.11`:

```
docker compose run --rm reinvent bash -c "git clone --depth 1 --branch v4.5.11 --filter=blob:none --no-checkout https://github.com/MolecularAI/REINVENT4.git /tmp/p && git -C /tmp/p checkout HEAD -- priors/reinvent.prior && cp /tmp/p/priors/reinvent.prior /work/priors/ && sha512sum /work/priors/reinvent.prior"
```

Confira o SHA-512 impresso contra o `sha512.sum` do próprio repositório:

```
3ce451f01e19d87ca5a80a83b4d3dbc1f715d63a88af05441d39f95f1dbc935a90ba251dc92fca29b1346e63fe63737bd627dce7d16bfe4f78f2a28e4b7c0fb0
```

**3. Imagens.**

```
docker compose build
```

### Rodando (cmd do Windows, uma linha por comando)

```
docker compose run --rm tools python prep_bracoli.py BraCoLi_v1/BraCoLi_v1.sdf
```

```
docker compose run --rm reinvent reinvent -l sampling_baseline.log sampling_baseline.toml
```

```
docker compose run --rm reinvent reinvent -l tl_bracoli.log tl_bracoli.toml
```

Acompanhe a curva de validação:

```
docker compose --profile monitor up -d tensorboard
```

e abra <http://localhost:6006>. Escolha a época **anterior** àquela em que a
*loss* de validação parar de cair, edite `model_file` em
`sampling_agent.toml`, e siga:

```
docker compose run --rm reinvent reinvent -l sampling_agent.log sampling_agent.toml
```

```
docker compose run --rm tools python filter_generated.py results/sampled_agent.csv --reference data/bracoli_clean.smi --label agent --outdir results/filtered
```

```
docker compose run --rm tools python filter_generated.py results/sampled_prior.csv --reference data/bracoli_clean.smi --label prior --outdir results/filtered
```

### Replicatas (etapa 8)

O orquestrador amostra e filtra os dois modelos dez vezes, com sementes 1 a 10,
guardando os dados de cada replicata em sua própria pasta. Roda no **Git Bash**
(não no `cmd`), a partir de `Artigo/`:

```
bash run_replicates.sh
```

É retomável: replicatas já concluídas são puladas, então uma interrupção não
custa o que já rodou. Para refazer uma, apague a pasta `results/replicates/rNN`
correspondente. Parâmetros por variável de ambiente:

```
NREP=10 NSMILES=10000 AGENT_CKPT=models/bracoli_agent.chkpt.85.chkpt bash run_replicates.sh
```

A análise comparativa — tabelas, testes de hipótese e figuras:

```
docker compose run --rm tools python compare_replicates.py
```

### Dados disponíveis no repositório

Ao contrário do resto das saídas de execução, `results/` **é versionado**: os
dados que sustentam as tabelas e figuras ficam disponíveis sem exigir que o
leitor rode o pipeline inteiro.

| Caminho | Conteúdo |
|---|---|
| `results/replicates/RELATORIO.md` | desenho, funil médio e tabela completa de testes |
| `results/replicates/comparacao_prior_vs_agente.csv` | médias, DP, CV, U, p bruto e corrigido, Welch, Cliff, Hedges |
| `results/replicates/metricas_por_replicata.csv` | uma linha por replicata e modelo |
| `results/replicates/funil_medio.csv` | funil médio em N absoluto e percentual |
| `results/replicates/figuras/` | três figuras em PNG 300 dpi e PDF vetorial |
| `results/replicates/rNN/` | dados brutos de cada replicata: SMILES gerados, log, config com a semente, funil, descritores e métricas |

São moléculas **geradas** e escores de similaridade. Nenhuma estrutura do
BraCoLi é redistribuída — a regra de não versionar material de terceiros
continua valendo para `data/`, `models/`, `padel/` e os `.pkl`.

---

## Arquitetura dos contêineres

Dois ambientes isolados sobre o mesmo volume montado em `/work`:

| Serviço | Base | Função |
|---|---|---|
| `reinvent` | `python:3.12-slim` + REINVENT 4.8 | *transfer learning*, amostragem |
| `tools` | `python:3.11-slim` + RDKit, PaDEL, scikit-learn | curadoria, filtragem, predição |
| `tensorboard` | mesma imagem do `reinvent` | monitoramento do TL |

A separação é necessária: o REINVENT fixa suas dependências com *lock file*,
e instalar RDKit/PaDEL/scikit-learn no mesmo ambiente quebra essa resolução.

### Correções aplicadas sobre a instalação padrão

Documentadas nos comentários dos respectivos `Dockerfile`:

- **`scipy` ausente do `pyproject.toml` do REINVENT 4.8.** O arquivo
  `runmodes/utils/plot.py` o importa no topo do módulo, sem proteção, e o
  módulo é carregado antes de qualquer configuração ser lida. Instalar com
  `-d all` não resolve, pois `scipy` não pertence a nenhum extra.
- **Bibliotecas X11.** O REINVENT importa `rdkit.Chem.Draw` na partida, e esse
  módulo é ligado ao `libXrender`, mesmo sem interface gráfica.
- **Imagem de pós-processamento em pip, não conda.** Instalar no ambiente
  `base` do miniforge com uma versão diferente de Python derruba o próprio
  gerenciador no meio da transação.

---

## Decisões metodológicas em aberto

Pontos que merecem discussão antes de consolidar os resultados.

**1. Prior: ChEMBL ou PubChem?** O registro atual do Zenodo oferece apenas
`reinvent_pubchem.prior`. O `reinvent.prior` (ChEMBL), que é o descrito no
artigo do REINVENT 4, só existe no histórico do git. A escolha adotada aqui é
o de ChEMBL, por coerência com a literatura citada, mas a alternativa é
defensável e precisa ser declarada corretamente nos métodos.

**2. Os 166 duplicados.** A deduplicação ocorre **após** a remoção de
estereoquímica, então pares de estereoisômeros colapsam num registro único.
Isso é correto para este gerador (o prior Reinvent não modela estereoquímica
e não conseguiria distingui-los), mas precisa estar escrito, senão a diferença
entre 1176 e 982 parece perda inexplicada de dados.

**3. Domínio de aplicabilidade.** Os modelos QSAR foram treinados em ChEMBL e
aplicados ao BraCoLi: dois conjuntos de moléculas reais, com perfil estrutural
semelhante. Moléculas geradas por *transfer learning* tendem a cair fora desse
espaço. O critério adotado combina similaridade de Tanimoto ao conjunto de
treino com um envelope de propriedades. **Se nenhuma molécula gerada cair
dentro do domínio, isso é um resultado publicável**, não uma falha: seria
evidência de que modelos QSAR treinados em bibliotecas reais não transferem
para espaço químico generativo. O erro seria afrouxar o limiar até aparecerem
*hits*.

**4. Pós-processamento, não *reinforcement learning*.** Os filtros são
aplicados depois da geração. A alternativa seria embutir SA score e os modelos
de atividade como componentes de *scoring* num *staged learning*, deixando o
gerador aprender a produzir moléculas sintetizáveis e ativas desde o início.
É mais poderoso, mas agrava o problema do domínio de aplicabilidade: o RL
explora ativamente as fraquezas do modelo de *scoring* para maximizar
recompensa. Fica como trabalho futuro.

**5. Versão do scikit-learn.** Um `RandomForest` salvo com `pickle` só carrega
de forma confiável na mesma versão menor em que foi treinado. Versão diferente
pode falhar ao desempacotar ou, pior, carregar e produzir predições erradas em
silêncio. O `docker-compose.yml` está sem *pin* até que a versão correta seja
confirmada.

---

## Reprodutibilidade

Versões fixadas para registro nos métodos:

| Componente | Versão |
|---|---|
| REINVENT | `v4.8` (tag fixada no `Dockerfile`) |
| Prior | `reinvent.prior`, SHA-512 `3ce451f0...4b7c0fb0` |
| BraCoLi | v1, 1176 compostos |
| RDKit, scikit-learn, NumPy, pandas | impressos ao final do `docker compose build tools` |

---

## Referências

- Loeffler HH, He J, Tibo A, Janet JP, Voronov A, Mervin LH, Engkvist O.
  *Reinvent 4: Modern AI-driven generative molecule design.*
  J Cheminform 16:20 (2024). <https://doi.org/10.1186/s13321-024-00812-5>
- Veríssimo GC *et al.* *The Brazilian compound library (BraCoLi) database.*
  Mol Divers 26:3387-3397 (2022). <https://doi.org/10.1007/s11030-022-10386-9>
- Olivecrona M, Blaschke T, Engkvist O, Chen H. *Molecular de-novo design
  through deep reinforcement learning.* J Cheminform 9:48 (2017).
- Ertl P, Schuffenhauer A. *Estimation of synthetic accessibility score.*
  J Cheminform 1:8 (2009).
- Código do REINVENT 4: <https://github.com/MolecularAI/REINVENT4>
