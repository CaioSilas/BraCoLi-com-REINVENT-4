# Expansão do espaço químico do BraCoLi por *transfer learning* com REINVENT 4

Geração *de novo* de moléculas a partir de um prior treinado em ChEMBL,
redirecionado por *transfer learning* para a química da Brazilian Compound
Library (BraCoLi), com avaliação estatística do deslocamento obtido.

---

## Objetivo

O BraCoLi reúne 1176 compostos sintetizados e caracterizados por grupos de
pesquisa brasileiros. É uma biblioteca pequena para os padrões de triagem
virtual. A pergunta é se um modelo generativo pode **expandir** esse espaço
químico: produzir moléculas inéditas que preservem as características
estruturais da coleção original sem serem cópias dela.

O método é o *transfer learning* do REINVENT 4: parte-se de um prior
(RNN/GRU treinada em ~1-2 milhões de SMILES do ChEMBL) e continua-se o
treinamento sobre o BraCoLi, deslocando a distribuição de probabilidade do
gerador para a região de interesse.

A segunda pergunta, derivada da primeira, é metodológica: **modelos QSAR
treinados em bibliotecas de compostos reais transferem para moléculas
geradas?** Modelos de florestas aleatórias não extrapolam e não sinalizam
quando estão extrapolando. Avaliar isso com critério de domínio de
aplicabilidade declarado é parte do resultado.

---

## Resultados

### Desenho experimental

Dez replicatas de cada modelo, 10 000 SMILES por replicata, variando apenas a
semente de amostragem. O checkpoint do agente e o arquivo do prior são fixos,
então a variância medida é a da amostragem do gerador, não a do treino. A
unidade experimental é a replicata, não a molécula.

Teste de Mann-Whitney bilateral por permutação exata, com correção de
Holm-Bonferroni sobre a família de 23 métricas.

### Funil comparativo (% das amostradas, média ± DP)

| Etapa | Prior | Agente | Diferença |
|---|---:|---:|---:|
| Válidas | 98,28 ± 0,13 | 92,90 ± 0,44 | −5,4 |
| Únicas | 98,27 ± 0,13 | 86,12 ± 0,56 | −12,1 |
| Fora do BraCoLi | 98,26 ± 0,13 | 83,65 ± 0,58 | −14,6 |
| Novas (Tanimoto ≤ 0,6) | 98,07 ± 0,12 | 65,47 ± 0,41 | **−32,6** |
| Sintetizáveis (SA ≤ 4,5) | 94,22 ± 0,20 | 55,35 ± 0,47 | −38,9 |
| *Drug-like* | 85,36 ± 0,39 | 48,39 ± 0,53 | −37,0 |

As **23 de 23** métricas avaliadas têm separação completa entre os modelos
(delta de Cliff = ±1): nenhuma replicata de um modelo se sobrepõe a nenhuma do
outro.

### O deslocamento, em medidas contínuas

| Métrica | Prior | Agente |
|---|---:|---:|
| Tanimoto máximo médio vs BraCoLi | 0,260 ± 0,001 | 0,468 ± 0,002 |
| Quase-cópias do BraCoLi (% das únicas) | 0,19 ± 0,06 | 21,73 ± 0,38 |
| Scaffolds de Murcko únicos | 8544 ± 33 | 4172 ± 19 |
| Massa molecular média (Da) | 384,1 ± 1,0 | 345,7 ± 1,2 |
| LogP médio | 3,48 ± 0,02 | 1,88 ± 0,03 |
| TPSA média (Å²) | 76,4 ± 0,3 | 95,5 ± 0,6 |
| Número médio de anéis | 3,38 ± 0,01 | 2,32 ± 0,02 |
| SA score médio | 2,83 ± 0,01 | 3,19 ± 0,01 |

O Tanimoto máximo médio é a medida mais limpa do deslocamento, porque é
contínua e não depende de um limiar arbitrário.

O perfil de descritores descreve a direção do deslocamento: o agente gera
moléculas menores, bem mais polares e com um anel a menos, em média, do que o
prior — e mais difíceis de sintetizar.

O colapso de esqueletos de Murcko, de 8544 para 4172, é o custo do foco, e é
maior do que a taxa de duplicatas internas sugeriria.

### Convergência do transfer learning

Cem épocas. A perda de validação atingiu platô por volta da época 85 (mínimo
22,51 na época 86), com tendência residual de 0,39 — inferior ao
desvio-padrão de 0,48 das últimas dez épocas. A distância entre as perdas de
treino e validação permaneceu em 1,12, sem indício de sobreajuste. O
checkpoint da época 85 foi o selecionado.

Uma primeira rodada de 30 épocas foi descartada: a validação ainda descia, e
o critério ingênuo de parada (última época acima do mínimo global) é
enganoso quando a diferença cai dentro do ruído da curva.

### Curadoria do BraCoLi

| | |
|---|---:|
| Estruturas lidas do SDF | 1176 |
| Descartadas na padronização | 28 |
| Duplicatas removidas | 166 |
| **Retidas** | **982** |
| Treino / validação | 884 / 98 |

A deduplicação ocorre **após** a remoção de estereoquímica, então pares de
estereoisômeros colapsam num registro único. Isso é correto para este gerador:
o prior Reinvent não modela configuração espacial.

---

## Estado do trabalho

| Etapa | Situação |
|---|---|
| 1. Curadoria | Concluída — 982 de 1176 |
| 2. Baseline (prior puro) | Concluída — 10 replicatas |
| 3. Transfer learning | Concluída — 100 épocas, checkpoint 85 |
| 4. Amostragem do agente | Concluída — 10 replicatas |
| 5. Filtragem em cascata | Concluída |
| 6. Análise estatística | Concluída — ver `analise/RELATORIO.md` |
| 7. Domínio de aplicabilidade e atividade | **Bloqueada** — material de terceiros |

As etapas 6 e 7 do pipeline original dependem de material cedido pela autora
da dissertação de referência: conjunto de treino dos modelos, os três `.pkl`
de MRSA, os XML de *fingerprint* do PaDEL, e as versões exatas de
scikit-learn e PaDEL-Descriptor usadas no treino.

---

## Estrutura

```
.
├── docker/                 imagens: REINVENT e pós-processamento
├── analise/                RESULTADOS — relatório, tabelas e figuras
│   ├── RELATORIO.md
│   ├── metricas_por_replicata.csv
│   ├── comparacao_prior_vs_agente.csv
│   ├── funil_medio.csv
│   └── figuras/
├── prep_bracoli.py         curadoria do SDF
├── filter_generated.py     cascata de filtros
├── predict_activity.py     domínio de aplicabilidade + QSAR (bloqueado)
├── read_tl_curve.py        leitura da curva de validação
├── tl_bracoli.toml         configuração do transfer learning
├── sampling_baseline.toml  amostragem do prior
└── sampling_agent.toml     amostragem do agente
```

Não versionados (ver `.gitignore`): priors, dados do BraCoLi, checkpoints,
saídas brutas de amostragem e material de terceiros.

---

## Reprodução

### Pré-requisitos

Docker Desktop e cerca de 10 GB livres.

### Montagem

**1. Dados do BraCoLi.** Baixe de
<https://www.farmacia.ufmg.br/qf/downloads/> e coloque o `BraCoLi_v1.sdf` em
`BraCoLi_v1/`.

**2. Prior do REINVENT.** O `reinvent.prior` (treinado em ChEMBL) foi removido
da árvore de trabalho do repositório e da versão atual do registro no Zenodo.
Permanece no histórico do git, na tag `v4.5.11`:

```
docker compose run --rm reinvent bash -c "git clone --depth 1 --branch v4.5.11 --filter=blob:none --no-checkout https://github.com/MolecularAI/REINVENT4.git /tmp/p && git -C /tmp/p checkout HEAD -- priors/reinvent.prior && cp /tmp/p/priors/reinvent.prior /work/priors/ && sha512sum /work/priors/reinvent.prior"
```

SHA-512 esperado:

```
3ce451f01e19d87ca5a80a83b4d3dbc1f715d63a88af05441d39f95f1dbc935a90ba251dc92fca29b1346e63fe63737bd627dce7d16bfe4f78f2a28e4b7c0fb0
```

**3. Imagens.**

```
docker compose build
```

### Execução (cmd do Windows, uma linha por comando)

```
docker compose run --rm tools python prep_bracoli.py BraCoLi_v1/BraCoLi_v1.sdf
```

```
docker compose run --rm reinvent reinvent -l sampling_baseline.log sampling_baseline.toml
```

```
docker compose run --rm reinvent reinvent -l tl_bracoli.log tl_bracoli.toml
```

```
docker compose run --rm reinvent python read_tl_curve.py
```

```
docker compose run --rm reinvent reinvent -l sampling_agent.log sampling_agent.toml
```

```
docker compose run --rm tools python filter_generated.py results/sampled_agent.csv --reference data/bracoli_clean.smi --label agent --outdir results/filtered
```

Acompanhe o transfer learning em <http://localhost:6006> com
`docker compose --profile monitor up -d tensorboard`.

---

## Arquitetura dos contêineres

| Serviço | Base | Função |
|---|---|---|
| `reinvent` | `python:3.12-slim` + REINVENT 4.8 | *transfer learning*, amostragem |
| `tools` | `python:3.11-slim` + RDKit, PaDEL, scikit-learn | curadoria, filtragem, análise |
| `tensorboard` | mesma imagem do `reinvent` | monitoramento do TL |

A separação é necessária: o REINVENT fixa suas dependências com *lock file*, e
instalar RDKit e PaDEL no mesmo ambiente quebra essa resolução.

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

**1. Prior: ChEMBL ou PubChem?** O registro atual do Zenodo oferece apenas
`reinvent_pubchem.prior`. O `reinvent.prior` (ChEMBL), descrito no artigo do
REINVENT 4, só existe no histórico do git. A escolha adotada é o de ChEMBL,
por coerência com a literatura citada.

**2. O agente se deslocou na direção do BraCoLi, mas chegou lá?** O que está
demonstrado é o deslocamento. Falta comparar o perfil de descritores do agente
com o dos 982 compostos originais: se o *transfer learning* aproximou a
distribuição, os dois perfis coincidem. É o próximo passo.

**3. Diversidade é o custo.** Metade dos esqueletos de Murcko. Focalizar e
diversificar são objetivos opostos, e o trabalho mede o preço de um em termos
do outro.

**4. Domínio de aplicabilidade.** Se nenhuma molécula gerada cair dentro do
domínio dos modelos QSAR, isso é um resultado publicável sobre
transferibilidade, não uma falha. O erro seria afrouxar o limiar até
aparecerem acertos.

**5. Pós-processamento, não *reinforcement learning*.** Embutir SA score e
atividade como componentes de *scoring* num *staged learning* é mais poderoso,
mas agrava o problema do domínio de aplicabilidade: o RL explora ativamente as
fraquezas do modelo de score. Trabalho futuro.

---

## Reprodutibilidade

| Componente | Versão |
|---|---|
| REINVENT | `v4.8` (tag fixada no `Dockerfile`) |
| Prior | `reinvent.prior`, SHA-512 `3ce451f0...4b7c0fb0` |
| BraCoLi | v1, 1176 compostos |
| Checkpoint do agente | época 85 de 100 |
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
- Bemis GW, Murcko MA. *The properties of known drugs. 1. Molecular
  frameworks.* J Med Chem 39:2887-2893 (1996).
- Código do REINVENT 4: <https://github.com/MolecularAI/REINVENT4>
