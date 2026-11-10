# Prior vs agente pos-transfer-learning: comparacao com replicatas

## Desenho

- Replicatas por modelo: **10** (prior) e **10** (agente)
- SMILES por replicata: **10.000**
- Variou-se **apenas a semente de amostragem** (1..N). O checkpoint do
  agente e o arquivo do prior sao fixos, logo a variancia medida e a de
  amostragem do gerador, nao a de treino.
- Unidade experimental: a **replicata**. Uma metrica por replicata, nao
  uma por molecula - 10.000 moleculas de uma execucao nao sao 10.000
  observacoes independentes do modelo.
- Teste: Mann-Whitney bilateral para amostras independentes, **permutacao
  exata** (o `method="auto"` do SciPy cairia na aproximacao normal com
  n=10, que superestima o p em cerca de uma ordem de grandeza nesta
  faixa). Correcao de Holm-Bonferroni sobre a familia de metricas. Welch
  reportado como referencia parametrica.
- Descritores calculados sobre o conjunto **unico e valido** (etapa 2),
  antes dos filtros de novidade, SA score e drug-likeness.

Com n=10 contra n=10, o menor p bilateral alcancavel pela
permutacao exata e **1.08e-05**, e varias metricas saturam esse piso.
Saturar o piso nao mede a forca do efeito: significa apenas que o teste
chegou ao limite do que n=10 por grupo permite resolver.

As **23 de 23** metricas abaixo tem separacao completa
(delta de Cliff = +/-1): nenhuma replicata de um modelo se sobrepoe a
nenhuma do outro. Leia a diferenca de medias para o tamanho do efeito.

- Validade quimica (% das amostradas)
- Unicas (% das amostradas)
- Duplicacao interna (% das validas)
- Fora do BraCoLi (% das amostradas)
- Novas, Tanimoto <= 0.6 (% das amostradas)
- Sintetizaveis, SA <= 4.5 (% das amostradas)
- Drug-like, rendimento final (% das amostradas)
- Tanimoto maximo medio vs BraCoLi
- Tanimoto maximo mediano vs BraCoLi
- Quase-copias do BraCoLi (% das unicas)
- Similaridade interna media (Tanimoto)
- Scaffolds de Murcko unicos / molecula
- Numero de scaffolds de Murcko unicos
- Massa molecular media (Da)
- LogP medio
- TPSA media (A2)
- Doadores de H medios
- Aceptores de H medios
- Ligacoes rotaveis medias
- Numero medio de aneis
- QED medio
- SA score medio
- Atomos pesados medios

## Funil medio (% das amostradas)

| Etapa | Prior | Agente |
|---|---|---|
| Amostradas | 100.00 +/- 0.00 | 100.00 +/- 0.00 |
| Validas | 98.28 +/- 0.13 | 92.90 +/- 0.44 |
| Unicas | 98.27 +/- 0.13 | 86.12 +/- 0.56 |
| Fora do BraCoLi | 98.26 +/- 0.13 | 83.65 +/- 0.58 |
| Novas | 98.07 +/- 0.12 | 65.47 +/- 0.41 |
| Sintetizaveis | 94.22 +/- 0.20 | 55.35 +/- 0.47 |
| Drug-like | 85.36 +/- 0.39 | 48.39 +/- 0.53 |

## Comparacao metrica a metrica

Media +/- DP entre replicatas. `delta` = delta de Cliff do agente contra o
prior: +1 significa que toda replicata do agente supera toda do prior, -1 o
contrario, 0 nenhuma tendencia.

| Metrica | Prior | Agente | Diferenca | p (Holm) | delta | Efeito |
|---|---|---|---|---|---|---|
| Validade quimica (% das amostradas) | 98.283 +/- 0.130 | 92.903 +/- 0.439 | -5.380 | 0.0011 | -1.00 | grande |
| Unicas (% das amostradas) | 98.268 +/- 0.130 | 86.123 +/- 0.555 | -12.145 | 0.0011 | -1.00 | grande |
| Duplicacao interna (% das validas) | 0.015 +/- 0.011 | 7.298 +/- 0.417 | +7.283 | 0.0011 | +1.00 | grande |
| Fora do BraCoLi (% das amostradas) | 98.260 +/- 0.126 | 83.645 +/- 0.585 | -14.615 | 0.0011 | -1.00 | grande |
| Novas, Tanimoto <= 0.6 (% das amostradas) | 98.073 +/- 0.125 | 65.470 +/- 0.406 | -32.603 | 0.0011 | -1.00 | grande |
| Sintetizaveis, SA <= 4.5 (% das amostradas) | 94.224 +/- 0.202 | 55.354 +/- 0.466 | -38.870 | 2.5e-04 | -1.00 | grande |
| Drug-like, rendimento final (% das amostradas) | 85.363 +/- 0.394 | 48.393 +/- 0.530 | -36.970 | 2.5e-04 | -1.00 | grande |
| Tanimoto maximo medio vs BraCoLi | 0.260 +/- 0.001 | 0.468 +/- 0.002 | +0.208 | 2.5e-04 | +1.00 | grande |
| Tanimoto maximo mediano vs BraCoLi | 0.249 +/- 0.001 | 0.440 +/- 0.003 | +0.191 | 9.2e-04 | +1.00 | grande |
| Quase-copias do BraCoLi (% das unicas) | 0.190 +/- 0.058 | 21.728 +/- 0.381 | +21.537 | 2.5e-04 | +1.00 | grande |
| Similaridade interna media (Tanimoto) | 0.114 +/- 0.001 | 0.120 +/- 0.001 | +0.006 | 2.5e-04 | +1.00 | grande |
| Scaffolds de Murcko unicos / molecula | 0.870 +/- 0.003 | 0.499 +/- 0.003 | -0.371 | 2.5e-04 | -1.00 | grande |
| Numero de scaffolds de Murcko unicos | 8544.200 +/- 33.041 | 4171.700 +/- 19.334 | -4372.500 | 0.0011 | -1.00 | grande |
| Massa molecular media (Da) | 384.100 +/- 1.007 | 345.700 +/- 1.214 | -38.400 | 2.5e-04 | -1.00 | grande |
| LogP medio | 3.482 +/- 0.018 | 1.875 +/- 0.027 | -1.607 | 2.5e-04 | -1.00 | grande |
| TPSA media (A2) | 76.419 +/- 0.333 | 95.479 +/- 0.620 | +19.061 | 2.5e-04 | +1.00 | grande |
| Doadores de H medios | 1.455 +/- 0.014 | 1.615 +/- 0.017 | +0.160 | 2.5e-04 | +1.00 | grande |
| Aceptores de H medios | 4.635 +/- 0.024 | 5.927 +/- 0.045 | +1.292 | 2.5e-04 | +1.00 | grande |
| Ligacoes rotaveis medias | 5.386 +/- 0.023 | 5.735 +/- 0.027 | +0.349 | 2.5e-04 | +1.00 | grande |
| Numero medio de aneis | 3.378 +/- 0.008 | 2.319 +/- 0.015 | -1.060 | 2.5e-04 | -1.00 | grande |
| QED medio | 0.575 +/- 0.002 | 0.502 +/- 0.003 | -0.073 | 2.5e-04 | -1.00 | grande |
| SA score medio | 2.831 +/- 0.007 | 3.194 +/- 0.011 | +0.363 | 2.5e-04 | +1.00 | grande |
| Atomos pesados medios | 27.104 +/- 0.066 | 24.287 +/- 0.094 | -2.817 | 2.5e-04 | -1.00 | grande |

## Metricas sem diferenca detectavel (Holm, alfa = 0,05)

- (nenhuma: todas as metricas diferem)

Ausencia de significancia aqui significa que **estas replicatas nao
distinguem os modelos nesta metrica**, nao que os modelos sejam iguais.
Com n=10 por grupo a potencia so alcanca efeitos grandes.

## Arquivos

| Arquivo | Conteudo |
|---|---|
| `metricas_por_replicata.csv` | uma linha por replicata e modelo, todas as metricas |
| `comparacao_prior_vs_agente.csv` | tabela de testes completa, com Welch e Hedges g |
| `funil_medio.csv` | funil medio em N absoluto e em percentual, com DP |
| `figuras/funil_medio.{png,pdf}` | funil medio, barras agrupadas com DP |
| `figuras/metricas_por_replicata.{png,pdf}` | uma replicata por ponto, media +/- DP |
| `figuras/similaridade_bracoli.{png,pdf}` | distribuicao do Tanimoto maximo ao BraCoLi |
| `rNN/` | dados brutos de cada replicata: SMILES, funil, descritores, metricas |
