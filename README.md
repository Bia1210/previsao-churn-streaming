# # Previsão de churn em streaming

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bia1210/previsao-churn-streaming/blob/main/analisar_churn_clientes.ipynb)

Projeto de estudo de Machine Learning: prever quais clientes de um serviço de streaming vão cancelar nos próximos 60 dias, para priorizar ações de retenção.

> **Aviso:** os dados são **sintéticos**, gerados para este estudo. Os resultados mostram que o método funciona, mas não descrevem clientes reais.

## O problema

A equipe de retenção tem orçamento limitado e não pode abordar todos os clientes. A pergunta é: **quem abordar primeiro, e vale a pena abordar?**

- **Tipo de problema:** classificação binária (cancelou ou não nos 60 dias seguintes).
- **Base:** 50.000 clientes ativos em 01/01/2026, com 14,1% de churn.
- **Alvo:** `cancelou_60d`.

## Dados

- 24 colunas, sem dados pessoais (LGPD): perfil, plano, forma de pagamento, engajamento, suporte e falhas de pagamento.
- **Sem vazamento de dados:** todas as variáveis são calculadas com informações anteriores à data de corte. O alvo observa só os 60 dias seguintes.
- Nulos realistas em idade, renda, dispositivo e variação de engajamento (clientes novos não têm histórico).

## Método

1. Exploração dos dados e checagem de vazamento.
2. Divisão treino (80%) e teste (20%), estratificada, com o teste intocado até o fim.
3. Baseline com `DummyClassifier`.
4. `Pipeline` de pré-processamento (imputação, One-Hot Encoding, padronização) para que nada seja aprendido fora do treino.
5. Validação cruzada estratificada de 5 partes, comparando regressão logística com L2 e Gradient Boosting raso.
6. Avaliação final no teste, uma única vez.
7. Lift por decil, importância por permutação e escolha do corte de decisão pelo custo e ganho da ação de retenção.

**Métrica principal: PR-AUC**, porque as classes são desbalanceadas (14% de churn) e a acurácia engana.

## Resultados

| Modelo | PR-AUC no teste |
|---|---|
| Baseline (acaso) | 0,141 |
| Regressão logística L2 | **0,405** |

<!-- Substitua 0,405 pelo valor do seu teste se for outro. Com a divisão random_state=42 do notebook, obtive 0,419. -->

**O que mais pesa na previsão:** engajamento e variação do engajamento, seguidos de reclamações abertas e falhas de pagamento.

### Experimentos

| Experimento | Resultado |
|---|---|
| Remover as variáveis de engajamento | PR-AUC cai de 0,405 para 0,362 (queda de cerca de 11%) |
| `class_weight="balanced"` | PR-AUC igual (0,405 contra 0,404). Com corte de 0,5, a precision cai de 0,59 para 0,30 e o recall sobe de 0,15 para 0,71 |
| Criar `horas_por_dia_ativo` | Sem ganho (diferença de 0,0000), então a feature foi descartada |
| Incluir uma coluna derivada do alvo (vazamento de propósito) | PR-AUC de 1,000, resultado artificial que serve de alerta |

### Do modelo à decisão de negócio

Abordar um cliente compensa quando a probabilidade de churn dele supera:

```
corte = custo da ação ÷ (taxa de sucesso × valor retido)
```

Com custo de R$ 15, sucesso de 30% e R$ 120 salvos por ação bem-sucedida, o corte teórico é de **0,417**.

| Como o corte foi escolhido | Lucro estimado no teste |
|---|---|
| Fixo em 0,50 | R$ 3.312 |
| Escolhido numa validação separada (corte 0,39) | **R$ 3.474** |
| Escolhido olhando o teste | R$ 3.915 (otimista, pois o teste influenciou a escolha) |

O valor honesto é o de **R$ 3.474**. O de R$ 3.915 ilustra como escolher parâmetros olhando o teste infla o resultado.

## O que aprendi

- **Vazamento de dados é o erro mais perigoso.** Em uma base de 450 clientes, colunas como "total pago" e "último acesso" levaram a PR-AUC a cerca de 0,97. Sem elas, ficou perto de 0,25.
<!-- Resultado da etapa 3 do roteiro (base de 450 linhas). Confirme no seu Colab antes de manter. -->
- **O desempenho de treino não basta.** Comparar treino e validação revela overfitting, e quando dois modelos empatam, o mais simples é a melhor escolha.
- **PR-AUC antes de acurácia** em classes desbalanceadas.
- **O corte de decisão vem da economia do problema**, não de um 0,5 fixo.
- **Criar variáveis não garante melhora.** Cada feature nova precisa ser testada e descartada se não ajudar.
- **Variações pequenas entre execuções são ruído.** Duas divisões diferentes deram 0,419 e 0,405 para o mesmo modelo, então só vale comemorar diferenças maiores que isso.

## Limites

- Os dados são sintéticos e geram relações mais "limpas" que as reais.
- Os parâmetros de custo, sucesso e valor da ação são premissas e precisam ser substituídos por números reais.
- Não há validação temporal (treinar no passado e testar no futuro), o que seria essencial com dados reais.
- A regressão logística assume relações lineares. Com dados reais, vale comparar com modelos mais flexíveis e revisar a calibração das probabilidades.

## Próximos passos

- Repetir o fluxo com um dataset público, como o *Telco Customer Churn*.
- Adicionar validação temporal e calibração de probabilidades.
- Explicar o modelo com SHAP.

## Como reproduzir

1. Abra o notebook no Google Colab.
2. Envie o arquivo `dataset_streaming_churn_50k.csv` quando solicitado.
3. Execute as células na ordem (**Ambiente de execução → Executar tudo**).

## Estrutura

```
├── churn_passo_a_passo_colab.ipynb   # notebook com o passo a passo
├── dataset_streaming_churn_50k.csv   # dados sintéticos
├── dicionario_de_dados.csv           # descrição das colunas
└── README.md
```

## Tecnologias

Python, pandas, NumPy, scikit-learn, Matplotlib e Google Colab.
