# Duelo: Random Forest × SVM — quem vai cancelar o plano?

Comparação entre **Random Forest** e **SVM** para prever *churn* (cancelamento de plano) em uma operadora de telecomunicações, com foco em qual modelo atende melhor o time de retenção.

## Objetivo

Identificar os clientes com maior risco de cancelar para que o time de retenção entre em contato **antes** da saída, oferecendo desconto ou upgrade. A métrica principal é o **recall da classe churn**: de todos os clientes que realmente cancelaram, quantos o modelo conseguiu apontar.

## Dados

Dataset **Telco Customer Churn** (IBM / Kaggle): 7.043 clientes e 20 variáveis (tempo de contrato, tipo de contrato, internet, forma de pagamento, mensalidade, serviços adicionais etc.).

Após a limpeza, ficam **7.032 clientes** e **30 colunas**, com **26,6%** de cancelamentos.

Tratamentos aplicados:
- `TotalCharges` convertida de texto para número (`errors="coerce"`), removendo 11 clientes novos com valor vazio;
- remoção da coluna `customerID`;
- one-hot encoding das variáveis categóricas (`pd.get_dummies`).

## Metodologia

- Validação cruzada estratificada com 5 folds (`StratifiedKFold`, `random_state=42`);
- Métricas: acurácia, recall (churn), F1 e tempo;
- Modelos: `DummyClassifier` (linha de base), `RandomForestClassifier` (300 árvores, profundidade máxima 8) e `SVC` com `StandardScaler` em pipeline;
- Teste do parâmetro `class_weight="balanced"` nos dois modelos.

## Resultados

| Modelo | Acurácia | Recall (churn) | F1 |
|---|---|---|---|
| Chute: ninguém cancela | 73,4% | 0,0% | 0,0% |
| SVM sem escala | 73,4% | 0,0% | 0,0% |
| SVM com escala | 79,6% | 49,0% | 56,1% |
| Random Forest | 80,3% | 49,9% | 57,4% |
| SVM balanceado | 74,5% | 78,0% | 61,9% |
| Random Forest balanceada | 76,3% | 77,1% | 63,3% |

Os valores podem variar um pouco entre execuções e ambientes.

## Principais conclusões

- **Acurácia engana:** o chute "ninguém cancela" acerta 73,4%, mas não identifica nenhum cliente em risco.
- **Escalar é essencial para o SVM:** sem `StandardScaler`, o `TotalCharges` domina o cálculo de distância e o modelo se comporta como o chute.
- **Sem balanceamento**, os dois modelos pegam apenas cerca de metade dos clientes que cancelam.
- **Com `class_weight="balanced"`**, o recall sobe para cerca de 77–78% e o F1 melhora, ao custo de uma queda na acurácia e de mais alarmes falsos.
- **Recomendação:** Random Forest com balanceamento. Ela tem desempenho parecido com o do SVM balanceado e ainda indica quais fatores mais pesam no cancelamento.

## Fatores que mais influenciam o cancelamento

Segundo a importância das variáveis da Random Forest, as principais são `tenure` (tempo de casa), `TotalCharges`, `Contract_Two year`, `InternetService_Fiber optic`, `MonthlyCharges` e `PaymentMethod_Electronic check`.

Ações possíveis:
- acompanhar de perto os clientes nos primeiros meses;
- oferecer vantagens para migrar de contrato mensal para contratos de 1 ou 2 anos;
- investigar preço e qualidade do serviço de fibra ótica.

## Como executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/SEU_USUARIO/duelo-rf-svm-churn.git
   cd duelo-rf-svm-churn
   ```
2. Instale as dependências:
   ```bash
   pip install pandas matplotlib scikit-learn jupyter
   ```
3. Abra o notebook `duelo_rf_svm.ipynb` e execute todas as células. O dataset é baixado automaticamente da URL indicada no código, então é preciso ter conexão com a internet.

## Tecnologias

Python, pandas, scikit-learn, matplotlib e Jupyter Notebook.

## Autor
Joao Pedro Lima Barbosa 
