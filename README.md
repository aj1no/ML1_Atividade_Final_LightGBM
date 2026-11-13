# Aprendizado de Máquina I — Atividade Final: Investigação do Algoritmo LightGBM (*Light Gradient Boosting Machine*)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aj1no/ML1_Atividade_Final_LightGBM/blob/main/Atividade_Final_ML1_LightGBM.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline%20%2B%20CV-orange.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

Repositório referente à **Atividade Final** da disciplina de **Aprendizado de Máquina I**, desenvolvida na **Modalidade A (Investigação de uma Técnica de Machine Learning)**.

---

## Informações Acadêmicas

- **Instituição:** FATEC Jundiaí — Faculdade de Tecnologia Deputado Ary Fossen
- **Curso:** Tecnologia em Ciência de Dados
- **Disciplina:** Aprendizado de Máquina I
- **Professor:** Prof. Me. Mateus Guilherme Fuini
- **Aluno:** Rodolfo Vinicius Cima Takemoto
- **Modalidade:** Modalidade A — Investigação de uma Técnica de Machine Learning
- **Técnica Investigada:** **LightGBM (*Light Gradient Boosting Machine*)** aplicada à Predição de Inadimplência de Crédito Bancário (*Credit Default Risk*) e comparada criticamente com as técnicas supervisionadas estudadas na disciplina (*Random Forest*, *Árvore de Decisão*, *SVM com Kernel RBF*, *K-NN* e *Naive Bayes*).

---

## Vídeo de Apresentação

▶️ **[Assistir à apresentação em vídeo (7 min 16 s)](video/Apresentacao_LightGBM_Rodolfo.mp4)**

Apresentação narrada da investigação: fundamentação teórica do LightGBM, problema de negócio, os três experimentos e a avaliação final no conjunto de teste.

---

## Estrutura do Projeto (Alinhada ao Edital e ao Checklist de Projetos de ML)

O notebook principal (`Atividade_Final_ML1_LightGBM.ipynb`) contempla integralmente as **8 seções obrigatórias do edital**:

1. **Seção 1 — Introdução:** Motivação histórica, finalidade do **LightGBM** (Ke et al., NeurIPS 2017) e problemas que resolve em dados tabulares heterogêneos.
2. **Seção 2 — Fundamentação Teórica e Matemática:** Formulação aditiva do *Gradient Boosting* ($F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$), expansão de Taylor de segunda ordem (gradientes $g_i$ e hessianas $h_i$), discretização em histogramas de 255 *bins* com subtração de histogramas, **GOSS** (*Gradient-based One-Side Sampling*), **EFB** (*Exclusive Feature Bundling*), crescimento ***leaf-wise* (*best-first*)** vs. ***level-wise* (*depth-first*)** e dicionário de hiperparâmetros.
3. **Seção 3 — Aplicações Práticas e Problema de Negócio:** Casos reais de uso industrial e modelagem do problema de **Risco de Inadimplência de Crédito Bancário (`N = 5.000` contratos, `22,0%` de inadimplentes)** com matriz de custos assimétricos ($\text{Custo}_{\text{FN}} = \text{R\$ } 8.500,00$ por inadimplente aprovado vs. $\text{Custo}_{\text{FP}} = \text{R\$ } 450,00$ por bom pagador revisado em mesa de crédito).
4. **Seção 4 — Implementação em Python (EDA, Prevenção de *Data Leakage* e Baselines):** Análise exploratória, diagnóstico de assimetria e *outliers*, divisão estratificada (`Treino = 3.500`, `Validação = 750`, `Teste Oficial = 750`) **antes** do pré-processamento, construção de `ColumnTransformer` + `Pipeline` livre de vazamento de dados e comparação inicial com `DummyClassifier`.
5. **Seção 5 — Experimentação Controlada:** Três experimentos sistemáticos conduzidos via validação cruzada estratificada de 5 *folds* (`StratifiedKFold`) e partição de validação.
6. **Seção 6 — Avaliação Comparativa Final no Conjunto de Teste Oficial (`N = 750`):** Confronto entre 7 modelos usando *Accuracy*, *Precision*, *Recall*, *F1-Score*, *ROC-AUC*, Curvas ROC, Importância de Variáveis, Tempo de Treinamento e Custo Financeiro Total.
7. **Seção 7 — Análise Crítica:** Vantagens, limitações operacionais e quadro comparativo com os algoritmos da disciplina.
8. **Seção 8 — Conclusão e Referências Bibliográficas.**

---

## Principais Resultados Experimentais

> Todos os números abaixo são gerados pelo próprio notebook (`SEED = 42`). Versões diferentes de `scikit-learn`/`LightGBM` podem alterar as últimas casas decimais e os tempos de treino.

### 1. Experimento 1 — Controle de Complexidade *Leaf-Wise* e Trade-off Viés × Variância (`5-Fold CV`)

| Configuração | `F1 Treino` | `F1 Val (5-Fold)` | `Gap F1 (Tr - Val)` | `AUC Treino` | `AUC Val (5-Fold)` | Diagnóstico Técnico |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `C1 (L=3, d=2, lr=0.02)` | 0,7814 | 0,7657 ± 0,0252 | +0,0157 | 0,9594 | 0,9537 | *Underfitting* (Alto Viés) |
| `C2 (L=7, d=3, lr=0.05)` | 0,9031 | 0,8572 ± 0,0186 | +0,0459 | 0,9911 | 0,9803 | Conservador |
| **`C3 (L=15, d=4, lr=0.05)`** | **0,9417** | **0,8589 ± 0,0246** | **+0,0828** | **0,9969** | **0,9825** | **Maior F1 e AUC de validação ($\text{num\_leaves} < 2^4$)** |
| `C4 (L=31, d=-1, lr=0.10)` | 1,0000 | 0,8565 ± 0,0218 | +0,1435 | 1,0000 | 0,9800 | Padrão LightGBM (memoriza o treino) |
| `C5 (L=63, d=-1, lr=0.15)` | 1,0000 | 0,8568 ± 0,0225 | +0,1432 | 1,0000 | 0,9795 | Memoriza o treino |
| `C6 (L=127, d=-1, lr=0.25)` | 1,0000 | 0,8569 ± 0,0254 | +0,1431 | 1,0000 | 0,9798 | Memoriza o treino |

De `C2` a `C6`, o F1 de validação empata dentro do desvio-padrão; a diferença está no *gap* — `C3` generaliza igual sem decorar o treino.

---

### 2. Experimento 2 — Sensibilidade à Escala de Atributos e *Outliers* (`5-Fold CV`)

| Algoritmo | `F1 Sem Escala` | `F1 MinMaxScaler` | `F1 StandardScaler` | `Variação F1 (Std − Bruto)` | `AUC Sem Escala` | `AUC StandardScaler` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `K-NN (n_neighbors=7)` | 0,1195 | 0,6560 | 0,8079 | **+68,84 p.p.** | 0,5824 | 0,9576 |
| `SVM (Kernel RBF, C=1)` | 0,0000 | 0,8445 | 0,8605 | **+86,05 p.p.** | 0,8938 | 0,9810 |
| `Random Forest (d=8)` | 0,8056 | 0,8056 | 0,8050 | −0,06 p.p. (invariante) | 0,9714 | 0,9714 |
| **`LightGBM (C3)`** | **0,8589** | **0,8604** | **0,8589** | **+0,00 p.p. (invariante)** | **0,9823** | **0,9825** |

---

### 3. Experimento 3 — Calibração do Limiar de Decisão ($p^*$) pelo Custo Financeiro na Validação (`N = 750`)

Custo $= \text{R\$ } 8.500 \times \text{FN} + \text{R\$ } 450 \times \text{FP}$. A varredura fina (passo `0,01`) encontrou **`p* = 0,06`**, próximo do limiar teórico de Bayes $450 / 8.950 \approx 0,05$:

- **Limiar Padrão (`p = 0,50`):** `Precision = 0,8581` | `Recall = 0,8061` | `F1 = 0,8312` | `FN = 32` | `FP = 22` | **Custo = R\$ 281.900,00**
- **`scale_pos_weight = 3,55` (`p = 0,50`):** `Precision = 0,8063` | `Recall = 0,9333` | `F1 = 0,8652` | `FN = 11` | `FP = 37` | **Custo = R\$ 110.150,00**
- **Limiar Calibrado (`p* = 0,06`):** `Precision = 0,6074` | **`Recall = 0,9939`** | `F1 = 0,7540` | **`FN = 1`** | `FP = 106` | **Custo = R\$ 56.200,00** (**−80,06%**)

---

### 4. Avaliação Comparativa Final no Conjunto de Teste Oficial (`N = 750`)

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC | VP | FP | FN | VN | Treino (s) | Custo no Teste (R\$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1. Baseline (Dummy Most Freq)` | 0,7800 | 0,0000 | 0,0000 | 0,0000 | 0,5000 | 0 | 0 | 165 | 585 | 0,013 | 1.402.500,00 |
| `2. Naive Bayes (GaussianNB)` | 0,9040 | 0,7541 | 0,8364 | 0,7931 | 0,9650 | 138 | 45 | 27 | 540 | 0,014 | 249.750,00 |
| `3. K-NN (k=7 + StandardScaler)` | 0,9160 | 0,8542 | 0,7455 | 0,7961 | 0,9598 | 123 | 21 | 42 | 564 | 0,013 | 366.450,00 |
| `4. Decision Tree Podada (d=5)` | 0,9080 | 0,8117 | 0,7576 | 0,7837 | 0,9495 | 125 | 29 | 40 | 556 | 0,024 | 353.050,00 |
| `5. SVM (Kernel RBF, C=1.0)` | 0,9400 | 0,8896 | 0,8303 | 0,8589 | **0,9814** | 137 | 17 | 28 | 568 | 0,607 | 245.650,00 |
| `6. Random Forest (Bagging 160 arv)` | 0,9160 | 0,8400 | 0,7636 | 0,8000 | 0,9695 | 126 | 24 | 39 | 561 | 0,278 | 342.300,00 |
| **`7. LightGBM C3 (p = 0,50)`** | **0,9413** | **0,9007** | 0,8242 | **0,8608** | 0,9808 | 136 | 15 | 29 | 570 | 0,089 | 253.250,00 |
| **`8. LightGBM C3 (p* = 0,06)`** | 0,8400 | 0,5801 | **0,9879** | 0,7309 | 0,9808 | 163 | 118 | 2 | 467 | 0,089 | **70.100,00** |

No limiar padrão, LightGBM e SVM RBF empatam na prática (diferenças de 1–2 clientes), com o LightGBM treinando ≈ 7× mais rápido. Com o limiar de custo, o LightGBM reduz o prejuízo no teste em 72,3% em relação ao próprio limiar padrão.

---

## Estrutura do Repositório

```text
ML1_Atividade_Final_LightGBM/
├── Atividade_Final_ML1_LightGBM.ipynb   # Notebook completo com 8 secoes, 3 experimentos, graficos e analises
├── README.md                            # Documentacao academica e sintese dos resultados
├── video/
│   └── Apresentacao_LightGBM_Rodolfo.mp4  # Video de apresentacao (7 min 16 s)
├── LICENSE                              # Licenca MIT
└── .gitignore                           # Arquivos ignorados pelo Git
```

---

## Como Executar

1. Clique no selo **Open in Colab** no topo deste README para abrir e executar o notebook diretamente no Google Colab.
2. Ou execute localmente em um ambiente com Python 3.10+:
   ```bash
   pip install numpy pandas scikit-learn lightgbm matplotlib seaborn jupyter
   jupyter notebook Atividade_Final_ML1_LightGBM.ipynb
   ```
