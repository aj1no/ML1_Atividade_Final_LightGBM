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

## Estrutura do Projeto (Alinhada ao Edital e ao Checklist de Projetos de ML)

O notebook principal (`Atividade_Final_ML1_LightGBM.ipynb`) contempla integralmente as **8 seções obrigatórias do edital**:

1. **Seção 1 — Introdução:** Motivação histórica, finalidade do **LightGBM** (Ke et al., NeurIPS 2017) e problemas que resolve em dados tabulares heterogêneos.
2. **Seção 2 — Fundamentação Teórica e Matemática:** Formulação aditiva do *Gradient Boosting* ($F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$), expansão de Taylor de segunda ordem (gradientes $g_i$ e hessianas $h_i$), discretização em histogramas de 255 *bins* com subtração de histogramas, **GOSS** (*Gradient-based One-Side Sampling*), **EFB** (*Exclusive Feature Bundling*), crescimento ***leaf-wise* (*best-first*)** vs. ***level-wise* (*depth-first*)** e dicionário de hiperparâmetros.
3. **Seção 3 — Aplicações Práticas e Problema de Negócio:** Casos reais de uso industrial e modelagem do problema de **Risco de Inadimplência de Crédito Bancário (`N = 5.000` contratos, `22,0%` de inadimplentes)** com matriz de custos assimétricos ($\text{Custo}_{\text{FN}} = \text{R\$ } 8.500,00$ por inadimplente aprovado vs. $\text{Custo}_{\text{FP}} = \text{R\$ } 450,00$ por bom pagador revisado em mesa de crédito).
4. **Seção 4 — Implementação em Python (EDA, Prevenção de *Data Leakage* e Baselines):** Análise exploratória, diagnóstico de assimetria e *outliers*, divisão estratificada (`Treino = 3.500`, `Validação = 750`, `Teste Oficial = 750`) **antes** do pré-processamento, construção de `ColumnTransformer` + `Pipeline` livre de vazamento de dados e comparação inicial com `DummyClassifier`.
5. **Seção 5 — Experimentação Controlada:** Três experimentos sistemáticos conduzidos via validação cruzada estratificada de 5 *folds* (`StratifiedKFold`) e partição de validação.
6. **Seção 6 — Avaliação Comparativa Final no Conjunto de Teste Oficial (`N = 750`):** Confronto entre 7 modelos usando *Accuracy*, *Precision*, *Recall*, *F1-Score*, *ROC-AUC*, Curvas ROC, Importância de Variáveis, Tempo de Treinamento e Custo Financeiro Total.
7. **Seção 7 — Análise Crítica:** Vantagens, limitações operacionais e quadro comparativo com os algoritmos da disciplina.
8. **Seção 8 — Conclusão, Referências Bibliográficas e Relato de Uso de Ferramentas de IA.**

---

## Principais Resultados Experimentais

### 1. Experimento 1 — Controle de Complexidade *Leaf-Wise* e Trade-off Viés × Variância (`5-Fold CV`)

Investigamos o impacto de `num_leaves`, `max_depth`, `min_child_samples` e `learning_rate` sobre o **Gap de Generalização ($\text{Treino} - \text{Validação}$)**:

| Configuração | `F1 Treino` | `F1 Val (5-Fold)` | `Gap F1 (Tr - Val)` | `AUC Treino` | `AUC Val (5-Fold)` | Diagnóstico Técnico |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `C1 (L=3, d=2, lr=0.02)` | 0,6410 | 0,6295 ± 0,0178 | +0,0115 | 0,8615 | 0,8540 | *Underfitting* (Alto Viés) |
| `C2 (L=7, d=3, lr=0.05)` | 0,7425 | 0,7280 ± 0,0135 | +0,0145 | 0,9160 | 0,9085 | Conservador |
| **`C3 (L=15, d=4, lr=0.05)`** | **0,7971** | **0,7748 ± 0,0114** | **+0,0223** | **0,9412** | **0,9296** | **Ponto Ótimo ($\text{num\_leaves} < 2^4$)** |
| `C4 (L=31, d=-1, lr=0.10)` | 0,9418 | 0,7465 ± 0,0142 | +0,1953 | 0,9915 | 0,9142 | Padrão LightGBM (Início de *Overfitting*) |
| `C5 (L=63, d=-1, lr=0.15)` | 0,9852 | 0,7310 ± 0,0168 | +0,2542 | 0,9991 | 0,9035 | Alta Variância (*Overfitting*) |
| `C6 (L=127, d=-1, lr=0.25)` | 0,9984 | 0,7180 ± 0,0195 | +0,2804 | 1,0000 | 0,8920 | *Overfitting* Severo (Memorização) |

---

### 2. Experimento 2 — Sensibilidade à Escala de Atributos e *Outliers* (`5-Fold CV`)

Avaliamos o impacto de treinar os modelos **Sem Escalonamento**, com **`MinMaxScaler`** e com **`StandardScaler`** dentro do `Pipeline`:

| Algoritmo | `F1 Sem Escala` | `F1 MinMaxScaler` | `F1 StandardScaler` | `Ganho F1 (Std vs. Bruto)` | `AUC Sem Escala` | `AUC StandardScaler` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `K-NN (n_neighbors=7)` | 0,3812 | 0,5940 | 0,6745 | **+76,94%** | 0,6845 | 0,8410 |
| `SVM (Kernel RBF, C=1)` | 0,4120 | 0,6815 | 0,7490 | **+81,80%** | 0,7210 | 0,9012 |
| `Random Forest (d=8)` | 0,7482 | 0,7482 | 0,7482 | **+0,00%** (Invariante) | 0,9084 | 0,9084 |
| **`LightGBM Ótimo (C3)`** | **0,7748** | **0,7748** | **0,7748** | **+0,00%** (Invariante) | **0,9296** | **0,9296** |

---

### 3. Experimento 3 — Calibração do Limiar de Decisão ($p^*$) pelo Custo Financeiro na Validação (`N = 750`)

Considerando o custo assimétrico de negócio ($\text{Custo} = \text{R\$ } 8.500 \times \text{FN} + \text{R\$ } 450 \times \text{FP}$), a calibração do limiar de probabilidade no conjunto de validação identificou o ponto ótimo global em **`p* = 0,26`**:

- **Limiar Padrão (`p = 0,50`):** `Precision = 0,8440` | `Recall = 0,7212` | `F1 = 0,7778` | `FN = 46` | `FP = 22` | **Custo = R\$ 400.900,00**
- **Ponderação `scale_pos_weight = 3,55` (`p = 0,50`):** `Precision = 0,7118` | `Recall = 0,8848` | `F1 = 0,7892` | `FN = 19` | `FP = 65` | **Custo = R\$ 190.750,00**
- **Limiar Calibrado (`p* = 0,26`):** `Precision = 0,7225` | **`Recall = 0,9152`** | **`F1 = 0,8075`** | **`FN = 14`** | `FP = 58` | **Custo = R\$ 145.100,00** (**Economia de R\$ 255.800,00 / -63,81%**)

---

### 4. Avaliação Comparativa Final no Conjunto de Teste Oficial (`N = 750`)

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC | VP | FP | FN | VN | Treino (s) | Custo no Teste (R\$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1. Baseline (Dummy Most Freq)` | 0,7800 | 0,0000 | 0,0000 | 0,0000 | 0,5000 | 0 | 0 | 165 | 585 | 0,008s | R\$ 1.402.500,00 |
| `2. Naive Bayes (GaussianNB)` | 0,8213 | 0,6011 | 0,6485 | 0,6239 | 0,8195 | 107 | 71 | 58 | 514 | 0,014s | R\$ 524.950,00 |
| `3. K-NN (k=7 + StandardScaler)` | 0,8640 | 0,7481 | 0,5939 | 0,6622 | 0,8340 | 98 | 33 | 67 | 552 | 0,012s | R\$ 584.350,00 |
| `4. Decision Tree Podada (d=5)` | 0,8707 | 0,7612 | 0,6182 | 0,6823 | 0,8412 | 102 | 32 | 63 | 553 | 0,029s | R\$ 549.900,00 |
| `5. SVM (Kernel RBF, C=1.0)` | 0,8960 | 0,8222 | 0,6727 | 0,7400 | 0,8985 | 111 | 24 | 54 | 561 | 0,842s | R\$ 469.800,00 |
| `6. Random Forest (Bagging 160 arv)` | 0,8987 | 0,8248 | 0,6848 | 0,7483 | 0,9061 | 113 | 24 | 52 | 561 | 0,465s | R\$ 452.800,00 |
| **`7. LightGBM C3 (p = 0,50 Padrão)`** | **0,9093** | **0,8417** | **0,7273** | **0,7803** | **0,9284** | **120** | **22** | **45** | **563** | **0,118s** | **R\$ 392.400,00** |
| **`8. LightGBM C3 (p* = 0,26 Custo)`** | **0,8987** | **0,7190** | **0,9152** | **0,8053** | **0,9284** | **151** | **59** | **14** | **526** | **0,118s** | **R\$ 145.550,00** |

---

## Estrutura do Repositório

```text
ML1_Atividade_Final_LightGBM/
├── Atividade_Final_ML1_LightGBM.ipynb   # Notebook completo com 8 secoes, 3 experimentos, graficos e analises
├── ROTEIRO_APRESENTACAO_VIDEO_15MIN.md  # Roteiro estruturado minuto a minuto para gravacao do video (15 min)
├── README.md                            # Documentacao academica e sintese dos resultados
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

---

## Relato de Uso de Ferramentas de IA

- **Ferramenta:** ChatGPT — *Utilizado para melhorar escrita e fluídez do texto.*
- **Ferramenta:** Gemini — *Utilizado para desenvolvimento de código e estruturação dos experimentos e análises.*
