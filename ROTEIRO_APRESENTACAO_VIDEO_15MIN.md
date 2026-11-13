# Roteiro de Apresentação em Vídeo (Até 15 Minutos) — Aprendizado de Máquina I

**Aluno:** Rodolfo Vinicius Cima Takemoto  
**Professor:** Prof. Me. Mateus Guilherme Fuini  
**Disciplina:** Aprendizado de Máquina I — FATEC Jundiaí  
**Modalidade:** Modalidade A — Investigação de uma Técnica de Machine Learning (**LightGBM**)  

---

## Como Gravar a Apresentação (Dicas Rápidas)

1. Abra o notebook `Atividade_Final_ML1_LightGBM.ipynb` no **Google Colab** (ou no VS Code / Jupyter) com todas as saídas e gráficos já visíveis.
2. Utilize o **OBS Studio**, **Microsoft Clipchamp**, **Zoom** ou **Google Meet** para gravar sua tela + câmera/microfone.
3. Siga os 6 blocos abaixo para fechar o vídeo entre **12 e 14 minutos** (dentro do limite máximo de 15 minutos do edital), cobrindo 100% dos critérios de avaliação do Prof. Mateus:
   - Fundamentação e domínio técnico (25%)
   - Qualidade da implementação (25%)
   - Experimentos e análise dos resultados (20%)
   - Análise crítica e conclusões (15%)
   - Clareza e organização da apresentação (15%)

---

## Cronograma Minuto a Minuto

### Bloco 1 (00:00 – 02:00) — Introdução e Diferença entre Bagging e Boosting (Seção 1)
* **O que mostrar na tela:** Cabeçalho do Notebook e Seção 1.
* **O que falar:**
  - *"Olá, professor Mateus. Eu sou o Rodolfo Vinicius Cima Takemoto e neste trabalho final de Aprendizado de Máquina I escolhi a Modalidade A para investigar o algoritmo LightGBM (Light Gradient Boosting Machine), proposto pela Microsoft Research na NeurIPS de 2017."*
  - *"Na disciplina, estudamos árvores de decisão e a Random Forest, que usa o paradigma de Bagging: treinar várias árvores independentes em paralelo via Bootstrap para reduzir a variância."*
  - *"O LightGBM pertence à família do Gradient Boosting: aqui as árvores são treinadas em sequência de forma aditiva. Cada nova árvore não tenta acertar o alvo do zero, mas sim prever os resíduos (os gradientes negativos e hessianas da função de perda) deixados pelas árvores anteriores, atuando diretamente na redução do viés."*

---

### Bloco 2 (02:00 – 05:00) — Fundamentação Teórica: Histogramas, GOSS, EFB e Leaf-Wise (Seção 2)
* **O que mostrar na tela:** Seção 2 (Fórmulas 2.1, subseções 2.2, 2.3 e Tabela 2.4 de hiperparâmetros).
* **O que falar:**
  - *"Por que o LightGBM foi criado? Porque o Gradient Boosting clássico precisa ordenar todos os N registros para cada atributo contínuo a fim de achar o melhor corte, o que custa O(N x d). O LightGBM resolve isso com três inovações principais:"*
  - *"Primeiro: o algoritmo baseado em Histogramas. Ele agrupa variáveis contínuas em 255 bins discretos, reduzindo a busca para O(255 x d), e usa subtração de histogramas (o histograma do filho maior é obtido subtraindo o filho menor do nó pai)."*
  - *"Segundo: o GOSS (Gradient-based One-Side Sampling), que mantém todas as amostras com gradiente grande (aquelas que o modelo ainda erra) e sorteia apenas uma fração das amostras de gradiente pequeno."*
  - *"Terceiro: o EFB (Exclusive Feature Bundling), que empacota colunas esparsas mutuamente exclusivas, como variáveis one-hot."*
  - *"Além disso, enquanto árvores tradicionais crescem por nível (level-wise), o LightGBM cresce orientado à folha (leaf-wise): ele sempre divide a folha de maior ganho. Isso converge muito mais rápido, mas exige controlar `num_leaves` e `max_depth` pela regra `num_leaves < 2^max_depth` para evitar overfitting."*

---

### Bloco 3 (05:00 – 07:30) — Problema de Negócio, EDA e Pipeline Sem Data Leakage (Seções 3 e 4)
* **O que mostrar na tela:** Seção 3, Figura 1 (painel 2x2 da EDA) e código do `ColumnTransformer` + `Pipeline` (Seção 4.3).
* **O que falar:**
  - *"Para avaliar a técnica em um cenário realista de negócios, modelei uma carteira bancária de 5.000 solicitações de crédito com 12 variáveis e taxa de inadimplência de 22% (desbalanceada: 3.900 adimplentes e 1.100 inadimplentes)."*
  - *"Definimos uma matriz de custos assimétricos: um Falso Negativo (emprestar para quem vai dar calote) custa em média R$ 8.500,00 ao banco, enquanto um Falso Positivo (enviar um bom pagador para análise em mesa de crédito) custa R$ 450,00."*
  - *"Na Figura 1 da EDA, vemos dois pontos ligados às nossas aulas: primeiro, `renda_mensal` tem forte assimetria positiva (skew = 2,18) e outliers acima de R$ 40 mil; segundo, a taxa de inadimplência salta de 7,8% (zero atrasos) para 86,4% (4 ou mais atrasos)."*
  - *"Seguindo a Aula 03 e a Aula 04, dividi os dados de forma estratificada ANTES de qualquer pré-processamento (3.500 treino, 750 validação, 750 teste oficial) e encapsulei a imputação pela mediana, o escalonamento e o OneHotEncoder dentro de um `Pipeline` com `ColumnTransformer` para garantir zero Data Leakage."*

---

### Bloco 4 (07:30 – 11:30) — Os Três Experimentos Controlados (Seção 5)
* **O que mostrar na tela:** Tabelas e Figuras 2, 3 e 4 da Seção 5.
* **O que falar:**
  - **Experimento 1 (Figura 2 — Complexidade Leaf-Wise e Viés × Variância):** *"No Experimento 1, variei a complexidade de C1 (3 folhas, profundidade 2) até C6 (127 folhas, sem limite de profundidade) usando validação cruzada de 5 folds com `return_train_score=True`. Na Figura 2, vemos que no padrão irrestrito (C4 a C6) o F1 de treino vai a 0,9984, mas o F1 de validação cai para 0,7180 — um gap de generalização de +0,2804 (overfitting clássico). Já na configuração C3 (`num_leaves=15, max_depth=4, min_child_samples=30, lr=0.05`), respeitando `15 < 2^4`, atingimos o maior F1 de validação (0,7748) e ROC-AUC de 0,9296 com gap de apenas 2,23 pontos percentuais."*
  - **Experimento 2 (Figura 3 — Sensibilidade à Escala e Outliers):** *"No Experimento 2, conectei com a Aula 04 testando os modelos Sem Escala, com MinMaxScaler e com StandardScaler. Como mostra a Figura 3, o K-NN (+76,9%) e o SVM RBF (+81,8%) colapsam sem escalonamento porque `valor_emprestimo` domina a distância euclidiana, e o MinMaxScaler sofre com os outliers de renda. Já a Random Forest e o LightGBM tiveram exatamente 0,00% de variação nos três cenários, pois particionam por quantis e histogramas, sendo invariantes a transformações monotônicas de escala."*
  - **Experimento 3 (Figura 4 — Limiar de Decisão e Custo Financeiro):** *"No Experimento 3, conectei com a Aula 02 sobre limiar de decisão. No limiar padrão p = 0,50, o LightGBM teve 46 Falsos Negativos na validação, custando R$ 400.900,00. Ao calibrar o limiar na validação para p* = 0,26 (Figura 4), o Recall subiu de 72,1% para 91,5%, os Falsos Negativos caíram de 46 para 14 e o custo caiu para R$ 145.100,00 — uma economia de R$ 255.800,00 (-63,8%)."*

---

### Bloco 5 (11:30 – 14:00) — Avaliação Final no Teste Oficial e Análise Crítica (Seções 6 e 7)
* **O que mostrar na tela:** Tabela da Seção 6, Figura 5 (Curvas ROC e Feature Importance) e Quadro Comparativo da Seção 7.2.
* **O que falar:**
  - *"Na Seção 6, abrimos pela única vez o conjunto de Teste Oficial (750 clientes) para comparar o LightGBM contra todos os algoritmos vistos na disciplina: Dummy Baseline, Naive Bayes, K-NN, Árvore de Decisão, SVM RBF e Random Forest."*
  - *"Mesmo no limiar padrão p = 0,50, o LightGBM liderou em todas as métricas: Accuracy de 90,93%, F1 de 0,7803 e ROC-AUC de 0,9284, superando a Random Forest (AUC = 0,9061) e o SVM RBF (AUC = 0,8985), além de treinar 3,9 vezes mais rápido que a Random Forest e 7,1 vezes mais rápido que o SVM."*
  - *"Com o limiar calibrado p* = 0,26, o LightGBM alcançou F1 de 0,8053 e Recall de 91,52% no teste (detectou 151 dos 165 inadimplentes), reduzindo o prejuízo financeiro para R$ 145.550,00 contra R$ 452.800,00 da Random Forest."*
  - *"Na Figura 5(b), vemos que a quantidade de atrasos em 12 meses (28,4%), o score de bureau (24,1%) e o comprometimento de renda (16,8%) concentram quase 70% do ganho de informação."*
  - *"Por fim, na Seção 7 fazemos a análise crítica: a grande vantagem do LightGBM é a eficiência em dados tabulares, invariância à escala e redução ativa de viés; já sua principal limitação é o risco de overfitting do crescimento leaf-wise se `num_leaves` e `max_depth` não forem devidamente regularizados."*

---

### Bloco 6 (14:00 – 14:45) — Conclusão e Encerramento (Seção 8)
* **O que mostrar na tela:** Seção 8 (Conclusão e Referências Bibliográficas).
* **O que falar:**
  - *"Em conclusão, o trabalho mostrou que unir uma técnica moderna baseada em histogramas como o LightGBM às boas práticas vistas na disciplina — validação cruzada olhando o gap treino-validação, prevenção de data leakage com Pipeline e ajuste do limiar pelo custo do erro — gera ganhos expressivos tanto em métricas estatísticas quanto em retorno financeiro de negócio. Muito obrigado!"*
