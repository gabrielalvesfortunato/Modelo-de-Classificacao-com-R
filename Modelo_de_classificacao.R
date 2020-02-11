setwd("C:/Users/Gabriel/Desktop/Cursos/BigDataReAzure/cap11 - MachineLearning")
getwd()


# Definiçao do Problema de Negócio: Previsão de ocorrências de câncer de Mama
#http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29


## Etapa 1 - Coletando os dados

# Os dados de cancer de mama incluem 569 observações de biópsias de câncer.
# cada um com 32 características (variáveis). Uma caracteristica é um numero de
# identifição (ID), outro é o diagnostico de câncer, e 30 são medidas laboratoriais
# numéricas. O diagnostico é codificado como "M" para indicar maligno ou "B" para 
# indicar benigno
dados <- read.csv("dataset.csv", stringsAsFactors = FALSE)
str(dados)
View(dados)



## Etapa 2 - Pré-Processamento

# Excluindo a coluna ID
# Independentemente do método de apredizagem de máquina, deve sempre ser excluidas
# Variáveis de ID. Caso contrário, isso pode levar a resultados errados porque o ID
# pode ser usado para unicamente "prever" cada exemplo. Por conseguinte, um modelo
# que inclui um identificador pode sofrer de superajuste (overfitting),
# e será muito difícil usá-lo para generalizar outros dados.
dados$id <- NULL
View(dados)

# Ajustando o label da variável alvo
dados$diagnosis <- sapply(dados$diagnosis, function(x){ifelse(x=="M", "Maligno", "Benigno")})

# Muitos classificadores requerem que as variaveis sejam do tipo Fator
table(dados$diagnosis)
dados$diagnosis <- factor(dados$diagnosis, levels = c("Benigno", "Maligno"), labels = c("Benigno", "Maligno"))
str(dados$diagnosis)

# Verificando a proporção 
round(prop.table(table(dados$diagnosis)) * 100, digits = 1)

# Medidas de Têndencia Central
# Detectamos um problema de escala entre os dados, que entao precisam ser normalizados
# o cálculo de distância feito pelo knn é dependente das medidas de escala nos dados de entrada
summary(dados[c("radius_mean", "area_mean", "smoothness_mean")])

# Criando uma função de normalização
normalizar <- function(x){
  return((x - min(x)) / (max(x) - min(x)))
}

# Testando a função de normaização - os resultados devem ser identicos
normalizar(c(1, 2, 3, 4, 5))
normalizar(c(10, 20, 30, 40, 50))

# Normalizando os dados
dados_normalizados <- as.data.frame(lapply(dados[2:31], normalizar))
View(dados_normalizados)



## Etapa 3 - Treinando o modelo com KNN

# Carregando o pacote library
# install.packages("class")
library(class)
?knn

# Criando dados de treino e dados de teste
dados_treino <- dados_normalizados[1:469, ]
dados_teste <- dados_normalizados[470:569, ]

# Criando os labels para os dados de treino e de teste
dados_treino_labels <- dados[1:469, 1]
dados_teste_labels <- dados[470:569, 1]

length(dados_teste_labels)
length(dados_treino_labels)


## Criando o Modelo
modelo_knn_v1 <- knn(train = dados_treino,
                     test = dados_teste,
                     cl = dados_treino_labels,
                     k = 21)

summary(modelo_knn_v1)



## Etapa 4 - Avaliando e Interpretando o Modelo

# Carregando o gmodels
library(gmodels)

# Criando uma tabela cruzada dos dados previstos x dados atuais
# Usaremos amostra com 100 observações: length(dados)teste_labels
CrossTable(x = dados_teste_labels, y = modelo_knn_v1, prop.chisq = FALSE)


# Interpretando os resultados:
# A tabela cruzada mostra 4 possiveis valores, que representam os falso/verdadeiros
# Temos duas colunas listando os labels originais nos dados observados.
# Temos duas linhas listando os labels dos dados de teste

# Falso Positivo - Erro tipo I 
# Falso Negativo - Erro tipo II

# Taxa de acerto do Modelo: 98% (acertou 98 em 100)



## Etapa 5: Otimizando a performance do modelo

# Usando a função scale() para padronizar o z-score
?scale()
dados_z <- as.data.frame(scale(dados[-1]))

# Confirmando transformação realizada com sucesso
summary(dados_z$area_mean)

# Criando novos datasets de treino e de teste
dados_treino <- dados_z[1:469, ]
dados_teste <- dados_z[470:569, ]

View(dados_z)

dados_treino_labels <- dados[1:469, 1] 
dados_teste_labels <- dados[470:569, 1]


# Reclassificando
modelo_knn_v2 <- knn(train = dados_treino,
                     test = dados_teste,
                     cl = dados_treino_labels,
                     k = 21)


# Criando uma tabela cruzada dos dados previstos x dados atuais
CrossTable(x = dados_teste_labels, y = modelo_knn_v2, prop.chisq = FALSE)

# CONCLUIMOS QUE O DESEMPENHO DO ALGORITMO PIOROU :(




## Etapa - 6: Construindo um Modelo com Algoritmo Support Vector Machine(SVM)
# Este Algoritmo pode ser usado tanto para Regressão quanto para Classificação

# Definindo a semente para resultados reproduzíveis
set.seed(40)

# Prepara o dataset
dados <- read.csv("dataset.csv", stringsAsFactors = F)
View(dados)
dados$id <- NULL
dados[, "index"] <- ifelse(runif(nrow(dados)) < 0.8, 1, 0) # Definindo um index para a futura separaçao dos dados
View(dados)

# Dados de treino e teste
trainset <- dados[dados$index == 1,]
testset <- dados[dados$index == 0,]

# Obter o indice
trainColNum <- grep("index", names(trainset))
print(trainColNum)

# Removendo o indice dos datasets
trainset <- trainset[, -trainColNum]
testset <- testset[, -trainColNum]

# Obter indice de colunas da variável target no conjunto de dados
typeColNum <- grep("diag", names(dados))


# Criando o modelo 
# Nos ajustamos o kernel para radial, já que este conjunto de dados não tem
# plano linear que pode ser desenhado
install.packages("e1071")
library(e1071)
?svm
modelo_svm_v1 <- svm(diagnosis ~ .,
                     data = trainset,
                     type = "C-classification",
                     kernel = "radial")


# Previsões
pred_train <- predict(modelo_svm_v1, trainset)

# Percentual de previsões corretas com dataset de TREINO
mean(pred_train == trainset$diagnosis)

# Previsões com os dados de TESTE
pred_test <- predict(modelo_svm_v1, testset)

# Percentual de previsões corretas com dataset de TESTE
mean(pred_test == testset$diagnosis)

# Confusion Matrix
table(pred_test, testset$diagnosis)



### Construindo um modelo com Random Forest

# Criando o modelo
library(rpart)
modelo_rf_v1 <- rpart(diagnosis ~ ., data = trainset, control = rpart.control(cp = .0005))

# Previsões nos dados de teste
tree_pred <- predict(modelo_rf_v1, testset, type = "class")

# Previsões corretas nos dados de teste
mean(tree_pred == testset$diagnosis)

# Confusion Matrix
table(tree_pred, testset$diagnosis)
