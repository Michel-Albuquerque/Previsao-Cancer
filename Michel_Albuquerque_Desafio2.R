
##### Aprendizado de Máquinas #####

##### Desafio 2 ##########

# Aluno: Michel de Farias Albuquerque







# selecionando o diretório onde estão os dados
setwd("~/Michel/ESTATISTICA/8periodo/AprendizadoM/Desafios/Desafio3")


# importando bibliotecas

library(caret)
library(tidyverse)
library(ISLR)
library(kernlab)
library(epiDisplay)
library(mlr)
library(Amelia)





# importando base
load("desafio3.rda")

# Visualizando dados
#View(dados)


# obtendo um resumo da base de dados
dim(dados)
summary(dados)
str(dados)


# Analisando a variável resposta
tab1(dados$diagnosis)


# criando novo banco para modifica-lo
dados1 = dados

# tratando a variável categórica

dados1$diagnosis = factor(dados1$diagnosis,
                         levels = c("B","M"),
                         labels = c(1,0))


#head(dados$diagnosis,20)
#str(dados$diagnosis)


# Dividindo os dados em treino e teste

set.seed(319054040)
indices_sep = createDataPartition(y = dados1$diagnosis,
                                  p = 0.75,
                                  list = F)

dados1_treino = dados1[indices_sep,]
dados1_teste = dados1[-indices_sep,]

#dim(dados1_treino)
#dim(dados1_teste)


# verificando a presença de dados faltantes (NAs)

missmap(dados1_treino,
        col = c("yellow","black"),
        legend = FALSE)

# pode-se verificar que em todas as variáveis há
# dados NAs


# verificando a quantidade de NAs em cada variável
a = apply(X = dados1_treino[,2:31], MARGIN = 2, is.na)
b = apply(X = a, MARGIN = 2, sum)  
b

# quantidade de observações na amostra de treino
dim(dados1_treino)

# há 353 observações, então caso haja alguma variável com uma quantidade
# de NAs por volta de 160, ela será eliminada do banco de dados.
# Isso porque com uma quantidade grande de NAs fica inviável fazer até
# uma simulação desses valores pois afetaria o resultado final.

# verificando se alguma das variáveis tem 160 NAs ou mais
b > 160
sum(b>160)

# Portanto, como não temos um número muito grande de NAs em nenhuma variável,
# o que resultaria na eliminação dessa variável, e nem um 
# número muito pequeno, onde eliminaríamos as observações,
# então trataremos esses dados faltantes


# obs
# como podemos observar abaixo, também há
# muitas observações com dados faltantes, 
# ficando inviável a remoção delas

c = apply(X = dados1_treino[,2:31], MARGIN = 1, is.na)
d = apply(X = a, MARGIN = 1, sum) 
d



# tratando os NAs

#verificando o tipo de dados das variáveis
str(dados1_treino)  # somente dados numeric


treino_imp = impute(dados1_treino,
                    target = "diagnosis",
                    classes = list(numeric = imputeLearner("regr.gbm")))

dados2_treino = treino_imp$data

# verificando se há NAs no no banco de dados
missmap(dados2_treino, col = c("yellow","black"),
        legend = F)

par(mfrow = c(1,1))

# aplicando o mesmo procedimento na amostra teste
dados2_teste = reimpute(dados1_teste,treino_imp$desc)

# verificando NAs na amostra teste
missmap(dados1_teste,
        col = c("yellow","black"),
        legend = F)
missmap(dados2_teste,
        col = c("yellow","black"),
        legend = F)




# Padronização

# como os dados estão em escalas diferentes
# devemos fazer uma padronização
# para que essas diferencas não influenciem
# nos resultados do modelo

dados3_treino = dados2_treino
dados3_teste = dados2_teste

padr = preProcess(dados3_treino,
                  method = c("center","scale"))
dados3_treino = predict(padr,dados3_treino)

# View(dados3_treino)

# aplicando o modelo criado pra padronizar
# os dados de treino agora nos dados de teste
dados3_teste = predict(padr,dados3_teste)

#head(dados3_teste)








# verificando a variância


nearZeroVar(dados3_treino[,-1], saveMetrics = T)

#sem variáveis com baixa variância


# Correlação

dados_treino_cor = dados3_treino[,-1]

correlacao = cor(dados_treino_cor)
#View(correlacao)

findCorrelation(correlacao, cutoff = 0.75,
                verbose = T)

highcor = findCorrelation(correlacao, cutoff = 0.75,
                          names = T)
highcor

# eliminando variáveis com alta correlação com outras
dados4_treino = dados3_treino |> 
  dplyr::select(-highcor)


# aplicando aos dados de teste

dados4_teste = dados3_teste |> 
  dplyr::select(-highcor)



# verificando variáveis com dependencia linear
findLinearCombos(dados4_treino[,-1])

# sem variáveis com dependencia linear







# verificando proporção da variável dependente
tab1(dados4_treino$diagnosis)

# a proporção é próxima de 60/40, então
# não aplicaremos um redimensionamento



# treinando o modelo

train_Control = trainControl(method = "repeatedcv",
                             number = 5,
                             repeats = 3)

grid = expand.grid(interaction.depth = c(1,3,5),
                   n.trees = c(50,100,500),
                   shrinkage=c(0.1,0.01),
                   n.minobsinnode = 10)

# treinamento

set.seed(319054040)
modelo_v1 = caret::train(diagnosis~.,
                         data = dados4_treino,
                         method = "gbm",
                         distribution = "bernoulli",
                         trControl = train_Control,
                         tuneGrid = grid,
                         verbose = FALSE)

# peso das variáveis
head(summary(modelo_v1))
modelo_v1$results

# avaliando na amostra de teste

predicao = predict(modelo_v1,dados4_teste)
confusionMatrix(predicao,dados4_teste$diagnosis)


# Portanto:
modelFit = modelo_v1

# acurácia 0.9397
# sensibilidade 0.9726
# especificidade: 0.8837


# Gerar RData
# save.image(file = "Michel_AlbuquerqueD3.RData")





