### Aprendizado de Máquinas: Desafio 3
# Aluno: Michel de Farias Albuquerque





Desafio = function(x) {
  library(caret)
  library(kernlab)
  library(ISLR)
  library(epiDisplay)
  library(mlr)
  load("Michel_AlbuquerqueD3.RData")
  dados_desafio = x
  dados_desafio$diagnosis = factor(dados_desafio$diagnosis, # atribuindo os fatores corretamente
                                   levels = c("B","M"),
                                   labels = c(1,0))
  dados_desafio = reimpute(dados_desafio,treino_imp$desc)  # tratando possíveis NAs
  dados_desafio = predict(padr,dados_desafio)              # padronizando os dados
  dados_desafio = dados_desafio |>                         # retirando variáveis não importante para o modelo
    dplyr::select(-c(highcor))
  previsao = predict(modelFit,dados_desafio)               # aplicando o modelo aos novos dados
  confusionMatrix(data = previsao,                         # avaliando previsao
                  reference = dados_desafio$diagnosis)
}


#load("dados")
#Desafio(dados)
