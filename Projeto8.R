# Formação Cientista de Dados 

# Projeto com Feedback 8 

# Modelagem Preditiva em IoT - Previsão de Uso de Energia 
# STEFANI HENRIQUE RAMALHO

# Importando os modulos
library(caret) # machine learning
library(randomForest) # selecao de atributos
library(lubridate) # tratar atributos do tipo data
library(ggplot2) # criacao dos plots
library(gridExtra) # grid para plotar varios graficos agrupados
library(dplyr) # sumarizacao dos dados
library(corrplot) # plotar as correlacoes
library(e1071) # Modelagem com SVM

# Definindo o diretorio raiz
setwd('D:/MachineLearning/Projetos/projeto8')

### Coleta e limpeza dos dados ###

# Lendo o conjunto de dados para treinamento
train <- read.csv('dados/projeto8-training.csv')

# Lendo o conjunto de dados para teste
test <- read.csv('dados/projeto8-testing.csv')

# Criando um novo dataset com a juncao dos dados para tratamento e analise
df <- rbind(train, test)

# Shape dos dados
dim(train) # 14803 linhas
dim(test) # 4932 linhas
dim(df)

# Visualizando as primeiras linhas do dataset
head(df)

# Estrutura dos dados
# O atributo data esta como tipo fator e devera ser transformado em tipo data
str(df)

# Buscando por valores NA
# Nao existem valores nulos ou ausentes
any(is.na(df))

# Convertendo o atributo date para tipo datetime
df$date <- ymd_hms(df$date)

# Criando o atributo dia do mes
df$day <- day(df$date)

# Criando o atributo hora
df$hour <- hour(df$date)

# Criando o atributo mes
df$month <- month(df$date)

### Analise Exploratoria ###

# Estatistica descritiva
summary(df)

# O atributo target Appliances apresenta outliers, pois o 3 quartil tem valor de 100 e 
# o valor maximo esta como 1080
# Alem disso o atributo Appliances nao e uma variavel continual, onde posteriormente serao
# tratadas por uma transformacao logaritima

# Os atributos rv1 e rv2 sao iguais
any(df$rv1 == df$rv2)

# Excluindo o atributo rv2
df$rv2 <- NULL

# Distribuicao de frequencia
p1 <- ggplot(df, aes(x = Appliances)) +
  geom_histogram(bins = 30, fill = 'red') +
  theme_light() +
  xlab('Appliances') + ylab('Frequência') +
  ggtitle('Histograma do atributo Appliances')

p2 <- ggplot(df, aes(x = log(Appliances))) +
  geom_histogram(bins = 10, fill = 'blue') +
  theme_light() +
  xlab('Appliances') + ylab('Frequência') +
  ggtitle('Histograma do atributo Appliances com transformacao logaritima')

# plotando os graficos
grid.arrange(p1, p2, ncol=1)

# No periodo de fevereiro a maio a mediana se manteve a mesma
df %>% group_by(month) %>%
  summarise(mediana = median(Appliances)) %>%
  ggplot(aes(x = as.factor(month), y = mediana)) +
  geom_bar(fill = 'blue', stat = 'identity') + 
  theme_light() +
  ggtitle("Mediana por mes do atributo Appliances") +
  xlab("Mes") + ylab("Mediana")

# Variacao Appliances o longo do ano
p1 <- df %>% filter(month %in% c(1,2,3)) %>%
  ggplot(aes(x = date, y = Appliances)) + 
  geom_line(colour = "red") +
  theme_light() +
  ggtitle("Valor de Appliances a cada 10 min de jan a mar") +
  xlab("Data") + ylab("Appliances")

p2 <-  df %>% filter(month %in% c(4,5)) %>%
  ggplot(aes(x = date, y = Appliances)) +
  geom_line(colour = "red") +
  theme_light() +
  ggtitle("Valor de Appliances a cada 10 min de abr a mai") +
  xlab("Data") + ylab("Appliances")

# plotando os graficos
grid.arrange(p1, p2, ncol=1)

# Existe um crescimento de Appliances em determinados valores de NSM
# Possivelmente por ser horario comercial
ggplot(df, aes(x = NSM, y = Appliances)) +
  geom_point(size = 2, colour = "blue", alpha = 20/100) +
  theme_light() +
  xlab("NSM") + ylab("Appliances") +
  ggtitle("Variacao de Appliances por NSM")

# Media de Appliances por ligths
df %>% group_by(lights) %>%
  summarise(media = mean(Appliances)) %>%
  ggplot(aes(x = as.factor(lights), y = media)) +
  geom_bar(fill = 'blue', stat = 'identity') + 
  theme_light() +
  ggtitle("Media por ligths do atributo Appliances") +
  xlab("Ligths") + ylab("Media")

# Existem poucas observacoes e algumas com valores extremos com ligths acima de 30
df %>% group_by(lights) %>%
  summarise(contagem = n(),
            media = mean(Appliances))

# Ligths maior ou igual a 40 receberao valores de 40
# Appliances de ligths maiores que 40 receberao a media
df[df$lights > 40, "Appliances"] = mean(df$Appliances)
df[df$lights > 40, "lights"] = 40

# Media de Appliances por dia da semana
df %>% group_by(Day_of_week) %>%
  summarise(media = mean(Appliances)) %>%
  ggplot(aes(x = Day_of_week, y = media)) +
  geom_bar(fill = 'blue', stat = 'identity') + 
  theme_light() +
  ggtitle("Media por Day_of_week do atributo Appliances") +
  xlab("Day_of_week") + ylab("Media")

# Analisando as correlacoes

# Nao foi houve correlacao forte com o atributo target
# No entanto alguns atributos explicativos possuem correlacoes fortes entre eles e serao tratados por 
# selecao de atributos
correlacoes <- cor(df[,-c(1,30,31, 34)]) # desconsiderando a data e variaveis do tipo fator
corrplot(correlacoes, method = "circle")

# Excluindo o atributo date para preparar o modelo
df$date <- NULL

# Transformando o atributo preditor para log
df$Appliances <- log(df$Appliances)

### Feature Selection ###

# Criando modelo de random forest
modelo_rf <- randomForest(Appliances ~ ., data=df, importance=TRUE, ntree = 100)

# Plot das importancias
varImpPlot(modelo_rf)

# melhores atributos
atributos = c("lights", "NSM", "Day_of_week", "RH_5", "Visibility", "Press_mm_hg", "RH_9", "T8", "Tdewpoint",
              "RH_2", "RH_3", "RH_7",  "RH_4", "T7", "T4", "day", "T5", "T_out", "hour", "Windspeed")

# funncao para verificar colineariedade entre os atributos selecionados
colinear <- function(x){
  correlacoes <- cor(x)
  result_cor = data.frame()
  total = length(rownames(correlacoes))
  for (i in seq(1,total-1)){
    for (j in seq(i+1,total)){
      if (abs(correlacoes[i,j]) > 0.6) {
        result = data.frame("v1" = colnames(correlacoes)[j],
                            "v2" = rownames(correlacoes)[i],
                            "cor" = correlacoes[i,j])
        result_cor = rbind(result, result_cor)}
    }
  }
  return (result_cor)}

# Existe forte correlacao entre varios atributos
df_temp <-df[,atributos]
colinear(df_temp[,-3]) # Chamando a funcao desconsiderando o atributo dator day_of_week

# Nova selecao de atributos
atributos = c("Appliances", "lights", "NSM", "Day_of_week", "RH_5", "Visibility", "Press_mm_hg", "RH_9", "T8", "Tdewpoint",
              "day", "Windspeed")

# Dividindo os dados em treino e teste
train <- df[1:14803, atributos]
test <- df[14804:19735, atributos]

# Salvando os dados tratados
write.csv(train, "dados/train.csv", row.names = FALSE)
write.csv(test, "dados/test.csv", row.names = FALSE)

# lendo o conjunto de dados de treino e teste para construir o modelo
train = read.csv("dados/train.csv")
test = read.csv("dados/test.csv")

### Criando o modelo ###

# SVM
model_svm <- svm(Appliances ~ ., data = train, cost = 1, gamma = 0.05)

# Prevendo os dados de teste
pred <- predict(model_svm, test)

# Avaliando o modelo
postResample(pred, test$Appliances)

# Grid search com SVM
cost_ <- c(0.1, 10, 100)
gamma_ <- c(0.01, 0.25, 0.5, 1)

tuning_svm <- tune(svm,
                    Appliances ~., 
                    data = train, 
                    kernel = "radial", 
                    ranges = list(cost = cost_, gamma = gamma_))

# Melhores parametros
tuning_svm$best.parameters
#cost gamma
#10   0.5

# Prevendo os dados de teste
pred <- predict(tuning_svm$best.model, test)

# Avaliando o modelo
postResample(pred, test$Appliances)
#RMSE  Rsquared       MAE 
#0.4187478 0.5853214 0.2699297 


### Criando modelo com GBM ###

# Parametros de controle para validacao cruzada
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3)

# Criando um grid para selecao de hipermarametros
#gbmGrid <-  expand.grid(interaction.depth = c(8, 10, 12),
#                        n.trees = c(1000,1500, 3500), 
#                        shrinkage = c(0.1, 0.2, 0.3),
#                        n.minobsinnode = c(10, 20))

# Criando um modelo com GBM
#model_gbm <- train(Appliances ~., 
#                   data = train,
#                   method = "gbm", 
#                   trControl = fitControl, 
#                   verbose = FALSE,
#                   preProc = c("center", "scale"),
#                   tuneGrid = gbmGrid,
#                   bag.fraction=0.75)

# Melhores parametros
#interaction.depth = 12
#n.trees = 3500
#shrinkage = 0.1
#n.minobsinnode = 10

# Criando um modelo com GBM
set.seed(124)
model_gbm <- train(Appliances ~., 
                   data = train,
                   method = "gbm", 
                   trControl = fitControl, 
                   verbose = FALSE,
                   preProc = c("center", "scale"),
                   tuneGrid = data.frame(interaction.depth = 12,
                                         n.trees = 3500,
                                         shrinkage = 0.1,
                                         n.minobsinnode = 10),
                   bag.fraction=0.75)

# Prevendo os dados de teste
pred_2 <- predict(model_gbm, test)

# Avaliando o modelo
postResample(pred_2, test$Appliances)
#RMSE  Rsquared       MAE 
#0.3597249 0.6900402 0.2473768 

# Calculando e normalizando os residuos
residuos <- data.frame(scale(test$Appliances - pred_2, center = TRUE))
names(residuos) <- "residuos"
residuos$test <- test$Appliances


# Plotando os residuos
ggplot(residuos, aes(x = test, y = residuos)) +
  geom_point(color = "red") +
  geom_hline(yintercept=3, linetype="dotted", color = "black", size=1) + 
  geom_hline(yintercept=-3, linetype="dotted", color = "black", size=1) + 
  theme_light() +
  ggtitle("Residuos") +
  xlab("dados de teste") + ylab("residuos")
