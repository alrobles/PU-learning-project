library(tidyverse)
library(caret)
library(e1071)
library(kernlab)

#cargar datos y crear variable objetivo
datos <- readr::read_csv2("wine_quality.csv") %>%
  mutate_all(as.numeric) %>% 
  mutate(var_obj = ifelse(quality >= 7, "bueno", "malo") ) %>% 
  distinct()

train <- datos %>% distinct() %>% sample_frac(0.65) %>% select(-quality)
test <- anti_join(datos, train) %>% select(-quality)

minDataTrain <- train %>% count(var_obj) %>% pull(n) %>% min()
balancedTrain <- train %>% sample_n(minDataTrain)

#crear conjunto de datos balancedanceado 50% positivo -50% negativo
# svm no acepta pesos para las observaciones, asi que un submuestreo es necesario
#grid <- expand.grid(C=c(1:10), sigma = 0.07950468) # creo que este valor de sigma lo obutve con la funcion kernlab::sigest 
sigmas <- kernlab::sigest(var_obj~., data = test)
grid <- expand.grid(C = seq(1, 20, 2), sigma = sigmas)

svmNormal <- train(var_obj~.,
                   data = balancedTrain,
                   method = "svmRadial",
                   trControl = trainControl("cv", 
                                            2,
                                            classProbs = TRUE,
                                            summaryFunction = twoClassSummary),
                   metric = "ROC",
                   tuneGrid = grid)
plot(svmNormal)
n_pu = 500
pos <- datos %>% filter(var_obj == "bueno") %>% select(-quality)
neg <- datos %>% filter(var_obj == "malo") %>% select(-quality)

#crear datos de PU learning

P <- pos %>% 
  sample_n(n_pu) 

U <- anti_join(pos, P) %>% 
  full_join(neg) %>% 
  mutate(var_obj = "desconocido")
#generar 30-folds para validacion cruzada

#balancear U
#u_subsample <- U[sample( 1:nrow(U), nrow(P) ), ]
u_subsample <- U %>% sample_n(n_pu)
pu_train <- full_join(P, u_subsample)
n <- nrow(pu_train)

#validacion cruzada
#añadir folds a pop
pu_train <- pu_train %>%  
  mutate(pop = sample(1:10, n(), replace = TRUE))
model_formula = var_obj~fixed_acidity+volatile_acidity+citric_acid+residual_sugar+chlorides+free_sulfur_dioxide+total_sulfur_dioxide+density+pH+sulphates+alcohol

# estrategia:
#   1. Agrupar por Fold
#   2. Expandir Grid
#   (las columnas deben tener los mismos nombres que las categorias de entrenamiento) 
#   3. Entrenar para cada fold
#   4. Predecir con el resto de los datos
#   5. Calcular scores
#   6. Predecir con el mejor Score
#
pu_train_nested <- pu_train %>% 
  #arrange(pop) %>% x
  group_by(pop) %>% 
  nest() %>%
  mutate(values = map(data, function(x) select(x, var_obj) ) ) %>%  
  mutate(traindata = map(data, function(x) anti_join(pu_train, x) %>% select(-pop) ) ) %>% 
  tidyr::expand_grid(desconocido = 1:5, bueno = 2^(1:13) ) 

# Modelar
# Generar un data frame con un modelo por cada entrada
pu_train_nested <- pu_train_nested  %>% 
  mutate(model = pmap(.l = list(traindata, desconocido, bueno), function(x, y, z){
    modelSvm <- svm(model_formula,
      data = x, 
      type = 'C-classification', 
      kernel = 'radial', 
      class.weights = tibble(desconocido = y, bueno = z ) )
    return(modelSvm)
    })) 

# Predecir  
# Generar un data frame con las predicciones de cada modelo
pu_train_nested <- pu_train_nested %>% 
  mutate(predictions = pmap(.l = list(model, data),
                            function(x, y){ predict(x, y)}
                            )
         )

# Calcular F1score
# Generar las metricas de cada modelo
F1score <- function(confMat){
    #calcular r^2/p(f(x)=1)
    precision = confMat[2,2]/(confMat[2,2] + confMat[2,1] )
    recall = confMat[2,2]/(confMat[2,2] + confMat[1,2] )
    metrica = 2 * precision*recall/(precision+recall)
    return(metrica)
  }

pu_train_nested <- pu_train_nested %>% 
  mutate(F1Scores = pmap_dbl(.l = list(values, predictions),
                         function(x, y) {
                           tibble(var_obj_pred = y) %>% 
                             bind_cols(x) %>% 
                             table() %>% 
                             F1score()
                         }
                         ))
#metrica de caret
pu_train_nested <- pu_train_nested %>% 
  mutate(caretScore = pmap_dbl(.l = list(values, predictions),
                             function(x, y) {
                               tibble(var_obj_pred = y) %>% 
                                 bind_cols(x) %>% 
                                 table() %>% caret::F_meas()
                             }
  ))
# crear data frame con metricas de validacion cruzada
CV <- pu_train_nested %>% select(desconocido, bueno, F1Scores, caretScore) %>% 
  rename(ks = desconocido, js = bueno )

# encontrar los mejores hiperparametros
# max(CV$F1Scores)
# Si hay mas de uno entonces sólo tomar uno al azar
best_params <- CV %>% 
  na.exclude() %>% 
  filter(F1Scores == max(F1Scores)) %>% 
  sample_n(1)
#c_0=1, c_1=2
#entrenar el mejor modelo

PU.svm <- svm(
  formula = model_formula,
  data = select(pu_train, -pop), 
  type = 'C-classification', 
  kernel = 'radial', 
  class.weights = tibble(bueno = 2^best_params$js,
                            desconocido = best_params$ks),
  probability = TRUE)

#comparar AUC de ambos modelos 
library(ROCR)
pred.pu <- prediction( attr(predict(PU.svm, test,probability=TRUE),"prob")[,2],test$var_obj)
perf.pu <- performance(pred.pu,"tpr","fpr")
roc.pu <- tibble( ejeX = perf.pu@x.values %>% unlist,
                  ejeY = perf.pu@y.values %>% unlist) %>% 
  mutate(Metodo = "Biased SVM")

pred <- prediction( predict(svmNormal, test,type="prob")[,2], test$var_obj)
perf <- performance(pred,"tpr","fpr")
roc.normal <- tibble( ejeX = perf@x.values %>% unlist,
                      ejeY = perf@y.values %>% unlist) %>% 
  mutate(Metodo = "Standard SVM") 

rocsvm  <- full_join(roc.pu, roc.normal)

graficaroc <- rocsvm %>% 
  ggplot(aes(x = ejeX, y = ejeY, col = Metodo)) +
  geom_line(size = 1.5) + 
  xlab("1-especificidad")+
  ylab("sensibilidad") +
  theme_classic() + 
  theme(legend.position = c(0.75, 0.25))

graficaroc
