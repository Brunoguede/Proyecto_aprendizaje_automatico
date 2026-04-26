# =============================================================
# NAIVE BAYES
# =============================================================
# Función para ENTRENAR el modelo Naïve Bayes
# Recibe:
#   - X: dataframe con las variables predictoras (los síntomas)
#   - y: vector con la variable objetivo (0 o 1)
# Devuelve una lista con todo lo aprendido del dataset
# =============================================================

entrenar_naive_bayes <- function(X, y) {
  
  clases <- unique(y)
  
  prob_clases <- table(y) / length(y)
  
  parametros <- list()
  
  for (col in colnames(X)) {
    
    parametros[[col]] <- list()
    
    for (clase in clases) {
      
      valores <- X[y == clase, col]
      
      if (length(unique(na.omit(valores))) <= 2) {
        
        prob <- mean(valores, na.rm = TRUE)
        parametros[[col]][[as.character(clase)]] <- list(tipo = "binaria", prob = prob)
        
      } else {
        
        media <- mean(valores, na.rm = TRUE)
        desv  <- sd(valores, na.rm = TRUE)
        parametros[[col]][[as.character(clase)]] <- list(tipo = "continua", media = media, desv = desv)
        
      }
    }
  }
  
  return(list(prob_clases = prob_clases, parametros = parametros, clases = clases))
}

# =============================================================
# Función para PREDECIR con Naïve Bayes
# Recibe:
#   - modelo: lo que devolvió entrenar_naive_bayes()
#   - X_nuevo: dataframe con los pacientes a predecir
# Devuelve un vector con la predicción (0 o 1) para cada paciente
# =============================================================

predecir_naive_bayes <- function(modelo, X_nuevo) {
  
  predicciones <- c()
  
  for (i in 1:nrow(X_nuevo)) {
    
    scores <- c()
    
    for (clase in modelo$clases) {
      
      score <- log(modelo$prob_clases[as.character(clase)])
      
      for (col in names(modelo$parametros)) {
        
        valor <- X_nuevo[i, col]
        
        if (is.na(valor)) next
        
        params <- modelo$parametros[[col]][[as.character(clase)]]
        
        if (params$tipo == "binaria") {
          
          p <- ifelse(valor == 1, params$prob + 0.001, 1 - params$prob + 0.001)
          score <- score + log(p)
          
        } else {
          
          p <- dnorm(valor, mean = params$media, sd = params$desv + 0.001)
          score <- score + log(p + 0.001)
          
        }
      }
      
      scores[as.character(clase)] <- score
    }
    
    predicciones[i] <- as.integer(names(which.max(scores)))
  }
  
  return(predicciones)
}