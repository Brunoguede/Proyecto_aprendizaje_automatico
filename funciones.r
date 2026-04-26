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
  
  # Clases posibles (en nuestro caso 0 y 1)
  clases <- unique(y)
  
  # Probabilidad a priori de cada clase
  prob_clases <- table(y) / length(y)
  
  # Lista donde guardaremos los parámetros de cada variable para cada clase
  parametros <- list()
  
  for (col in colnames(X)) {
    
    parametros[[col]] <- list()
    
    for (clase in clases) {
      
      valores <- X[y == clase, col]
      valores <- valores[!is.na(valores)]
      
      if (length(unique(valores)) <= 2) {
        
        # Suavizado de Laplace: sumamos 1 al numerador y 2 al denominador
        # Evita probabilidades de 0 o 1 exactos
        n1 <- sum(valores == 1) + 1
        n0 <- sum(valores == 0) + 1
        prob <- n1 / (n1 + n0)
        parametros[[col]][[as.character(clase)]] <- list(tipo = "binaria", prob = prob)
        
      } else {
        
        media <- mean(valores, na.rm = TRUE)
        desv  <- max(sd(valores, na.rm = TRUE), 0.1)  # mínimo 0.1 para evitar desv=0
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
    
    predicciones[i] <- as.numeric(names(which.max(scores)))
  }
  
  return(predicciones)
}

# =============================================================
# K-NN (K VECINOS MÁS CERCANOS)
# =============================================================
# Función para PREDECIR con k-NN
# k-NN no tiene fase de entrenamiento como tal - simplemente
# memoriza todos los datos y compara cuando llega un caso nuevo
# Recibe:
#   - X_train: dataframe con los pacientes de entrenamiento
#   - y_train: vector con las clases de entrenamiento
#   - X_test: dataframe con los pacientes a predecir
#   - k: número de vecinos a considerar
# Devuelve un vector con la predicción (0 o 1) para cada paciente
# =============================================================

predecir_knn <- function(X_train, y_train, X_test, k = 3) {
  
  # Vector donde guardaremos las predicciones
  predicciones <- c()
  
  # Recorremos cada paciente nuevo que queremos predecir
  for (i in 1:nrow(X_test)) {
    
    # Calculamos la distancia entre este paciente y TODOS los del train
    # Usamos distancia euclídea: raíz cuadrada de la suma de diferencias al cuadrado
    distancias <- c()
    
    for (j in 1:nrow(X_train)) {
      
      # Cogemos los valores de ambos pacientes
      paciente_nuevo <- as.numeric(X_test[i, ])
      paciente_train <- as.numeric(X_train[j, ])
      
      # Ignoramos las posiciones donde alguno tiene valor missing
      posiciones_validas <- !is.na(paciente_nuevo) & !is.na(paciente_train)
      
      # Calculamos distancia euclídea solo con posiciones válidas
      diff <- paciente_nuevo[posiciones_validas] - paciente_train[posiciones_validas]
      distancias[j] <- sqrt(sum(diff^2))
    }
    
    # Ordenamos las distancias de menor a mayor y cogemos los k primeros índices
    indices_vecinos <- order(distancias)[1:k]
    
    # Miramos las clases de esos k vecinos
    clases_vecinos <- y_train[indices_vecinos]
    
    # La predicción es la clase más frecuente entre los k vecinos
    # table() cuenta cuántos hay de cada clase
    # which.max() devuelve el índice del máximo
    predicciones[i] <- as.integer(names(which.max(table(clases_vecinos))))
  }
  
  return(predicciones)
}

# =============================================================
# LDA (ANÁLISIS DISCRIMINANTE LINEAL)
# =============================================================
# LDA busca la combinación lineal de variables que mejor separa
# las clases. Asume que cada clase sigue una distribución normal
# con la misma matriz de covarianza.
# Recibe:
#   - X: dataframe con las variables predictoras
#   - y: vector con la variable objetivo (0 o 1)
# Devuelve una lista con los parámetros aprendidos
# =============================================================

entrenar_lda <- function(X, y) {
  
  # Convertimos X a matriz numérica para poder operar matemáticamente
  X <- as.matrix(X)
  
  # Clases posibles
  clases <- unique(y)
  
  # Número total de observaciones
  n <- nrow(X)
  
  # Número de variables
  p <- ncol(X)
  
  # Calculamos la media global de todas las variables
  media_global <- colMeans(X, na.rm = TRUE)
  
  # Para cada clase calculamos su media
  medias <- list()
  for (clase in clases) {
    # Cogemos solo las filas de esta clase
    X_clase <- X[y == clase, ]
    medias[[as.character(clase)]] <- colMeans(X_clase, na.rm = TRUE)
  }
  
  # Calculamos la matriz de covarianza dentro de los grupos (Within-class)
  # Es la varianza conjunta de todas las clases
  # Inicializamos con ceros
  SW <- matrix(0, nrow = p, ncol = p)
  
  for (clase in clases) {
    X_clase <- X[y == clase, ]
    # Centramos los datos restando la media de la clase
    X_centrada <- sweep(X_clase, 2, medias[[as.character(clase)]], "-")
    # Reemplazamos NAs por 0 para no perder filas enteras
    X_centrada[is.na(X_centrada)] <- 0
    # Sumamos la contribución de esta clase a SW
    SW <- SW + t(X_centrada) %*% X_centrada
  }
  
  # Dividimos por n-número de clases para obtener la covarianza media
  SW <- SW / (n - length(clases))
  
  # Añadimos una pequeña regularización para evitar matriz singular
  # Una matriz singular no se puede invertir, lo que daría error
  SW <- SW + diag(1e-4, p)
  
  # Calculamos la inversa de SW
  SW_inv <- solve(SW)
  
  # Probabilidades a priori de cada clase
  prob_clases <- table(y) / length(y)
  
  # Devolvemos todo lo aprendido
  return(list(
    medias = medias,
    SW_inv = SW_inv,
    prob_clases = prob_clases,
    clases = clases
  ))
}

# =============================================================
# Función para PREDECIR con LDA
# Recibe:
#   - modelo: lo que devolvió entrenar_lda()
#   - X_nuevo: dataframe con los pacientes a predecir
# Devuelve un vector con la predicción (0 o 1) para cada paciente
# =============================================================

predecir_lda <- function(modelo, X_nuevo) {
  
  # Convertimos a matriz
  X_nuevo <- as.matrix(X_nuevo)
  
  # Reemplazamos NAs por la media de cada columna
  for (col in 1:ncol(X_nuevo)) {
    na_pos <- is.na(X_nuevo[, col])
    if (any(na_pos)) {
      X_nuevo[na_pos, col] <- mean(X_nuevo[, col], na.rm = TRUE)
    }
  }
  
  predicciones <- c()
  
  for (i in 1:nrow(X_nuevo)) {
    
    scores <- c()
    
    for (clase in modelo$clases) {
      
      # Media de esta clase
      mu <- modelo$medias[[as.character(clase)]]
      
      # Diferencia entre el paciente y la media de la clase
      diff <- X_nuevo[i, ] - mu
      
      # Puntuación discriminante de Fisher
      # Cuanto menor es esta distancia, más probable es esta clase
      score <- -0.5 * t(diff) %*% modelo$SW_inv %*% diff +
        log(modelo$prob_clases[as.character(clase)])
      
      scores[as.character(clase)] <- score
    }
    
    # Predecimos la clase con mayor puntuación
    predicciones[i] <- as.integer(names(which.max(scores)))
  }
  
  return(predicciones)
}

# =============================================================
# BAGGING
# =============================================================
# Bagging entrena múltiples árboles de decisión, cada uno sobre
# una muestra aleatoria con reemplazamiento del dataset.
# La predicción final es la votación mayoritaria de todos los árboles.
# Necesitamos la librería rpart para los árboles de decisión.
# Recibe:
#   - X: dataframe con las variables predictoras
#   - y: vector con la variable objetivo (0 o 1)
#   - n_arboles: número de árboles a entrenar (por defecto 100)
# Devuelve una lista con todos los árboles entrenados
# =============================================================

entrenar_bagging <- function(X, y, n_arboles = 100) {
  
  # Cargamos rpart para poder entrenar árboles de decisión
  library(rpart)
  
  # Lista donde guardaremos todos los árboles entrenados
  arboles <- list()
  
  # Número total de pacientes en el train
  n <- nrow(X)
  
  # Entrenamos n_arboles árboles distintos
  for (i in 1:n_arboles) {
    
    # Creamos una muestra aleatoria CON reemplazamiento
    # sample() con replace=TRUE permite que un paciente salga varias veces
    indices_muestra <- sample(1:n, size = n, replace = TRUE)
    
    # Cogemos los pacientes de esta muestra
    X_muestra <- X[indices_muestra, ]
    y_muestra <- y[indices_muestra]
    
    # Juntamos X e y en un dataframe para rpart
    datos_muestra <- X_muestra
    datos_muestra$target <- as.factor(y_muestra)
    
    # Entrenamos un árbol de decisión con esta muestra
    # method="class" indica que es un problema de clasificación
    arbol <- rpart(target ~ ., data = datos_muestra, method = "class")
    
    # Guardamos el árbol en la lista
    arboles[[i]] <- arbol
  }
  
  return(list(arboles = arboles))
}

# =============================================================
# Función para PREDECIR con Bagging
# Recibe:
#   - modelo: lo que devolvió entrenar_bagging()
#   - X_nuevo: dataframe con los pacientes a predecir
# Devuelve un vector con la predicción (0 o 1) para cada paciente
# =============================================================

predecir_bagging <- function(modelo, X_nuevo) {
  
  library(rpart)
  
  # Número de pacientes a predecir
  n <- nrow(X_nuevo)
  
  # Número de árboles
  n_arboles <- length(modelo$arboles)
  
  # Matriz donde cada fila es un paciente y cada columna es un árbol
  # Guardaremos la predicción de cada árbol para cada paciente
  votos <- matrix(0, nrow = n, ncol = n_arboles)
  
  # Recorremos cada árbol
  for (i in 1:n_arboles) {
    
    # Predicción de este árbol para todos los pacientes
    # type="class" devuelve la clase directamente (0 o 1)
    pred <- predict(modelo$arboles[[i]], newdata = X_nuevo, type = "class")
    
    # Guardamos las predicciones como números
    votos[, i] <- as.integer(as.character(pred))
  }
  
  # Para cada paciente calculamos la votación mayoritaria
  # rowMeans nos da la proporción de árboles que votaron 1
  # Si más del 50% votaron 1, predecimos 1, si no predecimos 0
  predicciones <- ifelse(rowMeans(votos) >= 0.5, 1, 0)
  
  return(predicciones)
}

# =============================================================
# ADABOOST
# =============================================================
# AdaBoost entrena árboles de decisión secuencialmente.
# Cada árbol se centra más en los casos que el anterior falló.
# La predicción final es una votación ponderada — los árboles
# que aciertan más tienen más peso en la decisión final.
# Recibe:
#   - X: dataframe con las variables predictoras
#   - y: vector con la variable objetivo (0 o 1)
#   - n_arboles: número de árboles a entrenar (por defecto 50)
# Devuelve una lista con los árboles y sus pesos
# =============================================================

entrenar_adaboost <- function(X, y, n_arboles = 50) {
  
  library(rpart)
  
  # AdaBoost trabaja con clases -1 y +1 en vez de 0 y 1
  # Convertimos: 0 → -1, 1 → +1
  y_ab <- ifelse(y == 1, 1, -1)
  
  # Número de pacientes
  n <- nrow(X)
  
  # Inicializamos los pesos — todos igual de importantes al principio
  # Cada paciente tiene peso 1/n
  pesos <- rep(1/n, n)
  
  # Listas donde guardaremos los árboles y sus pesos
  arboles <- list()
  alfas   <- c()
  
  for (i in 1:n_arboles) {
    
    # Juntamos X, y y los pesos en un dataframe para rpart
    datos <- X
    datos$target <- as.factor(y_ab)
    
    # Entrenamos un árbol muy simple (maxdepth=1 significa solo una pregunta)
    # Los pesos le dicen al árbol en qué pacientes fijarse más
    # Un árbol de profundidad 1 se llama "stump" o tocón
    arbol <- rpart(target ~ ., data = datos, method = "class",
                   weights = pesos,
                   control = rpart.control(maxdepth = 1))
    
    # Predicción de este árbol para todos los pacientes
    pred <- as.integer(as.character(predict(arbol, newdata = X, type = "class")))
    
    # Calculamos el error ponderado
    # Es la suma de pesos de los pacientes que falló
    error <- sum(pesos[pred != y_ab])
    
    # Si el error es 0 o mayor que 0.5 paramos
    # Error=0 significa predicción perfecta
    # Error>0.5 significa que el árbol es peor que el azar
    if (error <= 0 || error >= 0.5) break
    
    # Calculamos el peso de este árbol (alfa)
    # Cuanto menor es el error, mayor es alfa (más peso en la votación)
    alfa <- 0.5 * log((1 - error) / error)
    
    # Actualizamos los pesos de los pacientes
    # Los que falló → su peso aumenta (multiplicamos por e^alfa)
    # Los que acertó → su peso disminuye (multiplicamos por e^-alfa)
    pesos <- pesos * exp(-alfa * y_ab * pred)
    
    # Normalizamos los pesos para que sumen 1
    pesos <- pesos / sum(pesos)
    
    # Guardamos el árbol y su peso
    arboles[[i]] <- arbol
    alfas[i]     <- alfa
  }
  
  return(list(arboles = arboles, alfas = alfas))
}

# =============================================================
# Función para PREDECIR con AdaBoost
# Recibe:
#   - modelo: lo que devolvió entrenar_adaboost()
#   - X_nuevo: dataframe con los pacientes a predecir
# Devuelve un vector con la predicción (0 o 1) para cada paciente
# =============================================================

predecir_adaboost <- function(modelo, X_nuevo) {
  
  library(rpart)
  
  # Número de pacientes a predecir
  n <- nrow(X_nuevo)
  
  # Vector donde acumulamos la votación ponderada
  # Empieza en 0 para cada paciente
  votos <- rep(0, n)
  
  # Recorremos cada árbol
  for (i in 1:length(modelo$arboles)) {
    
    # Predicción de este árbol (-1 o +1)
    pred <- as.integer(as.character(
      predict(modelo$arboles[[i]], newdata = X_nuevo, type = "class")
    ))
    
    # Sumamos la predicción ponderada por el peso del árbol (alfa)
    # Un árbol con alfa alto influye más en la decisión final
    votos <- votos + modelo$alfas[i] * pred
  }
  
  # Si la suma de votos es positiva predecimos 1 (familiar)
  # Si es negativa predecimos 0 (esporádico)
  predicciones <- ifelse(votos > 0, 1, 0)
  
  return(predicciones)
}

# =============================================================
# GRID SEARCH CON VALIDACIÓN CRUZADA
# =============================================================
# Busca el mejor hiperparámetro k para k-NN usando validación
# cruzada de 5 folds. Para cada valor de k divide el train en
# 5 partes, entrena con 4 y evalúa con 1, rotando 5 veces.
# El k con mejor accuracy medio es el seleccionado.
# Recibe:
#   - X_train: dataframe con las variables predictoras
#   - y_train: vector con la variable objetivo
#   - ks: vector de valores de k a probar
#   - n_folds: número de particiones (por defecto 5)
# Devuelve una tabla con el accuracy medio de cada k
# =============================================================

grid_search_knn <- function(X_train, y_train, ks = c(1,3,5,7,9,11,15), n_folds = 5) {
  
  # Número total de pacientes en train
  n <- nrow(X_train)
  
  # Creamos los índices de cada fold aleatoriamente
  # Dividimos los pacientes en n_folds grupos
  set.seed(42)
  indices_fold <- sample(rep(1:n_folds, length.out = n))
  
  # Dataframe donde guardaremos los resultados
  resultados <- data.frame(k = ks, accuracy_medio = NA)
  
  # Probamos cada valor de k
  for (idx in 1:length(ks)) {
    
    k <- ks[idx]
    accuracies <- c()
    
    # Validación cruzada — rotamos el fold de evaluación
    for (fold in 1:n_folds) {
      
      # Los pacientes de este fold son el conjunto de validación
      idx_val   <- which(indices_fold == fold)
      # El resto son el conjunto de entrenamiento
      idx_train <- which(indices_fold != fold)
      
      # Partimos los datos
      X_tr  <- X_train[idx_train, ]
      y_tr  <- y_train[idx_train]
      X_val <- X_train[idx_val, ]
      y_val <- y_train[idx_val]
      
      # Predecimos con este k
      preds <- as.integer(predecir_knn(X_tr, y_tr, X_val, k = k))
      y_val <- as.integer(y_val)
      
      # Guardamos el accuracy de este fold
      accuracies[fold] <- mean(preds == y_val)
    }
    
    # Guardamos el accuracy medio de todos los folds para este k
    resultados$accuracy_medio[idx] <- mean(accuracies)
    cat("k =", k, "-> Accuracy medio CV:", round(mean(accuracies) * 100, 2), "%\n")
  }
  
  # Mostramos el mejor k
  mejor_k <- resultados$k[which.max(resultados$accuracy_medio)]
  cat("\nMejor k:", mejor_k, "\n")
  
  return(resultados)
}