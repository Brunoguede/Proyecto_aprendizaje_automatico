
# ARCHIVO DE FUNCIONES - APRENDIZAJE AUTOMÁTICO I
# En este archivo implementamos desde cero los siguientes modelos:
#   - Naïve Bayes
#   - k-NN (k vecinos más cercanos)
#   - LDA (Análisis Discriminante Lineal)
#   - Bagging
#   - AdaBoost
#   - Grid Search con validación cruzada para k-NN


# 1. NAIVE BAYES
# La idea es calcular la probabilidad de que un paciente sea
# familiar o esporádico dado sus síntomas, usando el teorema
# de Bayes.
# El entrenamiento consiste en aprender las probabilidades de
# cada síntoma para cada clase a partir de los datos de entrenamiento.
# Para variables binarias (0/1) guardamos la proporción de 1s.
# Para variables continuas (edades) guardamos media y desviación típica.


entrenar_naive_bayes <- function(X, y) {
  
  # sacamos las clases que hay, en nuestro caso 0 y 1
  clases <- unique(y)
  
  # calculamos cuántos casos hay de cada clase respecto al total
  prob_clases <- table(y) / length(y)
  
  # aquí guardaremos los parámetros aprendidos de cada variable
  parametros <- list()
  
  # recorremos cada variable (columna) del dataset
  for (col in colnames(X)) {
    
    parametros[[col]] <- list()
    
    # para cada clase calculamos sus parámetros por separado
    for (clase in clases) {
      
      # cogemos los valores de esta variable solo para esta clase
      valores <- X[y == clase, col]
      valores <- valores[!is.na(valores)]  # quitamos los NA
      
      # si la variable es binaria (solo tiene 0s y 1s)
      if (length(unique(valores)) <= 2) {
        
        # usamos suavizado de Laplace: sumamos 1 arriba y 2 abajo
        # esto evita que alguna probabilidad sea exactamente 0
        n1 <- sum(valores == 1) + 1
        n0 <- sum(valores == 0) + 1
        prob <- n1 / (n1 + n0)
        parametros[[col]][[as.character(clase)]] <- list(tipo = "binaria", prob = prob)
        
      } else {
        
        # si es continua (como las edades) guardamos media y desv típica
        # con estos dos valores podemos calcular la densidad normal después
        media <- mean(valores, na.rm = TRUE)
        desv  <- max(sd(valores, na.rm = TRUE), 0.1)  # mínimo 0.1 para que no sea 0
        parametros[[col]][[as.character(clase)]] <- list(tipo = "continua", media = media, desv = desv)
        
      }
    }
  }
  
  # devolvemos todo lo que hemos aprendido
  return(list(prob_clases = prob_clases, parametros = parametros, clases = clases))
}


# Esta función usa lo que aprendió entrenar_naive_bayes para predecir
# nuevos pacientes. Para cada paciente calcula un score por clase
# y se queda con la clase de mayor score.

predecir_naive_bayes <- function(modelo, X_nuevo) {
  
  predicciones <- c()
  
  # recorremos cada paciente que queremos predecir
  for (i in 1:nrow(X_nuevo)) {
    
    scores <- c()
    
    for (clase in modelo$clases) {
      
      # empezamos el score con la probabilidad a priori de la clase
      # usamos log para evitar que los números se hagan demasiado pequeños
      # al multiplicar muchas probabilidades entre sí
      score <- log(modelo$prob_clases[as.character(clase)])
      
      # vamos sumando la contribución de cada síntoma
      for (col in names(modelo$parametros)) {
        
        valor <- X_nuevo[i, col]
        
        # si el valor falta lo ignoramos directamente
        if (is.na(valor)) next
        
        params <- modelo$parametros[[col]][[as.character(clase)]]
        
        if (params$tipo == "binaria") {
          
          # si el síntoma está presente usamos su prob, si no usamos 1-prob
          # el +0.001 es para evitar log(0) en caso de prob=0 o prob=1
          p <- ifelse(valor == 1, params$prob + 0.001, 1 - params$prob + 0.001)
          score <- score + log(p)
          
        } else {
          
          # para variables continuas usamos la densidad de la normal
          # dnorm nos da qué tan probable es ese valor dado media y desv
          p <- dnorm(valor, mean = params$media, sd = params$desv + 0.001)
          score <- score + log(p + 0.001)
          
        }
      }
      
      scores[as.character(clase)] <- score
    }
    
    # nos quedamos con la clase que tiene mayor puntuación
    predicciones[i] <- as.numeric(names(which.max(scores)))
  }
  
  return(predicciones)
}


# 2. K-NN (K VECINOS MÁS CERCANOS)
# k-NN no aprende ninguna fórmula, simplemente memoriza todos
# los pacientes de entrenamiento. Cuando llega uno nuevo,
# busca los k más parecidos y vota la clase más frecuente.
# La similitud se mide con distancia euclídea: cuanto más
# parecidos los síntomas, menor es la distancia.
# No hay función de entrenamiento porque no hay nada que aprender,
# solo se usan los datos directamente en la predicción.


predecir_knn <- function(X_train, y_train, X_test, k = 3) {
  
  predicciones <- c()
  
  # para cada paciente nuevo calculamos su distancia a todos los del train
  for (i in 1:nrow(X_test)) {
    
    distancias <- c()
    
    for (j in 1:nrow(X_train)) {
      
      paciente_nuevo <- as.numeric(X_test[i, ])
      paciente_train <- as.numeric(X_train[j, ])
      
      # ignoramos las variables donde alguno de los dos tiene NA
      posiciones_validas <- !is.na(paciente_nuevo) & !is.na(paciente_train)
      
      # distancia euclídea: raíz de la suma de diferencias al cuadrado
      diff <- paciente_nuevo[posiciones_validas] - paciente_train[posiciones_validas]
      distancias[j] <- sqrt(sum(diff^2))
    }
    
    # ordenamos por distancia y cogemos los k más cercanos
    indices_vecinos <- order(distancias)[1:k]
    
    # miramos qué clase tienen esos k vecinos
    clases_vecinos <- y_train[indices_vecinos]
    
    # la predicción es la clase más repetida entre los vecinos
    predicciones[i] <- as.integer(names(which.max(table(clases_vecinos))))
  }
  
  return(predicciones)
}


# 3. LDA (ANÁLISIS DISCRIMINANTE LINEAL)
# LDA busca la dirección que mejor separa las dos clases.
# Durante el entrenamiento calcula el "paciente típico" de cada
# clase (su media) y cómo de dispersos están los pacientes
# alrededor de ese típico (matriz de covarianza within-class SW).
# Para predecir, mira a cuál de los dos típicos se parece más
# el paciente nuevo, ajustando por la dispersión de cada clase.
# Asume que los datos siguen una distribución normal.


entrenar_lda <- function(X, y) {
  
  # necesitamos trabajar con matrices para hacer álgebra lineal
  X <- as.matrix(X)
  
  clases <- unique(y)
  n <- nrow(X)  # número de pacientes
  p <- ncol(X)  # número de variables
  
  # esta variable no se usa en la predicción pero es útil tenerla
  media_global <- colMeans(X, na.rm = TRUE)
  
  # calculamos la media de cada clase (el "paciente típico" de cada grupo)
  medias <- list()
  for (clase in clases) {
    X_clase <- X[y == clase, ]
    medias[[as.character(clase)]] <- colMeans(X_clase, na.rm = TRUE)
  }
  
  # calculamos SW: la matriz de covarianza dentro de los grupos
  # mide cómo de dispersos están los pacientes alrededor de su media de clase
  SW <- matrix(0, nrow = p, ncol = p)
  
  for (clase in clases) {
    X_clase <- X[y == clase, ]
    # centramos restando la media de la clase
    X_centrada <- sweep(X_clase, 2, medias[[as.character(clase)]], "-")
    X_centrada[is.na(X_centrada)] <- 0  # los NA los tratamos como 0
    SW <- SW + t(X_centrada) %*% X_centrada
  }
  
  # dividimos para obtener la covarianza media entre clases
  SW <- SW / (n - length(clases))
  
  # añadimos un valor pequeño en la diagonal para que SW sea invertible
  # sin esto puede dar error si hay variables muy correlacionadas
  SW <- SW + diag(1e-4, p)
  
  # invertimos SW, necesaria para calcular la puntuación discriminante
  SW_inv <- solve(SW)
  
  # probabilidad a priori de cada clase
  prob_clases <- table(y) / length(y)
  
  return(list(
    medias = medias,
    SW_inv = SW_inv,
    prob_clases = prob_clases,
    clases = clases
  ))
}


predecir_lda <- function(modelo, X_nuevo) {
  
  X_nuevo <- as.matrix(X_nuevo)
  
  # los NA los sustituimos por la media de cada columna
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
      
      mu <- modelo$medias[[as.character(clase)]]
      
      # diferencia entre el paciente y el centro de esta clase
      diff <- X_nuevo[i, ] - mu
      
      # puntuación discriminante de Fisher
      # mide qué tan cerca está el paciente del centro de esta clase
      # ajustando por la dispersión (SW_inv)
      # sumamos log de la prob a priori igual que en Naïve Bayes
      score <- -0.5 * t(diff) %*% modelo$SW_inv %*% diff +
        log(modelo$prob_clases[as.character(clase)])
      
      scores[as.character(clase)] <- score
    }
    
    # la clase con mayor puntuación es la predicción
    predicciones[i] <- as.integer(names(which.max(scores)))
  }
  
  return(predicciones)
}


# 4. BAGGING
# Bagging soluciona el overfitting de los árboles de decisión
# entrenando muchos árboles distintos y haciéndoles votar.
# Para conseguir árboles distintos usamos bootstrap: para cada
# árbol creamos una muestra aleatoria CON reemplazamiento de los
# datos de entrenamiento. Así cada árbol ve datos ligeramente
# distintos y comete errores distintos. Al votar juntos, los
# errores individuales se cancelan entre sí.


entrenar_bagging <- function(X, y, n_arboles = 100) {
  
  library(rpart)  # necesitamos rpart para los árboles de decisión
  
  arboles <- list()
  n <- nrow(X)
  
  for (i in 1:n_arboles) {
    
    # muestra bootstrap: cogemos n pacientes al azar CON repetición
    # algunos pacientes saldrán varias veces y otros no saldrán
    indices_muestra <- sample(1:n, size = n, replace = TRUE)
    
    X_muestra <- X[indices_muestra, ]
    y_muestra <- y[indices_muestra]
    
    # rpart necesita que X e y estén en el mismo dataframe
    datos_muestra <- X_muestra
    datos_muestra$target <- as.factor(y_muestra)
    
    # entrenamos el árbol con esta muestra
    # method="class" porque es clasificación, no regresión
    arbol <- rpart(target ~ ., data = datos_muestra, method = "class")
    
    arboles[[i]] <- arbol
  }
  
  return(list(arboles = arboles))
}


predecir_bagging <- function(modelo, X_nuevo) {
  
  library(rpart)
  
  n <- nrow(X_nuevo)
  n_arboles <- length(modelo$arboles)
  
  # matriz de votos: fila = paciente, columna = árbol
  votos <- matrix(0, nrow = n, ncol = n_arboles)
  
  for (i in 1:n_arboles) {
    
    # cada árbol hace su predicción
    pred <- predict(modelo$arboles[[i]], newdata = X_nuevo, type = "class")
    votos[, i] <- as.integer(as.character(pred))
  }
  
  # si más del 50% de los árboles votan 1, predecimos 1
  # rowMeans calcula la proporción de votos a 1 por paciente
  predicciones <- ifelse(rowMeans(votos) >= 0.5, 1, 0)
  
  return(predicciones)
}


# 5. ADABOOST
# AdaBoost también entrena múltiples árboles pero de forma
# secuencial, no independiente como Bagging.
#
# La clave está en los pesos: al principio todos los pacientes
# tienen el mismo peso. Después de cada árbol, los pacientes
# que se clasificaron mal ganan más peso para que el siguiente
# árbol se centre más en ellos.
# La predicción final es una votación ponderada: los árboles
# que acertaron más tienen más voz en la decisión final.


entrenar_adaboost <- function(X, y, n_arboles = 50) {
  
  library(rpart)
  
  # AdaBoost necesita clases -1 y +1 en vez de 0 y 1
  # la fórmula de actualización de pesos lo requiere así
  y_ab <- ifelse(y == 1, 1, -1)
  
  n <- nrow(X)
  
  # todos los pacientes empiezan con el mismo peso: 1/n
  pesos <- rep(1/n, n)
  
  arboles <- list()
  alfas   <- c()  # pesos de cada árbol en la votación final
  
  for (i in 1:n_arboles) {
    
    datos <- X
    datos$target <- as.factor(y_ab)
    
    # usamos árboles de profundidad 1 (solo hacen una pregunta)
    # se llaman "stumps" y son los clasificadores débiles típicos de AdaBoost
    # los pesos le dicen al árbol en qué pacientes fijarse más
    arbol <- rpart(target ~ ., data = datos, method = "class",
                   weights = pesos,
                   control = rpart.control(maxdepth = 1))
    
    pred <- as.integer(as.character(predict(arbol, newdata = X, type = "class")))
    
    # error ponderado: suma de pesos de los pacientes mal clasificados
    error <- sum(pesos[pred != y_ab])
    
    # paramos si el árbol es perfecto o peor que el azar
    if (error <= 0 || error >= 0.5) break
    
    # alfa: el peso de este árbol en la votación final
    # cuanto menor es el error, mayor es alfa
    alfa <- 0.5 * log((1 - error) / error)
    
    # actualizamos los pesos de los pacientes:
    # los que falló → peso mayor (e^alfa > 1)
    # los que acertó → peso menor (e^-alfa < 1)
    pesos <- pesos * exp(-alfa * y_ab * pred)
    
    # normalizamos para que los pesos sumen 1
    pesos <- pesos / sum(pesos)
    
    arboles[[i]] <- arbol
    alfas[i]     <- alfa
  }
  
  return(list(arboles = arboles, alfas = alfas))
}


predecir_adaboost <- function(modelo, X_nuevo) {
  
  library(rpart)
  
  n <- nrow(X_nuevo)
  
  # acumulamos la suma ponderada de votos para cada paciente
  votos <- rep(0, n)
  
  for (i in 1:length(modelo$arboles)) {
    
    # predicción de este árbol en escala -1/+1
    pred <- as.integer(as.character(
      predict(modelo$arboles[[i]], newdata = X_nuevo, type = "class")
    ))
    
    # sumamos ponderando por alfa: los mejores árboles influyen más
    votos <- votos + modelo$alfas[i] * pred
  }
  
  # suma positiva → más votos a familiar (1)
  # suma negativa → más votos a esporádico (0)
  predicciones <- ifelse(votos > 0, 1, 0)
  
  return(predicciones)
}


# 6. GRID SEARCH CON VALIDACIÓN CRUZADA PARA K-NN
# Para elegir el mejor k en k-NN usamos validación cruzada
# de 5 folds en lugar de evaluarlo solo una vez en test.
# La validación cruzada divide el train en 5 partes iguales.
# En cada iteración usa 4 partes para entrenar y 1 para validar,
# rotando qué parte se usa para validar. El accuracy final de
# cada k es la media de las 5 evaluaciones, lo que da una
# estimación más fiable que una sola partición.
# Grid search simplemente prueba todos los valores de k que
# le pasamos y se queda con el que da mejor accuracy medio.


grid_search_knn <- function(X_train, y_train, ks = c(1,3,5,7,9,11,15), n_folds = 5) {
  
  n <- nrow(X_train)
  
  # asignamos cada paciente a un fold aleatoriamente
  set.seed(42)
  indices_fold <- sample(rep(1:n_folds, length.out = n))
  
  resultados <- data.frame(k = ks, accuracy_medio = NA)
  
  for (idx in 1:length(ks)) {
    
    k <- ks[idx]
    accuracies <- c()
    
    # rotamos el fold de validación 5 veces
    for (fold in 1:n_folds) {
      
      # el fold actual es la validación, el resto es entrenamiento
      idx_val   <- which(indices_fold == fold)
      idx_train <- which(indices_fold != fold)
      
      X_tr  <- X_train[idx_train, ]
      y_tr  <- y_train[idx_train]
      X_val <- X_train[idx_val, ]
      y_val <- y_train[idx_val]
      
      preds <- as.integer(predecir_knn(X_tr, y_tr, X_val, k = k))
      y_val <- as.integer(y_val)
      
      accuracies[fold] <- mean(preds == y_val)
    }
    
    # guardamos el accuracy medio de los 5 folds para este k
    resultados$accuracy_medio[idx] <- mean(accuracies)
    cat("k =", k, "-> Accuracy medio CV:", round(mean(accuracies) * 100, 2), "%\n")
  }
  
  mejor_k <- resultados$k[which.max(resultados$accuracy_medio)]
  cat("\nMejor k:", mejor_k, "\n")
  
  return(resultados)
}

