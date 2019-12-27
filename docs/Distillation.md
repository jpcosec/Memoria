



# Generalidades

Division de formas de compresion:
- Quantization
- Pruning
- Distilation

## Por que entrenar intentando ajustarse al conocimiento de una red grande en vez de usar una red pequeña



Una pregunta no trivial que puede salir a la palestra al momento de revisar investigaciones relativas a transfer learning y destilacion de modelo es el por que no simplemente entrenar la red desde 0, en vez de complicar el procedimiento usando otro modelo, en general mas pesado para esta tarea. 

En escencia una red neuronal fully connected de solo una capa debiese ser suficiente para poder aproximar cualquier funcion con un nivel de presicion arbitrario, controlado solo por la cantidad de parametros del modelo. En el caso de la clasificacion de datos reales, la tarea no deja de ser la misma, la aproximacion de una función hipotetica capaz de realizar el mapeo de los datos $x$ a sus etiquetas $y$. Sin embargo, un detalle fundamental que muchas veces se pasa por alto es que al momento de analizar datos naturales la cantidad de sampleos disponibles de esta funcion es limitada, cosa que se acrecienta con datasets obtenidos de manera local \footnote{No por trasnacionales con recursos como Google.}.

Antiguamente, gran parte del data science, procesamiento de imagenes incluido, consistia en diseñar y descubrir patrones dentro de los datos de entrada, cosa de reducir la dimensionalidad del problema y facilitar la tarea del clasificado. La novedad del Deep Learning consiste precisamente en esa capacidad de las redes de descubrir  esos patrones sin necesidad de usar conocimiento experto, solamente en base a los datos. Lamentablemente, dado el modo de entrenamiento de la red\footnote{Y la falta de significado semantico que permita dar sentido a estos datos.}, es un peligro siempre presente el sobre ajustarse a los datos sobre los que se entrenó, perdiendo la capacidad de prediccion sobre los datos reales. La forma en que se logra combatir este efecto actualmente es mediante la introduccion de artificios matematicos que permitan suavizar la funcion, permitiéndole predecir mejor en su vecindad; es decir, regularizando tales como

- Usar una cantidad del orden de cientos de miles de datos para aproximar una mayor generalidad antes que un subconjunto del dominio sobre el que se trabaja.
- Incrementar la capacidad de aprendizaje de la red  aumentando el numero de parámetros, y diseñando la estructura de la misma via uso de capas convolucionales, pooling, capas de sampleo, recurrencias, etc.
- Tecnicas de regularizacion  como data augmentation, normalizacion por batch, aleatorizacion de entrada y pesos, etc.

En definitiva, si bien un dataset de entrenamiento probablemente puede ser aproximado de forma facil y limpia por redes poco profundas de menos parametros, el ajustarse directamente a este dataset no necesariamente resuelve el problema (task) real. Por suerte, en el ajuste parametrico de un modelo hay información sobre el dataset o conocimiento que puede ser usado para adelantar, acelerar o mejorar el entrenamiento de otros. De manera similar a como los sistemas biologicos y en particular los seres humanos aprovechan el conocimiento de antepasados para aprender.

## Cosas que inluyen en velocidad de modelo

- Esparsidad (Prunning) en neuronas SSI se usan tecnicas especiales (buscar info)
- Esparcidad en filtros (Prunning de filtro completo)
- cuantizacion y aproximacion de bajo rango

## Notaciones

- Red tutora $T$, red estudiante (todo: buscar un mejor termino) $S$
- Feature map o salida de una capa $F \in \mathcal{R}^{C \times WH}$, se considerará el valor despues de la activación (no linealidad $\sigma(.)$). 
- De un canal $F^{k\cdot} \in \mathcal{R}^{WH}$, de una posicion $F^{\cdot k} \in \mathcal{R}^{C}$. 

# Papers Base



### UNIFYING DISTILLATION AND PRIVILEGED INFORMATION

año: feb 2016

- Informacion privilegiada de Vapnik es un método que permite usar dos estrategias para aprovechar un modelo svm tutor para entrenar un estudiante; control de similaridad, el cual permite acelerar el entrenamiento relajando la variable de holgura $\xi$ de manera controlada por una red tutora y transferencia de conocimiento, la cual funciona de manera similar a la destilacion.
- El paper basicamente propone un framework llamado destilacion generalizada, bajo el cual la tecnica descrita por cite:hinton_2015 e informacion privilegiada de vapnik son equivalentes.

## Distilling the Knowledge in a Neural Network

año: mar 2015

- Un detalle interesante en el que vale la pena detenerse para el caso del trabajo de Hinton es que este se centra exclusivamente en la imitación de los logits en la capa de salida de una red neuronal, aprovechando que típicamente en el caso de clasificación la última capa usa una activación Softmax que produce scores sobre las clases normalizados en 1.


- La red estudiante se entrena sobre la predicción de la red tutora a la vez que sobre los labels originales, para esto se vale de una combinación de las pérdidas de crossentropia de ambas usando una versión de softmax que incluye un valor \(T\) de temperatura.

$$q_i =\frac{exp \left ( z_i/T \right)}{\sum_j exp \left ( z_j/T \right)}$$

- Formulacion exclusiva para softmax. En caso de regresiones puede ser importante introducir normalizaciones en la perdida (recordar que el entrenamiento ocurre por gradiente) y en claso de clasificaciones binarias falta revisar.

## Learning Global Additive Explanations for Neural Nets Using Model Distillation

Año: dec 2018

Aplicación : 

Tipo: distillation, interpretabillity

abstract: Interpretability has largely focused on local explanations, i.e. explaining why a model made a particular prediction for a sample. These explanations are appealing due to their simplicity and local fidelity. However, they do not provide information about the general behavior of the model. We propose to leverage model distilla- tion to learn global additive explanations that describe the relationship between input features and model predictions. These global explanations take the form of feature shapes, which are more expressive than feature attributions. Through care- ful experimentation, we show qualitatively and quantitatively that global additive explanations are able to describe model behavior and yield insights about models such as neural nets. A visualization of our approach applied to a neural net as it is trained is available at https://youtu.be/ErQYwNqzEdc.

- Proponen usar destilacion para aprender explicaciones globales aditivas en la forma$$\hat{F}(x)=h_o + \sum_i h_i(xi) + \sum_{i \neq j} h_{i,j} (x_i,x_j) + \sum_{i\neq j}\sum_{j \neq k} h_{ijk} (x_i,x_j,x_k) + \cdots$$

Este tipo de modelos suelen aproximarse 

- Usan $\hat{F}$ para producir un modelo alternativo como un arbol de desiciones para predecir F



# Video distillation



### Back to the Future: Knowledge Distillation for Human Action Anticipation
año: apr 2019

abstract: We consider the task oftraining a neural network to anticipate human actions in video. This task is challenging given the complexity of video data, the stochastic nature of the future, and the limited amount of annotated train- ing data. In this paper, we propose a novel knowledge distillation framework that uses an action recognition net- work to supervise the training ofan action anticipation net- work, guiding the latter to attend to the relevant information needed for correctly anticipating the future actions. This framework is possible thanks to a novel loss function to ac- count for positional shifts ofsemantic concepts in a dynamic video. The knowledge distillation framework is a form of self-supervised learning, and it takes advantage of unlabeled data. Experimental results on JHMDB and EPIC- KITCHENS dataset show the effectiveness ofour approach.



### Paying More Attention to Motion: Attention Distillation for Learning Video Representations

año: apr 2019

abstract:We address the challenging problem of learning motion representations using deep models for video recognition. To this end, we make use of attention modules that learn to highlight regions in the video and aggregate features for recognition. Specifically, we propose to leverage out- put attention maps as a vehicle to transfer the learned rep- resentation from a motion (flow) network to an RGB net- work. We systematically study the design ofattention mod- ules, and develop a novel method for attention distillation. Our method is evaluated on major action benchmarks, and consistently improves the performance ofthe baseline RGB network by a significant margin. Moreover, we demon- strate that our attention maps can leverage motion cues in learning to identify the location ofactions in video frames. We believe our method provides a step towards learning motion-aware representations in deep models

- We provide the first systematic study of attention mechanisms for action recognition. We demonstrate that modeling attention as probabilistic variables can better facilitate the learning of deep model.

- We propose a novel method for learning motion-aware video presentations from RGB frames. Our method learns an RGB network that mimics the attention map of a flow network, thereby distilling important motion knowledge into the representation learning.

- Our method achieves consistent improvements of more than 1% across major datasets (UCF101 [41], HMDB51 [22] and 20BN-Something-Something [11, 28]) with almost no extra computational cost.

- Destilan movimiento desde una flow network

- Dificil de leer, técnica poco clara

  

### TKD: Temporal Knowledge Distillation for Active Perception

año: mar 2019

tip: video distillation, lstm

Abstract: Deep neural networks based methods have been proved to achieve outstanding performance on object detec- tion and classification tasks. Despite significant performance improvement, due to the deep structures, they still require prohibitive runtime to process images and maintain the highest possible performance for real-time applications. Observing the phenomenon that human vision system (HVS) relies heavily on the temporal dependencies among frames from the visual input to conduct recognition efficiently, we propose a novel framework dubbed as TKD: temporal knowledge distillation. This framework distills the temporal knowledge from a heavy neural networks based model over selected video frames (the perception of the moments) to a light-weight model. To en- able the distillation, we put forward two novel procedures: 1) an Long-short Term Memory (LSTM) based key frame selection method; and 2) a novel teacher-bounded loss design. To validate, we conduct comprehensive empirical evaluations using different object detection methods over multiple datasets including Youtube-Objects and Hollywood scene dataset. Our results show consistent improvement in accuracy-speed trad- offs for object detection over the frames of the dynamic scene, compare to other modern object recognition methods.

- Usan tiny-yolo como arquitectura estudiante
- Basicamente centran todo en entrenar tiny-yolo de manera "online", usando una lstm para detectar los frames interesantes (donde ocurren cambios importantes en la escena).
- No se ve nada muy interesante en este y esta mal escrito

## Destilacion en imagenes







## Transfer Learning

### A Survey on Deep Transfer Learning

año: agosto 2018

tipo: survey

Abstract. As a new classification platform, deep learning has recently received increasing attention from researchers and has been success- fully applied to many domains. In some domains, like bioinformatics and robotics, it is very difficult to construct a large-scale well-annotated dataset due to the expense of data acquisition and costly annotation, which limits its development. Transfer learning relaxes the hypothesis that the training data must be independent and identically distributed (i.i.d.) with the test data, which motivates us to use transfer learning to solve the problem of insufficient training data. This survey focuses on reviewing the current researches of transfer learning by using deep neural network and its applications. We defined deep transfer learning, category and review the recent research works based on the techniques used in deep transfer learning.

* [ ] Hacen una introduccion a una definicion formal al problema del transfer learning en contexto de deep learning. Luego separan taxonómicamente en 4 clases las tendencias que estan pasando actualmente.

* [ ] La definicion de transfer learning la analogan al aprovechamiento o adaptacion de un modelo entrenado en un par dominio-tarea particular a otro par dominio tarea.

  Un dominio se puede definir como $\mathcal{D} = \{ \chi,P(X) \}$ es decir un par compuesto por $\chi$, el espacio de características y $P(X)$, una probabilidad de distribucion sobre $X=\{x_1,\dots,x_n \} \in \chi$. Tomando como ejemplo el popular dataset MNIST, $\chi$ podria tratarse de todos los arreglos $\mathcal{b}^{28 \times 28} $  donde $ \mathcal{b}=\{0 \dots 255\} $ son todos los enteros representables en 8 bits, $X$ las 70.000 imagenes del dataset y $P(X)$ todos los digitos manuscritos que puedan ser escritos en esa resolución.

  Una tarea de aprendizaje (task) se puede definir como $\mathcal{T}=\{y,f(x)\}$, tambien consistente en dos partes, un espacio de etiquetas o clases $yh$ y una funcion objetivo $f(x)$, tambien describible como $P(y|x)$. Siguiendo el ejemplo de mnist, $y$ corresponde a los digitos del 0 al 9 y $f(x)$ a la prediccion de un modelo.

  Dadas estas definiciones, transfer learning corresponde a que dado una tarea *source* $\mathcal{T}_{s}$ razonablemente resuelta sobre un dominio *source* $\mathcal{D}_s$ y una tarea *target* $\mathcal{T}_t$ sobre un dominio target $\mathcal{D}_t$ no resueltas y donde ambos o uno de $\mathcal{D}_s \neq \mathcal{D}_t$ y  $\mathcal{T}_s \neq \mathcal{T}_t$ se cumplen, se hace uso del conocimiento latente del par *source* para mejorar el desempeño del par target. En el caso que hayan estructuras deep metidas entre medios, se puede hablar de deep transfer learning.

* [ ] La primera categoria descrita corresponde a habiendo cierta coincidencia entre los conjuntos $D_s$ y $D_t$, usar instancias ponderadas de estos para la prediccion objetivo, ejemplo de esto serían los algoritmos tipo ensemble. Otra se refiere al uso de parte de la red base que define $f_t(x)$ para extraer características que luego pueden clasificarse mediante un reentrenamiento de las capas no usadas o el uso de otros algoritmos. Finalmente quedan dos categorías, la primera referida al uso de funciones de mapeo entre uno y otro dominio para aprovechar el conocimiento de uno en otro y la segunda referida al uso de algoritmos adversariales para este  mismo fin.





## Otras aplicaciones

### KNOWLEDGE DISTILLATION FOR SMALL-FOOTPRINT HIGHWAY NETWORKS

Año: 2016

Tipo: Layer level distillation

Aplicacion: Audio

Abstract: Deep learning has significantly advanced state-of-the-art of speech recognition in the past few years. However, compared to conven- tional Gaussian mixture acoustic models, neural network models are usually much larger, and are therefore not very deployable in embed- ded devices. Previously, we investigated a compact highway deep neural network (HDNN) for acoustic modelling, which is a type of depth-gated feedforward neural network. We have shown that HDNN-based acoustic models can achieve comparable recognition accuracy with much smaller number of model parameters compared to plain deep neural network (DNN) acoustic models. In this pa- per, we push the boundary further by leveraging on the knowledge distillation technique that is also known as teacher-student training, i.e., we train the compact HDNN model with the supervision of a high accuracy cumbersome model. Furthermore, we also investigate sequence training and adaptation in the context of teacher-student training. Our experiments were performed on the AMI meeting speech recognition corpus. With this technique, we significantly im- proved the recognition accuracy of the HDNN acoustic model with less than 0.8 million parameters, and narrowed the gap between this model



- Aplican destilacion para un modelo compacto de neuronas parecido a LSTM.





# Bayesian Distillation

### Dropout Distillation

Año: 2016

La destilacion en este caso se centra en destilar una red bayesiana en la linea de lo expuesto por yarin gal en cite:gal2015; el uso de dropout al momento de realizar la inferencia sobre una red entrenada usando esa tecnica permite realizar sampleos sobre la version probabilistica de la red. De tal forma, usando una cantidad fija de sampleos se pueden estimar la media y varianza para regresion o la moda y algun otro estimador de dispersion para clasificación como medidas de la esperanza de la red original. Lo bueno es que esto mejora la inferencia y permite medir incerteza, lo malo es que requiere de una cantidad mayor de sampleos por inferencia (normalmente 10).

El paper se centra en la destilacion de la esperanza de la red bayesiana para poder realizar inferencia en tiempo real usando la esperanza en vez de estimar sobre una cantidad mayor de sampleos de la inferencia.

En su forma mas basica, esta tarea consiste en encontrar pesos de la red bayesiana $\hat{W}$ tal que el error de la inferencia sea el minimo. Esto se hace entrenando una red de la misma estructura de la red original usando divergencia KL entre ambas redes sobre la ultima capa. Esto en general permite mejorar la presicion de la red con respecto a la interpretación determinista de ensamble de redes, perdiendo de todas maneras el desempeño ante la interpretacion probabilista original.

### Zero-shot Knowledge Transfer via Adversarial Belief Matching

año: mayo 2019

- La gran mayoria de los modelos interesantes que se usan hoy en dia se entrenan via el uso de cantidades estratosfericas de datos, los cuales suelen ser de datasets privados. Razon por la cual entrenar una red via el uso del dataset original no es posible. El paper se centra en la destilacion del conocimiento de una red tutora sin datos (de ahi el zero shot).
- Toman una tecnica de GP llamada inducing point methods, que basicamente consiste en tomar puntos representativos del dataset, los cuales son usados para reducir la complejidad computacional all momento de la inferencia. En este contexto, wang introduce en cite:wang el uso de un metodo de destilacion de dataset para generar pseudoimagenes que cumplan el mismo objetivo. En la practica, los inducing points corresponden a sampleos de una red generadora entrenada con la red para maximizar la divergencia KL entre la red estudiante y el generador.
- Usan un generador, una red estudiante y una red tutora pre-entrenada. Suman a la perdida de la red estudiante el uso de una perdida que incluye attention maps en algunos bloques comunes de la red estudiante con la red tutoria, de la misma manera que las vistas en las de layer level distillation.
- La perdida para el generador es basicamente menos la divergencia KL entre la red estudiante y la red tutora. Para la red estudiante es la divergencia KL entre a red estudiante y la red tutora mas un termino de regularizacion sobre la atencion en las features de la red.
- Funciona relativamente bien en datasets pequeños del tipo cifar 10.



### Casos especiales

#### Knowledge Transfer with Jacobian Matching

Autores: Suraj Srinivas, Francois Fleuret

Año: 2018

En: ICML



Proponen un entrenamiento mediante matching de los jacobianos en alguna capa intermedia. El jacobiano se calcula con respecto al pixel de mayor intensidad dentro del espacio de entrada. Al parecer usan la red como un metodo de pre-inicializacion para finalmente entrenar con cross entropy. No tiene demasiado sentido la tecnica.

### Adversarial Distillation of Bayesian Neural Network Posteriors
año 2018

- Plantean destilacion como una forma de reducir el costo de producir samples en BNN. Usan GAN como una forma de exploracion MCMC sobre la distribucion de una red bayesiana. En otras palabras, usan GAN para aproximar la distribucion de la BNN original con una red generativa.
- Al parecer no funciona muy bien.
- No se entiende el objetivo

### KDGAN: Knowledge Distillation with Generative Adversarial Networks [MALO]

año: 2018

- Usan un gan en una simplificacion de la formulacion IR-GAN cite:wang-IRGAN, una adaptacion de GAN al contexto de information retrieval, es decir, recuperacion de un documento (o imagen) desde keywords o un caption. 
- Se centra en recomendacion de tags. No se profundizó en el documento.