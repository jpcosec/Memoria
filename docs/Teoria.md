### How transferable are features in deep neural networks?

Autores: Jason Yosinski, Jeff Clune, Yoshua Bengio, Hod Lipson

Año: 2014

En: NIPS 

KW: transfer learning, teoría

tldr: Estudio centrado en transferencia al inicializar redes. Al usar un modelo entrenado en una tarea como inicio del mismo modelo en otra tarea se nota:

* A mayor profundidad de la capa dentro de la red, mas especificos son los features. En las capas finales esto se corrige un poco, sin lograr los resultados de las primeras capas de todas maneras.
* El reentrenamiento de toda la red suele ser suficiente para mejorar cualquier problema de adaptacion.

Es un hecho conocido y facil de visualizar en las redes neuronales que las primeras capas toman formas parecidas a las de los filtros de Gabor o detectores de colores. Al medir las activaciones a medidas de profundiza la red se puede notar como estas tienden a activarse frente a partes de la imagen de entrada, buscando por ejemplo caras en humanos o texto en carteles. De tal manera es una afirmacion bastante compartida en el campo el hecho que las primeras capas tienden a ser mas generales entre tareas distintas, mientras que las capas mas profundas van especializandose mas a la tarea sobre la cual la red fue entrenada.

El paper parte desde esta premisa para realizar distintos estudios sobre la transferibilidad de las features en redes neuronales. Para esto comienzan por dividir las 1000 clases de ImageNet entre dos grupos A y B de 500 clases, luego, se entrena una red A de 8 capas convolucionales en el primer dataset y una B identica en el segundo.  Con esto se tienen 2 modelos, A y B entrenados sobre dos tareas distintas.

Luego, se toman las ultimas $n-1$ capas de cada red, se aleatorizan y entrenan sobre una de las particiones de los datasets. Una red que usa las primeras n capas del modelo entrenado sobre A y sobre el que se entrenan las ultimas $n-1$ capas sobre B se le llama AbB, si se entrenan todas las capas y no solo las ultimas $n-1$ se le llama AnB+. Notese que comparando los casos AnA y AnB se puede ver la diferencia entre un modelo en que se realiza transfer learning y otro en que no, de esta forma se puede tomar la diferencia de ambos.

Por ultimo, se aprovecha el hecho que ImageNet tiene una organizacion jerarquica y semántica entre sus clases para establecer una medida de distancia entre las distintas tareas. Así la forma mas distante es el entrenamiento entre elementos hechos por el hombre y elementos naturales y al avanzar por las ramas entre las que se separa el arbol jerarquico la distancia entre las tareas se acerca.




**Cuantificacion de especificidad de features según profundidad**

La primera contribucion importante es que encuentran una forma de cuantificar que tan transferible entre tareas distintas son los features de una red segun su profundidad dentro de la red base, randomizando la separacion entre los datasets A y B. 

Se nota que al reentrenar la red completa despues de transferir capas entre distintas tareas el performance final es generalmente mejor, independiente de la capa en que esto se realice. Por el contrario, al realizar transfer sin entrenar el performance disminuye de manera notoria al acercarse a las capas finales.

Extrañamente, al transferir pesos entre la misma tarea se nota un comportamiento consistente pero extraño,  al cortar en las redes intermedias (3,4,5, y en menor medida 6 de 8 capas) y reentrenar solo las ultimas capas el performance de la red disminuye tremendamente, cosa que no ocurre al cortar en las primeras y ultimas capas, ni al entrenar la red completa nuevamente. De tal forma afirman que este problema se debe a dificultades en la co-adaptacion de las capas iniciadas aleatoriamentes con respecto a sus capas vecinas, en otras palabras, se llegó a un punto en la optimizacion al que es muy dificil de llegar de manera aleatoria sin entrenar la red completa.



**Comparacion segun diferencia de task**

Se repite el experimento anterior sin realizar fine-tuning sobre la red completa comparando el decaimiento observado al realizar las particiones del dataset aleatorias con una particion del dataset entre imagenes naturales e imagenes de objetos hechos por el hombre y con una inicializacion aleatoria de los pesos de las primeras capas. 

Se logra notar que en el caso de los features aleatorios el accuracy decrece exponencialmente a partir de la segunda capa. Comparando las otras dos, la separacion aleatoria del dataset suele mostrar un ligero mejor comportamiento a partir de la segunda capa con respecto a la separacion entre imagenes naturales y de objetos hechos por el hombre, incrementandose la diferencia a medida aumenta la profundidad, llegando a una diferencia de $0.2$ en accuracy entre ambas en la capa 7. 

De esta manera se prueba por un lado que una separacion mayor entre las tasks entre las que se realiza la transferencia ocasiona un ligeramente peor resultado (sin fine tuning sobre toda la red de por medio) y que una inicializacion aleatoria de las primeras capas no es capaz de detectar correctamente patrones sobre la imagen.



### UNDERSTANDING DEEP LEARNING REQUIRES RE- THINKING GENERALIZATION

autores:  Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals

año: 2016

En:

Realizan varios tests para probar algunos conceptos heredados del machine learning antiguo dentro del contexto de deep learning, mas en particular, la concepcion sobre la generalizacion en modelos donde la cantidad de parametros es grande. En concreto se realizan tests de randomizacion con distintos tipos de regularizadores comunes en deep learning con lo que se logra llegar a las siguientes observaciones.

1. Las DNNs son capaces de ajustarse con facilidad a etiquetas *y* aleatorias, en base a lo que se concluye:
   1. La capacidad  efectiva de las redes neuronales es suficiente para memorizar un dataset completo.
   2. La optimizacion sobre etiquetas aleatorias solo aumenta una fraccion constante pequeña el tiempo de convergencia con respecto a las etiquetas reales.
   3. La aleatorizacion de la etiquetas solo representa una transformacion de los datos, ninguna otra propiedad del problema cambia.
2. Existen propiedades inherentes a la arquitectura de un modelo DNN y al entrenamiento usando SGD, batch normalization o early stopping que crean ciertas formas de regularizacion implícitas, además de otras tecnicas explicitas de regularización tales como la regularizacion $l_2$, el dropout y el data augmentation usadas desde los tiempos del ML.
3. En base a lo visto anteriormente se llega a que si bien la regularizacion explicita puede mejorar la generalizacion en un dataset, esta no es ni necesaria ni suficiente para cotrolar la generalizacion en general. En general si bien el uso de estas puede ayudar a mejorar el error en tests  algunos pocos puntos la mejor forma de arreglar estos problemas es mediante cambios en la arquitectura.
4. En general la expresividad de un modelo se mide en un espacio continuo, normalmente para una familia determinada de funciones en $(0,1)^d$. Proponen una forma de medir expresividad mas ad-hok a tareas sobre un dataset. Mas en concreto muestran que por ejemplo una red ReLU de 2 capas con $2n+d$ parametros es capaz de expresar cualquier etiquetado de tamaño $n$ en $d$ dimensiones. En una red de profundidad k, para cada capa basta con tener a lo mas $O(n/k)$ parámetros. 



### Generalization in Deep Learning [copiar bullshit a algun lado y redactar sin bullshit]

Autores: Kenji Kawaguchi, Leslie Pack Kaelbling,  Yoshua Bengio

Año: Mayo 2019

En:

Sean

$x \in X$ es un input

$y \in Y$ es un objetivo (etiqueta, valor)

$\mathcal{L}$ es una funcion de perdida (medida de la distancia de $f(x)$ la prediccion de nuestro modelo a $y$)

$\mathcal{R}[f]=\mathbb{E}_{x,y\sim \mathbb{P}(X,Y)} \left[ \mathcal{L}\left( f \left (x \right),y\right)\right]$ es el riesgo de una funcion (modelo) $f$ sobre una distribucion real $\mathbb{P}_{(X,Y)}$.

$f_{\mathcal{A}(S)}:\mathcal{X}\rightarrow \mathcal{Y}$ es un modelo aprendido por un algoritmo de aprendizaje $\mathcal{A}$ sobre un dataset de entrenamiento $S:=S_m$ de $m$ pares $(x_n,y_n)$.

$\mathcal{R}_S[f]=\frac{1}{m}\sum_{i=1}^m \mathcal{L}\left( f \left (x \right),y\right)$ es el riesgo empirico de la funcion de perdida sobre el dataset $S$.

$\mathcal{F}$ es un set de funciones dentro de una estructura o un espacio de hipotesis. Sea ademas $\mathcal{L}_\mathcal{F}= \left \{ g:f \in \mathcal{F}, g(x,y) = \mathcal{L}(f(x),y)\right\}$ una familia de funciones de perdidas asociadas a $\mathcal{F}$.

$d_v$ la dimensionalidad de la variable $v$

En general el fin en machine learning es  la minimizacion del riesgo esperado $\mathcal{R}[f_\mathcal{A}(S)]$, problema que se aborda minimizando la version empírica $\mathcal{R}_S[f_\mathcal{A}(S)]$ ya que la real no es computable. El objetivo del estudio sobre generalizacion  es el de explicar cuando la minimizacion de $\mathcal{R}_S[f_\mathcal{A}(S)]$ ocasiona una disminucion en $\mathcal{R}[f_\mathcal{A}(S)]$. Un acercamiento razonable a esto es el de ver la diferencia entre ambos $\textrm{generalization gap} = \mathcal{R}[f_\mathcal{A}(S)] -\mathcal{R}_S[f_\mathcal{A}(S)]$.

En teoria de aprendizaje estadístico, esta diferencia suele aproximarse mediante otras cantidades como la complejidad Rademacher [eq]  segun la cual es posible definir mediante [eq].

-o- Formas de límites de teoria estadistica al gap de generalizacion

complejidad rademacher depende de $\mathcal{F}$ 

Por otro lado, el acercamiento de la estabilidad depende de la estabilidad del algoritmo con respecto a distintos datasets

-o-

:todo:Definir generalizacion

:todo Remark 4 reescribir:La capacidad de generalizacion de un modelo $f$ y un dataset $S$ están totalmente determinados por la tupla $\mathbb{P}_{(X,Y)},S,f$ independiente de otros factores tales como el espacio de hipótesis $\mathcal{F}$ y sus propiedades como capacidad, complejidad rademacher, limite en normas. 

En otras palabras, la paradoja surge de que las herramientas de aprendizaje estadistico permiten establecer una implicancia entre estas propiedades y un limite en el gap de generalizacion. Sin embargo, esto no significa que todo modelo con pequeña capacidad de generalizacion deba cumplir con estas propiedades. 





