# Resumen

## Capacidad

### Aproximacion Continua

Existen estudios sobre su capacidad o expresividad en funciones. [Cyb89] muestra en 1989 que una red de una capa oculta sin limite de neuronas de activacion sigmoideas aproxima cualquier funcion continua con un error arbitrario, cosa extendida al caso multicapa con cualquier activacion no constante en Hor91.

Telgarsky (2015) compara Relus shallow y deep llegando a que una red de $\Theta(k^3)$ capas de $\Theta(1)$ unidades contiene mas información que una de orden $\Theta(K)$ a menos que cada una de sus capas, tenga al menos $\Theta(2^k)$ unidades.



En el caso de las actualmente mas populares activaciones ReLU hay 2 resultados interesantes que vale la pena ver. Hanin estudió la capacidad en funciones dentro del cubo $[0,1]^d$ centrandose primero en el ancho minimo  $w_\text{min}(d)$ y luego en la profundidad.

- Sin cota maxima de profundidad, cualquier funcion en el cubo es aproximable con un error arbitrario si el ancho de capa es al menos $w_\text{min}(d)\leq d+2$ 
  - En el caso de convexas positivas bastan  $d+1$ neuronas. 
  - Y si la funcion es afin por parte, puede ser representada exactamente con d+3 neuronas de ancho.

- Seteando un ancho cercano a $w_\text{min}(d)$, cualquier funcion que es afin por piezas con $N$ partes puede ser representada exactamente por una ReLU de ancho $d+1$ y profundidad $N$.

En torno al ancho necesario para representar una funcion Lu tiene 2 teoremas bastante fuertes.

- Teorema de aproximacion universal para redes ReLU de ancho acotado**

  Cualquier funcion Lebesge integrable en $\mathbb{R}^d \rightarrow \mathbb{R}$ puede ser aproximada por una red fully connected de activacion ReLU y de ancho $d+4$ con un error arbitrario con respecto a la distancia L1. 

- **Borde minimo polinomial de eficiencia**

  Para un entero $k$, existe una red ReLU de ancho del orden $O(k^2)$ y profundidad $2$ que no puede ser aproximada por ninguna red de ancho de orden $O(k^{1.5})$ y profundidad $k$. 



### Expresividad según cantidad de piezas de aproximacion lineal

Al ser una red basada en ReLU una composicion de funciones lineales por piezas, este es una funcion lineal por piezas. El modelo que crea ReLU es entonces una aproximación lineal por piezas de la funcion que se intenta imitar. De esta manera, es posible ver que una forma de medir la expresividad de una red contando la cantidad de aproximaciones. 

- Para un modelo deep de $n_0$ $O(1)$ entradas y $k$ capas ocultas de ancho $n$, el numero maximo de regiones de aproximacion tiene la siguiente cota $\Omega\left (\left \lfloor \frac{n}{n_0}  \right \rfloor ^{k-1}  \frac{n^{n_o-2} }{k} \right)$.

- Para un modelo shallow de similar cantidad de entradas y $kn$ unidades en una sola capa, el valor es del orden $O(k^{n_0-1}n^{n_o-1})$.

Un detalle importante que notar es que siempre que hayan mas neuronas que unidades en la entrada la cantidad de regiones va a crecer exponencialmente con la cantidad de capas.

### Sampleo discreto

En general la expresividad de un modelo se mide en un espacio continuo, normalmente para una familia determinada de funciones. En Zhang Bengio proponen una forma de medir expresividad mas ad-hok a tareas sobre un dataset. Mas en concreto, para cualquier dataset de tamaño $n$ en $d$ dimensiones:

* Una red ReLU de 2 capas con $2n+d$ puede aprender cualquier etiquetado.
* Una red de profundidad $k$  necesitará  $O(n/k)$ parámetros. 

### Generalizacion

En *Understanding Deep Learning Requires Rethinking Generalization* (2017), entrenaron de manera efectiva DNNs sobre datasets con distintos niveles de aleatorizacion, llegando a las siguientes conclusiones.

1. Las DNNs son capaces de ajustarse con facilidad a etiquetas *y* aleatorias, en base a lo que se concluye.

2. Hay dos fuentes de regularizacion en DL

   1. **Implicitas** como aquellas inherentes a la arquitectura de un modelo DNN, al entrenamiento usando SGD, batch normalization, early stopping etc.
   2. **Explicitas**  como la regularizacion $l_2$, el dropout y el data augmentation usadas desde los tiempos del ML.

3. Las predicciones de límites clasicos de generalización no funcionan tan bien, en general una mayor cantidad de parametros implica una mejor generalizacion, sin embargo las CNNs modernas tienen demasiados parámetros para la capacidad de generalizacion que tienen.

4. La regularizacion explicita puede mejorar la generalizacion en un dataset, esta no es ni necesaria ni suficiente para controlar la generalizacion en general. En general si bien el uso de estas puede ayudar a mejorar el error en tests  algunos pocos puntos la mejor forma de arreglar estos problemas es mediante cambios en la arquitectura.

   

   

### En resumen

1. Las redes neuronales en general son capaces de aproximar cualquier función o mareo.

2. Con un MLP de 2 capas es posible memorizar cualquier dataset de n samples con un ancho de $2n+d$, para aproximar cualquier funcion Lebesge integrable en $\mathbb{R}^d \rightarrow \mathbb{R}$ basta un MLP ReLU de ancho $d+4$ . 

3. La cantidad de regiones de aproximacion lineal aumenta de forma exponencial según aumenta la cantidad de parametros (y la profundidad de la red).

   Dibujo de over-fitting

### Que es lo que aprende la red

FEATURES

## Transfer learning

# Papers

## Capacidad

### Benefits of depth in neural networks

Año: 2016

Autor: Matus Telgarsky

En:

En resumen, existen redes deep con del orden de $\Theta(k^3)$ capas de $\Theta(1)$ unidades y $\Theta(1)$ parametros, no aproximables por una de profundidad $\Theta(K)$ a menos que cada una de sus capas sean exponencialmente mas grandes, teniendo al menos $\Theta(2^k)$ unidades por capa.

### On the number of response regions of deep feed forward networks with piece-wise linear activations

Año: 2013

Autores: Razvan Pascanu, Guido Montufar, Yoshua Bengio

En: ICLR

En el paper se estudia la flexibilidad de un modelo deep con respecto a uno shallow contando el numero de regiones lineales que definen sobre la entrada para un numero fijo de unidades. I.E, el numero de piezas de aproximacion lineal que tiene disponible el modelo para aproximar alguna region no linear de funciones. Algunos detalles importantes del analisis son que:

* Para aproximar de manera perfecta el borde entre 2 clases un mlp necesita usar infinitas regiones lineales, entonces, es lógico usar esta cantidad como una medida de la flexibilidad del modelo.
* Estas regiones de activacion no son independientes unas de otras, en un modelo deep hay una correlacion entre las regiones, cosa derivada de el hecho que se comparten parámetros. Esto mismo permite que el modelo sea capaz de generalizar mejor.
* En una red de una capa capa neurona separa el espacio de entrada en 2. En el caso multicapa cada capa realiza una especie de operacion or que se propaga hacia adelante.

Partiendo del de Zalavsky que dice que para un set finito de $m$ hiperplanos en un espacio $n_o$-dimensional común (arreglo $\mathcal{A}$ ), la cantidad total de regiones es $r(\mathcal{A})=\sum_{s=0}^{n_0}\binom{m}{s}$ y mientras que la cantidad de regiones delimitadas es $b(\mathcal{A})=\binom{m-1}{n_0}$ se logra llegar a que para un modelo mlp de una sola capa oculta, con $n_0$ entradas, $n_1$ unidades en la capa oculta está dado por el arreglo creado de $n_1$ hiperplanos en el espacio $n_0$ dimensional $\mathcal{R}(n_0,n_1,n_y)=\sum_{j=0}^{n_0}\binom{n_1}{j}$.

* Para un modelo deep de $n_0$ $O(1)$ entradas y $k$ capas ocultas de ancho $n$, el numero maximo de regiones de respuesta por parámetros se comporta como

$$\Omega\left (\left \lfloor \frac{n}{n_0}  \right \rfloor ^{k-1}  \frac{n^{n_o-2} }{k} \right)$$

* Para un modelo shallow de similar cantidad de entradas y $kn$ unidades en una sola capa, el valor es del orden.

$$O(k^{n_0-1}n^{n_o-1})$$ 

Se puede ver primero que un modelo deep puede representar potencialmente mas regiones.



### The expressive Power of Neural Networks: a View from the width

Año: 2017

Autores: Zhou Lu, Hongming Pu, Feicheng Wang, Zhiqiang Hu, Liwei Wang,

En: NIPS

Partiendo de los siguientes trabajos que prueban existencias.

- 5 que prueba que una red de 3 capas puede representar mas que una de 2 capas y tamaño sub-exponencial
- 2 prueba que ciertas cases de redes deep convolucionales pueden no ser representadas por redes shallow de un tamaño menor a uno exponencial.
- 15 prueba que existen redes de tamaño $k^3$ que no son representables por redes del orden $O(k)$ de tamaño menor a $2^k$.

A diferencia de estos que prueban existencias, proveen 2 contribuciones interesantes

* **Teorema de aproximacion universal para redes ReLU de ancho acotado**

  Cualquier funcion Lebesge integrable en $\mathbb{R}^n \rightarrow \mathbb{R}$ puede ser aproximada por una red fully connected de activacion ReLU y de ancho $n+4$ con un error arbitrario con respecto a la distancia L1. 

* **Borde minimo de eficiencia polinomial**

  Para un entero $k$, existe una red ReLU de ancho del orden $O(k^2)$ y profundidad $2$ que no puede ser aproximadoa por ninguna red del orden $O(k^{1.5})$ y ancho $k$.



### UNIVERSAL FUNCTION APPROXIMATION BY DEEP NEURAL NETS WITH BOUNDED WIDTH AND RELU ACTIVATIONS

autor: Boris Hanin

año: 2017

En:

En el paper se responden las siguientes preguntas.

* ¿Cual es el ancho minimo $w_\text{min}(d)$ de capa para que una red ReLU de profundidad arbitraria pueda aproximar cualquier funcion continua en el cubo $[0,1]^d$ con un error arbitrario?



R. $w_\text{min}(d)\leq d+2$ o $d+1$ si la funcion es convexa positiva. Si la funcion es afin por parte se puede probar que que esta puede ser representada exactamente por una red ReLu de a lo mas d+3 neuronas de ancho.

* Para redes ReLU de un ancho de capa cercano a $w_\text{min}(d)$, cual es la profundidad necesaria para aproximar esa funcion?

En general esto depende de la cantidad de pieas afines que esta contiene. Cualquier funcion que es afin por piezas con $N$ piezas puede ser representada exactamente poruna ReLU de ancho $d+1$ y profundidad $N$.



###  Expressiveness of Rectifier Networks [No tan interesante]

Autores: Xingyuan Pan, Vivek Srikumar

Año: 2016

En:

Buscan responder que funciones booleanas expresan las redes ReLU y compararlas con las funciones threshold. Prueban que:

1. Una red ReLU de 2 capas es equivalente a redes threshold exponencialmente mas grandes y que existen redes ReLU de 2 capas irrepresentables por ninguna funcion threshold mas pequeña.

2. Basado en lo anterior prueban que una red ReLU es logaritmicamente comprimible en una red ReLU mas pequeña.

   

## Generalizacion

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



## Features

### How transferable are features in deep neural networks?

Autores: Jason Yosinski, Jeff Clune, Yoshua Bengio, Hod Lipson

Año: 2014

En: NIPS 

KW: transfer learning, teoría

tldr: Estudio centrado en transferencia al inicializar redes. Al usar un modelo entrenado en una tarea como inicio del mismo modelo en otra tarea se nota:

- A mayor profundidad de la capa dentro de la red, mas especificos son los features. En las capas finales esto se corrige un poco, sin lograr los resultados de las primeras capas de todas maneras.
- El reentrenamiento de toda la red suele ser suficiente para mejorar cualquier problema de adaptacion.

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

