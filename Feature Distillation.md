

# Layer/ feature level



Dado el problema planteado se pasará a detallar algunas investigaciones que comparten en comun el hacer uso de los features para transferencia de conocimiento.

### A Comprehensive Overhaul of Feature Distillation 

Año: 2019

Autores: Byeongho Heo, Jeesoo Kim, Sangdoo Yun, Hyojin Park, Nojun Kwak, Jin Yor Choi

En:

Definen feature distillation de manera generalizada como un setting de transfer learning en el cual se usan las features de la red tutora para el entrenamiento de la red estudiante. Definiendo una medida de distancia $d$ y una función de adaptacion dimensional para cada red $T_T$, $T_S$, las cuales son aplicadas sobre las features de cada red $F_T$ y $F_S$, la perdida de cualquier destilacion de features puede ser definida de la siguiente forma:

$$\mathcal{L}_{FD}= d \left[ T_T \left( F_T\right),T_S \left( F_S\right)\right]$$

Además, definen toda perdida en terminos de transformada en los terminos expuestos, es decir, las transformaciones $T_T$ y $T_S$ y la distancia $d$ a lo que agregan una caracteristica de perdida de información, como por ejemplo las dimensiones de canal en Attention Transfer o el valor de los features en activation boundary. 

En funcion de esto define un metodo de destilacion basado en el trabajo anterior del mismo autor,  Activation boundaries con la diferencia que en vez de binarizar la expresion de ReLU como en el caso anterior, usa una modificacion llamada Margin Relu $\sigma_{\textrm{MReLU }}=max(x,m)$, siendo m un valor negativo. En esta version setean el margen como al esperanza del valor de las respuestas negativas sobre todas las imagenes, el cual puede ser calculado online en el entrenamiento o usando el los parametros de la ultima batch normalization. Esto es usado en las features de la red tutora, mientras que la red estudiante usa una adaptacion de convoluciones de 1x1 y batch normalization.

Además, como funcion de distancia se propone una distancia L2 parcial, que transfiere el conocimiento de la red tutora para una posicion i-esima en el tensor de activacion en caso que la respuesta del estudiante sea menor que la del tutor y menor que 0.

$$d_p(T,S)=\sum^{WHC}_i\begin{cases}
 0& \text{ si } F_{S_i}  \leq F_{Ti} \leq 0 \\ 
\left ( F_{Ti}-F_{S_i}  \right )& \text{ en cualquier otro caso } 
\end{cases}$$



### Distilling Object Detectors with Fine-grained Feature Imitation

Año: 2019

Autores: Tao Wang, Li Yuan, Xiaopeng Zhang, Jiashi Feng

En: CVPR

El estudio se centra en destilacion en el contexto de deteccion de objetos. Ya que a diferencia de la clasificion, en la cual se obtiene una sola etiqueta de clase para toda la imagen, en deteccion el objetivo es diferenciar los objetos del fondo y clasificar cada uno por separado (paralela o secuencialmente). De partida esto introduce dos diferencias importantisimas con respecto a la clasificacion, las cuales tienen consecuencias importantes al momento de destilar;

1. Los fondos en las imagenes pueden ser mucho mas variados que los objetos de primer plano, de esta manera en deteccion gran parte de la informacion de la imagen es o inutil o ruido que puede condicionar al detector a obtener respuestas falsas.
2. La destilacion propuesta por hinton debe ser completamente modificada ya que la etiqueta deja de ser solo un valor de clase, si no de bounding box y clase por cada objeto presente en la imagen.





### An Embarrassingly Simple Approach for Knowledge Distillation

Año: 2019

Autores: Mengya Gao, Yujun Shen, Quanquan Li, Junjie Yan, Liang Wan, Dahua Lin, Chen Change Loy, Xiao Tang 

En: Arxiv, aun no publicado.

En general en los aprendizajes multi-task es necesario usar varias perdidas distintas, cada una de ellas ponderada por un hiperparametro $\lambda$ para ponderar cuanto afecta cada una de las perdidas al momento de realizar backpropagation. La eleccion de estos hiperparametros puede ser una tarea dificil, especialmente considerando que el valor optimo suele cambiar de dataset a dataset sin otra forma de encontrarlo que el uso de varias instancias de prueba y error. El paper postula una manera simple y eficiente de evitar este problema desacoplando las etapas de aprendizaje en el caso de destilacion de features.

En general, la mayoria de las redes puede separarse entre una etapa de extraccion de features llamada backbone y una de clasificacion llamada task-head. Como se puede ver en How transferable are features in deep neural networks? estas suelen ser bastante independientes del task, cosa que se aprovecha en este paper para separar los entrenamientos en una etapa de aprendizaje "greedy" de las features de una red tutora y una etapa de aprendizaje de las labels finales usando ground truth y el backbone fijo.

Para la primera etapa se hace uso de un entrenamiento bloque por bloque, es decir, se toma cierta profundidad de capa a reproducir y se reproducen las features de la red tutora en la red estudiante, luego se fijan los pesos en la red estudiante y se aprenden los features a una profundidad un poco mayor y asi sucesivamente hasta destilar el backbone completo. Finalmente se reproduce la ultima capa usando solo la red estudiante y el ground truth del task a entrenarse. El aprendijaze de los features se realiza usando una perdida de distancia L2 entre los features de la red tutora y estudiante.

$$L_{SSKD}= \|f_t^2-f_s^2 \|_2$$

### Learning Student Networks via Feature Embedding

Año: 2018

Autores: Hanting Chen, Yunhe Wang, Chang Xu, Chao Xu and Dacheng Tao

En: IEEE (submited)

Los autores realizan transferencia del conocimiento de los features de una manera un poco distinta a la vista hasta el momento ya que lejos de intentar acercar la respuesta de la red estudiante a la de la red tutora según alguna forma, buscan replicar la distancia relativa entre las respuestas de la red estudiante a dos samples $x_j$,$x_{i}$ del dataset arbitrarios. De esta forma introducen una perdida llama local preserving loss, la cual toma la siguiente forma:

$$\mathcal{L}_{\text{LP}}=\frac{1}{m}\sum_{i,j}\alpha_{i,j} \|f_{s}^i-f_s^j\|_2^2$$

Donde $m$ es el tamaño del minibatch, y $\alpha_{i,j}$ es un coeficiente que toma un valor positivo en caso que las features del sample $j$ en la red tutora, $f_T^j$ sea el vecino mas cercano de las features del sample $i$, $f_T^i$ y toma valor 0 en cualquier otro caso, es decir:

$$\alpha_{i,j}=\left\{\begin{matrix}
 exp(\|f_{T}^i-f_T^j\|_2^2) & \text{si } j \in N(i) \\ 
 0 &  \text{en otro caso}
\end{matrix}\right.$$





### FD-GAN: Pose-guided Feature Distilling GAN for Robust Person Re-identification

Año: 2018

Autores:  Yixiao Ge, Zhuowan Li, Haiyu Zhao, Guojun Yin, Shuai Yi, Xiaogang Wang, Hongsheng Li

En: Nips

El problema de la re-identificacion de personas consiste en la comparacion de alguna representacion de dos imagenes para dilucidar si se trata del mismo sujeto. Para esto suele hacerse uso de una arquitectura llama red siamesa, la cual basicamente consiste en instanciar una red dos o mas veces y entrenar reduciendo la distancia entre las representaciones obtenidas de las distintas instancias para una misma persona o alejandola para personas distintas. 

El paper mezcla 4 tasks distintas en una misma arquitectura siamesa para lograr un aprendizaje de una representacion consistente de identidad entre individuos, desacoplada de la pose (orientacion de cuerpo y posicion de extremidades con respecto al cuerpo). La base es un encoder basado en resnet que recibe una imagen, una pose y ruido.  Sobre esta cual se acopla un generador que crea imagenes falsas del sujeto en la pose especificada. Estas imagenes son luego pasadas por un discriminador de pose y de identidad para comparar con el dato de entrada. La parte siamesa viene por replicar toda esta arquitectura usando dos imagenes de entrada que pueden tratarse de un mismo sujeto o no. Al momento de evaluar se usa solamente el encoder basado en resnet.

El resultado es bastante bueno, sin embargo la tecnica no corresponde mucho al concepto de destilacion como hemos visto en el resto de los  ya que en vez de usarse una red estudiante para aprender lo sabido por una red tutora lo que se hace es sumar mas tasks a una estructura ya entrenada para realizar fine-tuning. De todas maneras el trabajo es bastante cercano y hace uso de transfer learning, junto con features de algo ya entrenado. 



### Like What You Like: Knowledge Distill via Neuron Selectivity Transfer 

Año: DEC 2017

Autores: Zehao Huang, Naiyan Wang

En: ICLR

En NST interpretan el mapa de activacion de cada posicion como si fuera un sampleo de como la red neuronal interpreta la imagen de entrada, con esto se puede ver en que se centra la red neuronal para realizar la deteccion. Basado en esto evitan hacer un match directo de los feature maps ya que esto ignora la densidad de sampleo en el espacio de las features de la red tutora. En vez de eso busca realizar un alineamiento de las distribuciones de la red tutora y estudiante.  

Definen entonces el conocimiento selectivo de las neuronas como la activacion de cada neurona para un patron particular encontrado en la entrada $X$ bajo un una tarea particular.  Desde esto el metodo propuesto es neural selectivity transfer, o la transferencia de este conocimiento. Para el entrenamiento se usan dos perdidas distintas, una para los feature maps y otra para la clasificacion. Clasificacion se pena con cross entropy y feature maps con MMD, discrepancia maxima de media.

$$ \mathcal{L}_{NST}(WS) =\mathcal{L}_{ce}(Y_{true},ps)+\frac{\lambda}{2} \mathcal{L}_{MDD^2}(F_T, F_S) $$



El MMD se calcula de la siguiente forma. Dos sets de samples $S_p=\{p^i\}^N_{i=1}$ $S_q=\{q^i\}^m_{i=1}$, luego, la distancia MMD es:

$$ \mathcal{L}_{MDD^2}(S_p,S_q)= \mid \mid \frac{1}{N} \sum^N_{i=1}\phi(p^i) - \frac{1}{M} \sum^M_{j=1}\phi(q^j) \mid \mid$$

donde $\phi(.)$ es una funcion explicita de mapeo. Usando el kernel trick  se puede reformular como

$$ \mathcal{L}_{MDD^2}(S_p,S_q)= \mid \mid \frac{1}{N^2} \sum^N_{i=1}\sum^N_{i'=1} k (p^i,p^{i'}) + \frac{1}{M^2} \sum^M_{j=1}\sum^M_{j'=1} k (q^j,q^{j'}) -+ \frac{1}{MN} \sum^M_{i=1}\sum^M_{j'=1} k (p^i,q^{j})  $$

Lo cual finalmente se aplica usando sampleos  desde $F_T$ y $F_S$, sampleando la activacion a traves de todos los canales y normalizando, $p^i=\frac{f^i_T}{\mid\mid f^i_T \mid\mid_2}$ e identicamente para q con FT

Sobre el k usado, se usaron kernel lineal, polinomial de $d=2$ y $c=0$ y gaussiano con $\sigma^2$ igual al ECM entre los pares. El caso lineal tiene ciertas semejanzas con Att y el caso polinomial de orden 2 da la matriz de gramm usada en FSP. En general funciona mejor que todos los puntos con los que se compara el paper. En el caso de pascal VOC 2007 funciona mejor incluso que la base (Faster R-CNN). Un detalle interesante, si bien no lo probaron con GAN lo postulan como una forma interesante a probar en adelante ya que permite recorrer de mejor manera el espacio de las features.

### FITNETS: HINTS FOR THIN DEEP NETS

Año: MAR 2015 

Autores: Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, Yoshua Bengio

En: ICLR

La primera de estas investigaciones es fitnets del 2015, en esta se usan representaciones intermedias (features) como "hints" de lo que una red estudiante de menor complejidad debiese aprender de una red de mayor complejidad. Esto mejora la capacidad de generalizacion con respecto a un modelo enfocado solo en el resultado final reduciendo a su vez la carga computacional.

Definen un hint como la salida de una capa convolucional $F_{T}$, desde la cual la capa de la red estudiante debe aprender. Se supone que este aprendizaje sirve como una especie de regularizacion, por lo que recomiendan usar representaciones de la parte media de la red.

El entrenamiento se realiza usando una perdida en la siguiente forma (respetando la notacion de arriba). $r$ es un regresor consistente en una capa convolucional con la misma funcion de activacion de $F_T$ que se usa simplemente para poder ajustar el tamaño de $F_S$ al de $F_T$, el cual se preentrena antes de entrenar, este debe usar la misma funcion de activacion de $F_{T}$. 

$$ \mathcal{L}_{HT}=\frac{1}{2}\mid \mid F_T-r(F_s) \mid \mid^2$$

$$ \mathcal{L}_{NST}(W_S) =\mathcal{L}_{ce}(Y_{true},ps)+\lambda \mathcal{L}_{HT}(F_T, F_S) $$

El estudio se centra en el uso de redes estudiantes mas profundas que las redes destiladas pero de menor ancho y por lo mismo menor cantidad de parametros. El funcionamiento es ligeramente mejor al de hinton.

### Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer

Año: Dec. 2016

Autores: Sergey Zagoruyko, Nikos Komodakis

En: ICLR

Code and models for our experiments are available at https://github.com/szagoruyko/attention-transfer.

Se toma prestado el concepto de atencion desde la percepcion humana. Se distinguen 2 tipos de procesos de percepcion; atencionales y no atencionales. Los primeros permiten observar generalidades de una escena y recolectar informacion de alto nivel. Desde este proceso se logra navegar en ciertos detalles de una escena.

En el contexto de CNNs, se considera la atencion como mapas espaciales que permiten codificar donde enfocar mas el procesamiento. Estos mapas se pueden definir con respecto a varias capas de la red y segun en que se basen se dividen en 2 tipos, de activacion y de gradiente. Se realiza un estudio sobre como los mapas de atencion varían segun arquitecturas y como estos mapas pueden ser transferidos a redes estudiantes desde una red tutora ya entrenada. Se centraron en arquitecturas fully convolutional (redes donde la clasificacion o regresion final se realiza sin el uso de capas densas, si no aprovechando la reduccion dimensional que dan las convoluciones). 



- **Mapa de atencion basado en activaciones**

  Considerando una capa de una red y su activación $F_{T}$ se define una funcion de mapeo de activacion $ \mathcal{F}: \mathcal{R}^{C \times H \times W} \rightarrow \mathcal{R}^{ H \times W}$. Asumiendo que para una neurona particular, el valor absoluto de su activacion puede ser tomado como una medida de la importancia que da la red a un determinado input, para obtener con respecto a una posicion $h,w$ se pueden usar alguno de los siguientes estadisticos.

  1. Suma de los absolutos entre los $C$ canales: $\mathcal{F}_{sum}(A)=\sum_{i=1}^C \mid A_i \mid$
  2. Suma de potencias: $\mathcal{F}^p_{sum}(A)=\sum_{i=1}^C \mid A_i \mid ^p$
  3. Maximo de potencias: $\mathcal{F}^p_{max}(A)=\max _{i=1,c} \mid A_i \mid ^p$

  En general se nota que las redes de mejor accuracy suelen tener atencion mas marcada, y que las capas iniciales se "fijan" en detalles como ojos o narices mientras que las mas profundas se fijan en objetos de mayor nivel como caras. 

  La perdida de entrenamiento en este caso es la siguiente, donde $\frac{Q_i^j}{\left \| Q_i^j \right \|}_2 $ es simplemente una normalizacion de las activaciones :

  $$ \mathcal{L}_{NST}(WS) =\mathcal{L}_{ce}(Y_{true},ps)+\frac{\lambda}{2} \mathcal{L}_{at}(F_T, F_S) $$

  Donde $\mathcal{L}_{at}(F_T, F_S)=  \sum_{j \in \mathcal{C}} \left \|  \frac{Q_S^j}{\left \| Q_S^j \right \|}_2 + \frac{Q_T^j}{\left \| Q_T^j \right \|}_2 \right \|_p$

- **Mapa de atencion basado en gradiente**

Para el caso de gradiente, se asume que el gradiente de la perdida de clasificacion con respecto a una entrada permite medir la "sensibilidad" de la red ante el estimulo, para esto se define el gradiente como.

$$ J_i =\frac{\partial \mathcal{L}_{ce}(W_i,x)}{\partial x}$$

La perdida toma la siguiente forma, la cual puede ser dificil para analizarse analiticamente ya que implica realizar backpropagation dos veces pero con las tecnicas modernas de diferenciacion automatica no deberia ser problema.

$$ \mathcal{L}_{NST}(WS) =\mathcal{L}_{ce}(Y_{true},ps)+\frac{\lambda}{2} \left \| J_s - J_T \right \|_2 $$

En general ambos metodos funcionan bien.

### Paraphrasing Complex Network: Network Compression via Factor Transfer

Año: 2018

Autores: Jangho Kim, SeongUk Park, Nojun Kwak

En: NIPS

Destila a nivel de features, pero proponiendo el uso de  capas intermedias en un "autoencoder fashion" que sirva de "interprete" entre el conocimiento de la red tutora y la estudiante, de manera similar al regresor de fitsnets solo que con una mayor cantidad de abstraccion entre medio, le pone de nombre "factores".  

El traspaso de informacion se realiza en 2 niveles, primero se entrena un parafreasador desde la red tutora, reconstruyendo el input desde la capa desde la cual se quiere realizar el traspaso de informacion. Se minimiza entonces la siguiente perdida en esta etapa, $P(x)$.

$$\mathcal{L}_{rec}=\left \| x - P(x) \right \| ^2$$

Luego de entrenado el parafraseador, se entrena la red estudiante ubicando una capa de interfaz que sirve de traductor entre el factor (salida del parafraseador) y la salida de la red estudiante. Luego, se usa la misma funcion de perdida de attention transfer entre las salidas del traductor y el parafraseador, a lo cual se suma la perdida de cross entropy para la salida.

Funciona regularmente bien, no demasiado, si regularmente

### Layer-Level Knowledge Distillation for Deep Neural Network Learning

año: abril 2019

Autores:Hao-Ting Li, Shih-Chieh Lin, Cheng-Yeh Chen and Chen-Kuo Chiang

En: MDPI



El modelo consiste de dos partes, lsp o layer selectivity procedure que usa el hessiano entre capas para determinar que features linkear y ALs o auxiliary structure learning que realiza la transferencia de conocimiento entre las features de los modelos tutor y estudiantes .

ALS basicamente consiste en usar capas para proyectar las features de un modelo al otro. Mediante una capa de projeccion por modelo que mapee los features a un vector de una dimensionalidad en $\mathbb{R}^n$ y otra capa densa de alineamiento mediante las que penar las diferencias.

$$\mathcal{L}_{Align}^{(K)}(t,s) = \| X_t^{(i)} -X_s^{(j)}  \|_2$$

La perdida final incorpora la suma de todas las perdidas de todas las etapas de  ALS.

$$\mathcal{L}_{total}=\sum_{k=1}^n \mathcal{L}_{Align}^{(k)} + \mathcal{L}_{model}^{(p)} $$

La otra etapa, LSP consiste en seleccionar que capas de la la red tutora transferir usando el argumento minimo del grammiamo inter clases sumado al grammiano entre capas. El funcionamiento es ligeramente mejor a Fitnets.

### A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning

Año: 2017

Autores: Junho Yim, Donggyu Joo, Jihoon Bae, Junmo Kim

En: CVPR

Usan distiliacion para resolver tres tareas distintas pero complementarias; Optimizacion del entrenamiento de la red estudiante mediante inicializacion inteligente de los pesos, mejorar el performance en tiempo de la red estudiante dado su tamaño y mejorar el performace en accuracy dada la transferencia de conocimiento de la red.

En vez de destilar directamente las features de la red tutora, centran su problema en la destilacion del flujo del procedimiento de resolución (FSP *Flow of Solution Procedure*) de la red tutora, definida como la relacion entre dos features intermedios. Matematicamente hablando, definen este FSP entre dos capas como la matriz Gramiana entre los features de ambas. En el caso de dos capas $1$ y $2$, un input $x$ y pesos $W$, este valor sería:

$$G_{i,j}(x,W)= \sum_{s=1}^h \sum_{t=1}^w\frac{ F^1_{s,t,i}(x,W)  F^2_{s,t,i}(x,W)}{h \times w}$$

Nótese que la formula se define para capas sin perdida dimensional, como es el caso de las capas en los bloques res-net. De tal forma la perdida se define como la distancia $L_2$ entre las FSPs de ambas redes, donde se deja libre un parametro $\lambda_i$ sobre las n posiciones de las matrices para poder ponderar entre redes. En el caso expuesto este lambda se deja libre.

$$\mathcal{L}_{FSP} = \frac{1}{N} \sum_x \sum_{i=1}^n \lambda_i \left \| G_i^T(x;W_T)-G_i^S(x;W_S)  \right \|_2$$

Al momento del entrenamiento, el entrenamiento se divide en dos fases, inicializacion y entrenamiento sobre la tarea principal. La inicializacion se realiza seteando $W_S$ de tal forma que minimice $\mathcal{L}_{FSP}$ entre ambas redes, para seguidamente realizar un finetunning entrenando de manera regular contra los labels originales.

El funcionamiento ws regular, muy posiblemente por la manera de pos-entrenamiento (No hay entrenamiento al momento de destilar, solo en la inicializacion de los pesos).



### Learning Deep Representations with Probabilistic Knowledge Transfer

año: Marzo 2018

Autores: Nikolaos Passalis, Anastasios Tefas

En: ECCV

Rather than hacer un matching directo entre los features como en casi todos los casos de layer level, hacen aproximan la distribución de probabilidades de la red estudiante a la red tutora.

El uso de probabilidades permite realizar knowledge transfer usando

- Cross-modal knowledge
- Features creados a mano (sift, etc)
- Transferir features usando tasks distintos al buscado
- Incorporar domain-knowledge

Al ser difícil de obtener de manera directa la distribucion completa de las features de la red se minimiza una distribucion condicional. Esta se estima usando kernel density estimation. Tomando un kernel $K$, la distribucion condicional para la red tutora es:

$$p^t_{i \mid j}= \frac{K( F_t^{i} F_t^j; 2\sigma_t^2)}{\sum^N_{k=1,k\neq j} K( F_t^{i} F_t^j; 2\sigma_t^2)} \in [0,1]$$ 

La distribucion para la red estudiante es equivalente cambiando los features $F_s^{i} $ y la varianza $\sigma_s^2$. Como kernels se proponen el uso del kernel gaussiano y una mejora interesante, la distancia coseno $K=\frac{1}{2} \frac{a^{\top}b}{\|a\|_2\|b\|_2}+1\in [0]$.

Ambas distribuciones se aproximan minimizando la divergencia Kullback-Leibler en su variante batch sobre muestras $D_{\mathrm{KL}}(P\|Q) = \sum_{i=1}^N \sum_{j=1, j\neq i}^N p(x) \ln \frac{p(x)}{q(x)} $, entrena usando adam en batches de 64 y 128.

Dicen algo mas dificil y interesante sobre mutual information C/R a la divergencia kullback leibler

Funciona bien. todo:profundizar

### Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons

Autores: Byeongho Heo, Minsik Lee, Sangdoo Yun, Jin Young Choi

Año: 2019

En: AAAI

Partiendo desde el hecho que la combinacion de las activaciones en una capa forman una funcion binaria de "bordes" sobre el espacio de salida de esta, el paper propone la transferencia no de la magnitud de la respuesta de una neurona si no de las activaciones de esta. Comparan por ejemplo las activaciones logradas con MSE como es el caso de Fitnets con el metodo propuesto, logrando un mejor borde de separacion de clases, al costo de mover la magnitud de las activaciones.

La perdida es similar a la de Fitnets, reemplazando la funcion de activaciones de la capa $\sigma$ por una binarizacion de la misma.

$$ \rho(x)= \left\{\begin{matrix} 1&  \textrm{si x >  0} \\ 
0 & \textrm{si no}
\end{matrix}\right.$$

Dado que $\rho$ no es diferencial se usa en el entrenamiento una perdida similar a hinge loss de SVM, donde $\mu$ representa un margen para estabilidad en el entrenamiento, $\vec{1}$ es un vector de largo M y valor 1 en cada elemento y $\odot$ es el producto punto entre los elementos. Nótese que en este caso tanto $F_S$ como $F_T$ representan la salida de una capa previo a la funcion de activacion.

$$\mathcal{L}(I)=\| \rho(F_T) \odot \sigma \left (\mu \vec{1} -F_S \right)  -\left (\vec{1} - \rho(F_T)  \right ) \odot \sigma \left (\mu \vec{1} --F_S \right ) \|_2^2$$

Con respecto a la neurona iésima en la red tutora $t_i$ y estudiante $s_i$, esto tiene una derivada por partes igual a.

$$-\frac{\partial \mathcal{L} }{\partial s_i}=\left\{\begin{matrix}
2(s_i-\mu) & \textrm{ si }\rho(t_i)=1\textrm{ y }s_i < \mu\\ 
-2(s_i+\mu) & \textrm{ si }\rho(t_i)=0\textrm{ y }s_i > -\mu \\ 
 0& \textrm{en cualquier otro caso} 
\end{matrix}\right.$$

En casos en que hayan tamaños distintos en la cantidad de neuronas se usa un regresor en la misma manera que en Fitnets.

El desempeño es bastante real





### Learning Efficient Object Detection Models with Knowledge Distillation

Autores: Guobin Chen, Wongun Choi, Xiang Yu, Tony Han, Manmohan Chandraker

Año: 2017

En: NIPS

Proponen un metodo de destilacion sobre Faster-RCNN, una de las redes estado del arte en detección de objetos mutilase compuesta por 3 modulos, un backbone común de extraccion  de features, una red de propuesta de regiones **RPN** y una red de clasificacion y regresion del bounding box **RCN** sobre las regiones propuestas de la **RPN** y el backbone. Partiendo desde Fitnets, proponen una serie de modificaciones para adaptarse al target compuesto (en regresion y clasificacion) de la detección y el desbalance de clases.

Tomando $N$ y $M$ como tamaños de batch para **RCN** y **RPN**, $L_{cl}$ como la perdida de clasificacion, $L_{reg}$ como perdida de regrecion y $\lambda$ con $\gamma$ como hiperparametros de balance entre perdidas fijados en 1 y 0.5 respectivamente, las perdidas de los modulos **RCN** y **RPN** toman las siguientes formas respectivamente:

$$\mathcal{L}_{RCN}=\frac{1}{N}\sum_iL_{cl}^{RCN}+\lambda\frac{1}{N}\sum_iL_{reg}^{RCN}$$

$$\mathcal{L}_{RPN}=\frac{1}{M}\sum_iL_{cl}^{RPN}+\lambda\frac{1}{M}\sum_iL_{reg}^{RPN}$$

Con lo que la perdida total toma la forma.

$$\mathcal{L}=\mathcal{L}_{RPN}+\mathcal{L}_{RCN}+\gamma\mathcal{L}_{Hints}$$



Se explicaran en detalle el balance de clases y la perdida destilada con margen del framework. El uso de Hints es idéntico al expuesto en Fitnets.

**Balance de clases en clasificacion**

A diferencia del caso de destilacion original donde el objetivo es el de clasificacion de imágenes, el caso de deteccion en imagenes tiene que lidiar con un desbalance de clases entre el fondo y los objetos. Para esto la destilacion original propuesta por Hinton $\mathcal{L}_{cl} = \lambda L_{ce}+(1-\lambda)L_{dist}$  se modifica para balancear mejor las predicciones en la perdida de destilacion, introduciendo un $w_c$ que amplifica la penalizacion de la perdida cuando se trata de clase background.

$$L_{dist}=-\sum w_{c}P_t log P_S$$

**Perdida destilada con margen**

En casos de regresion del bounding box la perdida se modifica para penalizar solo en casos que la perdida de la red estudiante sea mayor que la perdida de la red tutora mas un margen $m$.

$$L_{bb}(R_S,R_T,y)=\left\{\begin{matrix}
 \|R_S-y\|^2_2 && \textrm{si  } \|R_S-y\|^2_2 +m > \|R_T-y\|^2_2 , \\ 
0 & & \textrm{si no}
\end{matrix}\right.$$

Con lo que la perdida de regresion se pasa a ser una combinacion entre la regresion de la perdida $L_1$ sobre los labels reales $L_S$ y la perdida destilada $L_b$. 

$$L_reg=L_{S}+ \nu L_b$$



### Knowledge Transfer with Jacobian Matching 

Autores: Suraj Srinivas, Francois Fleuret

Año: 2018

En: ICML

Proponen un entrenamiento mediante matching de los jacobianos en alguna capa intermedia. El jacobiano se calcula con respecto al pixel de mayor intensidad dentro del espacio de entrada. Al parecer usan la red como un metodo de pre-inicializacion para finalmente entrenar con cross entropy. No tiene demasiado sentido la tecnica.







## Malitos de layer level

### Accelerating Convolutional Neural Networks with Dominant Convolutional Kernel and Knowledge Pre-regression

Año: 2016 

Tipo: Layer level, model compression

Aplicacion: imagenes

Abstract: Aiming at accelerating the test time of deep convolutional neural networks (CNNs), we propose a model compression method that contains a novel dominant kernel (DK) and a new training method called knowledge pre-regression (KP). In the combined model DK2PNet, DK is presented to significantly accomplish a low-rank decomposition of convolutional kernels, while KP is employed to transfer knowledge of intermediate hidden layers from a larger teacher network to its compressed student network on the basis of a cross entropy loss function instead of previous Euclidean distance. Compared to the latest results, the experimental results achieved on CIFAR-10, CIFAR-100, MNIST, and SVHN benchmarks show that our DK2PNet method has the best performance in the light of being close to the state of the art accuracy and requiring dramatically fewer number of model parameters.

- Inentendible, intentan definir una arquitectura convolucional que decompone la convolucion y entrenar una red estudiante sobre eso pero no se entiende la primera parte.





### FEED: FEATURE-LEVEL ENSEMBLE EFFECT FOR KNOWLEDGE DISTILLATION

Año: 2019

Tipo: Layer level distillation

Aplicacion

Abstract: This paper proposes a versatile and powerful training algorithm named Feature- level Ensemble Effect for knowledge Distillation (FEED), which is inspired by the work of factor transfer. The factor transfer is one of the knowledge transfer methods that improves the performance of a student network with a strong teacher network. It transfers the knowledge of a teacher in the feature map level using high-capacity teacher network, and our training algorithm FEED is an extension of it. FEED aims to transfer ensemble knowledge, using either multiple teacher in parallel or multiple training sequences. Adapting peer-teaching framework, we introduce a couple of training algorithms that transfer ensemble knowledge to the student at the feature map level, both of which help the student network find more generalized solutions in the parameter space. Experimental results on CIFAR-100 and ImageNet show that our method, FEED, has clear performance enhancements, without introducing any additional parameters or computations at test time.

- Centran la destilacion en tanto metodo de detener el overfitting al destilar desde un ensamble.
- Comentarios de open review. Muy parecido a la destilacion de ensamble de hinton 2019 sin buenos resultados.
- In this paper, the authors present two methods, Sequential and Parallel-FEED for learning student networks that share architectures with their teacher.
- it isn't clear to me where the novelty lies in this work. Sequential-FEED appears to be identical to BANs (https://arxiv.org/abs/1805.04770) with an additional non-linear transformation on the network outputs as in https://arxiv.org/abs/1802.04977. Parallel-FEED is just an ensemble of teachers; please correct me if I'm wrong.

