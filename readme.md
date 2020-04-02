


### Experimentos
En general se quiere probar que
1- Los algoritmos permiten aprender
2- Los algoritmos generalizan
3- Los algoritmos entrenan mas rápido que la wea cruda (Estabilidad, velocidad)
4- Son capaces de aprender de alguna distribución generadora/en la vecindad del dataset
5- Cuanto depende del dataset ese aprendizaje
6- Permiten una mejor estabilidad frente al ruido que el entrenamiento en crudo (En que momento pasa el umbral de tanto x)
7- Feats permiten una mejor estabilidad frente al ruido que el uso de logits solo
8- Funcionamiento de feats > funcionamiento de KD > funcionamiento de CE solo


Por cada experimento interesa
1- velocidad de convergencia (estabilidad?)
2- Acc en test 
3- Acc en test / train
4- Mejores y peores
5- Delta sobre KD, CE y media de feats por capa
6- grafico que permita ver tendencia según tecnica/capa



* Normal: FEAT KD
* feat-KDCE
* feat-CE
* noise: FEAt Ruido rango 1:0,1:10
* GAN_exp: Feat Gan
* KD_normal - (KC_CE, KD)
* KD_noise
* KD_GAN

Podria faltar ruido con KDCE y CE
Podria faltar grid de ruido con temperatura
Podría faltar VAE
Podría ser interesante probar con alguna medida de distancia entre y_t y y_s, implica correr todo de nuevo

Falta algún test con datos reales


TODO: Meter gráficos de los interesantes de 1-4