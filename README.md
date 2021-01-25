

![](./imagenes/upiita_tt1.png)
## <h1><center>Trabajo Terminal II</center></h1>
##  <h1><center>Caracterización de precursores para patologías cardíacas mediante análisis complejo.</center></h1>


## Autores:
 * Cruz Reyes Juan Manuel
 * Portilla Brambila Sebastian
 * Ramírez Palomino Alan Jesús

## Asesores:

 * Dr. Lev Guzmán Vargas
 * M. en C. Alvaro Anzueto Rios
 

 ## Hipótesis:
Utilizando registros de series de tiempo de la actividad eléctrica del corazón (electrocardiograma), y por medio de la aplicación de Gráficos de Poincaré, Entropía de Información, Análisis Fractal y Transformada de ondeleta, se obtendrán patrones característicos que serían identificados como precursores de fibrilación atrial, infarto agudo al miocardio e insuficiencia cardíaca congestiva.

## Objetivo General
Analizar eventos de Fibrilación Arterial, Infarto Agudo del Miocardio e insuficiencia cardíaca congestiva por medio de análisis complejo de series de tiempo utilizando gráficos de Poincaré, Entropía de Información, Análisis Fractal y  Transformada de Ondeleta para la caracterización de precursores fisiológicos.

## Objetivos Específicos
* Obtener los registros electrocardiográficos correspondientes a Infarto agudo al miocardio, fibrliación atrial e insuficiencia cardíaca congestiva en las bases de datos para su clasificación.
* Preprocesar las señales de cada registro para la obtención de la señal de variabilidad cardíaca (HRV). 
* Aplicar técnicas lineales (media,curtuosis, sesgo) para verificar la no diferenciabilidad y la separación respecto al régimen Gaussiano del conjunto de datos.

* Aplicar técnicas no lineales de entropía de información, transformada de ondeletas, gráficos de Poincaré y fractalidad a los registros ECG Y HRV para obtener patrones morfológicos, magnitud y de control en las ondas de ECG.
* Verificar la eficiencia del conjunto de precursores obtenidos con las técnicas no lineales mediante un algoritmo clasificador supervisado de patologías para comparar los registros del grupo de control saludable existente en la base de datos.

## Propuesta de solución

La propuesta de solución para el problema planteado consiste en una metodología compuesta por tres fases principales:
* Ingeniería de datos.
* Procesamiento y análisis de señales.
* Clasificación.
 
 El diseño de dicha solución puede verse de manera esquemática en la siguiente imagen.
 ![](./imagenes/ROADMAP_TT2.png)

## Trabajo Escrito

El trabajo escrito generado a partir de esta investigación se encuentra contenido en:
> TTB_2020_2_13/tesis

## Módulos y Paqueterías de referencia 

El presente proyecto hace uso de módulos de python ecternos, ya sea para optimizar el cálculo de métricas específicas ó para tomar como referencia en la implementación de código propio. Todas las rutas a dichas herramientas se encuentran referenciadas con un enlace directo a los respectivos repositorios a continuación:

* https://github.com/raphaelvallat/entropy
* https://github.com/PGomes92/pyhrv
* 