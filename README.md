# Latent Meta NCA Manifold for Image Segmentation

Este repositorio implementa una arquitectura híbrida de aprendizaje profundo orientada a la segmentación semántica de imágenes (aplicada sobre el dataset *Oxford-IIIT Pet*). El diseño unifica dos conceptos de investigación recientes en Sistemas Autoorganizados y Autómatas Celulares Neuronales (NCA): **Latent NCA** y **NCA Manifold (Meta-NCA)**.

---



### 1. Latent NCA (Eficiencia en Espacio Compacto)
Trabajar con NCA directamente en el espacio de píxeles de alta resolución es computacionalmente prohibitivo y propenso a inestabilidades. Siguiendo la filosofía de *Latent NCA*, la imagen de entrada se comprime primero mediante un codificador neuronal (*Encoder*) a una resolución espacial reducida, pero con una alta densidad de canales semánticos. 

El autómata opera exclusivamente dentro de este **espacio latente comprimido**. Esto no solo reduce drásticamente el consumo de memoria de video (VRAM), sino que permite que las reglas locales del NCA interactúen con características abstractas de alto nivel (como texturas complejas o partes del cuerpo del animal) en lugar de simples valores de color RGB.

### 2. NCA Manifold & Meta-Learning (Reglas Dinámicas)
En un NCA clásico, la regla de transición (los pesos de la red convolucional que dictan cómo evoluciona cada celda) es fija y universal para todos los datos. Para resolver una tarea heterogénea como la segmentación, adoptamos la estrategia de *NCA Manifold*:

Un módulo externo (actuando como una *HyperNetwork* o modulador) analiza el contexto global de la imagen de entrada y genera **factores de transcripción específicos**. Estos factores modifican dinámicamente las reglas de actualización del NCA en cada forward. En términos biológicos, la red no usa el mismo "ADN" estático para cada imagen; en su lugar, la presencia de un perro o un gato altera la expresión de las reglas para que el autómata guíe la segmentación de manera óptima según el contexto.



## Flujo del Pipeline

1.  **Encoder:** Procesa la imagen $I \in \mathbb{R}^{3 \times H \times W}$ y extrae el estado latente inicial $z_0 \in \mathbb{R}^{C \times h \times w}$ junto con las conexiones *skip*.
2.  **Meta-Modulation:** Condiciona las dinámicas del autómata basándose en la identidad y contexto global de $I$.
3.  **NCA Evolution:** El estado latente evoluciona iterativamente durante $N$ pasos fijos mediante actualizaciones locales:
    $$z_{t+1} = z_t + \text{MetaNCA}(z_t)$$
4.  **Decoder:** Toma el estado latente estabilizado $z_N$ y las conexiones *skip* para proyectar el resultado final a la máscara de segmentación de tres clases (Background, Foregound, Boundary).

---

## Estrategia de Entrenamiento (Fases)

Entrenar un NCA dinámico de extremo a extremo (*End-to-End*) desde cero genera un acoplamiento caótico en los gradientes debido a *Backpropagation Through Time* (BPTT). Para asegurar la convergencia, el entrenamiento se divide:

* **Fase 1: Estabilización del Espacio Latente.** Se entrena el Autoencoder base como un cuello de botella tradicional para asegurar que el espacio latente codifique representaciones estables y decodificables. El NCA permanece inactivo.
* **Fase 2: Aprendizaje de la Morfogénesis.** Se congelan los pesos del Autoencoder y se entrena exclusivamente el módulo NCA y su red de condicionamiento dinámico, forzando al autómata a aprender a corregir, refinar y estabilizar las máscaras latentes en el tiempo.