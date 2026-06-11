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


*Poner diagrama*
---

## Estrategia de Entrenamiento (Fases)

Entrenar un NCA dinámico de extremo a extremo (*End-to-End*) desde cero genera un acoplamiento caótico en los gradientes debido a *Backpropagation Through Time* (BPTT). Para asegurar la convergencia, el entrenamiento se divide:

* **Fase 1: Estabilización del Espacio Latente.** Se entrena el Autoencoder base como un cuello de botella tradicional para asegurar que el espacio latente codifique representaciones estables y decodificables. El NCA permanece inactivo.
* **Fase 2: Aprendizaje de la Morfogénesis.** Se congelan los pesos del Autoencoder y se entrena exclusivamente el módulo NCA y su red de condicionamiento dinámico, forzando al autómata a aprender a corregir, refinar y estabilizar las máscaras latentes en el tiempo.

##  Documentación Específica del Código

Esta sección detalla el funcionamiento interno de las clases y algoritmos implementados, especificando el flujo de tensores (*tensor shapes*) y las decisiones de diseño matemático.

---

### 1. Componentes de la Red (`model.py`)

#### `AutoEncoderDown3(nn.Module)`
Es la infraestructura convolucional base que se encarga de la compresión espacial y la posterior reconstrucción de la máscara.
* **Camino Convolucional Corto vs. Pass-Through:** En el Encoder, la imagen se procesa en paralelo por dos rutas. `conv_layer_1` y `conv_layer_2` reducen la resolución espacial aplicando regularización (`Dropout2d`) y normalización (`BatchNorm2d`). En paralelo, `pass_through_1` y `pass_through_2` mantienen un flujo de gradientes limpio sin dropout elevado. Ambos caminos convergen mediante una **suma directa** antes de generar el latente final en `conv_layer_3`.
* **Inyección de Skip Connection:** El Decoder no depende únicamente de la salida del NCA. La salida de baja resolución de `trans_conv_2` (dimensión $64 \times 128 \times 128$) se suma con `c1_out` (extraída del inicio del encoder con dimensión $64 \times 128 \times 128$) justo antes de ingresar a la capa de mezcla (`mix_layer`), recuperando la geometría fina del contorno.

#### `ParameterPredictor(nn.Module)`
Implementa la lógica de *Meta-Learning* mapeando las características de la imagen hacia el espacio de hiperparámetros del autómata.
* **Cálculo Dinámico de Pesos:** Recibe un vector colapsado de tamaño `[Batch, latent_dim]` (donde `latent_dim = 256`). A través de una red totalmente conectada, proyecta este vector a un tamaño plano equivalente a la suma exacta de todos los parámetros requeridos por las convoluciones del NCA:
  $$\text{Total Params} = (H_{dims} \times (C_{out} \times 3) \times 1 \times 1) + H_{dims} + (C_{out} \times H_{dims} \times 1 \times 1) + C_{out}$$
* **Desempaquetado Genérico:** Mediante operaciones nativas de `.view()`, el tensor resultante se segmenta por rangos de índices y se transforma en tensores de pesos compatibles con convoluciones funcionales 2D por lote.

#### `DynamicLatentNCA(nn.Module)`
Representa el motor recurrente del autómata celular que opera directamente sobre la rejilla latente de $32 \times 32$.
* **Operador de Percepción Lineal:** Se utiliza una convolución depthwise agrupada (`groups=C`) con pesos congelados (`get_sobel_kernel`) que calcula las derivadas espaciales de la rejilla. Multiplica los 256 canales por 3 (Sobel X, Sobel Y e Identidad), generando un tensor intermedio de 768 canales.
* **Loop de Actualización por Muestra:** Debido a que cada imagen del lote posee sus propios pesos dinámicos ($w_1, b_1, w_2, b_2$), el loop interno ejecuta `F.conv2d` iterando muestra por muestra (`perceived[b:b+1]`). Esto mantiene el grafo de PyTorch ligero y aislado por elemento.
* **Morfogénesis Estocástica:** En cada uno de los $T$ pasos, se genera una máscara probabilística de actualización:
  $$\text{mask} = \mathbb{I}(\text{rand}(B, 1, H, W) > 0.5)$$
  El estado latente se actualiza sutilmente guiado por un parámetro alfa entrenable acotado: $z_{t+1} = z_t + (\eta \times dx \times \text{mask})$, donde $\eta$ es el `leak_factor`.

---

### 2. Infraestructura de Entrenamiento (`train.py`)

#### Gestión de la Estabilidad: `NCAPool`
Los Autómatas Celulares recurrentes son propensos a divergir o sufrir colapsos catastróficos de gradiente si se entrenan siempre desde el paso $t=0$. Para solucionar esto, se implementa un pool de estados latentes persistente de tamaño fijo (512 ranuras).

* **Muestreo Estocástico (95/5):** Al recibir un lote de imágenes, el pool selecciona índices al azar y extrae los estados latentes históricos de esas muestras. 
  * Con un **95% de probabilidad**, el NCA arranca su simulación tomando como estado inicial el estado evolucionado de la época anterior (`sampled_states`). Esto fuerza al modelo a aprender a estabilizar la rejilla de forma infinita a largo plazo.
  * Con un **5% de probabilidad** (o si la ranura está vacía), el estado se resetea por completo al valor devuelto nativamente por el Encoder (`current_latent`).
* **Sincronización de Contexto:** Para evitar desajustes debido al orden aleatorio del DataLoader, el pool almacena de forma emparejada el estado latente intermedio, la máscara de segmentación real (`target_masks`) y el mapa estructural (`c1_out`). Tras completar los pasos del NCA, los nuevos estados se reescriben en sus respectivos índices mediante `.detach()`.

#### Optimizadores y Blindaje del Gradiente (`train_metanca`)
Durante la Fase 2 (optimización del autómata), el Autoencoder debe permanecer estrictamente estático para evitar que el espacio latente cambie sus propiedades semánticas mientras el NCA intenta aprender sus reglas físicas.
* **Congelación Estricta:** Se apagan los gradientes de la fase 1 (`param.requires_grad = False`) y se purgan los tensores residuales asignándolos a `None`.
* **Desactivación de Capas Dinámicas:** Para evitar que las capas de `BatchNorm2d` sigan recalculando medias móviles o que `Dropout2d` altere la estructura durante el loop de entrenamiento, se pisa temporalmente el método `.train()` del bloque base mediante una función de bypass que fuerza el modo `.eval()` exclusivamente en sub-módulos estocásticos.
* **Restricción del Atractor:** Al finalizar cada optimización de parámetros de Adam, se aplica un truncamiento matemático directo sobre el factor de fuga:
  $$\eta \leftarrow \text{clamp}(\eta, 10^{-3}, 10^{-1})$$
  Esto impide que el autómata celular anule las actualizaciones o sature la rejilla latente con amplitudes incontrolables.