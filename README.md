# Sign Language Interpreter

Un intérprete de lenguaje de señas (ASL - American Sign Language) implementado en Python usando deep learning. Este proyecto utiliza una arquitectura CNN (Convolutional Neural Network) para reconocer y clasificar señas del alfabeto ASL.

## 🏗️ Arquitectura

El proyecto sigue los principios SOLID y una arquitectura en capas:

```
src/
├── domain/              # Capa de dominio (lógica de negocio)
│   ├── entities/        # Entidades del dominio
│   ├── repositories/    # Interfaces de repositorios (abstracciones)
│   └── services/        # Servicios del dominio
├── infrastructure/      # Capa de infraestructura (implementaciones)
│   ├── camera/          # Implementación de cámara (OpenCV)
│   ├── data_loaders/    # Carga de datos desde sistema de archivos
│   ├── models/          # Implementación del modelo CNN
│   ├── preprocessing/   # Preprocesamiento de imágenes
│   └── services/        # Servicios de infraestructura (formación de palabras)
├── application/         # Capa de aplicación (casos de uso)
│   └── services/        # Servicios de aplicación (entrenamiento, predicción, cámara)
└── interfaces/          # Capa de interfaces
    ├── cli/             # Interfaz de línea de comandos
    └── camera/          # Interfaz de cámara (alternativa)
```

### Principios SOLID aplicados:

- **Single Responsibility**: Cada clase tiene una única responsabilidad
- **Open/Closed**: Las interfaces permiten extender funcionalidad sin modificar código existente
- **Liskov Substitution**: Las implementaciones pueden sustituirse por sus interfaces
- **Interface Segregation**: Interfaces específicas y pequeñas (DataRepository, ModelRepository, PreprocessingService)
- **Dependency Inversion**: Las capas superiores dependen de abstracciones, no de implementaciones concretas

## 📋 Requisitos

- Python 3.8 o superior
- TensorFlow 2.15+
- NumPy, OpenCV, Pillow
- scikit-learn

## 🚀 Inicio Rápido

### Paso 1: Instalación

1. Crea un entorno virtual (recomendado):
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

### Paso 2: Entrenar el Modelo (OBLIGATORIO la primera vez)

**⚠️ IMPORTANTE:** Debes entrenar el modelo ANTES de usar la cámara.

```bash
python3 -m src.interfaces.cli.main train
```

Este proceso puede tardar 10-30 minutos dependiendo de tu hardware.

### Paso 3: Ejecutar la Cámara

Una vez entrenado el modelo, puedes usar la cámara en tiempo real:

```bash
python3 -m src.interfaces.cli.main camera
```

**¡Listo!** Ahora puedes hacer señas frente a la cámara y el sistema las detectará.

> 📖 Para más detalles, consulta [QUICKSTART.md](QUICKSTART.md)

## 📊 Datos

El proyecto utiliza el dataset ASL Alphabet que debe estar en `data/asl_alphabet_train/`. El dataset contiene imágenes organizadas por carpetas, una para cada letra del alfabeto (A-Z) y clases especiales (space, del, nothing).

## 🎯 Uso Detallado

### 1. Entrenar el modelo

```bash
python3 -m src.interfaces.cli.main train
```

Opciones adicionales:
```bash
python -m src.interfaces.cli.main train \
    --data-dir data/asl_alphabet_train \
    --model-path models/asl_model.keras \
    --epochs 20 \
    --batch-size 32
```

### 2. Hacer predicciones desde una imagen

```bash
python3 -m src.interfaces.cli.main predict path/to/image.jpg
```

O especificar un modelo diferente:
```bash
python3 -m src.interfaces.cli.main predict path/to/image.jpg --model-path models/asl_model.keras
```

### 3. 🎥 Modo Cámara en Tiempo Real

El intérprete puede activar la cámara y detectar letras y palabras en tiempo real:

```bash
python3 -m src.interfaces.cli.main camera
```

**⚠️ Requisito:** Debes haber entrenado el modelo primero (Paso 2 del Inicio Rápido).

**Controles durante la ejecución:**
- `q`: Salir del programa
- `c`: Limpiar la palabra actual
- `r`: Resetear la posición del ROI (región de interés) al centro

**Opciones adicionales:**
```bash
python -m src.interfaces.cli.main camera \
    --model-path models/asl_model.keras \
    --camera-index 0 \
    --min-confidence 0.7 \
    --stability-threshold 10 \
    --space-delay 2.0
```

**Parámetros:**
- `--camera-index`: Índice de la cámara a usar (por defecto: 0)
- `--min-confidence`: Confianza mínima para aceptar una letra (por defecto: 0.7)
- `--stability-threshold`: Número de detecciones consecutivas necesarias para agregar una letra (por defecto: 10)
- `--space-delay`: Segundos de detección de 'space' para agregar un espacio (por defecto: 2.0)

**Cómo usar:**
1. Coloca tu mano dentro del rectángulo verde (ROI) en la pantalla
2. Realiza la seña de la letra que deseas
3. El sistema detectará la letra y la agregará a la palabra actual
4. Para agregar un espacio, mantén la seña de 'space' por 2 segundos
5. Para borrar la última letra, realiza la seña de 'del'
6. Las palabras completadas se mostrarán en la parte superior de la pantalla

## 🔧 Configuración

Puedes modificar los parámetros de configuración en `config/settings.py`:

- `IMAGE_SIZE`: Tamaño de las imágenes (por defecto: 64x64)
- `BATCH_SIZE`: Tamaño del batch para entrenamiento (por defecto: 32)
- `EPOCHS`: Número de épocas (por defecto: 10)
- `LEARNING_RATE`: Tasa de aprendizaje (por defecto: 0.001)
- `VALIDATION_SPLIT`: Porcentaje de datos para validación (por defecto: 0.2)

## 📁 Estructura del Proyecto

```
sign-language-interpreter/
├── config/              # Configuraciones
├── data/                # Datos de entrenamiento y prueba
├── models/              # Modelos entrenados (generados)
├── src/                 # Código fuente
│   ├── domain/         # Lógica de negocio
│   ├── infrastructure/ # Implementaciones
│   ├── application/    # Casos de uso
│   └── interfaces/     # Interfaces de usuario
├── tests/              # Tests (por implementar)
├── requirements.txt    # Dependencias
└── README.md          # Este archivo
```

## 🧠 Modelo

El modelo utiliza una arquitectura CNN con:
- 4 bloques convolucionales con MaxPooling y BatchNormalization
- Capas de Dropout para regularización
- Capa densa final con activación softmax para clasificación multiclase

## 📝 Notas

- El entrenamiento puede tardar varios minutos dependiendo del hardware
- Se recomienda usar GPU para acelerar el entrenamiento
- Los modelos entrenados se guardan en el directorio `models/`
- Los checkpoints se guardan automáticamente durante el entrenamiento
- Para el modo cámara, asegúrate de tener buena iluminación y coloca tu mano dentro del rectángulo verde
- El sistema requiere detecciones estables antes de agregar letras a la palabra (configurable con `--stability-threshold`)
- Para formar palabras, realiza las señas de las letras en secuencia. Usa 'space' para separar palabras y 'del' para borrar

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, asegúrate de seguir los principios SOLID y mantener la arquitectura en capas.

## 📄 Licencia

[Especificar licencia]

