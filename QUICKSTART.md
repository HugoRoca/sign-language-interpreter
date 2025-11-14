# 🚀 Guía Rápida de Inicio

Esta guía te ayudará a ejecutar el intérprete de lenguaje de señas paso a paso.

## 📋 Paso 1: Instalación

### 1.1 Crear entorno virtual (recomendado)

```bash
python3 -m venv venv
source venv/bin/activate  # En macOS/Linux
# O en Windows: venv\Scripts\activate
```

### 1.2 Instalar dependencias

```bash
pip install -r requirements.txt
```

## 🎯 Paso 2: Entrenar el Modelo

**IMPORTANTE:** Debes entrenar el modelo ANTES de usar la cámara. El modelo es necesario para hacer las predicciones.

### 2.1 Entrenar el modelo (básico)

```bash
python3 -m src.interfaces.cli.main train
```

Este comando:
- Cargará las imágenes de `data/asl_alphabet_train/`
- Entrenará el modelo CNN
- Guardará el modelo en `models/asl_model.keras`

**⏱️ Tiempo estimado:** 10-30 minutos dependiendo de tu hardware (más rápido con GPU)

### 2.2 Entrenar con opciones personalizadas

```bash
python3 -m src.interfaces.cli.main train \
    --epochs 20 \
    --batch-size 32
```

## 🎥 Paso 3: Usar la Cámara en Tiempo Real

Una vez que el modelo esté entrenado, puedes activar la cámara:

```bash
python3 -m src.interfaces.cli.main camera
```

### Controles durante la ejecución:

- **`q`**: Salir del programa
- **`c`**: Limpiar la palabra actual
- **`r`**: Resetear la posición del ROI al centro

### Cómo usar:

1. **Coloca tu mano** dentro del rectángulo verde (ROI) en la pantalla
2. **Realiza la seña** de la letra que deseas (A, B, C, etc.)
3. El sistema **detectará la letra** y la agregará a la palabra actual
4. Para **agregar un espacio**, mantén la seña de 'space' por 2 segundos
5. Para **borrar la última letra**, realiza la seña de 'del'
6. Las **palabras completadas** se mostrarán en la parte superior

## 📸 Paso Alternativo: Predecir desde una Imagen

Si prefieres probar con una imagen en lugar de la cámara:

```bash
python3 -m src.interfaces.cli.main predict data/asl_alphabet_test/A/A_test.jpg
```

## 🔧 Solución de Problemas

### Error: "Model not found"
- **Solución:** Debes entrenar el modelo primero con `python3 -m src.interfaces.cli.main train`

### Error: "Could not start camera"
- **Solución:** Verifica que tu cámara esté conectada y no esté siendo usada por otra aplicación
- Prueba con otro índice: `--camera-index 1`

### Error: "Data directory does not exist"
- **Solución:** Asegúrate de que `data/asl_alphabet_train/` existe y contiene las carpetas de las letras

### El modelo no detecta bien las señas
- Aumenta el número de épocas en el entrenamiento: `--epochs 20`
- Ajusta la confianza mínima: `--min-confidence 0.6`
- Asegúrate de tener buena iluminación
- Coloca tu mano completamente dentro del rectángulo verde

## 📝 Resumen de Comandos

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Entrenar modelo (OBLIGATORIO la primera vez)
python3 -m src.interfaces.cli.main train

# 3. Usar cámara en tiempo real
python3 -m src.interfaces.cli.main camera

# O predecir desde imagen
python3 -m src.interfaces.cli.main predict path/to/image.jpg
```

## 💡 Tips

- **Primera vez:** Entrena con pocas épocas (5-10) para probar rápidamente
- **Mejor precisión:** Entrena con más épocas (20-30) para mejor rendimiento
- **Iluminación:** Usa buena iluminación para mejores resultados
- **Fondo:** Un fondo simple ayuda a la detección
- **Estabilidad:** Mantén la seña estable por un momento para que se detecte

