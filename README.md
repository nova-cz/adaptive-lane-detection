# 🚗 Detección de Líneas de Carril con Procesamiento de Imágenes

<div align="center">

**Un sistema avanzado para la detección de líneas de carril en tiempo real utilizando técnicas de procesamiento de imágenes y visión por computadora.**

[[🎥 Ver demostración del proyecto](./videoREADME.mp4)](https://github.com/user-attachments/assets/db600a52-63bc-4457-a4d5-7c2f5fdc4647)


</div>

## 🛠️ Tecnologías Utilizadas

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/OpenCV-4.8.0-green?style=for-the-badge&logo=opencv" alt="OpenCV 4.8.0">
  <img src="https://img.shields.io/badge/numpy-1.24.3-blue?style=for-the-badge&logo=numpy" alt="NumPy">
  <img src="https://img.shields.io/badge/argparse-1.4.0-lightgrey?style=for-the-badge" alt="argparse">
</p>

## 🌟 Características Principales

- **Detección en Tiempo Real**: Procesamiento de video en tiempo real para la detección de líneas de carril.
- **Ajuste de Parámetros Interactivo**: Control en tiempo real de los parámetros de procesamiento.
- **Operaciones Morfológicas Avanzadas**: Incluye erosión, dilatación, apertura y cierre para mejorar la detección.
- **Ajuste de Imagen**: Corrección gamma, ecualización de histograma y ajuste de contraste.
- **Interfaz de Usuario Intuitiva**: Controles interactivos para ajustar parámetros durante la ejecución.

## 📋 Requisitos Previos

- Python 3.9 o superior
- OpenCV 4.8.0 o superior
- NumPy 1.24.3 o superior

## 🚀 Instalación

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/tu-repositorio.git
   cd tu-repositorio
   ```

2. **Crea un entorno virtual (recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Uso

### Procesar un video:
```bash
python main.py ruta/al/video.mp4
```

### Parámetros opcionales:
```bash
python main.py video.mp4 k=256 gamma=3 kernel=5 erosion=2 dilation=2
```

### Controles interactivos:
- `ESPACIO`: Pausar/Reanudar
- `q`: Salir
- `r`: Reiniciar
- `+`/`-`: Ajustar velocidad
- `o`: Alternar operación de apertura
- `c`: Alternar operación de cierre
- `e`/`E`: Aumentar/disminuir iteraciones de erosión
- `d`/`D`: Aumentar/disminuir iteraciones de dilatación
- `k`/`K`: Aumentar/disminuir tamaño del kernel

## 🏗️ Estructura del Proyecto

```
├── main.py          # Script principal
├── utils.py         # Funciones de utilidad
├── requirements.txt # Dependencias del proyecto
└── videoREADME.mp4  # Video demostrativo
```

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más información.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, lee las pautas de contribución antes de enviar un pull request.

## 📧 Contacto

¿Preguntas o sugerencias? Por favor, abre un issue o contáctame en [tu@email.com](mailto:tu@email.com)

## 📚 Recursos Adicionales

- [Documentación de OpenCV](https://docs.opencv.org/4.8.0/)
- [Documentación de NumPy](https://numpy.org/doc/stable/)
- [Guía de Python](https://docs.python.org/3/)
