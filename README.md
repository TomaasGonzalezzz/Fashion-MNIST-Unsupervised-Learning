# Aprendizaje no supervisado sobre Fashion-MNIST

Trabajo práctico individual de **I302 — Aprendizaje Automático y Aprendizaje Profundo**,
Universidad de San Andrés. **Autor: Tomás González.**  
Informe fechado el **9 de junio de 2026**.

Comparación de reducción de dimensionalidad y clustering mediante PCA, autoencoders,
β-VAE, k-Means, GMM y visualización t-SNE.

![Comparación de reconstrucciones PCA, AE y VAE, figura original](figures/figura_11_reconstrucciones_pca_ae_vae.png)

## Documentos

- [Consigna original](TP4_I302_1er_semestre_2026.pdf).
- [Informe de la entrega](Gonzalez_Tomas_Informe_TP4.pdf): 10 páginas.
- [Notebook original](Gonzalez_Tomas_Notebook_TP4.ipynb): código, experimentos y salidas.
- [Plantilla de informe de la cátedra](template_informe.tex).

## Dataset

[Descargar fashion_mnist_subset.csv desde la release](https://github.com/TomaasGonzalezzz/i302-tp4-aprendizaje-no-supervisado/releases/download/tp4-dataset-2026/fashion_mnist_subset.csv)

El CSV original pesa aproximadamente **137 MB** y se distribuye como adjunto de
release. Descargarlo y colocarlo en:

```text
data/fashion_mnist_subset.csv
```

Crear la carpeta `data/` si no existe. El archivo está excluido del historial Git.
Su hash SHA-256 figura en [docs/procedencia.json](docs/procedencia.json).

Contiene **25.000 imágenes**, con **784 píxeles** normalizados a [0, 1] y una
etiqueta por imagen: diez clases balanceadas, con 2.500 ejemplos por clase.
El split estratificado reserva 20.000 muestras para entrenamiento y 5.000 para evaluación.
Para clustering se utiliza una muestra estratificada de 3.000 imágenes.

## Recorrido del trabajo

1. Inspeccionar imágenes, distribución de clases, intensidades y valores faltantes.
2. Estandarizar píxeles con estadísticas de entrenamiento e implementar PCA con NumPy.
3. Retener el 90 % de varianza y comparar reconstrucciones con un autoencoder PyTorch.
4. Entrenar un β-VAE y estudiar reconstrucciones, promedios latentes y muestreo normal.
5. Implementar k-Means y GMM diagonal con NumPy; comparar K entre 5 y 15 en PCA y AE.
6. Analizar ganancias decrecientes y Silhouette; visualizar los clusters con t-SNE propio.
7. Examinar tamaños de clusters, composición por clase y pureza externa.

Las etiquetas no se usan para ajustar PCA, los autoencoders o los algoritmos de
clustering; se utilizan para muestreo estratificado, inspección y evaluación externa.

## Resultados originales

- **PCA:** 135 componentes conservan al menos el 90 % de la varianza.
- **AE:** usa la misma dimensión latente para comparar con PCA.
- **β-VAE:** dimensión latente 135 y β = 2, según el informe.
- **Configuración seleccionada:** k-Means sobre PCA, K = 10.
- **Silhouette:** 0,1702; **pureza externa ponderada:** 0,5207.

Son resultados del informe, notebook y tablas originales. No se recalcularon los
entrenamientos completos al publicar. Silhouette y pureza miden aspectos distintos;
la pureza externa no equivale a accuracy de un clasificador supervisado.

## Estructura

| Ruta | Función |
| --- | --- |
| `src/tp4_data.py` | Carga, etiquetas, split y muestreo estratificado. |
| `src/tp4_linear.py` | Estandarización, PCA, reconstrucción y distancias. |
| `src/tp4_clustering.py` | k-Means, GMM diagonal, Silhouette y análisis de clusters. |
| `src/tp4_tsne.py` | Implementación de t-SNE. |
| `src/tp4_torch_models.py`, `src/tp4_torch_train.py` | Autoencoder, β-VAE y entrenamiento PyTorch. |
| `src/tp4_plots.py` | Visualizaciones. |
| `figures/` | 21 figuras originales. |
| `csvs/` | Siete tablas originales de historiales y resultados. |

## Ejecutar

El notebook registra **Python 3.14.3**. Las versiones originales de las bibliotecas
no quedaron registradas; `requirements.txt` enumera las dependencias identificadas.
No se verificó una instalación completa del entorno PyTorch durante esta publicación.

Desde la raíz del repositorio, con el dataset en `data/`:

```sh
python -m venv .venv
```

Activar el entorno:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```sh
# macOS / Linux
source .venv/bin/activate
```

Instalar dependencias y abrir el notebook:

```sh
python -m pip install -r requirements.txt
python -m notebook Gonzalez_Tomas_Notebook_TP4.ipynb
```

Mantener el directorio de trabajo en la raíz del proyecto, porque el notebook usa
rutas relativas para `src/`, `data/`, `figures/` y `csvs/`. Ejecutar las celdas en orden.

El informe, las figuras y las salidas guardadas se pueden leer sin reentrenar.
La ejecución completa sí entrena los autoencoders y recalcula clustering y t-SNE;
puede tardar y sobrescribe figuras y tablas. No se incluyen checkpoints de las redes
ni embeddings guardados que sustituyan ese entrenamiento. Usar una copia o rama
para conservar los resultados entregados al comparar nuevas ejecuciones.

## Conservación y verificación

Los 40 archivos de la entrega se conservan sin modificaciones: 39 en Git y el CSV
original como adjunto de release. La consigna, plantilla y dataset del segundo ZIP
coinciden byte a byte con los incluidos en la entrega y se publican una sola vez.
Se añadieron este README, dependencias, reglas de Git y
[procedencia con hashes SHA-256](docs/procedencia.json).

Se verificaron la sintaxis de los módulos y las celdas, las dimensiones y rango del
dataset, y la separación del split. Sobre datos sintéticos se comprobaron la inversión
del escalado, la reconstrucción PCA completa, k-Means, GMM y Silhouette. También se
confirmó K = 10 como máximo de Silhouette en la tabla guardada de k-Means/PCA.
El [registro de verificación](docs/verificacion_basica.json) detalla alcance y entorno.
No se ejecutaron entrenamientos AE/VAE, t-SNE ni el barrido completo de clustering.

Se mantiene la autoría del trabajo; la consigna, plantilla y datos corresponden a
los materiales de la cátedra. No se añade una licencia sobre estos materiales ni se
reconstruye un historial de desarrollo anterior.
