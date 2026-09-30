# Asistente Inteligente para la Planeación del Súper Semanal

Proyecto final de **Inteligencia Artificial Aplicada** que combina aprendizaje automático y optimización para apoyar a los hogares en la planeación de menús y compras semanales.

**Autora:** Imelda Natalia Jiménez Altamirano

## Descripción del proyecto

La compra semanal de alimentos implica decisiones relacionadas con presupuesto, inventario, disponibilidad de productos, preferencias alimentarias, alergias y cantidades esperadas de consumo. Una planeación inadecuada puede provocar compras innecesarias, inventario ocioso y desperdicio de alimentos.

Este proyecto propone un asistente inteligente que:

- Analiza patrones históricos de compra, consumo, inventario y desperdicio.
- Predice el consumo esperado de ingredientes mediante un árbol de decisión.
- Selecciona recetas y cantidades de compra mediante optimización CP-SAT.
- Respeta restricciones de presupuesto, inventario, disponibilidad, preferencias y alergias.
- Detecta escenarios infactibles y evita generar recomendaciones incompletas o inseguras.

## Objetivo

Construir un prototipo reproducible que genere un menú semanal y una lista de compra optimizada, minimizando el costo, los sobrantes estimados y la selección de recetas con baja aceptación, sin incumplir las restricciones del hogar.

## Enfoque de Inteligencia Artificial

La solución integra dos componentes complementarios:

### 1. Aprendizaje automático

Se utiliza `DecisionTreeRegressor` de scikit-learn para estimar la cantidad consumida de un ingrediente a partir de variables como:

- Número de integrantes del hogar.
- Número de adultos y niños.
- Cantidad comprada anteriormente.
- Inventario final.
- Nivel de aceptación.

Resultados obtenidos con la partición de prueba de los datos sintéticos:

- **MAE:** 28.64
- **RMSE:** 45.97
- **R²:** 0.96

El valor R² indica la proporción de variabilidad explicada por el modelo en el conjunto de prueba. No representa un porcentaje directo de aciertos.

### 2. Inteligencia Artificial simbólica y optimización

Se utiliza Google OR-Tools CP-SAT para representar el problema mediante variables enteras, una función objetivo y restricciones.

El optimizador decide:

- Cuántas veces seleccionar cada receta.
- Cuántos paquetes comprar de cada producto.
- Cómo utilizar el inventario disponible.
- Cuánto producto podría quedar como sobrante.

La función objetivo minimiza una combinación de:

- Costo total de compra.
- Sobrantes estimados.
- Penalización por recetas con baja aceptación.

Las principales restricciones son:

- Presupuesto semanal.
- Número de preparaciones requeridas.
- Compatibilidad con preferencias alimentarias.
- Exclusión de alérgenos.
- Inventario disponible.
- Existencia de productos.
- Compra de paquetes completos.

## Flujo de trabajo

```text
Archivos CSV sintéticos
        |
        v
Preparación y validación de datos
        |
        v
Análisis exploratorio de datos
        |
        v
Árbol de decisión
        |
        v
Predicción de consumo
        |
        v
Optimización CP-SAT
        |
        v
Menú semanal + lista de compra + diagnóstico
```

## Datos utilizados

El proyecto utiliza seis archivos CSV con datos completamente sintéticos y destinados exclusivamente a fines académicos.

| Archivo | Contenido |
|---|---|
| `hogares.csv` | Integrantes, presupuesto, preferencias y alergias de 10 hogares ficticios. |
| `ingredientes.csv` | 24 ingredientes, categoría, unidad, vida útil y alérgenos asociados. |
| `recetas.csv` | 15 recetas con porciones, tipo de alimentación, aceptación y tiempo de preparación. |
| `receta_ingredientes.csv` | 77 relaciones entre recetas, ingredientes y cantidades requeridas. |
| `catalogo_supermercado.csv` | 48 productos con marca ficticia, presentación, precio y disponibilidad. |
| `historial_consumo.csv` | 768 registros históricos sintéticos de compra, consumo, desperdicio e inventario. |

## Estructura del repositorio

```text
Asistente-Super-Semanal-IA/
|
|-- README.md
|-- requirements.txt
|
|-- data/
|   |-- hogares.csv
|   |-- ingredientes.csv
|   |-- recetas.csv
|   |-- receta_ingredientes.csv
|   |-- catalogo_supermercado.csv
|   `-- historial_consumo.csv
|
|-- notebooks/
|   `-- Proyecto_Final_Asistente_Super.ipynb
|
|-- src/
|   `-- asistente_super_cp_sat.py
|
|-- resultados/
|   |-- menu_H001.csv
|   |-- compras_H001.csv
|   `-- predicciones_H001.csv
|
`-- docs/
    |-- Presentacion_Asistente_Super_Semanal.pptx
    `-- Reporte_Final.pdf
```

Los archivos de `resultados/` y `docs/` pueden aparecer conforme se ejecuta y documenta el proyecto.

## Requisitos

- Python 3.10 o posterior.
- Jupyter Notebook o JupyterLab.
- Dependencias incluidas en `requirements.txt`.

Principales librerías:

- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Google OR-Tools
- Jupyter

## Instalación

Desde la carpeta principal del proyecto, crear un entorno virtual es recomendable para aislar las dependencias.

### Windows PowerShell

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

## Ejecución del notebook

Desde la carpeta principal:

```powershell
py -m notebook
```

Después, abrir:

```text
notebooks/Proyecto_Final_Asistente_Super.ipynb
```

Ejecutar las celdas en orden para reproducir:

1. Preparación del entorno.
2. Carga y validación de los seis CSV.
3. Análisis exploratorio de datos.
4. Entrenamiento y evaluación del árbol de decisión.
5. Predicción para un hogar de prueba.
6. Optimización CP-SAT con recetas y productos de los CSV.
7. Generación del menú y la lista de compra.
8. Prueba de un escenario infactible.

## Ejecución del programa Python

Caso normal para el hogar `H001`:

```powershell
py src/asistente_super_cp_sat.py --datos data --hogar H001 --salida resultados
```

Prueba intencional de presupuesto insuficiente:

```powershell
py src/asistente_super_cp_sat.py --datos data --hogar H001 --presupuesto 0 --salida resultados
```

## Resultados generados

Cuando el modelo encuentra una solución `OPTIMAL` o `FEASIBLE`, genera:

- Predicciones de consumo por ingrediente.
- Menú semanal.
- Lista de compra.
- Número de paquetes por producto.
- Costo estimado.
- Sobrantes estimados.
- Estado del solucionador.

Los archivos se guardan en la carpeta `resultados/`.

## Manejo de casos infactibles

CP-SAT puede devolver diferentes estados, entre ellos:

- `OPTIMAL`: se encontró y demostró la mejor solución para el modelo.
- `FEASIBLE`: se encontró una solución válida.
- `INFEASIBLE`: no existe una solución que cumpla simultáneamente todas las restricciones.
- `MODEL_INVALID`: la formulación del modelo no es válida.
- `UNKNOWN`: no se obtuvo una conclusión dentro de las condiciones de ejecución.

Ante un estado `INFEASIBLE`, el prototipo no genera una recomendación parcial. En su lugar, comunica posibles causas, como presupuesto insuficiente, inventario limitado, falta de productos disponibles o número insuficiente de recetas compatibles.

Las alergias se consideran restricciones obligatorias y no se relajan automáticamente.

## Riesgos éticos y de seguridad

| Riesgo | Mitigación propuesta |
|---|---|
| Recomendación incompatible con una alergia | Modelar alergias como restricciones duras y validar recetas e ingredientes. |
| Datos personales o hábitos de consumo expuestos | Utilizar identificadores anónimos, minimizar datos y definir mecanismos de eliminación. |
| Favorecimiento de marcas | Mostrar la función objetivo y separar criterios de optimización de intereses comerciales. |
| Precios o inventarios desactualizados | Registrar fecha de actualización y solicitar validación antes de la compra. |
| Confianza excesiva en la predicción | Mostrar métricas, limitaciones y mantener revisión humana. |

El prototipo apoya la toma de decisiones y no sustituye asesoría médica o nutricional.

## Limitaciones

- Los datos, precios, marcas, hogares y hábitos de consumo son sintéticos.
- El modelo predictivo fue evaluado únicamente con el conjunto de datos generado para el proyecto.
- No existe integración en tiempo real con supermercados.
- La función objetivo utiliza pesos definidos para el prototipo.
- Las recomendaciones nutricionales no forman parte del alcance actual.
- Los resultados no deben interpretarse como evidencia de desempeño en hogares reales.

## Trabajo futuro

- Integración autorizada con catálogos reales de supermercados.
- Actualización automática de precios y existencias.
- Interfaz web o móvil para captura de preferencias e inventario.
- Validación con datos reales anonimizados y consentimiento informado.
- Comparación con otros modelos predictivos.
- Incorporación de información nutricional validada.
- Análisis de sensibilidad de los pesos de la función objetivo.

## Declaración de uso de Inteligencia Artificial generativa

Se utilizaron herramientas de Inteligencia Artificial generativa como apoyo para estructurar ideas, revisar redacción, diseñar la organización del proyecto, generar datos sintéticos iniciales y producir borradores de código y documentación.

La selección metodológica, ejecución, validación de resultados, interpretación, análisis crítico y conclusiones son responsabilidad de la autora.

## Uso académico

Este repositorio fue desarrollado como proyecto académico. Los datos son ficticios y no representan hogares, precios, marcas, condiciones médicas ni patrones reales de consumo.

## Licencia

Este proyecto se comparte con fines educativos. Antes de reutilizarlo en un contexto productivo, se deben revisar sus dependencias, licencias, requisitos de privacidad, seguridad alimentaria y calidad de datos.
