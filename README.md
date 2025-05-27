# Back-SV-Route-Optimization CareConnect

## Descripción
El servicio `back-sv-route-optimization` es un componente del sistema CareConnect, desarrollado para el Hospital Universitario San Ignacio. Optimiza las rutas de enfermeras para visitas domiciliarias utilizando un algoritmo genético (PyGAD). Procesa coordenadas, ventanas de tiempo y tiempos de atención, calcula distancias con la fórmula Haversine y genera rutas eficientes, integrándose con otros módulos a través de un API Gateway.

## Funcionalidades
- **Optimización de Rutas**: Genera rutas óptimas (`POST /rutas`) minimizando tiempo total, respetando turnos (6 o 12 horas) y ventanas de tiempo.
- **Cálculo de Distancias**: Crea matriz de distancias a partir de coordenadas geográficas.
- **Resultados**: Devuelve rutas, tiempo total, distancia recorrida, tiempo de espera y tiempo de ejecución.
- **Validación**: Maneja errores de entrada (e.g., datos vacíos, número inválido de enfermeras).

## Tecnologías
- **Framework**: FastAPI
- **Lenguaje**: Python 3.9+
- **Algoritmo**: PyGAD
- **Dependencias**:
  - `fastapi`
  - `pydantic`
  - `pygad`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `pyngrok`
  - `uvicorn`

## Requisitos
- Python 3.9+
- Entorno virtual
- API Gateway activo
- Archivo `.env` (opcional para configuraciones)

## Instalación
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/careconnect/back-sv-route-optimization.git
   cd back-sv-route-optimization
   ```

2. Configurar el entorno virtual e instalar dependencias (ver "Reconstruir el entorno virtual").

3. Iniciar el servicio:
   ```bash
   python main.py
   ```

   Disponible en `http://localhost:8000`, accesible vía API Gateway o ngrok.

## Reconstruir el Entorno Virtual en Otra Máquina
Si otra persona (o tú mismo en otro equipo) clona el repositorio, deberá seguir estos pasos para configurar el entorno virtual correctamente:

1. **Crear un nuevo entorno virtual**:
   ```bash
   python -m venv venv
   ```

2. **Activarlo**:
   - **Windows (CMD o PowerShell)**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

3. **Instalar las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

Estos pasos aseguran que el entorno virtual esté correctamente configurado y que todas las dependencias necesarias sean instaladas para ejecutar el proyecto sin problemas.

## Uso
- **Endpoint**: `POST /rutas`
  - **Cuerpo**:
    ```json
    {
      "coordenadas": [[lat1, lon1], [lat2, lon2]],
      "matrizVentanaTiempo": [[inicioTemprano1, inicioTarde1], ...],
      "tiempoAtencion": [tiempo1, tiempo2],
      "tipoTurno": 6,
      "numMaxEnfermeras": 10
    }
    ```
  - **Respuesta**: Rutas, tiempos, distancias y tiempos de espera.
- **Ejemplo**:
  ```bash
  curl -X POST "http://localhost:8000/rutas" -H "Content-Type: application/json" -d '{"coordenadas": [[4.6, -74.1], [4.7, -74.2]], "matrizVentanaTiempo": [[8, 12], [9, 13]], "tiempoAtencion": [0.5, 0.5], "tipoTurno": 6, "numMaxEnfermeras": 2}'
  ```
  
## Contacto
- Juan David González
- Lina María Salamanca
- Laura Alexandra Rodríguez
- Axel Nicolás Caro

**Pontificia Universidad Javeriana**  
**Mayo 26, 2025**