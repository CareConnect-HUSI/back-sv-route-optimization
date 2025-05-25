# Imagen base
FROM python:3.10-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos al contenedor
COPY . /app

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto interno
EXPOSE 8086

# Comando para iniciar FastAPI con Uvicorn
CMD ["uvicorn", "moduloOptimizacion:app", "--host", "0.0.0.0", "--port", "8086"]
