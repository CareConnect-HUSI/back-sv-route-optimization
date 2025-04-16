import pandas as pd  # type: ignore
import pygad # type: ignore
import numpy as np # type: ignore
import time
from fastapi import FastAPI, HTTPException, Request # type: ignore
from fastapi.exceptions import RequestValidationError # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from sklearn.metrics import pairwise_distances  # type: ignore
from pydantic import BaseModel, ValidationError # type: ignore
from pyngrok import ngrok # type: ignore
import uvicorn # type: ignore
from datetime import datetime, timedelta
import json
from typing import Optional

#Instalar FastAPI
# %pip install fastapi
# %pip install uvicorn
#Instalar ngrok
# %pip install pyngrok
#Instalar pydantic
# %pip install pydantic
#Instalar pygad
# %pip install pygad
#Instalar scikit-learn
# %pip install scikit-learn

# Convertir inputs a un array de NumPy

# RECIBIR COORDENADAS

numPersonas = 0
numMaxEnfermeras = 10
tiempoAtencion = [] # Horas
tiempoMaximo = 0.0 # Horas
matrizVentanaTiempo = [[]]
matrizDistancias = [[]]
inicioTemprano = 0
inicioTarde = 1
horaInicial = datetime.strptime("00:00", "%H:%M").time()

rutasEnfermeras = [[]]
velPromedio = 20 # Velocidad promedio es 20km/h

def fitness_func(ga_instance, solution, solution_idx):
    # Calcular el fitness de la solución calculando la FO
    tiempoTotal = 0.0
    solucionInicial = solution.copy()
    numEnfermeras = 0
    rutasEnfermeras.clear()
    enfermerasUtilizadas = []

    while solucionInicial.size > 0 and len(enfermerasUtilizadas) <= numMaxEnfermeras:
    # while solucionInicial.size > 0:
        rutaActual = []

        fila_con_indices = [(i, valor) for i, valor in enumerate(matrizDistancias[solucionInicial[0]])]
        fila_ordenada = sorted(fila_con_indices, key=lambda x: x[1])
        fila_ordenada.pop(0)

        while fila_ordenada:
          if fila_ordenada[0][0] >= numMaxEnfermeras or fila_ordenada[0][0] in enfermerasUtilizadas:
            fila_ordenada.pop(0)
          else:
            break

        if fila_ordenada:
          enfermerasUtilizadas.append(fila_ordenada[0][0])

          tiempoEnfermera = 0.0

          rutaActual.append(enfermerasUtilizadas[-1])

          tiempoEnfermera += matrizVentanaTiempo[solucionInicial[0]][inicioTemprano] #Tiempo de espera

          #tiempoEnfermera += matrizDistancias[rutaActual[-1]][solucionInicial[0]] / velPromedio #Tiempo desde la enfermera hasta el paciente, no se tiene en cuenta
          rutaActual.append(solucionInicial[0])

          tiempoEnfermera += tiempoAtencion[solucionInicial[0]]
          
          solucionInicial = np.delete(solucionInicial,0)

          while tiempoEnfermera < tiempoMaximo and solucionInicial.size > 0:
            if matrizVentanaTiempo[solucionInicial[0]][inicioTemprano] > tiempoEnfermera:
              if (tiempoEnfermera + (matrizDistancias[rutaActual[-1]][solucionInicial[0]] / velPromedio)+ tiempoAtencion[solucionInicial[0]]) + (matrizVentanaTiempo[solucionInicial[0]][inicioTemprano] - tiempoEnfermera) > tiempoMaximo:
                break
              else:
                tiempoEnfermera += (matrizVentanaTiempo[solucionInicial[0]][inicioTemprano] - tiempoEnfermera) #Tiempo de espera
                tiempoEnfermera += tiempoAtencion[solucionInicial[0]]
                tiempoEnfermera += matrizDistancias[rutaActual[-1]][solucionInicial[0]] / velPromedio
                rutaActual.append(solucionInicial[0])
                solucionInicial = np.delete(solucionInicial,0)

                
            elif matrizVentanaTiempo[solucionInicial[0]][inicioTarde] > tiempoEnfermera:
              if (tiempoEnfermera + (matrizDistancias[rutaActual[-1]][solucionInicial[0]] / velPromedio)+ tiempoAtencion[solucionInicial[0]]) > tiempoMaximo:
                break
              else:
                tiempoEnfermera += tiempoAtencion[solucionInicial[0]]
                tiempoEnfermera += matrizDistancias[rutaActual[-1]][solucionInicial[0]] / velPromedio
                rutaActual.append(solucionInicial[0])
                solucionInicial = np.delete(solucionInicial,0)
            else:
              break

          tiempoTotal += tiempoEnfermera
          
          rutasEnfermeras.append(rutaActual)
        else:
          break

    if len(solucionInicial) > 0: #NO atiende a todos los pacientes se penaliza
        fitness = 0.0
    else:
      fitness = 1.0 / (tiempoTotal + 0.0000000001)  # Para evitar división por cero
    return fitness

def calcularFO( solution, solution_idx):
    global horaInicial
    
    rutasFinales = []
    rutasEnfermeras.clear()

    horasIniciales = []
    horasIniciales.clear()

    horasFinales = []
    horasFinales.clear()

    tiempoTotal = 0.0
    solucionInicial = solution.copy()
    numEnfermeras = 0
    distanciaEnfermeras = []
    tiempoEnfermeras = []
    enfermerasUtilizadas = []
    tiempoEspera = []


    while solucionInicial.size > 0 and len(enfermerasUtilizadas) <= numMaxEnfermeras:
        rutaActual = []
        horaInicio = []
        horaFinal = []
        tiempoEsperaEnfermera = 0.0


        fila_con_indices = [(i, valor) for i, valor in enumerate(matrizDistancias[solucionInicial[0]])]
        fila_ordenada = sorted(fila_con_indices, key=lambda x: x[1])
        fila_ordenada.pop(0)

        while fila_ordenada:
          if fila_ordenada[0][0] >= numMaxEnfermeras or fila_ordenada[0][0] in enfermerasUtilizadas:
            fila_ordenada.pop(0)
          else:
            break
        if fila_ordenada:
          enfermerasUtilizadas.append(fila_ordenada[0][0])

          tiempoEnfermera = 0.0
          horaActualFloat = horas_a_decimal(horaInicial)
          distanciaEnfermera = 0.0
          rutaActual.append(enfermerasUtilizadas[-1])

          horaActual = decimal_a_horas(horaActualFloat)
          horaInicio.append(horaActual) #Hora inicio y hora final enfermera
          horaFinal.append(horaActual) #Hora inicio y hora final enfermera
          

          tiempoEnfermera += matrizVentanaTiempo[solucionInicial[0]][inicioTemprano] #Tiempo de espera
          tiempoEsperaEnfermera += matrizVentanaTiempo[solucionInicial[0]][inicioTemprano]

          horaActualFloat = horas_a_decimal(horaInicial) + tiempoEnfermera  #Se suma el tiempo inicial con el tiempo de espera de la enfermera
          horaActual = decimal_a_horas(horaActualFloat)
          horaInicio.append(horaActual) # Se agrega la hora de inicio del primer paciente

          #tiempoEnfermera += matrizDistancias[rutaActual[-1]][solucionInicial[0]] / velPromedio #Tiempo de recorrido NO SE CUENTA EL TIEMPO DESDE SU CASA AL PRIMER PACIENTE
          distanciaEnfermera += matrizDistancias[rutaActual[-1]][solucionInicial[0]]
          rutaActual.append(solucionInicial[0])  

          tiempoEnfermera += tiempoAtencion[solucionInicial[0]]
          solucionInicial = np.delete(solucionInicial,0)
          

          while tiempoEnfermera < tiempoMaximo and solucionInicial.size > 0:  # Verificamos si hay más pacientes en solucionInicial
            if matrizVentanaTiempo[solucionInicial[0]][inicioTemprano] > tiempoEnfermera:
              if (tiempoEnfermera + (matrizDistancias[rutaActual[-1]][solucionInicial[0]] / velPromedio) + tiempoAtencion[solucionInicial[0]])+(matrizVentanaTiempo[solucionInicial[0]][inicioTemprano] - tiempoEnfermera) > tiempoMaximo:
                horaActualFloat = horas_a_decimal(horaInicial) + tiempoEnfermera  #Se suma el tiempo inicial con el tiempo de recorrido y atención de la enfermera
                horaActual = decimal_a_horas(horaActualFloat)
                horaFinal.append(horaActual) # hora final enfermera
                break
              else:
                tiempoEnfermera += matrizDistancias[rutaActual[-1]][solucionInicial[0]] / velPromedio #Tiempo de recorrido
                horaActualFloat = horas_a_decimal(horaInicial) + tiempoEnfermera
                horaActual = decimal_a_horas(horaActualFloat)
                horaFinal.append(horaActual) # hora final enfermera

                tiempoEnfermera += (matrizVentanaTiempo[solucionInicial[0]][inicioTemprano] - tiempoEnfermera) #Tiempo de espera
                tiempoEsperaEnfermera += (matrizVentanaTiempo[solucionInicial[0]][inicioTemprano] - tiempoEnfermera)

                horaActualFloat = horas_a_decimal(horaInicial) + tiempoEnfermera
                horaActual = decimal_a_horas(horaActualFloat)
                horaInicio.append(horaActual) # Se agrega la hora de inicio sin contar el tiempo de espera

                tiempoEnfermera += tiempoAtencion[solucionInicial[0]] #Tiempo de atención
                distanciaEnfermera += matrizDistancias[rutaActual[-1]][solucionInicial[0]]
                rutaActual.append(solucionInicial[0])
                solucionInicial = np.delete(solucionInicial,0)
                if solucionInicial.size == 0:
                  horaActualFloat = horas_a_decimal(horaInicial) + tiempoEnfermera
                  horaActual = decimal_a_horas(horaActualFloat)
                  horaFinal.append(horaActual) # hora final enfermera
                
            elif matrizVentanaTiempo[solucionInicial[0]][inicioTarde] > tiempoEnfermera:
              if (tiempoEnfermera + (matrizDistancias[rutaActual[-1]][solucionInicial[0]] / velPromedio) + tiempoAtencion[solucionInicial[0]]) > tiempoMaximo:
                horaActualFloat = horas_a_decimal(horaInicial) + tiempoEnfermera  #Se suma el tiempo inicial con el tiempo de recorrido y atención de la enfermera
                horaActual = decimal_a_horas(horaActualFloat)
                horaFinal.append(horaActual) # hora final enfermera
                break
              else:
                tiempoEnfermera += matrizDistancias[rutaActual[-1]][solucionInicial[0]] / velPromedio #Tiempo de recorrido

                horaActualFloat = horas_a_decimal(horaInicial) + tiempoEnfermera  #Se suma el tiempo inicial con el tiempo de recorrido y atención de la enfermera
                horaActual = decimal_a_horas(horaActualFloat)
                horaFinal.append(horaActual) # hora final enfermera

                horaActualFloat = horas_a_decimal(horaInicial) + tiempoEnfermera
                horaActual = decimal_a_horas(horaActualFloat)
                horaInicio.append(horaActual) # Se agrega la hora de inicio sin contar el tiempo de espera

                tiempoEnfermera += tiempoAtencion[solucionInicial[0]] #Tiempo de atención
                distanciaEnfermera += matrizDistancias[rutaActual[-1]][solucionInicial[0]]
                rutaActual.append(solucionInicial[0])
                solucionInicial = np.delete(solucionInicial,0)
                if solucionInicial.size == 0:
                  horaActualFloat = horas_a_decimal(horaInicial) + tiempoEnfermera
                  horaActual = decimal_a_horas(horaActualFloat)
                  horaFinal.append(horaActual) # hora final enfermera
            else:
              horaActualFloat = horas_a_decimal(horaInicial) + tiempoEnfermera  #Se suma el tiempo inicial con el tiempo de recorrido y atención de la enfermera
              horaActual = decimal_a_horas(horaActualFloat)
              horaFinal.append(horaActual) # hora final enfermera
              break
          
          horasIniciales.append(horaInicio) #Se agregan las horas iniciales de la enfermera
          horasFinales.append(horaFinal) #Se agregan las horas finales de la enfermera
          tiempoTotal += tiempoEnfermera
          rutasEnfermeras.append(rutaActual)
          tiempoEnfermeras.append(tiempoEnfermera)
          rutasFinales.append(rutaActual)
          distanciaEnfermeras.append(distanciaEnfermera)
          tiempoEspera.append(tiempoEsperaEnfermera)
        else:
          break
    if len(solucionInicial) > 0: #NO atiende a todos los pacientes se penaliza
        fitness = 0.0
    else:
      fitness = 1.0 / (tiempoTotal + 0.0000000001)  # Para evitar división por cero
    return rutasFinales, tiempoTotal, tiempoEnfermeras, distanciaEnfermeras, tiempoEspera, horasIniciales, horasFinales


def principal(tiempoTurnoSelec, numMaxEnfermerasSelec, matrizDistanciasIn, matrizVentanaTiempoIn, tiempoAtencionIn):
  
  
  global tiempoMaximo, numMaxEnfermeras, matrizVentanaTiempo, matrizDistancias, numPersonas, tiempoAtencion

  matrizVentanaTiempo = matrizVentanaTiempoIn
  matrizDistancias = matrizDistanciasIn
  tiempoAtencion = tiempoAtencionIn

  numMaxEnfermeras = numMaxEnfermerasSelec
  numPersonas = len(matrizDistancias)

  tiempoMaximo = tiempoTurnoSelec

  numPacientes = numPersonas - numMaxEnfermeras

  gene_space = [i for i in range(numMaxEnfermeras, numPersonas)]

  # Configuración del algoritmo genético
  ga_instance = pygad.GA(num_generations=1000,          # Número de generaciones
                        sol_per_pop=30,                 # Población por generación
                        num_genes= numPacientes,        # Número de genes en cada solución
                        gene_space=gene_space,          # Espacio de genes
                        fitness_func=fitness_func,      # Función de fitness
                        gene_type=int,                  # Tipo de genes (enteros)
                        allow_duplicate_genes= False,   # Permitir genes duplicados
                        init_range_low=numMaxEnfermeras,             # Límite inferior del rango de genes
                        init_range_high=numPersonas,    # Límite superior del rango de genes
                        num_parents_mating=6,           # Padres que se aparean
                        mutation_type="random",         # Tipo de mutación
                        mutation_probability=0.1)       # Probabilidad de mutación
  
  tiempo_inicio = time.time()
  for i in range(0,3):
    fitnessEvaluar = 0.0
    # Ejecución del algoritmo genético

    ga_instance.run()
    # Obtener la mejor solución
    
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    if solution_fitness > fitnessEvaluar:
      rutasFinal, tiempoTotal, tiempoEnfermeras, distanciaEnfermeras, tiempoEspera, horasInicio, horasFinal = calcularFO(solution, solution_idx)
      fitnessEvaluar = solution_fitness
  tiempo_fin = time.time()

  

  if fitnessEvaluar == 0.0:
      resultadosGenerales = pd.DataFrame(
          {"Mensaje": ["No se encontró una solución válida. Trate de cambiar el número de enfermeras o tipo de turno!!!"]}
      )
      return [resultadosGenerales, ""]
  else:
      resultadosGenerales = pd.DataFrame({
          "Numero de enfermeras": [len(rutasFinal)],
          "Tiempo total (horas)": [round(tiempoTotal, 2)],
          "Distancia total (km)": [round(sum(distanciaEnfermeras), 2)],
          "Tiempo espera total (horas)": [round(sum(tiempoEspera), 2)],
          "Tiempo ejecución algoritmo (segundos)": [round(tiempo_fin - tiempo_inicio, 2)]
      })

      resultados = ""
      for i, (ruta, tiempo, distancia, tiempoEsperaEnfermera) in enumerate(zip(rutasFinal, tiempoEnfermeras, distanciaEnfermeras, tiempoEspera)):
          resultados += f"Enfermera {int(ruta[0])}:\n"
          resultados += f"  Ruta: {[int(n) for n in ruta]}\n"
          resultados += f"  Tiempo utilizado: {tiempo:.2f} horas\n"
          resultados += f"  Distancia recorrida: {distancia:.2f} km\n"
          resultados += f"  Tiempo Espera: {tiempoEsperaEnfermera:.2f} horas\n"

      return [resultadosGenerales, resultados, rutasFinal, horasInicio, horasFinal]
  
def decimal_a_horas(decimal):
    horas = int(decimal)
    minutos = int((decimal - horas) * 60)
    return f"{horas:02d}:{minutos:02d}"

def horas_a_decimal(horas_str):
    if not isinstance(horas_str, str):
        horas_str = horas_str.strftime("%H:%M")  # Convertir datetime.time a string

    horas, minutos = map(int, horas_str.split(":"))
    return horas + minutos / 60



#///////////////////////////////////////// FastAPI //////////////////////////////////////////

class InputData(BaseModel):
    coordenadas: list
    matrizVentanaTiempo: list
    tiempoAtencion: list
    tipoTurno: float
    numMaxEnfermeras: int
    ids: Optional[list] = None  # Hacer conjunto opcional
    horaInicio: Optional[str] = None  # Hacer conjunto opcional

app = FastAPI()

@app.get("/")
def home():
    return {"mensaje": "¡Hola, este es un web service para CareConnect!"}

# Manejador global para errores de validación (422)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": "Error de validación: El cuerpo de la solicitud es inválido o faltan campos requeridos."},
    )

@app.post("/rutas")
def rutas(datos: InputData):

  try:
    global horaInicial
    if datos.horaInicio: 
       horaInicial = datetime.strptime(datos.horaInicio, "%H:%M").time()
    if not datos.coordenadas:
      raise HTTPException(status_code=400, detail="Las coordenadas son obligatorias.")
    if not datos.matrizVentanaTiempo:
      raise HTTPException(status_code=400, detail="La matriz de ventana de tiempo es obligatoria.")
    if not datos.tiempoAtencion:
      raise HTTPException(status_code=400, detail="Los tiempos de atención son obligatorios.")
    if datos.numMaxEnfermeras <= 0:
      raise HTTPException(status_code=400, detail="El número de enfermeras debe ser mayor a 0.")

    # Convertir datos de entrada a matrices NumPy
    coordenadas = np.array(datos.coordenadas, dtype=float)
    matrizVentanaTiempo = np.array(datos.matrizVentanaTiempo, dtype=float)
    tiempoAtencion = np.array(datos.tiempoAtencion, dtype=float)
    tipoTurnoSelec = datos.tipoTurno
    numMaxEnfermeras = datos.numMaxEnfermeras

    # Crear DataFrame con coordenadas
    df_coordenadas = pd.DataFrame(coordenadas, columns=['Latitude', 'Longitude'])

    # Convertir coordenadas a radianes
    df_coordenadas['Latitude'] = np.radians(df_coordenadas['Latitude'])
    df_coordenadas['Longitude'] = np.radians(df_coordenadas['Longitude'])
    df_coordenadas[['Latitude','Longitude']].to_numpy()

    # Calcular la matriz de distancias utilizando Haversine
    dist = pairwise_distances(df_coordenadas[['Latitude', 'Longitude']].to_numpy(), metric='haversine') * 6373 # Radio de la Tierra en km
    matrizDistancias = pd.DataFrame(dist)

    if tipoTurnoSelec == 6:
      tiempoMaximoSelec = 5.75
    elif tipoTurnoSelec == 12:
      tiempoMaximoSelec = 11

    # Llamar a la función principal con los datos recibidos
    resultados = principal(tiempoMaximoSelec, numMaxEnfermeras, matrizDistancias, matrizVentanaTiempo, tiempoAtencion)

    resultadosGenerales = resultados[0]
    
    if resultados[1] == "":
      return {"resultados_generales": "DATOS LEIDOS CORRECTAMENTE",
      "resultados": resultadosGenerales.to_dict(orient="records")}
    else:
      # Preparar la respuesta en JSON
      
      rutas_brutas = resultados[2]
      horas_inicio = resultados[3]
      horas_fin = resultados[4]
      rutas_con_horas = {}

      for idx_ruta, ruta in enumerate(rutas_brutas):
        ruta_con_horas = []
        for i, indice_paciente in enumerate(ruta):
          if not datos.ids:
            paciente_id = str(indice_paciente)
          else:
              paciente_id = datos.ids[indice_paciente]

          entrada = {
                  "paciente": paciente_id,
                  "hora_inicio": horas_inicio[idx_ruta][i],
                  "hora_fin": horas_fin[idx_ruta][i]
              }
          ruta_con_horas.append(entrada)

          # Usar el primer paciente de la ruta como clave
        if ruta_con_horas:
          rutas_con_horas[ruta_con_horas[0]["paciente"]] = ruta_con_horas
        else:
          print(f"Ruta vacía detectada en índice {idx_ruta}: {ruta}")
      return {
          "resultados_generales": resultadosGenerales.to_dict(orient="records"),
          "rutas": rutas_con_horas}
  except ValidationError as e:
        
        raise HTTPException(status_code=422, detail=f"Error de validación: {str(e)}")
    
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))

public_url = ngrok.connect(8000, "http")
print(f"Web service accesible en: {public_url}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


