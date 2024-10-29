# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:29:14 2024

@author: nacha
"""

import numpy as np
from scipy.linalg import solve_triangular
'''
intercambiarfilas
Esta función toma una matriz A y cambia de lugar dos filas elegidas 
'''
def intercambiarfilas(A, fila1, fila2):
    A[[fila1, fila2]] = A[[fila2, fila1]]
    return A

'''
Facotirzación LU
Esta función toma a una matriz A  y le calcula su factorización LU
con permutaciones si es necesario
'''

def calcularLU(A):
    m, n = A.shape
    '''Si no es matriz cuadrada, no es invertible, 
    entonces no podemos calcular la factorización LU'''
    if m != n:
        print('Matriz no cuadrada')
        return
    '''
    Iniciamos el vector de permutaciones
    '''
    P = np.eye(n)
    Ac = A.copy()
    '''
    Recorremos las filas de la matriz A y si el pivote es cero, intercambiamos
    la fila con la siguiente
    '''  
    for fila in range(m):
        if Ac[fila, fila] == 0:
            '''
            Nos aseguramos de no estar en la última fila
            '''
            if fila + 1 < m: 
                intercambiarfilas(Ac, fila, fila + 1)
                intercambiarfilas(P, fila, fila + 1)
            else:
                print("La matriz no tiene factorización LU.")
                return
        '''Recorremos la matriz Ac. En cada paso, se calcula un factor 
        y se utiliza para restar las filas y obtener la eliminación gaussiana'''
        for i in range(fila + 1, m):
            factor = Ac[i, fila] / Ac[fila, fila]
            Ac[i, fila] = factor  
            Ac[i, fila + 1:] -= factor * Ac[fila, fila + 1:]
        '''Calculamos las matrices L y U que componen la factorización LU de la matriz original.
        L toma la parte triangular inferior estricta de la matriz Ac y le añadimos una matriz identidad''' 
        L = np.tril(Ac, -1) + np.eye(m) 
        U = np.triu(Ac) 
    return L, U, P

'''
inversaLU
Esta función toma la descomposición LU de una matriz y
calcula la inversa de la misma
'''
def inversaLU(L, U, P):
    n = L.shape[0]
    Inv = np.zeros((n, n))  # Inicializa una matriz de ceros
    id = np.eye(n)  # Crea una matriz identidad

    for i in range(n):
        y = solve_triangular(L, np.dot(P, id[:, i]), lower=True)  # Resuelve L * y = P * e_i
        x = solve_triangular(U, y)  # Resuelve U * x = y
        Inv[:, i] = x  # Almacena la columna en Inv

    return Inv

A = np.array([[0.7, 0 ,-0.1],
     [-0.05, 0, -0.2],
     [-0.1, -0.15, 0.9]])

B = np.array([[0.65, 0, 0], 
              [-0.05, 0.5, -0.15],
              [-0.2, -0.3, 0.45]])

L,U,P = calcularLU(A)
LB, UB, PB = calcularLU(B)
inversa = inversaLU(L,U, P)
inversaB = inversaLU(LB, UB, PB)
d = [100, 100, 300]
p = inversa@d


'''
FUNCIONES DEL PUNTO 7
'''
import pandas as pd
data = pd.read_excel("matrizlatina2011_compressed_0.xlsx",sheet_name=1)

'''
generadorMatrizZ 
Esta función toma como parámetros el excel con la información de los paises
y dos paises seleccionados para armar las matrices de flujo de capitales intrarregionales 
e interregionales
'''
def generadorMatrizZ(data,PAIS1,PAIS2):
    '''
    PAIS 1 corresponde a filas y PAIS 2 corresponde a columnas
    '''
    Columnas = data.drop([col for col in data.columns if not col.startswith(PAIS2) and not col.startswith("Country_iso3")], axis = 1)
    FilasYColumnas = Columnas[(Columnas["Country_iso3"]== PAIS1 )]
    Matriz = FilasYColumnas.reset_index(drop=True).drop([col for col in FilasYColumnas.columns if col.startswith("Country_iso3")],axis = 1)
    
    return Matriz
'''
produccionesPais 
Esta función toma como parámetros el excel y un país para calcular el total de producción del mismo
'''
def produccionesPais(data,PAIS):
    total = data.drop([col for col in data.columns if not col.startswith("Output") and not col.startswith("Country_iso3")], axis = 1)
    totalPAIS = total[(total["Country_iso3"]==PAIS)].reset_index(drop=True)
    totalPAIS = totalPAIS.drop([col for col in totalPAIS.columns if col.startswith("Country")],axis = 1)
    totalPAIS = totalPAIS.to_numpy()
    return totalPAIS

'''
ZGrande 
Esta función toma como parámetros las matrices de flujo de capitales intrarregionales 
e interregionales de cada país creadas anteriormente y nos devuelve una matriz con 
toda esa información junta
'''

def ZGrande(ZP1P1,ZP1P2,ZP2P1,ZP2P2):
    arriba = np.hstack((ZP1P1,ZP2P2))
    abajo = np.hstack((ZP2P1,ZP2P2))
    ZMatriz = np.vstack((arriba,abajo))
    return ZMatriz

'''
IdxP
Esta función toma como parámetro un país y devuleve la matriz diagonal con el total de producción
'''
def IdxP(pPAIS):   
  n = pPAIS.shape[0]
  Id = np.eye(n)
  for i in range(len(Id)):
    for j in range(len(Id[i])):
      if i == j :
        Id[i][j] = Id[i][j] * pPAIS[j]
        if Id [i][j] == 0:
            Id[i][j] = 1

  return(Id)

'''
AInsumoProducto
Esta función devuelve la matriz de coeficientes técnicos intrarregional 
'''
def AInsumoProducto(ZP1P2,Id_InvP2):
    AP1P2 = ZP1P2 @ Id_InvP2
    AP1P2 = AP1P2.to_numpy()
    
    return AP1P2
    
'''
AInsumoProductoMultiRegional 
Esta función nos devuelve la Matriz A que nos aporta la información de insumo-producto en economias regionales e interregionales
'''
def AInsumoProductoMultiRegional(ZP1P1,ZP1P2,ZP2P1,ZP2P2,IdP1_inv,IdP2_inv):
    
    #volvemos a crearlas pero dentro de la función a cada matriz de relaciones entre dos paises
    AP1P1 = ZP1P1 @ IdP1_inv
    
    AP1P2 = ZP1P2 @ IdP2_inv
    
    AP2P1 = ZP2P1 @ IdP1_inv
    
    AP2P2 = ZP2P2 @ IdP2_inv
    
    #Pegamos las matrices
    AUp = np.hstack((AP1P1,AP1P2))
    
    Adown = np.hstack((AP2P1,AP2P2))
    #Formamos la matriz de insumo producto que nos interesa ver
    A = np.vstack((AUp,Adown))
    
    return A

'''
SHOCK DE DEMANDA 
la demanda trabaja desde la fila 1 a la 40 sobre la region de Costa rica, y de la 41 a la 80 sobre Nicaragua
USAMOS LA ECUACION DE LEONTIEF PARA MATRICES DE DOS REGIONES COMO SI FUERA RESOLUCION DE SISTEMAS
'''
'''
demandaDeA
Esta función nos calcula la demanda para la matriz A 
'''
def demandaSimple(AP1P1,pPAIS1):
    
    idAP1 = np.eye(AP1P1.shape[0])
    
    dP1= (idAP1 - AP1P1) @ pPAIS1
    
    return dP1
    

def demandaCompleja(AP1P1,AP1P2,AP2P1,AP2P2,pPAIS1,pPAIS2):
    
    idAP1 = np.eye(AP1P1.shape[0])
    
    idAP2 = np.eye(AP2P2.shape[0])
    
    #la demanda vendra como se ve en la ecuación 4 de la Matriz de Leontief como
    #dos vectores traspuestos con 40 valores cada uno de ellos
    
    dP1 = ((idAP1 - AP1P1) @ pPAIS1) + (-AP1P2 @ pPAIS2) #demanda de los sectores del país 1
    
    dP2 = (-AP2P2 @ pPAIS1)  + ((idAP2 - AP2P1) @ pPAIS2) #demanda de los sectores del pais 2
    
    dtotal = np.vstack((dP1,dP2)) #demanda total vector traspuesto de los dos paises y sus 40 sectores de producción cada uno
    
    return dtotal

'''
diferencialShock
Esta función aplica el shock sobre las demandas calculadas
'''
def diferencialShock(demanda,shocks):
    #construyo a dPrima 
    copiaD = demanda.copy()
    n = demanda.shape[0]
    m= demanda.shape[1]
    deltaD = np.zeros((n,m))
    # Ahora generamos a la variación de la demanda como la resta de la 
    #nueva demanda alterada y la demanda despejada de los valores que partimos    
    for i in range(len(shocks)):
        deltaD[[shocks[i][0]-1]] = (copiaD[[shocks[i][0]-1]] * shocks[i][1])
           
    #obtenemos a la variación de la demanda (COMPLETA, tanto de Costa Rica como Nicaragua)
    return deltaD
