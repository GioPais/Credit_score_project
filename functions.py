#Importación de Librerias
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
import sklearn as sk
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import itertools

#Definimos la función para extraer las caracteristicas de cada CSV analizado.
def extraccionDeCaracteristicas(sensado,clase):
    caracts = np.array([clase])
    caracts = np.concatenate((caracts,np.mean(sensado, axis=0)))
    caracts = np.concatenate((caracts,np.var(sensado, axis=0)))
    maxsen = np.amax(sensado, axis=0)
    minsen = np.amin(sensado, axis=0)
    caracts = np.concatenate((caracts,maxsen))
    caracts = np.concatenate((caracts,minsen))
    caracts = np.concatenate((caracts,maxsen - minsen))
    corrfullsen = np.corrcoef(sensado)
    if sensado.shape[1] == 6:
        caracts = np.concatenate((caracts,np.array([corrfullsen[0,1],corrfullsen[0,2],corrfullsen[1,2],corrfullsen[3,4],corrfullsen[3,5],corrfullsen[4,5]])))
    else:
        caracts = np.concatenate((caracts,np.array([corrfullsen[0,1],corrfullsen[0,2],corrfullsen[1,2],corrfullsen[3,4],corrfullsen[3,5],corrfullsen[4,5],corrfullsen[6,7],corrfullsen[6,8],corrfullsen[7,8]])))
    return caracts

def divisionDatos(size, step, datos):
    datos_ = datos.copy()
    datos_ = [datos_[i:i+size,:] for i in range(0,len(datos_)-(size-step),step)]
    return datos_

def extraccionDeCaracteristicasConjuntos(datosConjuntos, clase):
    arregloFinal = np.zeros((len(datosConjuntos),37))
    for i in range(len(datosConjuntos)):
        arregloFinal[i] = extraccionDeCaracteristicas(datosConjuntos[i], clase)
    return arregloFinal

def plot_confusion_matrix(tipo, confusion, confusionmatrix, classes, classes2, normalize, title, cmap=plt.cm.Greens):
    plt.figure(figsize=(6,5))   
    plt.imshow(confusionmatrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes2)

    fmt = '.2f' if normalize else tipo
    thresh = confusionmatrix.max() / 2.
    for i, j in itertools.product(range(confusionmatrix.shape[0]), range(confusionmatrix.shape[1])):
        plt.text(j, i, format(confusionmatrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusionmatrix[i, j] > thresh else "black")

    plt.tight_layout()
    if confusion:
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    plt.show()

def plot_scores(selector, scoresKBest, caractstr):
    fig, ax = plt.subplots(figsize=(8,10))
    y_pos = np.arange(len(caractstr))
    ax.barh(y_pos, selector.scores_, align='center',
            color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(caractstr)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Score')
    ax.set_title('Score de cada característica utilizando KBest')
    plt.show()