import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# ==== OPGAVE 1 ====
def plotNumber(nrVector):
    # Let op: de manier waarop de data is opgesteld vereist dat je gebruik maakt
    # van de Fortran index-volgorde – de eerste index verandert het snelst, de 
    # laatste index het langzaamst; als je dat niet doet, wordt het plaatje 
    # gespiegeld en geroteerd. Zie de documentatie op 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html

    nrVector = np.reshape(nrVector, (20, 20), 'F')
    plt.matshow(nrVector)
    plt.show()

# ==== OPGAVE 2a ====
def sigmoid(z):
    # Maak de code die de sigmoid van de input z teruggeeft. Zorg er hierbij
    # voor dat de code zowel werkt wanneer z een getal is als wanneer z een
    # vector is.
    # Maak gebruik van de methode exp() in NumPy.

    # if type(z) is int:
    #     g = 1 / (1 + np.exp(-z))
    # else:
    #     m, n = z.shape
    #     g = np.zeros((m, n))
    #     for x in range(m):
    #         for y in range(n):
    #             g[x, y] = 1 / (1 + np.exp(-z[x, y]))
    return 1 / (1 + np.exp(-z))


# ==== OPGAVE 2b ====
def get_y_matrix(y, m):
    # Gegeven een vector met waarden y_i van 1...x, retourneer een (ijle) matrix
    # van m×x met een 1 op positie y_i en een 0 op de overige posities.
    # Let op: de gegeven vector y is 1-based en de gevraagde matrix is 0-based,
    # dus als y_i=1, dan moet regel i in de matrix [1,0,0, ... 0] zijn, als
    # y_i=10, dan is regel i in de matrix [0,0,...1] (in dit geval is de breedte
    # van de matrix 10 (0-9), maar de methode moet werken voor elke waarde van 
    # y en m

    data = np.ones(m)
    row = np.array(np.arange(m))
    column = np.reshape(y, -1)
    return csr_matrix((data, (row, column-1)), shape=(m, np.amax(y))).toarray()


# ==== OPGAVE 2c ==== 
# ===== deel 1: =====
def predictNumber(Theta1, Theta2, X):
    # Deze methode moet een matrix teruggeven met de output van het netwerk
    # gegeven de waarden van Theta1 en Theta2. Elke regel in deze matrix 
    # is de waarschijnlijkheid dat het sample op die positie (i) het getal
    # is dat met de kolom correspondeert.

    # De matrices Theta1 en Theta2 corresponderen met het gewicht tussen de
    # input-laag en de verborgen laag, en tussen de verborgen laag en de
    # output-laag, respectievelijk. 

    # Een mogelijk stappenplan kan zijn:

    #    1. voeg enen toe aan de gegeven matrix X; dit is de input-matrix a1
    #    2. roep de sigmoid-functie van hierboven aan met a1 als actuele
    #       parameter: dit is de variabele a2
    #    3. voeg enen toe aan de matrix a2, dit is de input voor de laatste
    #       laag in het netwerk
    #    4. roep de sigmoid-functie aan op deze a2; dit is het uiteindelijke
    #       resultaat: de output van het netwerk aan de buitenste laag.

    # Voeg enen toe aan het begin van elke stap en reshape de uiteindelijke
    # vector zodat deze dezelfde dimensionaliteit heeft als y in de exercise.

    a1 = np.c_[np.ones(X.shape[0]), X].transpose()
    z2 = np.dot(Theta1, a1)
    a2 = sigmoid(z2)
    a2 = np.vstack((np.ones(a2.shape[1]), a2))
    z3 = np.dot(Theta2, a2)
    a3 = sigmoid(z3).transpose()
    return a3



# ===== deel 2: =====
def computeCost(Theta1, Theta2, X, y):
    # Deze methode maakt gebruik van de methode predictNumber() die je hierboven hebt
    # geïmplementeerd. Hier wordt het voorspelde getal vergeleken met de werkelijk 
    # waarde (die in de parameter y is meegegeven) en wordt de totale kost van deze
    # voorspelling (dus met de huidige waarden van Theta1 en Theta2) berekend en
    # geretourneerd.
    # Let op: de y die hier binnenkomt is de m×1-vector met waarden van 1...10. 
    # Maak gebruik van de methode get_y_matrix() die je in opgave 2a hebt gemaakt
    # om deze om te zetten naar een matrix.

    m, n = X.shape
    y_matrix = get_y_matrix(y, m)
    predict = predictNumber(Theta1, Theta2, X)
    log_predict = np.log(predict)

    cost = np.multiply(-y_matrix, log_predict) - np.multiply(1 - y_matrix, np.log(1 - predict))

    return np.sum(cost) / m


# ==== OPGAVE 3a ====
def sigmoidGradient(z): 
    # Retourneer hier de waarde van de afgeleide van de sigmoïdefunctie.
    # Zie de opgave voor de exacte formule. Zorg ervoor dat deze werkt met
    # scalaire waarden en met vectoren.
    amount = sigmoid(z)
    g = amount * (1 - amount)
    return g

# ==== OPGAVE 3b ====
def nnCheckGradients(Theta1, Theta2, X, y): 
    # Retourneer de gradiënten van Theta1 en Theta2, gegeven de waarden van X en van y
    # Zie het stappenplan in de opgaven voor een mogelijke uitwerking.

    Delta1 = np.zeros(Theta1.shape)
    Delta2 = np.zeros(Theta2.shape)

    m, n = X.shape
    y_matrix = get_y_matrix(y, m)

    a1 = np.c_[np.ones(X.shape[0]), X].transpose()
    z2 = np.dot(Theta1, a1)
    a2 = sigmoid(z2)
    a2 = np.vstack((np.ones(a2.shape[1]), a2))
    z3 = np.dot(Theta2, a2)
    a3 = sigmoid(z3).transpose()

    delta3 = a3 - y_matrix
    delta2 = np.empty((5000, 26))
    a2_transposed = np.transpose(a2)
    a1_transposed = np.transpose(a1)

    # Delta2
    for i in range(m):
        dot_product = np.dot(np.transpose(Theta2), delta3[i])
        delta2[i] = dot_product * sigmoidGradient(a2_transposed[i])

    total2 = np.sum((a2_transposed * delta2), axis=0)

    for i in range(10):
        Delta2[i] = total2


    # Delta1
    delta1 = np.empty((5000, 401))

    for i in range(m):
        dot_product = np.dot(np.transpose(Theta1), delta2[i][1:])
        delta1[i] = dot_product * sigmoidGradient(a1_transposed[i])

    total1 = np.sum((a1_transposed * delta1), axis=0)

    for i in range(25):
        Delta1[i] = total1


    Delta1_grad = Delta1 / m
    Delta2_grad = Delta2 / m
    
    return Delta1_grad, Delta2_grad
