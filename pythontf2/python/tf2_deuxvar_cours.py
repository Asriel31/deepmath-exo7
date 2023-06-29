import numpy as np
from tensorflow import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
# COPIER-COLLER A PARTIR D'ICI

# Partie A. Données

# Fonction à approcher
def f(x,y):
    return x*np.cos(y)

n = 25  # pour le nb de points dans la grille
xmin, xmax, ymin, ymax = -4.0, 4.0, -4.0, 4.0

VX = np.linspace(xmin, xmax, n)
print(f'VX.shape : {VX.shape}')
VY = np.linspace(ymin, ymax, n)
print(f' VY.shape : {VY.shape}')
X, Y = np.meshgrid(VX, VY)
Z = f(X, Y)

print(f' X.shape : {X.shape} X.reshape(-1,1).shape {X.reshape(-1,1).shape}')
print(f' Y.shape : {Y.shape} Y.reshape(-1,1).shape {Y.reshape(-1,1).shape}')
print(f' Z.shape : {Z.shape} Z.reshape(-1,1).shape {Z.reshape(-1,1).shape}')

entree = np.append(X.reshape(-1,1), Y.reshape(-1,1), axis=1)
print(f' entree.shape : {entree.shape}')
sortie = Z.reshape(-1, 1)
print(f' sortie.shape : {sortie.shape}')

# Partie B. Réseau 

modele = Sequential()
p = 10
modele.add(Dense(p, input_dim=2, activation='relu'))
modele.add(Dense(p, activation='relu'))
modele.add(Dense(p, activation='relu'))
modele.add(Dense(1, activation='linear'))

# Méthode de gradient : descente de gradient classique
mysgd = optimizers.SGD(learning_rate=0.01)
modele.compile(loss='mean_squared_error', optimizer=mysgd)
print(modele.summary())

# Partie C. Apprentissage
# Apprentissage époque par époque à la main

for k in range(1000):
    loss = modele.train_on_batch(entree, sortie)
    print('Erreur :',loss)

# Partie D. Visualisation

sortie_produite = modele.predict(entree)
ZZ = sortie_produite.reshape(Z.shape)  

# sortie produite aux bonnes dimensions
# Affichage

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, color='blue', alpha=0.7)
ax.plot_surface(X, Y, ZZ, color='red', alpha=0.7)
plt.show()




