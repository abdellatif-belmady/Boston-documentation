Tout d'abord, exécutons la cellule ci-dessous pour importer tous les paquets dont vous aurez besoin au cours de cette étude.

- [pandas](https://pandas.pydata.org/) est une bibliothèque écrite pour le langage de programmation Python permettant la manipulation et l'analyse des données.

- [numpy](https://numpy.org/doc/1.20/) est une bibliothèque pour langage de programmation Python, destinée à manipuler des matrices ou tableaux multidimensionnels ainsi que des fonctions mathématiques opérant sur ces tableaux.

- [matplotlib](http://matplotlib.org) est une bibliothèque du langage de programmation Python destinée à tracer et visualiser des données sous forme de graphiques.

- [seaborn](https://seaborn.pydata.org/) est une bibliothèque de visualisation Python basée sur matplotlib. Elle fournit une interface de haut niveau pour dessiner des graphiques statistiques attrayants.

- [keras](https://keras.io/) est l'API de haut niveau de TensorFlow.

- [sklearn](https://scikit-learn.org/stable/) est une bibliothèque libre Python destinée à l'apprentissage automatique. 
- [pickle](https://docs.python.org/3/library/pickle.html) est principalement utilisé pour sérialiser et désérialiser une structure objet Python. En d'autres termes, c'est le processus de conversion d'un objet Python en un flux d'octets pour le stocker dans un fichier/base de données, maintenir l'état du programme entre les sessions ou transporter des données sur le réseau.

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
import seaborn as sns
import pickle
from sklearn import svm
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```