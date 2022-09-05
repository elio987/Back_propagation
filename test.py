#Leonardo Gracida Munoz A01379812
import pickle
import matplotlib.pyplot as plt
import numpy as np

#Abrimos la dataset de test
with open('test.pkl', 'rb') as file:
    myvar = pickle.load(file)
#Abrimos los pesos geenrados
with open('model.pkl', 'rb') as file:
    myvar2 = pickle.load(file)

X_test,y_test_cod = myvar
w0,w1,w2 = myvar2

#Funcion de activacion de la neuronas
def sigmoid(x):
    return(1/(1 + np.exp(-x)))

#Funcion de perdida
def loss(out, Y):
    s =(np.square(out-Y))
    #print(s)
    s = np.sum(s)/4
    #print(np.sum(s),len(Y))
    return s

#Funcion para predecir con la red entranada
def feed_fwd(x,w0,w1,w2):
    h1 = np.dot(w0,x)
    a1 = sigmoid(h1)
    h2 = np.dot(w1,a1)
    a2 = sigmoid(h2)
    h3 = np.dot(w2,a2)
    a3 = sigmoid(h3)
    return a3

#Funcion que traduce la salida de la red a una label numerico
def y_decof(y):
    if (y == np.array([[0,0,0,0]]).T).all():
        return 0.0
    elif (y == np.array([[0,0,0,1]]).T).all():
        return 1.0
    elif (y == np.array([[0,0,1,0]]).T).all():
        return 2.0
    elif (y == np.array([[0,0,1,1]]).T).all():
        return 3.0
    elif (y == np.array([[0,1,0,0]]).T).all():
        return 4.0
    elif (y == np.array([[0,1,0,1]]).T).all():
        return 5.0
    elif (y == np.array([[0,1,1,0]]).T).all():
        return 6.0
    elif (y == np.array([[0,1,1,1]]).T).all():
        return 7.0
    elif (y == np.array([[1,0,0,0]]).T).all():
        return 8.0
    elif (y == np.array([[1,0,0,1]]).T).all():
        return 9.0

#Obtenemos la accuray del modelo con el dataset de test
l =[]
for i in range(len(X_test)):
  out = feed_fwd(X_test[i], w0, w1, w2)
  l.append(loss(out, y_test_cod[i]))
print("acc:", (1-(sum(l)/len(X_test)))*100," ======== loss:",sum(l)/len(X_test)) 

#Ploteamos la unas cuantas imagenes y mostramos la predccion junto al label real
fig = plt.figure()
lugar = 0
for i in range(1, 7):
    plt.subplot(2, 3, i)
    #Predecimos
    out = feed_fwd(X_test[lugar],w0,w1,w2)
    for i in range(out.shape[0]):
        if out[i] <= 0.4:
            out[i] = 0
        else:
            out[i] = 1
    #Lo pasamos a un label numerico
    out = y_decof(out)
    #pasamos el label real a un label numerico
    y = y_decof(y_test_cod[lugar])
    print("Prediccion: ",out,", Real: ",y)
    plt.imshow(np.reshape(X_test[lugar],(8,8)), cmap='gray')
    plt.title("real: "+str(y)+", pred: "+str(out))
    lugar += 1
#Guardamos el modelo
plt.savefig('predicciones.png')
plt.show()