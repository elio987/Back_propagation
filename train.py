from random import random
from sklearn.datasets import load_digits
import pickle
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.model_selection import train_test_split
from time import time

digits = load_digits()
print(digits.data.shape)

digitos = digits.images/np.max(digits.images)
X = []
for i in range(len(digitos)):
    X.append(np.reshape(digitos[i],(64,1)))
X = np.array(X)
print(X.shape)

def time_random():
    return time() - float(str(time()).split('.')[0])

def gen_random_range(min, max):
    return int(time_random() * (max - min) + min)

def unique(list1):
  
    # initialize a null list
    unique_list = {}
  
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list[x] = 0
    return unique_list

def train_test_split(X,Y,test_size):
    test_size = int(X.shape[0]*test_size)
    train_size = int(X.shape[0] - test_size)
    print(train_size,test_size)
    randon_numbers = []
    X_test = []
    Y_test = []
    unique_dic = unique(Y)
    size = int(test_size/len(unique_dic))
    paso = 0
    while paso < test_size:
        random = gen_random_range(0,X.shape[0]-1)
        while (random in randon_numbers) and (unique_dic[Y[random]] >= size):
            random = gen_random_range(0,X.shape[0]-1)
        if random not in randon_numbers:
            X_test.append(X[random])
            Y_test.append(Y[random])
            randon_numbers.append(random)
            unique_dic[Y[random]] = unique_dic[Y[random]] + 1
            paso += 1
    X_train = []
    Y_train = []
    for i in range(train_size):
        if i not in randon_numbers:
            X_train.append(X[i])
            Y_train.append(Y[i])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    return (X_train,X_test,Y_train,Y_test)

X_train, X_test, y_train, y_test = train_test_split(X, digits.target, test_size=0.30)
print(X_train.shape)

def sigmoid(x):
    return(1/(1 + np.exp(-x)))

def generate_wt(x, y):
    l =[]
    for i in range(x * y):
        l.append(np.random.randn())
    return(np.array(l).reshape(x, y))

def feed_fwd(x,w0,w1,w2):
    h1 = np.dot(w0,x)
    a1 = sigmoid(h1)
    h2 = np.dot(w1,a1)
    a2 = sigmoid(h2)
    h3 = np.dot(w2,a2)
    a3 = sigmoid(h3)
    return a3
def back_prop(x,y,w0,w1,w2,alpha):
    #Input first layer
    h1 = np.dot(w0,x)
    #Output first layer
    a1 = sigmoid(h1)
    #Input second layer
    h2 = np.dot(w1,a1)
    #Output second layer
    a2 = sigmoid(h2)
    #Input third layer
    h3 = np.dot(w2,a2)
    #Output layer (pred)
    a3 = sigmoid(h3)
  
    op = np.multiply(np.multiply(a3,(1-a3)),(y-a3))
    oh2 = np.multiply(np.multiply(a2,(1-a2)),np.dot(w2.T,op))
    oh3 = np.multiply(np.multiply(a1,(1-a1)),np.dot(w1.T,oh2))
    w2_adj = np.dot(op,a2.T)
    w1_adj = np.dot(oh2,a1.T)
    w0_adj = np.dot(oh3,x.T)

    w0 = w0 + alpha*w0_adj
    w1 = w1 + alpha*w1_adj
    w2 = w2 + alpha*w2_adj
    return (w0,w1,w2)

def train(x, Y, w0, w1, w2, alpha = 0.01, epoch = 10):
    acc =[]
    losss =[]
    for j in range(epoch):
        l =[]
        for i in range(len(x)):
            out = feed_fwd(x[i], w0, w1, w2)
            l.append((loss(out, Y[i])))
            w0, w1, w2 = back_prop(x[i], Y[i], w0, w1, w2, alpha)
        print("epochs:", j + 1, "======== acc:", (1-(sum(l)/len(x)))*100," ======== loss:",sum(l)/len(x))  
        acc.append((1-(sum(l)/len(x))))
        losss.append(sum(l)/len(x))
    return(acc, losss, w0, w1, w2)
def loss(out, Y):
    s =(np.square(out-Y))
    #print(s)
    s = np.sum(s)/4
    #print(np.sum(s),len(Y))
    return s

w0 = generate_wt(32,64)
print(w0.shape)
w1 = generate_wt(16,32)
print(w1.shape)
w2 = generate_wt(4,16)
print(w2.shape)

def target_cod(y):
    salida = []
    for i in y:
        if i == 0:
            salida.append(np.array([[0,0,0,0]]).T)
        elif i == 1:
            salida.append(np.array([[0,0,0,1]]).T)
        elif i == 2:
            salida.append(np.array([[0,0,1,0]]).T)
        elif i == 3:
            salida.append(np.array([[0,0,1,1]]).T)
        elif i == 4:
            salida.append(np.array([[0,1,0,0]]).T)
        elif i == 5:
            salida.append(np.array([[0,1,0,1]]).T)
        elif i == 6:
            salida.append(np.array([[0,1,1,0]]).T)
        elif i == 7:
            salida.append(np.array([[0,1,1,1]]).T)
        elif i == 8:
            salida.append(np.array([[1,0,0,0]]).T)
        elif i == 9:
            salida.append(np.array([[1,0,0,1]]).T)
    return np.array(salida)
y_test_cod = target_cod(y_test)
y_train_cod = target_cod(y_train)
print(y_test_cod.shape)
print(y_train_cod.shape)

alpha = 0.01
epchos = 125
acc,loss,w0,w1,w2 = train(X_train,y_train_cod,w0,w1,w2,alpha,epoch=epchos)

ws = (w0,w1,w2)

with open('model.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(ws, file)

print("Modelo guardado")

test = (X_test,y_test_cod)

with open('test.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(test, file)

print("Test guardado")

fig = plt.figure()

plt.plot(np.arange(1,epchos+1,1), acc)
plt.plot(np.arange(1,epchos+1,1), loss)
plt.title("Acc and loss vs epochs")
plt.legend(["Accuracy","Loss"])
plt.savefig('acc_loss.png')
plt.show()