import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('test.pkl', 'rb') as file:
    myvar = pickle.load(file)
with open('model.pkl', 'rb') as file:
    myvar2 = pickle.load(file)
X_test,y_test_cod = myvar
print(X_test.shape)
w0,w1,w2 = myvar2

def sigmoid(x):
    return(1/(1 + np.exp(-x)))

def loss(out, Y):
    s =(np.square(out-Y))
    #print(s)
    s = np.sum(s)/4
    #print(np.sum(s),len(Y))
    return s

def feed_fwd(x,w0,w1,w2):
    h1 = np.dot(w0,x)
    a1 = sigmoid(h1)
    h2 = np.dot(w1,a1)
    a2 = sigmoid(h2)
    h3 = np.dot(w2,a2)
    a3 = sigmoid(h3)
    return a3

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

l =[]
for i in range(len(X_test)):
  out = feed_fwd(X_test[i], w0, w1, w2)
  l.append(loss(out, y_test_cod[i]))
print("acc:", (1-(sum(l)/len(X_test)))*100," ======== loss:",sum(l)/len(X_test)) 

fig = plt.figure(figsize=(25,25))
lugar = 0
#index = [5, 100,250,300]
for i in range(1, 7):
    plt.subplot(2, 3, i)
    out = np.round(feed_fwd(X_test[lugar],w0,w1,w2))
    out = y_decof(out)
    y = y_decof(y_test_cod[lugar])
    print(out,y)
    plt.imshow(np.reshape(X_test[lugar],(8,8)))
    #plt.plot(np.reshape(X_test,(8,8))*16)
    lugar += 1
    plt.text(-0.5, 0.5, "real: "+str(y)+", pred: "+str(out),
             fontsize=10, ha='right')
plt.show()