# Back_propagation
Programa donde se aplica el algoritmo de back propagation para poder predecir imagenes de numeros escritos a mano usando la dataset de scikit-learn y Python.
- Para este ejercicio la dataset que utilziamos es la digitos de scikit learn, que van de 0 al 9, imagenes de numeros escritos a mano, son imagenes con un tamaño de 8x8.
- En este repositorio utulizamos una red con la siguiente estrucutra:
![Alt text](neural_network.png?raw=true "Red nauronal")
- Como son diez tipos de salidas diferentes pasamos el label de cada imagen a su equivalente en binario de 4 bits.
- Ademas de utilziar las siguientes formulas para poder actualizar los pesos de la red:
- Formula del error de la capa de salida:
$\delta_{k} = O_{k}(1-O_{k})(t_{k}-O_{k})$
- Formula del error de la capas ocultas:
$\delta_{h} = O_{h}(1-O_{h}) \sum_{i = j}^{n} w_{jh}\delta_{k}$
- Formula de la delta w:
$\Delta w = n*\delta_{z}*x_{i}$
- Fomrula de actualziacion del peso:
$w_{i} = w + \alpha * \Delta w$
- Curva de comportamiento del entranamiento del modelo:
![Alt text](acc_loss.png?raw=true "Acc Loss")
- Algunas predicciones hechas:
![Alt text](predicciones.png?raw=true "Acc Loss")
- El archivo train.py entrena la red y guarda los pesos obtenidos en un archivo pkl como la dataset de test.
- El archivo test.py abre estos pesos guardados y la dataset de test del los archivos pkl y muestra la precision del modelo en la dataset de test.
