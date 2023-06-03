# -*- coding: utf-8 -*-

import tensorflow as tf 
import numpy as np
from matplotlib import pyplot as plt

celsius = np.array([12, -10, 23, 40, 30, 25, -20, 15, 7,], dtype=float)
fahrenheit = np.array([53, 14, 73, 104, 86, 77, -4, 59, 44], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Aprendiendo...")
historial = modelo.fit(celsius, fahrenheit, epochs=400, verbose=False)
print("Listo!")

plt.xlabel("# Epoca")
plt.ylabel("Pérdida")
plt.plot(historial.history["loss"])

print("Prediccion!")
resultado = modelo.predict([100.0])
print("Conversión completada " + str(resultado) + " " "fahrenheit")