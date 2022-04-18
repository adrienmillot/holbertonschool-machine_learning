#!/usr/bin/env python3
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

person_names = ['Farrah', 'Fred', 'Felicia']
plt.bar(
    person_names,
    fruit[0],
    color='red',
    label='apples',
    width=0.5
)
plt.bar(
    person_names,
    fruit[1],
    bottom=fruit[0], color='yellow',
    label='bananas',
    width=0.5
)
plt.bar(
    person_names,
    fruit[2],
    bottom=fruit[0]+fruit[1], color='#ff8000',
    label='oranges',
    width=0.5
)
plt.bar(
    person_names,
    fruit[3],
    bottom=fruit[0]+fruit[1]+fruit[2], color='#ffe5b4',
    label='peaches',
    width=0.5
)
plt.title("Number of Fruit per Person")
plt.ylabel("Quantity of Fruit")
plt.ylim(0, 80)
plt.legend()
plt.show()
