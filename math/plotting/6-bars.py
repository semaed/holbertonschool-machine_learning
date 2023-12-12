#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.ylim(0, 80)
plt.xticks(np.arange(3), ('Farrah', 'Fred', 'Felicia'))
plt.yticks(np.arange(0, 81, 10))
plt.bar(np.arange(3), fruit[0], width=0.5, color='red', label='apples')
plt.bar(np.arange(3), fruit[1], width=0.5,
        color='yellow', bottom=fruit[0], label='bananas')
plt.bar(np.arange(3), fruit[2], width=0.5, color='#ff8000',
        bottom=fruit[0]+fruit[1], label='oranges')
plt.bar(np.arange(3), fruit[3], width=0.5, color='#ffe5b4',
        bottom=fruit[0]+fruit[1]+fruit[2], label='peaches')
plt.legend()
plt.show()
