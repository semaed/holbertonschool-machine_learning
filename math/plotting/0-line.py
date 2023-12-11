#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

plt.plot(y, color='red')
plt.xticks(np.arange(11))
plt.show()
