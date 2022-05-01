import matplotlib.pyplot as plt
import numpy as np

plt.interactive(True)

arr = np.array([i**2 for i in range(100)])


fig, ax = plt.subplots()


ax.plot(arr)
plt.show()