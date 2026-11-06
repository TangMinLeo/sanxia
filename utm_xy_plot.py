import matplotlib.pyplot as plt
import numpy as np

file_path = './track_2024/utm_11.npy'
cordi = np.load(file_path)
plt.scatter(cordi[::1000, 0], cordi[::1000, 1], s=1)
plt.show()
