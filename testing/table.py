import matplotlib.pyplot as plt
import numpy as np

h = np.array([[1, 2, -2, 5, 4]]).T
hh = h.astype(str)

fig, ax = plt.subplots()

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

ax.table(cellText = hh)

fig.tight_layout()

plt.show()
