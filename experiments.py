import os

import matplotlib.pyplot as plt
import numpy
import numpy as np

# Example matrices A and B with 2 columns each (x, y)
A = np.array([[50.000, 216.500],
              [100.000, 472.000],
              [150.000, 697.500],
              [200.000, 702.040],
              [250.000, 994.590],
              [300.000, 1185.880]]
)



print(A)
aseco_node10 = np.array([[50.000, 40.000],
                         [100.000, 240.000],
                         [150.000, 440.000],
                         [200.000, 520.000],
                         [250.000, 570.000],
                         [300.000, 620.000]])



# Plot for matrix A with a specific color, marker, and format
plt.plot(A[:, 0], A[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
#
# Plot for matrix B with a different color, marker, and format
plt.plot(aseco_node10[:, 0], aseco_node10[:, 1], label='NEPTUNE+', color='green', marker='x', linestyle='--')

# Add labels for x and y axes
plt.xlabel('Workload')
plt.ylabel('Network delay (ms)')

# Add legend
plt.legend()

# Show the plot
save_path = 'plots'
if not os.path.exists(save_path):
    print('AAAAAAAAAA')
    os.makedirs(save_path)
# plt.savefig("plots/neptune.pdf")
plt.savefig(os.path.join(save_path, 'plot_complex(10nodes_varyWorkload).pdf'), format='pdf')

t =numpy.zeros((6,2))
for i in range(6):
    t[i]=i,2*i
print(f'T={t}')

print(f'MEAN={np.mean([2,4,6,8])}')
