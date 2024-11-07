import matplotlib.pyplot as plt
import numpy as np

x = [100,200,300,400,500]  # Generates 100 values from 0 to 10
y1 = [94.28, 99.17, 563.97, 946, 1021.83]               # First curve: sine function
y2 = [93.61, 96.76, 547.38, 851.32, 867.51]             # Second curve: cosine function

plt.plot(x, y1, label='naive', color='blue', linestyle='-')  # Plot first curve
plt.plot(x, y2, label='memory-based', color='red', linestyle='--')  # Plot second curve

plt.xlabel('number of queries')
plt.ylabel('makespan (seconds)')
plt.title('Doris Scheduling Result')
plt.legend()
plt.savefig('doris_scheduling_result.png')