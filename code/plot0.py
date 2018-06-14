import numpy as np
import matplotlib.pyplot as plt

plt.plot(0.693412, 0.900639, '.', label='Boosting')
plt.plot(0.887354, 0.900639, 's', label='KCF')
plt.plot(0.828978, 0.900639, '^', label='Median')
plt.plot(0.416822, 0.932611, '*', label='MIL')
plt.plot(0.382347, 1.000000, 'o', label='TLD')

plt.plot(0.317974, 0.635408, '.', label='Boosting-Kalman')
plt.plot(0.414395, 0.900639, 's', label='KFC-Kalman')
plt.plot(0.093893, 0.000002, '^', label='Medianflow-Kalman')
plt.plot(0.342144, 0.705508, '*', label='MIL-Kalman')
plt.plot(0.137459, 0.000006, 'o', label='TLD-Kalman')
plt.title('Dataset 0')
plt.legend(loc='best')
plt.ylim(-0.1,1.1)
plt.xlim(0,1)
plt.ylabel('Robustness')
plt.xlabel('Jaccard')

plt.show()
