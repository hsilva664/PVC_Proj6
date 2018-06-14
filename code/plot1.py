import numpy as np
import matplotlib.pyplot as plt

plt.plot(0.564843, 0.901292, '.', label='Boosting')
plt.plot(0.581136, 0.926619, 's', label='KCF')
plt.plot(0.777721, 0.356175, '^', label='Median')
plt.plot(0.324462, 0.926619, '*', label='MIL')
plt.plot(0.397417, 0.031190, 'o', label='TLD')

plt.plot(0.142367, 0.001664, '.', label='Boosting-Kalman')
plt.plot( 0.159076, 0.000046, 's', label='KFC-Kalman')
plt.plot(0.044651, 0.000001, '^', label='Medianflow-Kalman')
plt.plot(0.147397, 0.003282, '*', label='MIL-Kalman')
plt.plot(0.028832, 0.000001, 'o', label='TLD-Kalman')

plt.title('Dataset 1')
plt.legend(loc='best')
plt.ylim(-0.1,1.1)
plt.xlim(0,1)
plt.ylabel('Robustness')
plt.xlabel('Jaccard')

plt.show()
