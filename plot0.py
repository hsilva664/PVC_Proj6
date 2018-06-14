import numpy as np
import matplotlib.pyplot as plt

jac = 100*np.array([0.693412,0.887354,0.828978,0.416822,0.382347])
rob = np.array([0.900639,0.900639,0.900639,0.932611,1.000000],dtype=np.int32)
#e = 100*np.array([0.205769, 0.067192, 0.221534,0.250995,0.196488])

#jac2 = np.array([0, 0.076156, 0.098249, 0.114167, 0.151805, 0.156505])
#e2 = np.array([0, 0.115459, 0.131384, 0.145297, 0.152405, 0.182161])

plt.plot(0.693412, 0.900639, '-', 0.887354, 0.900639, 's', 0.828978, 0.900639, '^',0.416822, 0.932611,'*',0.382347,1.000000,'o')
plt.show()
plt.axis([-0.1, 1.1, 0, 101])
plt.ylabel('Robustness')
plt.xlabel('Jaccard')

plt.savefig('results_0.png', bbox_inches='tight')
