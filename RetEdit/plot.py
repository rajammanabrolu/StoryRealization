import matplotlib.pyplot as plt

d = [float(x) for x in open('dist.txt', 'r').read().split(',')]
p = [float(x) for x in open('prob.txt', 'r').read().split(',')]

plt.hist(d)
plt.show()
plt.hist(p)
plt.show()