import numpy as np
import matplotlib.pyplot as plt
import TurkstraMichael_Project4 as proj4

nspace, ntime, tau = 100, 100, 0.1
psi, x_grid, t_grid, prob = proj4.sch_eqn(nspace, ntime, tau, 'ftcs', 200, [], [10, 0, 0.5])
#psi, x_grid, t_grid, prob = proj4.sch_eqn(nspace, ntime, tau, 'ftcs', 200, [0], [10, 0, 0.5])
#psi, x_grid, t_grid, prob = proj4.sch_eqn(nspace, ntime, tau, 'ftcs', 200, [], [10, 0, 0.5])
#psi, x_grid, t_grid, prob = proj4.sch_eqn(nspace, ntime, tau, 'crank', 200, [0], [10, 0, 0.5])
#psi, x_grid, t_grid, prob = proj4.sch_eqn(nspace, ntime, tau, 'crank', 200, [199], [10, 0.5, 10])

total_prob = []
for i in range(len(t_grid)):
    total_prob.append(np.sum(prob[:, i]))

plt.figure()
plt.plot(t_grid, total_prob)
plt.title("Total Probability vs Time")
plt.xlabel("Time")
plt.ylabel("Total Probaility")
plt.ylim(0, 1)
plt.grid()
plt.show()
