import numpy as np
import matplotlib.pyplot as plt
import TurkstraMichael_Project4 as proj4

nspace, ntime, tau = 1000, 1000, 0.1

#psi1, x_grid1, t_grid1, prob1 = proj4.sch_eqn(nspace, ntime, tau)
#psi2, x_grid2, t_grid2, prob2 = proj4.sch_eqn(nspace, ntime, tau, 'ftcs', 200, [], [10, 0, 0.5])
#psi3, x_grid3, t_grid3, prob3 = proj4.sch_eqn(nspace, ntime, tau, 'ftcs', 200, [4], [10, 0, 0.5])
#psi4, x_grid4, t_grid4, prob4 = proj4.sch_eqn(nspace, ntime, tau, 'crank', 200, [4], [10, 0, 0.5])
#psi5, x_grid5, t_grid5, prob5 = proj4.sch_eqn(nspace, ntime, tau, 'crank', 200, [4], [5, 0, 1])

#proj4.sch_animate(psi2, x_grid2, t_grid2, 'psi')
#proj4.sch_animate(prob2, x_grid2, t_grid2, 'prob')

#proj4.sch_animate(psi4, x_grid4, t_grid4, 'psi')
#proj4.sch_animate(prob4, x_grid4, t_grid4, 'prob')

#proj4.sch_animate(psi5, x_grid5, t_grid5, 'psi')
#proj4.sch_animate(prob5, x_grid5, t_grid5, 'prob')

nspace, ntime, tau = 100, 100, 0.1
psi, x_grid, t_grid, prob = proj4.sch_eqn(nspace, ntime, tau, 'ftcs', 200, [], [10, 0, 0.5])

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
