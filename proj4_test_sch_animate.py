import TurkstraMichael_Project4 as proj4

nspace, ntime, tau = 1000, 10000, 0.1
psi, x_grid, t_grid, prob = proj4.sch_eqn(nspace, ntime, tau, 'crank', potential=[800])

proj4.sch_animate(psi, x_grid, t_grid, 'psi')
proj4.sch_animate(prob, x_grid, t_grid, 'prob')