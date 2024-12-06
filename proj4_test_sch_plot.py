import TurkstraMichael_Project4 as proj4

nspace, ntime, tau = 100, 20, 0.1
psi, x_grid, t_grid, prob = proj4.sch_eqn(nspace, ntime, tau)

proj4.sch_plot(psi, x_grid, t_grid, 'psi', False, 10)
proj4.sch_plot(prob, x_grid, t_grid, 'prob', False, 10)