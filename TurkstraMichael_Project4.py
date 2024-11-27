import numpy as np

def sch_init_cond(x_points, sig_0, k_0, x_0):
    init_cond = (1/np.sqrt(sig_0)*np.sqrt(np.pi))*np.exp((k_0*x_points*1j) - ((x_points-x_0)**2)/((2*sig_0)**2))


def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=[], wparam=[10, 0, 0.5]):
    sig0, x0, k0 = wparam
    x_points = np.linspace(-length/2, length/2, nspace)
    t_points = np.arange(0, ntime*tau, tau)

    init_cond = sch_init_cond(x_points, sig0, k0, x0)

    psi = np.zeros((ntime, nspace), dtype=np.complex128)
    psi[0, :] = init_cond

    prob = np.zeros(ntime)
    for i in range(ntime):
        prob[i] = psi[i, :]*np.conjugate(psi[i, :])
    
    return psi, x_points, t_points, prob


def main():
    nspace, ntime, tau = 100, 100, 0.1
    psi, x_grid, t_grid, prob = sch_eqn(nspace, ntime, tau)

if __name__ == '__main__':
    main()