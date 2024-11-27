import numpy as np
import numpy.linalg as nplinalg

h_bar, m = 1, 1/2

def sch_init_cond(x_points, sig_0, k_0, x_0):
    init_cond = (1/np.sqrt(sig_0)*np.sqrt(np.pi))*np.exp((k_0*x_points*1j) - ((x_points-x_0)**2)/((2*sig_0)**2))
    return init_cond

def make_tridiagonal(Nspace, Ntime, b, d, a):                                                                               ## From Lab 10
    '''
    Function to make matrix following format from lab outline
    Args: Takes in the L length of matrix to make, b value of elements on diagonal-1, d value of elements on diagonal, a value of elements on diagonal+1
    Returns: Matrix that was made
    '''
    if (Nspace > Ntime):
        D = np.zeros((Nspace, Ntime))
        for i in range(Ntime):
            D[i, i] = d
            D[(i + 1)%Nspace, i] = a
            D[(i - 1)%Nspace, i] = b
    elif (Nspace < Ntime):
        D = np.zeros((Nspace, Ntime))
        for i in range(Ntime):
            D[i%Nspace, i] = d
            D[(i + 1)%Nspace, i] = a
            D[(i - 1)%Nspace, i] = b
    else:
        D = d*np.identity(Nspace)+a*np.diagflat(np.ones(Nspace-1),1)+b*np.diagflat(np.ones(Nspace-1),-1)                       ## Equation for making required matrix, d*diagonal + b*diagonal-1, + a*diagonal+1, takes insperation from “lecture10a-FTCS-Diffusion_mtx.ipynb”
        D[Nspace-1, 0], D[0, Nspace-1] = b, a
    return D                                                                                                    ## Return matrix


def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=[], wparam=[10, 0, 0.5]):
    sig0, x0, k0 = wparam
    h = length/nspace
    x_points = np.linspace(-length/2, length/2, nspace)
    t_points = np.arange(0, ntime*tau, tau)
    
    V = np.zeros(nspace)
    for i in potential:
        V[i] = 1

    init_cond = sch_init_cond(x_points, sig0, k0, x0)

    psi = np.zeros((nspace, ntime), dtype=np.complex128)
    psi[:, 0] = init_cond

    H = make_tridiagonal(nspace, ntime, (-(h_bar**2)/(2*m*(h**2))), (2*(h_bar**2)/(2*m*(h**2)))+V, (-(h_bar**2)/(2*m*(h**2))))

    if (method == 'ftcs'):
        for i in range(ntime-1):
            psi[:, i+1] = (make_tridiagonal(nspace, ntime, 0, 1, 0) - (1j*tau/h_bar)*H).dot(psi[:, i])   
    elif (method == 'crank'):
        print(method)
        for i in range(ntime-1):
            psi[:, i+1] = nplinalg.inv(make_tridiagonal(nspace, ntime, 0, 1, 0) + (1j*tau/(2*h_bar))*H).dot(make_tridiagonal(nspace, ntime, 0, 1, 0) - (1j*tau/(2*h_bar))*H).dot(psi[:, i])
    
    prob = np.zeros(ntime)
    
    return psi, x_points, t_points, prob


def main():
    nspace, ntime, tau = 7, 7, 0.1
    psi, x_grid, t_grid, prob = sch_eqn(nspace, ntime, tau, 'crank')
    print(psi)

if __name__ == '__main__':
    main()