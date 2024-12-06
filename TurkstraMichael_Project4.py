import numpy as np
import numpy.linalg as nplinalg
import matplotlib.pyplot as plt
from celluloid import Camera

h_bar, m = 1, 1/2

def sch_init_cond(x_points, sig_0, k_0, x_0):
    init_cond = (1/np.sqrt(sig_0)*np.sqrt(np.pi))*np.exp((k_0*x_points*1j) - ((x_points-x_0)**2)/((2*sig_0)**2))
    return init_cond

def make_tridiagonal(Nspace, b, d, a):                                                                               ## From Lab 10
    '''
    Function to make matrix following format from lab outline
    Args: Takes in the L length of matrix to make, b value of elements on diagonal-1, d value of elements on diagonal, a value of elements on diagonal+1
    Returns: Matrix that was made
    '''
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

    prob = np.zeros((nspace, ntime), dtype=np.complex128)
    prob[:, 0] = np.conjugate(psi[:, 0]).dot(psi[:, 0])*h*tau

    #H = make_tridiagonal(nspace, (-(h_bar**2)/(2*m*(h**2))), (2*(h_bar**2)/(2*m*(h**2)))+V, (-(h_bar**2)/(2*m*(h**2))))

    H = -((h_bar**2)/2*m)*make_tridiagonal(nspace, 1/(h**2), -2/(h**2), 1/(h**2)) + V*np.identity(nspace)

    if (method == 'ftcs'):
        for i in range(ntime-1):
            psi[:, i+1] = (make_tridiagonal(nspace, 0, 1, 0) - (1j*tau/h_bar)*H).dot(psi[:, i])
            #prob[i+1] = np.sum(np.abs(psi[:, i+1])**2)*h*tau
            prob[:, i+1] = np.conjugate(psi[:, i+1]).dot(psi[:, i+1])*h*tau
    elif (method == 'crank'):
        for i in range(ntime-1):
            psi[:, i+1] = nplinalg.inv(make_tridiagonal(nspace, 0, 1, 0) + (1j*tau/(2*h_bar))*H).dot(make_tridiagonal(nspace, 0, 1, 0) - (1j*tau/(2*h_bar))*H).dot(psi[:, i])
            #prob[i+1] = np.sum(np.abs(psi[:, i+1])**2)*h*tau
            prob[:, i+1] = np.conjugate(psi[:, i+1]).dot(psi[:, i+1])*h*tau
    else:
        print("Invalid Method\n")
    
    #prob = psi.dot(np.conjugate(psi))

    return psi, x_points, t_points, prob

def sch_animate(psi, x_points, t_points,):
    fig = plt.figure()
    camera = Camera(fig)
    for i in range(0, len(t_points), 5):
        plt.plot(x_points, psi[:, i], color='blue', label=f"Schrodinger Eq at t = {t_points[i]}")
        camera.snap()                                                                                   ## Capturing frame at current time point
    animation = camera.animate()                                                                        ## Animating frames together 
    animation.save("TurkstraMichael_Lab7_coupled_oscillators.mp4",fps=60)                               ## Saving animation
    


def main():
    nspace, ntime, tau = 100, 50000, 0.1
    psi, x_grid, t_grid, prob = sch_eqn(nspace, ntime, tau, 'crank', potential=[80])
    sch_animate(psi, x_grid, t_grid)
    print(prob)

if __name__ == '__main__':
    main()