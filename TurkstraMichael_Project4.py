import numpy as np
import numpy.linalg as nplinalg
import matplotlib.pyplot as plt
from celluloid import Camera
from numpy import linalg as LA

h_bar, m = 1, 1/2                                                                                                   ## Initializing h_bar and m to values outlined in project description

def sch_init_cond(x_points, sig_0, k_0, x_0):                                                                       ## Similar to lab 10
    '''
    Function to apply gaussian initial conditions to inputted array using equation 9.42 from the textbook
    Args: x_points grid spacing, sig_0 standard deviation, k_0 wave number, x_0 center
    Returns: 1d array holding initial conditions applied to x_points
    '''
    init_cond = (1/np.sqrt(sig_0)*np.sqrt(np.pi))*np.exp((k_0*x_points*1j) - ((x_points-x_0)**2)/((2*sig_0)**2))    ## Calculating initial conditions array using equation 9.42
    return init_cond                                                                                                ## Returning initial conditions array

def make_tridiagonal(Nspace, b, d, a):                                                                              ## From Lab 10
    '''
    Function to make matrix following format from lab outline
    Args: Takes in the L length of matrix to make, b value of elements on diagonal-1, d value of elements on diagonal, a value of elements on diagonal+1
    Returns: Matrix that was made
    '''
    D = d*np.identity(Nspace)+a*np.diagflat(np.ones(Nspace-1),1)+b*np.diagflat(np.ones(Nspace-1),-1)                ## Equation for making required matrix, d*diagonal + b*diagonal-1, + a*diagonal+1, takes insperation from “lecture10a-FTCS-Diffusion_mtx.ipynb”
    D[Nspace-1, 0], D[0, Nspace-1] = a, b                                                                           ## Setting periodic boundary conditions
    return D                                                                                                        ## Return matrix

def check_stability(input_2d_array):
    '''
    Function to calculate eigenvalues of the input matrix, check if any eigen values are unstable and notify user if input will give stable results.
    Args: input_2d_array matrix to calculate eigenvalues
    Returns: void
    '''
    stable = True                                                                                                   ## Initialize boolean for tracking if stable or not
    eigenvalues = LA.eigvals(input_2d_array)                                                                        ## Using linalg.eig to calculate eigenvalues of input matrix
    for i in range(len(eigenvalues)):                                                                               ## Loop through the eigenvalues
        if (np.real(eigenvalues[i]) > 0):                                                                           ## Check if real part of current eigenvalue is negative
            stable = False                                                                                          ## If so update stable flag
    if (stable):                                                                                                    ## Check if stable flag is still stable (i.e. no unstable eigenvalues)
        print("The FTCS simulation is expected to be stable\n")                                                     ## Notify user of stability
    else:
        print("The FTCS simulation is not expected to be stable\n")                                                 ## Notify user of instability


def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=[], wparam=[10, 0, 0.5]):
    '''
    Function to solve the one dimentional, time dependant, Schrodinger equation using the FTCS or Crank-Nicholson method. 
    Args: nspace integer determining the number of spacial divisions, ntime integer determining the number of time divisions to simulate, tau time step, method string to select which
    method to use, length the length of the 1d simulation, potential array holding indexes for nspace where potential barrier will be located, wparam array of size 3 to hold inputs to
    sch_init_cond to get inital conditions
    Returns: psi 2d numpy array of size (nspace, ntime) holding the value of phi at every point in space and time simulated (i.e. Ψ(x,t)), x_points the spacial grid being simulated,
    t_points the time steps being simulated, prob 2d numpy array of size (nspace, ntime) holding the particle probability density at every point in space and time simulated.
    '''
    sig0, x0, k0 = wparam                                                                                           ## Unpacking initial condition parameters from wparam
    h = length/nspace                                                                                               ## Calculating grid spacing 
    x_points = np.linspace(-length/2, length/2, nspace)                                                             ## Initializing x_points the spacial grid being simulated
    t_points = np.arange(0, ntime*tau, tau)                                                                         ## Initializing t_points the time steps being simulated
    
    V = np.zeros(nspace)                                                                                            ## Initializing array to hold track hold potential barriers along the x axis  
    V[potential] = 1                                                                                                ## Setting potential barrier at inputted index's

    init_cond = sch_init_cond(x_points, sig0, k0, x0)                                                               ## Calling sch_init_cond with inputted parameters to get inital conditions

    psi = np.zeros((nspace, ntime), dtype=np.complex128)                                                            ## Initializing 2d numpy array to hold phi at every x and time point
    psi[:, 0] = init_cond                                                                                           ## Setting psi at time = 0 to initial conditions

    prob = np.zeros((nspace, ntime), dtype=np.complex128)                                                           ## Initializing 2d numpy array to hold particle probability density at every x and time point
    prob[:, 0] = (np.abs(psi[:, 0])**2)*h*tau                                                                       ## Calculating particle probability density for t = 0, with initial conditions

    H = -((h_bar**2)/2*m)*make_tridiagonal(nspace, 1/(h**2), -2/(h**2), 1/(h**2)) + V*np.identity(nspace)           ## Defining the H matrix from 9.31 from the textbook

    if (method == 'ftcs'):                                                                                          ## Check if user inputted FTCS method
        A = make_tridiagonal(nspace, 0, 1, 0) - (1j*tau/h_bar)*H                                                    ## Make A matrix using equation 9.32 from the textbook
        check_stability(A)                                                                                          ## Calling check_stability to notify user of stability
        for i in range(ntime-1):                                                                                    ## Looping through the time points
            psi[:, i+1] = A.dot(psi[:, i])                                                                          ## Calculating psi at n+1 using FTCS method
            prob[:, i+1] = (np.abs(psi[:, i+1])**2)*h*tau                                                           ## Calculating particle probability density using equation from project outline
    elif (method == 'crank'):                                                                                       ## Check if user inputted Crank-Nicholson method
        A = nplinalg.inv(make_tridiagonal(nspace, 0, 1, 0) + (1j*tau/(2*h_bar))*H).dot(make_tridiagonal(nspace, 0, 1, 0) - (1j*tau/(2*h_bar))*H) ## Calculate A matrix using equation 9.40 from the textbook
        for i in range(ntime-1):                                                                                    ## Looping through the time points
            psi[:, i+1] = A.dot(psi[:, i])                                                                          ## Calclulating psi at n+1 using Crank-Nicholson method
            prob[:, i+1] = (np.abs(psi[:, i+1])**2)*h*tau                                                           ## Calculating particle probability density using equation from project outline
    else:                                                                                                           ## Else not recognized method
        print("Invalid Method\n")                                                                                   ## Notify user of invalid method inputted

    return psi, x_points, t_points, prob                                                                            ## Returning psi matrix, x_points array, t_points array, prob matrix

def sch_animate(data, x_points, t_points, type):
    '''
    Function to visualize data from sch_eqn using an animation showing either psi or prob of x at every time point
    Args: data 2d numpy array can be psi or prob, x_points data's spacial grid, t_points data's time grid, type string to determine type of data
    Returns: void
    '''
    fig = plt.figure()                                                                                              ## Initializing figure
    camera = Camera(fig)                                                                                            ## Initializting camera to figure
    if type == 'psi':                                                                                               ## Checking if type is psi
        plt.title("Ψ(x) Animation")
        plt.xlabel("x")                                                                                             ## Labeling plot for psi animation
        plt.ylabel("Ψ(x)")
        for i in range(0, len(t_points), 5):                                                                        ## Looping through every 5 time points
            plt.plot(x_points, np.real(data[:, i]), color='blue', label=f"Schrodinger Eq at t = {t_points[i]}")     ## Plotting data at current time point
            camera.snap()                                                                                           ## Capturing frame
    elif type == 'prob':                                                                                            ## Checking if type is prob
        plt.title("Ψ*Ψ(x) Animation")
        plt.xlabel("x")                                                                                             ## Labeling plot for prob animation
        plt.ylabel("Ψ*Ψ(x)")
        for i in range(0, len(t_points), 5):                                                                        ## Looping through every 5 time points
            plt.plot(x_points, np.real(data[:, i]), color='blue')                                                   ## Plotting data at current time point
            camera.snap()                                                                                           ## Capturing frame
    animation = camera.animate()                                                                                    ## Compiling frames together
    animation.save(f"TurkstraMichael_Project4_Animation_{type}.mp4",fps=60)                                         ## Saving animation
    
def sch_plot(data, x_points, t_points, type, save, time):
    '''
    Funtion to plot either psi or prob data from sch_eqn at a time index.
    Args: data 2d numpy array can be psi or prob, x_points array to hold spacial grid of data, t_points array to hold time grid of data, type string to determine type of data
    Returns: void
    '''
    plt.figure()                                                                                                    ## Initializing Figure
    if type == 'psi':                                                                                               ## Checking if the inputted type is psi
        plt.plot(x_points, np.real(data[:, time]))                                                                  ## Plotting data at time index
        plt.title(f"Ψ(x) at t = {t_points[time]}")
        plt.xlabel("x")                                                                                             ## Labeling psi plot
        plt.ylabel("Ψ(x)")
    elif type == 'prob':                                                                                            ## Checking if the inputted type is prob
        plt.plot(x_points, np.real(data[:, time]))                                                                  ## Plotting data at time index
        plt.title(f"Ψ*Ψ(x) at t = {t_points[time]}")
        plt.xlabel("x")                                                                                             ## Labeling prob plot
        plt.ylabel("Ψ*Ψ(x)")
    else:                                                                                                           ## Else unrecognized type
        print("Invalid plot inputed\n")                                                                             ## Notify user
    plt.grid()                                                                                                      ## Adding grid to plot
    if save:                                                                                                        ## Checking is user requested plot to be saved
        plt.savefig(f'TurkstraMichael_Project4_Fig_{type}.png')                                                     ## Saving plot
    else:
        plt.show()                                                                                                  ## Showing plot

def main():
    '''
    Function to set parameters and call functions.
    '''
    nspace, ntime, tau = 1000, 10000, 0.1                                                                           ## Arbitrary parameter values for testing
    psi, x_grid, t_grid, prob = sch_eqn(nspace, ntime, tau, 'crank', potential=[800])                               ## Calling sch_eqn to solve 1 dimentional, time dependant, Schrodinger Equation
    sch_plot(psi, x_grid, t_grid, 'psi', True, 10)                                                                  ## Calling sch_plot to plot psi data
    sch_plot(prob, x_grid, t_grid, 'prob', True, 99)                                                                ## Calling sch_plot to plot prob data
    sch_animate(psi, x_grid, t_grid, 'psi')                                                                         ## Calling sch_animate to make animation of psi data
    sch_animate(prob, x_grid, t_grid, 'prob')                                                                       ## Calling sch_animate to make animation of prob data

if __name__ == '__main__':
    main()