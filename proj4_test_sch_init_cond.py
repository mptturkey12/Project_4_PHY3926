import numpy as np
import matplotlib.pyplot as plt

def sch_init_cond(x_points, sig_0, k_0, x_0):
    init_cond = (1/np.sqrt(sig_0)*np.sqrt(np.pi))*np.exp((k_0*x_points*1j))*np.exp(-((x_points-x_0)**2)/((2*sig_0)**2)) 
    return init_cond

x_grid = np.linspace(-50, 50, 500)
rslt = sch_init_cond(x_grid, 10, 0, 0.5)

plt.figure()
plt.plot(x_grid, rslt, label="Wave packet")
plt.title("Gaussian Wave Packet")
plt.xlabel("x")
plt.ylabel("Ψ(x)")
plt.grid()
plt.show()


