import TurkstraMichael_Project4 as proj4
import numpy as np
import matplotlib.pyplot as plt

x_grid = np.linspace(-50, 50, 500)
rslt = proj4.sch_init_cond(x_grid, 10, 0.5, 0)

plt.figure()
plt.plot(x_grid, rslt, label="Wave packet")
plt.title("Gaussian Wave Packet")
plt.xlabel("x")
plt.ylabel("Ψ(x)")
plt.grid()
plt.show()
