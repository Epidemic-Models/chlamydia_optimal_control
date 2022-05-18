import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def main():
    t_max = 350.
    r_max = 1.
    r_steps = 100
    t_steps = 1000000
    r = np.linspace(0, r_max, r_steps)
    t = np.linspace(0, t_max, t_steps)
    dr = r[1] - r[0]
    dt = t[1] - t[0]

    n_iterations = 1000

    n = 200.0
    k = 1
    eta = 0.7
    alpha_1 = 1 / 8
    alpha_2 = 1 / 10
    beta = 2
    xi = 0.5
    mu_c = 0.1
    mu_a = 1 / 10
    delta = 0.01
    theta = 0.1
    zeta = 0.09
    omega = 0.001


    k_rho = 2. * np.ones(r_steps)
    k_i = 2. * np.ones(r_steps)

    w_1 = 1.
    w_2 = 1.
    w_3 = 1.
    w_4 = 1.
    w_5 = 50000.
    w_6 = 100.
    w_rho = np.ones(r_steps)
    w_i = np.ones(r_steps)

    u_1_max = 0.9
    u_2_max = 0.9

    c = 100.0 * np.ones(t_steps)
    a = np.zeros(t_steps)
    i_1 = np.zeros(t_steps)
    i_2 = np.zeros(t_steps)
    p = np.zeros(t_steps)
    rho = np.zeros((r_steps, t_steps))
    i = np.zeros((r_steps, t_steps))

    u_1 = np.zeros(t_steps)
    u_2 = np.zeros(t_steps)

    for ind_t in range(1, t_steps):
        # ODEs:
        c[ind_t] = c[ind_t - 1] + (
                (1 - u_1[ind_t - 1]) * n * k * i[r_steps - 1, ind_t - 1] - beta * c[ind_t - 1]
                - mu_c * c[ind_t - 1]) * dt
        a[ind_t] = a[ind_t - 1] + (omega * i_1[ind_t - 1] - mu_a * a[ind_t - 1]) * dt
        i_1[ind_t] = i_1[ind_t - 1] + ((1 - theta) * beta * c[ind_t - 1] -
                                       alpha_1 * i_1[ind_t - 1]) * dt
        i_2[ind_t] = i_2[ind_t - 1] + (
                alpha_1 * i_1[ind_t - 1] - alpha_2 * i_2[ind_t - 1] - zeta * a[ind_t - 1] * i_2[ind_t - 1]) * dt
        p[ind_t] = p[ind_t - 1] + (
                (1 - u_2[ind_t - 1]) * eta * k_rho[r_steps - 1] * rho[r_steps - 1, ind_t - 1] -
                xi * p[ind_t - 1] - delta * p[ind_t - 1]) * dt

        # BCs:
        rho[0, ind_t] = alpha_2 * i_2[ind_t]
        i[0, ind_t] = (1 - eta) * k_rho[r_steps - 1] * rho[r_steps - 1, ind_t - 1] + xi * p[ind_t - 1]

        # PDEs:
        rho[1:r_steps, ind_t] = rho[1:r_steps, ind_t - 1] + (- (k_rho[1:r_steps] * rho[1:r_steps, ind_t - 1] -
                                                                k_rho[0:r_steps - 1] *
                                                                rho[0:r_steps - 1, ind_t - 1]) / dr -
                                                             zeta * a[ind_t - 1] * rho[1:r_steps, ind_t - 1]
                                                             ) * dt
        i[1:r_steps, ind_t] = i[1:r_steps, ind_t - 1] + (- (k_i[1:r_steps] * i[1:r_steps, ind_t - 1] -
                                                            k_i[0:r_steps - 1] * i[0:r_steps - 1, ind_t - 1]) / dr -
                                                         zeta * a[ind_t - 1] * i[1:r_steps, ind_t - 1]
                                                         ) * dt

    cost = np.sum(w_1 * np.square(c) + w_2 * np.square(i_1) + w_3 * np.square(i_2) + w_4 * p +
                  + np.square(np.sum(w_rho @ rho + w_i @ i) * dr) + w_5 * np.square(u_1) +
                  w_6 * np.square(u_2)) * dt

    print('The cost is ', format(cost, '.2f'), '.')



    plt.figure()
    plt.plot(t, c, "b")
    plt.legend(["C(t)"])
    plt.xlabel("t")
    plt.ylabel("")
    plt.xlim([0, t_max])
    plt.ylim([0, 1700])
    plt.show()

    plt.figure()
    plt.plot(t, i_1, "r")
    plt.plot(t, i_2, "m")
    plt.plot(t, p, "k")
    plt.legend(["I1(t)", "I2(t)", "P(t)"])
    plt.xlabel("t")
    plt.ylabel("")
    plt.xlim([0, t_max])
    plt.ylim([0, 10500])
    plt.show()

    plt.figure()
    plt.plot(t, i_1, "r")
    plt.legend(["I1(t)"])
    plt.xlabel("t")
    plt.ylabel("")
    plt.xlim([0, t_max])
    plt.ylim([0, 10500])
    plt.show()

    plt.figure()
    plt.plot(t, i_2, "m")
    plt.legend(["I2(t)"])
    plt.xlabel("t")
    plt.ylabel("")
    plt.xlim([0, t_max])
    plt.ylim([0, 600])
    plt.show()

    plt.figure()
    plt.plot(t, p, "k")
    plt.legend(["P(t)"])
    plt.xlabel("t")
    plt.ylabel("")
    plt.xlim([0, t_max])
    plt.ylim([0, 60])
    plt.show()

    # 3D diagrams
    t_mesh, r_mesh = np.meshgrid(t, r)

    plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(r_mesh, t_mesh, rho,
                    cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('r')
    ax.set_ylabel('t')
    ax.set_zlabel('\u03C1')
    ax.set_xlim3d(0, r_max)
    ax.set_ylim3d(0, t_max)
    ax.set_zlim3d(0, 50)
    plt.show()

    plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(r_mesh, t_mesh, rho,
                    cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('r')
    ax.set_ylabel('t')
    ax.set_zlabel('\u03C1')
    ax.set_xlim3d(0, r_max)
    ax.set_ylim3d(0, t_max)
    ax.set_zlim3d(0, 50)
    plt.show()



    plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(r_mesh, t_mesh, i,
                    cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('r')
    ax.set_ylabel('t')
    ax.set_zlabel('i')
    ax.set_xlim3d(0, r_max)
    ax.set_ylim3d(0, t_max)
    ax.set_zlim3d(0, 20)
    plt.show()


if __name__ == '__main__':
    main()