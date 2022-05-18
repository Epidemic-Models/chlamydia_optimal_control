import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def main():
    t_max = 150
    r_max = 1
    r_steps = 100
    t_steps = 1000000
    r = np.linspace(0, r_max, r_steps)
    t = np.linspace(0, t_max, t_steps)
    dr = r_max / r_steps
    dt = t_max / t_steps

    n_iterations = 20
    update_coefficient = 0.2
    tolerance = 0.1

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
    omega = 0.1




    k_rho = np.ones(r_steps)
    k_i = np.ones(r_steps)

    w_1 = 1000
    w_2 = 10
    w_3 = 10
    w_4 = 1000
    w_5 = 10. * np.ones(r_steps)
    w_6 = 10. * np.ones(r_steps)
    w_7 = 10.
    w_8 = 10.
    w_9 = 10.
    w_10 = 10.
    w_11 = 10. * np.ones(r_steps)
    w_12 = 10. * np.ones(r_steps)
    w_13 = 100
    w_14 = 100


    c = 200. * np.ones(t_steps)
    a = np.zeros(t_steps)
    i_1 = np.zeros(t_steps)
    i_2 = np.zeros(t_steps)
    p = np.zeros(t_steps)
    rho = np.zeros((r_steps, t_steps))
    i = np.zeros((r_steps, t_steps))

    c[0] = 65.3061
    a[0] = 9.40408
    i_1[0] = 940.408
    i_2[0] = 124.213
    p[0] = 372.986
    rho[:, 0] = 12.4213 * np.exp(- zeta * a[0] / k_rho[0] * r)
    i[:, 0] = 1.59851 * np.exp(- zeta * a[0] / k_i[0] * r)

    u_1_max = 0.9
    u_2_max = 0.9

    u_1 = 0.9 * u_1_max * np.ones(t_steps)
    u_2 = 0.8 * u_2_max * np.ones(t_steps)

    # u_1 = 0.62 * np.ones(t_steps)
    # u_2 = 0.72 * np.ones(t_steps)

    lambda_c = np.zeros(t_steps)
    lambda_a = np.zeros(t_steps)
    lambda_i_1 = np.zeros(t_steps)
    lambda_i_2 = np.zeros(t_steps)
    lambda_p = np.zeros(t_steps)

    mu_rho = np.zeros((r_steps, t_steps))
    mu_i = np.zeros((r_steps, t_steps))

    new_u_1 = np.zeros(t_steps)
    new_u_2 = np.zeros(t_steps)

    for ind_iter in range(n_iterations):
        old_c = np.copy(c)
        old_a = np.copy(a)
        old_i_1 = np.copy(i_1)
        old_i_2 = np.copy(i_2)
        old_p = np.copy(p)
        old_u_1 = np.copy(u_1)
        old_u_2 = np.copy(u_2)
        old_rho = np.copy(rho)
        old_i = np.copy(i)
        old_lambda_c = np.copy(lambda_c)
        old_lambda_a = np.copy(lambda_a)
        old_lambda_i_1 = np.copy(lambda_i_1)
        old_lambda_i_2 = np.copy(lambda_i_2)
        old_lambda_p = np.copy(lambda_p)
        old_mu_rho = np.copy(mu_rho)
        old_mu_i = np.copy(mu_i)
        old_u_1 = np.copy(u_1)
        old_u_2 = np.copy(u_2)

        for ind_t in range(1, t_steps):
            # ODEs:
            c[ind_t] = c[ind_t - 1] + (
                    (1 - u_1[ind_t - 1]) * n * k * i[r_steps - 1, ind_t - 1] - beta * c[ind_t - 1] -
                    mu_c * c[ind_t - 1]) * dt
            a[ind_t] = a[ind_t - 1] + (omega * i_1[ind_t - 1] - mu_a * a[ind_t - 1]) * dt
            i_1[ind_t] = i_1[ind_t - 1] + ((1 - theta) * beta * c[ind_t - 1] - alpha_1 * i_1[ind_t - 1]) * dt
            i_2[ind_t] = i_2[ind_t - 1] + (
                    alpha_1 * i_1[ind_t - 1] - alpha_2 * i_2[ind_t - 1] - zeta * a[ind_t - 1] * i_2[ind_t - 1]) * dt
            p[ind_t] = p[ind_t - 1] + (
                    (1 - u_2[ind_t - 1]) * eta * k_rho[r_steps - 1] * rho[r_steps - 1, ind_t - 1] -
                    xi * u_2[ind_t - 1] * p[ind_t - 1] - delta * p[ind_t - 1]) * dt

            # BCs:
            rho[0, ind_t] = alpha_2 * i_2[ind_t]
            i[0, ind_t] = (1 - eta) * k_rho[r_steps - 1] * rho[r_steps - 1, ind_t - 1] + \
                          xi * u_2[ind_t - 1] * p[ind_t - 1]

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
        # transversality
        lambda_c[t_steps - 1] = 2 * w_1 * c[t_steps - 1]
        lambda_a[t_steps - 1] = 0
        lambda_i_1[t_steps - 1] = 2 * w_2 * i_1[t_steps - 1]
        lambda_i_2[t_steps - 1] = 2 * w_3 * i_2[t_steps - 1]
        lambda_p[t_steps - 1] = 2 * w_4 * p[t_steps - 1]
        d_3_f = 2 * (w_11 @ rho[:, t_steps - 1] + w_12 @ i[:, t_steps - 1])
        mu_rho[:, t_steps - 1] = d_3_f * w_5
        mu_i[:, t_steps - 1] = d_3_f * w_6

        for ind_t in range(t_steps - 2, -1, -1):

            # adjoint ODEs:
            lambda_c[ind_t] = lambda_c[ind_t + 1] - (-2 * w_7 * c[ind_t + 1] + (beta + mu_c) * lambda_c[ind_t + 1]) * dt
            lambda_a[ind_t] = lambda_a[ind_t + 1] - (mu_a * lambda_a[ind_t + 1] +
                                                     zeta * i_2[ind_t + 1] * lambda_i_2[ind_t + 1] +
                                                     zeta * (rho[:, ind_t + 1] @ mu_rho[:, ind_t + 1]) * dr +
                                                     zeta * (i[:, ind_t + 1] @ mu_i[:, ind_t + 1]) * dr) * dt
            lambda_i_1[ind_t] = lambda_i_1[ind_t + 1] - (- 2 * w_8 * i_1[ind_t + 1] - omega * lambda_a[ind_t + 1]
                                                         + alpha_1 * lambda_i_1[ind_t + 1]
                                                         - alpha_1 * lambda_i_2[ind_t + 1]) * dt
            lambda_i_2[ind_t] = lambda_i_2[ind_t + 1] - (- 2 * w_9 * i_2[ind_t + 1] + (alpha_2 + zeta * a[ind_t + 1]) *
                                                         lambda_i_2[ind_t + 1]
                                                         - alpha_2 * k_rho[0] * mu_rho[0, ind_t + 1]) * dt
            lambda_p[ind_t] = lambda_p[ind_t + 1] - (- 2 * w_10 * p[ind_t + 1] +
                                                     (xi * u_2[ind_t - 1] + delta) * lambda_p[ind_t + 1] -
                                                     xi * u_2[ind_t - 1] * k_i[0] * mu_i[0, ind_t + 1]) * dt

            # adjoint BCs:
            mu_rho[r_steps - 1, ind_t] = (1 - u_2[ind_t]) * eta * lambda_p[ind_t] +\
                                         (1 - eta) * k_rho[0] * mu_i[0, ind_t + 1]
            mu_i[r_steps - 1, ind_t] = ((1 - u_1[ind_t]) * n * k * lambda_c[ind_t]) / k_i[r_steps - 1]

            # adjoint PDEs:
            d_3_g = 2 * (w_11 @ rho[:, ind_t + 1] + w_12 @ i[:, ind_t + 1])
            mu_rho[0:r_steps - 1, ind_t] = mu_rho[0:r_steps - 1, ind_t + 1] -\
                                           (- k_rho[0:r_steps - 1]
                                            * (mu_rho[1:r_steps, ind_t + 1] - mu_rho[0:r_steps - 1, ind_t + 1]) / dr
                                            - d_3_g * w_11[0:r_steps - 1] +
                                            zeta * a[ind_t + 1] * mu_rho[0:r_steps - 1, ind_t + 1]) * dt
            mu_i[0:r_steps - 1, ind_t] = mu_i[0:r_steps - 1, ind_t + 1] -\
                                         (- k_i[0:r_steps - 1] * (
                                                 mu_i[1:r_steps, ind_t + 1] - mu_i[0:r_steps - 1, ind_t + 1]) / dr
                                          - d_3_g * w_12[0:r_steps - 1] +
                                          zeta * a[ind_t + 1] * mu_i[0:r_steps - 1, ind_t + 1]) * dt

        for ind_t in range(t_steps):
            # control:
            frac_1 = lambda_c[ind_t] * n * k * i[r_steps - 1, ind_t] / (2 * w_13)
            new_u_1[ind_t] = min(max(frac_1, 0), u_1_max)
            frac_2 = (lambda_p[ind_t] * eta * k_rho[0] * rho[r_steps - 1, ind_t] + xi * p[ind_t] * lambda_p[ind_t] -
                      xi * p[ind_t] * k_i[0] * mu_i[0, ind_t]) / (2 * w_14)
            new_u_2[ind_t] = min(max(frac_2, 0), u_2_max)

        u_1 = (1 - update_coefficient) * old_u_1 + update_coefficient * new_u_1
        u_2 = (1 - update_coefficient) * old_u_2 + update_coefficient * new_u_2

        cost = w_1 * c[t_steps - 1] * c[t_steps - 1] + w_2 * i_1[t_steps - 1] * i_1[t_steps - 1] +\
               w_3 * i_2[t_steps - 1] * i_2[t_steps - 1] + w_4 * p[t_steps - 1] * p[t_steps - 1] +\
               np.square(np.sum(w_5 @ rho[:, t_steps - 1] + w_6 @ i[:, t_steps - 1]) * dr) +\
               np.sum(w_7 * np.square(c) + w_8 * np.square(i_1) + w_9 * np.square(i_2) + w_10 * p +
                      + np.square(np.sum(w_11 @ rho + w_12 @ i) * dr) + w_13 * np.square(u_1) +
                      w_14 * np.square(u_2)) * dt

        avg_u_1 = np.sum(u_1) * dt / t_max
        avg_u_2 = np.sum(u_2) * dt / t_max

        step_size = np.sum(np.abs(c - old_c)) * dt + np.sum(np.abs(a - old_a)) * dt \
                    + np.sum(np.abs(i_1 - old_i_1)) * dt + np.sum(np.abs(i_2 - old_i_2)) * dt + np.sum(
            np.abs(p - old_p)) * dt \
                    + np.sum(np.sum(np.abs(rho - old_rho))) * dr * dt \
                    + np.sum(np.sum(np.abs(i - old_i))) * dr * dt + np.sum(np.abs(lambda_c - old_lambda_c)) * dt \
                    + np.sum(np.abs(lambda_a - old_lambda_a)) * dt \
                    + np.sum(np.abs(lambda_i_1 - old_lambda_i_1)) * dt + np.sum(
            np.abs(lambda_i_2 - old_lambda_i_2)) * dt \
                    + np.sum(np.abs(lambda_p - old_lambda_p)) * dt \
                    + np.sum(np.sum(np.abs(mu_rho - old_mu_rho))) * dr * dt + np.sum(
            np.sum(np.abs(mu_i - old_mu_i))) * dr * dt \
                    + np.sum(np.abs(u_1 - old_u_1)) * dt + np.sum(np.abs(u_2 - old_u_2)) * dt

        print('Step ', ind_iter, ', cost: ', format(cost, '.2f'), ', average u_1: ', format(avg_u_1, '.2f'),
              ', average u_2: ', format(avg_u_2, '.2f'), ', step size: ', format(step_size, '.2f'), '.')

        if step_size < tolerance:
            break

    # 2D diagrams
    plt.plot(t, u_1, "r")
    plt.plot(t, u_2, "g")
    plt.legend(["u1(t)", "u2(t)"])
    plt.xlabel("t")
    plt.ylabel("")
    plt.xlim([0, t_max])
    plt.ylim([0, 1])
    plt.show()

    plt.figure()
    plt.plot(t, c, "b")
    plt.legend(["C(t)"])
    plt.xlabel("t")
    plt.ylabel("")
    plt.xlim([0, t_max])
    plt.ylim([0, 100])
    plt.show()

    plt.figure()
    plt.plot(t, c, "r")
    plt.plot(t, a, "g")
    plt.legend(["C(t)", "A(t)"])
    plt.xlabel("t")
    plt.ylabel("")
    plt.xlim([0, t_max])
    plt.ylim([0, 200])
    plt.show()

    plt.figure()
    plt.plot(t, i_1, "b")
    plt.plot(t, i_2, "m")
    plt.plot(t, p, "k")
    plt.legend(["I1(t)", "I2(t)", "P(t)"])
    plt.xlabel("t")
    plt.ylabel("")
    plt.xlim([0, t_max])
    plt.ylim([0, 1400])
    plt.show()

    plt.figure()
    plt.plot(t, i_1, "r")
    plt.legend(["I1(t)"])
    plt.xlabel("t")
    plt.ylabel("")
    plt.xlim([0, t_max])
    plt.ylim([0, 1000])
    plt.show()

    plt.figure()
    plt.plot(t, i_2, "m")
    plt.legend(["I2(t)"])
    plt.xlabel("t")
    plt.ylabel("")
    plt.xlim([0, t_max])
    plt.ylim([0, 150])
    plt.show()

    plt.figure()
    plt.plot(t, p, "k")
    plt.legend(["P(t)"])
    plt.xlabel("t")
    plt.ylabel("")
    plt.xlim([0, t_max])
    plt.ylim([0, 400])
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
    ax.set_zlim3d(0, 30)
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
    ax.set_zlim3d(0, 100)
    plt.show()


if __name__ == '__main__':
    main()