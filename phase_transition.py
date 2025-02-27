import torch as t
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

"""
This python script simulates a gradient flow to compute the loss for the Theorem 10 orf the paper.
Since all the system can be computed in closed-form, we directly compute the loss instead of running a gradient descent.
The .guf and .png produced show two clear phase transitions for very large embedding dimensions.
"""


# Data for the animation
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2, c='k', label=r"$L_{n}(t)$")
Vbar1, = ax.plot([], [], lw=2, c='grey', label=r"$L_{\infty}(t)$")
Vbar2, = ax.plot([], [], lw=2, c='grey')
Hbar1, = ax.plot([], [], lw=2, c='grey')
Hbar2, = ax.plot([], [], lw=2, c='grey')
Hbar3, = ax.plot([], [], lw=2, c='grey')
plt.legend()

nb_points = 1000
target_1=1.
target_2=2.
List_targets = [target_1, target_2]
max_Loss = (target_1+target_2**2)/4
plateau_loss = target_1/4
p = 2
cos_angle = 1.
norm_0 = 1.



def scalar_product(n: int, target_norm: float, p: int, cos_angle: float, times: t.Tensor) -> t.Tensor:
    if cos_angle == 1.:
        numerator = 1+t.zeros_like(times)
        denominator = 1+t.zeros_like(times)
    else:
        normalized_time = times*target_norm/(n*p)
        numerator = t.sinh(2*normalized_time) + cos_angle*t.cosh(2*normalized_time)
        denominator = t.cosh(2*normalized_time) + cos_angle*t.sinh(2*normalized_time)
    return numerator/denominator

def norm_2(n: int, target_norm: float, p: int, cos_angle: float, norm_0: float, times: t.Tensor) -> t.Tensor:
    normalized_time = times*target_norm/(n*p)
    if cos_angle == 1.:
        numerator = p*target_norm*(norm_0**2)*t.exp(normalized_time)
        denominator = p*target_norm*t.exp(-normalized_time) + (norm_0**2)*(t.exp(normalized_time)-t.exp(-normalized_time))
    else:
        numerator = p*target_norm*(norm_0**2)*(t.cosh(2*normalized_time) + cos_angle*t.sinh(2*normalized_time))
        denominator = p*target_norm + (norm_0**2)*(t.sinh(2*normalized_time) + cos_angle*(t.cosh(2*normalized_time)-1))
    return numerator/denominator

def loss(n: int, target_norm: float, p: int, norm_2: t.Tensor, scalar_product: t.Tensor) -> t.Tensor:
    return (target_norm**2 + (norm_2**2)/(p**2) - 2*norm_2*target_norm*scalar_product/p)/(2*n)


def init():
    ax.set_xlabel("Time")
    ax.set_ylabel("Loss")
    return line, Vbar1, Vbar2, Hbar1, Hbar2, Hbar3

def update(frame):
    n = 100+int(np.exp(frame/5))

    List_target_norm = [taregt*np.sqrt(n) for taregt in List_targets]
    min_target_norm = min(List_target_norm)
    max_target_norm = max(List_target_norm)
    t_second_1 = (n*p/(min_target_norm*2))*np.log(p*min_target_norm)
    t_second_2 = (n*p/(max_target_norm*2))*np.log(p*max_target_norm)
    t_second = max(t_second_1, t_second_2)

    times = t.linspace(0, 1.75*t_second, nb_points)
    Loss = t.zeros_like(times)
    for target_norm in List_target_norm:
        norm = norm_2(n, target_norm, p, cos_angle, norm_0, times)
        s_p = scalar_product(n, target_norm, p, cos_angle, times)
        Loss += loss(n, target_norm, p, norm, s_p)/p

    Vbar1.set_data([t_second_1/t_second, t_second_1/t_second],[0, plateau_loss])
    Vbar2.set_data([t_second_2/t_second, t_second_2/t_second],[plateau_loss, max_Loss])
    Hbar1.set_data([0, t_second_2/t_second],[max_Loss, max_Loss])
    Hbar2.set_data([t_second_2/t_second, 1],[plateau_loss, plateau_loss])
    Hbar3.set_data([1, 1.75],[0, 0])
    Vbar1.set_dashes([2,1])
    Vbar2.set_dashes([2,1])
    Hbar1.set_dashes([2,1])
    Hbar2.set_dashes([2,1])
    Hbar3.set_dashes([2,1])
    line.set_data(times/t_second, Loss)
    ax.set_xlim(0-0.05, 1.75-0.05)
    ax.set_ylim(0-0.05, max_Loss*1.1)
    ax.set_xticks([0, t_second_1/t_second, t_second_2/t_second])
    ax.set_xticklabels([0, r"$\frac{np}{||D_1||}\log(||D_1||)$", r"$\frac{np}{||D_2||}\log(||D_2||)$"])
    ax.set_yticks([max_Loss, 0])
    ax.set_yticklabels([r"$L_n(\theta(0))$", 0])
    exponent = int(np.log(n)/np.log(10))
    decimal = int((n/(10**exponent))*10)/10
    ax.set_title(rf"Loss trajectory in dimension ${decimal}\times 10^{ {exponent} }$")
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    return line, Vbar1, Vbar2, Hbar1, Hbar2, Hbar3

ani = FuncAnimation(fig, update, frames=210, init_func=init, blit=True)
ani.save("Images/Phase_Transition.gif", fps=25)

### Final image
n = 100+int(np.exp(210/5))
List_target_norm = [targets*np.sqrt(n) for targets in List_targets]
min_target_norm = min(List_target_norm)
max_target_norm = max(List_target_norm)
t_second_1 = (n*p/(min_target_norm*2))*np.log(p*min_target_norm)
t_second_2 = (n*p/(max_target_norm*2))*np.log(p*max_target_norm)
t_second = max(t_second_1, t_second_2)

times = t.linspace(0, 1.75*t_second, nb_points)
Loss = t.zeros_like(times)
for target_norm in List_target_norm:
    norm = norm_2(n, target_norm, p, cos_angle, norm_0, times)
    s_p = scalar_product(n, target_norm, p, cos_angle, times)
    Loss += loss(n, target_norm, p, norm, s_p)/p


fig, ax = plt.subplots(figsize=(6.4+0.15, 4.8+0.15))
VBarre1, = ax.plot([t_second_1/t_second, t_second_1/t_second],[0, plateau_loss], lw=2, c='grey', label=r"$L_{\infty}(t)$")
VBarre1.set_dashes([2,1])
VBarre2, = ax.plot([t_second_2/t_second, t_second_2/t_second],[plateau_loss, max_Loss], lw=2, c='grey')
VBarre2.set_dashes([2,1])
HBarre1, = ax.plot([0, t_second_2/t_second],[max_Loss, max_Loss], lw=2, c='grey')
HBarre1.set_dashes([2,1])
HBarre2, = ax.plot([t_second_2/t_second, 1],[plateau_loss, plateau_loss], lw=2, c='grey')
HBarre2.set_dashes([2,1])
HBarre3, = ax.plot([1, 1.75],[0, 0], lw=2, c='grey')
HBarre3.set_dashes([2,1])
ax.plot(times/t_second, Loss, lw=2, c='k', label=r"$L_{n}(t)$")
ax.set_xlim(0-0.05, 1.75-0.05)
ax.set_ylim(0-0.05, max_Loss*1.1)
ax.set_xticks([0, t_second_1/t_second, t_second_2/t_second])
ax.set_xticklabels(labels=[0, r"$\frac{1}{||D_1||}$", r"$\frac{1}{||D_2||}$"], fontsize=16)
ax.set_yticks([max_Loss, 0])
ax.set_yticklabels(labels=[r"$L_n(0)$", 0], fontsize=16)
exponent = int(np.log(n)/np.log(10))
decimal = int((n/(10**exponent))*10)/10
ax.set_xlabel("Normalized time", fontsize=16)
ax.set_ylabel("Loss", fontsize=16)
ax.set_title(rf"Rescaled loss trajectory in dimension ${decimal}\times 10^{ {exponent} }$", fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.legend(fontsize=16)
plt.subplots_adjust(bottom=0.15, left=0.15)
plt.savefig("Images/Phase_Transition.png", dpi=300)