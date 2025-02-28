import numpy as np 
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation

from dataclasses import dataclass, field

fps = 60 # 1 / seconds
time = 15 # seconds

@dataclass
class Config():
    D = 2.1
    # Konstanten
    D = 2.1     # Fenderkonstante für beide Federn gleichzeitig [D] = N/m
    m = 0.075   # Masse [m] = kg
    k = 0.05    # Dämpfung [k] = Ns/m

    # Resonanz
    resonance = False
    s = 0.1     # Auslenkung [s] = m
    f = 0.793   # Frequenz [f] = 1/s

    x0 = 0
    v0 = 0

steps = time * fps
t = np.linspace(0, time, steps)


def resonance_delta(t, c):
    return c.s * np.sin(2 * np.pi * c.f * t) if c.resonance else 0

# solve differential equation(s)
def solve(t, c):
    def ivp(t, x):    
        F_Hooke     = -c.D * x[0]
        F_Dämpfung  = -c.k * x[1]
        F_Resonanz  = +c.D * resonance_delta(t, c)
        return (x[1], (F_Hooke + F_Dämpfung + F_Resonanz) / c.m)
    
    sol = solve_ivp(ivp, [0, t[-1]], y0=[c.x0, c.v0], t_eval=t)
    return sol.y[0], sol.y[1]

def E_kin(v):   return 1/2 * c.m * v**2
def E_Spann(x): return 1/2 * c.D * x**2

c = Config()
c.x0 = 1
xt, vt = solve(t, c)

# calculate amplitude (without resonance!)
at = c.x0 * np.e ** (-(c.k/(2*c.m)) * t)

class Style():
    # background_color = "#fcf5e4"
    def setup():
        # setup visuals
        plt.style.use("Solarize_Light2")
        
        options = {
            "font": {
                # "family" : "default", "weight" : "bold",
                "size" : 16
                },
            "axes" : {
                "labelsize" : 22
                }
            }
        
        for k, v in options.items():
            matplotlib.rc(k, **v)

class EnergyVisualisation():
    def __init__(self, ax):
        ax.set_ylabel("$E$ in $J$")
        ax.set_ylim(0, 1.25 * E_Spann(np.max(xt)))

        self.labels = ["$E_{Spann}$", "$E_{Kin}$", "$E_{Gesamt}$"]
        self.bar = ax.bar(self.labels, [0, 0, 0], color=["C0", "C6", "C1"])

    def update(self, i):
        for b, h in zip(self.bar, [E_Spann(xt[i]), E_kin(vt[i]), E_Spann(xt[i]) + E_kin(vt[i])]):
            b.set_height(h)
        return (self.bar)

class GraphVisualisation():
    def __init__(self, ax):
        ax.set_xlabel("$t$ in $s$")
        ax.set_xlim(0, time)
        ax.set_ylabel("$x$ in $m$" if c.resonance else "$x$, $a$ in $m$")
        # ax.set_ylim(-x0*1.25, x0*1.25)

        ax.plot(t, xt, linestyle="--", color="C0")
        self.line_x, = ax.plot([], [], label="$x$", color="C0")

        if not c.resonance:
            ax.plot(t, at, linestyle="--", color="C2")
        self.line_a, = ax.plot([], [], label="$a$",  color="C2")
        self.line_a.set_visible(not c.resonance)

        if not c.resonance:
            ax.legend()
        
    def update(self, i):
        self.line_x.set_data(t[:i], xt[:i])
        self.line_a.set_data(t[:i], at[:i])
        return (self.line_x, self.line_a)


class SimVisualisation():
    def __init__(self, ax):
        ax.axis('off') # warning: disables display of axes, lines and labels
        ax.axis("equal")
        s = 2.0
        ax.set_xlim(-s, s); ax.set_ylim(-s, s)

        [self.line_s1,], [self.line_s2,] = [ax.plot([], [], lw=4, color="gray") for i in range(2)]

        self.rect = plt.Rectangle((0, 0), width=0, height=0, color="C0")
        ax.add_patch(self.rect)
        
    def update(self, i):
        r = 0.25
        n = 12; l = 1.1
        def spring(b):
            dl = l - b * resonance_delta(t[i], c)
            return ([(b*dl + xt[i]) * (m/(n-1)) - b*(dl + r) for m in range(n)],
                    [2*r * (m % 2 - 0.5) * (0<m<n-1)         for m in range(n)])
        
        self.line_s1.set_data(*spring( 1))
        self.line_s2.set_data(*spring(-1))
        
        self.rect.set(xy=(xt[i]-r, 0-r))
        self.rect.set(width=2*r, height=2*r)
        
        return (self.line_s1, self.line_s2, self.rect)

def render(classes, rows, cols, gridspec_kw = {}, name = None):
    Style.setup()
    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, figsize=(16, 8), 
        gridspec_kw=gridspec_kw
        )

    visualisations = [vis(ax) for ax, vis in zip(axes, classes)]
    
    def animate(i, visualisations):
        arr = []
        for vis in visualisations:
            arr += vis.update(i)
        return arr

    anim = animation.FuncAnimation(
        fig, animate, steps, 
        interval=1/fps * 1000, blit=True, 
        fargs=(visualisations,)
        )

    if name != None:
        from os import path
        directory = path.dirname(path.realpath(__file__))
        
        anim.save(path.join(directory, f"{name}.mp4"), 
                writer = "ffmpeg", fps=fps, 
                # savefig_kwargs={"facecolor": Style.background_color} # needed to fix bug
                )
        
    plt.show()


render([SimVisualisation, EnergyVisualisation],
    1, 2, gridspec_kw={"width_ratios": [3, 1]}, 
    # name="energie"
    )

render([SimVisualisation, GraphVisualisation],
    1, 2, gridspec_kw={"width_ratios": [1, 1]}, 
    # name="graphen"
    )

def graph_resonance():
    Style.setup()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    frequencies = np.linspace(0, 2, 200)
    amplitudes = []

    for f in frequencies:
        x = solve(t, 0, 0)
        amplitudes.append(np.max(np.abs(x[-fps * 5:])))
        
    ax.set_xlabel("$f_{Resonanz}$ in $\\frac{1}{s}$")
    ax.set_ylabel("$a$ in $m$")

    ax.plot(frequencies, amplitudes, lw=4, color="C1")

    plt.show()

def print_values():
    print("Messwerte t in s, a in m:")
    print([(round(t[i], 2), round(at[i], 2)) for i in range(0, len(t), len(t) // 10)])