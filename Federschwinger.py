import numpy as np 
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation

fps = 60 # 1 / seconds
time = 30 # seconds

# Konstanten
D = 2.1     # Fenderkonstante für beide Federn gleichzeitig [D] = N/m
m = 0.075   # Masse [m] = kg
k = 0.05    # Dämpfung [k] = Ns/m

# Resonanz
resonance = True
s = 0.1     # Auslenkung [s] = m
f = 0.793   # Frequenz [f] = 1/s

# Anfangswerte
if not resonance:
    x0 = 1  # Auslenkung zu Beginn [x] = m
    v0 = 0  # Geschwindigkeit zu Beginn [v] = m/s
else:
    x0 = 0
    v0 = 0

steps = time * fps
t = np.linspace(0, time, steps)

def resonance_delta(t):
    return s * np.sin(2 * np.pi * f * t) if resonance else 0

# solve differential equation(s)
def solve(t, x0, v0):
    def ivp(t, x):    
        F_Hooke     = -D * x[0]
        F_Dämpfung  = -k * x[1]
        F_Resonanz  = +D * resonance_delta(t)
        return (x[1], (F_Hooke + F_Dämpfung + F_Resonanz) / m)
    
    sol = solve_ivp(ivp, [0, t[-1]], y0=[x0, v0], t_eval=t)
    return sol.y[0], sol.y[1]

def E_kin(v):   return 1/2 * m * v**2
def E_Spann(x): return 1/2 * D * x**2

xt, vt = solve(t, x0, v0)

# calculate amplitude (without resonance!)
at = x0 * np.e ** (-(k/(2*m)) * t)

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
        ax.set_ylabel("$x$ in $m$" if resonance else "$x$, $a$ in $m$")
        # ax.set_ylim(-x0*1.25, x0*1.25)

        ax.plot(t, xt, linestyle="--", color="C0")
        self.line_x, = ax.plot([], [], label="$x$", color="C0")

        if not resonance:
            ax.plot(t, at, linestyle="--", color="C2")
        self.line_a, = ax.plot([], [], label="$a$",  color="C2")
        self.line_a.set_visible(not resonance)

        if not resonance:
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
            dl = l - b * resonance_delta(t[i])
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
    # name="video"
    )


render([SimVisualisation, GraphVisualisation],
    1, 2, gridspec_kw={"width_ratios": [1, 1]}, 
    # name="video"
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