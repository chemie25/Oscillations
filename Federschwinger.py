import numpy as np 
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation

from dataclasses import dataclass, field

from os import path
directory = path.dirname(path.realpath(__file__))


fps = 30 # 1 / seconds
time = 10 # seconds

@dataclass
class Config():
    # Konstanten
    D = 1.5     # Fenderkonstante [D] = N/m
    m = 0.075   # Masse [m] = kg
    k = 0.05    # Dämpfung [k] = Ns/m

    x0 = 0
    v0 = 0
    
    # Resonanz
    resonance = False
    s = 0.1     # Auslenkung [s] = m
    f = 0.71   # Frequenz [f] = 1/s



def resonance_delta(t, c):
    return c.s * np.sin(2*np.pi * c.f*t) if c.resonance else 0

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

t = np.linspace(0, time, time * fps)

class Style():
    # background_color = "#fcf5e4"
    def setup(set_style=True):
        # setup visuals
        if set_style:
            plt.style.use("Solarize_Light2")
        
        options = {
            "font": {
                "family" : "Calibri", 
                # "weight" : "bold",
                "size" : 18
                },
            "axes" : {
                "labelsize" : 24
                }
            }
        
        for k, v in options.items():
            matplotlib.rc(k, **v)

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
        # self.line_s2.set_data(*spring(-1))
        
        self.rect.set(xy=(xt[i]-r, 0-r))
        self.rect.set(width=2*r, height=2*r)
        
        return (self.line_s1, self.line_s2, self.rect)

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
        ax.set_ylim(-1*1.25, 1*1.25)
    
        graphs = []
        graphs.append([xt, "$x$", "C0"])
        
        # plot amplitude
        if not c.resonance and c.k != 0:
            # calculate amplitude (without resonance!)
            at = c.x0 * np.e ** (-(c.k/(2*c.m)) * t)
            graphs.append([at, "$A$", "C2"])
            
            print("Messwerte t in s, A(t) in m")
            indicies = range(0, len(t), len(t) // 4)
            print("t in s: ",                   [round(t[i], 2)  for i in indicies])
            print("Amplitude A(t) in m: ",      [round(at[i], 2) for i in indicies])
        
        self.lines = []
        self.lines_data = []
        
        for [data, label, color] in graphs:
            # plot dashed
            ax.plot(t, data, linestyle="--", color=color)
            line, = ax.plot([], [], label=label, color=color)
            self.lines.append(line)
            self.lines_data.append(data)

        ax.set_ylabel(", ".join([l for [_, l, _] in graphs]) + " in $m$")
        
        if len(graphs) > 1:
            ax.legend()
            
    def update(self, i):
        for l, d in zip(self.lines, self.lines_data):
            l.set_data(t[:i], d[:i])
        return self.lines

def render(classes, rows, cols, gridspec_kw = {}, name = None, export = False):
    Style.setup()
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8), gridspec_kw=gridspec_kw)

    visualisations = [vis(ax) for ax, vis in zip(axes, classes)]
    
    def animate(i, visualisations):
        arr = []
        for vis in visualisations:
            arr += vis.update(i)
        return arr

    anim = animation.FuncAnimation(
        fig, animate, time * fps, 
        interval=1/fps * 1000, blit=True, 
        fargs=(visualisations,)
        )

    if export and name != None:       
        anim.save(path.join(directory, f"videos/{name}.mp4"), 
                writer = "ffmpeg", fps=fps,
                # savefig_kwargs={"facecolor": Style.background_color} # needed to fix bug
                )
    else:
        plt.show()


def graph_engergy():
    Style.setup()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    
    ax.set_xlabel("$t$ in $s$")
    ax.set_ylabel("$E_{Ges}$ in $J$")

    ax.plot(t, [E_Spann(xt[i]) + E_kin(vt[i]) for i in range(len(t))], lw=4, color="C1")
    
    plt.show()


def graph_resonance():
    Style.setup()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    frequencies = np.linspace(0, 2, 200)
    amplitudes = []

    for f in frequencies:
        c = Config()
        c.resonance = True; c.f = f
        xt, _ = solve(t, c)
        amplitudes.append(np.max(np.abs(xt)))
        
    ax.set_xlabel("$f_{Erreger}$ in $\\frac{1}{s}$")
    ax.set_ylabel("$\hat{x}$ in $m$")

    ax.plot(frequencies, amplitudes, lw=4, color="C1")

    plt.show()

def graph_position():
    Style.setup(False)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    
    ax.set_xticks(np.arange(0, 10+1, 0.5))
    
    ax.set_xlabel("$t$ in $s$")
    ax.set_xlim(0, time)
    ax.set_ylim(-1*1.25, 1*1.25)

    ax.plot(t, xt, color="C0")

    ax.set_ylabel("$x(t)$ in $m$")
    plt.show()

export = False

# Ungedämpft
c = Config()
c.x0 = 0.6; c.k = 0
xt, vt = solve(t, c)

render([SimVisualisation, EnergyVisualisation],
    1, 2, gridspec_kw={"width_ratios": [3, 1]}, 
    name="Ungedämpft Energie Umwandlungen", export=export
    )

render([SimVisualisation, GraphVisualisation],
    1, 2, gridspec_kw={"width_ratios": [2, 3]}, 
    name="Ungedämpft Graphen", export=export
    )


c = Config()
c.x0 = 1
xt, vt = solve(t, c)

render([SimVisualisation, EnergyVisualisation],
    1, 2, gridspec_kw={"width_ratios": [3, 1]}, 
    name="Gedämpft Energie Umwandlungen", export=export
    )

render([SimVisualisation, GraphVisualisation],
    1, 2, name="Gedämpft Graphen", export=export)

# graph_engergy()
# graph_position()

for ff in [0.5, 1.0, 1.5]:
    c = Config()
    c.resonance = True; c.f = ff * c.f
    xt, vt = solve(t, c)

    render([SimVisualisation, GraphVisualisation],
        1, 2, name=f"Resonanz Frequenz {int(ff * 100)}", export=export)

graph_resonance()