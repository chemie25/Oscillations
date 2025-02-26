import numpy as np 
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation

# from: https://medium.com/analytics-vidhya/understanding-oscillators-python-2813ec38781d
# https://www.leifiphysik.de/mechanik/mechanische-schwingungen/grundwissen/federpendel-gedaempft

fps = 60 # 1 / seconds
time = 30 # seconds
save = False

# Konstanten
D = 2.1     # Fenderkonstante für beide Federn gleichzeitig [D] = N/m
m = 0.075   # Masse [m] = kg
k = 0.05    # Dämpfung [k] = Ns/m

# Resonanz
resonanz = True
s = 0.1     # Auslenkung [s] = m
f = 0.793   # Frequenz [f] = 1/s

# Anfangswerte
if not resonanz:
    x0 = 1  # Auslenkung zu Beginn [x] = m
    v0 = 0  # Geschwindigkeit zu Beginn [v] = m/s
else:
    x0 = 0
    v0 = 0

steps = time * fps
t = np.linspace(0, time, steps)

def resonance_delta(t):
    return s * np.sin(2 * np.pi * f * t) if resonanz else 0

# solve differential equation
def sim_x(t, x0 = 0, v0 = 0):
    def ivp(t, x):    
        F_Hooke     = -D * x[0]
        F_Dämpfung  = -k * x[1]
        F_Resonanz  = +D * resonance_delta(t)
        return (x[1], (F_Hooke + F_Dämpfung + F_Resonanz) / m)
    
    sol = solve_ivp(ivp, [0, t[-1]], y0=[x0, v0], t_eval=t)
    return sol.y[0]

x = sim_x(t, x0, v0)

# calculate amplitude (without resonance!)
a = x0 * np.e ** (-(k/(2*m)) * t)

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

class GraphVisualisation():
    def __init__(self, ax):
        ax.set_xlabel("Zeit $t$ in $s$")
        ax.set_xlim(0, time)
        ax.set_ylabel("Auslenkung $x$" if resonanz else "Auslenkung $x$, Amplitude $a$ in $m$ in $m$")
        # ax.set_ylim(-x0*1.25, x0*1.25)

        ax.plot(t, x, linestyle="--", color="C0")
        self.line_x, = ax.plot([], [], label="Auslenkung $x$", color="C0")

        if not resonanz:
            ax.plot(t, a, linestyle="--", color="C2")
        self.line_a, = ax.plot([], [], label="Amplitude $a$",  color="C2")
        self.line_a.set_visible(not resonanz)

        if not resonanz:
            ax.legend()
        
    def update(self, i):
        self.line_x.set_data(t[:i], x[:i])
        self.line_a.set_data(t[:i], a[:i])
        return self.line_x, self.line_a


class SimVisualisation():
    def __init__(self, ax):
        ax.axis('off') # warning: disables display of axes, lines and labels
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)

        [self.line_s1,], [self.line_s2,] = [ax.plot([], [], lw=4, color="gray") for i in range(2)]

        self.rect = plt.Rectangle((0, 0), width=0, height=0, color="C0")
        ax_sim.add_patch(self.rect)
        
    def update(self, t, x):
        r = 0.25
        n = 12; l = 1.1
        def spring(b):
            dl = l - b * resonance_delta(t)
            return ([(b*dl + x) * (m/(n-1)) - b*(dl + r) for m in range(n)],
                    [2*r * (m % 2 - 0.5) * (0<m<n-1)     for m in range(n)])
        
        self.line_s1.set_data(*spring( 1))
        self.line_s2.set_data(*spring(-1))
        
        self.rect.set(xy=(x-r, 0-r))
        self.rect.set(width=2*r, height=2*r)
        
        return self.line_s1, self.line_s2, self.rect

def animate(i):
    return *sim_vis.update(i), *graph_vis.update(t[i], x[i])

# create plot(s)
Style.setup()
fig, (ax_sim, ax_graph) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

sim_vis     = GraphVisualisation(ax_graph)
graph_vis   = SimVisualisation(ax_sim)

anim = animation.FuncAnimation(fig, animate, steps, interval=1/fps * 1000, blit=True)

if save:
    from os import path
    directory = path.dirname(path.realpath(__file__))
    
    anim.save(path.join(directory, "video.mp4"), 
            writer = "ffmpeg", fps=fps, 
            # savefig_kwargs={"facecolor": Style.background_color} # needed to fix bug
            )

plt.show()



Style.setup()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

frequencies = np.linspace(0, 2, 200)
amplitudes = []

for f in frequencies:
    x = sim_x(t, 0, 0)
    amplitudes.append(np.max(np.abs(x[-fps * 5:])))
    

ax.set_xlabel("Resonanzfrequenz $f_{Resonanz}$ in $\\frac{1}{s}$")
ax.set_ylabel("Amplitude $a$ in $m$")
    
ax.plot(frequencies, amplitudes, lw=4, color="C1")

plt.show()


# Plot resonance

print("Messwerte t in s, a in m:")
print([(round(t[i], 2), round(a[i], 2)) for i in range(0, len(t), len(t) // 10)])