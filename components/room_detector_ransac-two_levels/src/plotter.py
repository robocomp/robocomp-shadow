# pf_plotter.py
import sys, time, json, os, math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

def ensure_keys(d):
    for k in ["tick","loss_best","num_features","ess","births","deaths", "n_particles", "ess_pct", "weight_entropy",
              "x_std","y_std","theta_std", "period"]:
        d.setdefault(k, [])
    return d

def main():
    if len(sys.argv) < 2:
        print("Usage: pf_plotter.py /path/to/history.json")
        sys.exit(1)
    path = sys.argv[1]
    print(f"[plotter] watching {path}")

    plt.ioff()
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7,1, figsize=(7,11), sharex=True)
    fig.canvas.manager.set_window_title("PF Diagnostics")

    (l1,) = ax1.plot([], [], label="loss_best")
    ax1.set_ylabel("loss"); ax1.grid(True, alpha=0.3); ax1.legend(loc="upper right")

    (l2,) = ax2.plot([], [], label="#features")
    ax2.set_ylabel("features"); ax2.set_ylim(-0.2, 3.2)
    ax2.grid(True, alpha=0.3); ax2.legend(loc="upper right")

    (l3,) = ax3.plot([], [], label="ESS%")
    ax3.set_ylabel("ESS%"); ax3.set_xlabel("tick")
    ax3.grid(True, alpha=0.3); ax3.legend(loc="upper right")

    (l4,) = ax4.plot([], [], label="particles")
    ax4.set_ylabel("particles");  ax4.set_xlabel("tick")
    ax4.grid(True, alpha=0.3);  ax4.legend(loc="upper right")
    ax4.yaxis.get_major_locator().set_params(integer=True)

    (l5,) = ax5.plot([], [], label="weight_entropy")
    ax5.set_ylabel("weight_entropy"); ax5.set_xlabel("tick")
    ax5.grid(True, alpha=0.3); ax5.legend(loc="upper right")

    l6x, = ax6.plot([], [], label="x_std")
    l6y, = ax6.plot([], [], label="y_std")
    l6t, = ax6.plot([], [], label="theta_std")
    ax6.set_ylabel("std"); ax6.set_xlabel("tick")
    ax6.grid(True, alpha=0.3); ax6.legend(loc="upper right")

    (l7,) = ax7.plot([], [], label="period")
    ax7.set_ylabel("period (ms)"); ax7.set_xlabel("tick")
    ax7.grid(True, alpha=0.3); ax7.legend(loc="upper right")

    last_mtime = 0
    while True:
        try:
            st = os.stat(path)
            if st.st_mtime != last_mtime and st.st_size > 0:
                last_mtime = st.st_mtime
                with open(path, "r") as f:
                    hist = ensure_keys(json.load(f))

                t = hist["tick"]
                if len(t) >= 2:
                    l1.set_data(t, hist["loss_best"]); ax1.relim(); ax1.autoscale_view()
                    l2.set_data(t, hist["num_features"]); ax2.relim(); ax2.autoscale_view()
                    l3.set_data(t, hist["ess_pct"]); ax3.relim(); ax3.autoscale_view()
                    l4.set_data(t, hist["n_particles"]); ax4.relim(); ax4.autoscale_view()
                    l5.set_data(t, hist["weight_entropy"]); ax5.relim(); ax5.autoscale_view()
                    l6x.set_data(t, hist["x_std"])
                    l6y.set_data(t, hist["y_std"])
                    l6t.set_data(t, hist["theta_std"])
                    ax6.relim(); ax6.autoscale_view()
                    l7.set_data(t, hist["period"]); ax7.relim(); ax7.autoscale_view()

                    # births/deaths markers on ax2
                    births_x = [ti for ti,b in zip(t, hist["births"]) if b>0]
                    deaths_x = [ti for ti,d in zip(t, hist["deaths"]) if d>0]
                    ymin, ymax = ax2.get_ylim()
                    yb = ymax - 0.1*(ymax-ymin)
                    yd = ymax - 0.2*(ymax-ymin)
                    for line in list(ax2.lines)[1:]:
                        line.remove()
                    if births_x: ax2.plot(births_x, [yb]*len(births_x), "g|", markersize=12)
                    if deaths_x: ax2.plot(deaths_x, [yd]*len(deaths_x), "r|", markersize=12)

                    fig.canvas.draw_idle()

            plt.pause(0.5)
            time.sleep(0.1)
        except KeyboardInterrupt:
            break
        except Exception as e:
            # keep running even if file is temporarily empty/invalid
            time.sleep(0.2)

if __name__ == "__main__":
    main()
