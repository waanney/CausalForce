import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json

def segments_from_binary(seq):
    segs=[]
    in_seg=False
    start=0
    for i,v in enumerate(seq+[0]):
        if v and not in_seg:
            in_seg=True; start=i
        elif not v and in_seg:
            in_seg=False; segs.append((start,i))
    return segs

def intersect_segments(a,b):
    out=[]; i=j=0
    while i<len(a) and j<len(b):
        a0,a1=a[i]; b0,b1=b[j]
        s=max(a0,b0); e=min(a1,b1)
        if s<e: out.append((s,e))
        if a1<b1: i+=1
        else: j+=1
    return out

def subtract_segments(a,b):
    out=[]
    for s,e in a:
        cur_s=s
        for bs,be in intersect_segments([(s,e)], b):
            if cur_s<bs: out.append((cur_s,bs))
            cur_s=max(cur_s,be)
        if cur_s<e: out.append((cur_s,e))
    return out

def clip_segments(segs, end_excl):
    out=[]
    for s,e in segs:
        s2=max(0,min(s,end_excl)); e2=max(0,min(e,end_excl))
        if s2<e2: out.append((s2,e2))
    return out

def plot_timeline_frame(img, gt, methods, t, out_png,
                        row_h=0.35, y_fontsize=18,
                        bar_h=0.06, bar_pad=0.02):
    from matplotlib.ticker import MaxNLocator
    from matplotlib.patches import Patch, Rectangle

    T = len(gt); names = list(methods.keys())
    gt_segs = clip_segments(segments_from_binary(list(gt)), t+1)


    # Aligned Video
    left  = 0.28          # left margin (legend area)
    right = 1.00          # right edge aligned with figure width
    bottom= 0.2
    gap   = 0.05          # gap between video and axes below
    fig_w = 10
    img_h_in  = fig_w * (256/900)
    plot_h_in = 1.6 + row_h*len(names)
    H = img_h_in + plot_h_in
    frac_img = (right-left) * fig_w * (256/900) / H
    top   = 1.0 - frac_img - gap
    
    fig = plt.figure(figsize=(fig_w, H))

    # ---------- Top: video axis (aligned with left and bottom) ----------
    ax_img = fig.add_axes([left, 1.0-frac_img, right-left, frac_img])
    if getattr(img, "mode", None) != "RGB":
        img = img.convert("RGB")
    ax_img.imshow(img, aspect="auto", interpolation="lanczos")  # 'nearest' also works
    ax_img.set_axis_off(); ax_img.set_frame_on(False)

    # ---------- Bottom: timeline axis (share the same left/right) ----------
    ax = fig.add_axes([left, bottom, right-left, top-bottom])

    ystep = 1.0; y0 = 0.5
    ax.set_xlim(0, T); ax.set_ylim(0, 0.5 + 1.0*len(names))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks([]); ax.set_xlabel("")
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.set_title("")
    ax.set_yticks([0.5 + i*1.0 for i in range(len(names))])
    ax.set_yticklabels(names, fontsize=y_fontsize)
    for lb in ax.get_yticklabels(): lb.set_fontweight("bold")


    # GT brake window (gray)
    for (s, e) in gt_segs:
        ax.axvspan(s, e, ymin=0, ymax=1, alpha=0.15, color="gray")

    # Brakes per method: red=in-window, orange=off-window
    for i, name in enumerate(names):
        y = y0 + i*ystep - 0.3; h = 0.6
        br = clip_segments(segments_from_binary(list(methods[name])), t+1)
        inwin = intersect_segments(br, gt_segs)
        offwin = subtract_segments(br, gt_segs)
        for (s, e) in inwin:
            ax.broken_barh([(s, e-s)], (y, h), facecolors="#EE4741FF")
        for (s, e) in offwin:
            ax.broken_barh([(s, e-s)], (y, h), facecolors="#F39F4C")

    # Current-frame cursor (optional)
    ax.axvline(t + 0.5, color="black", linewidth=1, linestyle="--", alpha=0.6)

    # Legend
    legend_elems = [
        Patch(facecolor="gray", alpha=0.15, label="GT Brake window"),
        Patch(facecolor="#EE4741FF", label="Brake in-window"),
        Patch(facecolor="#F39F4C", label="Brake off-window"),
    ]
 
    #ax.legend(handles=legend_elems, loc="upper right")
    fig.legend(
        handles=legend_elems,
        loc="center left",   # in figure coordinates
        bbox_to_anchor=(0.02, 0.50),   # (0.02,0.35) for full video
        frameon=True,
        prop={"size": 10}

    )   

    p = (t+1)/T  # progress in [0,1]
    
    # Gray background bar (from x=0 to 1; exactly aligned with axis width)
    ax.text(-0.02, 1 + bar_pad + bar_h/2, "Timeline",
        transform=ax.transAxes, ha="right", va="center",
        fontsize=14, fontweight="bold", clip_on=False, zorder=12)

    # Gray background bar
    ax.add_patch(Rectangle((0, 1 + bar_pad), 1 , bar_h,
                        transform=ax.transAxes, clip_on=False,
                        facecolor="0.9", edgecolor="none", zorder=10))

    # Red progress
    ax.add_patch(Rectangle((0, 1 + bar_pad), p, bar_h,
                        transform=ax.transAxes, clip_on=False,
                        facecolor="#8CEC7F", edgecolor="none", zorder=11))

    fig.subplots_adjust(
        left=0.28,   # ← left margin (0–1). Increase (e.g., 0.30) for a wider left gutter
        right=0.98,
        bottom=0.08,
        top=0.92     # reserve top space for the timeline
    )
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)



scenario = "Scenario_Name"

distance_braking_path = f'/path/to/your/{scenario}_brake.npy' # distance method brake output path
ca_braking_path = f'/path/to/your/{scenario}_brake.npy'       # collision anticipation method brake output path
bp_braking_path = f'/path/to/your/{scenario}_brake.npy'       # behavior prediction method brake output path
tp_braking_path = f'/path/to/your/{scenario}_brake.npy'       # trajectory prediction method brake output path
causalforce_braking_path = f'/path/to/your/{scenario}_brake.npy'     # CausalForce method brake output path


gt_brake = []
data_path = f'/path/to/your/{scenario}/' # path to data folder, containing only one scenario

measurement_path = data_path + 'measurements'
measurement_list = sorted(os.listdir(measurement_path))

# Load ground truth brake data
brake_count = 0
for measurement_file in measurement_list:
    with open(measurement_path + '/' + measurement_file) as f:
        measurement_data = json.load(f)

    if measurement_data['brake']:
        gt_brake.append(1)
    else:
        gt_brake.append(0)

# Load frame images
frame_images = []   
image_paths = sorted(os.listdir(data_path + 'rgb_front'))
for i,img_path in enumerate(image_paths):
    image = Image.open(data_path + 'rgb_front/' + img_path).convert("RGB")
    frame_images.append(image)

# Prepare ground truth and method outputs
gt_brake = np.array(gt_brake[2:])
distance_braking = np.load(distance_braking_path, allow_pickle=True)
ca_braking = np.load(ca_braking_path, allow_pickle=True)
bp_braking = np.load(bp_braking_path, allow_pickle=True)
tp_braking = np.load(tp_braking_path, allow_pickle=True)
causalforce_braking = np.load(causalforce_braking_path, allow_pickle=True)

# Prepare methods dict
methods = {
    "CausalForce": causalforce_braking.tolist(), 
    "Distance": distance_braking.tolist(),
    "CA": ca_braking.tolist(),
    "BP": bp_braking.tolist(),
    "TP": tp_braking.tolist(),
}

# Generate GIF
gif_path = make_timeline_gif(
    frame_images = frame_images,
    gt=list(gt_brake),            
    methods=methods,             
    out_gif=f"/home/hcis-s02/Desktop/vis/brake_vis/{scenario}.gif", # output gif path
    step=1,                      
    duration_ms=120,             
    scenario = scenario
)
print(gif_path)