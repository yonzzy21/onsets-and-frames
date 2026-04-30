import re
import matplotlib.pyplot as plt
import os
import numpy as np

def smooth(scalars, weight):
    if not scalars: return []
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def parse_log(filename):
    iterations, f1_scores, losses = [], [], []
    iter_regex = re.compile(r"\[Iteration (\d+)\]")
    f1_regex = re.compile(r"note f1\s+:\s+([\d.]+)")
    loss_regex = re.compile(r"loss[:\s]+([\d.]+)")
    resume_regex = re.compile(r"Resuming from iteration (\d+)")
    start_iter = 0

    if not os.path.exists(filename):
        return [], [], [], 0

    with open(filename, 'r') as f:
        current_iter = None
        for line in f:
            r_match = resume_regex.search(line)
            if r_match: start_iter = int(r_match.group(1))
            i_match = iter_regex.search(line)
            if i_match: current_iter = int(i_match.group(1))
            f_match = f1_regex.search(line)
            if f_match and current_iter is not None:
                f1_scores.append(float(f_match.group(1)))
                iterations.append(current_iter)
                current_iter = None 
            l_match = loss_regex.search(line)
            if l_match: losses.append(float(l_match.group(1)))
    return iterations, f1_scores, losses, start_iter

# Training speeds (observed from logs)
MEL_SPEED = 35.15 # it/s
CQT_SPEED = 24.99 # it/s

mel_log, cqt_log = "slurm_7406299.out", "slurm-7423774.out"
m_it, m_f1, m_loss, m_s = parse_log(mel_log)
c_it, c_f1, c_loss, c_s = parse_log(cqt_log)

# Calculate Peaks
m_peak_f1 = max(m_f1) if m_f1 else 0
m_peak_iter = m_it[m_f1.index(m_peak_f1)] if m_f1 else 0
m_peak_time = (m_peak_iter - m_s) / MEL_SPEED / 60 # minutes

c_peak_f1 = max(c_f1) if c_f1 else 0
c_peak_iter = c_it[c_f1.index(c_peak_f1)] if c_f1 else 0
c_peak_time = (c_peak_iter - c_s) / CQT_SPEED / 60 # minutes

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# TOP PANEL: F1 SCORE
ax1.plot(m_it, m_f1, color='#3498db', alpha=0.15)
ax1.plot(m_it, smooth(m_f1, 0.85), label=f'Mel (Peak: {m_peak_f1:.3f})', color='#3498db', linewidth=2)
ax1.plot(c_it, c_f1, color='#e74c3c', alpha=0.15)
ax1.plot(c_it, smooth(c_f1, 0.85), label=f'CQT (Peak: {c_peak_f1:.3f})', color='#e74c3c', linewidth=2)

# Add Peak Annotations
if m_peak_iter:
    ax1.axvline(x=m_peak_iter, color='#3498db', linestyle='--', alpha=0.5)
    ax1.text(m_peak_iter, 0.1, f' Mel Peak\n Iter: {m_peak_iter/1000:.1f}k\n Time: {m_peak_time:.1f}m', color='#2980b9', fontweight='bold')

if c_peak_iter:
    ax1.axvline(x=c_peak_iter, color='#e74c3c', linestyle='--', alpha=0.5)
    ax1.text(c_peak_iter, 0.3, f' CQT Peak\n Iter: {c_peak_iter/1000:.1f}k\n Time: {c_peak_time:.1f}m', color='#c0392b', fontweight='bold')

ax1.set_title('Validation Performance (Note F1) with Peak Convergence Analysis', fontsize=14, fontweight='bold')
ax1.set_ylabel('F1 Score', fontsize=12)
ax1.legend(loc='upper right'); ax1.grid(True, alpha=0.3); ax1.set_ylim(0, 1.0)

# BOTTOM PANEL: LOSS
if m_loss:
    m_l_it = [m_s + i for i in range(len(m_loss))]
    ax2.plot(m_l_it, m_loss, alpha=0.1, color='#3498db')
    ax2.plot(m_l_it, smooth(m_loss, 0.995), color='#3498db', label='Mel Loss', linewidth=2)
if c_loss:
    c_l_it = [c_s + i for i in range(len(c_loss))]
    ax2.plot(c_l_it, c_loss, alpha=0.1, color='#e74c3c')
    ax2.plot(c_l_it, smooth(c_loss, 0.995), color='#e74c3c', label='CQT Loss', linewidth=2)

ax2.set_title('Training Convergence (Loss)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Training Iteration (Steps)', fontsize=12)
ax2.set_ylabel('Loss Value (Log Scale)', fontsize=12)
ax2.grid(True, alpha=0.3); ax2.set_yscale('log'); ax2.legend()

plt.tight_layout()
plt.savefig("training_results_comparison_final.png", dpi=300)
print(f"Success! Final annotated graph saved as training_results_comparison_final.png")
