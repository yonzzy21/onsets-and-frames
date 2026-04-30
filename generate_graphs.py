import re
import matplotlib.pyplot as plt
import os
import numpy as np

def parse_log(filename):
    iterations, f1_scores = [], []
    iter_regex = re.compile(r"\[Iteration (\d+)\]")
    f1_regex = re.compile(r"note f1\s+:\s+([\d.]+)")
    resume_regex = re.compile(r"Resuming from iteration (\d+)")
    start_iter = 0

    if not os.path.exists(filename):
        return [], [], 0

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
    return iterations, f1_scores, start_iter

MEL_SPEED, CQT_SPEED = 35.15, 24.99
mel_log, cqt_log = "slurm_7406299.out", "slurm-7423774.out"
m_it, m_f1, m_s = parse_log(mel_log)
c_it, c_f1, c_s = parse_log(cqt_log)

# Calculate Peaks
m_peak_f1 = max(m_f1) if m_f1 else 0
m_peak_iter = m_it[m_f1.index(m_peak_f1)] if m_f1 else 0
m_peak_time = (m_peak_iter - m_s) / MEL_SPEED / 60

c_peak_f1 = max(c_f1) if c_f1 else 0
c_peak_iter = c_it[c_f1.index(c_peak_f1)] if c_f1 else 0
c_peak_time = (c_peak_iter - c_s) / CQT_SPEED / 60

# Single Panel Plot
plt.figure(figsize=(14, 8))

# Plot raw data
plt.plot(m_it, m_f1, label=f'Mel (Peak: {m_peak_f1:.3f})', color='#3498db', linewidth=2, alpha=0.9)
plt.plot(c_it, c_f1, label=f'CQT (Peak: {c_peak_f1:.3f})', color='#e74c3c', linewidth=2, alpha=0.9)

# Add Peak Annotations
if m_peak_iter:
    plt.axvline(x=m_peak_iter, color='#3498db', linestyle='--', alpha=0.5)
    plt.text(m_peak_iter + 2000, 0.1, f' Mel Peak\n {m_peak_iter/1000:.1f}k Iter\n {m_peak_time:.1f}m Time', color='#2980b9', fontweight='bold', fontsize=11)
if c_peak_iter:
    plt.axvline(x=c_peak_iter, color='#e74c3c', linestyle='--', alpha=0.5)
    plt.text(c_peak_iter + 2000, 0.3, f' CQT Peak\n {c_peak_iter/1000:.1f}k Iter\n {c_peak_time:.1f}m Time', color='#c0392b', fontweight='bold', fontsize=11)

plt.title('GuitarSet Transcription: Validation Performance (Note F1)', fontsize=16, fontweight='bold')
plt.xlabel('Training Iteration (Steps)', fontsize=14, fontweight='bold')
plt.ylabel('Note F1 Score', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim(0, 1.0)
plt.xlim(0, 150000) # Focusing on the interesting part of training

plt.tight_layout()
plt.savefig("final_performance_graph.png", dpi=300)
print("Success! Final single-panel graph saved as final_performance_graph.png")
