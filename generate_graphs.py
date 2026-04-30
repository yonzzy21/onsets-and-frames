import re
import matplotlib.pyplot as plt
import os

def parse_log(filename):
    iterations, f1_scores, losses = [], [], []
    
    iter_regex = re.compile(r"\[Iteration (\d+)\]")
    f1_regex = re.compile(r"note f1\s+:\s+([\d.]+)")
    loss_regex = re.compile(r"loss[:\s]+([\d.]+)")
    
    # We need to find the "Resuming from iteration XXX" to align the loss steps
    resume_regex = re.compile(r"Resuming from iteration (\d+)")
    start_iter = 0

    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return [], [], [], 0

    with open(filename, 'r') as f:
        current_iter = None
        for line in f:
            resume_match = resume_regex.search(line)
            if resume_match:
                start_iter = int(resume_match.group(1))

            iter_match = iter_regex.search(line)
            if iter_match:
                current_iter = int(iter_match.group(1))
            
            f1_match = f1_regex.search(line)
            if f1_match and current_iter is not None:
                f1_scores.append(float(f1_match.group(1)))
                iterations.append(current_iter)
                current_iter = None 
            
            loss_match = loss_regex.search(line)
            if loss_match:
                losses.append(float(loss_match.group(1)))
                
    return iterations, f1_scores, losses, start_iter

mel_log, cqt_log = "slurm_7406299.out", "slurm-7423774.out"

mel_iters, mel_f1, mel_loss, mel_start = parse_log(mel_log)
cqt_iters, cqt_f1, cqt_loss, cqt_start = parse_log(cqt_log)

# Create x-axis for loss by adding the start_iter offset
mel_loss_iters = [mel_start + i for i in range(len(mel_loss))]
cqt_loss_iters = [cqt_start + i for i in range(len(cqt_loss))]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# TOP PANEL: F1 SCORE
ax1.plot(mel_iters, mel_f1, label='Mel (Baseline)', color='#3498db', linewidth=2)
ax1.plot(cqt_iters, cqt_f1, label='CQT (Experimental)', color='#e74c3c', linewidth=2)
ax1.set_title('Validation Performance (Note F1)', fontsize=14, fontweight='bold')
ax1.set_ylabel('F1 Score', fontsize=12)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_ylim(0, 1.0)

# BOTTOM PANEL: TRAINING LOSS
if mel_loss:
    ax2.plot(mel_loss_iters, mel_loss, alpha=0.2, color='#3498db', label='Mel Loss')
if cqt_loss:
    ax2.plot(cqt_loss_iters, cqt_loss, alpha=0.2, color='#e74c3c', label='CQT Loss')

ax2.set_title('Training Convergence (Loss)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Training Iteration', fontsize=12)
ax2.set_ylabel('Loss Value (Log Scale)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig("training_results_comparison.png", dpi=300)
print("Success! Graph with iteration axis saved as training_results_comparison.png")
