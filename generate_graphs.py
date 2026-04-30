import re
import matplotlib.pyplot as plt
import os

def parse_log(filename):
    iterations = []
    f1_scores = []
    losses = []
    
    # Matches "[Iteration 128500]"
    iter_regex = re.compile(r"\[Iteration (\d+)\]")
    # Matches "note f1        : 0.767"
    f1_regex = re.compile(r"note f1\s+:\s+([\d.]+)")
    # Matches tqdm output for loss: "loss: 0.1234"
    loss_regex = re.compile(r"loss:\s+([\d.]+)")
    
    with open(filename, 'r') as f:
        current_iter = None
        for line in f:
            iter_match = iter_regex.search(line)
            if iter_match:
                current_iter = int(iter_match.group(1))
            
            f1_match = f1_regex.search(line)
            if f1_match and current_iter is not None:
                f1_scores.append(float(f1_match.group(1)))
                iterations.append(current_iter)
                # Try to find a loss value in the surrounding lines
                current_iter = None 
            
            loss_match = loss_regex.search(line)
            if loss_match:
                # We'll just collect all losses we see
                losses.append(float(loss_match.group(1)))
                
    return iterations, f1_scores, losses

# Paths to your logs
mel_log = "slurm_7406299.out"
cqt_log = "slurm-7423774.out"

print(f"Parsing {mel_log}...")
mel_iters, mel_f1, mel_loss = parse_log(mel_log)

print(f"Parsing {cqt_log}...")
cqt_iters, cqt_f1, cqt_loss = parse_log(cqt_log)

# Create a two-panel plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# TOP PANEL: F1 SCORE
ax1.plot(mel_iters, mel_f1, label='Mel Spectrogram (Baseline)', alpha=0.8, color='#3498db', linewidth=2)
ax1.plot(cqt_iters, cqt_f1, label='CQT (Experimental)', alpha=0.8, color='#e74c3c', linewidth=2)
ax1.set_title('Validation Performance (Note F1)', fontsize=14, fontweight='bold')
ax1.set_ylabel('F1 Score', fontsize=12)
ax1.legend(loc='lower right')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_ylim(0, 1.0)

# BOTTOM PANEL: TRAINING LOSS
# Note: Losses are sampled more frequently than validation, so we just plot them against a generated x-axis
if mel_loss:
    ax2.plot(range(len(mel_loss)), mel_loss, alpha=0.3, color='#3498db', label='Mel Loss')
if cqt_loss:
    ax2.plot(range(len(cqt_loss)), cqt_loss, alpha=0.3, color='#e74c3c', label='CQT Loss')

ax2.set_title('Training Convergence (Loss)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Training Steps', fontsize=12)
ax2.set_ylabel('Loss Value', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_yscale('log') # Loss is often better viewed on a log scale

plt.tight_layout()

# Save the plot
output_file = "training_results_comparison.png"
plt.savefig(output_file, dpi=300)
print(f"Success! Final comparison graph saved as {output_file}")
