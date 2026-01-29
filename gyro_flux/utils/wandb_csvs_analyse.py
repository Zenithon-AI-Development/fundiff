import wandb

api = wandb.Api()
run = api.run("Zenithon-AI/gyro_flux_target_training/l51tawpx")

# Get history with all diagnostic columns
history = run.history(keys=[
    "loss/train",
    "diag/padding_ratio",
    "diag/early_phase_ratio",
    "diag/late_phase_ratio",
    "diag/mean_seq_len",
])

# Drop rows with NaN (steps where not all metrics were logged)
df = history.dropna()

print(f"Total data points: {len(df)}")
print()
print("=== Correlations with loss/train ===")
print(f"corr(loss, padding_ratio):    {df['loss/train'].corr(df['diag/padding_ratio']):.3f}")
print(f"corr(loss, early_phase_ratio): {df['loss/train'].corr(df['diag/early_phase_ratio']):.3f}")
print(f"corr(loss, late_phase_ratio):  {df['loss/train'].corr(df['diag/late_phase_ratio']):.3f}")
print(f"corr(loss, mean_seq_len):      {df['loss/train'].corr(df['diag/mean_seq_len']):.3f}")
print()
print("=== Summary Stats ===")
print(df.describe())
