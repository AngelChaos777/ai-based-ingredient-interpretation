#!/usr/bin/env python3
"""
Ensemble training for allergen multi-label classifier.
Trains 5 BCE / ASL MobileBERT models with different random seeds,
then evaluates ensemble averaging.

Supports Asymmetric Loss (ASL) for false-positive suppression and
label-count regularisation for calibrated set-size predictions.

Usage:
    # Baseline BCE ensemble
    python scripts/train_ensemble.py

    # ASL with default gamma_minus=2 + set-size regularisation
    python scripts/train_ensemble.py --asl-gamma-minus 2 --set-size-alpha 0.05

    # Stronger ASL focusing
    python scripts/train_ensemble.py --asl-gamma-minus 4 --asl-gamma-plus 0.5 --set-size-alpha 0.1

Output:
    models/mobilebert_allergen_ensemble/seed_{N}/  (each seed's best model)
    Prints ensemble evaluation on test set.
"""

import os, sys, random, warnings, json, shutil, argparse
sys.path.insert(0, 'notebooks')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score
from scipy.special import expit

from utils.data_utils import (
    get_data_directories, load_labeled_data, create_stratified_splits,
    augment_dataframe, augment_targeted, clean_training_data,
    create_counter_examples
)
from utils.text_processing import (
    clean_html, get_allergen_list, allergens_to_binary
)
from utils.model_utils import (
    compute_class_weights, find_best_thresholds, apply_thresholds,
    AsymmetricLoss, label_count_regularization, TemperatureScaling,
    ClassifierChainMobileBERT,
)
from utils.evaluation import (
    print_classification_report, print_per_class_metrics
)

# ─── Classifier chain orderings per ensemble seed ───
# Each seed uses a different label ordering to reduce chain-order bias.

CHAIN_ORDERINGS = {
    42:    ['milk', 'wheat', 'soy', 'eggs', 'peanuts', 'fish', 'tree_nuts', 'shellfish'],
    123:   ['shellfish', 'tree_nuts', 'fish', 'peanuts', 'eggs', 'soy', 'wheat', 'milk'],
    456:   ['wheat', 'milk', 'soy', 'eggs', 'peanuts', 'fish', 'shellfish', 'tree_nuts'],
    789:   ['eggs', 'milk', 'fish', 'wheat', 'peanuts', 'soy', 'tree_nuts', 'shellfish'],
    1111:  ['tree_nuts', 'shellfish', 'fish', 'peanuts', 'eggs', 'soy', 'wheat', 'milk'],
}

# ─── CLI ───
parser = argparse.ArgumentParser(description="Train ensemble of MobileBERT allergen classifiers")
parser.add_argument("--asl-gamma-plus", type=float, default=0.0,
                    help="ASL gamma+ for positives (default 0.0 = off)")
parser.add_argument("--asl-gamma-minus", type=float, default=0.0,
                    help="ASL gamma- for negatives (default 0.0 = off, set >0 to enable ASL)")
parser.add_argument("--set-size-alpha", type=float, default=0.0,
                    help="Label-count regularisation strength (default 0.0 = off)")
parser.add_argument("--temperature-scale", action="store_true",
                    help="Apply post-hoc temperature scaling after training")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--early-stopping", type=int, default=3)
args = parser.parse_args()

use_asl = args.asl_gamma_minus > 0
use_set_size = args.set_size_alpha > 0
use_temp_scale = args.temperature_scale

# ─── Configuration ───
ENSEMBLE_SEEDS = [42, 123, 456, 789, 1111]
SEED = 42  # Base seed for data split
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
WEIGHT_DECAY = args.weight_decay
EARLY_STOPPING = args.early_stopping
model_name = "google/mobilebert-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Loss: {'ASL' if use_asl else 'BCE'}"
      f"(gamma_plus={args.asl_gamma_plus}, gamma_minus={args.asl_gamma_minus})"
      f" | Set-size reg: {args.set_size_alpha}"
      f" | Temp scaling: {use_temp_scale}")

# Set seed for data split
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ─── Load & Prepare Data ───
dirs = get_data_directories()
df = load_labeled_data(filepath=os.path.join(dirs['final'], 'labeled_dataset_enhanced_fixed.csv'))
df["ingredients_cleaned"] = df["ingredients_text_en"].apply(clean_html)
df["labels"] = df["detected_allergens"].apply(allergens_to_binary)

X = df["ingredients_cleaned"].values
y = np.array(df["labels"].tolist())

train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = create_stratified_splits(
    X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=SEED
)
print(f"Data: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")

# ─── Tokenizer & Max Length ───
tokenizer = AutoTokenizer.from_pretrained(model_name)
sample_lengths = [len(tokenizer.encode(t, truncation=False)) for t in train_texts[:500]]
MAX_LEN = min(int(np.percentile(sample_lengths, 95)), 256)
print(f"Max length: {MAX_LEN}")

# Pre-tokenize val and test once (shared across all seeds)
val_encodings = tokenizer(
    list(val_texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
)
test_encodings = tokenizer(
    list(test_texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
)

class FoodDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()} | {"labels": torch.tensor(self.labels[idx], dtype=torch.float)}
    def __len__(self):
        return len(self.labels)

val_dataset = FoodDataset(val_encodings, np.array(val_labels))
test_dataset = FoodDataset(test_encodings, np.array(test_labels))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ─── Training Utilities ───
def run_train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device, set_size_alpha=0.0):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            prev_preds=batch["labels"],
        )
        # Reorder labels to match chain order for loss computation
        labels_chain = batch["labels"][:, model._chain_ord]
        loss = loss_fn(logits, labels_chain)
        # Optional set-size regularisation
        if set_size_alpha > 0:
            loss = loss + label_count_regularization(
                logits, labels_chain, alpha=set_size_alpha
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits_canonical = model.reorder_logits_to_canonical(logits)
            # Reorder labels to match chain order for loss computation
            labels_chain = batch["labels"][:, model._chain_ord]
            loss = loss_fn(logits, labels_chain)
            total_loss += loss.item()
            all_logits.append(logits_canonical.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = expit(logits)
    preds = (probs > 0.5).astype(int)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    samples_f1 = f1_score(labels, preds, average="samples", zero_division=0)
    return total_loss / len(dataloader), macro_f1, samples_f1, logits, labels

def predict(model, dataloader, device):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits_canonical = model.reorder_logits_to_canonical(logits)
            all_logits.append(logits_canonical.cpu().numpy())
    return np.concatenate(all_logits)

# ─── Train Each Seed ───
n_classes = len(get_allergen_list())
ensemble_models = {}
ensemble_val_f1s = {}
ensemble_dir = os.path.join(dirs['base'], "models", "mobilebert_allergen_ensemble")
os.makedirs(ensemble_dir, exist_ok=True)

loss_label = "ASL" if use_asl else "BCE"
if use_set_size:
    loss_label += f"+set{args.set_size_alpha}"

for seed in ENSEMBLE_SEEDS:
    print(f"\n{'='*50}")
    print(f"Training seed {seed}/1111  [{loss_label}]")
    print(f"{'='*50}")

    # Set all seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Re-augment with this seed (different shuffle -> different augmentations)
    train_df = pd.DataFrame({
        'text': list(train_texts),
        'ingredients_cleaned': list(train_texts),
        'labels': [list(l) for l in np.array(train_labels)],
    })
    # Clean training data: remove non-food items, fix label errors
    train_df = clean_training_data(train_df)
    print(f"  After cleaning: {len(train_df)} train samples")

    # Augmentation
    aug_df = augment_dataframe(train_df, num_augmented=2)
    target_df = augment_targeted(train_df, target_allergens={"tree_nuts": 8, "shellfish": 3, "fish": 2, "peanuts": 2})
    # Note: augment_targeted already returns the original rows plus targeted
    # copies, so do NOT concat train_df again here — that would double-count
    # every original sample in the training distribution.
    combined = pd.concat([target_df, aug_df], ignore_index=True)

    # Counter-examples: teach model that certain ingredients do NOT imply allergens
    counter_df = create_counter_examples(
        train_df,
        target_ingredients={
            "eggs": ["whey", "whey powder", "whey permeate"],
            "soy": ["soya oil", "soya lecithin", "soy lecithin", "soya"],
        },
        copies=3,
        random_state=seed,
    )
    if len(counter_df) > 0:
        print(f"  Counter-examples added: {len(counter_df)} rows")
        combined = pd.concat([combined, counter_df], ignore_index=True)

    tr_texts = combined["text"].tolist()
    tr_labels = np.array(combined["labels"].tolist())

    # Fresh tokenize + dataloader
    enc = tokenizer(tr_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    tr_dataset = FoodDataset(enc, tr_labels)
    tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Class weights (slightly different per seed due to augmentation)
    cw = compute_class_weights(tr_labels).to(device)

    # Fresh model
    model = ClassifierChainMobileBERT(
        backbone_name=model_name,
        n_classes=n_classes,
        chain_order=CHAIN_ORDERINGS[seed],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(tr_labels) // BATCH_SIZE)
    total_steps = steps_per_epoch * EPOCHS
    warmup = max(200, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)

    # Loss function — ASL or BCE
    if use_asl:
        loss_fn = AsymmetricLoss(
            gamma_plus=args.asl_gamma_plus,
            gamma_minus=args.asl_gamma_minus,
            pos_weight=cw,
            reduction='mean'
        )
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=cw, reduction='mean')

    # Training loop
    best_macro = 0.0
    best_samples = 0.0
    best_state = None
    patience = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_train_epoch(
            model, tr_loader, optimizer, scheduler, loss_fn, device,
            set_size_alpha=args.set_size_alpha if use_set_size else 0.0
        )
        val_loss, val_macro, val_samples, _, _ = evaluate(model, val_loader, loss_fn, device)

        # Track best by macro F1 (standard) but log samples F1 too
        if val_macro > best_macro:
            best_macro = val_macro
            best_samples = val_samples
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        marker = '*' if epoch > 1 and val_macro == best_macro else ''
        print(f"  Epoch {epoch:2d}/{EPOCHS} | train: {train_loss:.4f}"
              f" | val macro: {val_macro:.4f} samples: {val_samples:.4f} {marker}", end='')
        if patience > 0:
            print(f" (patience {patience})", end='')
        print()

        if patience >= EARLY_STOPPING:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Save chain model weights + order marker
    seed_dir = os.path.join(ensemble_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(seed_dir, "pytorch_model.bin"))
    with open(os.path.join(seed_dir, "chain_order.json"), "w") as f:
        json.dump(model.chain_order, f)
    with open(os.path.join(seed_dir, "model_config.json"), "w") as f:
        json.dump({"backbone_name": model_name, "n_classes": n_classes}, f, indent=2)
    tokenizer.save_pretrained(seed_dir)

    ensemble_models[seed] = model
    ensemble_val_f1s[seed] = best_macro
    print(f"  ✓ Seed {seed} done — Best Val Macro F1: {best_macro:.4f}  Samples F1: {best_samples:.4f}")

# ─── Ensemble Evaluation ───
print(f"\n\n{'='*60}")
print("All seeds trained! Evaluating ensemble...")
print(f"{'='*60}")
for s in ENSEMBLE_SEEDS:
    print(f"  Seed {s:4d}: Val Macro F1 = {ensemble_val_f1s[s]:.4f}")

# Collect test/val logits from each seed (pre-sigmoid for temp scaling)
all_test_logits = []
all_val_logits = []
for s in ENSEMBLE_SEEDS:
    m = ensemble_models[s]
    all_test_logits.append(predict(m, test_loader, device))
    all_val_logits.append(predict(m, val_loader, device))

# Optional temperature scaling on validation ensemble logits
if use_temp_scale:
    val_logits_ensemble = np.mean(all_val_logits, axis=0)
    scaler = TemperatureScaling()
    t_opt = scaler.calibrate(val_logits_ensemble, np.array(val_labels))
    print(f"\n  Temperature scaling: T = {t_opt:.4f}")

    # Apply to test logits
    test_logits_ensemble = np.mean(all_test_logits, axis=0)
    test_probs_ensemble = expit(scaler(test_logits_ensemble))
    val_probs_ensemble = expit(scaler(val_logits_ensemble))
else:
    test_logits_ensemble = np.mean(all_test_logits, axis=0)
    val_logits_ensemble = np.mean(all_val_logits, axis=0)
    test_probs_ensemble = expit(test_logits_ensemble)
    val_probs_ensemble = expit(val_logits_ensemble)

# Find optimal thresholds on validation set
best_thresholds_ensemble = find_best_thresholds(val_probs_ensemble, np.array(val_labels))

# Apply to test set
preds_ensemble = apply_thresholds(test_probs_ensemble, best_thresholds_ensemble)
test_labels_np = np.array(test_labels)

print(f"\n{'='*60}")
print("  Ensemble — Test Set Performance")
print(f"{'='*60}")
print_classification_report(test_labels_np, preds_ensemble, get_allergen_list(), prefix="Ensemble")

# Samples avg
samples_f1 = f1_score(test_labels_np, preds_ensemble, average='samples', zero_division=0)
macro_f1 = f1_score(test_labels_np, preds_ensemble, average='macro', zero_division=0)
micro_f1 = f1_score(test_labels_np, preds_ensemble, average='micro', zero_division=0)

# Single model (seed 42) for comparison
single_probs = expit(all_test_logits[0])
single_preds = apply_thresholds(single_probs, best_thresholds_ensemble)
single_samples_f1 = f1_score(test_labels_np, single_preds, average='samples', zero_division=0)

# Global threshold for samples avg
def find_best_samples_threshold(probs, labels, step=0.01):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.01, 0.99, step):
        p = (probs >= t).astype(int)
        f = f1_score(labels, p, average='samples', zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1

best_t, best_samples_f1 = find_best_samples_threshold(val_probs_ensemble, np.array(val_labels))
global_preds = (test_probs_ensemble >= best_t).astype(int)
ensemble_global_samples_f1 = f1_score(test_labels_np, global_preds, average='samples', zero_division=0)

# Comparison
print(f"\n{'='*60}")
print("  Final Comparison")
print(f"{'='*60}")
print(f"{'Approach':<30} {'Samples F1':>12} {'Macro F1':>12}")
print(f"{'─'*56}")
print(f"{'Single BCE (seed 42)':<30} {single_samples_f1:>12.4f} {'N/A':>12}")
print(f"{'Ensemble (per-class thresh)':<30} {samples_f1:>12.4f} {macro_f1:>12.4f}")
print(f"{'Ensemble (global thresh T='+str(best_t)+')':<30} {ensemble_global_samples_f1:>12.4f} {'N/A':>12}")
print(f"{'─'*56}")
print(f"{'Improvement vs single':<30} {samples_f1 - single_samples_f1:>+12.4f}")
print(f"{'Loss config':<30} {loss_label:>12}")

# Save best config — the chain model is written to its own directory so it
# does NOT clobber models/mobilebert_allergen_final, the standard
# AutoModelForSequenceClassification model loaded by load_model_and_tokenizer()
# and notebooks/08. Load the chain model with ClassifierChainMobileBERT.from_pretrained().
final_dir = os.path.join(dirs['models'], "mobilebert_allergen_chain_final")

# Pick the threshold strategy on the VALIDATION set. Comparing test-set F1s
# (samples_f1 / ensemble_global_samples_f1 above) would leak the test set
# into model selection and inflate the reported numbers.
val_preds_per_class = apply_thresholds(val_probs_ensemble, np.array(val_labels))
val_samples_f1_per_class = f1_score(np.array(val_labels), val_preds_per_class, average='samples', zero_division=0)
val_global_preds = (val_probs_ensemble >= best_t).astype(int)
val_samples_f1_global = f1_score(np.array(val_labels), val_global_preds, average='samples', zero_division=0)

if val_samples_f1_global > val_samples_f1_per_class:
    chosen_thresholds = np.array([best_t] * n_classes)
    best_source = f"Ensemble (global T={best_t})"
    best_f1_val = val_samples_f1_global
else:
    chosen_thresholds = best_thresholds_ensemble
    best_source = "Ensemble (per-class)"
    best_f1_val = val_samples_f1_per_class

# Report the chosen config on the test set (held out from selection)
test_preds_chosen = apply_thresholds(test_probs_ensemble, chosen_thresholds)
test_samples_f1_chosen = f1_score(test_labels_np, test_preds_chosen, average='samples', zero_division=0)

if os.path.exists(final_dir):
    shutil.rmtree(final_dir)
os.makedirs(final_dir, exist_ok=True)

# Save seed 42 as canonical chain model file
torch.save(ensemble_models[42].state_dict(), os.path.join(final_dir, "pytorch_model.bin"))
with open(os.path.join(final_dir, "chain_order.json"), "w") as f:
    json.dump(ensemble_models[42].chain_order, f)
with open(os.path.join(final_dir, "model_config.json"), "w") as f:
    json.dump({"backbone_name": model_name, "n_classes": n_classes}, f, indent=2)
tokenizer.save_pretrained(final_dir)

np.save(os.path.join(dirs['models'], "best_thresholds.npy"), chosen_thresholds)

hybrid_config = {
    "ml_thresholds": chosen_thresholds.tolist(),
    "rule_conf_threshold": 0.5,
    "mode": "hard_override",
    "ensemble": {
        "n_models": len(ENSEMBLE_SEEDS),
        "seeds": ENSEMBLE_SEEDS,
        "method": "avg_probabilities"
    },
    "training": {
        "loss": loss_label,
        "asl_gamma_plus": args.asl_gamma_plus,
        "asl_gamma_minus": args.asl_gamma_minus,
        "set_size_alpha": args.set_size_alpha,
        "temperature_scaling": use_temp_scale,
    }
}
with open(os.path.join(dirs["models"], "hybrid_config.json"), "w") as f:
    json.dump(hybrid_config, f, indent=2)

print(f"\n{'='*60}")
print(f"  Saved: {best_source} (selected on validation)")
print(f"  Val Samples Avg F1: {best_f1_val:.4f}")
print(f"  Test Samples Avg F1 (chosen config): {test_samples_f1_chosen:.4f}")
print(f"  → {final_dir}")
print(f"  Loss config: {loss_label}")
print(f"{'='*60}")
