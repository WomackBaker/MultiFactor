# IP Spoofing Detection Branch

This branch simulates and detects **IP spoofing attacks** within a multi-factor authentication (MFA) context.  
It generates synthetic user behavior, augments data with GAN-produced samples and trust scores, injects realistic attacker patterns, and trains an SVM classifier to detect spoofed login sessions.

---

## End-to-end Pipeline

generate.py → sample_data.csv

gan.py → output_with_trust_scores.csv

split.py → output/train.csv + output/test.csv (with attackers)

svm.py → models/svm_model.sav + results/test_result_svm.csv + ROC/EER visualization

---

## Files & Purpose

### `generate.py` — Data generation

**Purpose:** Simulate realistic login activity for many users across regions, device types, and time windows.

**Key features produced:**
  - region/timezone, GPS (lat/lon), home/work location simulation
  - OS/manufacturer/device type codes
  - session start epoch, session duration, time since last login
  - `ip_address_as_int` (IP converted to integer), `ip_reputation_code`
  - `vpn_tor` usage, typing speed, click pattern, role/scope codes, `historic_risk_score`

**Output:** `sample_data.csv`
- **Usage:**
  ```bash
  python generate.py <num_users> <rows_per_user>
  # Example:
  python generate.py 100 1000
---
Gan.py - GAN-based data synthesis + trust scoring
Purpose: Train a PyTorch GAN to generate additional realistic continuous/categorical samples and produce dynamic trust scores per sample.
Design highlights:
•	Generator outputs continuous features + categorical logits (e.g., region_tz_code).
•	Discriminator distinguishes real vs. fake on continuous features.
•	Preprocessing: categorical mappings, boolean conversion, MinMax scaling.
•	FEATURE_TRUST_SCORES maps features → base trust weight; calculate_dynamic_trust adjusts trust based on context (ip reputation, device type, visit count).
•	After generating samples, inverse-scaling and mapping restores human-readable values, and trust scores are interleaved into output columns.
Outputs: output_with_trust_scores.csv and saved snapshots in ./saved/
Usage:
python gan.py <num_samples>
# Example:
python gan.py 500
---
split.py — Attack injection & dataset prep
•	Purpose: Inject IP-spoofing attacks into the dataset (modifies IPs, location radius, login timing), add labels, normalize, and split into train/test sets.
•	Attack simulation logic:
o	Select a normal sample → mutate:
	ip_address_as_int += random.randint(2000, 5000) (simulate different IP)
	location_conf_radius += random.randint(200, 500) (larger location uncertainty)
	time_since_last_login_mins set to small value to simulate abnormal rapid logins
o	Assign label = 1 for spoofed entries.
o	Set trust_score low (0.0–0.5) 70% of the time, or moderate-high (0.7–0.9) 30% to simulate camouflage.
•	Realism enhancements:
o	Flip a small fraction of labels (~2–3%) to inject noise.
o	Slightly adjust trust scores (attackers ↓, normals ↑).
o	Normalize features using MinMaxScaler.
o	Stratified train/test split (70/30).
•	Outputs:
o	output/train.csv
o	output/test.csv
•	Usage:
python  split.py <attacker_count> <path_to_data>
# Example:
python split.py 500 output_with_trust_scores.csv
---
svm.py — Model training & evaluation
•	Purpose: Train an SVM to detect spoofed sessions and evaluate using ROC / EER.
•	Modes:
o	q — Quick (default) — fixed SVM hyperparameters for fast training
o	f — Full GridSearch — searches C and gamma over a small grid with cross-validation (slower)
•	Evaluation metrics & outputs:
o	Computes ROC curve, AUC, EER, and threshold corresponding to EER.
o	Saves model to models/svm_model.sav (pickle).
o	Saves test scores & labels to results/test_result_svm.csv.
o	Shows ROC plot.
•	Usage:
python svm.py
Then choose mode: (q) Quick or (f) Full
---
Outputs Summary
•	sample_data.csv — raw synthetic login records (from generate.py)
•	output_with_trust_scores.csv — GAN-augmented data including trust scores (from gan.py)
•	output/train.csv, output/test.csv — normalized training and testing data with attacker labels (from split.py)
•	models/svm_model.sav — trained SVM model (from svm.py)
•	results/test_result_svm.csv — model test scores and labels
•	ROC/EER plot displayed by svm.py
---

# Full Run Example
1. Go to GenerateData folder
2. python generate.py 100 1000
3. Go to GAN folder
4. python gan.py 500
5. Go to SVM folder
6. python split.py 500 output_with_trust_scores.csv
7. python svm.py
8. choose 'q' or 'f' when prompted
