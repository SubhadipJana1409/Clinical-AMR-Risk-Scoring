"""
================================================================
Day 14 — Clinical AMR Risk Scoring (REAL DATA)
Author  : Subhadip Jana
Dataset : example_isolates — AMR R package
          2,000 clinical isolates × 40 antibiotics (R/S/I)

What is a Clinical Risk Score?
  A point-based system that converts patient/isolate features
  into a single interpretable score — similar to:
    • Wells Score (DVT/PE risk)
    • CHA₂DS₂-VASc (stroke risk in AFib)
    • CURB-65 (pneumonia severity)
    • APACHE II (ICU severity)

  Advantage over ML: transparent, actionable at bedside,
  no computer required after derivation.

Outcome: MDR status (resistant to ≥3 antibiotic classes)
         — the most clinically actionable AMR endpoint

Scoring Pipeline:
  1. Logistic regression → log-odds coefficients
  2. Convert coefficients → integer point scores
  3. Sum points → total risk score (0–20)
  4. Calibrate thresholds → Low / Moderate / High / Critical
  5. Validate: AUC, calibration curve, DCA
  6. Compare: Risk Score vs full ML model

Predictors:
  • Age group (clinically meaningful bands)
  • Ward (ICU > Clinical > Outpatient)
  • Bacterial species (Gram +ve vs –ve vs other)
  • Isolation year (temporal resistance trend)
  • Gender
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                              confusion_matrix, f1_score,
                              precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
import pickle, warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# SECTION 1: LOAD & ENGINEER CLINICAL FEATURES
# ─────────────────────────────────────────────────────────────

print("🔬 Loading example_isolates dataset...")
df = pd.read_csv("data/isolates.csv")
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year

META = ["date","patient","age","gender","ward","mo","year"]
AB_COLS = [c for c in df.columns if c not in META]

print(f"✅ {len(df)} isolates | {df.mo.nunique()} species | 2002–2017")

# ── MDR Outcome ──
# Antibiotic class mapping
AB_CLASS = {
    "PEN":"Penicillin","OXA":"Penicillin","FLC":"Penicillin",
    "AMX":"Penicillin","AMC":"Penicillin","AMP":"Penicillin",
    "TZP":"Cephalosporin","CZO":"Cephalosporin","FEP":"Cephalosporin",
    "CXM":"Cephalosporin","FOX":"Cephalosporin","CTX":"Cephalosporin",
    "CAZ":"Cephalosporin","CRO":"Cephalosporin",
    "GEN":"Aminoglycoside","TOB":"Aminoglycoside",
    "AMK":"Aminoglycoside","KAN":"Aminoglycoside",
    "TMP":"Sulfonamide","SXT":"Sulfonamide",
    "CIP":"Fluoroquinolone","MFX":"Fluoroquinolone",
    "VAN":"Glycopeptide","TEC":"Glycopeptide",
    "ERY":"Macrolide","CLI":"Macrolide","AZM":"Macrolide",
    "IPM":"Carbapenem","MEM":"Carbapenem",
    "NIT":"Other","FOS":"Other","LNZ":"Other","TCY":"Tetracycline",
    "TGC":"Tetracycline","DOX":"Tetracycline",
    "MTR":"Other","CHL":"Other","COL":"Polymyxin","MUP":"Other","RIF":"Other",
}

def count_resistant_classes(row):
    rc = set()
    for ab in AB_COLS:
        if row[ab] == "R" and ab in AB_CLASS:
            rc.add(AB_CLASS[ab])
    return len(rc)

df["n_resistant_classes"] = df.apply(count_resistant_classes, axis=1)
df["MDR"] = (df["n_resistant_classes"] >= 3).astype(int)
print(f"   MDR prevalence: {df['MDR'].mean()*100:.1f}% ({df['MDR'].sum()} isolates)")

# ── Clinical Feature Engineering ──
# Age groups — clinically meaningful
df["age_group"] = pd.cut(df["age"],
    bins=[0, 18, 40, 60, 75, 120],
    labels=["Paediatric (<18)", "Young adult (18–40)",
            "Middle-aged (40–60)", "Older adult (60–75)", "Elderly (75+)"])

# Species gram-stain category
GRAM_NEG = {"B_ESCHR_COLI","B_KLBSL_PNMN","B_PSDMN_AERG","B_PROTS_MRBL",
            "B_HLCBC_PYLR","B_ACINB_BUMN","B_ENTBC_CLOC","B_SERRT_MRCS"}
GRAM_POS = {"B_STPHY_AURS","B_STPHY_CONS","B_STPHY_EPDR","B_STPHY_HMNS",
            "B_STRPT_PNMN","B_STRPT_PYOG","B_ENCCS_FCLS","B_ENCCS_FECM"}

def gram_category(mo):
    if mo in GRAM_NEG: return "Gram-negative"
    if mo in GRAM_POS: return "Gram-positive"
    return "Other/Unknown"

df["gram"] = df["mo"].apply(gram_category)

# Year era
df["era"] = pd.cut(df["year"],
    bins=[2001, 2006, 2011, 2017],
    labels=["Early (2002–06)", "Mid (2007–11)", "Recent (2012–17)"])

print("\n   Clinical feature distribution:")
print(f"   Age groups:\n{df['age_group'].value_counts().to_string()}")
print(f"\n   Wards: {df['ward'].value_counts().to_dict()}")
print(f"\n   Gram: {df['gram'].value_counts().to_dict()}")
print(f"\n   Era: {df['era'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: UNIVARIATE RISK ANALYSIS
# ─────────────────────────────────────────────────────────────

print("\n📊 Univariate MDR risk analysis...")
from scipy.stats import chi2_contingency, mannwhitneyu

def odds_ratio(df, col, val):
    """Compute OR and 95% CI for a binary predictor."""
    a = ((df[col]==val) & (df["MDR"]==1)).sum()
    b = ((df[col]==val) & (df["MDR"]==0)).sum()
    c = ((df[col]!=val) & (df["MDR"]==1)).sum()
    d = ((df[col]!=val) & (df["MDR"]==0)).sum()
    if b==0 or c==0: return np.nan, np.nan, np.nan
    OR   = (a*d) / (b*c)
    logse = np.sqrt(1/a + 1/b + 1/c + 1/d)
    return OR, OR*np.exp(-1.96*logse), OR*np.exp(1.96*logse)

univar = []
# Ward
for ward in ["ICU","Clinical","Outpatient"]:
    mdr_pct = df[df["ward"]==ward]["MDR"].mean()*100
    OR, lo, hi = odds_ratio(df,"ward",ward)
    ct = pd.crosstab(df["ward"]==ward, df["MDR"])
    _, pval, _, _ = chi2_contingency(ct)
    univar.append({"Variable":"Ward","Category":ward,
                   "n":int((df["ward"]==ward).sum()),
                   "MDR_pct":round(mdr_pct,1),"OR":round(OR,2),
                   "CI_lo":round(lo,2),"CI_hi":round(hi,2),"p":round(pval,4)})
# Age group
for ag in df["age_group"].cat.categories:
    mdr_pct = df[df["age_group"]==ag]["MDR"].mean()*100
    mask    = df["age_group"]==ag
    ct = pd.crosstab(mask, df["MDR"])
    if ct.shape == (2,2):
        _, pval,_,_ = chi2_contingency(ct)
    else: pval=1.0
    univar.append({"Variable":"Age group","Category":str(ag),
                   "n":int(mask.sum()),"MDR_pct":round(mdr_pct,1),
                   "OR":np.nan,"CI_lo":np.nan,"CI_hi":np.nan,"p":round(pval,4)})
# Gram
for gram in ["Gram-negative","Gram-positive","Other/Unknown"]:
    mdr_pct = df[df["gram"]==gram]["MDR"].mean()*100
    OR, lo, hi = odds_ratio(df,"gram",gram)
    ct = pd.crosstab(df["gram"]==gram, df["MDR"])
    _, pval,_,_ = chi2_contingency(ct)
    univar.append({"Variable":"Gram stain","Category":gram,
                   "n":int((df["gram"]==gram).sum()),
                   "MDR_pct":round(mdr_pct,1),"OR":round(OR,2),
                   "CI_lo":round(lo,2),"CI_hi":round(hi,2),"p":round(pval,4)})
# Era
for era in df["era"].cat.categories:
    mdr_pct = df[df["era"]==era]["MDR"].mean()*100
    mask    = df["era"]==era
    ct = pd.crosstab(mask, df["MDR"])
    _, pval,_,_ = chi2_contingency(ct)
    univar.append({"Variable":"Era","Category":str(era),
                   "n":int(mask.sum()),"MDR_pct":round(mdr_pct,1),
                   "OR":np.nan,"CI_lo":np.nan,"CI_hi":np.nan,"p":round(pval,4)})

univar_df = pd.DataFrame(univar)
univar_df.to_csv("outputs/univariate_analysis.csv", index=False)

print("\n   MDR % by Ward:")
for _, r in univar_df[univar_df["Variable"]=="Ward"].iterrows():
    print(f"   {r['Category']:12s}: {r['MDR_pct']:.1f}% MDR | OR={r['OR']:.2f} | p={r['p']:.4f}")

# ─────────────────────────────────────────────────────────────
# SECTION 3: DERIVE POINT-BASED RISK SCORE
# ─────────────────────────────────────────────────────────────

print("\n🏥 Deriving clinical risk score (logistic regression coefficients)...")

# Build categorical feature matrix for scoring
score_features = pd.DataFrame({
    "age_60plus"    : (df["age"] >= 60).astype(int),
    "age_75plus"    : (df["age"] >= 75).astype(int),
    "ward_ICU"      : (df["ward"] == "ICU").astype(int),
    "ward_Clinical" : (df["ward"] == "Clinical").astype(int),
    "gram_neg"      : (df["gram"] == "Gram-negative").astype(int),
    "male"          : (df["gender"] == "M").astype(int),
    "era_recent"    : (df["year"] >= 2012).astype(int),
    "era_mid"       : ((df["year"] >= 2007) & (df["year"] < 2012)).astype(int),
})

y_mdr = df["MDR"].values

# Logistic regression → coefficients
lr = LogisticRegression(C=1.0, class_weight="balanced",
                         max_iter=2000, random_state=42)
lr.fit(score_features.values, y_mdr)

coef_df = pd.DataFrame({
    "Feature" : score_features.columns,
    "Coef"    : lr.coef_[0],
}).sort_values("Coef", ascending=False)

# Convert to integer points (scale: divide by min non-zero, round)
min_abs = coef_df["Coef"].abs()[coef_df["Coef"].abs() > 0].min()
coef_df["Points"] = (coef_df["Coef"] / min_abs).round(0).astype(int)
coef_df["Points"] = coef_df["Points"].clip(-3, 5)  # cap for interpretability

# Human-readable labels
POINT_LABELS = {
    "age_75plus"    : "Age ≥75 years",
    "age_60plus"    : "Age 60–74 years",
    "ward_ICU"      : "ICU admission",
    "ward_Clinical" : "Clinical ward",
    "gram_neg"      : "Gram-negative organism",
    "male"          : "Male gender",
    "era_recent"    : "Isolation 2012–2017",
    "era_mid"       : "Isolation 2007–2011",
}
coef_df["Label"] = coef_df["Feature"].map(POINT_LABELS)

print("\n   ┌─────────────────────────────────────────┬────────┐")
print("   │ Risk Factor                             │ Points │")
print("   ├─────────────────────────────────────────┼────────┤")
for _, row in coef_df.iterrows():
    sign = "+" if row["Points"] > 0 else ""
    print(f"   │ {row['Label']:39s} │  {sign}{row['Points']:3d}   │")
print("   └─────────────────────────────────────────┴────────┘")

# Compute total score per patient
points_map = dict(zip(coef_df["Feature"], coef_df["Points"]))
df["risk_score"] = sum(
    df_col * points_map[col]
    for col, df_col in score_features.items()
)

# Normalise to 0-based (shift min to 0)
df["risk_score"] = df["risk_score"] - df["risk_score"].min()

print(f"\n   Score range: {df['risk_score'].min()}–{df['risk_score'].max()}")
print(f"   Mean score : {df['risk_score'].mean():.2f} ± {df['risk_score'].std():.2f}")

# ─────────────────────────────────────────────────────────────
# SECTION 4: CALIBRATE RISK TIERS
# ─────────────────────────────────────────────────────────────

print("\n📐 Calibrating risk tiers...")

# Find thresholds: Low=<25th, Moderate=25–60th, High=60–85th, Critical>85th
thresholds = {
    "Low"      : (df["risk_score"].min(),
                  df["risk_score"].quantile(0.30)),
    "Moderate" : (df["risk_score"].quantile(0.30),
                  df["risk_score"].quantile(0.65)),
    "High"     : (df["risk_score"].quantile(0.65),
                  df["risk_score"].quantile(0.88)),
    "Critical" : (df["risk_score"].quantile(0.88),
                  df["risk_score"].max() + 1),
}

def assign_tier(score):
    for tier, (lo, hi) in thresholds.items():
        if lo <= score < hi:
            return tier
    return "Critical"

df["risk_tier"] = df["risk_score"].apply(assign_tier)

TIER_COLORS = {"Low":"#2ECC71","Moderate":"#F39C12",
               "High":"#E74C3C","Critical":"#8E44AD"}
TIER_ORDER  = ["Low","Moderate","High","Critical"]

print("\n   Risk Tier Statistics:")
print(f"   {'Tier':10s} {'n':>6s} {'Score range':>15s} {'MDR %':>8s}")
for tier in TIER_ORDER:
    sub = df[df["risk_tier"]==tier]
    sr  = f"{sub['risk_score'].min():.0f}–{sub['risk_score'].max():.0f}"
    mdr = sub["MDR"].mean()*100
    print(f"   {tier:10s} {len(sub):>6d} {sr:>15s} {mdr:>7.1f}%")

# ─────────────────────────────────────────────────────────────
# SECTION 5: VALIDATE RISK SCORE
# ─────────────────────────────────────────────────────────────

print("\n✅ Validating risk score (5-fold CV AUC)...")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_score = df["risk_score"].values.reshape(-1, 1)

# Score AUC
score_probs = cross_val_predict(
    LogisticRegression(C=1.0, max_iter=1000), X_score, y_mdr,
    cv=kf, method="predict_proba")[:,1]
score_auc  = roc_auc_score(y_mdr, score_probs)
score_brier= brier_score_loss(y_mdr, score_probs)

# Full ML model AUC (for comparison)
top_species = df["mo"].value_counts().head(15).index
df["sp_grp"] = df["mo"].apply(lambda x: x if x in top_species else "Other")
sp_dum = pd.get_dummies(df["sp_grp"], prefix="sp")
wd_dum = pd.get_dummies(df["ward"],   prefix="ward")
X_ml = pd.concat([sp_dum, wd_dum,
                   (df["gender"]=="M").astype(int).rename("gender_M"),
                   ((df["age"]-df["age"].mean())/df["age"].std()).rename("age"),
                   ((df["year"]-df["year"].mean())/df["year"].std()).rename("year")
                  ], axis=1).astype(float)

rf_ml = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=42, n_jobs=-1)
ml_probs = cross_val_predict(rf_ml, X_ml.values, y_mdr,
                              cv=kf, method="predict_proba")[:,1]
ml_auc   = roc_auc_score(y_mdr, ml_probs)
ml_brier = brier_score_loss(y_mdr, ml_probs)

print(f"\n   Risk Score AUC  : {score_auc:.4f} | Brier: {score_brier:.4f}")
print(f"   Full ML AUC     : {ml_auc:.4f} | Brier: {ml_brier:.4f}")
print(f"   ML gain over Score: {(ml_auc-score_auc)*100:+.2f} pp")

# Calibration data (for plot)
frac_pos_score, mean_pred_score = calibration_curve(y_mdr, score_probs, n_bins=10)
frac_pos_ml,    mean_pred_ml    = calibration_curve(y_mdr, ml_probs,    n_bins=10)

# ROC data
fpr_s, tpr_s, _ = roc_curve(y_mdr, score_probs)
fpr_ml,tpr_ml,_ = roc_curve(y_mdr, ml_probs)

# Save results
df[["age","gender","ward","mo","gram","age_group","era",
    "risk_score","risk_tier","MDR","n_resistant_classes"]].to_csv(
    "outputs/patient_risk_scores.csv", index=False)
coef_df.to_csv("outputs/scoring_coefficients.csv", index=False)

# Tier-level summary
tier_summary = df.groupby("risk_tier", observed=True).agg(
    n=("MDR","count"),
    mdr_pct=("MDR",lambda x: round(x.mean()*100,1)),
    mean_score=("risk_score","mean"),
    mean_age=("age","mean"),
).loc[TIER_ORDER]
tier_summary.to_csv("outputs/tier_summary.csv")

# ─────────────────────────────────────────────────────────────
# SECTION 6: DASHBOARD (9 panels)
# ─────────────────────────────────────────────────────────────

print("\n🎨 Generating dashboard...")

fig = plt.figure(figsize=(24, 20))
fig.suptitle(
    "Clinical AMR Risk Scoring System — REAL CLINICAL DATA\n"
    "Point-based MDR Risk Score derived from Logistic Regression coefficients\n"
    "example_isolates (AMR R package) | 2,000 isolates | 2002–2017",
    fontsize=15, fontweight="bold", y=0.99
)

# ── Plot 1: Forest plot — OR per risk factor ──
ax1 = fig.add_subplot(3, 3, 1)
forest_df = coef_df.sort_values("Coef")
colors_f  = ["#E74C3C" if c > 0 else "#2ECC71" for c in forest_df["Coef"]]
y_pos = np.arange(len(forest_df))
ax1.barh(y_pos, forest_df["Coef"].values,
         color=colors_f, edgecolor="black", linewidth=0.4, alpha=0.85)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(forest_df["Label"].values, fontsize=8)
ax1.axvline(0, color="black", lw=1.5)
for i, (val, pts) in enumerate(zip(forest_df["Coef"], forest_df["Points"])):
    sign = "+" if pts > 0 else ""
    ax1.text(val + (0.02 if val >= 0 else -0.02),
             i, f"{sign}{pts}pt", va="center",
             ha="left" if val >= 0 else "right", fontsize=8, fontweight="bold")
ax1.set_xlabel("Log-odds coefficient")
ax1.set_title("Risk Factor Coefficients\n(+pt = increases MDR risk)",
              fontweight="bold", fontsize=10)
ax1.legend(handles=[mpatches.Patch(color="#E74C3C",label="Risk factor"),
                    mpatches.Patch(color="#2ECC71",label="Protective")],
           fontsize=8)

# ── Plot 2: MDR % by risk tier (bar with CI) ──
ax2 = fig.add_subplot(3, 3, 2)
tier_mdr_pct  = [df[df["risk_tier"]==t]["MDR"].mean()*100 for t in TIER_ORDER]
tier_counts   = [len(df[df["risk_tier"]==t]) for t in TIER_ORDER]
tier_colors   = [TIER_COLORS[t] for t in TIER_ORDER]
bars2 = ax2.bar(TIER_ORDER, tier_mdr_pct, color=tier_colors,
                edgecolor="black", linewidth=0.6, alpha=0.87, width=0.6)
for bar, pct, n in zip(bars2, tier_mdr_pct, tier_counts):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{pct:.1f}%\n(n={n})", ha="center", fontsize=9, fontweight="bold")
ax2.set_ylabel("MDR Prevalence (%)")
ax2.set_title("MDR % by Risk Tier\n(Low → Critical)",
              fontweight="bold", fontsize=10)
ax2.set_ylim(0, 105)
ax2.axhline(df["MDR"].mean()*100, color="gray", lw=1.5,
            linestyle="--", label=f"Overall ({df['MDR'].mean()*100:.1f}%)")
ax2.legend(fontsize=8)

# ── Plot 3: Risk score distribution by tier ──
ax3 = fig.add_subplot(3, 3, 3)
for tier in TIER_ORDER:
    sub_scores = df[df["risk_tier"]==tier]["risk_score"]
    ax3.hist(sub_scores, bins=15, alpha=0.65,
             color=TIER_COLORS[tier], label=tier,
             edgecolor="white", linewidth=0.5)
for tier, (lo, hi) in thresholds.items():
    if hi < df["risk_score"].max()+1:
        ax3.axvline(hi, color=TIER_COLORS[tier], lw=1.5,
                    linestyle="--", alpha=0.7)
ax3.set_xlabel("Risk Score (points)")
ax3.set_ylabel("Number of Isolates")
ax3.set_title("Risk Score Distribution\n(coloured by tier)",
              fontweight="bold", fontsize=10)
ax3.legend(fontsize=9)

# ── Plot 4: ROC curve comparison ──
ax4 = fig.add_subplot(3, 3, 4)
ax4.plot(fpr_s,  tpr_s,  lw=2.5, color="#E74C3C",
         label=f"Risk Score (AUC={score_auc:.3f})", zorder=5)
ax4.plot(fpr_ml, tpr_ml, lw=2.5, color="#3498DB",
         linestyle="--", label=f"Full ML RF (AUC={ml_auc:.3f})")
ax4.plot([0,1],[0,1],"k:",lw=1,alpha=0.4)
ax4.fill_between(fpr_s, tpr_s, alpha=0.07, color="#E74C3C")
ax4.set_xlabel("False Positive Rate")
ax4.set_ylabel("True Positive Rate")
ax4.set_title("ROC Curve: Risk Score vs Full ML\n(5-fold CV)",
              fontweight="bold", fontsize=10)
ax4.legend(fontsize=9)

# ── Plot 5: Calibration curve ──
ax5 = fig.add_subplot(3, 3, 5)
ax5.plot(mean_pred_score, frac_pos_score, "s-",
         lw=2, color="#E74C3C", label=f"Risk Score (Brier={score_brier:.3f})")
ax5.plot(mean_pred_ml,    frac_pos_ml,    "o-",
         lw=2, color="#3498DB", linestyle="--",
         label=f"Full ML RF (Brier={ml_brier:.3f})")
ax5.plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="Perfect calibration")
ax5.fill_between([0,1],[0,1],alpha=0.04,color="gray")
ax5.set_xlabel("Mean Predicted Probability")
ax5.set_ylabel("Fraction Positive (Observed)")
ax5.set_title("Calibration Curve\n(closer to diagonal = better calibrated)",
              fontweight="bold", fontsize=10)
ax5.legend(fontsize=8)

# ── Plot 6: Risk tier by ward ──
ax6 = fig.add_subplot(3, 3, 6)
tier_ward = pd.crosstab(df["ward"], df["risk_tier"],
                         normalize="index")[TIER_ORDER] * 100
tier_ward.plot(kind="bar", ax=ax6, color=[TIER_COLORS[t] for t in TIER_ORDER],
               edgecolor="black", linewidth=0.4, width=0.7)
ax6.set_title("Risk Tier Distribution by Ward\n(% of ward patients)",
              fontweight="bold", fontsize=10)
ax6.set_xlabel(""); ax6.set_ylabel("% of Patients")
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=0, fontsize=10)
ax6.legend(title="Risk Tier", fontsize=8)

# ── Plot 7: Risk score vs age (scatter + regression) ──
ax7 = fig.add_subplot(3, 3, 7)
for tier in TIER_ORDER:
    sub = df[df["risk_tier"]==tier]
    ax7.scatter(sub["age"], sub["risk_score"],
                c=TIER_COLORS[tier], alpha=0.25, s=12,
                label=tier)
# Regression line
z = np.polyfit(df["age"], df["risk_score"], 1)
p = np.poly1d(z)
x_line = np.linspace(df["age"].min(), df["age"].max(), 100)
ax7.plot(x_line, p(x_line), "k-", lw=2, zorder=5, label="Trend")
from scipy.stats import spearmanr
r_age, p_age = spearmanr(df["age"], df["risk_score"])
ax7.set_xlabel("Patient Age (years)")
ax7.set_ylabel("Clinical Risk Score")
ax7.set_title(f"Risk Score vs Age\nSpearman r={r_age:.3f}, p={p_age:.4f}",
              fontweight="bold", fontsize=10)
ax7.legend(fontsize=7, ncol=2)

# ── Plot 8: Score card — point summary ──
ax8 = fig.add_subplot(3, 3, 8)
ax8.axis("off")
# Scorecard visualization
scorecard_rows = []
for _, row in coef_df.sort_values("Points", ascending=False).iterrows():
    sign = "+" if row["Points"] > 0 else ""
    scorecard_rows.append([row["Label"], f"{sign}{int(row['Points'])}"])

scorecard_rows.append(["─"*35, "─"*6])
scorecard_rows.append(["TOTAL SCORE", "0–20"])
scorecard_rows.append(["", ""])
scorecard_rows.append(["Low Risk",      f"< {thresholds['Moderate'][0]:.0f} pts"])
scorecard_rows.append(["Moderate Risk", f"{thresholds['Moderate'][0]:.0f}–{thresholds['High'][0]:.0f} pts"])
scorecard_rows.append(["High Risk",     f"{thresholds['High'][0]:.0f}–{thresholds['Critical'][0]:.0f} pts"])
scorecard_rows.append(["Critical Risk", f"≥ {thresholds['Critical'][0]:.0f} pts"])

tbl8 = ax8.table(cellText=scorecard_rows,
                 colLabels=["Clinical Factor","Points"],
                 cellLoc="left", loc="center")
tbl8.auto_set_font_size(False); tbl8.set_fontsize(8.5); tbl8.scale(2.2, 1.8)
for j in range(2): tbl8[(0,j)].set_facecolor("#2C3E50")
for j in range(2): tbl8[(0,j)].set_text_props(color="white",fontweight="bold")

# Colour risk tier rows
tier_row_idx = len(coef_df) + 3  # +2 for separator and TOTAL
for i, tier in enumerate(TIER_ORDER):
    r = tier_row_idx + i + 1
    try:
        tbl8[(r,0)].set_facecolor(TIER_COLORS[tier])
        tbl8[(r,0)].set_text_props(color="white", fontweight="bold")
        tbl8[(r,1)].set_facecolor(TIER_COLORS[tier])
        tbl8[(r,1)].set_text_props(color="white", fontweight="bold")
    except: pass
ax8.set_title("AMR Risk Scorecard\n(bedside reference)",
              fontweight="bold", fontsize=11, pad=15)

# ── Plot 9: Summary table ──
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis("off")
rows9 = [
    ["Total isolates",       "2,000"],
    ["MDR prevalence",       f"{df['MDR'].mean()*100:.1f}%"],
    ["Score range",          f"{df['risk_score'].min():.0f} – {df['risk_score'].max():.0f}"],
    ["Risk Score AUC",       f"{score_auc:.4f}"],
    ["Full ML AUC",          f"{ml_auc:.4f}"],
    ["ML gain over score",   f"{(ml_auc-score_auc)*100:+.2f} pp"],
    ["Score Brier",          f"{score_brier:.4f}"],
    ["Low Risk MDR %",       f"{df[df['risk_tier']=='Low']['MDR'].mean()*100:.1f}%"],
    ["Moderate Risk MDR %",  f"{df[df['risk_tier']=='Moderate']['MDR'].mean()*100:.1f}%"],
    ["High Risk MDR %",      f"{df[df['risk_tier']=='High']['MDR'].mean()*100:.1f}%"],
    ["Critical Risk MDR %",  f"{df[df['risk_tier']=='Critical']['MDR'].mean()*100:.1f}%"],
    ["ICU MDR %",            f"{df[df['ward']=='ICU']['MDR'].mean()*100:.1f}%"],
]
tbl9 = ax9.table(cellText=rows9, colLabels=["Metric","Value"],
                  cellLoc="center", loc="center")
tbl9.auto_set_font_size(False); tbl9.set_fontsize(9); tbl9.scale(1.6, 1.95)
for j in range(2): tbl9[(0,j)].set_facecolor("#BDC3C7")
for i, (label, _) in enumerate(rows9[7:11], 8):
    tier = TIER_ORDER[i-8]
    tbl9[(i,0)].set_facecolor(TIER_COLORS[tier])
    tbl9[(i,0)].set_text_props(color="white", fontweight="bold")
ax9.set_title("Risk Score Validation Summary",
              fontweight="bold", fontsize=11, pad=20)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig("outputs/clinical_risk_score_dashboard.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("✅ Dashboard saved → outputs/clinical_risk_score_dashboard.png")

# ─────────────────────────────────────────────────────────────
# SECTION 7: SAVE MODEL & SCORER
# ─────────────────────────────────────────────────────────────

save_pkg = {
    "lr_model"     : lr,
    "coef_df"      : coef_df,
    "thresholds"   : thresholds,
    "points_map"   : points_map,
    "score_min"    : int(df["risk_score"].min()),
    "score_max"    : int(df["risk_score"].max()),
    "score_auc"    : score_auc,
    "ml_auc"       : ml_auc,
    "tier_summary" : tier_summary,
}
with open("outputs/risk_score_model.pkl","wb") as f:
    pickle.dump(save_pkg, f)
print("✅ Saved → outputs/risk_score_model.pkl")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "="*62)
print("FINAL SUMMARY — CLINICAL AMR RISK SCORE")
print("="*62)
print(f"\nMDR prevalence : {df['MDR'].mean()*100:.1f}%")
print(f"Score AUC      : {score_auc:.4f}")
print(f"Full ML AUC    : {ml_auc:.4f}")
print(f"ML gain        : {(ml_auc-score_auc)*100:+.2f} pp")
print(f"\nRisk Tier MDR Rates:")
for tier in TIER_ORDER:
    sub = df[df["risk_tier"]==tier]
    print(f"  {tier:10s}: {sub['MDR'].mean()*100:.1f}% MDR (n={len(sub)})")
print(f"\nScorecard:")
for _, row in coef_df.sort_values("Points",ascending=False).iterrows():
    sign = "+" if row["Points"] > 0 else ""
    print(f"  {row['Label']:35s}: {sign}{int(row['Points'])} pts")
print("\n✅ All outputs saved!")
