"""
DEP 803 Group Assignment 1, Part B (Group 04)
Reproduces every number in the Part B report.

How to run: put this file in the same folder as
    hh_survey_group04.csv
    labour_transitions_group04.csv
and run:  python GROUP_04_code.py
(To keep the data somewhere else, change DATA_FOLDER below.)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save the figure to a file instead of opening a window
import matplotlib.pyplot as plt

DATA_FOLDER = os.path.dirname(os.path.abspath(__file__))

hh = pd.read_csv(os.path.join(DATA_FOLDER, "hh_survey_group04.csv"))
n = len(hh)
cons = hh["consumption_ghs"].to_numpy()
y = np.log(cons)

# ---------------------------------------------------------------- B1(a)
print("\n=== B1(a) Sample covariance with the centring matrix ===")
Z = hh[["educ_years", "hh_size", "consumption_ghs"]].to_numpy()
H = np.eye(n) - np.ones((n, n)) / n
S_formula = Z.T @ H @ Z / (n - 1)
S_numpy = np.cov(Z, rowvar=False)
print(np.array2string(S_formula, precision=4))
print("max difference from np.cov:", np.abs(S_formula - S_numpy).max())

# ---------------------------------------------------------------- B1(b)
print("\n=== B1(b) One-pass vs two-pass variance ===")
x = cons + 1e11
var_onepass = (np.sum(x**2) - n * x.mean()**2) / (n - 1)
var_twopass = np.sum((x - x.mean())**2) / (n - 1)
eps = np.finfo(float).eps
print("one-pass :", var_onepass)
print("two-pass :", var_twopass)
print("true var :", np.var(cons, ddof=1))
print("eps:", eps, " sum(x^2):", np.sum(x**2), " eps*sum(x^2):", eps * np.sum(x**2))
print("(n-1)*var:", (n - 1) * np.var(cons, ddof=1))

# ---------------------------------------------------------------- B2(a)
print("\n=== B2(a) Dummy variable trap ===")
region_dummies = pd.get_dummies(hh["region"], prefix="region", dtype=float)
base = hh[["educ_years", "hh_size", "urban"]].to_numpy(dtype=float)
X_full = np.column_stack([np.ones(n), base, region_dummies.to_numpy()])
print("shape:", X_full.shape, " rank:", np.linalg.matrix_rank(X_full))

# ---------------------------------------------------------------- B2(b)
print("\n=== B2(b) Normal equations and QR (region A dropped) ===")
X = np.column_stack([np.ones(n), base, region_dummies.drop(columns="region_A").to_numpy()])
beta_ne = np.linalg.solve(X.T @ X, X.T @ y)
Q, R = np.linalg.qr(X)
beta_qr = np.linalg.solve(R, Q.T @ y)
names = ["intercept", "educ_years", "hh_size", "urban",
         "region_B", "region_C", "region_D", "region_E", "region_F"]
print(pd.DataFrame({"normal equations": beta_ne, "QR": beta_qr}, index=names).to_string(float_format=lambda v: f"{v:.4g}"))
print("max difference:", np.abs(beta_ne - beta_qr).max())
e = y - X @ beta_qr
R2 = 1 - np.sum(e**2) / np.sum((y - y.mean())**2)
print("R-squared:", R2)

# ---------------------------------------------------------------- B2(c)
print("\n=== B2(c) Education coefficient ===")
beta1 = beta_qr[1]
print("beta1:", beta1, " 100*beta1:", 100 * beta1, " exact %:", 100 * (np.exp(beta1) - 1))

# ---------------------------------------------------------------- B2(d)
print("\n=== B2(d) Frisch-Waugh-Lovell ===")
tmp = hh[["region", "educ_years", "hh_size", "urban"]].copy()
tmp["log_cons"] = y
cols = ["log_cons", "educ_years", "hh_size", "urban"]
dm = tmp[cols] - tmp.groupby("region")[cols].transform("mean")
yd = dm["log_cons"].to_numpy()
Xd = dm[["educ_years", "hh_size", "urban"]].to_numpy()
beta_fwl = np.linalg.solve(Xd.T @ Xd, Xd.T @ yd)
print("FWL coefficients:", beta_fwl)
print("max difference from full model:", np.abs(beta_fwl - beta_qr[1:4]).max())

# ---------------------------------------------------------------- B3(a)
print("\n=== B3(a) Condition numbers and VIFs ===")
def standardise(df):
    return ((df - df.mean()) / df.std(ddof=1)).to_numpy()
cols_one = ["educ_years", "hh_size", "urban", "income_ghs"]
cols_both = cols_one + ["income_reported_ghs"]
Z_one, Z_both = standardise(hh[cols_one]), standardise(hh[cols_both])
print("cond (one income):", np.linalg.cond(Z_one))
print("cond (both incomes):", np.linalg.cond(Z_both))
C = np.corrcoef(Z_both, rowvar=False)
vif_both = np.diag(np.linalg.inv(C))
print(pd.Series(vif_both, index=cols_both).map("{:.4g}".format).to_string())

# ---------------------------------------------------------------- B3(b)
print("\n=== B3(b) First half and second half ===")
X5 = np.column_stack([np.ones(n), base,
                      hh["income_ghs"].to_numpy() / 1000,
                      hh["income_reported_ghs"].to_numpy() / 1000])
half = n // 2
def qr_fit(Xm, ym):
    Qm, Rm = np.linalg.qr(Xm)
    return np.linalg.solve(Rm, Qm.T @ ym)
b1, b2 = qr_fit(X5[:half], y[:half]), qr_fit(X5[half:], y[half:])
names5 = ["intercept", "educ_years", "hh_size", "urban", "income_k", "income_reported_k"]
print(pd.DataFrame({"first half": b1, "second half": b2}, index=names5).to_string(float_format=lambda v: f"{v:.4g}"))
print("sum of income coefficients:", b1[4] + b1[5], b2[4] + b2[5])

# ---------------------------------------------------------------- B3(c)
print("\n=== B3(c) Drop income_reported_ghs ===")
X4 = X5[:, :5]
c1, c2 = qr_fit(X4[:half], y[:half]), qr_fit(X4[half:], y[half:])
names4 = ["intercept", "educ_years", "hh_size", "urban", "income_k"]
print(pd.DataFrame({"first half": c1, "second half": c2}, index=names4).to_string(float_format=lambda v: f"{v:.4g}"))

# ---------------------------------------------------------------- B4(a)
print("\n=== B4(a) PCA: variance explained ===")
assets = ["radio", "tv", "fridge", "mobile_phone", "motorbike", "car",
          "electricity", "piped_water", "improved_toilet", "cement_floor"]
Za = standardise(hh[assets])
U, S, Vt = np.linalg.svd(Za, full_matrices=False)
lam = S**2 / (n - 1)
share = lam / lam.sum()
print(pd.Series(share, index=[f"PC{i}" for i in range(1, 11)]).map("{:.4g}".format).to_string())
print("PC1:", share[0], " PC1+PC2:", share[:2].sum())
plt.figure(figsize=(7, 4))
plt.plot(range(1, 11), share, marker="o")
plt.xlabel("Principal component")
plt.ylabel("Share of variance explained")
plt.title("Scree plot of the ten asset indicators (group 04)")
plt.xticks(range(1, 11))
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(DATA_FOLDER, "scree_plot_group04.png"), dpi=150, bbox_inches="tight")
plt.close()
print("scree plot saved as scree_plot_group04.png")

# ---------------------------------------------------------------- B4(b)
print("\n=== B4(b) PC1 loadings and wealth index ===")
v1 = Vt[0]
if v1.sum() < 0:
    v1 = -v1
print(pd.Series(v1, index=assets).sort_values(ascending=False).map("{:.4g}".format).to_string())
wealth_index = Za @ v1
print("first three index values:", wealth_index[:3])

# ---------------------------------------------------------------- B4(c)
print("\n=== B4(c) Correlation with log consumption ===")
print(np.corrcoef(wealth_index, y)[0, 1])

# ---------------------------------------------------------------- B4(d)
print("\n=== B4(d) Targeting errors ===")
cut_w = np.quantile(wealth_index, 0.30)
cut_c = np.quantile(cons, 0.30)
targeted = wealth_index <= cut_w
poor = cons <= cut_c
print("targeted, poor, both:", targeted.sum(), poor.sum(), (targeted & poor).sum())
exclusion_error = (poor & ~targeted).sum() / poor.sum()
inclusion_error = (targeted & ~poor).sum() / targeted.sum()
print("exclusion error:", exclusion_error, " inclusion error:", inclusion_error)

# ---------------------------------------------------------------- B5(a)
print("\n=== B5(a) Transition matrix ===")
tr = pd.read_csv(os.path.join(DATA_FOLDER, "labour_transitions_group04.csv"))
counts = tr[["formal", "informal", "unemployed"]].to_numpy(dtype=float)
P_hat = counts / counts.sum(axis=1, keepdims=True)
print("row totals:", counts.sum(axis=1))
print(P_hat)
print("row sums:", P_hat.sum(axis=1))

# ---------------------------------------------------------------- B5(b)
print("\n=== B5(b) Long-run distribution ===")
vals, vecs = np.linalg.eig(P_hat.T)
print("eigenvalues:", vals)
i = np.argmax(vals.real)
pi_eig = vecs[:, i].real
pi_eig = pi_eig / pi_eig.sum()
print("stationary (eigenvector):", pi_eig)
print("pi @ P_hat:", pi_eig @ P_hat)
pi_pow = np.array([1.0, 0.0, 0.0])
for _ in range(1000):
    pi_pow = pi_pow @ P_hat
print("stationary (power iteration):", pi_pow)
print("max difference:", np.abs(pi_pow - pi_eig).max())
mods = np.sort(np.abs(vals))[::-1]
print("moduli:", mods)
print("second-largest modulus:", mods[1])
print("periods to get within 1e-8:", np.log(1e-8) / np.log(mods[1]))
