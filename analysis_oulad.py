#!/usr/bin/env python3
"""OULAD: youth (0-35) retention — full quantitative analysis. Reproducible."""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE = "/Users/daniilpikulev/Documents/ИССЛЕД 2"
DATA = os.path.join(BASE, "data"); FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)
ALPHA = 0.05; SEED = 42

def J(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return [J(v) for v in o.tolist()]
    if isinstance(o, dict): return {str(k): J(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [J(v) for v in o]
    return o

def prop_diff_ci(n1, d1, n2, d2, z=1.96):
    p1, p2 = n1/d1, n2/d2; diff = p1-p2
    se = np.sqrt(p1*(1-p1)/d1 + p2*(1-p2)/d2)
    return diff, diff-z*se, diff+z*se

def cohens_h(p1, p2):
    return float(2*np.arcsin(np.sqrt(p1)) - 2*np.arcsin(np.sqrt(p2)))

def cramers_v(chi2, n, r, c): return float(np.sqrt(chi2/(n*(min(r,c)-1))))

# ---------- 1. LOAD ----------
info = pd.read_csv(f"{DATA}/studentInfo.csv")
sA = pd.read_csv(f"{DATA}/studentAssessment.csv")
ass = pd.read_csv(f"{DATA}/assessments.csv")
courses = pd.read_csv(f"{DATA}/courses.csv")
reg = pd.read_csv(f"{DATA}/studentRegistration.csv")
vle = pd.read_csv(f"{DATA}/studentVle.csv")
vlemeta = pd.read_csv(f"{DATA}/vle.csv")
fl = {"studentInfo": info, "studentAssessment": sA, "assessments": ass,
      "courses": courses, "studentRegistration": reg, "studentVle": vle, "vle": vlemeta}

dq = {"n_rows": {k: int(len(v)) for k, v in fl.items()},
      "columns": {k: list(v.columns) for k, v in fl.items()},
      "missing": {k: J(v.isna().sum()[v.isna().sum() > 0].to_dict()) for k, v in fl.items()},
      "missing_pct": {k: J((v.isna().mean()*100).round(2)[v.isna().sum() > 0].to_dict()) for k, v in fl.items()},
      "exact_duplicates": {}}
for k, v in fl.items():
    dq["exact_duplicates"][k] = int(v.duplicated().sum())
info_key_dup = int(info.duplicated(subset=["code_module","code_presentation","id_student"]).sum())
dq["studentInfo_key_duplicates"] = info_key_dup
# drop exact dup rows
for k in fl:
    fl[k] = fl[k].drop_duplicates()
info, sA, ass, courses, reg, vle, vlemeta = (fl["studentInfo"], fl["studentAssessment"], fl["assessments"],
      fl["courses"], fl["studentRegistration"], fl["studentVle"], fl["vle"])
dq["n_rows_after_dedup"] = {k: int(len(v)) for k, v in fl.items()}

dq["final_result_dist"] = J(info["final_result"].value_counts().to_dict())
dq["final_result_dist_pct"] = J((info["final_result"].value_counts(normalize=True)*100).round(2).to_dict())
dq["age_band_dist"] = J(info["age_band"].value_counts().to_dict())
dq["vle_date_min"] = int(vle["date"].min()); dq["vle_date_max"] = int(vle["date"].max())
dq["vle_negative_date_rows"] = int((vle["date"] < 0).sum())
dq["vle_negative_date_pct"] = round(float((vle["date"] < 0).mean()*100), 3)
sc = vle["sum_click"]
dq["sum_click_describe"] = J(sc.describe(percentiles=[.5,.9,.99,.999]).round(3).to_dict())
dq["sum_click_min"] = float(sc.min()); dq["sum_click_zeros"] = int((sc == 0).sum())
dq["info_dtypes"] = {c: str(t) for c, t in info.dtypes.items()}
dq["notes"] = ("Exact-duplicate rows dropped per counts above; no other rows removed. "
               "VLE negative dates (=pre-start activity) kept and reported separately. "
               "No winsorization; log scale used for plots.")

# ---------- 2. EDA ----------
info["youth"] = (info["age_band"] == "0-35").astype(int)
info["withdrawn"] = (info["final_result"] == "Withdrawn").astype(int)
info["passed"] = info["final_result"].isin(["Pass","Distinction"]).astype(int)
N = len(info)

def wr_table(col, df=info):
    t = df.groupby(col)["withdrawn"].agg(["sum","count"]); t["rate"] = t["sum"]/t["count"]
    return t.sort_values("rate", ascending=False)

overall = {"n": N, "withdrawn_n": int(info["withdrawn"].sum()),
           "withdrawn_rate": float(info["withdrawn"].mean())}
by_age = wr_table("age_band")
youth_n = int((info["youth"]==1).sum()); rest_n = N - youth_n
youth_w = int(info.loc[info["youth"]==1,"withdrawn"].sum()); rest_w = int(info.loc[info["youth"]==0,"withdrawn"].sum())
mid_n = int((info["age_band"]=="35-55").sum()); old_n = int((info["age_band"]=="55<=").sum())
mid_w = int(info.loc[info["age_band"]=="35-55","withdrawn"].sum()); old_w = int(info.loc[info["age_band"]=="55<=","withdrawn"].sum())
by_module = wr_table("code_module"); by_pres = wr_table("code_presentation")
by_gender = wr_table("gender"); by_edu = wr_table("highest_education")
by_imd = wr_table("imd_band"); by_credits = wr_table("studied_credits"); by_prev = wr_table("num_of_prev_attempts")

# engagement per student (vectorized on a single packed integer key)
KEYS = ["code_module","code_presentation","id_student"]
_mod = {v: i for i, v in enumerate(sorted(set(vle["code_module"]) | set(info["code_module"])))}
_pres = {v: i for i, v in enumerate(sorted(set(vle["code_presentation"]) | set(info["code_presentation"])))}
def skey(df):
    c1 = df["code_module"].map(_mod).to_numpy(np.int64)
    c2 = df["code_presentation"].map(_pres).to_numpy(np.int64)
    c3 = df["id_student"].to_numpy(np.int64)
    return (c1 << 40) | (c2 << 32) | c3
vle["skey"] = skey(vle)
eng = info.copy(); eng["skey"] = skey(info)

tot = vle.groupby("skey")["sum_click"].sum().rename("total_clicks")
ad = vle.groupby("skey")["date"].nunique().rename("active_days")
early14 = vle[vle["date"].between(0,13)].groupby("skey")["sum_click"].sum().rename("early14")
early30 = vle[vle["date"].between(0,29)].groupby("skey")["sum_click"].sum().rename("early30")
ad30 = vle[vle["date"].between(0,29)].groupby("skey")["date"].nunique().rename("ad30")
eng = eng.join(pd.concat([tot, ad, early14, early30, ad30], axis=1), on="skey")
eng[["total_clicks","active_days","early14","early30","ad30"]] = eng[["total_clicks","active_days","early14","early30","ad30"]].fillna(0)

def gs(s):
    return {"mean": float(s.mean()), "median": float(s.median()), "std": float(s.std()),
            "q25": float(s.quantile(.25)), "q75": float(s.quantile(.75)), "n": int(len(s))}
engagement = {"total_clicks_overall": gs(eng["total_clicks"]),
    "total_clicks_youth": gs(eng.loc[eng["youth"]==1,"total_clicks"]),
    "total_clicks_rest": gs(eng.loc[eng["youth"]==0,"total_clicks"]),
    "active_days_overall": gs(eng["active_days"]),
    "active_days_youth": gs(eng.loc[eng["youth"]==1,"active_days"]),
    "active_days_rest": gs(eng.loc[eng["youth"]==0,"active_days"]),
    "early14_youth": gs(eng.loc[eng["youth"]==1,"early14"]),
    "early14_rest": gs(eng.loc[eng["youth"]==0,"early14"]),
    "early30_youth": gs(eng.loc[eng["youth"]==1,"early30"]),
    "early30_rest": gs(eng.loc[eng["youth"]==0,"early30"]),
    "by_final_result": {k: gs(g["total_clicks"]) for k, g in eng.groupby("final_result")}}
zero_click = {"n_zero_click_students": int((eng["total_clicks"]==0).sum()),
              "pct_zero": round(float((eng["total_clicks"]==0).mean()*100),2)}

# funnel: registered -> VLE-active -> TMA1 submitted -> Pass/Distinction
tma = ass[ass["assessment_type"]=="TMA"]
tma1_ids = tma.sort_values("date").groupby(["code_module","code_presentation"]).first().reset_index()[["code_module","code_presentation","id_assessment"]]
sub = sA[sA["score"].notna()].merge(ass[["id_assessment","code_module","code_presentation"]], on="id_assessment", how="left")
tma1sub = sub.merge(tma1_ids, on=["code_module","code_presentation","id_assessment"], how="inner")
tma1_keys = set(skey(tma1sub))
eng["tma1"] = eng["skey"].isin(tma1_keys)
eng["vle_active"] = (eng["total_clicks"] > 0).astype(int)
funnel = {"registered": N, "vle_active": int(eng["vle_active"].sum()),
          "tma1_submitted": int(eng["tma1"].sum()), "pass_distinction": int(eng["passed"].sum())}
funnel_rates = {"vle_active_of_registered": funnel["vle_active"]/N,
    "tma1_of_registered": funnel["tma1_submitted"]/N, "tma1_of_active": funnel["tma1_submitted"]/max(funnel["vle_active"],1),
    "pass_of_registered": funnel["pass_distinction"]/N,
    "pass_of_tma1": funnel["pass_distinction"]/max(funnel["tma1_submitted"],1)}
funnel_y = {"youth_registered": youth_n, "youth_vle_active": int(eng.loc[eng["youth"]==1,"vle_active"].sum()),
    "youth_tma1": int(eng.loc[eng["youth"]==1,"tma1"].sum()), "youth_pass": int(eng.loc[eng["youth"]==1,"passed"].sum())}
funnel_r = {"rest_registered": rest_n, "rest_vle_active": int(eng.loc[eng["youth"]==0,"vle_active"].sum()),
    "rest_tma1": int(eng.loc[eng["youth"]==0,"tma1"].sum()), "rest_pass": int(eng.loc[eng["youth"]==0,"passed"].sum())}

# weekly activity (% active students per week, date>=0)
vv = vle[vle["date"]>=0].copy(); vv["week"] = (vv["date"]//7).astype(int)
youth_keys = set(eng.loc[eng["youth"]==1,"skey"])
vv["is_youth"] = vv["skey"].isin(youth_keys)
wk_all = vv.groupby("week")["skey"].nunique()/N*100
wk_y = vv[vv["is_youth"]].groupby("week")["skey"].nunique()/max(youth_n,1)*100
wk_r = vv[~vv["is_youth"]].groupby("week")["skey"].nunique()/max(rest_n,1)*100

# ---------- 3. SEGMENT youth vs rest ----------
seg = {"youth_n": youth_n, "rest_n": rest_n,
    "youth_withdrawn_rate": youth_w/youth_n, "rest_withdrawn_rate": rest_w/rest_n,
    "youth_final_result": J(info[info["youth"]==1]["final_result"].value_counts(normalize=True).round(4).to_dict()),
    "rest_final_result": J(info[info["youth"]==0]["final_result"].value_counts(normalize=True).round(4).to_dict()),
    "youth_pass_rate": float(info.loc[info["youth"]==1,"passed"].mean()),
    "rest_pass_rate": float(info.loc[info["youth"]==0,"passed"].mean())}

# ---------- 4. STATS TESTS ----------
ct = pd.crosstab(info["youth"], info["withdrawn"])
chi2, p_chi, dof, _ = sps.chi2_contingency(ct)
V = cramers_v(chi2, N, *ct.shape)
d_w, lo_w, hi_w = prop_diff_ci(youth_w, youth_n, rest_w, rest_n)
test_a = {"name": "chi-square Withdrawn(yes/no) x youth(yes/no)",
    "H0": "withdrawal independent of youth status", "H1": "withdrawal associated with youth status",
    "contingency": J(ct.to_dict()), "chi2": float(chi2), "dof": int(dof), "p": float(p_chi),
    "cramers_V": V, "diff_youth_minus_rest": float(d_w), "CI95": [float(lo_w), float(hi_w)],
    "cohens_h": cohens_h(youth_w/youth_n, rest_w/rest_n), "alpha": ALPHA, "reject_H0": bool(p_chi < ALPHA)}

comp = eng.loc[eng["final_result"].isin(["Pass","Distinction"]),"total_clicks"]
wd = eng.loc[eng["final_result"]=="Withdrawn","total_clicks"]
sh_c = sps.shapiro(comp.sample(min(5000,len(comp)), random_state=SEED))
sh_w = sps.shapiro(wd.sample(min(5000,len(wd)), random_state=SEED))
U, p_mw = sps.mannwhitneyu(comp, wd, alternative="two-sided")
n1, n2 = len(comp), len(wd)
r_rb = float(1 - 2*U/(n1*n2))
test_b = {"name": "Mann-Whitney U total_clicks completed(Pass/Distinction) vs Withdrawn",
    "choice_justification": f"clicks strongly right-skewed (skew={float(comp.skew()):.2f}/{float(wd.skew()):.2f}); Shapiro on n=5000 subsamples p_c={sh_c.pvalue:.3g}, p_w={sh_w.pvalue:.3g} -> normality rejected, rank test used",
    "H0": "equal distributions of total clicks", "H1": "distributions differ",
    "n_completed": n1, "n_withdrawn": n2, "median_completed": float(comp.median()), "median_withdrawn": float(wd.median()),
    "U": float(U), "p": float(p_mw), "rank_biserial_r": r_rb, "alpha": ALPHA, "reject_H0": bool(p_mw < ALPHA)}

yp, yn = int(info.loc[info["youth"]==1,"passed"].sum()), youth_n
op, on = int(info.loc[info["youth"]==0,"passed"].sum()), rest_n
p1, p2 = yp/yn, op/on; pp = (yp+op)/(yn+on)
se_pool = np.sqrt(pp*(1-pp)*(1/yn+1/on)); z = (p1-p2)/se_pool
p_z = float(2*(1-sps.norm.cdf(abs(z))))
d_p, lo_p, hi_p = prop_diff_ci(yp, yn, op, on)
test_c = {"name": "two-proportion z-test Pass/Distinction rate youth vs rest",
    "H0": "equal pass rates", "H1": "pass rates differ",
    "youth_pass": [yp, yn, p1], "rest_pass": [op, on, p2],
    "z": float(z), "p": p_z, "diff": float(d_p), "CI95": [float(lo_p), float(hi_p)],
    "cohens_h": cohens_h(p1, p2), "alpha": ALPHA, "reject_H0": bool(p_z < ALPHA)}
ct_e = pd.crosstab(info["highest_education"], (info["final_result"].isin(["Fail","Withdrawn"])).astype(int))
chi2e, pe, dofe, _ = sps.chi2_contingency(ct_e)
test_c_edu = {"name": "chi-square Fail+Withdrawn vs Pass+Distinction x highest_education",
    "chi2": float(chi2e), "dof": int(dofe), "p": float(pe),
    "cramers_V": cramers_v(chi2e, N, *ct_e.shape), "reject_H0": bool(pe < ALPHA)}

# ---------- 5. LOGISTIC REGRESSION P(Withdrawn) ----------
reg["date_registration"] = pd.to_numeric(reg["date_registration"], errors="coerce")
reg["date_unregistration"] = pd.to_numeric(reg["date_unregistration"], errors="coerce")
reg1 = reg.sort_values("date_unregistration").drop_duplicates(["code_module","code_presentation","id_student"])
lr = info[["code_module","code_presentation","id_student","gender","highest_education","imd_band",
           "studied_credits","num_of_prev_attempts","youth","withdrawn"]].merge(
    eng[["code_module","code_presentation","id_student","early30","ad30"]], on=["code_module","code_presentation","id_student"], how="left").merge(
    reg1[["code_module","code_presentation","id_student","date_registration"]], on=["code_module","code_presentation","id_student"], how="left")
lr[["early30","ad30"]] = lr[["early30","ad30"]].fillna(0)
reg_na = int(lr["date_registration"].isna().sum())
lr["date_registration"] = lr["date_registration"].fillna(lr["date_registration"].median())
lr["log_early30"] = np.log1p(lr["early30"])
num = ["studied_credits","num_of_prev_attempts","log_early30","ad30","date_registration"]
X_num = lr[num].copy(); Xs = StandardScaler().fit_transform(X_num)
Xcat = pd.get_dummies(lr[["gender","highest_education","imd_band"]], drop_first=True, dtype=float)
X = np.hstack([lr[["youth"]].values, Xs, Xcat.values])
fnames = ["youth_0-35"] + [n+"(std)" for n in num] + list(Xcat.columns)
y = lr["withdrawn"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
m = LogisticRegression(max_iter=5000); m.fit(Xtr, ytr)
auc_tr = float(roc_auc_score(ytr, m.predict_proba(Xtr)[:,1])); auc_te = float(roc_auc_score(yte, m.predict_proba(Xte)[:,1]))
coefs = [{"feature": f, "coef": float(c), "odds_ratio": float(np.exp(c))} for f, c in zip(fnames, m.coef_[0])]
coefs_sorted = sorted(coefs, key=lambda d: -abs(d["coef"]))
logreg = {"n": int(len(lr)), "withdrawn_rate": float(y.mean()), "date_registration_NA_filled_median": reg_na,
    "features": fnames, "numeric_standardized": True, "note": "OR for standardized numerics = per +1 SD; youth/gender/education/IMD dummies unscaled",
    "AUC_train": auc_tr, "AUC_test": auc_te, "intercept": float(m.intercept_[0]),
    "coefficients": coefs_sorted,
    "conclusion": "Largest-|coef| features (see sorted list) are most strongly associated with withdrawal; sign>0 raises odds."}

# ---------- 6. FIGURES ----------
plt.rcParams.update({"figure.dpi": 150})
figs = []
def save(nm):
    p = f"{FIG}/{nm}"; plt.tight_layout(); plt.savefig(p); plt.close(); figs.append(nm); return p

plt.figure(figsize=(7,4.5)); o = by_age["rate"].loc[["0-35","35-55","55<="]]
plt.bar(o.index, o.values); plt.ylabel("Withdrawn rate"); plt.title("Withdrawn rate by age_band (N=32,593)")
for i,v in enumerate(o.values): plt.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=9)
save("01_withdrawn_rate_ageband.png")

plt.figure(figsize=(9,4.5)); o = by_module["rate"]
plt.bar(o.index, o.values); plt.ylabel("Withdrawn rate"); plt.title("Withdrawn rate by module")
for i,v in enumerate(o.values): plt.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=8)
save("02_withdrawn_rate_module.png")

plt.figure(figsize=(8,4.5))
x = eng["total_clicks"].replace(0, np.nan).dropna()
plt.hist(np.log10(x), bins=60, alpha=.7, label="all (zeros excl.)")
for lab, f in [("Withdrawn", eng[eng["withdrawn"]==1]["total_clicks"]), ("Pass/Distinction", eng[eng["passed"]==1]["total_clicks"])]:
    plt.hist(np.log10(f.replace(0,np.nan).dropna()), bins=60, alpha=.45, label=lab)
plt.xlabel("log10(total clicks)"); plt.ylabel("students"); plt.title("Total clicks per student (log scale)"); plt.legend()
save("03_total_clicks_log.png")

plt.figure(figsize=(9,4.5))
plt.plot(wk_all.index, wk_all.values, label="all"); plt.plot(wk_y.index, wk_y.values, label="0-35"); plt.plot(wk_r.index, wk_r.values, label="35+")
plt.xlabel("course week (date>=0)"); plt.ylabel("% students active"); plt.title("Weekly activity (% active students)"); plt.legend()
save("04_weekly_activity.png")

plt.figure(figsize=(8,4.5))
labels = ["Registered","VLE-active","TMA1 submitted","Pass/Distinction"]
vals = [funnel["registered"], funnel["vle_active"], funnel["tma1_submitted"], funnel["pass_distinction"]]
plt.bar(labels, vals); plt.title("Funnel: registered to Pass/Distinction")
for i,v in enumerate(vals): plt.text(i, v+N*0.01, str(v), ha="center", fontsize=9)
plt.xticks(rotation=10); save("05_funnel.png")

plt.figure(figsize=(9,6))
top = coefs_sorted[:15]; yy = [d["feature"] for d in top][::-1]; vv2 = [d["coef"] for d in top][::-1]
plt.barh(yy, vv2); plt.axvline(0, color="k", lw=.8); plt.xlabel("coef (log-odds)"); plt.title("Top-15 logistic regression coefficients: P(Withdrawn)")
save("06_logreg_coefs.png")

plt.figure(figsize=(7,4.5)); o2 = wr_table("code_presentation").sort_index()["rate"]
plt.bar(o2.index, o2.values); plt.ylabel("Withdrawn rate"); plt.title("Withdrawn rate by presentation")
for i,v in enumerate(o2.values): plt.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=9)
save("07_withdrawn_rate_presentation.png")

# ---------- 7. JSON ----------
summary = {"dataset": "OULAD (Open University Learning Analytics Dataset, 2013-2014, UK Open University)",
 "data_quality": dq, "N_students_rows": N, "overall": overall,
 "withdrawn_by_age_band": J(by_age.assign(rate=by_age["rate"].round(4)).to_dict("index")),
 "youth_vs_mid_vs_old": {"0-35": [youth_w, youth_n, youth_w/youth_n], "35-55": [mid_w, mid_n, mid_w/mid_n], "55+": [old_w, old_n, old_w/old_n]},
 "by_module": J(by_module.assign(rate=by_module["rate"].round(4)).to_dict("index")),
 "by_presentation": J(by_pres.assign(rate=by_pres["rate"].round(4)).to_dict("index")),
 "by_gender": J(by_gender.assign(rate=by_gender["rate"].round(4)).to_dict("index")),
 "by_education": J(by_edu.assign(rate=by_edu["rate"].round(4)).to_dict("index")),
 "by_imd": J(by_imd.assign(rate=by_imd["rate"].round(4)).to_dict("index")),
 "by_credits": J(by_credits.assign(rate=by_credits["rate"].round(4)).to_dict("index")),
 "by_prev_attempts": J(by_prev.assign(rate=by_prev["rate"].round(4)).to_dict("index")),
 "engagement": engagement, "zero_click": zero_click,
 "funnel": funnel, "funnel_rates": J(funnel_rates), "funnel_youth": funnel_y, "funnel_rest": funnel_r,
 "segmentation": seg, "test_a_chi2_youth_withdrawn": test_a, "test_b_mannwhitney_clicks": test_b,
 "test_c_ztest_passrate": test_c, "test_extra_edu_chi2": test_c_edu,
 "logreg_withdrawn": logreg, "figures": figs, "alpha": ALPHA, "seed": SEED}
with open(f"{BASE}/results_summary.json","w") as f: json.dump(J(summary), f, indent=2, ensure_ascii=False)
print("OK N=",N," overall_withdrawn=",round(overall["withdrawn_rate"],4))
print("youth_wr=",round(youth_w/youth_n,4)," rest_wr=",round(rest_w/rest_n,4))
print("chi2 p=",p_chi," MWU p=",p_mw," z p=",p_z," AUC_test=",round(auc_te,4))
