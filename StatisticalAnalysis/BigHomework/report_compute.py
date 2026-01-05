from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import sys

import mpmath as mp
import numpy as np


STEPS = np.array(
    [
        812,
        1247,
        1793,
        963,
        2218,
        3076,
        15684,
        2684,
        4189,
        3827,
        5126,
        4573,
        2439,
        5297,
        6148,
        4861,
        3472,
        5924,
        18946,
        5638,
        6152,
        6217,
        4284,
        2796,
        5071,
        3928,
        3346,
        5763,
        4479,
        6018,
        6386,
        7193,
        6827,
        17891,
        8234,
        7049,
        9037,
        8472,
        7816,
        9483,
        10137,
        10984,
        9826,
        11973,
        13482,
        12461,
        13927,
        20861,
        15863,
        14472,
        16834,
        15429,
        5094,
        16372,
        5871,
        17418,
        5587,
        18394,
        6097,
        6924,
        21836,
        20317,
        15924,
        9874,
        7218,
        4837,
        2986,
        5183,
        6097,
        8726,
        9184,
        10493,
        6479,
        4026,
        2284,
        14962,
        17384,
        827,
        3379,
        6924,
        8793,
        4686,
        12473,
        4327,
        5094,
        14918,
        7634,
        9781,
        15500,
        19573,
    ],
    dtype=float,
)

SLEEP = np.array(
    [
        2.3,
        6.4,
        4.1,
        7.9,
        5.3,
        6.7,
        8.6,
        7.2,
        6.1,
        8.0,
        6.6,
        7.8,
        4.0,
        7.1,
        6.7,
        8.7,
        5.1,
        7.5,
        9.6,
        6.3,
        7.3,
        6.2,
        8.9,
        4.8,
        7.8,
        6.5,
        8.4,
        7.1,
        5.7,
        7.5,
        6.1,
        8.0,
        7.3,
        8.8,
        8.8,
        5.9,
        7.7,
        9.1,
        6.2,
        7.9,
        6.6,
        8.5,
        6.0,
        7.0,
        9.7,
        6.4,
        8.6,
        9.8,
        10.0,
        6.9,
        9.5,
        7.1,
        6.3,
        6.3,
        8.0,
        7.2,
        7.2,
        6.7,
        6.7,
        6.6,
        13.2,
        6.1,
        8.3,
        6.8,
        7.6,
        8.5,
        5.0,
        7.3,
        6.7,
        8.7,
        6.1,
        9.0,
        7.5,
        5.9,
        6.8,
        6.2,
        7.7,
        3.1,
        9.1,
        6.6,
        7.4,
        8.2,
        8.7,
        8.0,
        6.3,
        7.3,
        7.9,
        5.8,
        8.4,
        9.9,
    ],
    dtype=float,
)

START_DATE = date(2025, 9, 16)


@dataclass(frozen=True)
class Desc:
    n: int
    mean: float
    sd: float
    median: float
    q1: float
    q3: float
    iqr: float
    min: float
    max: float


def describe(x: np.ndarray) -> Desc:
    x = np.asarray(x, dtype=float)
    q1 = float(np.percentile(x, 25, method="linear"))
    q3 = float(np.percentile(x, 75, method="linear"))
    return Desc(
        n=int(x.size),
        mean=float(x.mean()),
        sd=float(x.std(ddof=1)),
        median=float(np.median(x)),
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        min=float(x.min()),
        max=float(x.max()),
    )


def t_cdf(t: float, df: float) -> mp.mpf:
    t = mp.mpf(t)
    df = mp.mpf(df)
    x = df / (df + t * t)
    ib = mp.betainc(df / 2, mp.mpf("0.5"), 0, x, regularized=True)
    if t >= 0:
        return 1 - mp.mpf("0.5") * ib
    return mp.mpf("0.5") * ib


def t_p_two_sided(t: float, df: float) -> float:
    c = t_cdf(t, df)
    return float(2 * min(c, 1 - c))


def norm_ppf(p: float) -> float:
    p = float(p)
    return float(mp.sqrt(2) * mp.erfinv(2 * mp.mpf(p) - 1))


def chi2_cdf(x: float, df: float) -> mp.mpf:
    x = mp.mpf(x)
    df = mp.mpf(df)
    return mp.gammainc(df / 2, 0, x / 2, regularized=True)


def chi2_ppf(p: float, df: float) -> float:
    p = mp.mpf(p)
    df_m = mp.mpf(df)
    if p <= 0:
        return 0.0
    if p >= 1:
        return float("inf")
    z = norm_ppf(float(p))
    guess = df_m * (1 - 2 / (9 * df_m) + z * mp.sqrt(2 / (9 * df_m))) ** 3
    guess = max(guess, mp.mpf("1e-12"))
    f = lambda x: chi2_cdf(x, df_m) - p
    try:
        root = mp.findroot(f, guess)
    except Exception:
        root = mp.findroot(f, (guess * mp.mpf("0.5"), guess * mp.mpf("1.5")))
    return float(root)


def t_ppf(p: float, df: float) -> float:
    p = mp.mpf(p)
    df_m = mp.mpf(df)
    if p <= 0:
        return float("-inf")
    if p >= 1:
        return float("inf")
    if p == mp.mpf("0.5"):
        return 0.0

    sign = 1
    target = p
    if p < mp.mpf("0.5"):
        sign = -1
        target = 1 - p

    z = mp.mpf(norm_ppf(float(target)))
    guess = z * mp.sqrt((df_m) / (df_m - 2)) if df_m > 2 else z

    f = lambda x: t_cdf(x, df_m) - target
    try:
        root = mp.findroot(f, guess)
    except Exception:
        root = mp.findroot(f, (guess * mp.mpf("0.5"), guess * mp.mpf("1.5")))
    return float(sign * root)


def welch_ttest(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = x.size, y.size
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se2 = vx / nx + vy / ny
    t = float((mx - my) / np.sqrt(se2))
    df = float(se2**2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)))
    p = t_p_two_sided(t, df)
    return t, df, p


def rankdata_avg(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def mannwhitney_u(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n1, n2 = len(x), len(y)
    allv = np.concatenate([x, y])
    ranks = rankdata_avg(allv)
    r1 = float(ranks[:n1].sum())
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    _, counts = np.unique(allv, return_counts=True)
    tie_term = float((counts**3 - counts).sum())
    N = n1 + n2
    var_u = n1 * n2 / 12 * (N + 1 - tie_term / (N * (N - 1)))
    mean_u = n1 * n2 / 2
    z = float((u - mean_u + 0.5) / np.sqrt(var_u))
    p = float(2 * (0.5 * mp.erfc(abs(z) / mp.sqrt(2))))
    return float(u1), float(u2), z, p


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x - x.mean()
    y = y - y.mean()
    return float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata_avg(np.asarray(x))
    ry = rankdata_avg(np.asarray(y))
    return pearsonr(rx, ry)


def pearsonr_ci_95(r: float, n: int) -> tuple[float, float]:
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    z_lo = z - 1.959963984540054 * se
    z_hi = z + 1.959963984540054 * se
    r_lo = (np.exp(2 * z_lo) - 1) / (np.exp(2 * z_lo) + 1)
    r_hi = (np.exp(2 * z_hi) - 1) / (np.exp(2 * z_hi) + 1)
    return float(r_lo), float(r_hi)


def corr_test_zero(r: float, n: int) -> tuple[float, float, float]:
    df = n - 2
    t = float(r * np.sqrt(df / (1 - r * r)))
    p = t_p_two_sided(t, df)
    return t, float(df), float(p)


def normal_mean_ci_95(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    n = int(x.size)
    mean = float(x.mean())
    s = float(x.std(ddof=1))
    tcrit = t_ppf(0.975, n - 1)
    half = float(tcrit * s / np.sqrt(n))
    return mean - half, mean + half


def normal_sigma2_ci_95(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    n = int(x.size)
    s2 = float(x.var(ddof=1))
    df = n - 1
    chi2_lo = chi2_ppf(0.975, df)
    chi2_hi = chi2_ppf(0.025, df)
    return (df * s2) / chi2_lo, (df * s2) / chi2_hi


def lognormal_mle(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    lx = np.log(x)
    mu = float(lx.mean())
    sigma = float(lx.std(ddof=0))
    return mu, sigma


def lognormal_mu_ci_95(mu: float, sigma: float, n: int) -> tuple[float, float]:
    half = 1.959963984540054 * sigma / np.sqrt(n)
    return mu - half, mu + half


def lognormal_sigma2_ci_95(sigma: float, n: int) -> tuple[float, float]:
    df = n
    s2 = float(sigma * sigma)
    chi2_lo = chi2_ppf(0.975, df)
    chi2_hi = chi2_ppf(0.025, df)
    return (df * s2) / chi2_lo, (df * s2) / chi2_hi


def ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    n, k = x.shape
    sigma2 = float((resid @ resid) / (n - k))
    xtx_inv = np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    tstats = beta / se
    pvals = np.array([t_p_two_sided(float(t), n - k) for t in tstats], dtype=float)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid**2).sum())
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)
    df1 = k - 1
    df2 = n - k
    F = ((ss_tot - ss_res) / df1) / (ss_res / df2)
    xx = (df1 * F) / (df1 * F + df2)
    F_cdf = mp.betainc(df1 / 2, df2 / 2, 0, xx, regularized=True)
    F_p = float(1 - F_cdf)
    return beta, se, pvals, float(r2), float(adj_r2), float(F), float(F_p)


def build_dates_and_weekend(n: int) -> tuple[np.ndarray, np.ndarray]:
    dates = np.array([START_DATE + timedelta(days=i) for i in range(n)])
    weekend = np.array([1 if d.weekday() >= 5 else 0 for d in dates], dtype=float)
    return dates, weekend


def simple_regression_line(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    b = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    a = float(y.mean() - b * x.mean())
    return a, b


def print_text_report() -> None:
    dates, weekend = build_dates_and_weekend(len(STEPS))

    steps_desc = describe(STEPS)
    sleep_desc = describe(SLEEP)

    min_idx = int(np.argmin(STEPS))
    max_idx = int(np.argmax(STEPS))
    sleep_min_idx = int(np.argmin(SLEEP))
    sleep_max_idx = int(np.argmax(SLEEP))

    weekday_steps = STEPS[weekend == 0]
    weekend_steps = STEPS[weekend == 1]
    wd_desc = describe(weekday_steps)
    we_desc = describe(weekend_steps)

    t, df, p = welch_ttest(weekend_steps, weekday_steps)
    u1, u2, z, mw_p = mannwhitney_u(weekend_steps, weekday_steps)

    pear = pearsonr(STEPS, SLEEP)
    spear = spearmanr(STEPS, SLEEP)
    pear_lo, pear_hi = pearsonr_ci_95(pear, len(STEPS))
    pear_t, pear_df, pear_p = corr_test_zero(pear, len(STEPS))

    X1 = np.column_stack([np.ones(len(STEPS)), SLEEP])
    beta1, se1, pvals1, r2_1, adj_r2_1, F1, F1_p = ols(STEPS, X1)

    X2 = np.column_stack([np.ones(len(STEPS)), SLEEP, weekend])
    beta2, se2, pvals2, r2_2, adj_r2_2, F2, F2_p = ols(STEPS, X2)

    a, b = simple_regression_line(STEPS, SLEEP)

    sleep_mean_ci = normal_mean_ci_95(SLEEP)
    sleep_sigma2_ci = normal_sigma2_ci_95(SLEEP)
    logn_mu, logn_sigma = lognormal_mle(STEPS)
    logn_mu_ci = lognormal_mu_ci_95(logn_mu, logn_sigma, len(STEPS))
    logn_sigma2_ci = lognormal_sigma2_ci_95(logn_sigma, len(STEPS))
    logn_mean = float(np.exp(logn_mu + 0.5 * logn_sigma * logn_sigma))

    print(f"DATE_START {dates[0].isoformat()} DATE_END {dates[-1].isoformat()}")
    print(f"STEPS_DESC {steps_desc}")
    print(f"SLEEP_DESC {sleep_desc}")
    print(f"STEPS_MIN {steps_desc.min} {dates[min_idx].isoformat()} day {min_idx+1}")
    print(f"STEPS_MAX {steps_desc.max} {dates[max_idx].isoformat()} day {max_idx+1}")
    print(f"SLEEP_MIN {sleep_desc.min} {dates[sleep_min_idx].isoformat()} day {sleep_min_idx+1}")
    print(f"SLEEP_MAX {sleep_desc.max} {dates[sleep_max_idx].isoformat()} day {sleep_max_idx+1}")
    print(f"WEEKDAY_DESC {wd_desc}")
    print(f"WEEKEND_DESC {we_desc}")
    print(f"WELCH t={t:.6f} df={df:.6f} p={p:.12g}")
    print(f"MANN_WHITNEY u1={u1:.6f} u2={u2:.6f} z={z:.6f} p={mw_p:.12g}")
    print(f"CORR pearson={pear:.6f} pearson_ci95=({pear_lo:.6f},{pear_hi:.6f}) spearman={spear:.6f}")
    print(f"CORR_TEST pearson_t={pear_t:.6f} df={pear_df:.0f} p={pear_p:.12g}")
    print(
        f"REG1 beta={beta1} se={se1} p={pvals1} r2={r2_1:.6f} adj_r2={adj_r2_1:.6f} F={F1:.6f} F_p={F1_p:.12g}"
    )
    print(
        f"REG2 beta={beta2} se={se2} p={pvals2} r2={r2_2:.6f} adj_r2={adj_r2_2:.6f} F={F2:.6f} F_p={F2_p:.12g}"
    )
    print(f"SCATTER_LINE a={a:.6f} b={b:.6f}")
    print(f"SLEEP_NORM mean_ci95=({sleep_mean_ci[0]:.6f},{sleep_mean_ci[1]:.6f}) sigma2_ci95=({sleep_sigma2_ci[0]:.6f},{sleep_sigma2_ci[1]:.6f})")
    print(f"STEPS_LOGN mu={logn_mu:.6f} sigma={logn_sigma:.6f} implied_mean={logn_mean:.6f}")
    print(f"STEPS_LOGN_CI mu_ci95=({logn_mu_ci[0]:.6f},{logn_mu_ci[1]:.6f}) sigma2_ci95=({logn_sigma2_ci[0]:.6f},{logn_sigma2_ci[1]:.6f})")


def print_latex_pgfplots_table() -> None:
    dates, weekend = build_dates_and_weekend(len(STEPS))
    ma7_steps = np.full(len(STEPS), np.nan, dtype=float)
    ma7_sleep = np.full(len(STEPS), np.nan, dtype=float)
    for i in range(6, len(STEPS)):
        ma7_steps[i] = float(STEPS[i - 6 : i + 1].mean())
        ma7_sleep[i] = float(SLEEP[i - 6 : i + 1].mean())

    print("day steps sleep weekend ma7_steps ma7_sleep")
    for i, (s, sl, w, ms, ml) in enumerate(
        zip(STEPS.astype(int), SLEEP, weekend.astype(int), ma7_steps, ma7_sleep),
        start=1,
    ):
        ms_str = "nan" if np.isnan(ms) else f"{ms:.2f}"
        ml_str = "nan" if np.isnan(ml) else f"{ml:.2f}"
        print(f"{i} {int(s)} {sl:.1f} {int(w)} {ms_str} {ml_str}")


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "text"
    if mode == "text":
        print_text_report()
        return 0
    if mode == "pgfplots":
        print_latex_pgfplots_table()
        return 0
    sys.stderr.write("Usage: python report_compute.py [text|pgfplots]\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
