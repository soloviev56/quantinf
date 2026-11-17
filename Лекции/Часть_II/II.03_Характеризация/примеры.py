"""
Лекция II.3. Характеризация квантовых устройств: томография состояний
(линейная инверсия, проекция Смолина–Гамбетты–Смита, максимальное
правдоподобие), томография процессов и ошибки SPAM, калибровочная свобода
GST, randomized benchmarking (модель A p^m + B, нечувствительность к SPAM,
interleaved RB), линейный XEB и тест квантового объёма (heavy outputs),
классические тени (паули- и хааровские/клиффордовы измерения).

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
Порядок множителей: первый — старший (np.kron); матрица Чоя — вход первый.
"""
import time
import itertools
import numpy as np
from math import pi, sqrt, log
from functools import reduce
from scipy.linalg import expm, sqrtm
from scipy.optimize import curve_fit

T_START = time.time()
rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [I2, X, Y, Z]
H1 = np.array([[1, 1], [1, -1]], dtype=complex) / sqrt(2)
S1 = np.diag([1, 1j])
Sdg = S1.conj().T
# поворот базиса измерения: после U измерение в Z эквивалентно измерению X / Y / Z
BASIS_ROT = {1: H1, 2: H1 @ Sdg, 3: I2}


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def dag(A):
    return A.conj().T


def kron(*ops):
    return reduce(np.kron, ops)


def ket(i, d):
    v = np.zeros(d, complex); v[i] = 1
    return v


def apply_kraus(K, rho):
    return sum(k @ rho @ dag(k) for k in K)


def fidelity(r, s):
    q = sqrtm(r)
    return np.real(np.trace(sqrtm(q @ s @ q))) ** 2


def haar_unitary(d):
    G = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / sqrt(2)
    Q, R = np.linalg.qr(G)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def pauli_string(idx):
    return kron(*[PAULIS[i] for i in idx])


# ---------------------------------------------------------------------------
section("1. Томография состояний: линейная инверсия, SGS-проекция, МП")


def settings(n):
    return list(itertools.product((1, 2, 3), repeat=n))


def measure_counts(rho, n, shots):
    """Для каждой настройки (X/Y/Z на каждом кубите) — частоты исходов."""
    data = {}
    for s in settings(n):
        U = kron(*[BASIS_ROT[b] for b in s])
        p = np.clip(np.real(np.diag(U @ rho @ dag(U))), 0, None)
        p = p / p.sum()
        data[s] = rng.multinomial(shots, p) / shots if shots else p
    return data


def bit_signs(n):
    bits = np.array(list(itertools.product((0, 1), repeat=n)))
    return 1 - 2 * bits  # (+1 для 0, -1 для 1)


def linear_inversion(data, n):
    sg = bit_signs(n)
    rho = np.zeros((2 ** n, 2 ** n), complex)
    for idx in itertools.product(range(4), repeat=n):
        # среднее по всем совместимым настройкам
        vals = []
        for s, f in data.items():
            if all(i == 0 or i == b for i, b in zip(idx, s)):
                mask = np.array([i != 0 for i in idx])
                vals.append(f @ np.prod(sg[:, mask], axis=1) if mask.any() else 1.0)
        rho += np.mean(vals) * pauli_string(idx)
    return rho / 2 ** n


def sgs_project(mu_mat):
    """Смолин–Гамбетта–Смит (2012): ближайшая (в норме 2) матрица плотности."""
    w, V = np.linalg.eigh(mu_mat)
    w = w[::-1]; V = V[:, ::-1]
    lam = w.copy()
    a = 0.0
    i = len(w)
    while i > 0 and lam[i - 1] + a / i < 0:
        a += lam[i - 1]
        lam[i - 1] = 0
        i -= 1
    lam[:i] = lam[:i] + a / i
    return (V * lam) @ dag(V)


def mle(data, n, iters=300):
    """Итерации R rho R (Храдил, Ржегачек) по проекторам всех настроек."""
    projs, freqs = [], []
    for s, f in data.items():
        U = kron(*[BASIS_ROT[b] for b in s])
        for k in range(2 ** n):
            v = dag(U)[:, k]
            projs.append(np.outer(v, v.conj())); freqs.append(f[k])
    projs = np.array(projs); freqs = np.array(freqs)
    rho = np.eye(2 ** n, dtype=complex) / 2 ** n
    for _ in range(iters):
        p = np.real(np.einsum("kij,ji->k", projs, rho))
        R = np.einsum("k,kij->ij", np.where(p > 1e-15, freqs / np.maximum(p, 1e-15), 0), projs)
        rho = R @ rho @ R
        rho = (rho + dag(rho)) / 2
        rho /= np.trace(rho).real
    return rho


# точная инверсия без шума
for n in (1, 2, 3):
    psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n); psi /= np.linalg.norm(psi)
    rho_t = np.outer(psi, psi.conj())
    assert np.allclose(linear_inversion(measure_counts(rho_t, n, 0), n), rho_t)
print("  без статистического шума линейная инверсия по 3^n настройкам восстанавливает rho точно (n = 1, 2, 3)")
print("  n   настроек 3^n   параметров 4^n - 1   проекторов 6^n")
for n in (1, 2, 3, 5, 10, 20):
    print(f"  {n:2d}   {3 ** n:12d}   {4 ** n - 1:18d}   {6 ** n:14d}")
# SGS — ближайшая матрица плотности: проверка против прямой оптимизации на случайном примере
A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)); A = (A + dag(A)) / 2
A = A - (np.trace(A) - 1) * np.eye(4) / 4
P = sgs_project(A)
assert np.linalg.eigvalsh(P).min() > -1e-12 and np.isclose(np.trace(P), 1)
def rand_dm(d):
    G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    R = G @ dag(G)
    return R / np.trace(R)


# сравнение с матрицами плотности рядом с P (P + малое возмущение, спроецированное на след 1 и PSD)
best_other = min(np.linalg.norm(A - ((1 - t) * P + t * rand_dm(4))) for t in (0.01, 0.05, 0.2) for _ in range(300))
assert np.linalg.norm(A - P) <= best_other + 1e-12
# статистика: GHZ / Белл / случайное чистое
targets = {}
psi1 = np.array([np.cos(0.6), np.exp(0.7j) * np.sin(0.6)])
targets[1] = psi1
targets[2] = haar_unitary(4)[:, 0]
targets[3] = haar_unitary(8)[:, 0]
TOMO = {}
trials = 20
print("  верность с исходным чистым состоянием <psi|rho|psi> (среднее по", trials, "повторам);")
print("  доля оценок линейной инверсии с отрицательным собственным значением")
print("  n  запусков/настр.   лин. инв.   SGS        МП         1-F(SGS)   1-F(МП)    доля rho<0")
for n in (1, 2, 3):
    psi = targets[n]; rho_t = np.outer(psi, psi.conj())
    for shots in (100, 1000, 10000):
        fl, fs, fm, neg = [], [], [], 0
        for _ in range(trials):
            data = measure_counts(rho_t, n, shots)
            rl = linear_inversion(data, n)
            neg += np.linalg.eigvalsh(rl).min() < -1e-12
            fl.append(np.real(psi.conj() @ rl @ psi))
            rs = sgs_project(rl)
            fs.append(np.real(psi.conj() @ rs @ psi))
            fm.append(np.real(psi.conj() @ mle(data, n, 1000) @ psi))
        TOMO[(n, shots)] = (np.mean(fl), np.mean(fs), np.mean(fm), neg / trials)
        fm_s = f"{np.mean(fm):.5f}"
        print(f"  {n}  {shots:8d}         {np.mean(fl):.5f}    {np.mean(fs):.5f}    {fm_s}    {1 - np.mean(fs):.2e}   {1 - np.mean(fm):.2e}   {neg / trials:.2f}")
# для чистых состояний неадаптивная томография: 1 - F ~ N^(-1/2)
SLOPES = {}
for n in (1, 2, 3):
    for k, nm in ((1, "SGS"), (2, "МП")):
        SLOPES[(n, nm)] = np.polyfit(np.log([100, 1000, 10000]), np.log([1 - TOMO[(n, s)][k] for s in (100, 1000, 10000)]), 1)[0]
print("  наклоны log(1 - F) по log N:", ", ".join(f"n={n} {nm}: {v:.2f}" for (n, nm), v in SLOPES.items()))
assert all(-0.8 < v < -0.3 for v in SLOPES.values())
assert all(TOMO[(n, s)][3] > 0.5 for n in (2, 3) for s in (100, 1000, 10000))
assert all(abs(TOMO[(n, s)][0] - 1) < 0.01 for n in (1, 2, 3) for s in (1000, 10000))  # линейная оценка несмещена

# ---------------------------------------------------------------------------
section("2. Томография процесса и SPAM; калибровочная свобода GST")


def choi_from_outputs(outs):
    """outs: Phi(|0><0|), Phi(|1><1|), Phi(|+><+|), Phi(|+i><+i|) -> J (вход первый)."""
    r00, r11, rp, rpi = outs
    r01 = rp - 0.5 * (r00 + r11) + 1j * (rpi - 0.5 * (r00 + r11))
    r10 = dag(r01)
    blocks = [[r00, r01], [r10, r11]]
    J = np.zeros((4, 4), complex)
    for i in range(2):
        for j in range(2):
            J += np.kron(np.outer(ket(i, 2), ket(j, 2)), blocks[i][j])
    return J


def choi_of(K):
    return sum(np.kron(np.outer(ket(i, 2), ket(j, 2)), apply_kraus(K, np.outer(ket(i, 2), ket(j, 2)))) for i in range(2) for j in range(2))


def F_pro_choi(J, U=I2):
    v = (np.kron(I2, U) @ np.eye(2).reshape(-1)) / sqrt(2)
    return np.real(v.conj() @ J @ v) / 2


IN_STATES = [ket(0, 2), ket(1, 2), (ket(0, 2) + ket(1, 2)) / sqrt(2), (ket(0, 2) + 1j * ket(1, 2)) / sqrt(2)]
# «гейт» X/2 с шумом
Ux = expm(-1j * pi / 4 * X)
g_ad, p_dep = 0.02, 0.01
Kg = [k @ Ux for k in [np.array([[1, 0], [0, np.sqrt(1 - g_ad)]], complex), np.array([[0, np.sqrt(g_ad)], [0, 0]], complex)]]
Kg = [sqrt(1 - 3 * p_dep / 4) * k for k in Kg] + [sqrt(p_dep / 4) * P @ k for P in (X, Y, Z) for k in Kg]
J_true = choi_of(Kg)
F_true = F_pro_choi(J_true, Ux)
outs = [apply_kraus(Kg, np.outer(v, v.conj())) for v in IN_STATES]
assert np.allclose(choi_from_outputs(outs), J_true)
print(f"  точные выходы на 4 входах восстанавливают J; F_pro(шумный X/2 относительно X/2) = {F_true:.5f}")


def readout_noisy_tomo(rho, e):
    """Томография кубита при симметричной ошибке считывания e (бит переворачивается с вероятностью e)."""
    r = []
    for b in (1, 2, 3):
        U = BASIS_ROT[b]
        p = np.real(np.diag(U @ rho @ dag(U)))
        p = (1 - e) * p + e * p[::-1]
        r.append(p[0] - p[1])
    return 0.5 * (I2 + r[0] * X + r[1] * Y + r[2] * Z)


SPAM = {}
for e in (0.0, 0.02, 0.05):
    for e_prep in (0.0, 0.02):
        # ошибка приготовления: вход с вероятностью e_prep заменяется ортогональным
        outs_e = []
        for v in IN_STATES:
            rv = np.outer(v, v.conj())
            rv = (1 - e_prep) * rv + e_prep * (np.eye(2) - rv)
            outs_e.append(readout_noisy_tomo(apply_kraus(Kg, rv), e))
        Je = choi_from_outputs(outs_e)
        SPAM[(e, e_prep)] = F_pro_choi(Je, Ux)
        print(f"  ошибка считывания {e:.2f}, приготовления {e_prep:.2f}: F_pro по томографии = {SPAM[(e, e_prep)]:.5f} (истинная {F_true:.5f})")
assert SPAM[(0.05, 0.0)] < F_true - 0.05 and SPAM[(0.02, 0.02)] < F_true - 0.04
# формула для идеального гейта и ошибки считывания: блоховские компоненты сжимаются в (1-2e)
J_id_e = choi_from_outputs([readout_noisy_tomo(np.outer(v, v.conj()), 0.05) for v in IN_STATES])
f_id = F_pro_choi(J_id_e)
assert np.isclose(f_id, (1 + 3 * (1 - 2 * 0.05)) / 4)
print(f"  идеальный тождественный гейт при ошибке считывания e = 0.05: F_pro = (1 + 3(1-2e))/4 = {f_id:.4f} — смещение 3e/2")

# калибровочная свобода: PTM-представление (вектор (1, r), 4x4)


def ptm(K):
    return np.array([[0.5 * np.real(np.trace(Pi @ apply_kraus(K, Pj))) for Pj in PAULIS] for Pi in PAULIS])


def vec_state(rho):
    return np.array([np.real(np.trace(P @ rho)) for P in PAULIS]) / sqrt(2)


def vec_effect(E):
    return np.array([np.real(np.trace(P @ E)) for P in PAULIS]) / sqrt(2)


G1, G2 = ptm(Kg), ptm([expm(-1j * pi / 4 * Y)])
rho0 = vec_state(np.diag([0.98, 0.02]).astype(complex))
E0 = vec_effect(np.diag([0.97, 0.04]).astype(complex))
B = ptm([expm(-1j * 0.07 * (0.6 * X + 0.8 * Z))])  # калибровочное преобразование (небольшой поворот)
Bi = np.linalg.inv(B)
seqs = [rng.integers(0, 2, size=rng.integers(1, 12)) for _ in range(50)]


def prob(seq, G, rho, E):
    v = rho
    for k in seq:
        v = G[k] @ v
    return E @ v


pa = [prob(s, [G1, G2], rho0, E0) for s in seqs]
pb = [prob(s, [B @ G1 @ Bi, B @ G2 @ Bi], B @ rho0, E0 @ Bi) for s in seqs]
assert np.allclose(pa, pb)
R_ideal = ptm([Ux])
Fp = lambda R: np.trace(R_ideal.T @ R) / 4
print(f"  калибровка G -> B G B^-1, rho -> B rho, E -> E B^-1: все {len(seqs)} вероятностей совпадают (max разница {np.abs(np.array(pa) - np.array(pb)).max():.1e}),")
print(f"    но F_pro(X/2) меняется: {Fp(G1):.5f} -> {Fp(B @ G1 @ Bi):.5f}")
assert abs(Fp(G1) - F_true) < 1e-12 and abs(Fp(B @ G1 @ Bi) - Fp(G1)) > 1e-3

# ---------------------------------------------------------------------------
section("3. Randomized benchmarking (один кубит)")


def canon(U):
    k = np.argmax(np.abs(U.reshape(-1)) > 1e-9)
    ph = U.reshape(-1)[k] / abs(U.reshape(-1)[k])
    return U / ph


C1 = [I2]
frontier = [I2]
while frontier:
    new = []
    for U in frontier:
        for G in (H1, S1):
            V = canon(G @ U)
            if not any(np.allclose(V, W) for W in C1):
                C1.append(V); new.append(V)
    frontier = new
assert len(C1) == 24
C_ptm = np.array([ptm([C]) for C in C1])


def F_pro_ptm(R):
    return np.trace(R) / 4


def rb_seq_survival(m, noise, rho_v, E_v, interleave=None, idx=None):
    if idx is None:
        idx = rng.integers(0, 24, size=m)
    v = rho_v.copy()
    tot = np.eye(4)
    for k in idx:
        v = noise @ (C_ptm[k] @ v)
        tot = C_ptm[k] @ tot
        if interleave is not None:
            v = interleave[1] @ (interleave[0] @ v)
            tot = interleave[0] @ tot
    inv = np.linalg.inv(tot)
    v = noise @ (inv @ v)
    return E_v @ v


# шум: амплитудное затухание + дефазировка + когерентный пере-поворот (не зависит от гейта)
Kn = [np.array([[1, 0], [0, sqrt(1 - 0.004)]], complex), np.array([[0, sqrt(0.004)], [0, 0]], complex)]
Kn = [sqrt(1 - 0.002) * k for k in Kn] + [sqrt(0.002) * Z @ k for k in Kn]
Kn = [k @ expm(-1j * 0.02 / 2 * X) for k in Kn]
N_ptm = ptm(Kn)
Fpro_n = F_pro_ptm(N_ptm)
r_true = 1 - (2 * Fpro_n + 1) / 3
p_true = (4 * Fpro_n - 1) / 3
print(f"  шум после каждого клиффорда: F_pro = {Fpro_n:.6f}, ожидаемые p = (4F_pro - 1)/3 = {p_true:.6f}, r = 1 - F_avg = {r_true:.3e}")
# точная проверка модели: среднее по ВСЕМ последовательностям длины m = 1, 2 даёт A p^m + B
rho_v = vec_state(np.diag([1, 0]).astype(complex))
E_v = vec_effect(np.diag([1, 0]).astype(complex))
exact = {}
for m in (1, 2, 3):
    vals = [rb_seq_survival(m, N_ptm, rho_v, E_v, idx=list(idx)) for idx in itertools.product(range(24), repeat=m)]
    exact[m] = np.mean(vals)
# A и B из m = 1, 2 при известном p; проверка на m = 3
Ae = (exact[1] - exact[2]) / (p_true - p_true ** 2)
Be = exact[1] - Ae * p_true
print(f"  полное усреднение по 24^m последовательностям: P(1) = {exact[1]:.8f}, P(2) = {exact[2]:.8f}, P(3) = {exact[3]:.8f}")
print(f"    A p^3 + B с A = {Ae:.6f}, B = {Be:.6f}: {Ae * p_true ** 3 + Be:.8f}")
assert abs(Ae * p_true ** 3 + Be - exact[3]) < 1e-10
# стандартный протокол: 30 случайных последовательностей на длину, 1000 запусков каждая
lengths = np.array([1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500])
model = lambda m, A, p, B: A * p ** m + B


def run_rb(rho_v, E_v, shots=1000, nseq=30, interleave=None):
    means = []
    for m in lengths:
        vals = []
        for _ in range(nseq):
            s = rb_seq_survival(int(m), N_ptm, rho_v, E_v, interleave)
            vals.append(rng.binomial(shots, min(max(s, 0), 1)) / shots if shots else s)
        means.append(np.mean(vals))
    popt, pcov = curve_fit(model, lengths, means, p0=[0.5, 0.99, 0.5], maxfev=20000)
    return popt, np.sqrt(np.diag(pcov)), np.array(means)


popt, perr, means_clean = run_rb(rho_v, E_v)
r_fit = (1 - popt[1]) / 2
print(f"  RB (30 посл. x 1000 запусков, m = 1..500): A = {popt[0]:.4f}, p = {popt[1]:.6f} +- {perr[1]:.6f}, B = {popt[2]:.4f}")
print(f"    r = (1 - p)/2 = {r_fit:.3e}; истинное 1 - F_avg = {r_true:.3e}")
assert abs(r_fit - r_true) / r_true < 0.1
# SPAM: ошибка приготовления 2 % и считывания 4 %
rho_spam = vec_state(np.diag([0.98, 0.02]).astype(complex))
E_spam = vec_effect(np.diag([0.96, 0.04]).astype(complex))
popt_s, perr_s, _ = run_rb(rho_spam, E_spam)
print(f"  с ошибками SPAM (2 % приготовление, 4 % считывание): A = {popt_s[0]:.4f}, p = {popt_s[1]:.6f} +- {perr_s[1]:.6f}, B = {popt_s[2]:.4f}")
popt_s0, _, _ = run_rb(rho_spam, E_spam, shots=0)
popt_c0, _, _ = run_rb(rho_v, E_v, shots=0)
print(f"  без дробового шума: p (без SPAM) = {popt_c0[1]:.6f}, p (со SPAM) = {popt_s0[1]:.6f}, A: {popt_c0[0]:.4f} -> {popt_s0[0]:.4f}, B: {popt_c0[2]:.4f} -> {popt_s0[2]:.4f}")
assert abs(popt_s[1] - p_true) < 4 * perr_s[1] + 2e-4 and abs(popt_s0[1] - popt_c0[1]) < 2e-4
assert abs(popt_s0[0] - popt_c0[0]) > 0.01
RB = dict(p=popt[1], dp=perr[1], r=r_fit, A=popt[0], B=popt[2], pS=popt_s[1], AS=popt_s[0], BS=popt_s[2])
# interleaved RB для X с дополнительной ошибкой
Kx_err = [sqrt(1 - 3 * 0.006 / 4) * I2] + [sqrt(0.006 / 4) * P for P in (X, Y, Z)]
X_ptm = ptm([X])
Xerr_ptm = ptm(Kx_err) @ N_ptm  # шум гейта X: собственная ошибка + общий шум
popt_i, perr_i, _ = run_rb(rho_v, E_v, interleave=(X_ptm, Xerr_ptm))
rX_est = (1 - popt_i[1] / popt[1]) / 2
rX_true = 1 - (2 * F_pro_ptm(Xerr_ptm) + 1) / 3
print(f"  interleaved RB для X: p_int = {popt_i[1]:.6f}; r_X = (1 - p_int/p)/2 = {rX_est:.3e}; истинное 1 - F_avg(шум X) = {rX_true:.3e}")
assert abs(rX_est - rX_true) / rX_true < 0.2
# RB видит F_avg, но не худший случай: когерентная ошибка
eps = 0.02
print(f"  когерентный пере-поворот 0.02 рад в шуме: вклад в r = sin^2(0.01)*2/3 = {2 / 3 * np.sin(eps / 2) ** 2:.2e}, а ||.||_dia = 2 sin(0.01) = {2 * np.sin(eps / 2):.2e}")
RB_i = dict(p=popt_i[1], r=rX_est, rt=rX_true)

# ---------------------------------------------------------------------------
section("4. Линейный XEB и квантовый объём")
n5 = 5
CZ = np.diag([1, 1, 1, -1]).astype(complex)


def apply_1q(rho, U, q, n):
    Uf = kron(*[U if i == q else I2 for i in range(n)])
    return Uf @ rho @ dag(Uf)


def apply_2q(rho, U, q, n):
    """U на соседних кубитах q, q+1."""
    Uf = kron(*([I2] * q + [U] + [I2] * (n - q - 2)))
    return Uf @ rho @ dag(Uf)


def dep_1q(rho, p, q, n):
    out = (1 - 3 * p / 4) * rho
    for P in (X, Y, Z):
        Pf = kron(*[P if i == q else I2 for i in range(n)])
        out = out + p / 4 * Pf @ rho @ Pf
    return out


def dep_2q(rho, lam, q, n):
    """lam rho + (1 - lam) (I/4 (x) Tr_{q,q+1} rho) на паре кубитов."""
    out = lam * rho
    acc = 0
    for a in range(4):
        for b in range(4):
            Pf = kron(*([I2] * q + [PAULIS[a], PAULIS[b]] + [I2] * (n - q - 2)))
            acc = acc + Pf @ rho @ Pf
    return out + (1 - lam) * acc / 16


def random_circuit(n, depth):
    layers = []
    for d in range(depth):
        ones = [haar_unitary(2) for _ in range(n)]
        pairs = list(range(d % 2, n - 1, 2))
        layers.append((ones, pairs))
    return layers


def run_circuit(layers, n, p1=0.0, lam2=1.0):
    rho = np.zeros((2 ** n, 2 ** n), complex); rho[0, 0] = 1
    n1 = n2 = 0
    for ones, pairs in layers:
        for q, U in enumerate(ones):
            rho = apply_1q(rho, U, q, n)
            if p1:
                rho = dep_1q(rho, p1, q, n)
            n1 += 1
        for q in pairs:
            rho = apply_2q(rho, CZ, q, n)
            if lam2 < 1:
                rho = dep_2q(rho, lam2, q, n)
            n2 += 1
    return rho, n1, n2


p1, lam2 = 0.002, 0.985
F1 = 1 - 3 * p1 / 4
F2 = lam2 + (1 - lam2) / 16
print(f"  n = {n5}, шум: деполяризация после однокубитных (F_pro = {F1:.4f}) и двухкубитных (F_pro = {F2:.5f}) гейтов")
print("  глубина   N sum p^2 - 1   F_XEB (ожидаемое)   норм. F_XEB   прод. F_pro   норм./прод.   верность <psi|rho|psi>")
XEB = []
for depth in (4, 8, 12, 16, 20):
    fx, fp, fid, pt, fn = [], [], [], [], []
    for _ in range(20):
        lay = random_circuit(n5, depth)
        rho_i, n1, n2 = run_circuit(lay, n5)
        rho_n, _, _ = run_circuit(lay, n5, p1, lam2)
        p = np.real(np.diag(rho_i)); q = np.real(np.diag(rho_n))
        N = 2 ** n5
        pt.append(N * (p ** 2).sum() - 1)
        fx.append(N * (p * q).sum() - 1)
        fn.append((N * (p * q).sum() - 1) / (N * (p ** 2).sum() - 1))
        fp.append(F1 ** n1 * F2 ** n2)
        fid.append(np.real(np.trace(rho_i @ rho_n)))
    XEB.append((depth, np.mean(pt), np.mean(fx), np.mean(fn), np.mean(fp), np.mean(fn) / np.mean(fp), np.mean(fid)))
    print(f"  {depth:5d}     {np.mean(pt):.4f}          {np.mean(fx):.4f}            {np.mean(fn):.4f}        {np.mean(fp):.4f}        {np.mean(fn) / np.mean(fp):.3f}        {np.mean(fid):.4f}")
assert all(abs(r[5] - 1) < 0.1 for r in XEB)
# квантовый объём: heavy output probability для квадратных схем со случайными SU(4)


def qv_circuit_hop(n, lam2_, ncirc=20):
    hops = []
    for _ in range(ncirc):
        psi_layers = []
        for _ in range(n):
            perm = rng.permutation(n)
            psi_layers.append((perm, [haar_unitary(4) for _ in range(n // 2)]))
        rho_i = np.zeros((2 ** n, 2 ** n), complex); rho_i[0, 0] = 1
        rho_n = rho_i.copy()
        for perm, Us in psi_layers:
            for k, U in enumerate(Us):
                a, b = perm[2 * k], perm[2 * k + 1]
                Uf = full_2q(U, a, b, n)
                rho_i = Uf @ rho_i @ dag(Uf)
                rho_n = Uf @ rho_n @ dag(Uf)
                if lam2_ < 1:
                    rho_n = dep_pair(rho_n, lam2_, a, b, n)
        p = np.real(np.diag(rho_i)); q = np.real(np.diag(rho_n))
        heavy = p > np.median(p)
        hops.append(q[heavy].sum())
    return np.mean(hops), np.std(hops) / sqrt(ncirc)


def full_2q(U, a, b, n):
    """Оператор U на кубитах (a, b) в n-кубитном пространстве (a — первый множитель U)."""
    d = 2 ** n
    M = np.zeros((d, d), complex)
    U4 = U.reshape(2, 2, 2, 2)
    for col in range(d):
        bits = [(col >> (n - 1 - i)) & 1 for i in range(n)]
        for oa in range(2):
            for ob in range(2):
                amp = U4[oa, ob, bits[a], bits[b]]
                if amp != 0:
                    nb = bits.copy(); nb[a] = oa; nb[b] = ob
                    row = int("".join(map(str, nb)), 2)
                    M[row, col] += amp
    return M


def dep_pair(R, lam, a, b, n):
    acc = 0
    for i in range(4):
        for j in range(4):
            ops = [I2] * n
            ops[a] = PAULIS[i]; ops[b] = PAULIS[j]
            Pf = kron(*ops)
            acc = acc + Pf @ R @ Pf
    return lam * R + (1 - lam) * acc / 16


hop_ideal = {}
for n in (4, 6):
    h, s = qv_circuit_hop(n, 1.0, 15)
    hop_ideal[n] = h
    print(f"  QV, n = {n}, идеальные схемы: средняя доля тяжёлых исходов = {h:.4f} +- {s:.4f}  ((1 + ln 2)/2 = {(1 + log(2)) / 2:.4f})")
assert all(abs(hop_ideal[n] - (1 + log(2)) / 2) < 0.06 for n in (4, 6))
QV = {}
for n in (4, 6):
    for lam_ in (0.99, 0.97, 0.94):
        h, s = qv_circuit_hop(n, lam_, 10)
        QV[(n, lam_)] = h
        print(f"  QV, n = {n}, двухкубитная деполяризация 1 - lam = {1 - lam_:.2f}: HOP = {h:.4f} +- {s:.4f} {'(> 2/3)' if h > 2 / 3 else '(< 2/3)'}")
assert QV[(4, 0.99)] > 2 / 3 and QV[(6, 0.94)] < 2 / 3

# ---------------------------------------------------------------------------
section("5. Классические тени")
# одно-кубитные «снимки»: 3 |s><s| - I обращает канал измерения
eig = {}
for b in (1, 2, 3):
    U = BASIS_ROT[b]
    eig[b] = [dag(U)[:, k] for k in range(2)]
# проверка обращения: E[3 U^+|s><s|U - I] = rho для случайного rho
rho_r = np.array([[0.7, 0.2 - 0.1j], [0.2 + 0.1j, 0.3]])
avg = 0
for b in (1, 2, 3):
    for v in eig[b]:
        pr = np.real(v.conj() @ rho_r @ v)
        avg = avg + pr * (3 * np.outer(v, v.conj()) - I2) / 3
assert np.allclose(avg, rho_r)
print("  однокубитный снимок 3|s><s| - I: среднее по базисам X, Y, Z и исходам = rho (обращение канала измерения)")


def ghz_outcome_probs(basis):
    """Вероятности исходов 2^n строк при измерении GHZ_n в базисах basis (кортеж из 1/2/3)."""
    n = len(basis)
    a = np.array([[eig[b][s][0].conj() for s in (0, 1)] for b in basis])  # <s|0>
    c = np.array([[eig[b][s][1].conj() for s in (0, 1)] for b in basis])  # <s|1>
    bits = np.array(list(itertools.product((0, 1), repeat=n)))
    amp = (np.prod(a[np.arange(n), bits], axis=1) + np.prod(c[np.arange(n), bits], axis=1)) / sqrt(2)
    return np.abs(amp) ** 2, bits


def ghz_fid_estimator(basis, bits):
    """<GHZ| (x)_i (3|s_i><s_i| - I) |GHZ> для каждой строки исходов."""
    n = len(basis)
    A = np.array([[3 * np.outer(eig[b][s], eig[b][s].conj()) - I2 for s in (0, 1)] for b in basis])  # [i, s, 2, 2]
    Ai = A[np.arange(n), bits]  # [rows, n, 2, 2]
    t = (np.prod(Ai[:, :, 0, 0], axis=1) + np.prod(Ai[:, :, 1, 1], axis=1) + np.prod(Ai[:, :, 0, 1], axis=1) + np.prod(Ai[:, :, 1, 0], axis=1)) / 2
    return np.real(t)


def pauli_estimator(basis, bits, obs):
    """obs: кортеж 0..3 по кубитам; оценка 3^k prod(собств. значений), если базисы совпадают, иначе 0."""
    sg = 1 - 2 * bits
    val = np.ones(len(bits))
    for i, o in enumerate(obs):
        if o == 0:
            continue
        if basis[i] != o:
            return np.zeros(len(bits))
        val = val * 3 * sg[:, i]
    return val


print("  GHZ_n, случайные паули-измерения: точная дисперсия одного снимка")
print("  n    Var[оценка верности]   Var[Z1 Z2] (=3^2-1)   Var[X..X] (=3^n-1)   снимков для eps = 0.05 (верность)   полная томография 3^n x 1000")
SH = {}
for n in range(2, 9):
    e2 = 0; ez = 0; ex = 0; ef = 0
    for basis in itertools.product((1, 2, 3), repeat=n):
        pr, bits = ghz_outcome_probs(basis)
        est = ghz_fid_estimator(basis, bits)
        ef += pr @ est / 3 ** n
        e2 += pr @ est ** 2 / 3 ** n
        ez += pr @ pauli_estimator(basis, bits, (3, 3) + (0,) * (n - 2)) ** 2 / 3 ** n
        ex += pr @ pauli_estimator(basis, bits, (1,) * n) ** 2 / 3 ** n
    assert abs(ef - 1) < 1e-9
    var_f = e2 - 1
    assert abs(ez - 1 - 8) < 1e-9 and abs(ex - 1 - (3 ** n - 1)) < 1e-6
    SH[n] = var_f
    print(f"  {n}    {var_f:12.3f}           {ez - 1:.1f}                  {ex - 1:10.0f}            {var_f / 0.05 ** 2:12.0f}                   {3 ** n * 1000:10d}")
ratios = [SH[n + 1] / SH[n] for n in range(4, 8)]
print(f"  рост дисперсии оценки верности: Var(n+1)/Var(n) = {', '.join(f'{r:.3f}' for r in ratios)} ")
# Монте-Карло: оценка верности и 10 наблюдаемых по 20 000 снимкам для n = 6
n = 6
Nsnap = 20000
obs_list = [tuple(3 if j in (i, i + 1) else 0 for j in range(n)) for i in range(n - 1)]  # Z_i Z_{i+1}
obs_list += [(1,) * n, (3,) * n, (1, 1) + (0,) * (n - 2), (3,) + (0,) * (n - 1), (2, 2) + (1,) * (n - 2)]
true_vals = []
psi_g = (ket(0, 2 ** n) + ket(2 ** n - 1, 2 ** n)) / sqrt(2)
for o in obs_list:
    true_vals.append(np.real(psi_g.conj() @ pauli_string(o) @ psi_g))
bases_idx = rng.integers(1, 4, size=(Nsnap, n))
sum_f = 0; sum_o = np.zeros(len(obs_list))
uniq, inv_idx, cnt = np.unique(bases_idx, axis=0, return_inverse=True, return_counts=True)
for u, c in zip(uniq, cnt):
    basis = tuple(int(x) for x in u)
    pr, bits = ghz_outcome_probs(basis)
    draws = rng.choice(len(pr), size=c, p=pr / pr.sum())
    sum_f += ghz_fid_estimator(basis, bits[draws]).sum()
    for k, o in enumerate(obs_list):
        sum_o[k] += pauli_estimator(basis, bits[draws], o).sum()
est_f = sum_f / Nsnap
est_o = sum_o / Nsnap
print(f"  n = {n}, {Nsnap} снимков: оценка верности = {est_f:.4f} (истинная 1; ст. ошибка ~ sqrt({SH[n]:.1f}/{Nsnap}) = {sqrt(SH[n] / Nsnap):.4f})")
names = [f"Z{i + 1}Z{i + 2}" for i in range(n - 1)] + ["X^n", "Z^n", "X1X2", "Z1", "Y1Y2X..X"]
for nm, e, t in zip(names, est_o, true_vals):
    print(f"     {nm:9s}: оценка {e:+.4f}, истинное {t:+.4f}")
assert abs(est_f - 1) < 5 * sqrt(SH[n] / Nsnap)
assert all(abs(e - t) < 5 * sqrt(3 ** sum(1 for x in o if x) / Nsnap) for e, t, o in zip(est_o, true_vals, obs_list))
# хааровские (= клиффордовы до 3-х моментов) снимки: (2^n + 1) U^+|b><b|U - I, дисперсия верности <= 3
n = 3
d = 2 ** n
psi_t = (ket(0, d) + ket(d - 1, d)) / sqrt(2)
vals = []
acc = np.zeros((d, d), complex)
for _ in range(20000):
    U = haar_unitary(d)
    pr = np.abs(U @ psi_t) ** 2
    b = rng.choice(d, p=pr / pr.sum())
    v = dag(U)[:, b]
    snap = (d + 1) * np.outer(v, v.conj()) - np.eye(d)
    acc += snap
    vals.append(np.real(psi_t.conj() @ snap @ psi_t))
acc /= 20000
vals = np.array(vals)
print(f"  глобальные хааровские снимки, GHZ_3, 20000 снимков: оценка верности {vals.mean():.4f}, выборочная дисперсия {vals.var():.3f} (<= 3)")
print(f"    ||E[снимок] - rho||_max = {np.abs(acc - np.outer(psi_t, psi_t.conj())).max():.3f} (сходится как 1/sqrt(N))")
assert abs(vals.mean() - 1) < 0.05 and vals.var() < 3.2
# точная дисперсия для глобальных снимков: для O = |psi><psi| и rho = |psi><psi|: E[o^2] = ((d+1)^2 * E|<psi|v>|^4 ... )
# E over Haar b with Born weights: E[|<psi|v>|^4] (взвешенное) = 6/((d+1)(d+2)); o = (d+1) x - 1, x = |<psi|v>|^2
Ex = 2 / (d + 1)
Ex2 = 6 / ((d + 1) * (d + 2))
var_exact = (d + 1) ** 2 * Ex2 - 2 * (d + 1) * Ex + 1 - 1  # E[o^2] - E[o]^2, E[o] = 1
print(f"    точная дисперсия: E_Born[x] = 2/(d+1), E_Born[x^2] = 6/((d+1)(d+2)) -> Var = 2(d-1)/(d+2) = {var_exact:.4f} (< 2 при любом d)")
assert abs(var_exact - 2 * (d - 1) / (d + 2)) < 1e-12 and abs(vals.var() - var_exact) < 0.1
SH_haar = (vals.mean(), vals.var(), var_exact)

# ---------------------------------------------------------------------------
section("6. Ответы к задачам семинара")
print("  1: rho = (I + <X>X + <Y>Y + <Z>Z)/2; при <X> = 0.6, <Y> = 0, <Z> = 0.9: собств. значения")
r_ex = 0.5 * (I2 + 0.6 * X + 0.9 * Z)
ev = np.linalg.eigvalsh(r_ex)
print(f"     {ev.round(4)} -> |r| = {np.hypot(0.6, 0.9):.4f} > 1; SGS-проекция: собств. значения {np.linalg.eigvalsh(sgs_project(r_ex)).round(4)}")
assert ev.min() < 0
ps = np.linalg.eigvalsh(sgs_project(r_ex))
assert np.allclose(ps, [0, 1])
print(f"  2: n = 5: 3^5 = {3 ** 5} настроек, 4^5 - 1 = {4 ** 5 - 1} параметров; процесс: 12^5 = {12 ** 5} конфигураций, 16^5 - 4^5 = {16 ** 5 - 4 ** 5} параметров")
print(f"  4: p = 0.995 -> r = (1-p)/2 = {(1 - 0.995) / 2:.4f}; F_avg = {1 - (1 - 0.995) / 2:.4f}; F_pro = (3 F_avg - 1)/2 = {(3 * (1 - 0.0025) - 1) / 2:.4f}")
q7 = 0.03
Dq = ptm([sqrt(1 - 3 * q7 / 4) * I2] + [sqrt(q7 / 4) * P for P in (X, Y, Z)])
A7 = vec_effect(np.diag([1, 0]).astype(complex)) @ Dq @ (rho_v - vec_state(I2 / 2))
B7 = vec_effect(np.diag([1, 0]).astype(complex)) @ Dq @ vec_state(I2 / 2)
assert np.isclose(A7, (1 - q7) / 2) and np.isclose(B7, 0.5)
print(f"  7: Lambda = Delta_q, идеальные SPAM: A = (1 - q)/2 = {A7:.4f} (q = {q7}), B = 1/2")
print(f"  5: двухкубитный RB: r = (1-p)(d-1)/d = 3(1-p)/4; p = 0.98 -> r = {3 * 0.02 / 4:.4f}")
print(f"  6: p_int/p = 0.990/0.995 -> r_C = (1 - p_int/p)/2 = {(1 - 0.990 / 0.995) / 2:.2e}")
print(f"  10: Var(Pauli веса k) = 3^k - <P>^2; k = 2: 9 - <P>^2 <= 9; для M = 100 наблюдаемых веса 2 с eps = 0.05, delta = 0.01:")
Nk = 9 / 0.05 ** 2 * 2 * log(2 * 100 / 0.01)
print(f"     N ~ (9/eps^2) * 2 ln(2M/delta) = {Nk:.0f} снимков (оценка через медиану средних с константами по порядку)")
F9 = 0.999 ** 1000 * 0.995 ** 500
print(f"  9: 1000 однокубитных гейтов с ошибкой 0.1 % и 500 двухкубитных с 0.5 %: F ~ {F9:.4f}; выборок для 5 sigma ~ 25/F^2 = {25 / F9 ** 2:.0f}")
from scipy.integrate import quad
hop = quad(lambda x: x * np.exp(-x), log(2), np.inf)[0]
print(f"  12: Портер–Томас: медиана Np = ln 2, HOP = int_ln2^inf x e^-x dx = {hop:.6f} = (1 + ln 2)/2 = {(1 + log(2)) / 2:.6f}")
assert abs(hop - (1 + log(2)) / 2) < 1e-10
print(f"  3: тождественный гейт, ошибка считывания e: F_pro = (1 + 3(1 - 2e))/4 = 1 - 3e/2; e = 0.02: {1 - 1.5 * 0.02:.3f}")

print(f"\nВсе проверки пройдены. Время работы: {time.time() - T_START:.1f} с")
