"""
Лекция II.12. Информация Холево и различимость: классико-квантовые
состояния и chi = I(X:B); свойства chi; граница Холево на случайных ансамблях
и POVM; доступная информация — численная оптимизация по POVM из k = 2..6
элементов ранга 1 (ансамбли {|0>,|+>}, трин, тетраэдр, BB84); два чистых
состояния и измерение Хелстрома; теорема Дэвиса (k <= d^2); субэнтропия —
нижняя граница; коллективные измерения: кодовые слова Переса–Вуттерса из пар
трин-состояний; неравенство обработки данных для chi и запрет клонирования;
n кубитов несут не больше n бит; сверхплотное кодирование; квантовые коды
случайного доступа и граница Наяка.

Соглашения (Лекции/Обозначения.md, §23–§24): log = log2; ансамбль {p_x, rho_x};
cq-состояние rho_XB = sum_x p_x |x><x| (x) rho_x; chi = S(sum p_x rho_x) - sum p_x S(rho_x);
I_acc = max по POVM взаимной информации I(X:Y). Трин-состояния
|t_k> = cos(pi k/3)|0> + sin(pi k/3)|1> (I.3, §3.3).

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # мелкие матрицы: многопоточность BLAS только мешает
import time
from math import log2, sqrt, pi, log
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import sqrtm

T_START = time.time()
rng = np.random.default_rng(2026)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def H(p):
    p = np.asarray(p, float).ravel()
    p = p[p > 1e-15]
    return float(-(p * np.log2(p)).sum())


def h(x):
    return H([x, 1 - x])


def S(rho):
    return H(np.clip(np.linalg.eigvalsh((rho + rho.conj().T) / 2), 0, None))


def proj(v):
    v = np.asarray(v, complex)
    return np.outer(v, v.conj()) / np.vdot(v, v).real


def chi(ps, rhos):
    avg = sum(p * r for p, r in zip(ps, rhos))
    return S(avg) - sum(p * S(r) for p, r in zip(ps, rhos))


def mutual_info(ps, rhos, povm):
    """I(X:Y) для ансамбля и POVM: p(y|x) = Tr(E_y rho_x)."""
    P = np.array([[np.trace(E @ r).real for E in povm] for r in rhos])   # x, y
    P = np.clip(P, 0, None)
    pxy = np.asarray(ps)[:, None] * P
    return H(pxy.sum(1)) + H(pxy.sum(0)) - H(pxy)


def povm_from_params(theta, d, k):
    """Параметризация, гарантирующая POVM из k элементов ранга 1: E_j = G^{-1/2} m_j m_j^+ G^{-1/2}, G = sum m_j m_j^+."""
    M = (theta[:d * k] + 1j * theta[d * k:]).reshape(d, k)
    G = M @ M.conj().T
    w, V = np.linalg.eigh(G)
    Gm = (V / np.sqrt(np.clip(w, 1e-14, None))) @ V.conj().T
    N = Gm @ M
    return [np.outer(N[:, j], N[:, j].conj()) for j in range(k)]


def accessible_info(ps, rhos, k, starts=12, real=False):
    d = rhos[0].shape[0]
    best, bestE = -1, None

    def f(th):
        return -mutual_info(ps, rhos, povm_from_params(th, d, k))

    for _ in range(starts):
        th0 = rng.normal(size=2 * d * k)
        if real:
            th0[d * k:] = 0
        res = minimize(f, th0, method="L-BFGS-B", options=dict(maxiter=3000))
        if -res.fun > best:
            best, bestE = -res.fun, povm_from_params(res.x, d, k)
    return best, bestE


def is_povm(Es, tol=1e-9):
    d = Es[0].shape[0]
    return np.allclose(sum(Es), np.eye(d), atol=tol) and all(np.linalg.eigvalsh(E).min() > -tol for E in Es)


def rand_rho(d, rank=None):
    rank = d if rank is None else rank
    G = rng.normal(size=(d, rank)) + 1j * rng.normal(size=(d, rank))
    r = G @ G.conj().T
    return r / np.trace(r).real


def rand_povm(d, k):
    return povm_from_params(rng.normal(size=2 * d * k), d, k)


def bloch_state(n):
    return 0.5 * (np.eye(2) + n[0] * X + n[1] * Y + n[2] * Z)


X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
k0, k1 = np.array([1, 0], complex), np.array([0, 1], complex)
plus, minus = (k0 + k1) / sqrt(2), (k0 - k1) / sqrt(2)

ENS = {
    "{|0>,|+>}": ([0.5, 0.5], [proj(k0), proj(plus)]),
    "трин": ([1 / 3] * 3, [proj([np.cos(pi * k / 3), np.sin(pi * k / 3)]) for k in range(3)]),
    "тетраэдр": ([0.25] * 4, [bloch_state(np.array(v) / sqrt(3)) for v in
                              ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]),
    "BB84": ([0.25] * 4, [proj(k0), proj(k1), proj(plus), proj(minus)]),
}

# ===========================================================================
section("1. Классико-квантовые состояния и информация Холево")
CHI = {}
for name, (ps, rhos) in ENS.items():
    CHI[name] = chi(ps, rhos)
    # chi = I(X:B) для cq-состояния
    dX, d = len(ps), 2
    rXB = sum(p * np.kron(proj(np.eye(dX)[x]), r) for x, (p, r) in enumerate(zip(ps, rhos)))
    rX = np.diag(ps)
    rB = sum(p * r for p, r in zip(ps, rhos))
    IXB = S(rX) + S(rB) - S(rXB)
    assert abs(IXB - CHI[name]) < 1e-10
    print(f"  {name:10s}: chi = {CHI[name]:.4f} бита = I(X:B);  H(X) = {H(ps):.4f}, S(rho) = {S(rB):.4f}")
assert abs(CHI["{|0>,|+>}"] - h(np.cos(pi / 8) ** 2)) < 1e-12
for name in ("трин", "тетраэдр", "BB84"):
    assert abs(CHI[name] - 1) < 1e-12
# свойства: 0 <= chi <= min(H(X), S(rho)) <= log d; равенство chi = H(X) <=> ортогональные носители
worst = 10
for _ in range(500):
    d, m = int(rng.integers(2, 5)), int(rng.integers(2, 6))
    ps = rng.dirichlet(np.ones(m))
    rhos = [rand_rho(d, int(rng.integers(1, d + 1))) for _ in range(m)]
    c = chi(ps, rhos)
    avg = sum(p * r for p, r in zip(ps, rhos))
    worst = min(worst, c, H(ps) - c, S(avg) - c, log2(d) - S(avg))
assert worst > -1e-10
print(f"  500 случайных ансамблей: min запаса в 0 <= chi <= min(H(X), S(rho)) <= log d: {worst:.2e}")
# два чистых состояния с перекрытием c: chi = h((1 + c)/2)
for c in (0.0, 0.3, 1 / sqrt(2), 0.95):
    v1 = c * k0 + sqrt(1 - c * c) * k1
    assert abs(chi([0.5, 0.5], [proj(k0), proj(v1)]) - h((1 + c) / 2)) < 1e-12

# ===========================================================================
section("2. Граница Холево: I(X:Y) <= chi для случайных ансамблей и POVM")
worst, cnt = 10, 0
for _ in range(1000):
    d, m = int(rng.integers(2, 4)), int(rng.integers(2, 6))
    k = int(rng.integers(d, 8))
    ps = rng.dirichlet(np.ones(m))
    rhos = [rand_rho(d, int(rng.integers(1, d + 1))) for _ in range(m)]
    Es = rand_povm(d, k)
    assert is_povm(Es)
    worst = min(worst, chi(ps, rhos) - mutual_info(ps, rhos, Es))
    cnt += 1
print(f"  {cnt} случайных пар (ансамбль, POVM): min[chi - I(X:Y)] = {worst:.3e} >= 0")
assert worst > -1e-10
# доказательство: канал измерения M(rho) = sum_y Tr(E_y rho)|y><y| на B; I(X:Y) = I(X:B') для rho_XB'
ps, rhos = ENS["{|0>,|+>}"]
Es = rand_povm(2, 3)
meas = lambda r: np.diag([np.trace(E @ r).real for E in Es])
assert abs(chi(ps, [meas(r) for r in rhos]) - mutual_info(ps, rhos, Es)) < 1e-10
print("  I(X:Y) = chi ансамбля после канала измерения (диагональные состояния) — проверено")

# ===========================================================================
section("3. Доступная информация: оптимизация по POVM из k элементов ранга 1")
ACC = {}
print("  ансамбль     chi      " + "  ".join(f"k = {k}" for k in range(2, 7)))
for name, (ps, rhos) in ENS.items():
    row = []
    for k in range(2, 7):
        val, E = accessible_info(ps, rhos, k, starts=10)
        assert is_povm(E, 1e-7)
        row.append(val)
    ACC[name] = row
    print(f"  {name:10s}  {CHI[name]:.4f}   " + "  ".join(f"{v:.4f}" for v in row))
I_hel = 1 - h(np.sin(pi / 8) ** 2)
assert abs(max(ACC["{|0>,|+>}"]) - I_hel) < 1e-6
assert abs(ACC["трин"][1] - (log2(3) - 1)) < 1e-6 and max(ACC["трин"]) < log2(3) - 1 + 1e-6
assert abs(ACC["тетраэдр"][2] - log2(4 / 3)) < 1e-6 and max(ACC["тетраэдр"]) < log2(4 / 3) + 1e-6
assert abs(max(ACC["BB84"]) - 0.5) < 1e-6
print(f"  {{|0>,|+>}}: I_acc = 1 - h(sin^2(pi/8)) = {I_hel:.4f} (измерение Хелстрома); chi - I_acc = {CHI['{|0>,|+>}'] - I_hel:.4f}")
print(f"  трин: I_acc = log2(3) - 1 = {log2(3) - 1:.4f} (k = 3: «анти-трин» POVM); тетраэдр: log2(4/3) = {log2(4 / 3):.4f}; "
      f"BB84: 0.5")
# явные оптимальные POVM
tr = ENS["трин"][1]
anti = [2 / 3 * (np.eye(2) - r) for r in tr]          # E_k ~ |t_k^perp><t_k^perp|
assert is_povm(anti)
I_anti = mutual_info(ENS["трин"][0], tr, anti)
tri_povm = [2 / 3 * r for r in tr]                    # трин-POVM I.3: p(k|j) = 2/3, 1/6
I_tri = mutual_info(ENS["трин"][0], tr, tri_povm)
print(f"  трин: «анти-трин» POVM (2/3)(I - |t_k><t_k|): I = {I_anti:.4f}; трин-POVM (I.3, §3.3): I = {I_tri:.4f}")
assert abs(I_anti - (log2(3) - 1)) < 1e-12 and abs(I_tri - (log2(3) - H([2 / 3, 1 / 6, 1 / 6]))) < 1e-12
assert I_tri < I_anti
# лучший двухисходный (проективный) для трин
print(f"  трин, проективное измерение (k = 2): {ACC['трин'][0]:.4f} < {ACC['трин'][1]:.4f}: POVM строго лучше")
assert ACC["трин"][0] < ACC["трин"][1] - 0.01
tet = ENS["тетраэдр"][1]
anti_t = [0.5 * (np.eye(2) - r) * 2 / 2 for r in tet]
anti_t = [(np.eye(2) - r) / 2 for r in tet]
assert is_povm(anti_t)
assert abs(mutual_info([0.25] * 4, tet, anti_t) - log2(4 / 3)) < 1e-12
# Дэвис: достаточно d^2 = 4 элементов; k = 5, 6 не лучше k = 4
for name in ENS:
    assert max(ACC[name][3:]) < ACC[name][2] + 1e-5
print("  теорема Дэвиса: для d = 2 оптимум достигается при k <= 4 (k = 5, 6 не лучше)")

# ===========================================================================
section("4. Два чистых состояния: chi, I_acc, Хелстром")
TWO = {}
print("    c = |<psi0|psi1>|   chi = h((1+c)/2)   I_acc (оптим.)   Хелстром 1 - h((1-sqrt(1-c^2))/2)   P_err")
for c in (0.2, 0.5, 1 / sqrt(2), 0.9):
    v1 = c * k0 + sqrt(1 - c * c) * k1
    ps, rhos = [0.5, 0.5], [proj(k0), proj(v1)]
    val, _ = accessible_info(ps, rhos, 4, starts=8)
    perr = (1 - sqrt(1 - c * c)) / 2
    hel = 1 - h(perr)
    TWO[c] = (chi(ps, rhos), val, hel, perr)
    print(f"    {c:.4f}            {TWO[c][0]:.4f}             {val:.4f}           {hel:.4f}                            {perr:.4f}")
    assert abs(val - hel) < 1e-6
# неравновероятные: численно
ps, rhos = [0.8, 0.2], [proj(k0), proj(plus)]
val_u, _ = accessible_info(ps, rhos, 4, starts=8)
print(f"  {{|0>,|+>}} с вероятностями (0.8, 0.2): chi = {chi(ps, rhos):.4f}, I_acc = {val_u:.4f}")
assert val_u < chi(ps, rhos)

# ===========================================================================
section("5. Субэнтропия Йожи–Робба–Вуттерса — нижняя граница I_acc чистых ансамблей")


def subentropy(rho):
    lam = np.sort(np.linalg.eigvalsh(rho))[::-1]
    lam = lam + 1e-7 * np.arange(len(lam))[::-1]   # снятие вырождения (предел непрерывен)
    q = 0.0
    for k, lk in enumerate(lam):
        pr = np.prod([lk / (lk - ll) for l_, ll in enumerate(lam) if l_ != k])
        q -= pr * lk * np.log2(lk)
    return q


Q_half = subentropy(np.eye(2) / 2)
print(f"  Q(I/2) = 1 - 1/(2 ln 2) = {Q_half:.4f} <= I_acc: трин {log2(3) - 1:.4f}, BB84 0.5, тетраэдр {log2(4 / 3):.4f}")
assert abs(Q_half - (1 - 1 / (2 * log(2)))) < 1e-5
r0p = 0.5 * (proj(k0) + proj(plus))
Q0p = subentropy(r0p)
print(f"  {{|0>,|+>}}: Q(rho) = {Q0p:.4f} <= I_acc = {I_hel:.4f} <= chi = S(rho) = {CHI['{|0>,|+>}']:.4f}")
assert Q0p < I_hel < CHI["{|0>,|+>}"]
assert Q_half < min(log2(3) - 1, 0.5, log2(4 / 3))
QMIN = Q_half

# ===========================================================================
section("6. Коллективное измерение: кодовые слова Переса–Вуттерса |t_k>|t_k>")
pw = [np.kron(r, r) for r in tr]
ps3 = [1 / 3] * 3
chi_pw = chi(ps3, pw)
# «довольно хорошее» измерение (PGM): E_k = rho^{-1/2} p_k rho_k rho^{-1/2} на носителе
avg = sum(p * r for p, r in zip(ps3, pw))
w, V = np.linalg.eigh(avg)
inv_sqrt = sum((1 / np.sqrt(wi)) * np.outer(V[:, i], V[:, i].conj()) for i, wi in enumerate(w) if wi > 1e-12)
supp = sum(np.outer(V[:, i], V[:, i].conj()) for i, wi in enumerate(w) if wi > 1e-12)
pgm = [inv_sqrt @ (p * r) @ inv_sqrt for p, r in zip(ps3, pw)]
pgm.append(np.eye(4) - supp)
assert is_povm(pgm)
I_pgm = mutual_info(ps3, pw, pgm)
I_pw_opt, _ = accessible_info(ps3, pw, 6, starts=10, real=True)
I_sep = 2 * (log2(3) - 1)


def best_product(k1, k2, starts=12):
    """Лучшее неадаптивное произведение POVM E_a (x) F_b на кодовых словах Переса–Вуттерса."""
    best = -1
    n1 = 2 * 2 * k1

    def f(th):
        E = povm_from_params(th[:n1], 2, k1)
        F = povm_from_params(th[n1:], 2, k2)
        return -mutual_info(ps3, pw, [np.kron(a, b) for a in E for b in F])

    for _ in range(starts):
        res = minimize(f, rng.normal(size=n1 + 4 * k2), method="L-BFGS-B", options=dict(maxiter=3000))
        best = max(best, -res.fun)
    return best


I_prod = max(best_product(3, 3), best_product(2, 3), best_product(4, 4, starts=6))
print(f"  chi(кодовые слова) = {chi_pw:.4f}; PGM: I = {I_pgm:.4f}; оптимизация (k = 6): {I_pw_opt:.4f}")
print(f"  лучшее произведение POVM на двух кубитах (неадаптивное, k <= 4): I = {I_prod:.4f}")
print(f"  на кубит: коллективное {max(I_pgm, I_pw_opt) / 2:.4f} > I_acc(трин) = {log2(3) - 1:.4f} "
      f"(посимвольное измерение без кодирования)")
assert I_pgm > I_sep and I_pw_opt >= I_pgm - 1e-6 and max(I_pgm, I_pw_opt) <= chi_pw + 1e-9
assert I_prod < I_pgm

# ===========================================================================
section("7. Неравенство обработки данных для chi и запрет клонирования")


def rand_channel(din, dout, nk):
    nk = max(nk, -(-din // dout))
    G = rng.normal(size=(dout * nk, din)) + 1j * rng.normal(size=(dout * nk, din))
    Qm, _ = np.linalg.qr(G)
    return [Qm[j * dout:(j + 1) * dout, :] for j in range(nk)]


worst = 10
for _ in range(300):
    m = int(rng.integers(2, 5))
    ps = rng.dirichlet(np.ones(m))
    rhos = [rand_rho(3, int(rng.integers(1, 4))) for _ in range(m)]
    Ks = rand_channel(3, int(rng.integers(2, 4)), int(rng.integers(1, 4)))
    out = [sum(K @ r @ K.conj().T for K in Ks) for r in rhos]
    worst = min(worst, chi(ps, rhos) - chi(ps, out))
print(f"  300 случайных каналов: min[chi(rho_x) - chi(Phi(rho_x))] = {worst:.3e} >= 0")
assert worst > -1e-10
CLONE = {}
for c in (0.5, 1 / sqrt(2), 0.9):
    v1 = c * k0 + sqrt(1 - c * c) * k1
    one = chi([0.5, 0.5], [proj(k0), proj(v1)])
    two = chi([0.5, 0.5], [np.kron(proj(k0), proj(k0)), np.kron(proj(v1), proj(v1))])
    CLONE[c] = (one, two)
    assert two > one + 1e-3 and abs(two - h((1 + c * c) / 2)) < 1e-12
print("  клонирование psi -> psi x psi увеличило бы chi: " +
      ", ".join(f"c = {c:.3f}: {o:.4f} -> {t:.4f}" for c, (o, t) in CLONE.items()))
# ортогональные состояния (c = 0): chi = 1 и после «клонирования» — клонирование не противоречит

# ===========================================================================
section("8. Следствия: n кубитов — не больше n бит; сверхплотное кодирование; QRAC")
worst = 10
for _ in range(200):
    m = int(rng.integers(2, 9))
    ps = rng.dirichlet(np.ones(m))
    rhos = [rand_rho(4, 1) for _ in range(m)]
    Es = rand_povm(4, int(rng.integers(4, 10)))
    worst = min(worst, 2 - mutual_info(ps, rhos, Es))
assert worst > 0
print(f"  200 случайных двухкубитных ансамблей и POVM: I(X:Y) <= 2 бита (min запас {worst:.3f})")
# сверхплотное кодирование: Алиса применяет X^y Z^x к своей половине Phi+; ансамбль у Боба из двух кубитов
phi = (np.kron(k0, k0) + np.kron(k1, k1)) / sqrt(2)
SD = []
for x in (0, 1):
    for y in (0, 1):
        U = np.linalg.matrix_power(X, y) @ np.linalg.matrix_power(Z, x)
        SD.append(proj(np.kron(U, np.eye(2)) @ phi))
chi_sd = chi([0.25] * 4, SD)
# без половины Боба (только отправленный кубит) — chi = 0
chi_sent = chi([0.25] * 4, [np.einsum("ijkj->ik", r.reshape(2, 2, 2, 2)) for r in SD])
print(f"  сверхплотное кодирование: chi(два кубита у Боба) = {chi_sd:.4f}; chi(только посланный кубит) = {chi_sent:.4f}")
assert abs(chi_sd - 2) < 1e-12 and abs(chi_sent) < 1e-12
# QRAC 2 -> 1: состояния с векторами Блоха ((-1)^{b2}, 0, (-1)^{b1})/sqrt2; бит 1 — измерение Z, бит 2 — X
qr = {}
for b1 in (0, 1):
    for b2 in (0, 1):
        qr[(b1, b2)] = bloch_state(np.array([(-1) ** b2, 0, (-1) ** b1]) / sqrt(2))
P0z, P0x = proj(k0), proj(plus)
p1 = np.mean([np.trace((P0z if b1 == 0 else np.eye(2) - P0z) @ r).real for (b1, b2), r in qr.items()])
p2 = np.mean([np.trace((P0x if b2 == 0 else np.eye(2) - P0x) @ r).real for (b1, b2), r in qr.items()])
nayak = 2 * (1 - h(p1))
print(f"  QRAC 2 -> 1: вероятность угадать любой бит = {p1:.4f} = cos^2(pi/8); граница Наяка: m >= (1 - h(p)) n = {nayak:.4f} <= 1")
assert abs(p1 - np.cos(pi / 8) ** 2) < 1e-12 and abs(p2 - p1) < 1e-12 and nayak <= 1
# классический 2 -> 1 код: лучшая вероятность 3/4 (детерминированно: послать первый бит, угадывать второй)
best_cl = 0
for f in range(16):  # функции {0,1}^2 -> {0,1}
    enc = [(f >> i) & 1 for i in range(4)]
    for g1 in range(4):
        for g2 in range(4):
            dec1 = [(g1 >> m_) & 1 for m_ in range(2)]
            dec2 = [(g2 >> m_) & 1 for m_ in range(2)]
            ok = 0
            for idx, (b1, b2) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                m_ = enc[idx]
                ok += (dec1[m_] == b1) + (dec2[m_] == b2)
            best_cl = max(best_cl, min(ok / 8, ok / 8))
print(f"  классический 2 -> 1 код (детерминированный, средняя по двум битам): {best_cl:.2f}")
assert abs(best_cl - 0.75) < 1e-12

# ===========================================================================
section("9. Ответы к задачам")
print(f"  1: {{|0>,|+>}}: chi = {CHI['{|0>,|+>}']:.4f}; I_acc = {I_hel:.4f}")
print(f"  2: трин: chi = {CHI['трин']:.4f}, I_acc = {log2(3) - 1:.4f}; трин-POVM: {I_tri:.4f}")
print(f"  3: тетраэдр: chi = 1, I_acc = log2(4/3) = {log2(4 / 3):.4f}")
print(f"  4: BB84: chi = 1, I_acc = 0.5")
print(f"  5: два состояния c = 1/sqrt2: chi = {TWO[1 / sqrt(2)][0]:.4f}, I_acc = {TWO[1 / sqrt(2)][2]:.4f}, P_err = {TWO[1 / sqrt(2)][3]:.4f}")
print(f"  7: клонирование: c = 1/sqrt2: {CLONE[1 / sqrt(2)][0]:.4f} -> {CLONE[1 / sqrt(2)][1]:.4f}")
print(f"  9: Перес–Вуттерс: PGM {I_pgm:.4f} бита на 2 кубита ({I_pgm / 2:.4f} на кубит) против {log2(3) - 1:.4f}")
print(f"  10: QRAC: p = {p1:.4f}, 2(1 - h(p)) = {nayak:.4f}")

print(f"\nВсе проверки пройдены. Время работы: {time.time() - T_START:.1f} с")
