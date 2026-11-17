"""
Лекция I.7. Сепарабельность и меры запутанности: PPT, свидетели, связанная
запутанность, LOCC и мажорирование, concurrence, negativity, состояния Вернера.
Численные иллюстрации ко всем утверждениям лекции.

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
Порядок подсистем: первый множитель — старший (как в N&C и np.kron).
"""
import numpy as np
from itertools import product
from scipy.optimize import minimize

rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def dag(A):
    return A.conj().T


def proj(v):
    return np.outer(v, v.conj())


def kron(*ops):
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def random_state(d):
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    return v / np.linalg.norm(v)


def random_rho_hs(d):
    """Случайная матрица плотности по мере Гильберта–Шмидта."""
    G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    r = G @ dag(G)
    return r / np.trace(r)


def random_separable(dA, dB, k=6):
    w = rng.dirichlet(np.ones(k))
    return sum(wi * kron(proj(random_state(dA)), proj(random_state(dB))) for wi in w)


def ptrans(rho, dA, dB, sys=1):
    """Частичное транспонирование подсистемы sys (0 — A, 1 — B)."""
    t = rho.reshape(dA, dB, dA, dB)
    t = t.transpose(0, 3, 2, 1) if sys == 1 else t.transpose(2, 1, 0, 3)
    return t.reshape(dA * dB, dA * dB)


def ptrace(rho, dA, dB, keep=0):
    t = rho.reshape(dA, dB, dA, dB)
    return np.einsum("ijkj->ik", t) if keep == 0 else np.einsum("ijil->jl", t)


def realign(rho, dA, dB):
    """Перестановка (realignment): R[(a a'),(b b')] = rho[(a b),(a' b')]."""
    return rho.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(dA * dA, dB * dB)


def trace_norm(A):
    return np.sum(np.linalg.svd(A, compute_uv=False))


def entropy(p):
    p = np.asarray(p, float)
    p = p[p > 1e-14]
    return float(-np.sum(p * np.log2(p)))


def h2(x):
    return entropy([x, 1 - x])


def negativity(rho, dA, dB):
    return (trace_norm(ptrans(rho, dA, dB)) - 1) / 2


def log_negativity(rho, dA, dB):
    return np.log2(trace_norm(ptrans(rho, dA, dB)))


YY = kron(Y, Y)


def concurrence(rho):
    """Формула Вуттерса."""
    rt = YY @ rho.conj() @ YY
    lam = np.sqrt(np.clip(np.sort(np.real(np.linalg.eigvals(rho @ rt)))[::-1], 0, None))
    return max(0.0, lam[0] - lam[1] - lam[2] - lam[3])


def eof_wootters(rho):
    C = concurrence(rho)
    return max(0.0, h2((1 + np.sqrt(1 - C ** 2)) / 2))


def ent_entropy(psi, dA, dB):
    s = np.linalg.svd(psi.reshape(dA, dB), compute_uv=False)
    return entropy(s ** 2)


def majorized(x, y):
    """x ≺ y (x мажорируется y)."""
    n = max(len(x), len(y))
    xs = np.sort(np.pad(x, (0, n - len(x))))[::-1]
    ys = np.sort(np.pad(y, (0, n - len(y))))[::-1]
    return bool(np.all(np.cumsum(xs) <= np.cumsum(ys) + 1e-12)) and np.isclose(xs.sum(), ys.sum())


ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)
Phi_p = (kron(ket0, ket0) + kron(ket1, ket1)) / np.sqrt(2)
Psi_m = (kron(ket0, ket1) - kron(ket1, ket0)) / np.sqrt(2)
I4 = np.eye(4)


def rho_iso(p):
    """rho_p из I.5–I.6: изотропное состояние двух кубитов."""
    return p * proj(Phi_p) + (1 - p) * I4 / 4


def rho_W(p):
    """Состояние Вернера в форме ЛР4: p |Psi-><Psi-| + (1-p) I/4."""
    return p * proj(Psi_m) + (1 - p) * I4 / 4


# ---------------------------------------------------------------------------
section("1. Сепарабельные состояния: классические корреляции; явное разложение rho_{1/3}")
rho_cl = 0.5 * kron(proj(ket0), proj(ket0)) + 0.5 * kron(proj(ket1), proj(ket1))
assert not np.allclose(rho_cl, kron(ptrace(rho_cl, 2, 2, 0), ptrace(rho_cl, 2, 2, 1)))
print("  (|00><00| + |11><11|)/2: не произведение, но сепарабельно по определению")
# шесть состояний ±x, ±y, ±z у Алисы и комплексно сопряжённые у Боба
axes = []
for P in (X, Y, Z):
    for s in (1, -1):
        axes.append((I2 + s * P) / 2)
mix = sum(kron(a, a.conj()) for a in axes) / 6
assert np.allclose(mix, rho_iso(1 / 3))
print("  rho_{1/3} = (1/6) sum_{n in ±x,±y,±z} |n><n| (x) |n*><n*|  — сепарабельно")
for p in [0, 0.1, 0.2, 1 / 3]:
    q = 3 * p          # rho_p = q rho_{1/3} + (1-q) I/4 при p <= 1/3
    assert np.allclose(rho_iso(p), q * rho_iso(1 / 3) + (1 - q) * I4 / 4)
print("  rho_p при p <= 1/3 — смесь rho_{1/3} и I/4, тоже сепарабельно")
# шар Гурвица–Барнума: ||rho - I/d||_2 <= 1/sqrt(d(d-1)) => сепарабельно; rho_{1/3} на его границе
rad = 1 / np.sqrt(4 * 3)
assert np.isclose(np.linalg.norm(rho_iso(1 / 3) - I4 / 4), rad)
for _ in range(2000):
    D = random_rho_hs(4) - I4 / 4
    r = I4 / 4 + D * rad / np.linalg.norm(D) * rng.uniform() ** 0.25
    assert np.min(np.linalg.eigvalsh(ptrans(r, 2, 2))) > -1e-12
print(f"  шар радиуса 1/sqrt(12) = {rad:.4f} вокруг I/4 сепарабелен (2000 точек — все PPT); rho_{{1/3}} на его границе")
# rho_W и rho_p локально эквивалентны: (I (x) XZ)|Phi+> = ±|Psi->
U = kron(I2, X @ Z)
assert np.isclose(abs(np.vdot(Psi_m, U @ Phi_p)), 1)
for p in np.linspace(0, 1, 6):
    assert np.allclose(U @ rho_iso(p) @ dag(U), rho_W(p))
print("  rho_W(p) = (I (x) XZ) rho_p (I (x) XZ)^dag: для кубитов это одно семейство")

# ---------------------------------------------------------------------------
section("2. Критерий PPT")
SWAP = np.eye(4)[[0, 2, 1, 3]]
assert np.allclose(ptrans(proj(Phi_p), 2, 2), SWAP / 2)
print("  (|Phi+><Phi+|)^Gamma = SWAP/2, собственные значения", np.round(np.linalg.eigvalsh(SWAP / 2), 3))
for p in np.linspace(0, 1, 11):
    for r in (rho_iso(p), rho_W(p)):
        mn = np.min(np.linalg.eigvalsh(ptrans(r, 2, 2)))
        assert np.isclose(mn, (1 - 3 * p) / 4)
print("  rho_p и rho_W: lambda_min(rho^Gamma) = (1 - 3p)/4 < 0  <=>  p > 1/3")
for _ in range(200):
    r = random_rho_hs(6)
    ea = np.linalg.eigvalsh(ptrans(r, 2, 3, 0)); eb = np.linalg.eigvalsh(ptrans(r, 2, 3, 1))
    assert np.allclose(ea, eb)                                     # T_A и T_B: один спектр
    UA, UB = (np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))[0] for d in (2, 3))
    W = kron(UA, UB)
    assert np.allclose(np.linalg.eigvalsh(ptrans(W @ r @ dag(W), 2, 3)), eb)   # локальные унитарные
print("  спектр rho^Gamma не зависит от выбора подсистемы, базиса и локальных унитарных (200 состояний 2x3)")
for dA, dB in [(2, 2), (2, 3), (3, 3), (2, 4)]:
    for _ in range(100):
        assert np.min(np.linalg.eigvalsh(ptrans(random_separable(dA, dB), dA, dB))) > -1e-12
print("  случайные сепарабельные состояния 2x2, 2x3, 3x3, 2x4 — все PPT")

# ---------------------------------------------------------------------------
section("3. Доля PPT (= сепарабельных) двухкубитных состояний по мере Гильберта–Шмидта")
N = 200_000
G = rng.normal(size=(N, 4, 4)) + 1j * rng.normal(size=(N, 4, 4))
R = G @ np.conj(np.transpose(G, (0, 2, 1)))
R /= np.trace(R, axis1=1, axis2=2)[:, None, None]
RG = R.reshape(N, 2, 2, 2, 2).transpose(0, 1, 4, 3, 2).reshape(N, 4, 4)
frac = np.mean(np.linalg.eigvalsh(RG)[:, 0] >= 0)
err = np.sqrt(frac * (1 - frac) / N)
print(f"  {N} состояний: доля PPT = {frac:.4f} ± {err:.4f};  8/33 = {8 / 33:.4f}")
assert abs(frac - 8 / 33) < 4 * err

# ---------------------------------------------------------------------------
section("4. Свидетели запутанности")
W_phi = 0.5 * I4 - proj(Phi_p)
W_psi = 0.5 * I4 - proj(Psi_m)
# на произведениях <W> >= 0: максимум |<Phi+|ab>|^2 = 1/2 (наибольший коэффициент Шмидта)
def min_on_products(W, dA=2, dB=2):
    def f(t):
        a = t[:dA] + 1j * t[dA:2 * dA]; b = t[2 * dA:2 * dA + dB] + 1j * t[2 * dA + dB:]
        v = kron(a / np.linalg.norm(a), b / np.linalg.norm(b))
        return np.real(np.vdot(v, W @ v))
    return min(minimize(f, rng.normal(size=2 * (dA + dB)), method="BFGS").fun for _ in range(20))
for W in (W_phi, W_psi):
    m = min_on_products(W)
    assert m > -1e-9 and np.isclose(m, 0, atol=1e-7)
print("  min по произведениям <ab|W|ab> = 0 для W = I/2 - |Phi+><Phi+| и I/2 - |Psi-><Psi-|")
# разложение по Паули: измеряется тремя локальными настройками
assert np.allclose(proj(Phi_p), (kron(I2, I2) + kron(X, X) - kron(Y, Y) + kron(Z, Z)) / 4)
assert np.allclose(proj(Psi_m), (kron(I2, I2) - kron(X, X) - kron(Y, Y) - kron(Z, Z)) / 4)
print("  |Phi+><Phi+| = (II + XX - YY + ZZ)/4,  |Psi-><Psi-| = (II - XX - YY - ZZ)/4")
for p in np.linspace(0, 1, 11):
    assert np.isclose(np.trace(W_psi @ rho_W(p)).real, 0.5 - (1 + 3 * p) / 4)
    assert np.trace(W_phi @ rho_W(p)).real >= 0.25 - 1e-12
print("  <I/2 - Psi-> на rho_W = 1/2 - (1+3p)/4 < 0 при p > 1/3; <I/2 - Phi+> >= 1/4 при всех p")
# свидетель из PPT: W = (|v><v|)^Gamma, v — собственный вектор rho^Gamma с отрицательным значением
for _ in range(100):
    r = random_rho_hs(4)
    w, V = np.linalg.eigh(ptrans(r, 2, 2))
    if w[0] < 0:
        Wv = ptrans(proj(V[:, 0]), 2, 2)
        assert np.isclose(np.trace(Wv @ r).real, w[0])
        for _ in range(20):
            assert np.trace(Wv @ random_separable(2, 2)).real > -1e-12
print("  W = (|v><v|)^Gamma: Tr(W rho) = lambda_min(rho^Gamma) < 0, Tr(W sigma) >= 0 на сепарабельных")

# ---------------------------------------------------------------------------
section("5. Критерий редукции; для двух кубитов он равносилен PPT")
def reduction_ok(r, dA, dB):
    return np.min(np.linalg.eigvalsh(kron(ptrace(r, dA, dB, 0), np.eye(dB)) - r)) > -1e-12
for p in np.linspace(0, 1, 21):
    assert reduction_ok(rho_iso(p), 2, 2) == (p <= 1 / 3 + 1e-12)
cnt = 0
for _ in range(2000):
    r = random_rho_hs(4)
    ppt = np.min(np.linalg.eigvalsh(ptrans(r, 2, 2))) > -1e-12
    assert ppt == reduction_ok(r, 2, 2)
    cnt += ppt
print(f"  rho_A (x) I - rho >= 0 <=> PPT на 2000 случайных двухкубитных состояниях (PPT из них {cnt})")

# ---------------------------------------------------------------------------
section("6. Связанная запутанность: состояние Городецкого 3x3; критерий перестановки")
def horodecki_3x3(a):
    s = np.zeros((9, 9))
    for i in (0, 4, 8):
        for j in (0, 4, 8):
            s[i, j] = a
    for i in (1, 2, 3, 5, 7):
        s[i, i] = a
    b, c = (1 + a) / 2, np.sqrt(1 - a ** 2) / 2
    s[6, 6] = b; s[8, 8] = b; s[6, 8] = s[8, 6] = c
    return s / (8 * a + 1)
for a in [0.01, 0.1, 0.236, 0.5, 0.9, 0.99]:
    s = horodecki_3x3(a)
    assert np.isclose(np.trace(s), 1) and np.min(np.linalg.eigvalsh(s)) > -1e-12
    lmin = np.min(np.linalg.eigvalsh(ptrans(s, 3, 3)))
    ccnr = trace_norm(realign(s, 3, 3)) - 1
    assert lmin > -1e-12 and ccnr > 0
    print(f"  a = {a:5.3f}: lambda_min(rho^Gamma) = {lmin:+.2e} (PPT);  ||R(rho)||_1 - 1 = {ccnr:.2e} > 0 -> запутано")
for _ in range(300):
    assert trace_norm(realign(random_separable(3, 3), 3, 3)) <= 1 + 1e-12
print("  на случайных сепарабельных 3x3: ||R||_1 <= 1")

# ---------------------------------------------------------------------------
section("7. LOCC и мажорирование (теорема Нильсена); катализ")
lam = lambda psi, d: np.linalg.svd(psi.reshape(d, d), compute_uv=False) ** 2
phi4 = np.ones(4) / 4
for _ in range(100):
    x = rng.dirichlet(np.ones(4))
    assert majorized(phi4, x)        # из максимально запутанного получается любое
print("  (1/4,1/4,1/4,1/4) ≺ любой вектор: из |Phi_4> по LOCC получается любое состояние 4x4")
psi = np.array([0.4, 0.4, 0.1, 0.1]); phi = np.array([0.5, 0.25, 0.25, 0])
assert not majorized(psi, phi) and not majorized(phi, psi)
cat = np.array([0.6, 0.4])
assert majorized(np.kron(psi, cat), np.kron(phi, cat))
print("  psi = (.4,.4,.1,.1), phi = (.5,.25,.25,0): несравнимы;  с катализатором (.6,.4): psi(x)c -> phi(x)c")
print(f"  энтропии: E(psi) = {entropy(psi):.3f} > E(phi) = {entropy(phi):.3f} — необходимое условие выполнено, но мало")
assert entropy(psi) > entropy(phi)

# ---------------------------------------------------------------------------
section("8. Concurrence и запутанность формирования (Вуттерс)")
for _ in range(200):
    v = random_state(4)
    rA = ptrace(proj(v), 2, 2, 0)
    C = concurrence(proj(v))
    assert np.isclose(C, 2 * np.sqrt(max(0, np.linalg.det(rA).real)))
    assert np.isclose(C, 2 * abs(v[0] * v[3] - v[1] * v[2]))
    assert np.isclose(eof_wootters(proj(v)), ent_entropy(v, 2, 2))
print("  чистые: C = 2|ad - bc| = 2 sqrt(det rho_A);  h((1+sqrt(1-C^2))/2) = S(rho_A)")
for p in np.linspace(0, 1, 11):
    assert np.isclose(concurrence(rho_iso(p)), max(0, (3 * p - 1) / 2))
print("  rho_p: C = max(0, (3p-1)/2)")
# прямая минимизация средней энтропии по разложениям (K = 6 чистых состояний) для ранга 2
def eof_numeric(r, K=6, tries=6):
    w, V = np.linalg.eigh(r)
    keep = w > 1e-12
    w, V = w[keep], V[:, keep]
    k = len(w)
    base = V * np.sqrt(w)                          # столбцы sqrt(w_j)|e_j>
    def f(t):
        M = (t[:K * k] + 1j * t[K * k:]).reshape(K, k)
        Q, _ = np.linalg.qr(M)                     # изометрия K x k
        tot = 0.0
        for i in range(K):
            v = base @ Q[i]
            pi = np.vdot(v, v).real
            if pi > 1e-14:
                tot += pi * ent_entropy(v / np.sqrt(pi), 2, 2)
        return tot
    return min(minimize(f, rng.normal(size=2 * K * k), method="Nelder-Mead",
                        options=dict(maxiter=20000, xatol=1e-9, fatol=1e-11)).fun for _ in range(tries))
for _ in range(3):
    a, b = random_state(4), random_state(4)
    r = 0.6 * proj(a) + 0.4 * proj(b)
    en, ew = eof_numeric(r), eof_wootters(r)
    print(f"  ранг 2: минимум по разложениям {en:.5f}, формула Вуттерса {ew:.5f}")
    assert en >= ew - 1e-6 and abs(en - ew) < 2e-3

# ---------------------------------------------------------------------------
section("9. Negativity и логарифмическая negativity")
for _ in range(200):
    v = random_state(4)
    s = np.linalg.svd(v.reshape(2, 2), compute_uv=False)
    assert np.isclose(negativity(proj(v), 2, 2), s[0] * s[1])
    assert np.isclose(2 * negativity(proj(v), 2, 2), concurrence(proj(v)))
    assert np.isclose(log_negativity(proj(v), 2, 2), 2 * np.log2(s[0] + s[1]))
print("  чистые: N = sqrt(l1 l2) = C/2, E_N = 2 log2(sqrt l1 + sqrt l2)")
for p in np.linspace(0, 1, 11):
    assert np.isclose(negativity(rho_iso(p), 2, 2), max(0, (3 * p - 1) / 4))
print("  rho_p: N = max(0, (3p-1)/4)")
worst = 0
for _ in range(2000):
    r = random_rho_hs(4)
    gap = concurrence(r) - 2 * negativity(r, 2, 2)
    assert gap > -1e-9
    worst = max(worst, gap)
print(f"  смешанные: 2N <= C на 2000 случайных состояниях (наибольший разрыв {worst:.3f})")
# монотонность: локальный канал (деполяризация Боба) не увеличивает N и C
def depol_B(r, q):
    return (1 - q) * r + q * kron(ptrace(r, 2, 2, 0), I2 / 2)
for _ in range(200):
    r = random_rho_hs(4); q = rng.uniform()
    assert negativity(depol_B(r, q), 2, 2) <= negativity(r, 2, 2) + 1e-12
    assert concurrence(depol_B(r, q)) <= concurrence(r) + 1e-9
print("  локальная деполяризация не увеличивает N и C (200 случаев)")
assert np.isclose(log_negativity(proj(Phi_p), 2, 2), 1) and np.isclose(eof_wootters(proj(Phi_p)), 1)
print("  нормировка: E_N(Phi+) = E_F(Phi+) = 1")

# ---------------------------------------------------------------------------
section("10. Сводка по rho_p и изотропные состояния в размерности d")
print("    p     C       N      E_F    E_N    S_max")
for p in [0.2, 1 / 3, 0.5, 1 / np.sqrt(2), 0.9, 1.0]:
    r = rho_iso(p)
    T = np.array([[np.trace(r @ kron(A, B)).real for B in (X, Y, Z)] for A in (X, Y, Z)])
    t = np.sort(np.linalg.eigvalsh(T.T @ T))[::-1]
    print(f"  {p:.3f}  {concurrence(r):.3f}  {negativity(r, 2, 2):.3f}  {eof_wootters(r):.3f}"
          f"  {log_negativity(r, 2, 2):.3f}  {2 * np.sqrt(t[0] + t[1]):.3f}")
for d in [2, 3, 4]:
    Om = sum(kron(np.eye(d)[i], np.eye(d)[i]) for i in range(d)) / np.sqrt(d)
    for p in np.linspace(0, 1, 41):
        r = p * proj(Om) + (1 - p) * np.eye(d * d) / d ** 2
        ppt = np.min(np.linalg.eigvalsh(ptrans(r, d, d))) > -1e-12
        assert ppt == (p <= 1 / (d + 1) + 1e-12)
print("  изотропные d x d: PPT <=> p <= 1/(d+1) (d = 2, 3, 4); для них PPT = сепарабельность")

# ---------------------------------------------------------------------------
section("11. Многочастичная запутанность: свидетель GHZ")
GHZ = (kron(ket0, ket0, ket0) + kron(ket1, ket1, ket1)) / np.sqrt(2)
Wst = (kron(ket1, ket0, ket0) + kron(ket0, ket1, ket0) + kron(ket0, ket0, ket1)) / np.sqrt(3)
# максимум перекрытия с бисепарабельными чистыми = max по разрезам наибольшего коэфф. Шмидта^2
def max_schmidt2(psi):
    cuts = [(2, 4, psi), (2, 4, psi.reshape(2, 2, 2).transpose(1, 0, 2).reshape(8)),
            (2, 4, psi.reshape(2, 2, 2).transpose(2, 0, 1).reshape(8))]
    return max(np.linalg.svd(v.reshape(a, b), compute_uv=False)[0] ** 2 for a, b, v in cuts)
assert np.isclose(max_schmidt2(GHZ), 0.5)
assert np.isclose(max_schmidt2(Wst), 2 / 3)
print("  max перекрытие с бисепарабельными: GHZ — 1/2, W — 2/3")
print("  => свидетели подлинной трёхчастичной запутанности: I/2 - |GHZ><GHZ|, (2/3) I - |W><W|")
for p in [0.4, 3 / 7, 0.5, 0.8]:
    r = p * proj(GHZ) + (1 - p) * np.eye(8) / 8
    F = np.vdot(GHZ, r @ GHZ).real
    print(f"  p = {p:.3f}: F_GHZ = {F:.3f} -> {'подлинно запутано' if F > 0.5 else 'свидетель молчит'}")
assert np.isclose((3 / 7) + (1 - 3 / 7) / 8, 0.5)

print("\nВсе проверки пройдены.")
