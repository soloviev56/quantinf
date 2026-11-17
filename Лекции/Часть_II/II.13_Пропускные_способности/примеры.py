"""
Лекция II.13. Пропускные способности квантовых каналов: информация Холево
канала chi(Phi) (оптимизация по ансамблям), C = 1 - h(p/2) для деполяризующего
канала, проверка аддитивности на Delta x Delta, минимальная выходная энтропия;
когерентная информация и Q^(1); деградируемость стирающего канала и
амплитудного затухания (явные деградирующие каналы), симметричный канал
стирания 1/2; деполяризующий канал: хэш-граница, оценка сверху через
клонирование (клонер Бужека–Хиллери), суперактивность когерентной информации
на кодах повторения (DiVincenzo–Shor–Smolin); пропускная способность с
поддержкой запутанности C_E = max I(R:B); сводная таблица; бозонный канал с
потерями: C, C_E, Q через гауссовы ковариационные матрицы, граница PLOB.

Соглашения (Лекции/Обозначения.md, §12, §21, §23–§25): log = log2;
Delta_p(rho) = (1-p) rho + p I/2; D_p — дефазировка; A_gamma — амплитудное
затухание; стирающий канал Phi^er_eps (выход в C^3, |e> = |2>);
I_c(rho, Phi) = S(Phi(rho)) - S(Phi^c(rho)); I(R:B) = S(rho) + I_c;
g(N) = (N+1) log(N+1) - N log N; дисперсия вакуума 1/2.

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
from math import log2, sqrt, pi, comb
import numpy as np
from scipy.optimize import minimize, brentq, minimize_scalar
from scipy.special import gammaln

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


def g(N):
    return 0.0 if N <= 0 else (N + 1) * log2(N + 1) - N * log2(N)


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
PAULI = [I2, X, Y, Z]


def apply(Ks, rho):
    return sum(K @ rho @ K.conj().T for K in Ks)


def compl(Ks, rho):
    """Дополнительный канал (II.1, (4.4)): [Phi^c(rho)]_{jk} = Tr(K_j rho K_k^+)."""
    n = len(Ks)
    return np.array([[np.trace(Ks[j] @ rho @ Ks[k].conj().T) for k in range(n)] for j in range(n)])


def Ic(Ks, rho):
    return S(apply(Ks, rho)) - S(compl(Ks, rho))


def IRB(Ks, rho):
    return S(rho) + Ic(Ks, rho)


def bloch(r):
    return 0.5 * (I2 + r[0] * X + r[1] * Y + r[2] * Z)


def ket_from_angles(th, ph):
    return np.array([np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)])


# --- каналы кубита (Краус) ---
def K_dep(p):          # Delta_p = (1-p) rho + p I/2 = (1 - 3p/4) rho + p/4 (X.X + Y.Y + Z.Z)
    return [sqrt(1 - 3 * p / 4) * I2] + [sqrt(p / 4) * P for P in (X, Y, Z)]


def K_deph(p):
    return [sqrt(1 - p) * I2, sqrt(p) * Z]


def K_ad(gam):
    return [np.array([[1, 0], [0, sqrt(1 - gam)]], complex), np.array([[0, sqrt(gam)], [0, 0]], complex)]


def K_er(eps):
    """Стирающий канал: C^2 -> C^3, |e> = |2>."""
    K0 = np.zeros((3, 2), complex); K0[0, 0] = K0[1, 1] = sqrt(1 - eps)
    K1 = np.zeros((3, 2), complex); K1[2, 0] = sqrt(eps)
    K2 = np.zeros((3, 2), complex); K2[2, 1] = sqrt(eps)
    return [K0, K1, K2]


def er_out(eps, rho):
    out = np.zeros((3, 3), complex)
    out[:2, :2] = (1 - eps) * rho
    out[2, 2] = eps * np.trace(rho)
    return out


# --- информация Холево канала: оптимизация по ансамблям из m чистых состояний ---
def chi_channel(Ks, m=4, starts=8, dim=2):
    best = -1
    npar = m * (2 * dim) + m

    def unpack(th):
        vecs = th[:m * 2 * dim].reshape(m, 2 * dim)
        vecs = vecs[:, :dim] + 1j * vecs[:, dim:]
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        w = np.exp(th[m * 2 * dim:] - th[m * 2 * dim:].max())
        return w / w.sum(), vecs

    def f(th):
        ps, vecs = unpack(th)
        outs = [apply(Ks, np.outer(v, v.conj())) for v in vecs]
        avg = sum(p * o for p, o in zip(ps, outs))
        return -(S(avg) - sum(p * S(o) for p, o in zip(ps, outs)))

    for _ in range(starts):
        res = minimize(f, rng.normal(size=npar), method="L-BFGS-B", options=dict(maxiter=4000))
        best = max(best, -res.fun)
    return best


def max_over_qubit_inputs(fun, starts=6):
    """max_rho fun(rho) по шару Блоха (параметризация r = tanh|u| u/|u|)."""
    best = -np.inf

    def rho_of(u):
        n = np.linalg.norm(u)
        r = np.zeros(3) if n < 1e-12 else np.tanh(n) * u / n
        return bloch(r)

    for s in range(starts):
        u0 = rng.normal(size=3) * (0.0 if s == 0 else 1.0)
        res = minimize(lambda u: -fun(rho_of(u)), u0, method="Nelder-Mead",
                       options=dict(xatol=1e-9, fatol=1e-12, maxiter=4000))
        best = max(best, -res.fun)
    return best


# ===========================================================================
section("1. Информация Холево канала и классическая пропускная способность")
CHI = {}
for p in (0.1, 0.3, 0.5):
    c_num = chi_channel(K_dep(p))
    c_th = 1 - h(p / 2)
    CHI[("dep", p)] = c_num
    print(f"  Delta_{p}: chi (оптимизация, 4 состояния) = {c_num:.6f};  1 - h(p/2) = {c_th:.6f}")
    assert abs(c_num - c_th) < 1e-6
# амплитудное затухание: ортогональный вход {|0>,|1>} против оптимального ансамбля
for gam in (0.3, 0.6):
    c_num = chi_channel(K_ad(gam))
    c_orth = -minimize_scalar(lambda q: -(h(q * (1 - gam)) - q * h(gam)) if 0 < q < 1 else 0,
                              bounds=(1e-9, 1 - 1e-9), method="bounded").fun
    CHI[("ad", gam)] = c_num
    print(f"  A_{gam}: chi = {c_num:.5f};  только {{|0>,|1>}}: {c_orth:.5f}  (неортогональные входы лучше)")
    assert c_num > c_orth + 1e-4
# дефазировка: chi = 1 (базис Z не портится)
c_dph = chi_channel(K_deph(0.3))
assert abs(c_dph - 1) < 1e-6
# стирание: chi = 1 - eps
for eps in (0.25, 0.5):
    Ks = K_er(eps)
    c_er = chi_channel(Ks)
    CHI[("er", eps)] = c_er
    assert abs(c_er - (1 - eps)) < 1e-6
print(f"  D_0.3: chi = {c_dph:.6f};  стирание eps = 0.25, 0.5: chi = {CHI[('er', 0.25)]:.6f}, {CHI[('er', 0.5)]:.6f} = 1 - eps")

# минимальная выходная энтропия деполяризующего канала и запутанные входы
p = 0.3
Smin = h(p / 2)
phi = np.array([1, 0, 0, 1]) / sqrt(2)
Kdd = [np.kron(A, B) for A in K_dep(p) for B in K_dep(p)]
S_bell = S(apply(Kdd, np.outer(phi, phi)))
S_prod = S(apply(Kdd, np.kron(np.diag([1.0, 0]), np.diag([1.0, 0]))))
print(f"  Delta_0.3: S_min = h(p/2) = {Smin:.4f}; на Delta x Delta: произведение {S_prod:.4f} = 2 S_min, "
      f"Белл {S_bell:.4f} > 2 S_min")
assert abs(S_prod - 2 * Smin) < 1e-10 and S_bell > 2 * Smin
# аддитивность chi для Delta x Delta: оптимизация по ансамблям из 4 и 6 двухкубитных чистых состояний
c2 = max(chi_channel(Kdd, m=4, starts=6, dim=4), chi_channel(Kdd, m=6, starts=4, dim=4))
print(f"  chi(Delta_0.3 x Delta_0.3) (оптимизация, в т.ч. запутанные входы) = {c2:.6f};  2 chi(Delta_0.3) = {2 * (1 - h(0.15)):.6f}")
assert c2 < 2 * (1 - h(0.15)) + 1e-6 and c2 > 2 * (1 - h(0.15)) - 1e-4
CHI2 = c2

# ===========================================================================
section("2. Когерентная информация; стирающий канал и деградируемость")
# проверка I_c = S(B) - S(RB) через очищение
rho = bloch([0.2, -0.3, 0.4])
w, V = np.linalg.eigh(rho)
psi_RA = sum(np.sqrt(w[i]) * np.kron(V[:, i].conj(), V[:, i]) for i in range(2))   # R x A
Ks = K_ad(0.3)
RB = sum(np.kron(I2, K) @ np.outer(psi_RA, psi_RA.conj()) @ np.kron(I2, K).conj().T for K in Ks)
SB = S(np.einsum("ijik->jk", RB.reshape(2, 2, 2, 2)))
assert abs((SB - S(RB)) - Ic(Ks, rho)) < 1e-10
assert abs(S(np.einsum("ijkj->ik", RB.reshape(2, 2, 2, 2))) - S(rho)) < 1e-10
print("  I_c(rho, Phi) = S(Phi(rho)) - S(Phi^c(rho)) = S(B) - S(RB) = -S(R|B): проверено")
Q1 = {}
for eps in (0.1, 0.25, 0.4, 0.5, 0.6):
    q1 = max(0.0, max_over_qubit_inputs(lambda r: Ic(K_er(eps), r)))
    Q1[("er", eps)] = q1
    assert abs(q1 - max(0, 1 - 2 * eps)) < 1e-6
print("  стирание: Q^(1) = max(0, 1 - 2 eps): " + ", ".join(f"eps = {e}: {Q1[('er', e)]:.4f}" for e in (0.1, 0.25, 0.4, 0.5, 0.6)))
# дополнительный канал стирания — стирание с 1 - eps (с переименованием базиса окружения)
perm = np.zeros((3, 3)); perm[2, 0] = perm[0, 1] = perm[1, 2] = 1      # |0>_E -> |e>, |1>_E -> |0>, |2>_E -> |1>
for eps in (0.2, 0.5, 0.7):
    for _ in range(5):
        r = bloch(rng.uniform(-0.57, 0.57, 3))
        assert np.allclose(perm @ compl(K_er(eps), r) @ perm.T, er_out(1 - eps, r))
# деградирующий канал при eps < 1/2: стирание с eps' = (1-2eps)/(1-eps) на выходе C^3
for eps in (0.1, 0.3, 0.45):
    e2 = (1 - 2 * eps) / (1 - eps)
    Kd = [np.diag([sqrt(1 - e2), sqrt(1 - e2), 1.0]).astype(complex)]
    for j in range(2):
        M = np.zeros((3, 3), complex); M[2, j] = sqrt(e2); Kd.append(M)
    assert np.allclose(sum(K.conj().T @ K for K in Kd), np.eye(3))
    for _ in range(5):
        r = bloch(rng.uniform(-0.57, 0.57, 3))
        assert np.allclose(apply(Kd, er_out(eps, r)), er_out(1 - eps, r))
print("  Phi^c = Phi^er_{1-eps}; при eps < 1/2 деградирующий канал — стирание с eps' = (1-2eps)/(1-eps): проверено")
print("  eps = 1/2: Phi^c = Phi (симметричный канал) => Q = 0 (иначе Боб и Ева оба восстановили бы состояние)")

# ===========================================================================
section("3. Амплитудное затухание и дефазировка: деградируемые каналы")
for gam in (0.1, 0.3, 0.45):
    g2 = (1 - 2 * gam) / (1 - gam)
    for _ in range(5):
        r = bloch(rng.uniform(-0.57, 0.57, 3))
        assert np.allclose(apply(K_ad(g2), apply(K_ad(gam), r)), compl(K_ad(gam), r))
print("  A_gamma^c = A_{1-gamma}; деградирующий канал A_{(1-2g)/(1-g)} при gamma < 1/2: проверено")


def Q_ad_formula(gam):
    if gam >= 0.5:
        return 0.0
    return -minimize_scalar(lambda q: -(h((1 - gam) * q) - h(gam * q)), bounds=(1e-9, 1 - 1e-9),
                            method="bounded", options=dict(xatol=1e-10)).fun


QAD = {}
for gam in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
    q1 = max(0.0, max_over_qubit_inputs(lambda r: Ic(K_ad(gam), r)))
    QAD[gam] = q1
    assert abs(q1 - Q_ad_formula(gam)) < 1e-6
print("  A_gamma: Q = max_q [h((1-g)q) - h(g q)] (0 при gamma >= 1/2): " +
      ", ".join(f"{gam}: {QAD[gam]:.4f}" for gam in QAD))
QDPH = {}
for p in (0.05, 0.1, 0.2, 0.3):
    q1 = max_over_qubit_inputs(lambda r: Ic(K_deph(p), r))
    QDPH[p] = q1
    assert abs(q1 - (1 - h(p))) < 1e-6
print("  D_p: Q = 1 - h(p): " + ", ".join(f"{p}: {QDPH[p]:.4f}" for p in QDPH))

# ===========================================================================
section("4. Деполяризующий канал: хэш-граница, оценка сверху, суперактивность")


def hashing(p):
    return 1 - H([1 - 3 * p / 4, p / 4, p / 4, p / 4])


p_h = brentq(hashing, 0.1, 0.4)
print(f"  хэш-граница Q >= 1 - H(1-3p/4, p/4, p/4, p/4) = 0 при p = {p_h:.4f} "
      f"(в конвенции QEC, X/Y/Z по q/3: q = 3p/4 = {3 * p_h / 4:.4f})")
assert abs(3 * p_h / 4 - 0.1893) < 1e-3
for p in (0.1, 0.2):
    q1 = max_over_qubit_inputs(lambda r: Ic(K_dep(p), r))
    assert abs(q1 - hashing(p)) < 1e-6
print("  Q^(1)(Delta_p) = хэш-граница (максимум — на I/2): проверено при p = 0.1, 0.2")
# клонер Бужека–Хиллери: оба выхода — Delta_{1/3}(rho)
U = np.zeros((8, 2), complex)        # |psi>_A -> копия1 x копия2 x анцилла
def kidx(a, b, c): return 4 * a + 2 * b + c
U[kidx(0, 0, 0), 0] = sqrt(2 / 3); U[kidx(0, 1, 1), 0] = U[kidx(1, 0, 1), 0] = sqrt(1 / 6)
U[kidx(1, 1, 1), 1] = sqrt(2 / 3); U[kidx(0, 1, 0), 1] = U[kidx(1, 0, 0), 1] = sqrt(1 / 6)
assert np.allclose(U.conj().T @ U, np.eye(2))
for _ in range(10):
    r = bloch(rng.uniform(-0.57, 0.57, 3))
    out = (U @ r @ U.conj().T).reshape(2, 2, 2, 2, 2, 2)
    c1 = np.einsum("abcdbc->ad", out)
    c2_ = np.einsum("abcaec->be", out)
    target = (2 / 3) * r + (1 / 3) * I2 / 2
    assert np.allclose(c1, target) and np.allclose(c2_, target)
print("  клонер Бужека–Хиллери: обе копии = Delta_{1/3}(rho) => Q(Delta_p) = 0 при p >= 1/3 (q >= 1/4)")
# Delta_p при p > 1/3 — композиция Delta_{1/3} с ещё одним деполяризующим каналом
pp = 0.5
p2 = 1 - (1 - pp) / (1 - 1 / 3)
r = bloch([0.3, 0.1, -0.5])
dep = lambda q, x: (1 - q) * x + q * I2 / 2
assert np.allclose(dep(p2, dep(1 / 3, r)), dep(pp, r))
# простая оценка сверху: Q <= C_E / 2
def CE_pauli(p):
    return 2 - H([1 - 3 * p / 4, p / 4, p / 4, p / 4])


print(f"  оценка Q <= C_E/2 = 1 - H(...)/2: при p = {p_h:.4f} это {CE_pauli(p_h) / 2:.4f}")
# суперактивность когерентной информации: коды повторения (кошачьи), DiVincenzo–Shor–Smolin


def dep_on_qubit(rho, p, j, n):
    """Delta_p на кубите j из n (rho — на n кубитах)."""
    d = 2 ** n
    r = rho.reshape([2] * (2 * n))
    out = (1 - p) * rho
    # p * (Tr_j rho) x I/2 на месте j
    tr = np.trace(r, axis1=j, axis2=j + n)                  # 2n-2 индексов
    tr = np.expand_dims(np.expand_dims(tr, j), j + n)
    full = tr * (np.eye(2).reshape([2 if k in (j, j + n) else 1 for k in range(2 * n)]) / 2)
    return out + p * full.reshape(d, d)


def cat_coherent_info(n, p):
    """I_c для входа (|0>_R|0^n> + |1>_R|1^n>)/sqrt2 через Delta_p^{(x)n}; R — первый кубит."""
    psi = np.zeros(2 ** (n + 1)); psi[0] = psi[-1] = 1 / sqrt(2)
    rho = np.outer(psi, psi)
    for j in range(1, n + 1):
        rho = dep_on_qubit(rho, p, j, n + 1)
    SB = S(np.einsum("ijik->jk", rho.reshape(2, 2 ** n, 2, 2 ** n)))
    return (SB - S(rho)) / n


SUPER = {}
p_test = p_h + 0.001
for n in (1, 3, 5, 7):
    SUPER[n] = cat_coherent_info(n, p_test)
print(f"  p = {p_test:.4f} (чуть выше хэш-границы): I_c/n для кошачьих кодов n = 1, 3, 5, 7: " +
      ", ".join(f"{SUPER[n]:+.5f}" for n in SUPER))
assert SUPER[1] < 0 and max(SUPER[n] for n in (3, 5, 7)) > 0
# порог положительности I_c/n для лучшего n
p_cat = {n: brentq(lambda q: cat_coherent_info(n, q), p_h - 0.02, p_h + 0.05) for n in (3, 5, 7)}
print("  порог I_c/n = 0: " + ", ".join(f"n = {n}: p = {p_cat[n]:.4f} (q = {3 * p_cat[n] / 4:.4f})" for n in p_cat)
      + f"; хэш: q = {3 * p_h / 4:.4f}")
assert max(p_cat.values()) > p_h

# ===========================================================================
section("5. Пропускная способность с поддержкой запутанности C_E = max I(R:B)")
CE = {}
for p in (0.1, 0.3, 0.5):
    v = max_over_qubit_inputs(lambda r: IRB(K_dep(p), r))
    CE[("dep", p)] = v
    assert abs(v - CE_pauli(p)) < 1e-6
for eps in (0.25, 0.5):
    v = max_over_qubit_inputs(lambda r: IRB(K_er(eps), r))
    CE[("er", eps)] = v
    assert abs(v - 2 * (1 - eps)) < 1e-6
for gam in (0.3, 0.6):
    v = max_over_qubit_inputs(lambda r: IRB(K_ad(gam), r))
    f_ad = -minimize_scalar(lambda q: -(h(q) + h((1 - gam) * q) - h(gam * q)), bounds=(1e-9, 1 - 1e-9),
                            method="bounded", options=dict(xatol=1e-10)).fun
    CE[("ad", gam)] = v
    assert abs(v - f_ad) < 1e-6
for p in (0.3,):
    v = max_over_qubit_inputs(lambda r: IRB(K_deph(p), r))
    CE[("dph", p)] = v
    assert abs(v - (2 - h(p))) < 1e-6
print("  Delta_p: C_E = 2 - H(1-3p/4, p/4 x3): " + ", ".join(f"p = {p}: {CE[('dep', p)]:.4f}" for p in (0.1, 0.3, 0.5)))
print(f"  стирание: C_E = 2(1-eps): {CE[('er', 0.25)]:.4f}, {CE[('er', 0.5)]:.4f};  "
      f"A_gamma: C_E = max_q[h(q) + h((1-g)q) - h(gq)]: {CE[('ad', 0.3)]:.4f}, {CE[('ad', 0.6)]:.4f};  D_0.3: {CE[('dph', 0.3)]:.4f}")
# отношение C_E / C для сильного шума
RATIO = {}
for p in (0.5, 0.9, 0.99):
    RATIO[p] = CE_pauli(p) / (1 - h(p / 2))
print("  C_E / C для Delta_p: " + ", ".join(f"p = {p}: {RATIO[p]:.3f}" for p in RATIO) + "  (-> 3 = d + 1 при p -> 1)")
assert 2.9 < RATIO[0.99] < 3.0
# асимптотика при lambda = 1 - p -> 0: C ~ lambda^2/(2 ln 2), C_E ~ 3 lambda^2/(2 ln 2)
lam_ = 1e-3
assert abs((1 - h((1 - lam_) / 2)) / (lam_ ** 2 / (2 * np.log(2))) - 1) < 1e-3
assert abs(CE_pauli(1 - lam_) / (3 * lam_ ** 2 / (2 * np.log(2))) - 1) < 1e-2
# сверхплотное кодирование / телепортация: Q_E = C_E / 2
assert abs(max_over_qubit_inputs(lambda r: IRB([I2], r)) - 2) < 1e-9

# ===========================================================================
section("6. Сводная таблица (кубитные каналы)")
TABLE = []
rows = [("Delta_0.1", K_dep(0.1), 1 - h(0.05), hashing(0.1), CE_pauli(0.1)),
        ("Delta_0.3", K_dep(0.3), 1 - h(0.15), hashing(0.3), CE_pauli(0.3)),
        ("D_0.1", K_deph(0.1), 1.0, 1 - h(0.1), 2 - h(0.1)),
        ("A_0.2", K_ad(0.2), None, Q_ad_formula(0.2), None),
        ("стирание 0.25", K_er(0.25), 0.75, 0.5, 1.5)]
print("  канал           C (chi)    Q        C_E")
for name, Ks, Cth, Qth, CEth in rows:
    c = chi_channel(Ks, starts=6)
    ce = max_over_qubit_inputs(lambda r: IRB(Ks, r))
    if Cth is not None:
        assert abs(c - Cth) < 1e-6
    if CEth is not None:
        assert abs(ce - CEth) < 1e-6
    TABLE.append((name, c, Qth, ce))
    qs = f"{max(0.0, Qth):.4f}" + (" (хэш < 0)" if Qth < 0 else "")
    print(f"  {name:14s}  {c:.4f}    {qs:16s}  {ce:.4f}" + ("   (для Delta — только нижняя граница Q)" if name.startswith("Delta") else ""))
    assert Qth <= c + 1e-9 <= ce + 1e-9

# ===========================================================================
section("7. Бозонный канал с потерями: C, C_E, Q, граница PLOB")


def symp_eigs(V):
    m = V.shape[0] // 2
    Om = np.kron(np.eye(m), np.array([[0, 1], [-1, 0]]))
    ev = np.abs(np.linalg.eigvals(1j * Om @ V))
    return np.sort(ev)[::2]


def S_gauss(V):
    return sum(g(nu - 0.5) for nu in symp_eigs(V))


def tmsv_through_loss(N, eta):
    ch2r = 2 * N + 1
    sh2r = sqrt(ch2r ** 2 - 1)
    V = 0.5 * np.block([[ch2r * np.eye(2), sh2r * Z.real], [sh2r * Z.real, ch2r * np.eye(2)]])
    Xm = np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), sqrt(eta) * np.eye(2)]])
    Ym = np.block([[np.zeros((2, 2)), np.zeros((2, 2))], [np.zeros((2, 2)), (1 - eta) / 2 * np.eye(2)]])
    return Xm @ V @ Xm.T + Ym


LOSS = {}
for N in (1.0, 5.0):
    for eta in (0.3, 0.7, 0.9):
        V = tmsv_through_loss(N, eta)
        SR, SBo, SRB = S_gauss(V[:2, :2]), S_gauss(V[2:, 2:]), S_gauss(V)
        assert abs(SR - g(N)) < 1e-9 and abs(SBo - g(eta * N)) < 1e-9 and abs(SRB - g((1 - eta) * N)) < 1e-8
        CEv, Icv = SR + SBo - SRB, SBo - SRB
        LOSS[(N, eta)] = (g(eta * N), CEv, max(0, Icv))
print("   N    eta   C = g(eta N)   C_E = g(N)+g(eta N)-g((1-eta)N)   Q_N = max(0, g(eta N)-g((1-eta)N))")
for (N, eta), (c, ce, q) in LOSS.items():
    print(f"  {N:4.1f}  {eta:.1f}     {c:.4f}            {ce:.4f}                          {q:.4f}")
# при N -> бесконечности Q_N -> log2(eta/(1-eta))
for eta in (0.7, 0.9):
    qN = g(eta * 1e6) - g((1 - eta) * 1e6)
    assert abs(qN - log2(eta / (1 - eta))) < 1e-5
print(f"  без ограничения энергии: Q = max(0, log2(eta/(1-eta))): eta = 0.7: {log2(0.7 / 0.3):.4f}, "
      f"eta = 0.9: {log2(9):.4f}; Q = 0 при eta <= 1/2")
# среднее гауссово распределённых когерентных состояний — тепловое (chi = g(eta N)): проверка в базисе Фока
Nf, NF = 0.8, 40
nn = np.arange(NF)
xs = np.linspace(-6, 6, 161)
dx = xs[1] - xs[0]
avg = np.zeros((NF, NF), complex)
for xr in xs:
    for xi in xs:
        al = (xr + 1j * xi) * sqrt(Nf)
        wgt = np.exp(-(xr ** 2 + xi ** 2)) / pi * dx * dx        # |alpha|^2 ~ экспоненциальное со средним Nf
        v = np.exp(-abs(al) ** 2 / 2 + nn * np.log(al + 1e-300 + 0j) - 0.5 * gammaln(nn + 1))
        avg += wgt * np.outer(v, v.conj())
therm = np.diag((Nf / (Nf + 1)) ** nn / (Nf + 1))
assert np.allclose(avg, therm, atol=1e-6)
assert abs(S(avg) - g(Nf)) < 1e-4
print(f"  среднее когерентных состояний с гауссовым распределением (N = {Nf}) = тепловое: S = {S(avg):.5f} = g(N) = {g(Nf):.5f}")
# PLOB для волокна 0.2 дБ/км
PLOB = {}
for L in (10, 50, 100, 200, 300):
    eta = 10 ** (-0.2 * L / 10)
    PLOB[L] = (eta, -log2(1 - eta))
    assert -log2(1 - eta) <= eta / np.log(2) / (1 - eta) + 1e-12
print("  граница PLOB -log2(1-eta) (бит на использование канала), волокно 0.2 дБ/км:")
for L, (eta, b) in PLOB.items():
    print(f"    L = {L:3d} км: eta = {eta:.3e}, PLOB = {b:.3e}, 1.44 eta = {1.4427 * eta:.3e}")
assert abs(PLOB[100][1] - 0.01449) < 1e-4

# ===========================================================================
section("8. Ответы к задачам")
print(f"  1: Delta_0.3: C = 1 - h(0.15) = {1 - h(0.15):.4f}; C_E = {CE_pauli(0.3):.4f}; хэш Q >= {max(0, hashing(0.3)):.4f}")
print(f"  2: стирание eps = 0.25: C = 0.75, Q = 0.5, C_E = 1.5")
print(f"  3: A_0.2: Q = {Q_ad_formula(0.2):.4f}; A_0.3: chi = {CHI[('ad', 0.3)]:.4f}")
print(f"  4: D_0.1: Q = 1 - h(0.1) = {1 - h(0.1):.4f}, C_E = {2 - h(0.1):.4f}")
print(f"  5: хэш-граница: p = {p_h:.4f} (q = {3 * p_h / 4:.4f}); клонер: p >= 1/3")
print(f"  6: C_E/C при p = 0.9: {RATIO[0.9]:.3f}")
print(f"  8: N = 5, eta = 0.7: C = {LOSS[(5.0, 0.7)][0]:.4f}, C_E = {LOSS[(5.0, 0.7)][1]:.4f}, Q_N = {LOSS[(5.0, 0.7)][2]:.4f}")
print(f"  9: PLOB при 100 км: {PLOB[100][1]:.4e}; при 200 км: {PLOB[200][1]:.4e}")

print(f"\nВсе проверки пройдены. Время работы: {time.time() - T_START:.1f} с")
