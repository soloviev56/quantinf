"""
Лекция I.16. Вариационные алгоритмы и квантовая линейная алгебра: VQE для H2
(UCC-анзац, кривая диссоциации), правило сдвига параметров, бюджет измерений,
hardware-efficient анзац для TFIM, QAOA для MaxCut (формула p = 1, связь с
адиабатикой), barren plateaus (глобальная и локальная функции стоимости),
алгоритм HHL (полная симуляция малой схемы, роль числа обусловленности).

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
Порядок кубитов: первый множитель — старший (как в N&C и np.kron).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # малые матрицы: многопоточность BLAS только мешает
import numpy as np
from math import pi, sqrt, erf, exp
from functools import reduce
from itertools import product, combinations
from scipy.optimize import minimize
from scipy.linalg import expm

rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H1 = (X + Z) / np.sqrt(2)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


import time
_T0 = time.time()


def section(title):
    print(f"\n[{time.time() - _T0:6.1f} с]\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def kron(*ops):
    return reduce(np.kron, ops, np.array([[1.0 + 0j]]))


def pstr(s):
    return kron(*[PAULI[c] for c in s])


def op_on(n, j, o):
    return kron(*[o if k == j else I2 for k in range(n)])


def pauli_decompose(Hm, n):
    out = {}
    for s in product("IXYZ", repeat=n):
        s = "".join(s)
        c = np.trace(pstr(s) @ Hm) / 2 ** n
        if abs(c) > 1e-12:
            out[s] = c.real
    return out


# ---------------------------------------------------------------------------
# Гамильтониан H2 в STO-3G (как в I.15, раздел 7: интегралы с нуля)
ANG = 1 / 0.529177210903
sto_a = np.array([3.42525091, 0.62391373, 0.16885540])
sto_d = np.array([0.15432897, 0.53532814, 0.44463454])
norm = (2 * sto_a / pi) ** 0.75


def F0(t):
    return 1.0 if t < 1e-12 else 0.5 * sqrt(pi / t) * erf(sqrt(t))


def h2_integrals(R):
    C = [np.array([0., 0, 0]), np.array([0., 0, R])]
    S = np.zeros((2, 2)); T = np.zeros((2, 2)); V = np.zeros((2, 2)); ERI = np.zeros((2, 2, 2, 2))
    prim = [(a, d * nn) for a, d, nn in zip(sto_a, sto_d, norm)]
    for i in range(2):
        for j in range(2):
            AB2 = np.sum((C[i] - C[j]) ** 2)
            for a, ca in prim:
                for b, cb in prim:
                    p = a + b; mu = a * b / p; P = (a * C[i] + b * C[j]) / p
                    K = exp(-mu * AB2); s = (pi / p) ** 1.5 * K
                    S[i, j] += ca * cb * s
                    T[i, j] += ca * cb * mu * (3 - 2 * mu * AB2) * s
                    for Cn in C:
                        V[i, j] += -ca * cb * 2 * pi / p * K * F0(p * np.sum((P - Cn) ** 2))
    for i, j, k, l in product(range(2), repeat=4):
        val = 0.0
        for a, ca in prim:
            for b, cb in prim:
                p = a + b; P = (a * C[i] + b * C[j]) / p
                Kab = exp(-a * b / p * np.sum((C[i] - C[j]) ** 2))
                for c, cc in prim:
                    for d, cd in prim:
                        q = c + d; Q = (c * C[k] + d * C[l]) / q
                        Kcd = exp(-c * d / q * np.sum((C[k] - C[l]) ** 2))
                        val += ca * cb * cc * cd * 2 * pi ** 2.5 / (p * q * sqrt(p + q)) * Kab * Kcd * F0(p * q / (p + q) * np.sum((P - Q) ** 2))
        ERI[i, j, k, l] = val
    return S, T + V, ERI, 1.0 / R


sig_minus = (X + 1j * Y) / 2


def a_op(j, n):
    return kron(*([Z] * j + [sig_minus] + [I2] * (n - j - 1)))


A4 = [a_op(j, 4) for j in range(4)]
NOP = sum(a.conj().T @ a for a in A4)


def h2_hamiltonian(R_ang):
    S, hcore, eri, Enuc = h2_integrals(R_ang * ANG)
    Cmo = np.array([[1, 1], [1, -1]], float)
    Cmo[:, 0] /= sqrt(2 * (1 + S[0, 1])); Cmo[:, 1] /= sqrt(2 * (1 - S[0, 1]))
    hmo = Cmo.T @ hcore @ Cmo
    emo = np.einsum("pi,qj,rk,sl,pqrs->ijkl", Cmo, Cmo, Cmo, Cmo, eri)
    so = [(0, 0), (0, 1), (1, 0), (1, 1)]
    Hq = Enuc * np.eye(16, dtype=complex)
    for p, q in product(range(4), repeat=2):
        if so[p][1] == so[q][1]:
            Hq += hmo[so[p][0], so[q][0]] * A4[p].conj().T @ A4[q]
    for p, q, r, s in product(range(4), repeat=4):
        if so[p][1] == so[r][1] and so[q][1] == so[s][1]:
            Hq += 0.5 * emo[so[p][0], so[r][0], so[q][0], so[s][0]] * A4[p].conj().T @ A4[q].conj().T @ A4[s] @ A4[r]
    return Hq


def fci(Hq):
    w, v = np.linalg.eigh(Hq)
    k = [k for k in range(16) if np.isclose(np.vdot(v[:, k], NOP @ v[:, k]).real, 2)][0]
    return w[k], v[:, k]


HF = np.zeros(16, complex); HF[int("1100", 2)] = 1
Tgen = A4[2].conj().T @ A4[3].conj().T @ A4[1] @ A4[0]
Kgen = Tgen - Tgen.conj().T                       # антиэрмитов генератор UCC (двойное возбуждение)


def ucc_state(theta):
    return expm(theta * Kgen) @ HF


def energy(Hq, psi):
    return np.vdot(psi, Hq @ psi).real


# ---------------------------------------------------------------------------
section("1. VQE для H2: UCC-анзац с одним параметром; кривая диссоциации")
Hq = h2_hamiltonian(0.7414)
E_fci, g = fci(Hq)
# генератор: 8 паули-строк веса 4, попарно коммутируют
dec = pauli_decompose(1j * Kgen, 4)
assert len(dec) == 8 and all(abs(abs(c) - 1 / 8) < 1e-12 for c in dec.values())
strs = list(dec)
assert all(np.allclose(pstr(a) @ pstr(b), pstr(b) @ pstr(a)) for a, b in combinations(strs, 2))
print("  i(T - T^dag), T = a3^dag a4^dag a2 a1: 8 попарно коммутирующих паули-строк веса 4 с коэффициентами +-1/8:")
print("   ", ", ".join(f"{'+' if dec[s] > 0 else '-'}{s}" for s in strs))
# анзац: cos(theta)|1100> + sin(theta)|0011> (с точностью до знака)
th = 0.3
psi = ucc_state(th)
assert np.isclose(abs(psi[int("1100", 2)]), np.cos(th)) and np.isclose(abs(psi[int("0011", 2)]), np.sin(th))
print("  exp(theta (T - T^dag))|1100> = cos(theta)|1100> -+ sin(theta)|0011>")
res = minimize(lambda t: energy(Hq, ucc_state(t[0])), [0.0], method="BFGS")
th_opt = res.x[0]
print(f"  R = 0.7414 А: theta_opt = {th_opt:.5f}, E_VQE = {res.fun:.6f}, E_FCI = {E_fci:.6f}, E_HF = {energy(Hq, HF):.6f} Ха")
assert abs(res.fun - E_fci) < 1e-9
print("  кривая диссоциации (Ха):")
print("   R, А    E_HF        E_VQE(UCC)  E_FCI       E_VQE-E_FCI   E_HF-E_FCI")
Rs = np.linspace(0.3, 2.5, 10)
curve = []
for R in Rs:
    Hr = h2_hamiltonian(R)
    ef, _ = fci(Hr)
    eh = energy(Hr, HF)
    best = min((minimize(lambda t: energy(Hr, ucc_state(t[0])), [x0], method="BFGS") for x0 in (0.0, 0.5)), key=lambda r: r.fun)
    curve.append((R, eh, best.fun, ef))
    print(f"   {R:.3f}  {eh:10.6f}  {best.fun:10.6f}  {ef:10.6f}  {best.fun - ef:10.1e}   {eh - ef:8.4f}")
    assert abs(best.fun - ef) < 1e-8
iR = int(np.argmin([c[3] for c in curve]))
print(f"  минимум на сетке FCI при R = {curve[iR][0]:.3f} А; E_HF - E_FCI растёт от {curve[0][1] - curve[0][3]:.4f} до {curve[-1][1] - curve[-1][3]:.4f} Ха")
assert curve[-1][1] - curve[-1][3] > 10 * 0.0016
# энергия диссоциации (2 E(H) в STO-3G = 2 * <1s|h|1s>)
Hfar = h2_hamiltonian(50.0 / ANG)
E_far, _ = fci(Hfar)
print(f"  R = 50 бор: E_FCI = {E_far:.6f} Ха (две изолированные H: 2 E_H(STO-3G) ~ -0.9332); E_HF = {energy(Hfar, HF):.4f} — неверная диссоциация RHF")
assert abs(E_far - 2 * (-0.466582)) < 2e-3

# ---------------------------------------------------------------------------
section("2. Правило сдвига параметров")


def apply_1q(psi, U, q, n):
    psi = psi.reshape([2] * n)
    psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)
    return psi.reshape(-1)


def ry(t):
    return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]], dtype=complex)


def rz(t):
    return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])


def cz_chain_phase(n):
    idx = np.arange(2 ** n)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))) & 1
    ph = np.ones(2 ** n)
    for i in range(n - 1):
        ph *= 1 - 2 * (bits[:, i] & bits[:, i + 1])
    return ph


def hea_state(params, n, layers, rot=("y", "z")):
    """Hardware-efficient анзац: слой поворотов (R_y, R_z на каждом кубите) + цепочка CZ; повторить."""
    psi = np.zeros(2 ** n, complex); psi[0] = 1
    ph = cz_chain_phase(n)
    k = 0
    for l in range(layers + 1):
        for q in range(n):
            for r in rot:
                psi = apply_1q(psi, ry(params[k]) if r == "y" else rz(params[k]), q, n)
                k += 1
        if l < layers:
            psi = ph * psi
    return psi


def n_params(n, layers, rot=("y", "z")):
    return n * len(rot) * (layers + 1)


# случайная наблюдаемая и случайный анзац на 3 кубитах
n, layers = 3, 2
Ob = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)); Ob = Ob + Ob.conj().T
f = lambda p: np.vdot(hea_state(p, n, layers), Ob @ hea_state(p, n, layers)).real
p0 = rng.uniform(0, 2 * pi, n_params(n, layers))
grad_ps = np.zeros_like(p0); grad_fd = np.zeros_like(p0)
for k in range(len(p0)):
    e = np.zeros_like(p0); e[k] = 1
    grad_ps[k] = (f(p0 + pi / 2 * e) - f(p0 - pi / 2 * e)) / 2
    grad_fd[k] = (f(p0 + 1e-6 * e) - f(p0 - 1e-6 * e)) / 2e-6
assert np.allclose(grad_ps, grad_fd, atol=1e-6)
print(f"  {len(p0)} параметров, случайная наблюдаемая: [f(theta + pi/2) - f(theta - pi/2)]/2 = df/dtheta (сверено с конечными разностями, max |разн.| = {np.max(np.abs(grad_ps - grad_fd)):.1e})")
# f(theta_k) = a + b cos + c sin
ts = np.linspace(0, 2 * pi, 9)
vals = [f(np.r_[t, p0[1:]]) for t in ts]
Mfit = np.c_[np.ones_like(ts), np.cos(ts), np.sin(ts)]
coef, resid, *_ = np.linalg.lstsq(Mfit, vals, rcond=None)
assert np.allclose(Mfit @ coef, vals)
print("  f(theta_k) = a + b cos(theta_k) + c sin(theta_k) точно (по 9 точкам)")
# UCC для H2: генератор со спектром {0, +-i} -> E(theta) = a + b cos 2theta + c sin 2theta; сдвиг pi/4
Eu = lambda t: energy(Hq, ucc_state(t))
for t in (0.0, 0.2, -0.4):
    d_ps = Eu(t + pi / 4) - Eu(t - pi / 4)
    d_fd = (Eu(t + 1e-6) - Eu(t - 1e-6)) / 2e-6
    assert abs(d_ps - d_fd) < 1e-6
print("  UCC для H2: dE/dtheta = E(theta + pi/4) - E(theta - pi/4) (генератор со спектром {0, +-i})")
# шум выборки: дисперсия оценки градиента с N запусками на каждое плечо
# одна паули-наблюдаемая Z на кубите 1 для простоты
Zo = op_on(n, 0, Z)
fz = lambda p: np.vdot(hea_state(p, n, layers), Zo @ hea_state(p, n, layers)).real
e0 = np.zeros_like(p0); e0[0] = 1
mp, mm = fz(p0 + pi / 2 * e0), fz(p0 - pi / 2 * e0)
true_g = (mp - mm) / 2
print("  шум выборки: оценка градиента по N запускам на каждое плечо (наблюдаемая Z_1):")
for Nsh in (100, 1000, 10000):
    est = []
    for _ in range(2000):
        xp = 2 * rng.binomial(Nsh, (1 + mp) / 2) / Nsh - 1
        xm = 2 * rng.binomial(Nsh, (1 + mm) / 2) / Nsh - 1
        est.append((xp - xm) / 2)
    sd = np.std(est)
    pred = 0.5 * sqrt((1 - mp ** 2 + 1 - mm ** 2) / Nsh)
    print(f"    N = {Nsh:5d}: std = {sd:.4f}, формула (1/2) sqrt((1-<Z>+^2 + 1-<Z>-^2)/N) = {pred:.4f}")
    assert abs(sd / pred - 1) < 0.08

# ---------------------------------------------------------------------------
section("3. Бюджет измерений VQE для H2")
decH = pauli_decompose(Hq, 4)
eps = 1.6e-3
lam = sum(abs(c) for s, c in decH.items() if s != "IIII")
var_terms = {s: 1 - energy(pstr(s), g) ** 2 for s in decH if s != "IIII"}
N_naive = lam ** 2 / eps ** 2
N_opt = sum(abs(decH[s]) * sqrt(var_terms[s]) for s in var_terms) ** 2 / eps ** 2
# группировка: все Z-строки в одну группу (одно измерение в базисе Z), XY-строки — по одной
zgroup = [s for s in var_terms if set(s) <= {"I", "Z"}]
xy = [s for s in var_terms if s not in zgroup]
Oz = sum(decH[s] * pstr(s) for s in zgroup)
var_z = energy(Oz @ Oz, g) - energy(Oz, g) ** 2
groups_sd = [sqrt(var_z)] + [abs(decH[s]) * sqrt(var_terms[s]) for s in xy]
N_group = sum(groups_sd) ** 2 / eps ** 2
# все 4 XY-строки попарно коммутируют -> одна группа (с запутывающей сменой базиса)
Oxy = sum(decH[s] * pstr(s) for s in xy)
assert all(np.allclose(pstr(a) @ pstr(b), pstr(b) @ pstr(a)) for a, b in combinations(xy, 2))
var_xy = energy(Oxy @ Oxy, g) - energy(Oxy, g) ** 2
N_group2 = (sqrt(var_z) + sqrt(var_xy)) ** 2 / eps ** 2
print(f"  lambda = sum|c_j| (без I) = {lam:.4f} Ха; точность eps = 1.6 мХа")
print(f"  верхняя оценка lambda^2/eps^2 = {N_naive:.2e} запусков")
print(f"  по слагаемым, оптимальное распределение (sum |c_j| sigma_j)^2/eps^2 = {N_opt:.2e}")
print(f"  группы: Z-строки ({len(zgroup)}) вместе + 4 XY-строки отдельно: {N_group:.2e};  Z-группа + XY-группа: {N_group2:.2e}")
print(f"  (дисперсии в основном состоянии: Var(Z-часть) = {var_z:.5f}, Var(XY-часть) = {var_xy:.5f})")
assert N_group2 <= N_group * (1 + 1e-9) and N_group <= N_naive and N_opt < N_naive
print(f"  число различных схем измерения: 14 (по слагаемым) -> 5 (Z-группа + 4 XY) -> 2 (Z-группа + XY-группа)")

# ---------------------------------------------------------------------------
section("4. VQE для TFIM n = 6 с hardware-efficient анзацем (без шума выборки)")


def tfim_dense(n, J, h):
    return sum(-J * op_on(n, i, Z) @ op_on(n, i + 1, Z) for i in range(n - 1)) + sum(-h * op_on(n, i, X) for i in range(n))


n = 6
Ht = tfim_dense(n, 1.0, 1.0)
E0t = np.linalg.eigvalsh(Ht)[0]
Ht_r = Ht.real
print(f"  TFIM n = 6, J = h = 1: E_0 = {E0t:.6f}")


def hea_y_energy_grad(params, n, layers, Hm):
    """Энергия и градиент для R_y-анзаца методом сопряжённого состояния (adjoint):
    один прямой и один обратный проход вместо 2P запусков схемы."""
    ph = cz_chain_phase(n)
    ops = []
    k = 0
    for l in range(layers + 1):
        for q in range(n):
            ops.append(("ry", q, k)); k += 1
        if l < layers:
            ops.append(("cz", None, None))
    psi = np.zeros(2 ** n); psi[0] = 1
    for o, q, k in ops:
        psi = apply_1q(psi, ry(params[k]).real, q, n) if o == "ry" else ph * psi
    lam = Hm @ psi
    E = psi @ lam
    grad = np.zeros(len(params))
    Ym = np.array([[0, -1], [1, 0]]) / 2          # -i Y/2 (вещественная)
    for o, q, k in reversed(ops):
        if o == "ry":
            grad[k] = 2 * lam @ apply_1q(psi, Ym, q, n)
            Ui = ry(-params[k]).real
            psi = apply_1q(psi, Ui, q, n); lam = apply_1q(lam, Ui, q, n)
        else:
            psi = ph * psi; lam = ph * lam
    return E, grad


# сверка градиента с правилом сдвига
pt = rng.uniform(0, 2 * pi, n_params(n, 2, rot=("y",)))
Eg, gg = hea_y_energy_grad(pt, n, 2, Ht_r)
for k in (0, 5, 11):
    e = np.zeros_like(pt); e[k] = 1
    gs = (hea_y_energy_grad(pt + pi / 2 * e, n, 2, Ht_r)[0] - hea_y_energy_grad(pt - pi / 2 * e, n, 2, Ht_r)[0]) / 2
    assert abs(gs - gg[k]) < 1e-10
print("  градиент методом сопряжённого состояния совпадает с правилом сдвига")
vqe_res = {}
for layers in (1, 2, 3, 6):
    P = n_params(n, layers, rot=("y",))
    best = None
    for trial in range(6):
        r = minimize(hea_y_energy_grad, rng.uniform(0, 2 * pi, P), args=(n, layers, Ht_r), jac=True, method="BFGS", options={"gtol": 1e-10, "maxiter": 2000})
        if best is None or r.fun < best.fun:
            best = r
    vqe_res[layers] = best.fun
    print(f"   слоёв {layers}: {P:2d} параметров, E_VQE = {best.fun:.6f}, ошибка {best.fun - E0t:.2e} (лучший из 6 случайных стартов)")
    assert best.fun >= E0t - 1e-9
assert vqe_res[6] - E0t < vqe_res[1] - E0t
assert abs(vqe_res[6] - vqe_res[3]) < 1e-4      # застревание: 3 и 6 слоёв дают одно и то же
# анзац, построенный по гамильтониану (HVA): |+>^n, затем слои e^{-i beta B} e^{-i gamma A}
zz = np.diag(sum(-op_on(n, i, Z) @ op_on(n, i + 1, Z) for i in range(n - 1))).real
Bx = sum(-op_on(n, i, X) for i in range(n))
wB, VB = np.linalg.eigh(Bx)
plus = np.full(2 ** n, 1 / np.sqrt(2 ** n), complex)


def hva_energy(p):
    L = len(p) // 2
    psi = plus.copy()
    for gm, bt in zip(p[:L], p[L:]):
        psi = np.exp(-1j * gm * zz) * psi
        psi = VB @ (np.exp(-1j * bt * wB) * (VB.conj().T @ psi))
    return np.vdot(psi, Ht @ psi).real


hva_res = {}
for L in (1, 2, 3, 4):
    best = min(minimize(hva_energy, rng.uniform(0, 1, 2 * L), method="BFGS").fun for _ in range(10))
    hva_res[L] = best - E0t
    print(f"   HVA, слоёв {L}: {2 * L} параметров, ошибка {best - E0t:.2e}")
assert all(hva_res[L + 1] < hva_res[L] for L in (1, 2, 3)) and hva_res[4] < vqe_res[6] - E0t

# ---------------------------------------------------------------------------
section("5. QAOA для MaxCut")


def cut_diag(n, edges):
    idx = np.arange(2 ** n)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))) & 1
    return sum((bits[:, u] ^ bits[:, v]).astype(float) for u, v in edges)


def qaoa_state(gammas, betas, n, Cd):
    psi = np.full(2 ** n, 1 / sqrt(2 ** n), complex)
    for gm, bt in zip(gammas, betas):
        psi = np.exp(-1j * gm * Cd) * psi
        rx = np.array([[np.cos(bt), -1j * np.sin(bt)], [-1j * np.sin(bt), np.cos(bt)]])   # e^{-i beta X}
        for q in range(n):
            psi = apply_1q(psi, rx, q, n)
    return psi


def qaoa_exp(params, n, Cd):
    p = len(params) // 2
    psi = qaoa_state(params[:p], params[p:], n, Cd)
    return np.dot(np.abs(psi) ** 2, Cd)


def qaoa_opt(p, n, Cd, starts=20):
    best = None
    for _ in range(starts):
        x0 = np.r_[rng.uniform(0, pi, p), rng.uniform(0, pi / 2, p)]
        r = minimize(lambda x: -qaoa_exp(x, n, Cd), x0, method="BFGS")
        if best is None or -r.fun > best[0]:
            best = (-r.fun, r.x)
    return best


# куб Q3: 3-регулярный граф без треугольников, 8 вершин, 12 рёбер, двудольный (MaxCut = 12)
cube = [(u, v) for u in range(8) for v in range(u + 1, 8) if bin(u ^ v).count("1") == 1]
Cc = cut_diag(8, cube)
assert len(cube) == 12 and Cc.max() == 12
# формула p = 1 для ребра в d-регулярном графе без треугольников:
# <C_uv> = 1/2 + (1/2) sin(4 beta) sin(gamma) cos^{d-1}(gamma)
for _ in range(5):
    gm, bt = rng.uniform(0, pi), rng.uniform(0, pi)
    val = qaoa_exp(np.array([gm, bt]), 8, Cc) / 12
    assert np.isclose(val, 0.5 + 0.5 * np.sin(4 * bt) * np.sin(gm) * np.cos(gm) ** 2)
best1 = qaoa_opt(1, 8, Cc)
print(f"  куб Q3 (3-регулярный, без треугольников): <C_uv> = 1/2 + (1/2) sin(4 beta) sin(gamma) cos^2(gamma) — проверено")
print(f"  p = 1: max <C>/MaxCut = {best1[0] / 12:.6f};  1/2 + 1/(3 sqrt 3) = {0.5 + 1 / (3 * sqrt(3)):.6f}")
assert abs(best1[0] / 12 - (0.5 + 1 / (3 * sqrt(3)))) < 1e-6
for p in (2, 3):
    b = qaoa_opt(p, 8, Cc)
    print(f"  p = {p}: <C>/MaxCut = {b[0] / 12:.4f}")
# кольцо: p = 1 даёт 3/4
ring = [(i, (i + 1) % 8) for i in range(8)]
Cr = cut_diag(8, ring)
br = qaoa_opt(1, 8, Cr)
print(f"  кольцо из 8 вершин, p = 1: <C>/MaxCut = {br[0] / Cr.max():.4f} (= 3/4)")
assert abs(br[0] / Cr.max() - 0.75) < 1e-6
# случайный 3-регулярный граф на 8 вершинах (ЛР8, доп. задание)
def random_3reg(nv, rng):
    while True:
        stubs = np.repeat(np.arange(nv), 3)
        rng.shuffle(stubs)
        e = [tuple(sorted(stubs[i:i + 2])) for i in range(0, len(stubs), 2)]
        if all(u != v for u, v in e) and len(set(e)) == len(e):
            return e
rg = random_3reg(8, rng)
Cg = cut_diag(8, rg)
tri = sum(1 for a, b, c in combinations(range(8), 3) if {(a, b), (a, c), (b, c)} <= set(rg))
print(f"  случайный 3-регулярный граф: рёбра {rg}; треугольников {tri}; MaxCut = {Cg.max():.0f}")
qaoa_rand = {}
for p in (1, 2, 3):
    b = qaoa_opt(p, 8, Cg)
    qaoa_rand[p] = b[0] / Cg.max()
    print(f"   p = {p}: <C>/MaxCut = {qaoa_rand[p]:.4f}")
assert qaoa_rand[1] <= qaoa_rand[2] + 1e-9 <= qaoa_rand[3] + 2e-9
# вероятность получить максимальный разрез при p = 3
b3 = qaoa_opt(3, 8, Cg)
psi3 = qaoa_state(b3[1][:3], b3[1][3:], 8, Cg)
pmax = np.sum(np.abs(psi3[Cg == Cg.max()]) ** 2)
print(f"   p = 3: вероятность измерить оптимальный разрез = {pmax:.3f} (случайная строка: {np.mean(Cg == Cg.max()):.3f})")
# связь с адиабатикой: троттеризованный линейный отжиг H(s) = -(1-s) B - s C как QAOA
print("  троттеризованный отжиг (gamma_k = (k - 1/2)/p * dt, beta_k = (1 - (k - 1/2)/p) * dt), куб Q3:")
for p, dt in ((5, 0.5), (20, 0.5), (60, 0.5)):
    gms = np.array([(k + 0.5) / p * dt for k in range(p)])
    bts = np.array([(1 - (k + 0.5) / p) * dt for k in range(p)])
    val = qaoa_exp(np.r_[gms, bts], 8, Cc) / 12
    print(f"   p = {p:3d}, T = p dt = {p * dt:5.1f}: <C>/MaxCut = {val:.4f}")

# ---------------------------------------------------------------------------
section("6. Barren plateaus: дисперсия dC/dtheta_1 от n")
print("  анзац: (R_y R_z на каждом кубите + цепочка CZ) x L, затем R_y R_z; 200 случайных наборов параметров")


def grad_first(params, n, layers, kind):
    def cost(p):
        psi = hea_state(p, n, layers)
        pr = np.abs(psi) ** 2
        if kind == "global":
            return 1 - pr[0]
        bits = (np.arange(2 ** n)[:, None] >> (n - 1 - np.arange(n))) & 1
        return 1 - np.mean([(pr * (bits[:, i] == 0)).sum() for i in range(n)])
    e = np.zeros_like(params); e[0] = 1
    return (cost(params + pi / 2 * e) - cost(params - pi / 2 * e)) / 2


bp = {}
ns = [2, 4, 6, 8, 10, 12]
for kind, depth in (("global", "n"), ("local", "n"), ("local", "2")):
    vs = []
    for nn in ns:
        L = nn if depth == "n" else 2
        P = n_params(nn, L)
        gr = [grad_first(rng.uniform(0, 2 * pi, P), nn, L, kind) for _ in range(200)]
        vs.append(np.var(gr))
    bp[(kind, depth)] = vs
    sl = np.polyfit(ns[1:], np.log2(vs[1:]), 1)[0]
    print(f"   {kind:6s}, глубина L = {depth}: " + "  ".join(f"n={nn}: {v:.1e}" for nn, v in zip(ns, vs)) + f"   log2 Var ~ {sl:.2f} n")
    bp[(kind, depth, "slope")] = sl
ll = np.polyfit(np.log(ns[1:]), np.log(bp[("local", "2")][1:]), 1)[0]
print(f"   local, L = 2: в двойном логарифмическом масштабе наклон {ll:.2f}; n^2 Var = " + " ".join(f"{nn ** 2 * v:.3f}" for nn, v in zip(ns, bp[("local", "2")])))
assert -2.5 < ll < -1.5      # степенное убывание ~ 1/n^2 (нормировка 1/n в функции стоимости)
assert bp[("global", "n", "slope")] < -0.8
assert bp[("local", "2", "slope")] > -0.4
assert bp[("global", "n")][-1] < bp[("local", "2")][-1] / 50

# ---------------------------------------------------------------------------
section("7. Алгоритм HHL: полная симуляция малой схемы")


def qft_mat(T):
    j, k = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
    return np.exp(2j * pi * j * k / T) / sqrt(T)


def hhl(A, b, t, C=None, evolve_scale=None):
    """Схема HHL: QPE (t кубитов часов) для U = exp(2 pi i A/T * s), поворот вспомогательного кубита
    на C/lambda, обратная QPE. Возвращает (вероятность успеха, состояние системы при успехе)."""
    d = A.shape[0]
    T = 2 ** t
    s = 1.0 if evolve_scale is None else evolve_scale       # собственное значение lambda <-> фаза lambda s / T
    U = expm(2j * pi * A * s / T)
    Hn = reduce(np.kron, [H1] * t)
    ctrl = sum(np.kron(np.diag(np.eye(T)[m]), np.linalg.matrix_power(U, m)) for m in range(T))
    QPE = np.kron(qft_mat(T).conj().T, np.eye(d)) @ ctrl @ np.kron(Hn, np.eye(d))
    lam_of_m = np.arange(T) / s
    if C is None:
        C = np.min(np.abs(np.linalg.eigvalsh(A)))
    # вспомогательный кубит — последний множитель
    state = np.kron(QPE @ np.kron(np.eye(T)[0], b), np.array([1, 0]))
    state = state.reshape(T, d, 2)
    out = np.zeros_like(state)
    for m in range(T):
        lam = lam_of_m[m]
        if lam == 0:
            out[m, :, 0] = state[m, :, 0]
            continue
        c = min(1.0, C / lam)
        out[m, :, 0] = sqrt(1 - c ** 2) * state[m, :, 0]
        out[m, :, 1] = c * state[m, :, 0]
    out = out.reshape(T * d, 2)
    anc1 = QPE.conj().T @ out[:, 1]
    anc1 = anc1.reshape(T, d)
    p_succ = np.sum(np.abs(anc1[0]) ** 2)           # анцилла = 1 и часы = 0
    x = anc1[0] / np.linalg.norm(anc1[0])
    return p_succ, x, C


# A с целыми собственными значениями 1..T-1: QPE точна
Q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
lams = np.array([1, 2, 5, 12])
A = Q @ np.diag(lams) @ Q.conj().T
b = rng.normal(size=4) + 1j * rng.normal(size=4); b /= np.linalg.norm(b)
xe = np.linalg.solve(A, b); xe_n = xe / np.linalg.norm(xe)
ps, x, C = hhl(A, b, 4)
fid = abs(np.vdot(xe_n, x)) ** 2
print(f"  A 4x4, собственные значения {list(lams)} (kappa = 12), часы t = 4: верность с A^-1 b/||A^-1 b|| = {fid:.10f}")
print(f"  вероятность успеха = {ps:.4f};  формула C^2 ||A^-1 b||^2 = {C ** 2 * np.linalg.norm(xe) ** 2:.4f}")
assert fid > 1 - 1e-10 and np.isclose(ps, C ** 2 * np.linalg.norm(xe) ** 2)
# зависимость вероятности от kappa: худший случай b — собственный вектор с наибольшим lambda
print("  вероятность успеха при b — собственный вектор с lambda_max (C = lambda_min = 1):")
for kap in (2, 4, 8, 15):
    lam2 = np.array([1, kap])
    A2 = np.diag(lam2).astype(complex)
    ps2, x2, _ = hhl(A2, np.array([0, 1], complex), 4)
    print(f"    kappa = {kap:2d}: P = {ps2:.4f} = 1/kappa^2 = {1 / kap ** 2:.4f}")
    assert np.isclose(ps2, 1 / kap ** 2)
# неточные собственные значения: верность растёт с числом кубитов часов
lams3 = np.array([1.3, 2.7, 4.45, 6.1])
A3 = Q @ np.diag(lams3) @ Q.conj().T
xe3 = np.linalg.solve(A3, b); xe3 /= np.linalg.norm(xe3)
print("  собственные значения 1.3, 2.7, 4.45, 6.1 (не на сетке): часы t, масштаб s = 2^t/8 (диапазон lambda < 8)")
fids = []
for t in (3, 4, 5, 6, 7):
    s = 2 ** t / 8
    ps3, x3, _ = hhl(A3, b, t, C=1.0, evolve_scale=s)
    f3 = abs(np.vdot(xe3, x3)) ** 2
    fids.append(f3)
    print(f"    t = {t}: верность = {f3:.4f}, P(успех) = {ps3:.4f}")
assert fids[-1] > fids[0] and fids[-1] > 0.95

# ---------------------------------------------------------------------------
section("8. Ответы к задачам семинара")
print(f"  3: H2: lambda = {lam:.4f}, N <= lambda^2/eps^2 = {N_naive:.2e}; с группировкой {N_group2:.2e}")
print(f"  5: d = 3, p = 1: max_gamma sin(gamma) cos^2(gamma) при tan^2(gamma) = 1/2: {np.sin(np.arctan(1 / sqrt(2))) * np.cos(np.arctan(1 / sqrt(2))) ** 2:.6f} = 2/(3 sqrt 3) = {2 / (3 * sqrt(3)):.6f}")
print(f"  5: кольцо (d = 2): max sin(gamma) cos(gamma) = 1/2 -> 1/2 + 1/4 = 3/4")
print(f"  8: kappa = 100: P ~ 1/kappa^2 = {1e-4:.0e}; с усилением амплитуды ~ kappa = 100 повторений")
print("\nВсе проверки пройдены.")
