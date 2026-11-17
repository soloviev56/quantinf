"""
Лекция II.2. Стандартные каналы и уравнение Линдблада: деполяризующий канал,
дефазировка, амплитудное и обобщённое амплитудное затухание (шар Блоха,
композиция, неподвижные точки), паули-твирлинг и клиффордов твирлинг,
рандомизированная компиляция, уравнение ГКСЛ (генератор, полугруппа,
вывод из короткого шага Крауса), вывод скорости распада из микроскопической
модели (Вигнер–Вайскопф), тепловая баня, T1/T2/T2*, Рамзи и спин-эхо,
уравнения Блоха с накачкой, немарковская динамика (Джейнс–Каммингс с
затухающим резонатором, мера Бройера–Лайне–Пиило), коллективная дефазировка.

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
Соглашения: порядок множителей — первый старший (np.kron); |0> — основное
состояние, H_S = -(w/2) Z, sigma_- = |0><1|; матрица Чоя — вход первый.
Векторизация по строкам: vec(A X B) = (A (x) B^T) vec(X).
"""
import time
import numpy as np
from math import pi, sqrt, sin, cos, exp, log
from functools import reduce
from scipy.linalg import expm, null_space
from scipy.optimize import curve_fit

T_START = time.time()
rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [I2, X, Y, Z]
SM = np.array([[0, 1], [0, 0]], dtype=complex)  # sigma_- = |0><1|
SP = SM.conj().T                                 # sigma_+ = |1><0|
H1 = np.array([[1, 1], [1, -1]], dtype=complex) / sqrt(2)
S1 = np.diag([1, 1j])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


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


def choi_map(channel, d):
    return sum(np.kron(np.outer(ket(i, d), ket(j, d)), channel(np.outer(ket(i, d), ket(j, d))))
               for i in range(d) for j in range(d))


def is_cptp_choi(J, d, tol=1e-9):
    w = np.linalg.eigvalsh((J + dag(J)) / 2)
    tr = J.reshape(d, d, d, d).trace(axis1=1, axis2=3)
    return w.min() > -tol and np.allclose(tr, np.eye(d), atol=tol)


def bloch(rho):
    return np.real([np.trace(rho @ P) for P in (X, Y, Z)])


def rho_of(r):
    return (I2 + r[0] * X + r[1] * Y + r[2] * Z) / 2


def affine(channel):
    """r -> M r + c (II.1, (5.1))."""
    M = np.array([[0.5 * np.trace(sk @ channel(sl)).real for sl in (X, Y, Z)] for sk in (X, Y, Z)])
    c = np.array([0.5 * np.trace(sk @ channel(I2)).real for sk in (X, Y, Z)])
    return M, c


def tnorm(A):
    return np.abs(np.linalg.eigvalsh((A + dag(A)) / 2)).sum()


def F_pro_kraus(K):
    d = K[0].shape[0]
    return sum(abs(np.trace(k)) ** 2 for k in K) / d ** 2


def F_pro_map(channel, d):
    J = choi_map(channel, d)
    phi = np.eye(d).reshape(-1) / sqrt(d)
    return (phi.conj() @ J @ phi).real / d


# стандартные каналы (Обозначения, §12)
def K_dep(p):
    return [sqrt(1 - 3 * p / 4) * I2, sqrt(p / 4) * X, sqrt(p / 4) * Y, sqrt(p / 4) * Z]


def K_deph(p):
    return [sqrt(1 - p) * I2, sqrt(p) * Z]


def K_ad(g):
    return [np.array([[1, 0], [0, sqrt(1 - g)]], complex), np.array([[0, sqrt(g)], [0, 0]], complex)]


def K_gad(g, p):
    """Обобщённое амплитудное затухание A_{g,p}: стационарное состояние diag(p, 1-p)."""
    return [sqrt(p) * np.array([[1, 0], [0, sqrt(1 - g)]], complex),
            sqrt(p) * np.array([[0, sqrt(g)], [0, 0]], complex),
            sqrt(1 - p) * np.array([[sqrt(1 - g), 0], [0, 1]], complex),
            sqrt(1 - p) * np.array([[0, 0], [sqrt(g), 0]], complex)]


def ch(K):
    return lambda r: apply_kraus(K, r)


# ---------------------------------------------------------------------------
section("1. Стандартные каналы: CPTP, действие на шар Блоха, композиция")
for name, K in [("Delta_0.3", K_dep(0.3)), ("D_0.2", K_deph(0.2)), ("A_0.4", K_ad(0.4)), ("A_{0.4,0.8}", K_gad(0.4, 0.8))]:
    assert np.allclose(sum(dag(k) @ k for k in K), I2)
    M, c = affine(ch(K))
    print(f"  {name:12s} M = diag{tuple(np.round(np.diag(M), 4))}, c = {tuple(np.round(c, 4))}")
    assert np.allclose(M, np.diag(np.diag(M)))
g, p = 0.4, 0.8
M, c = affine(ch(K_gad(g, p)))
assert np.allclose(np.diag(M), [sqrt(1 - g), sqrt(1 - g), 1 - g]) and np.allclose(c, [0, 0, g * (2 * p - 1)])
r_fix = np.array([0, 0, 2 * p - 1])
assert np.allclose(M @ r_fix + c, r_fix)
assert np.allclose(apply_kraus(K_gad(g, p), np.diag([p, 1 - p])), np.diag([p, 1 - p]))
print(f"  A_(g,p): M = diag(sqrt(1-g), sqrt(1-g), 1-g), c = (0, 0, g(2p-1)); неподвижная точка z = 2p - 1 = {2 * p - 1:.1f}, т.е. diag(p, 1-p)")
assert np.allclose(choi_map(ch(K_gad(g, 1.0)), 2), choi_map(ch(K_ad(g)), 2))
print("  A_(g,1) = A_g (затухание к |0>), A_(g,0) — затухание к |1>")
# композиция
for (a, b) in [(0.2, 0.3), (0.5, 0.1)]:
    J1 = choi_map(lambda r: apply_kraus(K_dep(a), apply_kraus(K_dep(b), r)), 2)
    assert np.allclose(J1, choi_map(ch(K_dep(1 - (1 - a) * (1 - b))), 2))
    J2 = choi_map(lambda r: apply_kraus(K_ad(a), apply_kraus(K_ad(b), r)), 2)
    assert np.allclose(J2, choi_map(ch(K_ad(1 - (1 - a) * (1 - b))), 2))
    J3 = choi_map(lambda r: apply_kraus(K_deph(a), apply_kraus(K_deph(b), r)), 2)
    pc = (1 - (1 - 2 * a) * (1 - 2 * b)) / 2
    assert np.allclose(J3, choi_map(ch(K_deph(pc)), 2))
print("  композиция: Delta_a Delta_b = Delta_{1-(1-a)(1-b)}, A_a A_b = A_{1-(1-a)(1-b)}, D_a D_b = D_c с 1-2c = (1-2a)(1-2b)")
J_AD = choi_map(lambda r: apply_kraus(K_ad(0.3), apply_kraus(K_deph(0.2), r)), 2)
J_DA = choi_map(lambda r: apply_kraus(K_deph(0.2), apply_kraus(K_ad(0.3), r)), 2)
assert np.allclose(J_AD, J_DA)
print("  A_g и D_p коммутируют (оба сохраняют ось z и вращательно симметричны вокруг неё)")
# ранг Чоя GAD
print(f"  ранг Чоя A_(0.4,0.8) = {np.linalg.matrix_rank(choi_map(ch(K_gad(0.4, 0.8)), 2), tol=1e-10)}, A_0.4 = {np.linalg.matrix_rank(choi_map(ch(K_ad(0.4)), 2), tol=1e-10)}")
# GAD — не унитальный при p != 1/2
Mg, cg = affine(ch(K_gad(0.4, 0.5)))
assert np.allclose(cg, 0)
print("  A_(g,1/2) унитален (бесконечная температура)")

# ---------------------------------------------------------------------------
section("2. Паули-твирлинг и клиффордов твирлинг")


def pauli_twirl(channel):
    return lambda r: sum(P @ channel(P @ r @ P) @ P for P in PAULIS) / 4


def pauli_probs(channel):
    """p_P для канала Паули (диагональ chi-матрицы в базисе Паули): p_P = <P,J P>/... через (5.2) II.1."""
    M, _ = affine(channel)
    lx, ly, lz = np.diag(M)
    return np.array([1 + lx + ly + lz, 1 + lx - ly - lz, 1 - lx + ly - lz, 1 - lx - ly + lz]) / 4


def chi_diag(K):
    """chi_PP = sum_j |Tr(P K_j)|^2 / d^2 — вероятности паули после твирлинга."""
    d = K[0].shape[0]
    n = int(round(np.log2(d)))
    import itertools
    out = []
    for idx in itertools.product(range(4), repeat=n):
        P = kron(*[PAULIS[i] for i in idx])
        out.append(sum(abs(np.trace(P @ k)) ** 2 for k in K) / d ** 2)
    return np.array(out)


for name, K in [("A_0.1", K_ad(0.1)), ("A_{0.3,0.7}", K_gad(0.3, 0.7))]:
    T = pauli_twirl(ch(K))
    Jt = choi_map(T, 2)
    pp = chi_diag(K)
    Jp = choi_map(lambda r: sum(q * P @ r @ P for q, P in zip(pp, PAULIS)), 2)
    assert np.allclose(Jt, Jp)
    assert abs(F_pro_map(T, 2) - F_pro_kraus(K)) < 1e-12
    print(f"  твирлинг {name}: (p_I, p_X, p_Y, p_Z) = {tuple(np.round(pp, 6))}; F_pro до = {F_pro_kraus(K):.6f}, после = {F_pro_map(T, 2):.6f}")
g = 0.1
pI_a = (1 + sqrt(1 - g)) ** 2 / 4
pX_a = g / 4
pZ_a = (1 - sqrt(1 - g)) ** 2 / 4
assert np.allclose(chi_diag(K_ad(g)), [pI_a, pX_a, pX_a, pZ_a])
print(f"  формулы: p_I = (1+sqrt(1-g))^2/4 = {pI_a:.6f}, p_X = p_Y = g/4 = {pX_a:.6f}, p_Z = (1-sqrt(1-g))^2/4 = {pZ_a:.6f}")
# когерентная ошибка
for eps in (0.1, 0.02):
    Rx = expm(-1j * eps / 2 * X)
    pp = chi_diag([Rx])
    assert np.allclose(pp, [cos(eps / 2) ** 2, sin(eps / 2) ** 2, 0, 0])
    print(f"  твирлинг R_x({eps}): p_I = cos^2(eps/2) = {pp[0]:.6f}, p_X = sin^2(eps/2) = {pp[1]:.3e}; ||.-id||_dia: 2 sin(eps/2) = {2 * sin(eps / 2):.4e} -> 2 sin^2(eps/2) = {2 * sin(eps / 2) ** 2:.3e}")
# канал Паули не меняется
pv = np.array([0.9, 0.05, 0.03, 0.02])
Pch = lambda r: sum(q * P @ r @ P for q, P in zip(pv, PAULIS))
assert np.allclose(choi_map(pauli_twirl(Pch), 2), choi_map(Pch, 2))
print("  канал Паули — неподвижная точка твирлинга; твирлинг идемпотентен")
# убирает недиагональные элементы M и сдвиг c
Mt, ct = affine(pauli_twirl(ch(K_gad(0.3, 0.7))))
M0, c0 = affine(ch(K_gad(0.3, 0.7)))
assert np.allclose(Mt, np.diag(np.diag(M0))) and np.allclose(ct, 0)
print("  твирлинг: M -> diag(M), c -> 0 (неунитальная часть исчезает)")

# группа Клиффорда одного кубита (по модулю фазы)


def canon(U):
    k = np.argmax(np.abs(U.reshape(-1)) > 1e-9)
    ph = U.reshape(-1)[k] / abs(U.reshape(-1)[k])
    return U / ph


def clifford_group_1q():
    group = [I2]
    frontier = [I2]
    while frontier:
        new = []
        for U in frontier:
            for G in (H1, S1):
                V = canon(G @ U)
                if not any(np.allclose(V, W) for W in group):
                    group.append(V); new.append(V)
        frontier = new
    return group


C1 = clifford_group_1q()
assert len(C1) == 24
print(f"  группа Клиффорда C_1 (по модулю фазы): {len(C1)} элементов")


def clifford_twirl(channel):
    return lambda r: sum(dag(C) @ channel(C @ r @ dag(C)) @ C for C in C1) / len(C1)


for name, K in [("A_0.1", K_ad(0.1)), ("R_x(0.1)", [expm(-0.05j * X)]), ("случайный", None)]:
    if K is None:
        G = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
        V, _ = np.linalg.qr(G)
        K = [V[2 * j:2 * j + 2, :] for j in range(3)]
    f = F_pro_kraus(K)
    pdep = (4 * f - 1) / 3
    Jc = choi_map(clifford_twirl(ch(K)), 2)
    Jdep = choi_map(lambda r: pdep * r + (1 - pdep) * np.trace(r) * I2 / 2, 2)
    assert np.allclose(Jc, Jdep)
    print(f"  клиффордов твирлинг {name:9s}: деполяризующий с lambda = (4F_pro-1)/3 = {pdep:.6f} (это Delta_(1-lambda))")
# рандомизированная компиляция: твирлинг шума CNOT
G = rng.normal(size=(4 * 3, 4)) + 1j * rng.normal(size=(4 * 3, 4))
V, _ = np.linalg.qr(G)
Kn_raw = [V[4 * j:4 * j + 4, :] for j in range(3)]
# слабый шум: смесь тождества и случайного канала + когерентный ZZ-поворот
eps = 0.05
UZZ = expm(-1j * 0.08 * np.kron(Z, Z))
Kn = [sqrt(1 - eps) * UZZ] + [sqrt(eps) * k for k in Kn_raw]
assert np.allclose(sum(dag(k) @ k for k in Kn), np.eye(4))
noisy_cnot = lambda r: apply_kraus(Kn, CNOT @ r @ dag(CNOT))
import itertools
P2 = [kron(PAULIS[a], PAULIS[b]) for a, b in itertools.product(range(4), repeat=2)]


def twirled_cnot(r):
    out = 0
    for P in P2:
        Pp = CNOT @ P @ dag(CNOT)  # паули после CNOT
        out = out + Pp @ noisy_cnot(P @ r @ P) @ dag(Pp)
    return out / 16


J_tw = choi_map(twirled_cnot, 4)
pp2 = chi_diag(Kn)
J_target = choi_map(lambda r: sum(q * P @ (CNOT @ r @ dag(CNOT)) @ P for q, P in zip(pp2, P2)), 4)
assert np.allclose(J_tw, J_target)
assert all(np.allclose(CNOT @ P @ dag(CNOT) @ dag(CNOT @ P @ dag(CNOT)), np.eye(4)) for P in P2)
print(f"  рандомизированная компиляция CNOT (16 паули-рамок): шум -> канал Паули, p_II = {pp2[0]:.5f} = F_pro шума {F_pro_kraus(Kn):.5f}")
assert abs(pp2[0] - F_pro_kraus(Kn)) < 1e-12

# ---------------------------------------------------------------------------
section("3. Уравнение ГКСЛ: генератор, полугруппа, кубит")


def lindbladian(H, Ls):
    d = H.shape[0]
    Id = np.eye(d)
    Lsup = -1j * (np.kron(H, Id) - np.kron(Id, H.T))
    for L in Ls:
        LdL = dag(L) @ L
        Lsup += np.kron(L, L.conj()) - 0.5 * np.kron(LdL, Id) - 0.5 * np.kron(Id, LdL.T)
    return Lsup


def evolve(Lsup, rho, t):
    d = rho.shape[0]
    return (expm(Lsup * t) @ rho.reshape(-1)).reshape(d, d)


def channel_of(Lsup, t, d):
    E = expm(Lsup * t)
    return lambda r: (E @ r.reshape(-1)).reshape(d, d)


# проверка векторизации
A_, B_, R_ = (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)) for _ in range(3))
assert np.allclose((A_ @ R_ @ B_).reshape(-1), np.kron(A_, B_.T) @ R_.reshape(-1))
w0, g1, gphi = 2.0, 0.3, 0.2
Hq = -w0 / 2 * Z
Lq = lindbladian(Hq, [sqrt(g1) * SM, sqrt(gphi / 2) * Z])
rho0 = rho_of(np.array([0.6, 0.0, -0.8]))
for t in (0.5, 2.0, 5.0):
    rt = evolve(Lq, rho0, t)
    assert np.isclose(rt[1, 1].real, rho0[1, 1].real * exp(-g1 * t))
    assert np.isclose(rt[0, 1], rho0[0, 1] * np.exp(1j * w0 * t) * exp(-(g1 / 2 + gphi) * t))
    assert is_cptp_choi(choi_map(channel_of(Lq, t, 2), 2), 2)
print("  кубит: rho_11(t) = rho_11(0) e^{-g1 t},  rho_01(t) = rho_01(0) e^{+i w t} e^{-(g1/2 + g_phi) t}")
print(f"  T1 = 1/g1 = {1 / g1:.4f}, T2 = 1/(g1/2 + g_phi) = {1 / (g1 / 2 + gphi):.4f}, T_phi = 1/g_phi = {1 / gphi:.4f}")
for t in (0.7, 3.0):
    assert np.allclose(choi_map(channel_of(lindbladian(0 * Z, [sqrt(g1) * SM]), t, 2), 2), choi_map(ch(K_ad(1 - exp(-g1 * t))), 2))
    pt = (1 - exp(-gphi * t)) / 2
    assert np.allclose(choi_map(channel_of(lindbladian(0 * Z, [sqrt(gphi / 2) * Z]), t, 2), 2), choi_map(ch(K_deph(pt)), 2))
    q = 1 - exp(-0.4 * t)
    Ldep = lindbladian(0 * Z, [sqrt(0.4 / 4) * P for P in (X, Y, Z)])
    assert np.allclose(choi_map(channel_of(Ldep, t, 2), 2), choi_map(ch(K_dep(q)), 2))
print("  e^{tL}: sqrt(g1) sigma_- -> A_g, g = 1 - e^{-g1 t};  sqrt(g_phi/2) Z -> D_p, 1 - 2p = e^{-g_phi t};")
print("          sqrt(G/4) {X,Y,Z} -> Delta_p, p = 1 - e^{-G t}")
# полугруппа
t1, t2 = 0.8, 1.7
assert np.allclose(expm(Lq * (t1 + t2)), expm(Lq * t1) @ expm(Lq * t2))
# неоднозначность (H, L): сдвиг L -> L + c I меняет H
c0 = 0.3 + 0.2j
L0 = sqrt(g1) * SM
Ha = Hq
Hb = Hq - 1j / 2 * (np.conj(c0) * L0 - c0 * dag(L0))
assert np.allclose(lindbladian(Ha, [L0]), lindbladian(Hb, [L0 + c0 * I2]))
print("  (H, L) -> (H - (i/2)(c* L - c L^+), L + c I) не меняет генератор")
# вывод из короткого шага Крауса
H_dr = -w0 / 2 * Z + 0.7 * X
Ls = [sqrt(g1) * SM, sqrt(gphi / 2) * Z]
Lf = lindbladian(H_dr, Ls)
t_tot = 2.0
E_ex = expm(Lf * t_tot)
errs = []
for n in (50, 100, 200, 400, 800):
    dt = t_tot / n
    K0 = I2 - (1j * H_dr + 0.5 * sum(dag(L) @ L for L in Ls)) * dt
    Kd = [K0] + [sqrt(dt) * L for L in Ls]
    Ssup = sum(np.kron(k, k.conj()) for k in Kd)
    errs.append((dt, np.abs(np.linalg.matrix_power(Ssup, n) - E_ex).max()))
sl = np.polyfit(np.log([e[0] for e in errs]), np.log([e[1] for e in errs]), 1)[0]
print(f"  короткий шаг Крауса K0 = I - (iH + 1/2 sum L^+L) dt, K_k = sqrt(dt) L_k: ошибка ~ dt^{sl:.3f}")
for dt, e in errs:
    print(f"     dt = {dt:.4f}: max|S^n - e^(tL)| = {e:.3e}")
assert abs(sl - 1) < 0.05
# ошибка Ли-расщепления «гамильтониан / диссипатор» (ЛР10, задание 3)
print("  расщепление: H, затем дискретные каналы A_g и D_p на шаге dt (накачка 0.7 X):")
Hs = lindbladian(H_dr, [])
Ds = lindbladian(0 * Z, Ls)
spl = []
for n in (20, 40, 80, 160):
    dt = t_tot / n
    step_lie = expm(Ds * dt) @ expm(Hs * dt)
    step_str = expm(Hs * dt / 2) @ expm(Ds * dt) @ expm(Hs * dt / 2)
    e1 = np.abs(np.linalg.matrix_power(step_lie, n) - E_ex).max()
    e2 = np.abs(np.linalg.matrix_power(step_str, n) - E_ex).max()
    spl.append((dt, e1, e2))
    print(f"     dt = {dt:.4f}: Ли {e1:.3e}, Стрэнг {e2:.3e}")
s1 = np.polyfit(np.log([s[0] for s in spl]), np.log([s[1] for s in spl]), 1)[0]
s2 = np.polyfit(np.log([s[0] for s in spl]), np.log([s[2] for s in spl]), 1)[0]
print(f"     наклоны: Ли {s1:.3f}, Стрэнг {s2:.3f}")
assert abs(s1 - 1) < 0.1 and abs(s2 - 2) < 0.1
# без накачки расщепление точное
H0 = -w0 / 2 * Z
E0 = expm(lindbladian(H0, Ls) * t_tot)
step = expm(Ds * 0.1) @ expm(lindbladian(H0, []) * 0.1)
assert np.allclose(np.linalg.matrix_power(step, 20), E0)
print("     без накачки (H = -(w/2) Z) расщепление точное: все части генератора коммутируют")

# ---------------------------------------------------------------------------
section("4. Микроскопический вывод: распад в широкополосную баню (Вигнер–Вайскопф)")
# |e> связан с N модами |k> (одновозбуждённый сектор), частоты равномерно в [-W/2, W/2]
N_modes, W = 2001, 100.0
delta = W / (N_modes - 1)
gc = 0.06
Gamma = 2 * pi * gc ** 2 / delta
Hww = np.zeros((N_modes + 1, N_modes + 1))
Hww[1:, 1:] = np.diag(np.linspace(-W / 2, W / 2, N_modes))
Hww[0, 1:] = Hww[1:, 0] = gc
ev, U = np.linalg.eigh(Hww)
ts = np.linspace(0, 4 / Gamma, 60)
ce = np.array([(U[0, :] ** 2 * np.exp(-1j * ev * t)).sum() for t in ts])
Pe = np.abs(ce) ** 2
dev = np.abs(Pe - np.exp(-Gamma * ts)).max()
t_rec = 2 * pi / delta
print(f"  N = {N_modes} мод, ширина W = {W}, шаг delta = {delta:.4f}, g = {gc}: Gamma = 2 pi g^2/delta = {Gamma:.4f}")
print(f"  max|P_e(t) - e^(-Gamma t)| на [0, 4/Gamma] = {dev:.4f};  время возврата 2 pi/delta = {t_rec:.0f} >> 4/Gamma = {4 / Gamma:.1f}")
assert dev < 0.01 and t_rec > 10 / Gamma
# узкая баня: ширина ~ g -> немарковская динамика (осцилляции)
W2, N2 = 0.3, 801
d2 = W2 / (N2 - 1)
g2 = sqrt(Gamma * d2 / (2 * pi))  # та же «золотая» скорость
H2 = np.zeros((N2 + 1, N2 + 1))
H2[1:, 1:] = np.diag(np.linspace(-W2 / 2, W2 / 2, N2))
H2[0, 1:] = H2[1:, 0] = g2
ev2, U2 = np.linalg.eigh(H2)
Pe2 = np.array([abs((U2[0, :] ** 2 * np.exp(-1j * ev2 * t)).sum()) ** 2 for t in ts])
dev2 = np.abs(Pe2 - np.exp(-Gamma * ts)).max()
print(f"  узкая баня W = {W2} (меньше Gamma = {Gamma:.3f}): max|P_e - e^(-Gamma t)| = {dev2:.3f} — марковское приближение неприменимо")
assert dev2 > 0.1
WW_dev, WW_G = dev, Gamma
# тепловая баня: GAD из Линдблада
print("  тепловая баня: L_down = sqrt(g0 (n+1)) sigma_-, L_up = sqrt(g0 n) sigma_+")
g0, wq = 1.0, 1.0
for Tb in (0.2, 0.5, 1.0):
    nb = 1 / (exp(wq / Tb) - 1)
    Lth = lindbladian(-wq / 2 * Z, [sqrt(g0 * (nb + 1)) * SM, sqrt(g0 * nb) * SP])
    ss = null_space(Lth)[:, 0].reshape(2, 2)
    ss = ss / np.trace(ss)
    boltz = np.array([1, exp(-wq / Tb)]) / (1 + exp(-wq / Tb))
    assert np.allclose(np.diag(ss).real, boltz)
    rho_e = evolve(Lth, np.diag([0, 1]).astype(complex), 1.0)
    g1_eff = -log((rho_e[1, 1].real - boltz[1]) / (1 - boltz[1]))
    assert np.isclose(g1_eff, g0 * (2 * nb + 1)) and np.isclose(g0 * (2 * nb + 1), g0 / np.tanh(wq / (2 * Tb)))
    pp = boltz[0]
    gt = 1 - exp(-g1_eff * 0.37)
    # точная проверка: диссипативная часть = A_(g, p) при H = 0
    Lth0 = lindbladian(0 * Z, [sqrt(g0 * (nb + 1)) * SM, sqrt(g0 * nb) * SP])
    assert np.allclose(choi_map(channel_of(Lth0, 0.37, 2), 2), choi_map(ch(K_gad(gt, pp)), 2))
    print(f"     T = {Tb}: n = {nb:.4f}, стационарное = Больцман diag({boltz[0]:.4f}, {boltz[1]:.4f}), 1/T1 = g0(2n+1) = g0 cth(w/2T) = {g1_eff:.4f}")
print("     и e^{tL} = A_(g,p) с g = 1 - e^{-t/T1}, p = 1/(1 + e^{-w/T})")
# пример: 5 ГГц при 50 мК
h, kB = 6.62607015e-34, 1.380649e-23
x5 = h * 5e9 / (kB * 0.05)
n5 = 1 / (exp(x5) - 1)
print(f"  кубит 5 ГГц при 50 мК: hf/kT = {x5:.3f}, n = {n5:.4f}, тепловая населённость |1> = {n5 / (2 * n5 + 1):.4f}")

# ---------------------------------------------------------------------------
section("5. T1, T2, T2*: инверсия-восстановление, Рамзи, спин-эхо")
T1, Tphi = 50.0, 60.0  # мкс
T2 = 1 / (1 / (2 * T1) + 1 / Tphi)
sig = 0.05  # рад/мкс — квазистатический шум расстройки
print(f"  модель: T1 = {T1}, T_phi = {Tphi} мкс -> T2 = {T2:.4f} мкс; квазистатическая расстройка sigma = {sig} рад/мкс")
Ls5 = [sqrt(1 / T1) * SM, sqrt(1 / (2 * Tphi)) * Z]


def Rx(th):
    return expm(-1j * th / 2 * X)


def free(delta_, t):
    # во вращающейся системе H = -(delta/2) Z
    return expm(lindbladian(-delta_ / 2 * Z, Ls5) * t)


def sup(U):
    return np.kron(U, U.conj())


gh_x, gh_w = np.polynomial.hermite_e.hermegauss(100)
gh_w = gh_w / gh_w.sum()


def p1(rho):
    return rho[1, 1].real


rho_g = np.diag([1, 0]).astype(complex)
# инверсия-восстановление
taus = np.linspace(0, 200, 41)
ir = [p1(((free(0, t) @ sup(Rx(pi))) @ rho_g.reshape(-1)).reshape(2, 2)) for t in taus]
pT1, _ = curve_fit(lambda t, a, T: a * np.exp(-t / T), taus, ir, p0=[1, 30])
# Рамзи (pi/2 - tau - pi/2) с намеренной расстройкой Delta и квазистатическим шумом
Dl = 0.3
ram = []
for t in taus:
    v = 0
    for x, w in zip(gh_x, gh_w):
        v += w * p1(((sup(Rx(pi / 2)) @ free(Dl + sig * x, t) @ sup(Rx(pi / 2))) @ rho_g.reshape(-1)).reshape(2, 2))
    ram.append(v)
ram = np.array(ram)
# аналитика: P1 = 1/2 (1 + cos(Delta t) e^{-t/T2} e^{-sigma^2 t^2/2}) с поправкой на T1 к z-компоненте
ram_th = []
for t in taus:
    # z после второго pi/2 = -(y-компонента до него) ... считаем по формуле Блоха явно
    zc = 1 - exp(-t / T1)  # возврат z к +1 из z = 0
    # после R_x(pi/2): z' = y sin + z cos -> вклад от y; y(t) = -cos(Delta t) e^{-t/T2} e^{-s^2t^2/2}
    y = -cos(Dl * t) * exp(-t / T2) * exp(-sig ** 2 * t ** 2 / 2)
    z = zc
    zz = y * sin(pi / 2) + z * cos(pi / 2)
    ram_th.append((1 - zz) / 2)
ram_th = np.array(ram_th)
assert np.abs(ram - ram_th).max() < 1e-6
print("  Рамзи: P_1(t) = [1 + cos(Delta t) e^{-t/T2} e^{-sigma^2 t^2/2}] / 2 — совпадение с численным < 1e-6")
# спин-эхо: pi/2 - t/2 - pi - t/2 - pi/2 (без намеренной расстройки)
echo = []
for t in taus:
    v = 0
    for x, w in zip(gh_x, gh_w):
        Ufr = free(sig * x, t / 2)
        v += w * p1(((sup(Rx(pi / 2)) @ Ufr @ sup(Rx(pi)) @ Ufr @ sup(Rx(pi / 2))) @ rho_g.reshape(-1)).reshape(2, 2))
    echo.append(v)
echo = np.array(echo)
# огибающая когерентности: амплитуда = |2 P_1 - 1| ... аппроксимация
pE, _ = curve_fit(lambda t, a, T, b: a * np.exp(-t / T) + b, taus, echo, p0=[0.5, 30, 0.5])
# Рамзи без намеренной расстройки: огибающая
ram0 = []
for t in taus:
    v = 0
    for x, w in zip(gh_x, gh_w):
        v += w * p1(((sup(Rx(pi / 2)) @ free(sig * x, t) @ sup(Rx(pi / 2))) @ rho_g.reshape(-1)).reshape(2, 2))
    ram0.append(v)
ram0 = np.array(ram0)
coh_r = 2 * ram0 - 1  # = e^{-t/T2} e^{-s^2 t^2/2}
coh_e = 1 - 2 * echo  # P_1 = (1 - e^{-t/T2})/2
assert np.allclose(coh_r, np.exp(-taus / T2) * np.exp(-sig ** 2 * taus ** 2 / 2), atol=1e-8)
assert np.allclose(coh_e, np.exp(-taus / T2), atol=1e-8)
pR, _ = curve_fit(lambda t, T: np.exp(-t / T), taus, coh_r, p0=[20])
from scipy.optimize import brentq
T2star = brentq(lambda t: exp(-t / T2) * exp(-sig ** 2 * t ** 2 / 2) - exp(-1), 1, 200)
print(f"  инверсия-восстановление: подгонка T1 = {pT1[1]:.3f} мкс (модель {T1})")
print(f"  спин-эхо: огибающая = e^(-t/T2) точно; подгонка T2 = {pE[1]:.3f} мкс (модель {T2:.3f})")
print(f"  Рамзи: огибающая = e^(-t/T2) e^(-sigma^2 t^2/2); T2* (уровень 1/e) = {T2star:.3f} мкс; чисто гауссово: sqrt(2)/sigma = {sqrt(2) / sig:.3f} мкс")
print(f"  проверка 1/T2 = 1/(2T1) + 1/T_phi: {1 / pE[1]:.6f} vs {1 / (2 * pT1[1]) + 1 / Tphi:.6f}")
assert abs(pT1[1] - T1) < 1e-3 and abs(pE[1] - T2) < 1e-3 and T2star < T2
# эхо не восстанавливает T1: инверсия-восстановление с pi посередине
ir_echo = [p1(((free(0, t / 2) @ sup(Rx(pi)) @ free(0, t / 2) @ sup(Rx(pi))) @ rho_g.reshape(-1)).reshape(2, 2)) for t in (20.0, 60.0)]
print(f"  T2 <= 2 T1: {T2:.2f} <= {2 * T1:.0f}; эхо не отменяет релаксацию: P_1 после 'pi, t/2, pi, t/2' при t = 60 мкс = {ir_echo[1]:.4f} != 0")
assert ir_echo[1] > 0.1
RAMSEY = dict(T1=pT1[1], T2=pE[1], T2s=T2star)

# ---------------------------------------------------------------------------
section("6. Раби с затуханием: стационарное решение уравнений Блоха")
for Om in (0.05, 0.2, 1.0):
    Ld = lindbladian(Om / 2 * X, Ls5)
    ss = null_space(Ld)
    ss = ss[:, 0].reshape(2, 2); ss = ss / np.trace(ss)
    r = bloch(ss)
    z_th = 1 / (1 + Om ** 2 * T1 * T2)
    y_th = -Om * T2 / (1 + Om ** 2 * T1 * T2)
    assert np.allclose(r, [0, y_th, z_th], atol=1e-10)
    print(f"  Omega = {Om}: r_ss = ({r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f});  z = 1/(1 + Om^2 T1 T2) = {z_th:.4f}, y = -Om T2/(1 + Om^2 T1 T2) = {y_th:.4f}")
# затухающие осцилляции Раби
Om = 0.5
Ld = lindbladian(Om / 2 * X, Ls5)
tt = np.linspace(0, 400, 801)
P1r = np.array([evolve(Ld, rho_g, t)[1, 1].real for t in tt])
P1_ss = (1 - 1 / (1 + Om ** 2 * T1 * T2)) / 2
print(f"  Omega = {Om}: P_1 осциллирует с периодом 2 pi/Omega = {2 * pi / Om:.2f} мкс, max P_1 = {P1r.max():.4f}; P_1(400 мкс) = {P1r[-1]:.4f} -> (1 - z_ss)/2 = {P1_ss:.4f}")
assert abs(P1r[-1] - P1_ss) < 1e-3

# ---------------------------------------------------------------------------
section("7. Немарковская динамика: кубит + затухающий резонатор")


def amp_map_(c):
    """Редуцированный канал кубита: rho_01 -> c rho_01, rho_11 -> |c|^2 rho_11."""
    return lambda r: np.array([[r[0, 0] + (1 - abs(c) ** 2) * r[1, 1], c * r[0, 1]], [np.conj(c) * r[1, 0], abs(c) ** 2 * r[1, 1]]])


def jc_amp(gq, kap, t):
    """Амплитуда |e,0> в одновозбуждённом секторе: H_eff = [[0, g], [g, -i kappa/2]] (резонанс)."""
    Heff = np.array([[0, gq], [gq, -1j * kap / 2]])
    return (expm(-1j * Heff * t) @ np.array([1, 0]))[0]


# сверка с полным уравнением Линдблада (кубит x резонатор, усечение Фока)
def jc_lindblad(gq, kap, nf, t):
    a = np.diag(np.sqrt(np.arange(1, nf)), 1).astype(complex)
    H = gq * (np.kron(SP, a) + np.kron(SM, dag(a)))  # sigma_+ = |1><0| — возбуждение кубита
    L = np.kron(I2, sqrt(kap) * a)
    rho = np.kron(np.diag([0, 1]), np.diag(ket(0, nf).real)).astype(complex)
    rt = evolve(lindbladian(H, [L]), rho, t)
    return rt.reshape(2, nf, 2, nf).trace(axis1=1, axis2=3)[1, 1].real


for nf in (2, 3, 5):
    vals = [jc_lindblad(1.0, 0.5, nf, t) for t in (1.0, 3.0)]
    ref = [abs(jc_amp(1.0, 0.5, t)) ** 2 for t in (1.0, 3.0)]
    assert np.allclose(vals, ref, atol=1e-10)
print("  полное уравнение Линдблада (усечение Фока 2, 3, 5) = эффективный 2x2 гамильтониан (один квант): совпадение < 1e-10")
# слабая связь: Парселл
gw, kw = 0.05, 2.0
tw = np.linspace(0, 6 / (4 * gw ** 2 / kw), 80)
Pw = np.array([abs(jc_amp(gw, kw, t)) ** 2 for t in tw])
fit = np.polyfit(tw[10:], np.log(Pw[10:]), 1)
print(f"  слабая связь g = {gw}, kappa = {kw}: скорость распада {-fit[0]:.6f}, Парселл 4g^2/kappa = {4 * gw ** 2 / kw:.6f}")
assert abs(-fit[0] / (4 * gw ** 2 / kw) - 1) < 0.01
# мера немарковости: D(rho1(t), rho2(t)) для |1> и |0>: = |c(t)|^2
gs, ks = 1.0, 0.2
ts7 = np.linspace(0, 20, 4001)
c7 = np.array([jc_amp(gs, ks, t) for t in ts7])
D7 = np.abs(c7) ** 2
# D для пары |1>, |0>: выходы A_gamma(t): z-компонента сжимается на |c|^2
# D(A(|1><1|), A(|0><0|)) = |c|^2 (z-компоненты 1 - 2|c|^2 и 1)
r_a = bloch(amp_map_(c7[1000])(np.diag([0, 1]).astype(complex))); r_b = bloch(amp_map_(c7[1000])(np.diag([1, 0]).astype(complex)))
assert np.isclose(0.5 * np.linalg.norm(r_a - r_b), D7[1000])
inc = np.clip(np.diff(D7), 0, None).sum()
dec_w = np.clip(np.diff(np.array([abs(jc_amp(gw, kw, t)) ** 2 for t in np.linspace(0, 200, 2001)])), 0, None).sum()
print(f"  сильная связь g = {gs}, kappa = {ks}: D(t) немонотонно; мера БЛП (сумма возрастаний D на [0, 20]) = {inc:.4f}")
print(f"  слабая связь: сумма возрастаний = {dec_w:.2e} (марковский режим)")
assert inc > 1.0 and dec_w < 1e-12
# промежуточное отображение Lambda(t2, t1) не CP: 1 - gamma = |c2|^2/|c1|^2 > 1
i1 = np.argmin(D7[:800]); i2 = i1 + np.argmax(D7[i1:i1 + 800])
ratio = D7[i2] / D7[i1]
print(f"  первый минимум |c|^2 = {D7[i1]:.1e} при t = {ts7[i1]:.3f}, затем возрождение до {D7[i2]:.4f} при t = {ts7[i2]:.3f}")
assert ratio > 1
# Choi промежуточного отображения на умеренной точке
ia, ib = 300, 500
ca, cb = c7[ia], c7[ib]
amp_map = amp_map_
Ja = choi_map(amp_map(ca), 2)
assert is_cptp_choi(Ja, 2)
# Lambda(tb, ta): c = cb/ca
cr = cb / ca
Jr = choi_map(amp_map(cr), 2)
print(f"  Lambda(t_b, t_a) = A(t_b) A(t_a)^-1 при t_a = {ts7[ia]:.2f}, t_b = {ts7[ib]:.2f}: |c_a|^2 = {abs(ca) ** 2:.4f}, |c_b|^2 = {abs(cb) ** 2:.4f},")
print(f"    |c_b/c_a|^2 = {abs(cr) ** 2:.2f} > 1, мин. собств. значение Чоя = {np.linalg.eigvalsh(Jr).min():.3f} — не канал (не CP-делимо)")
assert abs(cr) > 1 and np.linalg.eigvalsh(Jr).min() < 0

# ---------------------------------------------------------------------------
section("8. Коррелированный шум: коллективная дефазировка и подпространство без декогеренции")
gam = 0.1
Z1, Z2 = np.kron(Z, I2), np.kron(I2, Z)
L_ind = lindbladian(np.zeros((4, 4)), [sqrt(gam / 2) * Z1, sqrt(gam / 2) * Z2])
L_col = lindbladian(np.zeros((4, 4)), [sqrt(gam / 2) * (Z1 + Z2)])
phip = (ket(0, 4) + ket(3, 4)) / sqrt(2)
psip = (ket(1, 4) + ket(2, 4)) / sqrt(2)
t8 = 3.0
out = {}
for name, Ls8 in (("независимая", L_ind), ("коллективная", L_col)):
    rp = evolve(Ls8, np.outer(phip, phip.conj()), t8)
    rs = evolve(Ls8, np.outer(psip, psip.conj()), t8)
    cp = 2 * abs(rp[0, 3]); cs = 2 * abs(rs[1, 2])
    out[name] = (cp, cs)
    print(f"  {name:12s}: когерентность Phi+ = {cp:.4f}, Psi+ = {cs:.4f} (t = {t8})")
assert np.isclose(out["независимая"][0], exp(-2 * gam * t8)) and np.isclose(out["независимая"][1], exp(-2 * gam * t8))
assert np.isclose(out["коллективная"][0], exp(-4 * gam * t8)) and np.isclose(out["коллективная"][1], 1.0)
print(f"  e^(-2 g t) = {exp(-2 * gam * t8):.4f}, e^(-4 g t) = {exp(-4 * gam * t8):.4f}: Psi+ лежит в подпространстве без декогеренции")

# ---------------------------------------------------------------------------
section("9. Ответы к задачам семинара")
print(f"  3: твирлинг A_0.1: p_I = {pI_a:.6f}, p_X = p_Y = {pX_a:.6f}, p_Z = {pZ_a:.6f}; F_pro = {F_pro_kraus(K_ad(0.1)):.6f}")
print("  5: L = sqrt(k) Z: rho_01 затухает со скоростью 2k")
Lk = lindbladian(0 * Z, [sqrt(0.3) * Z])
assert np.isclose(evolve(Lk, rho_of([1, 0, 0]), 1.0)[0, 1], 0.5 * exp(-2 * 0.3))
Om7, T17, T27 = 0.1, 50.0, 37.5
z7 = 1 / (1 + Om7 ** 2 * T17 * T27)
print(f"  7: Omega = {Om7}, T1 = {T17}, T2 = {T27}: z_ss = {z7:.4f}, y_ss = {-Om7 * T27 * z7:.4f}, P_1 = {(1 - z7) / 2:.4f}")
Ld7 = lindbladian(Om7 / 2 * X, [sqrt(1 / T17) * SM, sqrt((1 / T27 - 1 / (2 * T17)) / 2) * Z])
ss7 = null_space(Ld7)[:, 0].reshape(2, 2); ss7 /= np.trace(ss7)
assert np.allclose(bloch(ss7), [0, -Om7 * T27 * z7, z7])
print(f"  8: sigma = {sig} рад/мкс: чисто гауссова огибающая e^(-sigma^2 t^2/2), T2* = sqrt(2)/sigma = {sqrt(2) / sig:.2f} мкс")
print(f"  9: 5 ГГц, 50 мК: n = {n5:.4f}; при 20 мК: n = {1 / (exp(h * 5e9 / (kB * 0.02)) - 1):.2e}; T1(T)/T1(0) = th(hf/2kT) = {np.tanh(x5 / 2):.4f}")
print(f"  13: коллективная дефазировка: Phi+ e^(-4 g t), Psi+ не затухает; независимая: обе e^(-2 g t)")

print(f"\nВсе проверки пройдены. Время работы: {time.time() - T_START:.1f} с")
