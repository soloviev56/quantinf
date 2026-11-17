"""
Лекция I.15. Моделирование гамильтонианов: разложение по паули-строкам и схема
для exp(-i theta P / 2), отображение Жордана–Вигнера, формулы Троттера–Сузуки
(1-й, 2-й, 4-й порядки) и коммутаторные оценки ошибки, модель Изинга в
поперечном поле (TFIM), LCU и блочное кодирование, кубитизация (полиномы
Чебышёва), молекула H2 в базисе STO-3G (интегралы вычисляются с нуля),
оценка фазы для энергии и проблема начального состояния.

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
Порядок кубитов: первый множитель — старший (как в N&C и np.kron).
"""
import numpy as np
from math import pi, sqrt, erf, exp, log
from functools import reduce
from itertools import product
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm

rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H1 = (X + Z) / np.sqrt(2)
S1 = np.diag([1, 1j])
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def kron(*ops):
    return reduce(np.kron, ops, np.array([[1.0 + 0j]]))


def pstr(s):
    return kron(*[PAULI[c] for c in s])


def op_on(n, j, o):
    """Оператор o на кубите j (с 0) из n."""
    return kron(*[o if k == j else I2 for k in range(n)])


def opnorm(A):
    return np.linalg.norm(A, 2)


def comm(A, B):
    return A @ B - B @ A


def pauli_decompose(Hm, n):
    out = {}
    for s in product("IXYZ", repeat=n):
        s = "".join(s)
        c = np.trace(pstr(s) @ Hm) / 2 ** n
        if abs(c) > 1e-12:
            out[s] = c
    return out


def cnot(n, c, t):
    P0 = np.diag([1, 0]).astype(complex); P1 = np.diag([0, 1]).astype(complex)
    a = [P0 if k == c else I2 for k in range(n)]
    b = [P1 if k == c else (X if k == t else I2) for k in range(n)]
    return kron(*a) + kron(*b)


# ---------------------------------------------------------------------------
section("1. Паули-разложение и схема для exp(-i theta P/2)")
n = 3
A = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
Hr = A + A.conj().T
dec = pauli_decompose(Hr, n)
assert np.allclose(sum(c * pstr(s) for s, c in dec.items()), Hr)
assert all(abs(c.imag) < 1e-12 for c in dec.values())
print(f"  случайный эрмитов 8x8: {len(dec)} паули-строк (из 64), коэффициенты Tr(P H)/2^n вещественны")


def pauli_exp_circuit(s, theta):
    """exp(-i theta P/2): смена базиса, лесенка CNOT, R_z, обратно. Возвращает (U, число CNOT)."""
    n = len(s)
    V = []
    for c in s:
        V.append({"I": I2, "Z": I2, "X": H1, "Y": S1 @ H1}[c])   # P_j = V Z V^dag
    Vt = kron(*V)
    act = [k for k, c in enumerate(s) if c != "I"]
    U = np.eye(2 ** n, dtype=complex)
    ladder = [cnot(n, act[i], act[i + 1]) for i in range(len(act) - 1)]
    Rz = op_on(n, act[-1], np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]))
    L = reduce(lambda a, b: b @ a, ladder, np.eye(2 ** n, dtype=complex))
    U = Vt @ L.conj().T @ Rz @ L @ Vt.conj().T
    return U, 2 * (len(act) - 1)


assert np.allclose(Y, S1 @ H1 @ Z @ H1.conj().T @ S1.conj().T)
for s in ("XYZ", "ZIZ", "YXI", "XXXX", "ZYIX"):
    th = rng.uniform(0, 2 * pi)
    U, nc = pauli_exp_circuit(s, th)
    assert np.allclose(U, expm(-1j * th / 2 * pstr(s)))
    print(f"  exp(-i theta {s}/2) = V (CNOT-лесенка)^dag R_z(theta) (CNOT-лесенка) V^dag: верно; CNOT = {nc} = 2(w-1)")

# ---------------------------------------------------------------------------
section("2. Отображение Жордана–Вигнера")
sig_minus = np.array([[0, 1], [0, 0]], dtype=complex)   # |0><1|: уничтожает частицу (|1> — занято)
assert np.allclose(sig_minus, (X + 1j * Y) / 2)
n = 4


def a_op(j, n):
    return kron(*([Z] * j + [sig_minus] + [I2] * (n - j - 1)))


ops = [a_op(j, n) for j in range(n)]
for i in range(n):
    for j in range(n):
        ac = ops[i] @ ops[j].conj().T + ops[j].conj().T @ ops[i]
        assert np.allclose(ac, np.eye(2 ** n) * (i == j))
        assert np.allclose(ops[i] @ ops[j] + ops[j] @ ops[i], 0)
print("  a_j = Z...Z (X+iY)/2 I...I: {a_i, a_j^dag} = delta_ij, {a_i, a_j} = 0 (n = 4)")
for j in range(n):
    assert np.allclose(ops[j].conj().T @ ops[j], (np.eye(2 ** n) - op_on(n, j, Z)) / 2)
print("  n_j = a_j^dag a_j = (I - Z_j)/2")
hop = ops[0].conj().T @ ops[1] + ops[1].conj().T @ ops[0]
assert np.allclose(hop, (pstr("XXII") + pstr("YYII")) / 2)
hop = ops[0].conj().T @ ops[3] + ops[3].conj().T @ ops[0]
assert np.allclose(hop, (pstr("XZZX") + pstr("YZZY")) / 2)
print("  a_1^dag a_2 + h.c. = (X1X2 + Y1Y2)/2;  a_1^dag a_4 + h.c. = (X Z Z X + Y Z Z Y)/2 — «хвост» Z")
# без хвоста антикоммутация нарушается
bad = [kron(*([I2] * j + [sig_minus] + [I2] * (n - j - 1))) for j in range(n)]
assert not np.allclose(bad[0] @ bad[1] + bad[1] @ bad[0], 0)
print("  без Z-хвоста (просто sigma^- на кубите) операторы на разных узлах коммутируют, а не антикоммутируют")

# ---------------------------------------------------------------------------
section("3. TFIM: H = -J sum Z_i Z_{i+1} - h sum X_i, открытая цепочка")


def tfim_sparse(n, J, h):
    N = 2 ** n
    idx = np.arange(N)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))) & 1
    zz = np.zeros(N)
    for i in range(n - 1):
        zz += (1 - 2 * bits[:, i]) * (1 - 2 * bits[:, i + 1])
    Hm = sp.diags(-J * zz).tocsr()
    for i in range(n):
        flip = idx ^ (1 << (n - 1 - i))
        Hm = Hm + sp.csr_matrix((-h * np.ones(N), (idx, flip)), shape=(N, N))
    return Hm


def tfim_parts(n, J, h):
    """Диагональная ZZ-часть (вектор) и X-часть (плотная матрица)."""
    N = 2 ** n
    idx = np.arange(N)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))) & 1
    zz = np.zeros(N)
    for i in range(n - 1):
        zz += (1 - 2 * bits[:, i]) * (1 - 2 * bits[:, i + 1])
    Bm = sum(op_on(n, i, X) for i in range(n)) * (-h)
    return -J * zz, Bm


n = 8
J = 1.0
E0 = {}
for h in (0.5, 1.0, 2.0):
    Hs = tfim_sparse(n, J, h)
    E0[h] = eigsh(Hs, k=1, which="SA")[0][0]
    print(f"  n = 8, h/J = {h}: E_0 = {E0[h]:.6f}")
Hd = tfim_sparse(4, 0.7, 1.3).toarray()
assert np.allclose(Hd, sum(-0.7 * op_on(4, i, Z) @ op_on(4, i + 1, Z) for i in range(3)) + sum(-1.3 * op_on(4, i, X) for i in range(4)))
# точное решение через свободные фермионы (Либ–Шульц–Маттис): проверка E0
def tfim_free_fermion_E0(n, J, h):
    # H = -J sum Z Z - h sum X; после поворота базиса — XY-модель с матрицей A, B (BdG)
    A = np.zeros((n, n)); B = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 2 * h
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = -J
        B[i, i + 1] = -J; B[i + 1, i] = J
    eps = np.sqrt(np.clip(np.linalg.eigvalsh((A - B) @ (A + B)), 0, None))
    return -0.5 * eps.sum()
for h in (0.5, 1.0, 2.0):
    ff = tfim_free_fermion_E0(n, J, h)
    assert abs(ff - E0[h]) < 1e-8, (ff, E0[h])
print("  E_0 совпадает с точным решением через свободные фермионы (Жордан–Вигнер + Боголюбов)")

# ---------------------------------------------------------------------------
section("4. Формулы Троттера: первый, второй и четвёртый порядок; коммутаторные оценки")
n = 8; J = 1.0; h = 1.0
zzA, Bm = tfim_parts(n, J, h)
Am = np.diag(zzA).astype(complex)
Hfull = Am + Bm
nA = opnorm(comm(Am, Bm))
nBBA = opnorm(comm(Bm, comm(Bm, Am)))
nAAB = opnorm(comm(Am, comm(Am, Bm)))
pairs = 2 * (n - 1)
print(f"  ||[A,B]|| = {nA:.3f};  сумма по парам ||[H_j,H_k]|| = 2(n-1)*2Jh = {pairs * 2:.0f};  ||[B,[B,A]]|| = {nBBA:.3f}, ||[A,[A,B]]|| = {nAAB:.3f}")
assert nA <= pairs * 2 + 1e-9

wB, VB = np.linalg.eigh(Bm)


def eB(t):
    return (VB * np.exp(-1j * wB * t)) @ VB.conj().T


def eA(t):
    return np.diag(np.exp(-1j * zzA * t))


def S1f(dt):
    return eA(dt) @ eB(dt)


def S2f(dt):
    return eB(dt / 2) @ eA(dt) @ eB(dt / 2)


p4 = 1 / (4 - 4 ** (1 / 3))


def S4f(dt):
    a = S2f(p4 * dt)
    return a @ a @ S2f((1 - 4 * p4) * dt) @ a @ a


wH, VH = np.linalg.eigh(Hfull)


def Uex(t):
    return (VH * np.exp(-1j * wH * t)) @ VH.conj().T


# оценки на один шаг (случайные t)
for dt in (0.01, 0.05, 0.1, 0.3):
    e1 = opnorm(Uex(dt) - S1f(dt)); b1 = dt ** 2 / 2 * nA
    e2 = opnorm(Uex(dt) - S2f(dt)); b2 = dt ** 3 / 12 * nBBA + dt ** 3 / 24 * nAAB
    assert e1 <= b1 * (1 + 1e-9) and e2 <= b2 * (1 + 1e-9)
print("  ||e^{-iHdt} - S1(dt)|| <= dt^2/2 ||[A,B]||;  ||e^{-iHdt} - S2(dt)|| <= dt^3/12 ||[B,[B,A]]|| + dt^3/24 ||[A,[A,B]]||: выполнено")
# то же для случайных матриц
for _ in range(20):
    a = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)); a = a + a.conj().T
    b = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)); b = b + b.conj().T
    for dt in (0.02, 0.1, 0.4):
        U = expm(-1j * (a + b) * dt)
        s1 = expm(-1j * a * dt) @ expm(-1j * b * dt)
        s2 = expm(-1j * a * dt / 2) @ expm(-1j * b * dt) @ expm(-1j * a * dt / 2)
        assert opnorm(U - s1) <= dt ** 2 / 2 * opnorm(comm(a, b)) * (1 + 1e-9)
        assert opnorm(U - s2) <= dt ** 3 * (opnorm(comm(b, comm(b, a))) / 12 + opnorm(comm(a, comm(a, b))) / 24) * (1 + 1e-9)
print("  те же оценки проверены на 20 парах случайных эрмитовых 6x6")

# ошибка от числа шагов r
print("\n  Ошибка ||U(t) - S(t/r)^r|| для TFIM n = 8, J = h = 1:")
res = {}
for t in (1.0, 5.0):
    Ut = Uex(t)
    rs = [4, 8, 16, 32, 64, 128, 256]
    errs = {1: [], 2: [], 4: []}
    for r in rs:
        dt = t / r
        for order, f in ((1, S1f), (2, S2f), (4, S4f)):
            errs[order].append(opnorm(Ut - np.linalg.matrix_power(f(dt), r)))
    res[t] = (rs, errs)
    for order in (1, 2, 4):
        e = np.array(errs[order])
        good = e > 1e-11
        sl = np.polyfit(np.log(np.array(rs)[good][-4:]), np.log(e[good][-4:]), 1)[0]
        print(f"   t = {t}: порядок {order}: " + " ".join(f"{x:.1e}" for x in e) + f"   наклон {sl:.2f}")
        if order == 1:
            assert abs(sl + 1) < 0.1
        if order == 2:
            assert abs(sl + 2) < 0.1
        if order == 4:
            assert abs(sl + 4) < 0.3
    print(f"   r = {rs}")
# сравнение с коммутаторной оценкой: t^2/(2r) ||[A,B]||
t = 1.0; rs, errs = res[t]
ratio1 = [errs[1][i] / (t ** 2 / (2 * r) * nA) for i, r in enumerate(rs)]
ratio2 = [errs[2][i] / (t ** 3 / r ** 2 * (nBBA / 12 + nAAB / 24)) for i, r in enumerate(rs)]
print(f"  t = 1: факт/оценка, 1-й порядок: {ratio1[-1]:.3f};  2-й порядок: {ratio2[-1]:.3f}")
assert max(ratio1) <= 1 and max(ratio2) <= 1
ratio1_t5 = res[5.0][1][1][-1] / (25 / (2 * 256) * nA)
print(f"  t = 5, r = 256: факт/оценка 1-го порядка {ratio1_t5:.3f}")

# число CNOT для eps = 1e-3: шаг 1-го/2-го порядка — 2(n-1) = 14 CNOT, 4-го — 5 * 14 = 70
cnot_step = 2 * (n - 1)
print("\n  Число шагов r и CNOT для ||U - S^r|| <= 1e-3 (n = 8, J = h = 1):")
cnot_res = {}
for t in (1.0, 5.0):
    Ut = Uex(t)
    for order, f, cs in ((1, S1f, cnot_step), (2, S2f, cnot_step), (4, S4f, 5 * cnot_step)):
        lo, hi = 1, 1
        while opnorm(Ut - np.linalg.matrix_power(f(t / hi), hi)) > 1e-3:
            lo, hi = hi, hi * 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if opnorm(Ut - np.linalg.matrix_power(f(t / mid), mid)) > 1e-3:
                lo = mid
            else:
                hi = mid
        cnot_res[(t, order)] = (hi, hi * cs)
        print(f"   t = {t}: порядок {order}: r = {hi:5d}, CNOT = {hi * cs}")
# оценка числа шагов по коммутаторной формуле для 1-го порядка
for t in (1.0, 5.0):
    r_bound = int(np.ceil(t ** 2 * nA / (2 * 1e-3)))
    print(f"   t = {t}: 1-й порядок по оценке t^2||[A,B]||/(2 eps): r = {r_bound} (факт {cnot_res[(t, 1)][0]})")
    assert r_bound >= cnot_res[(t, 1)][0]

# масштабирование с n: число некоммутирующих пар линейно
print("\n  Коммутаторная сумма растёт как n, а не как (число слагаемых)^2:")
for nn in (4, 6, 8, 10):
    zzA_, Bm_ = tfim_parts(nn, 1.0, 1.0)
    L = 2 * nn - 1
    nAB = opnorm(np.diag(zzA_) @ Bm_ - Bm_ @ np.diag(zzA_))
    print(f"   n = {nn:2d}: слагаемых L = {L}, пар L(L-1)/2 = {L * (L - 1) // 2}, некоммутирующих пар 2(n-1) = {2 * (nn - 1)}, ||[A,B]|| = {nAB:.2f}")

# ---------------------------------------------------------------------------
section("5. Динамика после квенча: <Z_1>(t) из |0...0>, h/J = 1")
n = 8
psi0 = np.zeros(2 ** n, complex); psi0[0] = 1
Z1 = np.diag(op_on(n, 0, Z)).real
ts = np.arange(0, 5.01, 0.5)
for dt in (0.25, 0.1):
    errs = []
    for tt in ts:
        r = int(round(tt / dt))
        ex = Uex(tt) @ psi0
        tr = np.linalg.matrix_power(S2f(dt), r) @ psi0 if r > 0 else psi0
        errs.append(abs(np.vdot(ex, Z1 * ex).real - np.vdot(tr, Z1 * tr).real))
    print(f"  шаг dt = {dt}: |<Z1>_exact - <Z1>_Trotter2| при t = 1, 2, 3, 4, 5: " + " ".join(f"{errs[int(k / 0.5)]:.1e}" for k in (1, 2, 3, 4, 5)))
vals = [np.vdot(Uex(tt) @ psi0, Z1 * (Uex(tt) @ psi0)).real for tt in (0, 1, 2, 3, 4, 5)]
print("  <Z1>(t), t = 0..5: " + " ".join(f"{v:.3f}" for v in vals))

# ---------------------------------------------------------------------------
section("6. LCU и блочное кодирование; кубитизация")
n = 3; J = 1.0; h = 0.6
terms = []
for i in range(n - 1):
    s = ["I"] * n; s[i] = s[i + 1] = "Z"; terms.append((-J, "".join(s)))
for i in range(n):
    s = ["I"] * n; s[i] = "X"; terms.append((-h, "".join(s)))
Hm = sum(c * pstr(s) for c, s in terms)
alpha = sum(abs(c) for c, _ in terms)
L = len(terms)
m = int(np.ceil(np.log2(L)))
Lp = 2 ** m
amps = np.zeros(Lp); amps[:L] = np.sqrt([abs(c) / alpha for c, _ in terms])
G = amps.astype(complex)
# SELECT = sum_j |j><j| (x) U_j, U_j = sign_j P_j; неиспользуемые индексы — I
SEL = np.zeros((Lp * 2 ** n, Lp * 2 ** n), complex)
for j in range(Lp):
    Uj = np.sign(terms[j][0]) * pstr(terms[j][1]) if j < L else np.eye(2 ** n)
    Pj = np.zeros((Lp, Lp)); Pj[j, j] = 1
    SEL += np.kron(Pj, Uj)
Gm = np.kron(G[:, None], np.eye(2 ** n))          # |G> (x) I
block = Gm.conj().T @ SEL @ Gm
assert np.allclose(block, Hm / alpha)
print(f"  TFIM n = 3, J = 1, h = 0.6: L = {L} слагаемых, alpha = sum|c_j| = {alpha:.2f}, {m} кубита-указателя")
print("  (<G| (x) I) SELECT (|G> (x) I) = H/alpha: блочное кодирование проверено")
# вероятность успеха постселекции для состояния psi: ||H psi||^2/alpha^2
wH3, VH3 = np.linalg.eigh(Hm)
p_succ_ground = (wH3[0] / alpha) ** 2
print(f"  на основном состоянии (E_0 = {wH3[0]:.4f}) постселекция успешна с вероятностью (E_0/alpha)^2 = {p_succ_ground:.3f}")
# кубитизация: W = (2|G><G| (x) I - I) SELECT;  (<G| (x) I) W^k (|G> (x) I) = T_k(H/alpha)
Refl = np.kron(2 * np.outer(G, G.conj()) - np.eye(Lp), np.eye(2 ** n))
W = Refl @ SEL
assert np.allclose(SEL @ SEL, np.eye(Lp * 2 ** n))
Hn = Hm / alpha
Tprev, Tcur = np.eye(2 ** n), Hn
Wk = np.eye(Lp * 2 ** n)
for k in range(0, 8):
    Tk = np.eye(2 ** n) if k == 0 else (Hn if k == 1 else None)
    if k >= 2:
        Tk = 2 * Hn @ Tcur - Tprev
        Tprev, Tcur = Tcur, Tk
    assert np.allclose(Gm.conj().T @ Wk @ Gm, Tk)
    Wk = W @ Wk
print("  W = (2|G><G| - I) SELECT:  (<G|) W^k (|G>) = T_k(H/alpha), k = 0..7 (многочлены Чебышёва)")
evW = np.linalg.eigvals(W)
for E in wH3:
    ph = np.arccos(E / alpha)
    assert np.min(np.abs(evW - np.exp(1j * ph))) < 1e-8 and np.min(np.abs(evW - np.exp(-1j * ph))) < 1e-8
print("  спектр W содержит exp(+-i arccos(E_k/alpha)) для всех собственных значений E_k оператора H")

# ---------------------------------------------------------------------------
section("7. Молекула H2, базис STO-3G: интегралы, гамильтониан на 4 кубитах")
ANG = 1 / 0.529177210903    # бор в ангстреме
sto_a = np.array([3.42525091, 0.62391373, 0.16885540])
sto_d = np.array([0.15432897, 0.53532814, 0.44463454])
norm = (2 * sto_a / pi) ** 0.75


def F0(t):
    return 1.0 if t < 1e-12 else 0.5 * sqrt(pi / t) * erf(sqrt(t))


def h2_integrals(R):
    """Атомные интегралы для двух 1s-орбиталей STO-3G на расстоянии R (бор)."""
    C = [np.array([0., 0, 0]), np.array([0., 0, R])]
    S = np.zeros((2, 2)); T = np.zeros((2, 2)); V = np.zeros((2, 2)); ERI = np.zeros((2, 2, 2, 2))
    prim = [(a, d * nn) for a, d, nn in zip(sto_a, sto_d, norm)]
    for i in range(2):
        for j in range(2):
            AB2 = np.sum((C[i] - C[j]) ** 2)
            for a, ca in prim:
                for b, cb in prim:
                    p = a + b; mu = a * b / p
                    P = (a * C[i] + b * C[j]) / p
                    K = exp(-mu * AB2)
                    s = (pi / p) ** 1.5 * K
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


def h2_qubit_hamiltonian(R_ang):
    R = R_ang * ANG
    S, hcore, eri, Enuc = h2_integrals(R)
    # МО по симметрии: sigma_g, sigma_u
    Cmo = np.array([[1, 1], [1, -1]], float)
    Cmo[:, 0] /= sqrt(2 * (1 + S[0, 1])); Cmo[:, 1] /= sqrt(2 * (1 - S[0, 1]))
    hmo = Cmo.T @ hcore @ Cmo
    emo = np.einsum("pi,qj,rk,sl,pqrs->ijkl", Cmo, Cmo, Cmo, Cmo, eri)   # (ij|kl), химическая запись
    # спин-орбитали: 0 = g up, 1 = g down, 2 = u up, 3 = u down
    nso = 4
    sp_orb = [(0, 0), (0, 1), (1, 0), (1, 1)]
    a = [a_op(j, nso) for j in range(nso)]
    Hq = Enuc * np.eye(16, dtype=complex)
    for p, q in product(range(nso), repeat=2):
        (P, sP), (Q, sQ) = sp_orb[p], sp_orb[q]
        if sP == sQ:
            Hq += hmo[P, Q] * a[p].conj().T @ a[q]
    for p, q, r, s in product(range(nso), repeat=4):
        (P, sP), (Q, sQ), (Rr, sR), (Ss, sS) = sp_orb[p], sp_orb[q], sp_orb[r], sp_orb[s]
        if sP == sR and sQ == sS:
            # 1/2 sum <pq|rs> a+_p a+_q a_s a_r,  <pq|rs> = (pr|qs)
            val = emo[P, Rr, Q, Ss]
            if abs(val) > 0:
                Hq += 0.5 * val * a[p].conj().T @ a[q].conj().T @ a[s] @ a[r]
    return Hq, S, hmo, emo, Enuc


Hq, S, hmo, emo, Enuc = h2_qubit_hamiltonian(0.7414)
assert np.allclose(Hq, Hq.conj().T)
Nop = sum(a_op(j, 4).conj().T @ a_op(j, 4) for j in range(4))
assert np.allclose(Hq @ Nop, Nop @ Hq)
hf = np.zeros(16); hf[int("1100", 2)] = 1
E_HF = np.vdot(hf, Hq @ hf).real
w, v = np.linalg.eigh(Hq)
two = [k for k in range(16) if np.isclose(np.vdot(v[:, k], Nop @ v[:, k]).real, 2)]
E_FCI = w[two[0]]
print(f"  R = 0.7414 А: S_12 = {S[0, 1]:.4f}, E_nuc = {Enuc:.4f} Ха")
print(f"  E_HF = {E_HF:.6f} Ха,  E_FCI = {E_FCI:.6f} Ха,  энергия корреляции {E_FCI - E_HF:.6f} Ха")
assert abs(E_HF - (-1.1167)) < 1e-3 and abs(E_FCI - (-1.1373)) < 1e-3
assert np.isclose(w[0], E_FCI)    # глобальный минимум — в секторе N = 2
# сверка с табличными интегралами Сабо–Остлунда при R = 1.4 бор
S14, h14, eri14, _ = h2_integrals(1.4)
assert abs(S14[0, 1] - 0.6593) < 2e-4 and abs(eri14[0, 0, 0, 0] - 0.7746) < 2e-4
print(f"  контроль при R = 1.4 бор: S_12 = {S14[0, 1]:.4f}, (11|11) = {eri14[0, 0, 0, 0]:.4f} (Сабо–Остлунд: 0,6593 и 0,7746)")
dec = pauli_decompose(Hq, 4)
dec = {k: c.real for k, c in dec.items()}
print(f"  паули-разложение: {len(dec)} слагаемых (включая I); sum|c_j| без I = {sum(abs(c) for k, c in dec.items() if k != 'IIII'):.4f}")
for k, c in sorted(dec.items(), key=lambda kv: -abs(kv[1])):
    print(f"     {k}: {c:+.6f}")
assert len(dec) == 15
# перекрытие HF с основным состоянием
g = v[:, two[0]]
ov = {0.7414: abs(np.vdot(hf, g)) ** 2}
for Rr in (1.5, 2.5):
    Hq2 = h2_qubit_hamiltonian(Rr)[0]
    w2, v2 = np.linalg.eigh(Hq2)
    k2 = [k for k in range(16) if np.isclose(np.vdot(v2[:, k], Nop @ v2[:, k]).real, 2)][0]
    ov[Rr] = abs(np.vdot(hf, v2[:, k2])) ** 2
print("  |<HF|E_0>|^2: " + ", ".join(f"R = {k} А: {x:.4f}" for k, x in ov.items()))

# ---------------------------------------------------------------------------
section("8. QPE для энергии: вероятность = перекрытие; проблема начального состояния")
# U = exp(-i H tau), фаза phi_k = -E_k tau/(2 pi) mod 1; распределение QPE (формула I.12, (5.3))
tau = 1.0
tq = 8
T = 2 ** tq


def qpe_probs(phi):
    m = np.arange(T)
    d = phi - m / T
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.sin(pi * T * d) ** 2
        den = (T * np.sin(pi * d)) ** 2
        p = np.where(np.abs(np.sin(pi * d)) < 1e-14, 1.0, num / den)
    return p


cs = np.abs(v.conj().T @ hf) ** 2
dist = sum(c * qpe_probs((-w[k] * tau / (2 * pi)) % 1) for k, c in enumerate(cs) if c > 1e-14)
phi0 = (-E_FCI * tau / (2 * pi)) % 1
m_near = [int(np.floor(phi0 * T)) % T, int(np.ceil(phi0 * T)) % T]
p_near = dist[m_near].sum()
p_ground_total = cs[two[0]]
mbest = int(np.argmax(dist))
E_est = -2 * pi * (mbest / T) / tau
if E_est < -pi / tau:
    E_est += 2 * pi / tau
print(f"  H2 при 0.7414 А, вход |HF>, tau = 1, t = {tq}: самый вероятный исход m = {mbest}, E = {E_est:.4f} Ха (точное {E_FCI:.4f}); шаг сетки 2 pi/(tau T) = {2 * pi / T:.4f} Ха")
print(f"  P(два ближайших к E_0 исхода) = {p_near:.3f} <= |<HF|E0>|^2 = {p_ground_total:.4f}")
assert p_near <= p_ground_total + 1e-12 and abs(E_est - E_FCI) < 2 * pi / T
# TFIM: максимальное перекрытие произведения одинаковых однокубитных состояний с основным состоянием
print("  TFIM, J = h = 1: max_a |<(cos a|0> + sin a|1>)^n | E_0>|^2 от n:")
ovs = []
ns = [4, 6, 8, 10, 12, 14]
for nn in ns:
    Hs = tfim_sparse(nn, 1.0, 1.0)
    e, vec = eigsh(Hs, k=1, which="SA")
    g0 = vec[:, 0]
    best = 0
    for aa in np.linspace(0, pi / 2, 181):
        q = np.array([np.cos(aa), np.sin(aa)])
        prod_state = reduce(np.kron, [q] * nn)
        best = max(best, abs(np.dot(prod_state, g0)) ** 2)
    ovs.append(best)
print("   " + "  ".join(f"n = {nn}: {o:.4f}" for nn, o in zip(ns, ovs)))
sl = np.polyfit(ns, np.log(ovs), 1)[0]
print(f"   убывание экспоненциальное: ln(перекрытие) ~ {sl:.4f} n, т.е. множитель {np.exp(sl):.4f} на кубит")
assert sl < 0 and all(np.diff(ovs) < 0)

# ---------------------------------------------------------------------------
section("9. Ответы к задачам семинара")
# задача 2: ||[Z1Z2, X1]|| = 2
print(f"  2: ||[Z Z, X I]|| = {opnorm(comm(pstr('ZZ'), pstr('XI'))):.1f}")
# задача 3: оценка первого порядка для TFIM общего n
print(f"  3: TFIM n = 8, J = h = 1: сумма по парам = 4(n-1)Jh = {4 * 7}; ||[A,B]|| = {nA:.3f}; при t = 1, eps = 1e-3: r >= {int(np.ceil(28 / 2e-3))} (через сумму) и {int(np.ceil(nA / 2e-3))} (через ||[A,B]||)")
# задача 5: Suzuki p4
assert abs(4 * p4 ** 3 + (1 - 4 * p4) ** 3) < 1e-12
print(f"  5: p_2 = 1/(4 - 4^(1/3)) = {p4:.6f}, 1 - 4 p_2 = {1 - 4 * p4:.6f}")
print("     4 p^3 + (1 - 4p)^3 = 0: главный член ошибки сокращается")
# задача 6: число CNOT
print(f"  6: e^(-i theta XYZ/2): CNOT = 4; e^(-i theta X Z Z X/2): CNOT = 6")
# задача 7: LCU alpha для TFIM n
for nn in (3, 8):
    print(f"  7: TFIM n = {nn}, J = 1, h = 0.6: alpha = (n-1)J + n h = {(nn - 1) + 0.6 * nn:.1f}, указатель на {int(np.ceil(np.log2(2 * nn - 1)))} кубитах")
print(f"  9: H2: |<HF|E0>|^2 = {ov[0.7414]:.4f} (0.7414 А), {ov[2.5]:.4f} (2.5 А)")
print("\nВсе проверки пройдены.")
