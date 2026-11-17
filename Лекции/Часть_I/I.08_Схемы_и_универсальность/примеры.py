"""
Лекция I.8. Квантовые схемы и универсальность: ZYZ, управляемые гейты, Toffoli,
отложенное измерение, двухуровневые разложения, плотность {H, T}, считающий
аргумент, алгоритм Соловея–Китаева, T-count поворотов, вставка SWAP.
Численные иллюстрации ко всем утверждениям лекции.

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
Порядок кубитов: первый множитель — старший (как в N&C и np.kron).
"""
import numpy as np
from itertools import product
from scipy.spatial import cKDTree
from scipy.stats import unitary_group

rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (X + Z) / np.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])
P0, P1 = np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)


def dag(A):
    return A.conj().T


def kron(*ops):
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def Rx(t): return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * X
def Ry(t): return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * Y
def Rz(t): return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * Z


def op(U, k, n):
    return kron(*[U if i == k else I2 for i in range(n)])


def controlled(U, c, t, n):
    """Управляемый однокубитный U: управление c, цель t, всего n кубитов."""
    a = kron(*[P0 if i == c else I2 for i in range(n)])
    b = kron(*[P1 if i == c else (U if i == t else I2) for i in range(n)])
    return a + b


def cnot(c, t, n): return controlled(X, c, t, n)


def dist(U, V):
    """d(U,V) = min_phi ||U - e^{i phi} V||_inf (I.1, пример 10.2)."""
    w = np.angle(np.linalg.eigvals(U @ dag(V)))
    w = np.sort(np.mod(w, 2 * np.pi))
    gaps = np.diff(np.concatenate([w, [w[0] + 2 * np.pi]]))
    spread = 2 * np.pi - np.max(gaps)            # длина наименьшей дуги, содержащей все собственные фазы
    return 2 * np.sin(min(spread, np.pi) / 4)


def equal_up_to_phase(U, V, tol=1e-9):
    return dist(U, V) < tol


def random_u(d):
    return unitary_group.rvs(d, random_state=rng)


# ---------------------------------------------------------------------------
section("1. ZYZ-разложение и следствие U = e^{ia} A X B X C, ABC = I")
def zyz(U):
    """U = e^{i a} Rz(b) Ry(c) Rz(d)."""
    a = np.angle(np.linalg.det(U)) / 2
    V = U * np.exp(-1j * a)                       # V из SU(2)
    c = 2 * np.arctan2(abs(V[1, 0]), abs(V[0, 0]))
    s = np.angle(V[1, 1]) if abs(V[1, 1]) > 1e-12 else 0.0     # (b+d)/2
    dlt = np.angle(V[1, 0]) if abs(V[1, 0]) > 1e-12 else 0.0  # (b-d)/2
    b, d = s + dlt, s - dlt
    W = np.exp(1j * a) * Rz(b) @ Ry(c) @ Rz(d)
    if not np.allclose(W, U):                     # ветвь корня из det: сдвиг фазы на pi
        a += np.pi
        W = np.exp(1j * a) * Rz(b) @ Ry(c) @ Rz(d)
    return a, b, c, d
for _ in range(1000):
    U = random_u(2)
    a, b, c, d = zyz(U)
    assert np.allclose(np.exp(1j * a) * Rz(b) @ Ry(c) @ Rz(d), U)
    A = Rz(b) @ Ry(c / 2)
    B = Ry(-c / 2) @ Rz(-(d + b) / 2)
    C = Rz((d - b) / 2)
    assert np.allclose(A @ B @ C, I2)
    assert np.allclose(np.exp(1j * a) * A @ X @ B @ X @ C, U)
print("  1000 случайных U: U = e^{ia} Rz(b) Ry(c) Rz(d) и U = e^{ia} A X B X C, ABC = I")
assert np.allclose(X @ Ry(0.7) @ X, Ry(-0.7)) and np.allclose(X @ Rz(0.7) @ X, Rz(-0.7))
print("  ключ к следствию: X Ry(t) X = Ry(-t), X Rz(t) X = Rz(-t)")

# ---------------------------------------------------------------------------
section("2. Управляемые гейты: CZ, разворот CNOT, SWAP, controlled-U, phase kickback")
CNOT01, CNOT10 = cnot(0, 1, 2), cnot(1, 0, 2)
CZ = np.diag([1, 1, 1, -1]).astype(complex)
assert np.allclose(controlled(Z, 0, 1, 2), controlled(Z, 1, 0, 2))
assert np.allclose(CNOT01, kron(I2, H) @ CZ @ kron(I2, H))
assert np.allclose(CNOT10, kron(H, H) @ CNOT01 @ kron(H, H))
SWAP = np.eye(4)[[0, 2, 1, 3]]
assert np.allclose(CNOT01 @ CNOT10 @ CNOT01, SWAP)
print("  CZ симметричен; CNOT = (I(x)H) CZ (I(x)H); (H(x)H) CNOT (H(x)H) — CNOT в обратную сторону; SWAP = 3 CNOT")
for _ in range(200):
    U = random_u(2)
    a, b, c, d = zyz(U)
    A, B, C = Rz(b) @ Ry(c / 2), Ry(-c / 2) @ Rz(-(d + b) / 2), Rz((d - b) / 2)
    circ = kron(np.diag([1, np.exp(1j * a)]), A) @ CNOT01 @ kron(I2, B) @ CNOT01 @ kron(I2, C)
    assert np.allclose(circ, controlled(U, 0, 1, 2))
print("  controlled-U = [diag(1,e^{ia}) (x) A] CNOT [I(x)B] CNOT [I(x)C]: два CNOT (200 случайных U)")
for _ in range(100):
    U = random_u(2)
    w, V = np.linalg.eig(U)
    u = V[:, 0]
    plus = np.array([1, 1]) / np.sqrt(2)
    out = controlled(U, 0, 1, 2) @ kron(plus, u)
    assert np.allclose(out, kron(np.array([1, w[0]]) / np.sqrt(2), u))
minus = np.array([1, -1]) / np.sqrt(2)
for x in (0, 1):
    v = kron(np.eye(2)[x], minus)
    assert np.allclose(CNOT01 @ v, (-1) ** x * v)
print("  phase kickback: C-U |+>|u> = (|0> + e^{i phi}|1>)/sqrt2 |u>;  CNOT |x>|-> = (-1)^x |x>|->")

# ---------------------------------------------------------------------------
section("3. Toffoli: 6 CNOT + 7 T; относительно-фазовые варианты; AND с раскомпьютацией измерением")
n = 3
TOF = np.eye(8, dtype=complex); TOF[[6, 7]] = TOF[[7, 6]]
Td = dag(T)
seq = [op(H, 2, n), cnot(1, 2, n), op(Td, 2, n), cnot(0, 2, n), op(T, 2, n), cnot(1, 2, n), op(Td, 2, n),
       cnot(0, 2, n), op(T, 1, n), op(T, 2, n), op(H, 2, n), cnot(0, 1, n), op(T, 0, n), op(Td, 1, n),
       cnot(0, 1, n)]
U = np.eye(8, dtype=complex)
for g in seq:
    U = g @ U
assert np.allclose(U, TOF)
nT = sum(1 for g in seq if any(np.allclose(g, op(G, k, n)) for G in (T, Td) for k in range(n)))
nC = sum(1 for g in seq if any(np.allclose(g, cnot(c, t, n)) for c in range(n) for t in range(n) if c != t))
print(f"  стандартное разложение (N&C, рис. 4.9): точно Toffoli; CNOT = {nC}, T/T^dag = {nT}")
assert (nC, nT) == (6, 7)
# относительно-фазовый Toffoli: 4 T и 3 CNOT (Марголус); с точностью до диагональной фазы
marg = [op(Ry(np.pi / 4), 2, n), cnot(1, 2, n), op(Ry(np.pi / 4), 2, n), cnot(0, 2, n),
        op(Ry(-np.pi / 4), 2, n), cnot(1, 2, n), op(Ry(-np.pi / 4), 2, n)]
M = np.eye(8, dtype=complex)
for g in marg:
    M = g @ M
D = M @ dag(TOF)
assert np.allclose(D, np.diag(np.diag(D)))
print("  гейт Марголуса (3 CNOT, 4 поворота Ry(±pi/4)) = Toffoli x диагональная фаза", np.round(np.diag(D).real, 3))
# 4 T: фазовый полином CCZ без членов, не содержащих цель: CCZ * CS^dag(a,b)
pp = [op(H, 2, n), op(T, 2, n), cnot(0, 2, n), op(Td, 2, n), cnot(1, 2, n), op(T, 2, n), cnot(0, 2, n),
      op(Td, 2, n), cnot(1, 2, n), op(H, 2, n)]
Q = np.eye(8, dtype=complex)
for g in pp:
    Q = g @ Q
CSd = np.diag([1, 1, 1, -1j]).astype(complex)      # управляемый S^dag на (a, b)
assert np.allclose(Q, TOF @ kron(CSd, I2))
print("  4 T + 4 CNOT: Toffoli с точностью до управляемого S^dag на управляющих кубитах")
# раскомпьютация AND без T: H на цели, измерение; при исходе 1 — CZ(a,b)
for _ in range(50):
    psi = rng.normal(size=4) + 1j * rng.normal(size=4); psi /= np.linalg.norm(psi)
    st = TOF @ kron(psi, np.array([1, 0]))          # |a b>|ab>
    st = op(H, 2, n) @ st
    for m in (0, 1):
        v = st.reshape(4, 2)[:, m]
        pm = np.vdot(v, v).real
        v = v / np.sqrt(pm)
        if m == 1:
            v = CZ @ v
        assert np.isclose(pm, 0.5) and np.isclose(abs(np.vdot(psi, v)), 1)
print("  раскомпьютация a AND b: H + измерение + (при исходе 1) CZ — T-гейты не нужны")

# ---------------------------------------------------------------------------
section("4. Принцип отложенного измерения")
for _ in range(100):
    psi = rng.normal(size=4) + 1j * rng.normal(size=4); psi /= np.linalg.norm(psi)
    U = random_u(2)
    # (а) измерить кубит 0, затем классически управляемый U на кубите 1
    rho_a, probs_a = np.zeros((2, 2), complex), []
    for m in (0, 1):
        v = psi.reshape(2, 2)[m]
        pm = np.vdot(v, v).real; probs_a.append(pm)
        v = (U if m else I2) @ v
        rho_a += np.outer(v, v.conj())
    # (б) квантово управляемый U, измерение в конце
    out = controlled(U, 0, 1, 2) @ psi
    probs_b = [np.vdot(out.reshape(2, 2)[m], out.reshape(2, 2)[m]).real for m in (0, 1)]
    rho_b = sum(np.outer(out.reshape(2, 2)[m], out.reshape(2, 2)[m].conj()) for m in (0, 1))
    assert np.allclose(probs_a, probs_b) and np.allclose(rho_a, rho_b)
print("  измерение + классическое управление == квантовое управление + измерение в конце (100 случаев)")

# ---------------------------------------------------------------------------
section("5. Двухуровневые унитарные: разложение d x d за <= d(d-1)/2 шагов")
def two_level_decompose(U):
    """Возвращает список двухуровневых V_k с V_K ... V_1 U = I (с точностью до диагонали)."""
    d = U.shape[0]
    W = U.copy()
    mats = []
    for col in range(d - 1):
        for row in range(col + 1, d):
            a, b = W[col, col], W[row, col]
            if abs(b) < 1e-14:
                continue
            r = np.sqrt(abs(a) ** 2 + abs(b) ** 2)
            V = np.eye(d, dtype=complex)
            V[col, col], V[col, row] = a.conj() / r, b.conj() / r
            V[row, col], V[row, row] = b / r, -a / r
            W = V @ W
            mats.append(V)
        # фазу диагонального элемента убираем двухуровневым с соседней строкой
        ph = W[col, col]
        if not np.isclose(ph, 1):
            V = np.eye(d, dtype=complex)
            V[col, col], V[col + 1, col + 1] = ph.conj(), ph
            W = V @ W
            mats.append(V)
    return mats, W
for d in [3, 4, 8]:
    U = random_u(d)
    mats, W = two_level_decompose(U)
    assert np.allclose(W, np.diag(np.diag(W)))            # осталась диагональ
    assert np.isclose(abs(W[-1, -1]), 1) and np.allclose(np.diag(W)[:-1], 1)
    prod = np.eye(d, dtype=complex)
    for V in mats:
        prod = V @ prod
    assert np.allclose(prod @ U, W)
    rot = [V for V in mats if np.count_nonzero(np.abs(V - np.eye(d)) > 1e-12) > 2]
    print(f"  d = {d}: исключений (повороты) {len(rot)} <= d(d-1)/2 = {d * (d - 1) // 2}; всего двухуровневых {len(mats)}")
    assert len(rot) <= d * (d - 1) // 2

# ---------------------------------------------------------------------------
section("6. Плотность: поворот T H T H на угол theta, cos(theta/2) = cos^2(pi/8)")
R = T @ H @ T @ H
R = R / np.sqrt(np.linalg.det(R))
theta = 2 * np.arccos(np.real(np.trace(R)) / 2)
assert np.isclose(np.cos(theta / 2), np.cos(np.pi / 8) ** 2)
print(f"  theta = {theta:.10f} рад, theta/pi = {theta / np.pi:.10f} (иррационально)")
for N in [10, 100, 1000, 10000]:
    pts = np.sort(np.mod(theta * np.arange(1, N + 1), 4 * np.pi))
    gap = np.max(np.diff(np.concatenate([[0], pts, [4 * np.pi]])))
    print(f"  первые {N:5d} степеней: наибольший промежуток на окружности {gap:.4f}")
# рациональность theta/pi с малым знаменателем исключена
for q in range(1, 2000):
    assert abs(theta / np.pi * q - round(theta / np.pi * q)) > 1e-9

# ---------------------------------------------------------------------------
section("7. Считающий аргумент: нижняя оценка числа гейтов")
def lower_bound_gates(n, eps, K=3):
    D = 4 ** n - 1                                     # размерность SU(2^n)
    return D * np.log2(1 / eps) / np.log2(K * n * n)
for n in [2, 5, 10, 20]:
    print(f"  n = {n:2d}, eps = 0.01: нужно g >= ~{lower_bound_gates(n, 0.01):.2e} гейтов для почти всех U")
print("  (для состояний — то же с 2^{n+1} - 2 вместо 4^n - 1)")

# ---------------------------------------------------------------------------
section("8. Сеть Clifford+T: нормальные формы Мацумото–Амано; T-count против точности")
def su2(U):
    return U / np.sqrt(np.linalg.det(U))
def quat(U):
    V = su2(U)
    return np.array([V[0, 0].real, V[0, 1].imag, V[0, 1].real, V[0, 0].imag])
# 24 клиффорда (по модулю фазы) с длинами слов в {H, S}
cliffords, frontier = [(I2, 0)], [(I2, 0)]
def known(U, lst):
    return any(equal_up_to_phase(U, V) for V, _ in lst)
while frontier:
    new = []
    for U, l in frontier:
        for G in (H, S):
            V = G @ U
            if not known(V, cliffords) and not known(V, new):
                new.append((V, l + 1))
    cliffords += new
    frontier = new
assert len(cliffords) == 24
CL = np.array([c for c, _ in cliffords]); CL_len = np.array([l for _, l in cliffords])
HT, SHT = H @ T, S @ H @ T
KMAX = 12
mats, tcount, glen = [], [], []
layer = np.array([I2, T]); layer_t = np.array([0, 1]); layer_g = np.array([0, 1])
for k in range(KMAX + 1):
    for ci in range(24):
        mats.append(layer @ CL[ci]); tcount.append(layer_t); glen.append(layer_g + CL_len[ci])
    if k < KMAX:
        layer = np.concatenate([layer @ HT, layer @ SHT])
        layer_t = np.concatenate([layer_t + 1, layer_t + 1])
        layer_g = np.concatenate([layer_g + 2, layer_g + 3])
mats = np.concatenate(mats); tcount = np.concatenate(tcount); glen = np.concatenate(glen)
Q = np.array([quat(U) for U in mats])
print(f"  сеть: {len(mats)} элементов с T-count <= {KMAX + 1}")
# различные элементы (нормальная форма единственна)
first = np.argmax(np.abs(Q) > 1e-9, axis=1)                    # первая ненулевая компонента
sgn = np.sign(Q[np.arange(len(Q)), first])
uniq = len(np.unique(np.round(Q * sgn[:, None], 8), axis=0))
print(f"  попарно различных по модулю фазы: {uniq}")
assert uniq == len(mats)
trees = {}
def nearest(U, tmax=KMAX + 1):
    if tmax not in trees:
        idx = np.where(tcount <= tmax)[0]
        trees[tmax] = (cKDTree(np.concatenate([Q[idx], -Q[idx]])), np.concatenate([idx, idx]))
    tree, idx = trees[tmax]
    _, j = tree.query(quat(U))
    return idx[j]
# T-count против точности: точный (исчерпывающий) поиск по сети
targets_U = [random_u(2) for _ in range(300)]
targets_Rz = [Rz(t) for t in rng.uniform(0, 2 * np.pi, 300)]
def mean_bits(targets, tmax):
    nearest(I2, tmax)                                  # построить дерево
    tree, _ = trees[tmax]
    d, _ = tree.query(np.array([quat(U) for U in targets]))
    return np.mean(np.log2(1 / d))                     # хордовое расстояние = d(U,V) в SU(2)
print("  T-count   log2(1/eps): случайные U   R_z(theta)   3 log2(1/eps) для R_z")
xs, ys = [], []
for tmax in range(5, KMAX + 1):
    bu, bz = mean_bits(targets_U, tmax), mean_bits(targets_Rz, tmax)
    xs.append(bu); ys.append(tmax)
    print(f"     {tmax:2d}            {bu:5.2f}            {bz:5.2f}          {3 * bz:5.1f}")
    assert tmax <= 3 * bz + 1e-9
slope = np.polyfit(xs, ys, 1)[0]
print(f"  случайные U: наклон T-count по log2(1/eps) = {slope:.2f} (асимптотически 3: объём сети ~ 2^t, размерность 3)")
assert 2.8 < slope < 4.0
print("  R_z: оптимальный T-count пока ниже 3 log2(1/eps); асимптотика Росса–Селинджера 3 log2(1/eps) + O(log log)")
# проверка: хордовое расстояние кватернионов совпадает с d(U,V)
for _ in range(20):
    U, V = random_u(2), random_u(2)
    qd = min(np.linalg.norm(quat(U) - quat(V)), np.linalg.norm(quat(U) + quat(V)))
    assert np.isclose(qd, dist(U, V))

# ---------------------------------------------------------------------------
section("9. Алгоритм Соловея–Китаева (Доусон–Нильсен)")
def axis_angle(U):
    V = su2(U)
    if V[0, 0].real < 0:
        V = -V
    c = np.clip(V[0, 0].real, -1, 1)
    th = 2 * np.arccos(c)
    # V = cos I - i sin (nx X + ny Y + nz Z): V01 = -i sin (nx - i ny) = -sin ny - i sin nx
    v = np.array([-V[0, 1].imag, -V[0, 1].real, -V[0, 0].imag])
    s = np.linalg.norm(v)
    return th, (v / s if s > 1e-15 else np.array([0, 0, 1.0]))
def rot(nvec, th):
    nvec = nvec / np.linalg.norm(nvec)
    return np.cos(th / 2) * I2 - 1j * np.sin(th / 2) * (nvec[0] * X + nvec[1] * Y + nvec[2] * Z)
def gc_decompose(D):
    """D = V W V^dag W^dag, V и W — повороты на угол ~ sqrt(угла D)."""
    th, nD = axis_angle(D)
    s = np.sqrt((1 - np.cos(th / 2)) / 2)
    phi = 2 * np.arcsin(np.sqrt(s))
    V0, W0 = Rx(phi), Ry(phi)
    G = V0 @ W0 @ dag(V0) @ dag(W0)
    thG, nG = axis_angle(G)
    ax = np.cross(nG, nD)
    if np.linalg.norm(ax) < 1e-12:
        Sm = I2 if nG @ nD > 0 else rot(np.array([1.0, 0, 0]) if abs(nG[0]) < 0.9 else np.array([0, 1.0, 0]), np.pi)
    else:
        Sm = rot(ax, np.arccos(np.clip(nG @ nD, -1, 1)))
    return Sm @ V0 @ dag(Sm), Sm @ W0 @ dag(Sm)
for _ in range(50):
    D = rot(rng.normal(size=3), rng.uniform(0, 0.3))
    V, W = gc_decompose(D)
    assert equal_up_to_phase(V @ W @ dag(V) @ dag(W), D, 1e-8)
print("  групповой коммутатор: D = V W V^dag W^dag проверен на 50 поворотах")
BASE_T = 12                   # базовая сеть: T-count <= 12
def sk(U, level):
    """Возвращает (матрица, T-count, длина)."""
    if level == 0:
        j = nearest(U, BASE_T)
        return mats[j], tcount[j], glen[j]
    Un, tn, ln = sk(U, level - 1)
    V, W = gc_decompose(U @ dag(Un))
    Vn, tv, lv = sk(V, level - 1)
    Wn, tw, lw = sk(W, level - 1)
    return Vn @ Wn @ dag(Vn) @ dag(Wn) @ Un, 2 * tv + 2 * tw + tn, 2 * lv + 2 * lw + ln
targets = [random_u(2) for _ in range(20)]
prev = None
for level in range(4):
    res = [sk(U, level) for U in targets]
    err = np.median([dist(r[0], U) for r, U in zip(res, targets)])
    tc = np.median([r[1] for r in res]); ln = np.median([r[2] for r in res])
    extra = f"; eps/eps_prev^1.5 = {err / prev ** 1.5:.2f}" if prev else ""
    print(f"  уровень {level}: медианная ошибка {err:.2e}, T-count {tc:.0f}, длина {ln:.0f}{extra}")
    if prev:
        assert err < prev
    prev = err
print("  длина растёт ~5x за уровень, ошибка ~ eps^{3/2}: L = O(log^{ln5/ln1.5}(1/eps)) = O(log^{3.97}(1/eps))")
print(f"  ln5/ln(3/2) = {np.log(5) / np.log(1.5):.3f}")
assert np.isclose(np.log(5) / np.log(1.5), 3.97, atol=0.005)

# ---------------------------------------------------------------------------
section("10. Связность: CNOT через соседа и цена SWAP")
n = 3
lhs = cnot(0, 2, n)
rhs = cnot(1, 2, n) @ cnot(0, 1, n) @ cnot(1, 2, n) @ cnot(0, 1, n)
assert np.allclose(lhs, rhs)
print("  CNOT(0->2) на цепочке 0-1-2 = CNOT(0,1) CNOT(1,2) CNOT(0,1) CNOT(1,2): 4 CNOT вместо 1")
sw = cnot(0, 1, n) @ cnot(1, 0, n) @ cnot(0, 1, n)
alt = sw @ cnot(1, 2, n) @ sw
assert np.allclose(alt, lhs)
print("  через SWAP: SWAP(0,1) CNOT(1,2) SWAP(0,1) — 7 CNOT; без обратного SWAP — 4 (раскладка меняется)")
for d in [2, 5, 10, 50]:
    print(f"  расстояние {d:2d}: через цепочку SWAP {3 * (d - 1) + 1:3d} CNOT (раскладка сдвинута), с возвратом {6 * (d - 1) + 1:3d}")

print("\nВсе проверки пройдены.")
