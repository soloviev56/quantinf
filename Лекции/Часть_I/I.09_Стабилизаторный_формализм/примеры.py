"""
Лекция I.9. Группа Паули, группа Клиффорда, стабилизаторный формализм:
симплектическая запись, таблицы сопряжения, порядок группы Клиффорда,
число стабилизаторных состояний, табличный симулятор Ааронсона–Готтесмана
(сверка с вектором состояния), энтропия запутанности стабилизаторных
состояний, магические состояния и стабилизаторный ранг, рост ранга Шмидта.

Запуск:  python примеры.py
Требуется: numpy. Все проверки оформлены как assert.
Порядок кубитов: первый множитель — старший (как в N&C и np.kron).
Нумерация кубитов в коде — с 0.
"""
import time
from math import prod, log2 as ilog2
import numpy as np
from itertools import product, combinations

rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (X + Z) / np.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def dag(A):
    return A.conj().T


def kron(*ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def op(U, k, n):
    return kron(*[U if i == k else I2 for i in range(n)])


def cnot(c, t, n):
    P0, P1 = np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)
    return kron(*[P0 if i == c else I2 for i in range(n)]) + \
        kron(*[P1 if i == c else (X if i == t else I2) for i in range(n)])


def pauli_string(s):
    return kron(*[PAULI[c] for c in s])


def pauli_xz(x, z):
    """P(x|z) = i^{x.z} X^x Z^z (эрмитов)."""
    x, z = np.asarray(x), np.asarray(z)
    M = kron(*[np.linalg.matrix_power(X, int(a)) @ np.linalg.matrix_power(Z, int(b)) for a, b in zip(x, z)])
    return (1j) ** int(x @ z) * M


def symp(u, v, n):
    return int(u[:n] @ v[n:] + u[n:] @ v[:n]) % 2


def canon(U):
    """Каноническая форма матрицы/вектора по модулю глобальной фазы (для хеширования)."""
    f = U.flatten()
    k = np.argmax(np.abs(f) > 1e-9)
    V = f * np.exp(-1j * np.angle(f[k]))
    return tuple(np.round(V, 6).view(float))


# ---------------------------------------------------------------------------
section("1. Группа Паули: симплектическая запись, коммутация, произведение")
n = 3
vecs = [np.array(v) for v in product([0, 1], repeat=2 * n)]
ops = {tuple(v): pauli_xz(v[:n], v[n:]) for v in vecs}
for v in vecs:
    P = ops[tuple(v)]
    assert np.allclose(P, dag(P)) and np.allclose(P @ P, np.eye(2 ** n))
assert np.allclose(pauli_xz([1], [1]), Y)
cnt = 0
for u in vecs:
    for v in vecs:
        A, B = ops[tuple(u)], ops[tuple(v)]
        comm = np.allclose(A @ B, B @ A)
        assert comm == (symp(u, v, n) == 0)
        # произведение: P(u) P(v) = i^{c} P(u+v), c — целое
        W = ops[tuple((u + v) % 2)]
        ph = np.trace(dag(W) @ A @ B) / 2 ** n
        assert np.isclose(abs(ph), 1) and np.isclose(ph ** 4, 1)
        cnt += 1
print(f"  n = {n}: все {len(vecs)} операторов P(x|z) эрмитовы, P^2 = I; P(1|1) = Y")
print(f"  {cnt} пар: [P(u),P(v)] = 0  <=>  <u,v> = x.z' + z.x' = 0 mod 2;  P(u)P(v) = i^c P(u+v)")
print(f"  |P_n| = 4^(n+1) = {4 ** (n + 1)} (с фазами i^k), |P_n/<iI>| = 4^n = {4 ** n}")
# ортогональность по Гильберту–Шмидту: базис пространства матриц
G = np.array([[np.trace(dag(ops[tuple(u)]) @ ops[tuple(v)]) / 2 ** n for v in vecs] for u in vecs])
assert np.allclose(G, np.eye(len(vecs)))
print("  Tr(P^dag Q)/2^n = delta_PQ: 4^n паули-строк — ортонормированный базис (I.1, §8)")

# ---------------------------------------------------------------------------
section("2. Таблицы сопряжения H, S, CNOT; T — не Клиффорд")
table = []
for name, U in [("H", H), ("S", S)]:
    for P in "XYZ":
        img = U @ PAULI[P] @ dag(U)
        for sgn in (1, -1):
            for Q in "XYZ":
                if np.allclose(img, sgn * PAULI[Q]):
                    table.append(f"{name} {P} {name}^dag = {'+' if sgn > 0 else '-'}{Q}")
print("  " + ";  ".join(table))
assert "H Y H^dag = -Y" in table and "S X S^dag = +Y" in table and "S Y S^dag = -X" in table
C = cnot(0, 1, 2)
cn_table = {}
for P in ["XI", "IX", "ZI", "IZ", "YI", "IY", "YY", "XZ"]:
    img = C @ pauli_string(P) @ C
    for sgn in (1, -1):
        for Q in ["".join(q) for q in product("IXYZ", repeat=2)]:
            if np.allclose(img, sgn * pauli_string(Q)):
                cn_table[P] = ("" if sgn > 0 else "-") + Q
print("  CNOT:", ", ".join(f"{k} -> {v}" for k, v in cn_table.items()))
assert cn_table["XI"] == "XX" and cn_table["IX"] == "IX" and cn_table["ZI"] == "ZI" and cn_table["IZ"] == "ZZ"
assert cn_table["YY"] == "-XZ"
TXT = T @ X @ dag(T)
assert np.allclose(TXT, (X + Y) / np.sqrt(2))
assert np.allclose(TXT, np.exp(-1j * np.pi / 4) * S @ X)
assert np.allclose(T @ Z @ dag(T), Z)
print("  T X T^dag = (X+Y)/sqrt2 — не паули: T не Клиффорд; но T X T^dag = e^{-i pi/4} S X — Клиффорд (иерархия: T в C_3)")

# ---------------------------------------------------------------------------
section("3. Порядок группы Клиффорда (поиск в ширину по модулю фазы)")
def clifford_bfs(n):
    gens = [op(H, k, n) for k in range(n)] + [op(S, k, n) for k in range(n)] + \
           [cnot(c, t, n) for c in range(n) for t in range(n) if c != t]
    start = np.eye(2 ** n, dtype=complex)
    seen = {canon(start): start}
    frontier = [start]
    while frontier:
        new = []
        for U in frontier:
            for g in gens:
                V = g @ U
                k = canon(V)
                if k not in seen:
                    seen[k] = V
                    new.append(V)
        frontier = new
    return list(seen.values())
def clifford_order(n):
    return 2 ** (n * n + 2 * n) * prod(4 ** j - 1 for j in range(1, n + 1))
def sp_order(n):
    return 2 ** (n * n) * prod(4 ** j - 1 for j in range(1, n + 1))
_PV = {}
def symplectic_image(U, n):
    """Двоичная 2n x 2n матрица действия U на (x|z) (столбцы — образы X_j, Z_j)."""
    if n not in _PV:
        vs = np.array(list(product([0, 1], repeat=2 * n)))
        _PV[n] = (vs, np.array([pauli_xz(v[:n], v[n:]) for v in vs]),
                  [pauli_xz(np.eye(2 * n, dtype=int)[k][:n], np.eye(2 * n, dtype=int)[k][n:]) for k in range(2 * n)])
    vs, Ps, E = _PV[n]
    cols = []
    for k in range(2 * n):
        img = U @ E[k] @ dag(U)
        ov = np.abs(np.einsum('kij,ij->k', Ps.conj(), img)) / 2 ** n
        j = int(np.argmax(ov))
        assert np.isclose(ov[j], 1), "не Клиффорд"
        cols.append(vs[j])
    return np.array(cols).T
t0 = time.time()
cliff = {}
for n in (1, 2):
    group = clifford_bfs(n)
    cliff[n] = group
    syms = {symplectic_image(U, n).tobytes() for U in group}
    print(f"  n = {n}: |C_n / U(1)| = {len(group)} (формула 2^(n^2+2n) prod(4^j-1) = {clifford_order(n)}); "
          f"различных симплектических матриц {len(syms)} = |Sp({2 * n},2)| = {sp_order(n)}")
    assert len(group) == clifford_order(n) and len(syms) == sp_order(n)
    assert len(group) == len(syms) * 4 ** n
print(f"  |C_n/U(1)| = |Sp(2n,2)| * 4^n: ядро отображения в Sp — сами паули; время {time.time() - t0:.1f} с")
for n in (3, 5, 10, 20):
    print(f"  n = {n:2d}: |C_n/U(1)| = {clifford_order(n):.3e},  log2 = {ilog2(clifford_order(n)):.1f};  2n^2 + 3n = {2 * n * n + 3 * n}")
    assert abs(ilog2(clifford_order(n)) - (2 * n * n + 3 * n)) < 1
# симплектичность: M^T Omega M = Omega
n = 2
Om = np.block([[np.zeros((n, n), int), np.eye(n, dtype=int)], [np.eye(n, dtype=int), np.zeros((n, n), int)]])
for U in cliff[2][::97]:
    M = symplectic_image(U, n)
    assert np.array_equal((M.T @ Om @ M) % 2, Om)
print("  M^T Omega M = Omega (mod 2) для образов клиффордов: сопряжение сохраняет коммутацию")

# ---------------------------------------------------------------------------
section("4. Стабилизаторные состояния: число, GHZ, проектор")
def stab_states(n):
    gens = [op(H, k, n) for k in range(n)] + [op(S, k, n) for k in range(n)] + \
           [cnot(c, t, n) for c in range(n) for t in range(n) if c != t]
    v0 = np.zeros(2 ** n, complex); v0[0] = 1
    seen = {canon(v0): v0}; frontier = [v0]
    while frontier:
        new = []
        for v in frontier:
            for g in gens:
                w = g @ v; k = canon(w)
                if k not in seen:
                    seen[k] = w; new.append(w)
        frontier = new
    return list(seen.values())
def n_stab(n):
    return 2 ** n * prod(2 ** k + 1 for k in range(1, n + 1))
STAB = {}
for n in (1, 2, 3):
    STAB[n] = stab_states(n)
    print(f"  n = {n}: стабилизаторных состояний {len(STAB[n])}, формула 2^n prod(2^k+1) = {n_stab(n)}")
    assert len(STAB[n]) == n_stab(n)
for n in (10, 50):
    print(f"  n = {n}: log2(#STAB) = {ilog2(n_stab(n)):.1f};  (n^2 + 3n)/2 = {(n * n + 3 * n) / 2:.0f}")
    assert abs(ilog2(n_stab(n)) - (n * n + 3 * n) / 2) < 2
ghz = np.zeros(8, complex); ghz[[0, 7]] = 1 / np.sqrt(2)
for g in ["XXX", "ZZI", "IZZ"]:
    assert np.allclose(pauli_string(g) @ ghz, ghz)
proj = (np.eye(8) + pauli_string("XXX")) @ (np.eye(8) + pauli_string("ZZI")) @ (np.eye(8) + pauli_string("IZZ")) / 8
assert np.allclose(proj, np.outer(ghz, ghz.conj()))
# через все 8 элементов группы
elems = [np.eye(8)]
for g in ["XXX", "ZZI", "IZZ"]:
    elems = elems + [e @ pauli_string(g) for e in elems]
assert np.allclose(sum(elems) / 8, proj)
print("  GHZ: XXX, ZZI, IZZ стабилизируют; prod (I+g_i)/2 = (1/8) sum_{g in S} g = |GHZ><GHZ|")
print("  ожидания паули на стабилизаторном состоянии: <P> in {0, +1, -1}")
for v in STAB[3][::37]:
    for s in product("IXYZ", repeat=3):
        e = np.vdot(v, pauli_string("".join(s)) @ v).real
        assert min(abs(e), abs(abs(e) - 1)) < 1e-9

# ---------------------------------------------------------------------------
section("5. Табличный симулятор Ааронсона–Готтесмана (CHP) и сверка с вектором состояния")
class Tableau:
    def __init__(self, n):
        self.n = n
        self.x = np.zeros((2 * n + 1, n), np.uint8)
        self.z = np.zeros((2 * n + 1, n), np.uint8)
        self.r = np.zeros(2 * n + 1, np.uint8)
        for i in range(n):
            self.x[i, i] = 1          # дестабилизаторы X_i
            self.z[n + i, i] = 1      # стабилизаторы Z_i
    def h(self, a):
        self.r ^= self.x[:, a] & self.z[:, a]
        self.x[:, a], self.z[:, a] = self.z[:, a].copy(), self.x[:, a].copy()
    def s(self, a):
        self.r ^= self.x[:, a] & self.z[:, a]
        self.z[:, a] ^= self.x[:, a]
    def cx(self, a, b):
        self.r ^= self.x[:, a] & self.z[:, b] & (self.x[:, b] ^ self.z[:, a] ^ 1)
        self.x[:, b] ^= self.x[:, a]
        self.z[:, a] ^= self.z[:, b]
    @staticmethod
    def g(x1, z1, x2, z2):
        x1, z1, x2, z2 = (np.asarray(v, int) for v in (x1, z1, x2, z2))
        return np.where((x1 == 1) & (z1 == 1), z2 - x2,
               np.where((x1 == 1) & (z1 == 0), z2 * (2 * x2 - 1),
               np.where((x1 == 0) & (z1 == 1), x2 * (1 - 2 * z2), 0)))
    def rowsum(self, hs, i):
        """Строки hs := строки hs * строка i (векторизовано по hs)."""
        hs = np.atleast_1d(hs)
        if hs.size == 0:
            return
        ph = 2 * self.r[hs].astype(int) + 2 * int(self.r[i]) + \
            self.g(self.x[i][None, :], self.z[i][None, :], self.x[hs], self.z[hs]).sum(axis=1)
        stab_rows = hs >= self.n
        assert np.all(ph[stab_rows] % 2 == 0)       # фаза строк-дестабилизаторов не нужна (может быть ±i)
        self.r[hs] = ((ph % 4) // 2).astype(np.uint8)
        self.x[hs] ^= self.x[i]
        self.z[hs] ^= self.z[i]
    def measure(self, a, forced=None):
        n = self.n
        ps = np.nonzero(self.x[n:2 * n, a])[0]
        if ps.size:                                 # случайный исход
            p = n + ps[0]
            rows = np.array([i for i in np.nonzero(self.x[:2 * n, a])[0] if i != p])
            self.rowsum(rows, p)
            self.x[p - n], self.z[p - n], self.r[p - n] = self.x[p], self.z[p], self.r[p]
            self.x[p] = 0; self.z[p] = 0; self.z[p, a] = 1
            self.r[p] = rng.integers(2) if forced is None else forced
            return int(self.r[p]), False
        sc = 2 * n                                  # детерминированный исход
        self.x[sc] = 0; self.z[sc] = 0; self.r[sc] = 0
        for i in np.nonzero(self.x[:n, a])[0]:
            self.rowsum(np.array([sc]), i + n)
        return int(self.r[sc]), True
    def stabilizers(self):
        n = self.n
        return [(self.x[n + i].copy(), self.z[n + i].copy(), int(self.r[n + i])) for i in range(n)]

def apply1(psi, U, a, n):
    psi = psi.reshape([2] * n)
    psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [a])), 0, a)
    return psi.reshape(-1)
def apply_cx(psi, a, b, n):
    psi = psi.reshape([2] * n).copy()
    idx = [slice(None)] * n; idx[a] = 1
    sub = psi[tuple(idx)]
    bb = b if b < a else b - 1
    psi[tuple(idx)] = np.flip(sub, axis=bb)
    return psi.reshape(-1)
def prob0(psi, a, n):
    p = np.abs(psi.reshape([2] * n)) ** 2
    return np.moveaxis(p, a, 0)[0].sum()
def project(psi, a, m, n):
    p = psi.reshape([2] * n).copy()
    idx = [slice(None)] * n; idx[a] = 1 - m
    p[tuple(idx)] = 0
    p = p.reshape(-1)
    return p / np.linalg.norm(p)

n_det = n_rand = 0
for trial in range(60):
    n = int(rng.integers(2, 7))
    tab = Tableau(n)
    psi = np.zeros(2 ** n, complex); psi[0] = 1
    for step in range(40):
        kind = rng.integers(4)
        a = int(rng.integers(n))
        if kind == 0:
            tab.h(a); psi = apply1(psi, H, a, n)
        elif kind == 1:
            tab.s(a); psi = apply1(psi, S, a, n)
        elif kind == 2:
            b = int(rng.choice([q for q in range(n) if q != a]))
            tab.cx(a, b); psi = apply_cx(psi, a, b, n)
        else:
            m, det = tab.measure(a)
            p0 = prob0(psi, a, n)
            if det:
                assert np.isclose(p0, 1 - m); n_det += 1
            else:
                assert np.isclose(p0, 0.5); n_rand += 1
            psi = project(psi, a, m, n)
    for x, z, r in tab.stabilizers():
        P = (-1) ** r * pauli_xz(x, z)
        assert np.allclose(P @ psi, psi)
print(f"  60 случайных схем (n = 2..6, 40 операций): {n_det} детерминированных и {n_rand} случайных измерений")
print("  совпадают с вектором состояния (p0 = 0/1 или 1/2), итоговые стабилизаторы ±P(x|z) стабилизируют psi")
# GHZ на n кубитах: генераторы X...X и Z_k Z_{k+1}
for n in range(3, 11):
    tab = Tableau(n)
    tab.h(0)
    for k in range(n - 1):
        tab.cx(k, k + 1)
    # приводим к виду X..X, Z_k Z_{k+1}: проверяем, что эти операторы лежат в группе (ранг над GF(2))
    M = np.array([np.concatenate([x, z]) for x, z, r in tab.stabilizers()], int)
    target = [np.concatenate([np.ones(n, int), np.zeros(n, int)])] + \
             [np.concatenate([np.zeros(n, int), np.eye(n, dtype=int)[k] + np.eye(n, dtype=int)[k + 1]]) for k in range(n - 1)]
    def rank2(A):
        A = A.copy() % 2; r = 0
        for c in range(A.shape[1]):
            piv = [i for i in range(r, A.shape[0]) if A[i, c]]
            if not piv: continue
            A[[r, piv[0]]] = A[[piv[0], r]]
            for i in range(A.shape[0]):
                if i != r and A[i, c]: A[i] ^= A[r]
            r += 1
        return r
    assert rank2(M) == n and rank2(np.vstack([M, np.array(target)])) == n
print("  GHZ на n = 3..10: группа стабилизатора = <X...X, Z_k Z_{k+1}> (тест ЛР5)")
# масштабирование: время одного измерения
print("  время: случайная клиффордова схема из 10 n гейтов, затем измерение всех n кубитов")
times = []
for n in (100, 200, 400, 800):
    tab = Tableau(n)
    for _ in range(10 * n):
        k = rng.integers(3); a = int(rng.integers(n))
        if k == 0: tab.h(a)
        elif k == 1: tab.s(a)
        else:
            b = int(rng.integers(n - 1)); b += b >= a; tab.cx(a, b)
    t0 = time.time()
    for a in range(n):
        tab.measure(a)
    dt = (time.time() - t0) / n
    times.append(dt)
    print(f"    n = {n:4d}: {1e3 * dt:7.3f} мс на измерение")
slope = np.polyfit(np.log([100, 200, 400, 800]), np.log(times), 1)[0]
print(f"  наклон log(время)/log(n) = {slope:.2f} (теория: 2 — O(n^2) на измерение)")
assert 1.3 < slope < 2.7

# ---------------------------------------------------------------------------
section("6. Запутанность стабилизаторных состояний: S_A = rank(M_A) - |A| (целое)")
def ent_entropy(psi, A, n):
    B = [q for q in range(n) if q not in A]
    M = psi.reshape([2] * n).transpose(list(A) + B).reshape(2 ** len(A), -1)
    s = np.linalg.svd(M, compute_uv=False) ** 2
    s = s[s > 1e-12]
    return -np.sum(s * np.log2(s)), s
def rank2(A):
    A = A.copy() % 2; r = 0
    for c in range(A.shape[1]):
        piv = [i for i in range(r, A.shape[0]) if A[i, c]]
        if not piv: continue
        A[[r, piv[0]]] = A[[piv[0], r]]
        for i in range(A.shape[0]):
            if i != r and A[i, c]: A[i] ^= A[r]
        r += 1
    return r
hist = {}
for trial in range(40):
    n = 6
    tab = Tableau(n)
    psi = np.zeros(2 ** n, complex); psi[0] = 1
    for _ in range(60):
        k = rng.integers(3); a = int(rng.integers(n))
        if k == 0: tab.h(a); psi = apply1(psi, H, a, n)
        elif k == 1: tab.s(a); psi = apply1(psi, S, a, n)
        else:
            b = int(rng.integers(n - 1)); b += b >= a
            tab.cx(a, b); psi = apply_cx(psi, a, b, n)
    stabs = tab.stabilizers()
    for size in (1, 2, 3):
        for A in combinations(range(n), size):
            MA = np.array([np.concatenate([x[list(A)], z[list(A)]]) for x, z, r in stabs], int)
            SA = rank2(MA) - len(A)
            Snum, spec = ent_entropy(psi, A, n)
            assert np.isclose(Snum, SA)
            assert np.allclose(spec, 2.0 ** (-SA))          # плоский спектр Шмидта
            hist[SA] = hist.get(SA, 0) + 1
print(f"  40 случайных стабилизаторных состояний n = 6, все разрезы |A| <= 3: формула совпадает с SVD")
print(f"  спектр Шмидта плоский (все lambda = 2^-S); распределение S_A: {dict(sorted(hist.items()))}")
print("  клиффордовы схемы могут давать максимальную запутанность (S_A = |A| при |A| <= n/2), но моделируются за poly(n)")

# ---------------------------------------------------------------------------
section("7. Магия: |T>, стабилизаторная верность, стабилизаторный ранг, гаджет T")
Tst = T @ np.array([1, 1]) / np.sqrt(2)
for t in (1, 2, 3):
    psi = kron(*[Tst.reshape(2, 1)] * t).reshape(-1)
    F = max(abs(np.vdot(s, psi)) ** 2 for s in STAB[t])
    print(f"  t = {t}: F_stab(|T>^t) = {F:.6f};  cos^2(pi/8)^t = {np.cos(np.pi / 8) ** (2 * t):.6f}")
    assert np.isclose(F, np.cos(np.pi / 8) ** (2 * t))
    assert F < 1 - 1e-6                                   # не стабилизаторное
# chi(|T>^{(x)2}) = 2 — явное разложение
TT = np.kron(Tst, Tst)
phi1 = np.array([1, 0, 0, 1j]) / np.sqrt(2)               # (|00> + i|11>)/sqrt2
phi2 = np.array([0, 1, 1, 0]) / np.sqrt(2)                # Psi^+
w = np.exp(1j * np.pi / 4)
assert np.allclose(TT, (phi1 + w * phi2) / np.sqrt(2))
keys = {canon(s) for s in STAB[2]}
assert canon(phi1) in keys and canon(phi2) in keys and canon(TT) not in keys
print("  |T>|T> = [(|00> + i|11>) + e^{i pi/4}(|01> + |10>)]/2: два стабилизаторных слагаемых, chi = 2")
# T = e^{i pi/8} (cos(pi/8) I - i sin(pi/8) Z) — сумма двух клиффордов
assert np.allclose(T, np.exp(1j * np.pi / 8) * (np.cos(np.pi / 8) * I2 - 1j * np.sin(np.pi / 8) * Z))
print("  T = e^{i pi/8}[cos(pi/8) I - i sin(pi/8) Z]: схема с t гейтами T — сумма 2^t клиффордовых схем")
for trial in range(5):
    n, tcount = 4, 6
    gates = []
    for _ in range(30):
        k = rng.integers(3); a = int(rng.integers(n))
        if k == 0: gates.append(("H", a))
        elif k == 1: gates.append(("S", a))
        else:
            b = int(rng.integers(n - 1)); b += b >= a; gates.append(("CX", a, b))
    for pos in sorted(rng.choice(len(gates), tcount, replace=False), reverse=True):
        gates.insert(int(pos), ("T", int(rng.integers(n))))
    def run(gates, branch=None):
        psi = np.zeros(2 ** n, complex); psi[0] = 1
        j = 0
        for g in gates:
            if g[0] == "H": psi = apply1(psi, H, g[1], n)
            elif g[0] == "S": psi = apply1(psi, S, g[1], n)
            elif g[0] == "CX": psi = apply_cx(psi, g[1], g[2], n)
            else:
                if branch is None:
                    psi = apply1(psi, T, g[1], n)
                else:
                    c = np.exp(1j * np.pi / 8) * (np.cos(np.pi / 8) if branch[j] == 0 else -1j * np.sin(np.pi / 8))
                    psi = c * (psi if branch[j] == 0 else apply1(psi, Z, g[1], n))
                    j += 1
        return psi
    full = run(gates)
    total = sum(run(gates, br) for br in product([0, 1], repeat=tcount))
    assert np.allclose(full, total)
print(f"  5 случайных схем Clifford+T (n = 4, t = 6): сумма 2^6 = 64 клиффордовых ветвей = точный вектор")
for t in (10, 20, 50):
    print(f"  t = {t:2d}: ветвей 2^t = {2.0 ** t:.1e};  по оценке 7^(t/6) (Бравый–Смит–Смолин): {7 ** (t / 6):.1e};"
          f"  по 2^(0.228 t): {2 ** (0.228 * t):.1e}")
xi = 1 / np.cos(np.pi / 8) ** 2
print(f"  1/F_stab(|T>) = 1/cos^2(pi/8) = {xi:.4f}, log2 = {np.log2(xi):.4f}; log2(7)/6 = {np.log2(7) / 6:.4f}")
assert np.isclose(np.log2(xi), 0.2284, atol=1e-4) and np.isclose(np.log2(7) / 6, 0.4679, atol=1e-4)
# гаджет: |psi> (x) |T>, CNOT(psi -> anc), измерение anc, при исходе 1 — S X на anc... проверяем стандартную форму
for _ in range(50):
    psi = rng.normal(size=2) + 1j * rng.normal(size=2); psi /= np.linalg.norm(psi)
    st = cnot(1, 0, 2) @ np.kron(psi, Tst)      # управляет кубит с |T>, цель — данные
    for m in (0, 1):
        v = st.reshape(2, 2)[m]                  # измерили кубит данных (первый) с исходом m
        pm = np.vdot(v, v).real
        v = v / np.sqrt(pm)
        if m == 1:
            v = S @ X @ v
        assert np.isclose(pm, 0.5) and np.isclose(abs(np.vdot(T @ psi, v)), 1)
print("  гаджет: CNOT(|T> -> psi), измерение psi в Z, при исходе 1 — SX: на месте |T> получается T|psi>")

# ---------------------------------------------------------------------------
section("8. Малая запутанность => эффективная симуляция: ранг Шмидта в 1D-схемах")
n = 12
cut = n // 2
psi = np.zeros(2 ** n, complex); psi[0] = 1
crossing = 0
print("  кирпичная схема: случайные однокубитные U + CZ на соседях; разрез посередине")
def haar2():
    A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    Q, R = np.linalg.qr(A)
    return Q * (np.diag(R) / abs(np.diag(R)))
def apply_cz(psi, a, b, n):
    p = psi.reshape([2] * n).copy()
    idx = [slice(None)] * n; idx[a] = 1; idx[b] = 1
    p[tuple(idx)] *= -1
    return p.reshape(-1)
for layer in range(8):
    for q in range(n):
        psi = apply1(psi, haar2(), q, n)
    for q in range(layer % 2, n - 1, 2):
        psi = apply_cz(psi, q, q + 1, n)
        if q == cut - 1:
            crossing += 1
    M = psi.reshape(2 ** cut, -1)
    sv = np.linalg.svd(M, compute_uv=False)
    rank = int(np.sum(sv > 1e-10))
    assert rank <= 2 ** crossing
    print(f"    слой {layer + 1}: CZ через разрез {crossing}, ранг Шмидта {rank:3d} <= 2^{crossing} = {2 ** crossing}")
print("  каждый CZ через разрез умножает ранг Шмидта не более чем на 2 (операторный ранг Шмидта CZ = 2);")
print("  MPS с bond dimension chi хранит состояние за O(n chi^2) чисел")

# ---------------------------------------------------------------------------
section("9. Ответы к задачам семинара")
# задача 2: стабилизаторы состояний Белла beta_xy: (-1)^x XX, (-1)^y ZZ
for xb, yb in product([0, 1], repeat=2):
    v = cnot(0, 1, 2) @ np.kron(H, I2) @ np.eye(4)[2 * xb + yb]
    assert np.allclose(pauli_string("XX") @ v, (-1) ** xb * v)
    assert np.allclose(pauli_string("ZZ") @ v, (-1) ** yb * v)
print("  2: beta_xy стабилизируется (-1)^x XX и (-1)^y ZZ")
# задача 3: сколько из 60 двухкубитных стабилизаторных состояний — произведения
prod_cnt = sum(1 for v in STAB[2] if np.linalg.matrix_rank(v.reshape(2, 2), tol=1e-9) == 1)
print(f"  3: из 60 стабилизаторных состояний n = 2 произведений {prod_cnt}, запутанных {60 - prod_cnt}")
assert prod_cnt == 36
# задача 4: |C_1| = 6 * 4 = 24 по образам X и Z
signed = [sg * PAULI[q] for q in "XYZ" for sg in (1, -1)]
pairs = sum(1 for A in signed for B in signed if np.allclose(A @ B, -B @ A))
print(f"  4: пар (образ X, образ Z) с антикоммутацией: {pairs}")
assert pairs == 24
# задача 5: H(0), CNOT(0,1), измерить 1, затем 0 — исходы совпадают
outs = set()
for _ in range(200):
    tb = Tableau(2); tb.h(0); tb.cx(0, 1)
    m1, d1 = tb.measure(1); m0, d0 = tb.measure(0)
    assert (not d1) and d0 and m0 == m1
    outs.add(m1)
assert outs == {0, 1}
print("  5: первое измерение случайно, второе детерминировано и совпадает с первым")
# задача 6: GHZ_n — S_A = 1 для любого непустого собственного A
for n in (4, 5):
    tb = Tableau(n); tb.h(0)
    for k in range(n - 1): tb.cx(k, k + 1)
    st = tb.stabilizers()
    for size in range(1, n):
        for A in combinations(range(n), size):
            MA = np.array([np.concatenate([x[list(A)], z[list(A)]]) for x, z, r in st], int)
            assert rank2(MA) - len(A) == 1
print("  6: GHZ_4, GHZ_5: S_A = 1 для всех разрезов")
# задача 7: линейное кластерное состояние на 4 кубитах
n = 4
cl = np.ones(16, complex) / 4
for a in range(n - 1):
    cl = apply_cz(cl, a, a + 1, n)
gens = ["XZII", "ZXZI", "IZXZ", "IIZX"]
for g in gens:
    assert np.allclose(pauli_string(g) @ cl, cl)
S12, _ = ent_entropy(cl, (0, 1), n)
S13, _ = ent_entropy(cl, (0, 2), n)
S1, _ = ent_entropy(cl, (0,), n)
print(f"  7: кластер 4 кубита: генераторы XZII, ZXZI, IZXZ, IIZX; S(12|34) = {S12:.3f}, S(13|24) = {S13:.3f}, S(1|234) = {S1:.3f}")
assert np.isclose(S12, 1) and np.isclose(S13, 2) and np.isclose(S1, 1)
# задача 10: генераторы слагаемых |T>|T>
assert np.allclose(pauli_string("ZZ") @ phi1, phi1) and np.allclose(pauli_string("XY") @ phi1, phi1)
assert np.allclose(pauli_string("XX") @ phi2, phi2) and np.allclose(-pauli_string("ZZ") @ phi2, phi2)
print("  10: (|00> + i|11>)/sqrt2 — <ZZ, XY>;  Psi^+ — <XX, -ZZ>")
# задача 8: при исходе 1 гаджета состояние = e^{i pi/4} X T^dag |psi>, SX X T^dag = T
assert np.allclose(S @ X @ X @ dag(T), T)
# задача 9: перекрытия |T> со стабилизаторными состояниями кубита
ov = sorted(abs(np.vdot(s_, Tst)) ** 2 for s_ in STAB[1])
print("  9: |<phi|T>|^2 по 6 состояниям:", np.round(ov, 4))
assert np.isclose(ov[-1], (1 + 1 / np.sqrt(2)) / 2)
# задача 11: память
n = 50
print(f"  11: n = 50: вектор состояния 2^50 * 16 байт = {2 ** 50 * 16 / 1e15:.1f} ПБ; таблица 2n(2n+1) бит = {2 * n * (2 * n + 1)} бит")
# задача 13: нижняя оценка числа гейтов для типичного клиффорда
for n in (10, 100, 1000):
    choices = 2 * n + n * (n - 1)
    g = ilog2(clifford_order(n)) / np.log2(choices)
    print(f"  13: n = {n:4d}: g >= log2|C_n| / log2(2n + n(n-1)) = {g:.0f}  (2n^2 + 3n)/(2 log2 n) = {(2 * n * n + 3 * n) / (2 * np.log2(n)):.0f}")

print("\nВсе проверки пройдены.")
