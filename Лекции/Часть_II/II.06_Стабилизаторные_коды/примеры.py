"""
Лекция II.6. Стабилизаторные коды: параметры [[n,k,d]], нормализатор и логические
операторы, дистанция; условия Книлла–Лафламма в стабилизаторной форме; измерение
синдрома и декодирование по таблице; декодирование по классам (ML) для вырожденных
кодов; CSS-коды, код Стина [[7,1,3]] и его трансверсальные гейты; код [[4,2,2]];
границы Синглтона, Хэмминга, Гильберта–Варшамова (CSS); подсистемный код
Бэкона–Шора; схемный шум при извлечении синдрома с одним вспомогательным кубитом
(«крючковые» ошибки).

Соглашения (Лекции/Обозначения.md, §7, §18): симплектическая запись (x|z);
деполяризующий шум с вероятностью p — X, Y, Z по p/3; двухкубитный — 15 паули по p/15.

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
Порядок множителей: первый — старший (np.kron); кубиты в коде нумеруются с 0.
"""
import time
import itertools
import numpy as np
from math import comb, log2
from functools import reduce
from scipy.optimize import brentq

T_START = time.time()
rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.diag([1, 1j])
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def dag(A):
    return A.conj().T


def kron(*ops):
    return reduce(np.kron, ops)


def pauli(s):
    return kron(*[PAULI[c] for c in s])


def sym(s):
    x = np.array([c in "XY" for c in s], dtype=np.int64)
    z = np.array([c in "ZY" for c in s], dtype=np.int64)
    return x, z


def to_str(x, z):
    return "".join("IXZY"[int(a) + 2 * int(b)] for a, b in zip(x, z))


def sym_inner(x1, z1, x2, z2):
    return int((np.dot(x1, z2) + np.dot(z1, x2)) % 2)


def gf2_rank(M):
    M = (np.array(M, dtype=np.int64) % 2).copy()
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = [i for i in range(r, rows) if M[i, c]]
        if not piv:
            continue
        M[[r, piv[0]]] = M[[piv[0], r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def weight(s):
    return sum(c != "I" for c in s)


class StabCode:
    """Стабилизаторный код: генераторы (строки), логические X_j, Z_j (списки строк)."""

    def __init__(self, name, gens, XL, ZL):
        self.name, self.gens, self.XL, self.ZL = name, gens, XL, ZL
        self.n = len(gens[0])
        self.r = len(gens)
        self.k = len(XL)
        G = np.array([np.concatenate(sym(g)) for g in gens])
        self.G = G
        assert gf2_rank(G) == self.r, "генераторы зависимы"
        assert self.k == self.n - self.r
        for a, b in itertools.combinations(gens, 2):
            assert sym_inner(*sym(a), *sym(b)) == 0
        for L in XL + ZL:
            for g in gens:
                assert sym_inner(*sym(L), *sym(g)) == 0
        for i in range(self.k):
            for j in range(self.k):
                assert sym_inner(*sym(XL[i]), *sym(ZL[j])) == (i == j)
                assert sym_inner(*sym(XL[i]), *sym(XL[j])) == 0
                assert sym_inner(*sym(ZL[i]), *sym(ZL[j])) == 0
        self.Sx = np.array([sym(g)[0] for g in gens]).T
        self.Sz = np.array([sym(g)[1] for g in gens]).T
        self._enum = None

    def in_stabilizer(self, s):
        v = np.concatenate(sym(s))
        return gf2_rank(np.vstack([self.G, v])) == self.r

    def syndrome(self, s):
        x, z = sym(s)
        return tuple(int(v) for v in (x @ self.Sz + z @ self.Sx) % 2)

    def distance(self, wmax=None):
        """Наименьший вес паули из N(S) \\ S (перебор по весу)."""
        n = self.n
        for w in range(1, (wmax or n) + 1):
            for qs in itertools.combinations(range(n), w):
                for ps in itertools.product("XYZ", repeat=w):
                    s = ["I"] * n
                    for q, P in zip(qs, ps):
                        s[q] = P
                    s = "".join(s)
                    if not any(self.syndrome(s)) and not self.in_stabilizer(s):
                        return w, s
        return None

    def enumerate(self):
        """Все 4^n паули: x, z, вес, синдром (целое), логический класс (целое из 2k битов)."""
        if self._enum is not None:
            return self._enum
        n = self.n
        idx = np.arange(4 ** n)
        digits = (idx[:, None] // 4 ** np.arange(n - 1, -1, -1)) % 4  # 0=I,1=X,2=Y,3=Z
        x = ((digits == 1) | (digits == 2)).astype(np.int64)
        z = ((digits == 2) | (digits == 3)).astype(np.int64)
        sb = (x @ self.Sz + z @ self.Sx) % 2
        syn = sb @ (2 ** np.arange(self.r - 1, -1, -1))
        cls = np.zeros(len(idx), dtype=np.int64)
        for L in self.XL + self.ZL:  # класс: коммутация с логическими операторами
            lx, lz = sym(L)
            cls = 2 * cls + (x @ lz + z @ lx) % 2
        w = (x | z).sum(1)
        self._enum = (x, z, w, syn, cls)
        return self._enum

    def table(self):
        """Таблица: синдром -> индекс паули минимального веса (первой по порядку)."""
        x, z, w, syn, cls = self.enumerate()
        order = np.lexsort((np.arange(len(w)), w))
        first = {}
        for i in order:
            s = int(syn[i])
            if s not in first:
                first[s] = i
        return first

    def pL_decoders(self, p):
        """Точная логическая ошибка при деполяризующем шуме: таблица (мин. вес) и ML по классам."""
        x, z, w, syn, cls = self.enumerate()
        prob = (p / 3) ** w * (1 - p) ** (self.n - w)
        ns, nc = 2 ** self.r, 4 ** self.k
        P = np.zeros((ns, nc))
        np.add.at(P, (syn, cls), prob)
        tab = self.table()
        ok_table = sum(P[s, cls[tab[s]]] for s in range(ns) if s in tab)
        ok_ml = P.max(axis=1).sum()
        return 1 - ok_table, 1 - ok_ml

    def fail_counts(self):
        x, z, w, syn, cls = self.enumerate()
        tab = self.table()
        corr_cls = np.array([cls[tab[int(s)]] for s in syn])
        fail = corr_cls != cls
        return (np.array([np.sum(fail & (w == k)) for k in range(self.n + 1)]),
                np.array([np.sum(w == k) for k in range(self.n + 1)]))

    def projector(self):
        P = np.eye(2 ** self.n, dtype=complex)
        for g in self.gens:
            P = P @ (np.eye(2 ** self.n) + pauli(g)) / 2
        return P


# ===========================================================================
section("1. Стабилизаторные коды: параметры, логические операторы, дистанция")
SHOR = StabCode("Шора [[9,1,3]]",
                ["ZZIIIIIII", "IZZIIIIII", "IIIZZIIII", "IIIIZZIII", "IIIIIIZZI", "IIIIIIIZZ",
                 "XXXXXXIII", "IIIXXXXXX"], XL=["ZIIZIIZII"], ZL=["XXXIIIIII"])
FIVE = StabCode("[[5,1,3]]", ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"], XL=["XXXXX"], ZL=["ZZZZZ"])
C422 = StabCode("[[4,2,2]]", ["XXXX", "ZZZZ"], XL=["XXII", "XIXI"], ZL=["ZIZI", "ZZII"])
hamming = ["0001111", "0110011", "1010101"]
steane_gens = ["".join("X" if b == "1" else "I" for b in h) for h in hamming] + \
              ["".join("Z" if b == "1" else "I" for b in h) for h in hamming]
STEANE = StabCode("Стина [[7,1,3]]", steane_gens, XL=["XXXXXXX"], ZL=["ZZZZZZZ"])
print("  генераторы кода Стина:", steane_gens)
DIST = {}
for code in (C422, FIVE, STEANE, SHOR):
    d, wit = code.distance()
    DIST[code.name] = d
    P = code.projector()
    assert np.isclose(np.trace(P).real, 2 ** code.k)
    print(f"   {code.name:16s} n = {code.n}, r = {code.r} генераторов, k = {code.k}, dim C = {np.trace(P).real:.0f}, "
          f"d = {d} (пример логического оператора минимального веса: {wit})")
assert DIST["[[4,2,2]]"] == 2 and DIST["[[5,1,3]]"] == 3 and DIST["Стина [[7,1,3]]"] == 3 and DIST["Шора [[9,1,3]]"] == 3

# нормализатор = централизатор: паули либо коммутируют, либо антикоммутируют
A_, B_ = pauli("XZY"), pauli("ZZX")
assert np.allclose(A_ @ B_, B_ @ A_) or np.allclose(A_ @ B_, -B_ @ A_)
# число элементов N(S) с точностью до фазы: 4^n / 2^r = 2^{n+k}; логических классов 4^k
x, z, w, syn, cls = STEANE.enumerate()
nN = int(np.sum(syn == 0))
assert nN == 2 ** (STEANE.n + STEANE.k)
assert len(set(cls[syn == 0])) == 4 ** STEANE.k
print(f"  Стин: паули с нулевым синдромом (N(S) без фаз) — {nN} = 2^(n+k); в каждом из 4 логических классов по {nN // 4} = |S|")
# представители минимального веса логических X и Z кода Стина
minXL = [s for s in (to_str(x[i], z[i]) for i in np.where((syn == 0) & (w == 3))[0]) if set(s) <= {"I", "X"}]
assert all(not STEANE.in_stabilizer(s) for s in minXL) and len(minXL) == 7
print(f"  Стин: логических X веса 3 — {len(minXL)} (носители — слова веса 3 кода Хэмминга), например {minXL[0]}")

# стабилизаторная форма условий КЛ: E_a^+ E_b не из N(S)\S для всех пар веса <= 1
singles7 = ["I" * 7] + ["I" * q + P_ + "I" * (6 - q) for q in range(7) for P_ in "XYZ"]
Pst = STEANE.projector()
for a, b in itertools.combinations(singles7, 2):
    prod_x = (sym(a)[0] + sym(b)[0]) % 2
    prod_z = (sym(a)[1] + sym(b)[1]) % 2
    s = to_str(prod_x, prod_z)
    in_N = not any(STEANE.syndrome(s))
    bad = in_N and not STEANE.in_stabilizer(s)
    assert not bad
    M = Pst @ pauli(s) @ Pst
    if in_N:
        assert np.allclose(M, Pst) or np.allclose(M, -Pst)
    else:
        assert np.allclose(M, 0)
print("  Стин: для всех пар ошибок веса <= 1 произведение не лежит в N(S)\\S; P E P = 0 вне N(S), = ±P в S (проверено)")

# [[4,2,2]] обнаруживает любую однокубитную ошибку, но не исправляет
syn422 = {C422.syndrome("I" * q + P_ + "I" * (3 - q)) for q in range(4) for P_ in "XYZ"}
assert (0, 0) not in syn422 and len(syn422) == 3
print("  [[4,2,2]]: каждая однокубитная ошибка даёт ненулевой синдром (обнаружение), но синдромов всего 3 — исправления нет")

# ===========================================================================
section("2. Измерение синдрома и декодирование")


def measure_pauli_circuit(g, psi):
    """Схема: анцилла |0>, H, controlled-g, H, измерение. Возвращает вероятности исходов 0/1."""
    n = len(g)
    P0 = np.diag([1, 0]).astype(complex)
    P1 = np.diag([0, 1]).astype(complex)
    CG = np.kron(P0, np.eye(2 ** n)) + np.kron(P1, pauli(g))
    Hn = np.kron(H, np.eye(2 ** n))
    out = Hn @ CG @ Hn @ np.kron(np.array([1, 0]), psi)
    out = out.reshape(2, -1)
    return np.sum(np.abs(out[0]) ** 2), np.sum(np.abs(out[1]) ** 2)


Pfive = FIVE.projector()
w5, v5 = np.linalg.eigh(Pfive)
code_vec = v5[:, -1]
for e in ("IIYII", "XIIII", "IIIIZ"):
    bad_vec = pauli(e) @ code_vec
    syn_c = tuple(int(round(measure_pauli_circuit(g, bad_vec)[1])) for g in FIVE.gens)
    assert syn_c == FIVE.syndrome(e)
print("  схема H-controlled-g-H на вспомогательном кубите даёт синдром = симплектическое произведение (проверено для [[5,1,3]])")
tab7 = STEANE.table()
print(f"  таблица синдромов: Стин — {2 ** STEANE.r} синдромов, Шор — {2 ** SHOR.r}, [[5,1,3]] — {2 ** FIVE.r}; "
      f"в общем случае 2^(n-k) строк")
n_distinct = len({STEANE.syndrome(s) for s in singles7})
assert n_distinct == 22
print(f"  Стин: 22 паули веса <= 1 дают {n_distinct} различных синдромов (невырожденный на весе 1)")

print("  деполяризующий шум (code capacity): таблица минимального веса против ML-декодера по классам")
print("     код                p      таблица        ML по классам")
DEC = {}
for code in (SHOR, STEANE, FIVE):
    for p in (0.01, 0.05, 0.1):
        pt, pm = code.pL_decoders(p)
        DEC[(code.name, p)] = (pt, pm)
        assert pm <= pt + 1e-12
        print(f"   {code.name:16s} {p:5.2f}   {pt:.6f}       {pm:.6f}")
assert all(abs(pt - pm) < 1e-12 for pt, pm in DEC.values())
for p in (0.15, 0.2, 0.25):  # и при больших p
    pt, pm = SHOR.pL_decoders(p)
    assert abs(pt - pm) < 1e-12
print("  для этих малых кодов таблица минимального веса совпадает с ML по классам (разница < 1e-12 до p = 0.25)")

# ===========================================================================
section("3. CSS-коды и код Стина")
Hm = np.array([[int(b) for b in h] for h in hamming])
# классический код Хэмминга [7,4,3]: ядро Hm
words = np.array(list(itertools.product((0, 1), repeat=7)))
C1 = words[np.all((words @ Hm.T) % 2 == 0, axis=1)]
assert len(C1) == 16
assert min(w_.sum() for w_ in C1 if w_.any()) == 3
C2 = np.array([w_ for w_ in words if any(np.array_equal(w_, (np.array(c) @ Hm) % 2) for c in itertools.product((0, 1), repeat=3))])
assert len(C2) == 8 and all(w_.sum() in (0, 4) for w_ in C2)
assert all(any(np.array_equal(c2, c1) for c1 in C1) for c2 in C2)  # C2 = C1^perp ⊂ C1
print("  C1 = Хэмминг [7,4,3] (16 слов, d = 3); C2 = C1^perp = симплекс [7,3,4] (8 слов, веса 0 и 4) ⊂ C1")
assert np.all((Hm @ Hm.T) % 2 == 0)
print("  условие коммутации H_X H_Z^T = H H^T = 0 (mod 2) выполнено")


def css_state(coset_shift):
    v = np.zeros(128, dtype=complex)
    for c in C2:
        v[int("".join(map(str, (c + coset_shift) % 2)), 2)] += 1
    return v / np.linalg.norm(v)


st0 = css_state(np.zeros(7, dtype=int))
st1 = css_state(np.ones(7, dtype=int))
w7 = np.linalg.eigh(STEANE.projector())[1][:, -2:]
for v in (st0, st1):
    assert np.isclose(np.linalg.norm(dag(w7) @ v), 1)
assert np.allclose(pauli("XXXXXXX") @ st0, st1) and np.allclose(pauli("ZZZZZZZ") @ st1, -st1)
w1 = sorted({int(c.sum()) for c in (C2 + 1) % 2})
print(f"  |0_L> = сумма 8 слов C2 (веса 0, 4); |1_L> = сумма слов смежного класса 1111111 + C2 (веса {w1})")
assert w1 == [3, 7]

V7 = np.stack([st0, st1], axis=1)


def logical_action(U):
    """Действие U на коде: V^+ U V (должно быть унитарным 2x2, если U сохраняет код)."""
    M = dag(V7) @ U @ V7
    assert np.allclose(M @ dag(M), np.eye(2), atol=1e-10), "не сохраняет кодовое пространство"
    return M


def eq_phase(A, B):
    k = np.argmax(np.abs(B))
    ph = A.flat[k] / B.flat[k]
    return np.isclose(abs(ph), 1) and np.allclose(A, ph * B)


TR = {}
for name, U1, target in (("H", H, H), ("S", S, dag(S)), ("S^+", dag(S), S), ("X", X, X), ("Z", Z, Z)):
    M = logical_action(kron(*[U1] * 7))
    assert eq_phase(M, target)
    TR[name] = M
print("  трансверсальные гейты кода Стина: H^7 = H_L, S^7 = (S_L)^+, X^7 = X_L, Z^7 = Z_L (проверено)")
T = np.diag([1, np.exp(1j * np.pi / 4)])
Mt = dag(V7) @ kron(*[T] * 7) @ V7
leak = 1 - np.linalg.norm(Mt @ np.array([1, 1]) / np.sqrt(2)) ** 2
assert leak > 1e-3
print(f"  T^7 не сохраняет код: из (|0_L>+|1_L>)/sqrt2 выходит из кода доля {leak:.4f} (трансверсального T нет — II.8)")

# трансверсальный CNOT между двумя блоками (14 кубитов): перестановка индексов
idx = np.arange(2 ** 14)
a_bits = idx >> 7
b_bits = idx & 127
cnot_t = np.empty_like(idx)
cnot_t[idx] = (a_bits << 7) | (b_bits ^ a_bits)
for a in (0, 1):
    for b in (0, 1):
        va, vb = (st0, st1)[a], (st0, st1)[b]
        out = np.zeros(2 ** 14, dtype=complex)
        out[cnot_t] = np.kron(va, vb)
        expect = np.kron(va, (st0, st1)[a ^ b])
        assert np.allclose(out, expect)
print("  трансверсальный CNOT (7 CNOT между блоками) = CNOT_L на |a_L>|b_L> (проверено, 14 кубитов)")

nf7, tot7 = STEANE.fail_counts()
print(f"  Стин, деполяризующий шум: N_fail = {list(nf7)} (из {list(tot7)})")
assert nf7[1] == 0
A7 = nf7[2] / 9
pth7 = brentq(lambda p: STEANE.pL_decoders(p)[0] - p, 1e-3, 0.3)
print(f"  p_L ~ {A7:.4f} p^2; псевдопорог (таблица) {pth7:.4f}; p_L(0.001) = {STEANE.pL_decoders(1e-3)[0]:.3e}")
# независимое CSS-декодирование X и Z частей кодом Хэмминга
x, z, w, syn, cls = STEANE.enumerate()


def hamming_correct(v):
    s = (v @ Hm.T) % 2
    pos = s @ np.array([4, 2, 1])  # столбец j матрицы Хэмминга — двоичная запись j+1
    out = v.copy()
    rows = np.where(pos > 0)[0]
    out[rows, pos[rows] - 1] ^= 1
    return out


assert all(int("".join(map(str, Hm[:, j])), 2) == j + 1 for j in range(7))
rx, rz = hamming_correct(x), hamming_correct(z)
fail_css = ((rx @ np.ones(7, int)) % 2 == 1) | ((rz @ np.ones(7, int)) % 2 == 1)
# остаток — элемент C1 (нулевой синдром); логическая ошибка, если вес нечётен (слово из C1 \ C2)
nf_css = np.array([np.sum(fail_css & (w == k)) for k in range(8)])
assert np.array_equal(nf_css, nf7)
print("  независимое декодирование X- и Z-частей кодом Хэмминга даёт те же N_fail, что и общая таблица")

# ===========================================================================
section("4. Границы")


def h2(t):
    return 0.0 if t <= 0 or t >= 1 else -t * log2(t) - (1 - t) * log2(1 - t)


print("   код         n-k   2(d-1)   Синглтон      Хэмминг (t = 1)")
BOUNDS = {}
for name, n, k, d in (("[[4,2,2]]", 4, 2, 2), ("[[5,1,3]]", 5, 1, 3), ("[[7,1,3]]", 7, 1, 3), ("[[9,1,3]]", 9, 1, 3)):
    sing = n - k >= 2 * (d - 1)
    ham = 2 ** k * (1 + 3 * n) <= 2 ** n if d == 3 else None
    BOUNDS[name] = (n - k, 2 * (d - 1))
    print(f"   {name:10s}  {n - k:3d}   {2 * (d - 1):3d}     {'равенство' if n - k == 2 * (d - 1) else 'выполнена'}"
          f"     {'' if ham is None else ('равенство' if 2 ** k * (1 + 3 * n) == 2 ** n else 'выполнена')}")
    assert sing
print("  асимптотика (k/n как функция delta = d/n или t/n):")
print("   delta   Синглтон 1-2delta   Хэмминг 1-(d/2n)log3-h(d/2n)   CSS-ГВ 1-2h(delta)   ГВ 1-delta*log3-h(delta)")
GV = {}
for dl in (0.02, 0.05, 0.1, 0.15):
    t = dl / 2
    vals = (1 - 2 * dl, 1 - t * log2(3) - h2(t), 1 - 2 * h2(dl), 1 - dl * log2(3) - h2(dl))
    GV[dl] = vals
    print(f"   {dl:5.2f}   {vals[0]:8.4f}            {vals[1]:8.4f}                      {vals[2]:8.4f}            {vals[3]:8.4f}")
    assert vals[2] <= vals[1] <= vals[0] + 1e-12
d_css0 = brentq(lambda t: 1 - 2 * h2(t), 1e-6, 0.3)
d_gv0 = brentq(lambda t: 1 - t * log2(3) - h2(t), 1e-6, 0.3)
print(f"  CSS-ГВ даёт положительную скорость при d/n < {d_css0:.4f}, стабилизаторная ГВ — при d/n < {d_gv0:.4f}")
print(f"  задача 8: CSS (Калдербанк–Шор), t/n = 0.01: k/n >= 1 - 2h(0.02) = {1 - 2 * h2(0.02):.4f}; "
      f"t/n = 0.05: {1 - 2 * h2(0.1):.4f}")
CSS_RATES = (1 - 2 * h2(0.02), 1 - 2 * h2(0.1))

# ===========================================================================
section("5. Подсистемный код Бэкона–Шора [[9,1,3]]")


def q(r, c):
    return 3 * r + c


def pstr(ops):
    s = ["I"] * 9
    for qq, P_ in ops:
        s[qq] = P_
    return "".join(s)


gauge = [pstr([(q(r, c), "X"), (q(r + 1, c), "X")]) for r in range(2) for c in range(3)] + \
        [pstr([(q(r, c), "Z"), (q(r, c + 1), "Z")]) for r in range(3) for c in range(2)]
Gm = np.array([np.concatenate(sym(g)) for g in gauge])
rank_G = gf2_rank(Gm)
stabs_bs = [pstr([(q(r, c), "X") for c in range(3)] + [(q(r + 1, c), "X") for c in range(3)]) for r in range(2)] + \
           [pstr([(q(r, c), "Z") for r in range(3)] + [(q(r, c + 1), "Z") for r in range(3)]) for c in range(2)]
# стабилизатор = центр калибровочной группы: коммутируют со всеми калибровочными
for s_ in stabs_bs:
    assert all(sym_inner(*sym(s_), *sym(g)) == 0 for g in gauge)
    assert gf2_rank(np.vstack([Gm, np.concatenate(sym(s_))])) == rank_G  # лежат в калибровочной группе
XL_bs = pstr([(q(0, c), "X") for c in range(3)])
ZL_bs = pstr([(q(r, 0), "Z") for r in range(3)])
assert all(sym_inner(*sym(L), *sym(g)) == 0 for L in (XL_bs, ZL_bs) for g in gauge + stabs_bs)
assert sym_inner(*sym(XL_bs), *sym(ZL_bs)) == 1
n_s = gf2_rank(np.array([np.concatenate(sym(s_)) for s_ in stabs_bs]))
n_g = (rank_G - n_s) // 2
k_bs = 9 - n_s - n_g
assert (rank_G, n_s, n_g, k_bs) == (12, 4, 4, 1)
print(f"  калибровочная группа: 12 операторов веса 2 (XX по вертикали, ZZ по горизонтали), ранг {rank_G}")
print(f"  стабилизаторы — 4 оператора веса 6 (центр); калибровочных кубитов {n_g}; k = 9 - 4 - 4 = {k_bs}")
print(f"  X_L = X на строке ({XL_bs}), Z_L = Z на столбце ({ZL_bs}) — коммутируют со всей калибровочной группой")
# стабилизатор X-типа = произведение трёх вертикальных XX-пар
prod = np.zeros(18, dtype=np.int64)
for c in range(3):
    prod = (prod + np.concatenate(sym(gauge[c]))) % 2
assert np.array_equal(prod, np.concatenate(sym(stabs_bs[0])))
# код Шора = Бэкон–Шор с калибровочными ZZ, добавленными к стабилизатору (блоки — строки)
shor_gens = ["ZZIIIIIII", "IZZIIIIII", "IIIZZIIII", "IIIIZZIII", "IIIIIIZZI", "IIIIIIIZZ", "XXXXXXIII", "IIIXXXXXX"]
for g_ in shor_gens:
    assert gf2_rank(np.vstack([Gm, np.concatenate(sym(g_))])) == rank_G  # лежит в калибровочной группе
Sh = np.array([np.concatenate(sym(g_)) for g_ in shor_gens])
for s_ in stabs_bs:
    assert gf2_rank(np.vstack([Sh, np.concatenate(sym(s_))])) == gf2_rank(Sh)  # стабилизаторы БШ ⊂ стабилизатор Шора
print("  синдром X-стабилизатора = чётность исходов трёх измерений XX (веса 2): длинные операторы измерять не нужно")

# ===========================================================================
section("6. Схемный шум: извлечение синдрома кода Стина с одним вспомогательным кубитом")
# кубиты 0..6 — данные, 7..12 — вспомогательные (3 X-проверки, затем 3 Z-проверки)
supports = [[j for j in range(7) if h[j] == "1"] for h in hamming]
circuit = []  # список CNOT (c, t); H на анциллах X-проверок не влияет на ошибки данных в модели «только CNOT»
for i, sup in enumerate(supports):  # X-проверки: анцилла |+>, CNOT анцилла -> данные
    for j in sup:
        circuit.append((7 + i, j))
for i, sup in enumerate(supports):  # Z-проверки: анцилла |0>, CNOT данные -> анцилла
    for j in sup:
        circuit.append((j, 10 + i))
NQ = 13
print(f"  {len(circuit)} CNOT; шум — двухкубитная деполяризация p (15 паули по p/15) после каждого CNOT;")
print("  после зашумлённого раунда — идеальный раунд, декодирование по нему (только ошибки данных)")

PAULI2 = [(a, b) for a in range(4) for b in range(4) if (a, b) != (0, 0)]  # 0=I,1=X,2=Y,3=Z


def bits(pc):
    return (pc in (1, 2)), (pc in (2, 3))


def propagate(xf, zf, start):
    """Распространение паули-рамки через CNOT circuit[start:]; xf, zf — массивы (shots, NQ)."""
    for c, t in circuit[start:]:
        xf[:, t] ^= xf[:, c]
        zf[:, c] ^= zf[:, t]
    return xf, zf


def data_fail(xf, zf):
    x7, z7 = xf[:, :7].astype(int), zf[:, :7].astype(int)
    rx, rz = hamming_correct(x7), hamming_correct(z7)
    return ((rx.sum(1) % 2) == 1) | ((rz.sum(1) % 2) == 1)


# точный первый порядок: перебор одиночных отказов
fails1, hook = 0, 0
for loc, (c, t) in enumerate(circuit):
    for (a, b) in PAULI2:
        xf = np.zeros((1, NQ), dtype=bool)
        zf = np.zeros((1, NQ), dtype=bool)
        xa, za = bits(a)
        xb, zb = bits(b)
        xf[0, c], zf[0, c], xf[0, t], zf[0, t] = xa, za, xb, zb
        propagate(xf, zf, loc + 1)
        if data_fail(xf, zf)[0]:
            fails1 += 1
            hook += 1 if (xf[0, :7] | zf[0, :7]).sum() >= 2 else 0
A1 = fails1 / 15
print(f"  одиночных отказов (место x паули), ведущих к логической ошибке: {fails1} из {len(circuit) * 15}; "
      f"из них с весом ошибки данных >= 2: {hook}")
print(f"  первый порядок: p_L = A1 p + O(p^2), A1 = {fails1}/15 = {A1:.3f}")
assert fails1 > 0 and hook == fails1


def monte_carlo(p, shots):
    xf = np.zeros((shots, NQ), dtype=bool)
    zf = np.zeros((shots, NQ), dtype=bool)
    for (c, t) in circuit:
        hit = rng.random(shots) < p
        kind = rng.integers(15, size=shots)
        pa = np.array([PAULI2[k][0] for k in range(15)])[kind]
        pb = np.array([PAULI2[k][1] for k in range(15)])[kind]
        xf[:, c] ^= hit & ((pa == 1) | (pa == 2))
        zf[:, c] ^= hit & ((pa == 2) | (pa == 3))
        xf[:, t] ^= hit & ((pb == 1) | (pb == 2))
        zf[:, t] ^= hit & ((pb == 2) | (pb == 3))
        xf[:, t] ^= xf[:, c]
        zf[:, c] ^= zf[:, t]
    return data_fail(xf, zf).mean()


print("     p        p_L (Монте-Карло, 4e5)   A1 p        p_L / p")
MC = {}
for p in (1e-3, 3e-3, 1e-2):
    pl = monte_carlo(p, 400_000)
    MC[p] = pl
    print(f"   {p:.0e}     {pl:.3e}              {A1 * p:.3e}   {pl / p:.2f}")
slope = np.polyfit(np.log(list(MC)), np.log(list(MC.values())), 1)[0]
print(f"  наклон log p_L от log p: {slope:.2f} (одна ошибка — логический отказ)")
assert 0.85 < slope < 1.15
assert abs(MC[1e-3] / (A1 * 1e-3) - 1) < 0.1
# для сравнения: та же схема, но ошибки только на кубитах данных (без распространения с анцилл)
xf = np.zeros((1, NQ), dtype=bool)
zf = np.zeros((1, NQ), dtype=bool)
xf[0, 7] = True  # X на анцилле первой X-проверки после 2-го CNOT
propagate(xf, zf, 2)
hook_err = to_str(xf[0, :7].astype(int), zf[0, :7].astype(int))
assert weight(hook_err) == 2 and data_fail(xf, zf)[0]
syn_hook = tuple(int(v) for v in (Hm @ xf[0, :7].astype(int)) % 2)
assert syn_hook == (0, 0, 1) and np.array_equal(hamming_correct(xf[:, :7].astype(int))[0], [1, 0, 0, 0, 0, 1, 1])
print(f"  пример «крючка»: X на анцилле проверки X4X5X6X7 после двух CNOT -> ошибка данных {hook_err} -> логическая ошибка")

# ===========================================================================
section("7. Ответы к задачам")
print(f"  3: [[4,2,2]]: k = 2, d = 2; одиночные ошибки дают синдромы {sorted(syn422)}")
print(f"  5: |1_L> Стина — слова весов {w1}; S^7 = (S_L)^+")
print(f"  7: Синглтон n-k >= 2(d-1): " + ", ".join(f"{k_}: {v[0]} >= {v[1]}" for k_, v in BOUNDS.items()))
print(f"  8: CSS-ГВ: t/n = 0.01 -> k/n >= {CSS_RATES[0]:.3f}; t/n = 0.05 -> {CSS_RATES[1]:.3f}")
pt, pm = DEC[("Шора [[9,1,3]]", 0.1)]
print(f"  10: Шор, p = 0.1: таблица {pt:.6f}, ML по классам {pm:.6f}")
print(f"  12: Стин: N_fail(2) = {nf7[2]} из {tot7[2]}, A = {A7:.3f}; псевдопорог {pth7:.4f}")

print(f"\nВсе проверки пройдены. Время работы: {time.time() - T_START:.1f} с")
