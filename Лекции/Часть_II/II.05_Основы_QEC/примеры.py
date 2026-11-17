"""
Лекция II.5. Основы квантовой коррекции ошибок: классический код повторения;
квантовые коды от X- и Z-ошибок (код на 3 кубитах), измерение синдрома на
вспомогательных кубитах, дискретизация ошибок; код Шора на 9 кубитах;
условия Книлла–Лафламма (проверка, вырожденность, построение восстановления);
квантовая граница Хэмминга и код [[5,1,3]]; точная логическая ошибка кодов
при независимом шуме.

Соглашения (Лекции/Обозначения.md, §18): p — вероятность ошибки на кубит;
«деполяризующий шум с вероятностью p» — X, Y, Z по p/3 (канал Delta_{4p/3});
bit flip X_p(rho) = (1-p) rho + p X rho X; phase flip — D_p (II.1).

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
Порядок множителей: первый — старший (np.kron); кубиты в коде нумеруются с 0.
"""
import time
import itertools
import numpy as np
from math import comb, log2
from functools import reduce
from scipy.linalg import expm, sqrtm, polar
from scipy.optimize import brentq

T_START = time.time()
rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def dag(A):
    return A.conj().T


def kron(*ops):
    return reduce(np.kron, ops)


def pauli(s):
    """Паули-строка 'XZZXI' -> матрица."""
    return kron(*[PAULI[c] for c in s])


def op_on(U, q, n):
    return kron(*[U if i == q else I2 for i in range(n)])


def ket(bits):
    v = np.zeros(2 ** len(bits), dtype=complex)
    v[int(bits, 2)] = 1
    return v


def cnot(c, t, n):
    P0 = np.diag([1, 0]).astype(complex)
    P1 = np.diag([0, 1]).astype(complex)
    return kron(*[P0 if i == c else I2 for i in range(n)]) + kron(*[P1 if i == c else (X if i == t else I2) for i in range(n)])


def rho_of(psi):
    return np.outer(psi, psi.conj())


def fid(psi, phi):
    return abs(np.vdot(psi, phi)) ** 2


# ---------------------------------------------------------------------------
# Симплектическая запись (I.9, §1.2): паули-строка <-> (x|z)
def sym(s):
    x = np.array([c in "XY" for c in s], dtype=np.uint8)
    z = np.array([c in "ZY" for c in s], dtype=np.uint8)
    return x, z


def sym_inner(x1, z1, x2, z2):
    return (np.dot(x1, z2) + np.dot(z1, x2)) % 2


class StabCode:
    """Стабилизаторный код с одним логическим кубитом: генераторы, X_L, Z_L."""

    def __init__(self, name, gens, XL, ZL, allowed="XYZ"):
        self.name, self.gens, self.XL, self.ZL = name, gens, XL, ZL
        self.allowed = allowed  # какие паули допускаются в исправлениях таблицы
        self.n = len(gens[0])
        self.Sx = np.array([sym(g)[0] for g in gens]).T  # n x r
        self.Sz = np.array([sym(g)[1] for g in gens]).T
        self.lx = [sym(XL), sym(ZL)]
        # проверки: генераторы коммутируют, логические коммутируют с ними и антикоммутируют друг с другом
        for a, b in itertools.combinations(gens, 2):
            assert sym_inner(*sym(a), *sym(b)) == 0
        for L in (XL, ZL):
            for g in gens:
                assert sym_inner(*sym(L), *sym(g)) == 0
        assert sym_inner(*sym(XL), *sym(ZL)) == 1

    def projector(self):
        n = self.n
        P = np.eye(2 ** n, dtype=complex)
        for g in self.gens:
            sign = -1 if g.startswith("-") else 1
            P = P @ (np.eye(2 ** n) + sign * pauli(g.lstrip("-"))) / 2
        return P

    def logical_states(self):
        P = self.projector()
        # |0_L>: +1 собственный вектор Z_L внутри кода
        PZ = P @ (np.eye(2 ** self.n) + pauli(self.ZL)) / 2
        w, v = np.linalg.eigh(PZ)
        k0 = v[:, -1]
        k1 = pauli(self.XL) @ k0
        return k0, k1

    def all_paulis(self):
        """Все 4^n паули-строк: массивы x, z (N x n), вес, синдром (целое)."""
        n = self.n
        idx = np.arange(4 ** n)
        digits = (idx[:, None] // 4 ** np.arange(n - 1, -1, -1)) % 4  # 0=I,1=X,2=Y,3=Z
        x = ((digits == 1) | (digits == 2)).astype(np.int64)
        z = ((digits == 2) | (digits == 3)).astype(np.int64)
        syn_bits = (x @ self.Sz + z @ self.Sx) % 2
        syn = syn_bits @ (2 ** np.arange(syn_bits.shape[1] - 1, -1, -1))
        w = (x | z).sum(1)
        return x, z, w, syn

    def decoder_table(self):
        """Таблица синдромов: для каждого синдрома — паули минимального веса (первая по порядку)."""
        if hasattr(self, "_table"):
            return self._table
        x, z, w, syn = self.all_paulis()
        idx = np.arange(4 ** self.n)
        digits = (idx[:, None] // 4 ** np.arange(self.n - 1, -1, -1)) % 4
        ok = np.all(np.isin(digits, [0] + ["IXYZ".index(c) for c in self.allowed]), axis=1)
        order = [i for i in np.lexsort((idx, w)) if ok[i]]
        table = {}
        for i in order:
            s = int(syn[i])
            if s not in table:
                table[s] = i
        self._table = (table, (x, z, w, syn))
        return self._table

    def correction(self, s):
        table, (x, z, _, _) = self.decoder_table()
        i = table[s]
        return "".join("IXZY"[x[i][q] + 2 * z[i][q]] for q in range(self.n))

    def isometry(self):
        if not hasattr(self, "_V"):
            k0, k1 = self.logical_states()
            self._V = np.stack([k0, k1], axis=1)
        return self._V

    def decode(self, rho):
        """Идеальное измерение синдрома + коррекция по таблице; результат — логическая матрица 2 x 2.
        Подпространство синдрома s натянуто на C_s|0_L>, C_s|1_L> (C_s — исправление из таблицы)."""
        if not hasattr(self, "_W"):
            r = len(self.gens)
            table, _ = self.decoder_table()
            assert len(table) == 2 ** r  # все синдромы достижимы
            V = self.isometry()
            self._W = [pauli(self.correction(s)) @ V for s in range(2 ** r)]
        if rho.ndim == 1:
            rho = rho_of(rho)
        return sum(dag(Ws) @ rho @ Ws for Ws in self._W)

    def failure_counts(self):
        """N_fail[w] — число паули-строк веса w, после которых таблица даёт логическую ошибку."""
        table, (x, z, w, syn) = self.decoder_table()
        cx = np.array([x[table[int(s)]] for s in syn])
        cz = np.array([z[table[int(s)]] for s in syn])
        rx, rz = (x + cx) % 2, (z + cz) % 2
        (Xx, Xz), (Zx, Zz) = self.lx
        comm_X = (rx @ Xz + rz @ Xx) % 2  # антикоммутирует с X_L -> есть Z_L-составляющая
        comm_Z = (rx @ Zz + rz @ Zx) % 2
        fail = (comm_X | comm_Z).astype(bool)
        N = np.array([np.sum(fail & (w == k)) for k in range(self.n + 1)])
        tot = np.array([np.sum(w == k) for k in range(self.n + 1)])
        return N, tot

    def pL(self, p, N=None):
        if N is None:
            N, _ = self.failure_counts()
        return sum(N[k] * (p / 3) ** k * (1 - p) ** (self.n - k) for k in range(self.n + 1))


def kl_matrix(V, errors):
    """Условия Книлла–Лафламма P E_a^+ E_b P = c_ab P, где P = V V^+ (V — изометрия кодирования).
    Эквивалентно V^+ E_a^+ E_b V = c_ab I. Возвращает (выполнены?, c)."""
    m, K = len(errors), V.shape[1]
    EV = [E @ V for E in errors]
    c = np.zeros((m, m), dtype=complex)
    ok = True
    for a in range(m):
        for b in range(m):
            M = dag(EV[a]) @ EV[b]
            c[a, b] = np.trace(M) / K
            if not np.allclose(M, c[a, b] * np.eye(K), atol=1e-10):
                ok = False
    return ok, c


# ===========================================================================
section("0. Классический код повторения")
for p in (0.1, 0.3, 0.5):
    pL3 = sum(comb(3, j) * p ** j * (1 - p) ** (3 - j) for j in (2, 3))
    assert np.isclose(pL3, 3 * p ** 2 - 2 * p ** 3)
print("  d = 3: p_L = 3p^2 - 2p^3 (проверено); точка безубыточности p_L = p:", end=" ")
roots = np.roots([-2, 3, -1, 0])
print(sorted(np.round(roots.real, 6)))
assert np.allclose(sorted(roots.real), [0, 0.5, 1])


def pL_rep(d, p):
    return sum(comb(d, j) * p ** j * (1 - p) ** (d - j) for j in range((d + 1) // 2, d + 1))


print("  p_L для кода повторения длины d (большинство голосов):")
print("     p      d=1      d=3        d=5        d=7        d=9")
REP = {}
for p in (0.01, 0.1, 0.3, 0.45, 0.55):
    vals = [pL_rep(d, p) for d in (1, 3, 5, 7, 9)]
    REP[p] = vals
    print(f"   {p:4.2f}  " + "  ".join(f"{v:.3e}" for v in vals))
assert all(np.diff(REP[0.45]) < 0) and all(np.diff(REP[0.55]) > 0)
print("  при p < 1/2 ошибка убывает с d, при p > 1/2 растёт: порог 1/2")
assert np.isclose(pL_rep(5, 0.1), 0.00856, atol=1e-5)

# ===========================================================================
section("1. Код от X-ошибок на трёх кубитах")
n3 = 3
alpha, beta = 0.6, 0.8j
psi_in = np.array([alpha, beta])
enc = cnot(0, 2, 3) @ cnot(0, 1, 3)
L3 = enc @ np.kron(psi_in, ket("00"))
assert np.allclose(L3, alpha * ket("000") + beta * ket("111"))
print("  кодирование CNOT(1->2) CNOT(1->3): a|0>+b|1> -> a|000> + b|111> (проверено)")

# схема синдрома: данные 0,1,2 и вспомогательные 3,4; анцилла 3 ← Z1Z2, анцилла 4 ← Z2Z3
n5 = 5
syn_circ = cnot(1, 4, n5) @ cnot(2, 4, n5) @ cnot(0, 3, n5) @ cnot(1, 3, n5)
SYN_TABLE3 = {"00": None, "10": 0, "11": 1, "01": 2}
print("  синдром (Z1Z2, Z2Z3) -> исправление: 00 -> нет, 10 -> X1, 11 -> X2, 01 -> X3")


def correct_bitflip(state3):
    """Полная схема: синдром на анциллах, измерение, условная коррекция. Возвращает список (вероятность, состояние)."""
    full = syn_circ @ np.kron(state3, ket("00"))
    out = []
    for s in ("00", "01", "10", "11"):
        proj = np.kron(np.eye(8), np.outer(ket(s), ket(s)))
        v = proj @ full
        pr = np.real(np.vdot(v, v))
        if pr < 1e-14:
            continue
        v = v / np.sqrt(pr)
        data = v.reshape(8, 4)[:, int(s, 2)]
        q = SYN_TABLE3[s]
        if q is not None:
            data = op_on(X, q, 3) @ data
        out.append((pr, s, data))
    return out


for q in range(3):
    res = correct_bitflip(op_on(X, q, 3) @ L3)
    assert len(res) == 1 and np.isclose(fid(res[0][2], L3), 1)
res = correct_bitflip(L3)
assert len(res) == 1 and res[0][1] == "00"
print("  каждая одиночная X-ошибка восстанавливается точно; синдром не зависит от a, b (проверено)")
# синдром не несёт информации о (a, b): вероятности синдромов одинаковы для разных входов
for a_, b_ in ((1, 0), (0, 1), (1 / np.sqrt(2), 1j / np.sqrt(2))):
    st = enc @ np.kron(np.array([a_, b_]), ket("00"))
    pr = {s: p_ for p_, s, _ in correct_bitflip(op_on(X, 1, 3) @ st)}
    assert np.isclose(pr["11"], 1)

# непрерывная ошибка: e^{-i theta X_2}
theta = 0.3
res = correct_bitflip(expm(-1j * theta * op_on(X, 1, 3)) @ L3)
probs = {s: p_ for p_, s, _ in res}
assert np.isclose(probs["00"], np.cos(theta) ** 2) and np.isclose(probs["11"], np.sin(theta) ** 2)
assert all(np.isclose(fid(d_, L3), 1) for _, _, d_ in res)
print(f"  ошибка e^(-i*{theta} X2): синдром 00 с вер. cos^2 = {probs['00']:.4f}, 11 с вер. sin^2 = {probs['11']:.4f}; "
      "после коррекции состояние точное в обоих случаях")


def apply_1q_channel(rho, kraus, q, n):
    """Канал на кубите q: свёртка по оси q (кет) и n+q (бра)."""
    r = rho.reshape([2] * (2 * n))
    out = np.zeros_like(r)
    for K in kraus:
        t = np.moveaxis(np.tensordot(K, r, axes=([1], [q])), 0, q)
        t = np.moveaxis(np.tensordot(K.conj(), t, axes=([1], [n + q])), 0, n + q)
        out += t
    return out.reshape(rho.shape)


def logical_pauli_probs(V, channel, decode):
    """Логический канал «кодирование V -> шум -> декодирование»: вероятности I, X, Y, Z и PTM R."""
    R = np.zeros((4, 4))
    paulis1 = [I2, X, Y, Z]
    for j, Pj in enumerate(paulis1):
        out = decode(channel(V @ Pj @ dag(V)))
        for i, Pi in enumerate(paulis1):
            R[i, j] = np.real(np.trace(Pi @ out)) / 2
    lam = np.diag(R)[1:]
    # канал Паули: lam_x = pI + pX - pY - pZ и т. д. (II.1, (5.2))
    M = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])
    probs = np.linalg.solve(M, np.concatenate([[1], lam]))
    return probs, R


REP3X = StabCode("repetition (X)", ["ZZI", "IZZ"], XL="XXX", ZL="ZII", allowed="X")
REP3Z = StabCode("repetition (Z)", ["XXI", "IXX"], XL="ZZZ", ZL="XII", allowed="Z")
V3 = REP3X.isometry()
assert np.isclose(fid(V3[:, 0], ket("000")), 1) and np.isclose(fid(V3[:, 1], ket("111")), 1)
k0_3, k1_3 = ket("000"), ket("111")


def pauli_channel_all(rho, kraus, n):
    for q in range(n):
        rho = apply_1q_channel(rho, kraus, q, n)
    return rho


for p in (0.05, 0.1, 0.2):
    bitflip = [np.sqrt(1 - p) * I2, np.sqrt(p) * X]
    dephase = [np.sqrt(1 - p) * I2, np.sqrt(p) * Z]
    probs, R = logical_pauli_probs(V3, lambda r: pauli_channel_all(r, bitflip, 3), REP3X.decode)
    assert np.isclose(probs[1], 3 * p ** 2 - 2 * p ** 3) and np.allclose(probs[2:], 0)
    probsZ, _ = logical_pauli_probs(V3, lambda r: pauli_channel_all(r, dephase, 3), REP3X.decode)
    assert np.isclose(probsZ[3], 3 * p - 6 * p ** 2 + 4 * p ** 3)
    probs_pf, _ = logical_pauli_probs(REP3Z.isometry(), lambda r: pauli_channel_all(r, dephase, 3), REP3Z.decode)
    assert np.isclose(probs_pf[1], 3 * p ** 2 - 2 * p ** 3)
    if p == 0.1:
        print(f"  bit flip X_p на каждом кубите, p = {p}: логический канал = (1-p_L) id + p_L X_L, "
              f"p_L = {probs[1]:.6f} = 3p^2 - 2p^3")
        print(f"  дефазировка D_p на каждом кубите, p = {p}: логическая Z-ошибка {probsZ[3]:.6f} = 3p - 6p^2 + 4p^3 "
              f"(хуже, чем без кода: {p})")
        print(f"  код от фазовых ошибок (|+++>, |--->, синдром X1X2, X2X3) при D_p: p_L = {probs_pf[1]:.6f} = 3p^2 - 2p^3")
        PZ_BITFLIP = probsZ
kp = REP3Z.isometry()[:, 0]
assert np.isclose(fid(kp, kron(H, H, H) @ ket("000")), 1)

# ===========================================================================
section("2. Код Шора на 9 кубитах")
SHOR = StabCode("Shor [[9,1,3]]",
                ["ZZIIIIIII", "IZZIIIIII", "IIIZZIIII", "IIIIZZIII", "IIIIIIZZI", "IIIIIIIZZ",
                 "XXXXXXIII", "IIIXXXXXX"],
                XL="ZIIZIIZII", ZL="XXXIIIIII")
blkp = (ket("000") + ket("111")) / np.sqrt(2)
blkm = (ket("000") - ket("111")) / np.sqrt(2)
sh0 = kron(blkp, blkp, blkp)
sh1 = kron(blkm, blkm, blkm)
P_shor = SHOR.projector()
assert np.isclose(np.trace(P_shor).real, 2)
assert np.allclose(P_shor @ sh0, sh0) and np.allclose(P_shor @ sh1, sh1)
del P_shor
assert np.allclose(pauli(SHOR.XL) @ sh0, sh1) and np.allclose(pauli(SHOR.ZL) @ sh1, -sh1)
assert np.allclose(pauli("ZZZZZZZZZ") @ sh0, sh1) and np.allclose(pauli("XXXXXXXXX") @ sh1, -sh1)
print("  |0_L> = (|000>+|111>)^3 / 2^(3/2), |1_L> = (|000>-|111>)^3 / 2^(3/2): кодовое пространство 8 стабилизаторов, dim 2")
print("  X_L = Z1 Z4 Z7 (~ Z^9), Z_L = X1 X2 X3 (~ X^9) — проверено")

# схема кодирования (N&C, рис. 10.6): фазовый код, затем побитовый в каждом блоке
enc9 = np.eye(512, dtype=complex)
for c, t in ((0, 3), (0, 6)):
    enc9 = cnot(c, t, 9) @ enc9
enc9 = kron(*[H if q in (0, 3, 6) else I2 for q in range(9)]) @ enc9
for b in (0, 3, 6):
    enc9 = cnot(b, b + 2, 9) @ cnot(b, b + 1, 9) @ enc9
assert np.allclose(enc9 @ ket("0" * 9), sh0) and np.allclose(enc9 @ ket("1" + "0" * 8), sh1)
print("  схема кодирования (2 CNOT, 3 H, 6 CNOT) даёт |0_L>, |1_L> (проверено)")

table_shor, (xs, zs, ws, syns) = SHOR.decoder_table()


def syndrome_of(code, s):
    x, z = sym(s)
    return tuple(int(v) for v in (x @ code.Sz + z @ code.Sx) % 2)


singles = ["I" * 9] + ["I" * q + P + "I" * (8 - q) for q in range(9) for P in "XYZ"]
syn_set = {}
for s in singles:
    syn_set.setdefault(syndrome_of(SHOR, s), []).append(s)
print(f"  28 паули веса <= 1 дают {len(syn_set)} различных синдромов (вырожденность: Z1, Z2, Z3 — один синдром)")
assert len(syn_set) == 22
assert syndrome_of(SHOR, "ZIIIIIIII") == syndrome_of(SHOR, "IZIIIIIII")
assert sym_inner(*sym("ZZIIIIIII"), *sym(SHOR.XL)) == 0  # Z1 Z2 — элемент стабилизатора

V9 = SHOR.isometry()
assert np.isclose(fid(V9[:, 0], sh0), 1) and np.isclose(fid(V9[:, 1], sh1), 1)
psi_l = np.array([0.6, 0.8j])
psiL = V9 @ psi_l


def logical_fid(rhoL, psi=psi_l):
    return np.real(np.vdot(psi, rhoL @ psi))


for s in singles:
    assert np.isclose(logical_fid(SHOR.decode(pauli(s) @ psiL)), 1)
print("  все 28 ошибок веса <= 1 исправляются таблицей синдромов (проверено на векторах размерности 512)")

# произвольная однокубитная ошибка (случайная унитарная) и произвольный канал (случайные Краусы)
for trial in range(3):
    A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    U = np.linalg.qr(A)[0]
    q = int(rng.integers(9))
    assert np.isclose(logical_fid(SHOR.decode(op_on(U, q, 9) @ psiL)), 1)
G = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
Vst = np.linalg.qr(G)[0]  # изометрия 2 -> 6: три оператора Крауса
kraus_rand = [Vst[2 * j:2 * j + 2, :] for j in range(3)]
assert np.allclose(sum(dag(K) @ K for K in kraus_rand), I2)
assert np.isclose(logical_fid(SHOR.decode(apply_1q_channel(rho_of(psiL), kraus_rand, 4, 9))), 1)
gam = 0.3
ad = [np.array([[1, 0], [0, np.sqrt(1 - gam)]], complex), np.array([[0, np.sqrt(gam)], [0, 0]], complex)]
assert np.isclose(logical_fid(SHOR.decode(apply_1q_channel(rho_of(psiL), ad, 7, 9))), 1)
# операторы Крауса A_gamma в базисе Паули: K0 = a I + b Z, K1 = sqrt(gamma)(X + iY)/2
a_, b_ = (1 + np.sqrt(1 - gam)) / 2, (1 - np.sqrt(1 - gam)) / 2
assert np.allclose(ad[0], a_ * I2 + b_ * Z) and np.allclose(ad[1], np.sqrt(gam) * (X + 1j * Y) / 2)
print("  случайные унитарные ошибки, случайный канал (3 Крауса) и A_0.3 на одном кубите исправляются точно: "
      "дискретизация ошибок")
# одна и та же ошибка на двух кубитах разных блоков (X1 X4) исправляется, X1 X2 — нет
assert np.isclose(logical_fid(SHOR.decode(pauli("XIIXIIIII") @ psiL)), 1)
assert logical_fid(SHOR.decode(pauli("XXIIIIIII") @ psiL)) < 0.99
print("  X1 X4 (два блока) исправляется, X1 X2 (один блок) — нет: дистанция 3")

# ===========================================================================
section("3. Условия Книлла–Лафламма")
errs_X = [np.eye(8)] + [op_on(X, q, 3) for q in range(3)]
ok, c = kl_matrix(V3, errs_X)
assert ok and np.allclose(c, np.eye(4))
errs_all3 = [np.eye(8)] + [op_on(P_, q, 3) for q in range(3) for P_ in (X, Y, Z)]
ok2, c2 = kl_matrix(V3, errs_all3)
assert not ok2
print("  код повторения: {I, X1, X2, X3} — условия выполнены, c = I; все однокубитные — нарушены")
P3 = V3 @ dag(V3)
M = P3 @ op_on(Z, 0, 3) @ P3
assert np.allclose(M, np.outer(k0_3, k0_3) - np.outer(k1_3, k1_3))
print("    P Z1 P = |000><000| - |111><111| — это Z_L на коде, не кратно P")

errs_shor = [pauli(s) for s in singles]
ok, c_shor = kl_matrix(V9, errs_shor)
offdiag = np.sum(np.abs(c_shor - np.diag(np.diag(c_shor))) > 1e-9)
assert ok and offdiag > 0
print(f"  код Шора, 28 ошибок: условия выполнены; c_ab недиагональна ({offdiag} ненулевых внедиагональных элементов) — вырожденный код")
assert np.isclose(c_shor[singles.index("ZIIIIIIII"), singles.index("IZIIIIIII")], 1)
rk = np.linalg.matrix_rank(c_shor, tol=1e-9)
print(f"    ранг c = {rk} (из 28)")
assert rk == 22

FIVE = StabCode("[[5,1,3]]", ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"], XL="XXXXX", ZL="ZZZZZ")
V5 = FIVE.isometry()
singles5 = ["I" * 5] + ["I" * q + P_ + "I" * (4 - q) for q in range(5) for P_ in "XYZ"]
ok, c5 = kl_matrix(V5, [pauli(s) for s in singles5])
assert ok and np.allclose(c5, np.eye(16))
# все нетривиальные элементы стабилизатора [[5,1,3]] имеют вес 4
for bits in itertools.product((0, 1), repeat=4):
    if any(bits):
        x_ = sum(b * sym(g)[0] for b, g in zip(bits, FIVE.gens)) % 2
        z_ = sum(b * sym(g)[1] for b, g in zip(bits, FIVE.gens)) % 2
        assert np.sum(x_ | z_) == 4
# внедиагональные элементы c у кода Шора — ровно пары Z_i Z_j внутри блока
nz = [(singles[a], singles[b]) for a in range(28) for b in range(28) if a != b and abs(c_shor[a, b]) > 1e-9]
assert all(s1.replace("I", "") == "Z" and s2.replace("I", "") == "Z" and s1.index("Z") // 3 == s2.index("Z") // 3 for s1, s2 in nz)
print("  код [[5,1,3]], 16 ошибок: условия выполнены, c = I — невырожденный код")

# построение восстановления по доказательству для кода Шора
w_, u_ = np.linalg.eigh(c_shor)
FV = [sum(u_[a, k] * (errs_shor[a] @ V9) for a in range(len(errs_shor))) for k in range(len(w_))]
keep = [k for k in range(len(w_)) if w_[k] > 1e-9]
assert len(keep) == rk
for k in keep:
    for l in keep:
        assert np.allclose(dag(FV[k]) @ FV[l], (w_[k] if k == l else 0) * np.eye(2), atol=1e-10)
A_rec = [dag(FV[k]) / np.sqrt(w_[k]) for k in keep]  # R_k = V^+ U_k^+ P_k в логических координатах
Ptot = sum(dag(A_) @ A_ for A_ in A_rec)
assert np.allclose(Ptot @ Ptot, Ptot) and np.isclose(np.trace(Ptot).real, 2 * len(keep))
print(f"  восстановление по доказательству: {len(keep)} операторов, подпространства F_k C взаимно ортогональны (проверено)")
for s in ("IIIIYIIII", "ZIIIIIIII", "IIIIIIIIX"):
    phi = pauli(s) @ psiL
    out = sum(np.outer(A_ @ phi, (A_ @ phi).conj()) for A_ in A_rec)
    assert np.isclose(logical_fid(out), 1)
print("  оно исправляет ошибки (проверено на Y5, Z1, X9)")

# ===========================================================================
section("4. Квантовая граница Хэмминга и код [[5,1,3]]")


def hamming_ok(n, k, t):
    return 2 ** k * sum(comb(n, j) * 3 ** j for j in range(t + 1)) <= 2 ** n


nmin = {}
for t in (1, 2, 3):
    nmin[t] = next(n for n in range(1, 60) if hamming_ok(n, 1, t))
    print(f"  k = 1, t = {t}: минимальное n по границе Хэмминга = {nmin[t]}")
assert nmin[1] == 5 and nmin[2] == 10 and nmin[3] == 15
assert 2 * (1 + 27 + 9 * comb(9, 2)) > 2 ** 9 and 2 * (1 + 30 + 9 * comb(10, 2)) <= 2 ** 10
assert 2 * (1 + 15) == 2 ** 5
syn5 = {syndrome_of(FIVE, s) for s in singles5}
assert len(syn5) == 16
print("  [[5,1,3]]: 2 * (1 + 3*5) = 32 = 2^5 — граница достигается; 16 ошибок веса <= 1 дают 16 разных синдромов")
k0_5, k1_5 = FIVE.logical_states()
amps = k0_5[np.abs(k0_5) > 1e-9]
assert len(amps) == 16 and np.allclose(np.abs(amps), 0.25)
print("  |0_L> кода [[5,1,3]] — суперпозиция 16 базисных векторов с амплитудами ±1/4")
# асимптотика: k/n <= 1 - (t/n) log2 3 - h(t/n)


def h2(x):
    return 0 if x in (0, 1) else -x * log2(x) - (1 - x) * log2(1 - x)


for tau in (0.05, 0.1, 0.189):
    print(f"    t/n = {tau}: k/n <= {1 - tau * log2(3) - h2(tau):.4f}")
tau0 = brentq(lambda x: 1 - x * log2(3) - h2(x), 0.01, 0.3)
print(f"  граница Хэмминга обращается в 0 при t/n = {tau0:.4f}")

# ===========================================================================
section("5. Логическая ошибка кодов при независимом шуме")
print("  деполяризующий шум с вероятностью p на каждом кубите (X, Y, Z по p/3), идеальное измерение синдрома,")
print("  таблица синдромов минимального веса. N_fail[w] — число паули-строк веса w с логической ошибкой:")
CODES = {}
for code in (REP3X, SHOR, FIVE):
    N, tot = code.failure_counts()
    CODES[code.name] = (code, N, tot)
    print(f"   {code.name:16s} N_fail = {list(N)}  (из {list(tot)})")
N_r, _ = CODES["repetition (X)"][1:]
assert N_r[1] == 6  # Y и Z на любом кубите — логическая ошибка: код не защищает от Z
N_s, tot_s = CODES["Shor [[9,1,3]]"][1:]
assert N_s[1] == 0
N_5, tot_5 = CODES["[[5,1,3]]"][1:]
assert N_5[1] == 0 and N_5[2] == tot_5[2]  # все ошибки веса 2 — логические (совершенный невырожденный код)
A_shor = N_s[2] / 9
A_five = N_5[2] / 9
print(f"  разложение p_L = A p^2 + ...: A(Шор) = N_fail[2]/9 = {A_shor:.4f}, A([[5,1,3]]) = {A_five:.4f}")
print(f"  ошибок веса 2, исправленных кодом Шора: {tot_s[2] - N_s[2]} из {tot_s[2]} "
      f"({(tot_s[2] - N_s[2]) / tot_s[2]:.3f}); у [[5,1,3]] — 0")
# разбор отказов кода Шора на ошибках веса 2
table_s, (xs_, zs_, ws_, syns_) = SHOR.decoder_table()
(Xx_, Xz_), (Zx_, Zz_) = SHOR.lx
n_w1corr, n_w2corr = 0, 0
for i in np.where(ws_ == 2)[0]:
    c_ = table_s[int(syns_[i])]
    rx_, rz_ = (xs_[i] + xs_[c_]) % 2, (zs_[i] + zs_[c_]) % 2
    if ((rx_ @ Xz_ + rz_ @ Xx_) % 2) or ((rx_ @ Zz_ + rz_ @ Zx_) % 2):
        if ws_[c_] == 1:
            n_w1corr += 1
        else:
            n_w2corr += 1
assert (n_w1corr, n_w2corr) == (63, 54)
print(f"  отказы Шора на весе 2: {n_w1corr} — синдром как у ошибки веса 1 (таблица выбирает её), "
      f"{n_w2corr} — неоднозначность среди ошибок веса 2")
PTH = {}
for name in ("Shor [[9,1,3]]", "[[5,1,3]]"):
    code, N, _ = CODES[name]
    pth = brentq(lambda p: code.pL(p, N) - p, 1e-4, 0.4)
    PTH[name] = pth
    print(f"   {name:16s}: p_L(0.001) = {code.pL(1e-3, N):.3e}, p_L(0.01) = {code.pL(1e-2, N):.3e}; "
          f"псевдопорог p_L = p при p = {pth:.4f}")
assert np.isclose(SHOR.pL(1e-3, N_s), A_shor * 1e-6, rtol=0.02)

# сверка p_L кода Шора с матричным расчётом при деполяризующем шуме (через логический канал)
p_chk = 0.05
dep_k = [np.sqrt(1 - p_chk) * I2, np.sqrt(p_chk / 3) * X, np.sqrt(p_chk / 3) * Y, np.sqrt(p_chk / 3) * Z]


def ch_dep9(rho):
    for q in range(9):
        rho = apply_1q_channel(rho, dep_k, q, 9)
    return rho


probs_shor, _ = logical_pauli_probs(V9, ch_dep9, SHOR.decode)
assert np.isclose(1 - probs_shor[0], SHOR.pL(p_chk, N_s), atol=1e-12)
print(f"  сверка с матричным расчётом (Шор, p = {p_chk}): 1 - p_I = {1 - probs_shor[0]:.6f} = p_L по перечислению")
print(f"    логические X, Y, Z: {probs_shor[1]:.5f}, {probs_shor[2]:.5f}, {probs_shor[3]:.5f}")

# амплитудное затухание на всех 9 кубитах кода Шора: логическая ошибка ~ gamma^2
print("  код Шора при амплитудном затухании A_gamma на всех кубитах (восстановление — таблица синдромов):")
AD_RES = {}
for gam in (0.01, 0.02, 0.04):
    adk = [np.array([[1, 0], [0, np.sqrt(1 - gam)]], complex), np.array([[0, np.sqrt(gam)], [0, 0]], complex)]

    def ch_ad9(rho, adk=adk):
        for q in range(9):
            rho = apply_1q_channel(rho, adk, q, 9)
        return rho
    pr, Rm = logical_pauli_probs(V9, ch_ad9, SHOR.decode)
    Fpro = (1 + Rm[1, 1] + Rm[2, 2] + Rm[3, 3]) / 4  # верность процесса логического канала
    AD_RES[gam] = 1 - Fpro
    print(f"    gamma = {gam}: 1 - F_pro = {1 - Fpro:.3e}, (1 - F_pro)/gamma^2 = {(1 - Fpro) / gam ** 2:.3f}")
ratio = AD_RES[0.02] / AD_RES[0.01]
assert 3.7 < ratio < 4.3
print(f"  отношение при удвоении gamma: {ratio:.3f} ~ 4: ошибка второго порядка")

# ===========================================================================
section("6. Ответы к задачам")
print(f"  1: p_L = 3p^2 - 2p^3; при p = 0.1: {3 * 0.01 - 2 * 0.001:.3f}; безубыточность p = 1/2")
print(f"  4: дефазировка на коде от X: P(нечётное число Z) = (1-(1-2p)^3)/2 = 3p - 6p^2 + 4p^3; при p = 0.1: {PZ_BITFLIP[3]:.4f}")
th3 = 0.3
print(f"  3: e^(-i theta X2), theta = {th3}: синдром 11 с вер. sin^2 = {np.sin(th3) ** 2:.4f}")
print(f"  7: минимальное n (k = 1) по Хэммингу: t = 1 -> {nmin[1]}, t = 2 -> {nmin[2]}")
print(f"  8: Шор при A_gamma на всех кубитах: (1 - F_pro)/gamma^2 = {AD_RES[0.01] / 0.01 ** 2:.2f} при gamma = 0.01")
print(f"  11: d = 5, p = 0.1: p_L = {pL_rep(5, 0.1):.5f}")
print(f"  12: A(Шор) = {A_shor:.3f}, псевдопорог {PTH['Shor [[9,1,3]]']:.4f}; A([[5,1,3]]) = {A_five:.3f}, псевдопорог {PTH['[[5,1,3]]']:.4f}")

print(f"\nВсе проверки пройдены. Время работы: {time.time() - T_START:.1f} с")
