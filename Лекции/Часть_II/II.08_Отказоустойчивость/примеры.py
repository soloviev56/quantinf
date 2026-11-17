"""
Лекция II.8. Отказоустойчивые вычисления: распространение ошибок через гейты;
отказоустойчивое извлечение синдрома кода Стина с флаговыми кубитами (точный
перебор одиночных и парных отказов, Монте-Карло); конкатенация и пороговая
оценка; теорема Истина–Книлла на примерах кода Стина и кода Рида–Маллера
[[15,1,3]] (трансверсальный T); дистилляция магических состояний 15 -> 1
(точный перебор 2^15 конфигураций ошибок и проверка вектором состояния);
оценка ресурсов.

Соглашения (Лекции/Обозначения.md, §7, §18–§20). Кубиты нумеруются с 0 (в тексте
лекции — с 1). Код Стина: проверочная матрица Хэмминга, столбец j — двоичная
запись j (II.6, (3.1)). Шум — двухкубитная деполяризация с вероятностью p после
каждого CNOT (15 паули по p/15), как в II.6, §6.

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
"""
import time
import itertools
from math import comb, log, log10, ceil
import numpy as np
from scipy.optimize import brentq

T_START = time.time()
rng = np.random.default_rng(2026)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def kron(*ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


# ===========================================================================
section("1. Распространение ошибок через гейты")
conj = lambda U, P: U @ P @ U.conj().T
assert np.allclose(conj(CNOT, kron(X, I2)), kron(X, X))      # X на управляющем -> на обоих
assert np.allclose(conj(CNOT, kron(I2, X)), kron(I2, X))
assert np.allclose(conj(CNOT, kron(I2, Z)), kron(Z, Z))      # Z на целевом -> на обоих
assert np.allclose(conj(CNOT, kron(Z, I2)), kron(Z, I2))
assert np.allclose(conj(H, X), Z) and np.allclose(conj(S, X), Y)
TXT = conj(T, X)
assert np.allclose(TXT, (X + Y) / np.sqrt(2))
assert np.allclose(TXT, np.exp(-1j * np.pi / 4) * S @ X)
print("  CNOT: X_c -> X_c X_t, Z_t -> Z_c Z_t; H: X <-> Z; S: X -> Y")
print("  T X T^+ = (X + Y)/sqrt2 = e^{-i pi/4} S X — не паули: после T ошибка X становится клиффордовой")

# ===========================================================================
section("2. Извлечение синдрома кода Стина: без флагов и с флагами")
HAM = np.array([[(j >> (2 - r)) & 1 for j in range(1, 8)] for r in range(3)], dtype=np.uint8)
SUPP = [list(np.nonzero(HAM[r])[0]) for r in range(3)]  # [3,4,5,6], [1,2,5,6], [0,2,4,6]
POW = np.array([4, 2, 1])


def build_circuit(flags):
    """Схема: X-проверки (анцилла |+>, CNOT анцилла -> данные), затем Z-проверки
    (анцилла |0>, CNOT данные -> анцилла). С флагами: после 1-го и перед 4-м CNOT —
    CNOT анцилла -> флаг (X-проверка, флаг |0>, измерение Z) или флаг -> анцилла
    (Z-проверка, флаг |+>, измерение X). Возвращает список CNOT и описание флагов."""
    ops, flag_meas = [], []
    q = 7
    for typ in ("X", "Z"):
        for sup in SUPP:
            a = q
            f = q + 1 if flags else None
            q += 2 if flags else 1
            for i, j in enumerate(sup):
                ops.append((a, j) if typ == "X" else (j, a))
                if flags and i in (0, 2):
                    ops.append((a, f) if typ == "X" else (f, a))
            if flags:
                flag_meas.append((f, "Z" if typ == "X" else "X"))
    return ops, flag_meas, q


PAULI2 = [(a, b) for a in range(4) for b in range(4) if (a, b) != (0, 0)]  # 0=I,1=X,2=Y,3=Z
PA = np.array([p[0] for p in PAULI2])
PB = np.array([p[1] for p in PAULI2])
XB = lambda c: (c == 1) | (c == 2)
ZB = lambda c: (c == 2) | (c == 3)


def single_faults(ops, flag_meas, nq):
    """Все одиночные отказы (место x 15 паули): ошибка данных и флаги в конце раунда."""
    nf = len(ops) * 15
    xf = np.zeros((nf, nq), dtype=bool)
    zf = np.zeros((nf, nq), dtype=bool)
    loc = np.repeat(np.arange(len(ops)), 15)
    kind = np.tile(np.arange(15), len(ops))
    for l, (c, t) in enumerate(ops):
        xf[:, t] ^= xf[:, c]
        zf[:, c] ^= zf[:, t]
        m = loc == l
        xf[m, c] ^= XB(PA[kind[m]]); zf[m, c] ^= ZB(PA[kind[m]])
        xf[m, t] ^= XB(PB[kind[m]]); zf[m, t] ^= ZB(PB[kind[m]])
    return outcome(xf, zf, flag_meas), loc


def outcome(xf, zf, flag_meas):
    """Вектор исхода: (флаги, x-часть ошибки данных, z-часть) — линейная функция отказов."""
    fl = np.array([xf[:, f] if b == "Z" else zf[:, f] for f, b in flag_meas]).T if flag_meas else \
        np.zeros((xf.shape[0], 0), dtype=bool)
    return np.hstack([fl, xf[:, :7], zf[:, :7]])


def syn(v):
    """Синдром Хэмминга (номер позиции) для массивов строк длины 7."""
    return ((v.astype(np.int64) @ HAM.T.astype(np.int64)) % 2) @ POW


def ham_corr(s):
    c = np.zeros((len(s), 7), dtype=bool)
    nz = s > 0
    c[np.nonzero(nz)[0], s[nz] - 1] = True
    return c


class Decoder:
    """Без флагов — независимая коррекция X и Z кодом Хэмминга (II.6, §3.3).
    С флагами: если флаги сработали, используется таблица (флаги, синдром) -> исправление,
    построенная по одиночным отказам; если такой пары нет — обычная таблица."""

    def __init__(self, O1, nflag):
        self.nflag = nflag
        self.table = {}
        self.consistent = True
        if nflag:
            fl, ex, ez = O1[:, :nflag], O1[:, nflag:nflag + 7], O1[:, nflag + 7:]
            keys = self.keys(fl, ex, ez)
            for i in np.nonzero(fl.any(1))[0]:
                k = keys[i]
                if k not in self.table:
                    self.table[k] = (ex[i].copy(), ez[i].copy())
                else:  # все ошибки в одной ячейке должны быть эквивалентны с точностью до стабилизатора
                    dx, dz = ex[i] ^ self.table[k][0], ez[i] ^ self.table[k][1]
                    if dx.sum() % 2 or dz.sum() % 2:
                        self.consistent = False

    def keys(self, fl, ex, ez):
        fint = fl.astype(np.int64) @ (1 << np.arange(fl.shape[1], dtype=np.int64)) if fl.shape[1] else 0
        return (fint * 64 + syn(ex) * 8 + syn(ez)).tolist() if fl.shape[1] else (syn(ex) * 8 + syn(ez)).tolist()

    def fails(self, O):
        nfl = self.nflag
        fl, ex, ez = O[:, :nfl], O[:, nfl:nfl + 7], O[:, nfl + 7:]
        cx, cz = ham_corr(syn(ex)), ham_corr(syn(ez))
        if nfl:
            flagged = np.nonzero(fl.any(1))[0]
            if len(flagged):
                keys = self.keys(fl[flagged], ex[flagged], ez[flagged])
                for i, k in zip(flagged, keys):
                    if k in self.table:
                        cx[i], cz[i] = self.table[k]
        rx, rz = ex ^ cx, ez ^ cz
        assert not syn(rx).any() and not syn(rz).any()
        return (rx.sum(1) % 2 == 1) | (rz.sum(1) % 2 == 1)


RES = {}
for flags in (False, True):
    ops, fm, nq = build_circuit(flags)
    O1, loc = single_faults(ops, fm, nq)
    dec = Decoder(O1, len(fm))
    f1 = dec.fails(O1)
    # парные отказы на разных местах: исход — XOR исходов (паули-рамка линейна)
    n1 = len(O1)
    f2 = 0
    for i in range(n1 - 1):
        m = loc[i + 1:] != loc[i]
        f2 += dec.fails(O1[i + 1:][m] ^ O1[i]).sum()
    RES[flags] = dict(ops=ops, fm=fm, nq=nq, n_loc=len(ops), f1=int(f1.sum()), f2=int(f2), dec=dec,
                      A1=f1.sum() / 15, A2=f2 / 225, consistent=dec.consistent, ntab=len(dec.table))
    name = "с флагами" if flags else "без флагов"
    print(f"  {name}: {len(ops)} CNOT, {nq} кубитов; одиночных отказов -> логическая ошибка: {int(f1.sum())} из {n1}; "
          f"парных: {int(f2)} из {(n1 * (n1 - 15)) // 2}")
    print(f"      A1 = {RES[flags]['A1']:.3f}, A2 = {RES[flags]['A2']:.1f}  (p_L = A1 p + A2 p^2 + O(p^3))")
nb, nfl = RES[False], RES[True]
assert nb["f1"] == 48 and nb["n_loc"] == 24          # совпадает с II.6, §6
assert nfl["f1"] == 0 and nfl["consistent"] and nfl["n_loc"] == 36 and nfl["nq"] == 19
print(f"  таблица флагов: {nfl['ntab']} пар (флаги, синдром); все одиночные отказы в каждой ячейке эквивалентны -> t = 1")

# какие ошибки данных дают одиночные отказы на флагованной X-проверке X4X5X6X7
ops, fm, nq = nfl["ops"], nfl["fm"], nfl["nq"]
O1, loc = single_faults(ops, fm, nq)
hooks = set()
for i in range(8 * 15):  # первые 6 CNOT (4 данных + 2 флага) — первая X-проверка; берём 8 мест с запасом
    if loc[i] < 6 and O1[i, 0] and not O1[i, 13:].any():
        hooks.add(tuple(np.nonzero(O1[i, 6:13])[0] + 1))
print(f"  ошибки X на данных при сработавшем флаге первой проверки (кубиты с 1): {sorted(hooks)}")
assert (6, 7) in hooks  # крючок X6X7 теперь помечен флагом


def monte_carlo(ops, fm, nq, dec, p, shots):
    xf = np.zeros((shots, nq), dtype=bool)
    zf = np.zeros((shots, nq), dtype=bool)
    for c, t in ops:
        xf[:, t] ^= xf[:, c]
        zf[:, c] ^= zf[:, t]
        hit = rng.random(shots) < p
        k = rng.integers(15, size=shots)
        xf[:, c] ^= hit & XB(PA[k]); zf[:, c] ^= hit & ZB(PA[k])
        xf[:, t] ^= hit & XB(PB[k]); zf[:, t] ^= hit & ZB(PB[k])
    return dec.fails(outcome(xf, zf, fm)).mean()


print("     p        p_L с флагами (MC, 3e5)   A2 p^2       p_L без флагов (MC, 1e5)")
MC = {}
for p in (2e-3, 4e-3, 8e-3):
    pl = monte_carlo(ops, fm, nq, nfl["dec"], p, 300_000)
    plb = monte_carlo(nb["ops"], nb["fm"], nb["nq"], nb["dec"], p, 100_000)
    MC[p] = (pl, plb)
    print(f"   {p:.0e}     {pl:.3e}                 {nfl['A2'] * p * p:.3e}    {plb:.3e}")
slope_f = np.polyfit(np.log(list(MC)), np.log([v[0] for v in MC.values()]), 1)[0]
slope_b = np.polyfit(np.log(list(MC)), np.log([v[1] for v in MC.values()]), 1)[0]
print(f"  наклоны log p_L от log p: с флагами {slope_f:.2f}, без флагов {slope_b:.2f}")
assert 1.75 < slope_f < 2.25 and 0.85 < slope_b < 1.2
assert abs(MC[2e-3][0] / (nfl["A2"] * 4e-6) - 1) < 0.25
p_star_flag = 1 / nfl["A2"]
print(f"  псевдопорог одного раунда с флагами в приближении A2 p^2 = p: p* = 1/A2 = {p_star_flag:.2e}")

# ===========================================================================
section("3. Конкатенация и пороговая теорема")
A2 = nfl["A2"]
pth = 1 / A2
print(f"  рекурсия p_(k+1) = A p_k^2 с A = A2 = {A2:.1f}: p_th = 1/A = {pth:.2e}")
print(f"  p_k = p_th (p/p_th)^(2^k)")
CONC = {}
for ratio in (0.1, 0.5):
    p = ratio * pth
    row = []
    pk = p
    for k in range(1, 5):
        pk = A2 * pk * pk
        row.append(pk)
        assert np.isclose(pk, pth * ratio ** (2 ** k))
    CONC[ratio] = row
    print(f"  p = {ratio} p_th: " + ", ".join(f"k={k + 1}: {v:.1e}" for k, v in enumerate(row)))


def levels_needed(ratio, target):
    k = 0
    while pth * ratio ** (2 ** k) > target:
        k += 1
    return k


LV = {(r, t): levels_needed(r, t) for r in (0.1, 0.5) for t in (1e-12, 1e-15)}
for (r, t), k in LV.items():
    print(f"  p = {r} p_th, цель {t:.0e}: уровней k = {k}, блок 7^k = {7 ** k} физических кубитов на логический")
assert LV[(0.1, 1e-15)] == 4 and LV[(0.5, 1e-15)] == 6
# полилогарифмический оверхед: размер ~ (log(1/eps))^(log2 7)
print(f"  log2 7 = {log(7, 2):.3f}: размер блока растёт как (log 1/eps)^2.81")
# грубая оценка снизу по числу пар мест
NLOC = nfl["n_loc"]
print(f"  грубая оценка: мест {NLOC}, пар C({NLOC},2) = {comb(NLOC, 2)}; если любая пара фатальна: "
      f"p_th >= 1/{comb(NLOC, 2)} = {1 / comb(NLOC, 2):.2e} (фатальна лишь доля {nfl['f2'] / (len(O1) * (len(O1) - 15) // 2):.2f} пар)")
assert 1 / comb(NLOC, 2) < pth

# ===========================================================================
section("4. Теорема Истина–Книлла: трансверсальные гейты кода Стина и кода [[15,1,3]]")


def css_codewords(gx_rows, n):
    """|0_L> = сумма по линейной оболочке X-стабилизаторов, |1_L> — со сдвигом на 1...1."""
    words = set()
    r = len(gx_rows)
    for coef in itertools.product((0, 1), repeat=r):
        w = np.zeros(n, dtype=np.uint8)
        for c, row in zip(coef, gx_rows):
            if c:
                w ^= row
        words.add(tuple(w))
    v0 = np.zeros(2 ** n, dtype=complex)
    v1 = np.zeros(2 ** n, dtype=complex)
    pw = 1 << np.arange(n - 1, -1, -1)
    for w in words:
        v0[int(np.dot(w, pw))] += 1
        v1[int(np.dot(np.array(w) ^ 1, pw))] += 1
    return v0 / np.linalg.norm(v0), v1 / np.linalg.norm(v1), words


def apply_1q_all(U, psi, n):
    """U^(x n) |psi>."""
    t = psi.reshape([2] * n)
    for q in range(n):
        t = np.moveaxis(np.tensordot(U, t, axes=([1], [q])), 0, q)
    return t.reshape(-1)


def apply_1q(U, psi, n, q):
    t = psi.reshape([2] * n)
    t = np.moveaxis(np.tensordot(U, t, axes=([1], [q])), 0, q)
    return t.reshape(-1)


def logical_matrix(U, c0, c1, n):
    """Матрица U^(x n) в базисе кода и доля, вышедшая из кода (максимум по базису и |+_L>)."""
    B = np.array([c0, c1]).T
    leak = 0.0
    M = np.zeros((2, 2), dtype=complex)
    for i, v in enumerate([c0, c1, (c0 + c1) / np.sqrt(2)]):
        w = apply_1q_all(U, v, n)
        proj = B.conj().T @ w
        leak = max(leak, 1 - np.linalg.norm(proj) ** 2)
        if i < 2:
            M[:, i] = proj
    return M, leak


def eq_phase(A, B):
    k = np.argmax(np.abs(B))
    ph = A.flat[k] / B.flat[k]
    return abs(abs(ph) - 1) < 1e-9 and np.allclose(A, ph * B, atol=1e-9)


# код Стина
c0s, c1s, _ = css_codewords(HAM, 7)
for name, U, target in (("H", H, H), ("S", S, S.conj().T), ("S^+", S.conj().T, S)):
    M, lk = logical_matrix(U, c0s, c1s, 7)
    assert lk < 1e-12 and eq_phase(M, target)
M, lkT7 = logical_matrix(T, c0s, c1s, 7)
assert abs(lkT7 - 0.4375) < 1e-12
print(f"  Стин: H^7 = H_L, S^7 = (S_L)^+; T^7 выводит из кода долю {lkT7:.4f} (II.6, §3.4)")
# все 24 однокубитных клиффорда трансверсальны в коде Стина (по модулю фазы)
gens = [H, S]
cliff = [I2]
frontier = [I2]
while frontier:
    new = []
    for A in frontier:
        for g in gens:
            B = g @ A
            if not any(eq_phase(B, C) for C in cliff):
                cliff.append(B)
                new.append(B)
    frontier = new
assert len(cliff) == 24
n_tr = sum(logical_matrix(U, c0s, c1s, 7)[1] < 1e-12 for U in cliff)
assert n_tr == 24
print(f"  все {len(cliff)} однокубитных клиффордов U дают логические U^7 (утечка 0)")

# условия Книлла–Лафламма для одиночных паули => Pi h_j Pi = (Tr h_j / 2) Pi для любой однокубитной h_j
B7 = np.array([c0s, c1s]).T
for q in range(7):
    for P in (X, Y, Z):
        blk = B7.conj().T @ apply_1q(P, c0s, 7, q), B7.conj().T @ apply_1q(P, c1s, 7, q)
        assert np.allclose(blk, 0)
hrand = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
hrand = hrand + hrand.conj().T
blk = np.array([B7.conj().T @ apply_1q(hrand, v, 7, 3) for v in (c0s, c1s)]).T
assert np.allclose(blk, np.trace(hrand) / 2 * np.eye(2))
print("  Стин: Pi P_j Pi = 0 для всех однокубитных паули => Pi h_j Pi = (Tr h_j/2) Pi: "
      "генератор sum_j h_j действует на коде как число")
# дискретность: однопараметрическое семейство (e^{i th Z})^{\otimes 7} сохраняет код лишь при th = k pi/4
thetas = np.arange(0, 128) * np.pi / 64
ok = [th for th in thetas if logical_matrix(np.diag([np.exp(1j * th), np.exp(-1j * th)]), c0s, c1s, 7)[1] < 1e-10]
ok_k = sorted(set(int(round(th / (np.pi / 4))) for th in ok))
assert all(abs(th / (np.pi / 4) - round(th / (np.pi / 4))) < 1e-9 for th in ok) and ok_k == list(range(8))
print(f"  (e^(i th Z))^7, th = 0..2pi с шагом pi/64: код сохраняется только при th = k pi/4, k = {ok_k}")

# код Рида–Маллера [[15,1,3]]
H4 = np.array([[(j >> (3 - r)) & 1 for j in range(1, 16)] for r in range(4)], dtype=np.uint8)
GZ = [H4[r] for r in range(4)] + [H4[a] & H4[b] for a in range(4) for b in range(a + 1, 4)]
c0r, c1r, words_rm = css_codewords(H4, 15)
# проверка стабилизаторов: X-генераторы — строки H4, Z-генераторы — строки H4 и их попарные произведения
PW15 = 1 << np.arange(14, -1, -1)
idx = np.arange(2 ** 15)
bits15 = ((idx[:, None] & PW15[None, :]) > 0).astype(np.uint8)
for v in (c0r, c1r):
    for g in GZ:
        ph = (-1.0) ** ((bits15 @ g) % 2)
        assert np.allclose(ph * v, v)
    for g in H4:
        flip = idx ^ int(np.dot(g, PW15))
        assert np.allclose(v[flip], v)
wts0 = sorted(set(sum(w) for w in words_rm))
wts1 = sorted(set(15 - sum(w) for w in words_rm))
assert wts0 == [0, 8] and wts1 == [7, 15] and len(GZ) == 10
M, lk = logical_matrix(T, c0r, c1r, 15)
assert lk < 1e-12 and eq_phase(M, T.conj().T)
Mt, lk2 = logical_matrix(T.conj().T, c0r, c1r, 15)
assert lk2 < 1e-12 and eq_phase(Mt, T)
_, lkH = logical_matrix(H, c0r, c1r, 15)
print(f"  [[15,1,3]]: 4 X-генератора (вес 8), 10 Z-генераторов; веса слов |0_L>: {wts0}, |1_L>: {wts1}")
print(f"  T^15 = (T_L)^+, (T^+)^15 = T_L (утечка 0); H^15 выводит из кода долю {lkH:.4f}")
assert lkH > 0.1
# дистанция 3 по Z-ошибкам (Z-логические в C_X^perp \ span(GZ) — нечётные слова кода Хэмминга [15,11,3])
ham15 = [e for e in itertools.product((0, 1), repeat=15) if not ((H4 @ np.array(e)) % 2).any()]
odd_w = sorted(set(sum(e) for e in ham15 if sum(e) % 2))
assert len(ham15) == 2 ** 11 and odd_w[0] == 3
n_w3 = sum(1 for e in ham15 if sum(e) == 3)
assert n_w3 == 35
print(f"  логические Z-ошибки — нечётные слова кода Хэмминга [15,11]: минимальный вес {odd_w[0]}, слов веса 3: {n_w3}")

# ===========================================================================
section("5. Дистилляция магических состояний 15 -> 1")
# Шум |T>: (1-eps)|T><T| + eps Z|T><T|Z (после твирлинга); Z-ошибки входов переходят в Z-ошибки блока кода.
# Принимаем, если 4 X-проверки дали 0 (e в коде Хэмминга); выход ошибочен, если e — нечётное слово.
# твирлинг W = e^{-i pi/4} S X (W^2 = I, W|T> = |T>, W Z|T> = -Z|T>) делает шум «Z с вероятностью eps»
Wt = np.exp(-1j * np.pi / 4) * S @ X
Tket = np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2)
Tperp = Z @ Tket
assert np.allclose(Wt @ Wt, I2) and np.allclose(Wt @ Tket, Tket) and np.allclose(Wt @ Tperp, -Tperp)
G = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
rho_noisy = 0.9 * np.outer(Tket, Tket.conj()) + 0.1 * (G @ G.conj().T) / np.trace(G @ G.conj().T)
rho_tw = (rho_noisy + Wt @ rho_noisy @ Wt.conj().T) / 2
Bt = np.array([Tket, Tperp]).T
rt = Bt.conj().T @ rho_tw @ Bt
assert abs(rt[0, 1]) < 1e-12 and np.isclose(rt[1, 1], (Tperp.conj() @ rho_noisy @ Tperp).real)
print(f"  твирлинг W = e^(-i pi/4) S X: шумное |T> -> (1-eps)|T><T| + eps Z|T><T|Z, eps = {rt[1, 1].real:.4f} (пример)")
W = np.bincount([sum(e) for e in ham15], minlength=16)
Wodd = np.array([W[w] if w % 2 else 0 for w in range(16)])
print(f"  весовой спектр кода Хэмминга [15,11]: {dict((w, int(c)) for w, c in enumerate(W) if c)}")


def distill(eps):
    w = np.arange(16)
    pw = eps ** w * (1 - eps) ** (15 - w)
    acc = (W * pw).sum()
    bad = (Wodd * pw).sum()
    return acc, bad / acc


for eps in (1e-2, 1e-3, 1e-4):
    acc, eout = distill(eps)
    print(f"  eps = {eps:.0e}: P_acc = {acc:.6f} (1 - 15 eps = {1 - 15 * eps:.6f}), eps_out = {eout:.3e} "
          f"(35 eps^3 = {35 * eps ** 3:.3e})")
    assert abs(eout / (35 * eps ** 3) - 1) < 25 * eps + 1e-6
    assert abs(acc - (1 - 15 * eps)) < 120 * eps ** 2
eps_th = brentq(lambda e: distill(e)[1] - e, 0.01, 0.4)
print(f"  порог протокола (eps_out = eps): eps* = {eps_th:.4f}")

# проверка вектором состояния: |+_L> -> T^15 -> Z^e -> синдром и логическое состояние
plusL = (c0r + c1r) / np.sqrt(2)
target = (c0r + np.exp(-1j * np.pi / 4) * c1r) / np.sqrt(2)  # (T_L)^+ |+_L>
base = apply_1q_all(T, plusL, 15)
assert np.isclose(abs(np.vdot(target, base)), 1)
BR = np.array([c0r, c1r]).T


def z_pattern(e, psi):
    return psi * (-1.0) ** ((bits15 @ np.array(e, dtype=np.uint8)) % 2)


examples = {
    "вес 1 (обнаружена)": tuple(1 if j == 4 else 0 for j in range(15)),
    "вес 3, слово Хэмминга (логическая)": next(e for e in ham15 if sum(e) == 3),
    "вес 4, слово Хэмминга (стабилизатор)": next(e for e in ham15 if sum(e) == 4 and
                                                  any(np.array_equal(np.array(e, np.uint8), g) for g in GZ)),
}
for name, e in examples.items():
    w = z_pattern(e, base)
    pin = np.linalg.norm(BR.conj().T @ w) ** 2
    fid = abs(np.vdot(target, w)) ** 2
    print(f"  {name}: доля в коде {pin:.3f}, верность с (T_L)^+|+_L> {fid:.3f}")
    if "обнаружена" in name:
        assert pin < 1e-12
    elif "логическая" in name:
        assert abs(pin - 1) < 1e-12 and fid < 1e-12
    else:
        assert abs(fid - 1) < 1e-12
# многоуровневая дистилляция
LEV = []
e = 1e-3
n_in = 1.0
while e > 1e-15:
    acc, e = distill(e)
    n_in = n_in * 15 / acc
    LEV.append((e, n_in))
print("  каскад от eps = 1e-3: " + "; ".join(f"уровень {k + 1}: eps = {v[0]:.1e}, входов на выход {v[1]:.0f}"
                                         for k, v in enumerate(LEV)))
assert len(LEV) == 2 and LEV[0][0] < 4e-8 and LEV[1][0] < 1e-20

# ===========================================================================
section("6. Оценка ресурсов: от физических кубитов к алгоритму")
# формула II.7, (5.1): p_L = A (p/p_th)^((d+1)/2), A = 0.1, p_th = 1e-2 — предположение для оценки
A_, pth_ = 0.1, 1e-2


def d_for(p, target):
    return next(d for d in range(3, 301, 2) if A_ * (p / pth_) ** ((d + 1) / 2) <= target * (1 + 1e-12))


# игрушечная оценка для RSA-2048 по параметрам Гидни–Экеро (I.13, §4.2): ~6200 логических кубитов, ~8 ч;
# цикл коррекции 1 мкс; суммарная вероятность отказа <= 1/3 (распределяем на все логические кубиты и циклы)
nL, hours, cycle = 6200, 8.0, 1e-6
rounds = hours * 3600 / cycle
target = (1 / 3) / (nL * rounds)
dR = d_for(1e-3, target)
qubits_per = 2 * dR ** 2 - 1
total = nL * qubits_per
print(f"  логических кубитов {nL}, циклов {rounds:.2e}, допустимая ошибка на кубит за цикл {target:.1e}")
print(f"  нужна дистанция d = {dR}: {qubits_per} физических на логический, всего {total:.2e} (без фабрик)")
assert dR == 27 and 8e6 < total < 1e7
# число T-состояний: ~4 T на Toffoli (или одно CCZ-состояние), 2.6e9 Toffoli (I.13, §4.2)
nT = 4 * 2.6e9
rate = nT / (hours * 3600)
print(f"  T-состояний ~ 4 x 2.6e9 = {nT:.1e}; темп {rate:.1e} в секунду = {rate * cycle:.2f} за цикл 1 мкс")
assert 0.3 < rate * cycle < 0.4
# требуемая ошибка одного T-состояния: <= (1/3)/nT
print(f"  требуемая ошибка одного T-состояния <= {1 / 3 / nT:.1e}: двух уровней 15->1 из eps = 1e-3 хватает "
      f"(eps_2 = {LEV[1][0]:.1e})")

# задача 9: то же при p = 2e-3
dR2 = d_for(2e-3, target)
total2 = nL * (2 * dR2 ** 2 - 1)
print(f"  задача 9: p = 2e-3: d = {dR2}, {2 * dR2 ** 2 - 1} на логический, всего {total2:.2e}")
assert dR2 == 39 and 1.8e7 < total2 < 2e7

# задача 8: гаджет телепортации T и sqrt(T): исход 1 даёт X U^+ |psi>, поправка U X U^+ ... -> U
def gadget(U, psi):
    """|psi> (x) U|+>, CNOT(2 -> 1), измерение кубита 1 в Z: состояния кубита 2 при исходах 0 и 1."""
    st = np.kron(psi, U @ np.array([1, 1]) / np.sqrt(2))
    C21 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])  # управляет кубит 2
    st = C21 @ st
    out = [st[2 * m:2 * m + 2] for m in (0, 1)]
    return [o / np.linalg.norm(o) for o in out]


psi = rng.normal(size=2) + 1j * rng.normal(size=2)
psi /= np.linalg.norm(psi)
o0, o1 = gadget(T, psi)
assert np.isclose(abs(np.vdot(T @ psi, o0)), 1) and np.isclose(abs(np.vdot(X @ T.conj().T @ psi, o1)), 1)
assert np.isclose(abs(np.vdot(T @ psi, S @ X @ o1)), 1)
sqT = np.diag([1, np.exp(1j * np.pi / 8)])
o0, o1 = gadget(sqT, psi)
assert np.isclose(abs(np.vdot(sqT @ psi, T @ X @ o1)), 1)
print("  задача 8: исход 1 даёт X U^+|psi>; поправка для T — SX (клиффорд), для sqrt(T) — TX (не клиффорд)")

# пример с Lambda (задача 12): p_L(3) = 3e-3, Lambda = 2 -> какая d для 1e-6?
pL3, Lam = 3e-3, 2.0
d12 = next(d for d in range(3, 101, 2) if pL3 / Lam ** ((d - 3) / 2) <= 1e-6)
print(f"  задача 12: p_L(3) = 3e-3, Lambda = 2: p_L <= 1e-6 при d = {d12} ({2 * d12 ** 2 - 1} кубитов); "
      f"при Lambda = 4: d = {next(d for d in range(3, 101, 2) if pL3 / 4 ** ((d - 3) / 2) <= 1e-6)}")
d12b = next(d for d in range(3, 101, 2) if pL3 / 4 ** ((d - 3) / 2) <= 1e-6)
assert d12 == 27 and d12b == 15

# ===========================================================================
section("7. Ответы к задачам")
print(f"  2: без флагов A1 = {nb['A1']:.2f} (48/15); с флагами A1 = 0, A2 = {nfl['A2']:.1f}")
print(f"  3: p = 0.1 p_th: p_1..p_4 = " + ", ".join(f"{v:.1e}" for v in CONC[0.1]) + f"; уровней для 1e-15: {LV[(0.1, 1e-15)]}")
print(f"  4: 1/C(36,2) = {1 / comb(36, 2):.2e}; по точному перебору p_th = 1/A2 = {pth:.2e}")
acc3, eo3 = distill(1e-3)
print(f"  7: eps = 1e-3: eps_out = {eo3:.2e}, P_acc = {acc3:.4f}; второй уровень {LEV[1][0]:.1e}; порог {eps_th:.3f}")
print(f"  9: RSA-2048 (игрушечно): p = 1e-3: d = {dR}, {total:.1e}; p = 2e-3: d = {dR2}, {total2:.1e} физических кубитов без фабрик")
print(f"  12: d = {d12} при Lambda = 2; d = {d12b} при Lambda = 4")

print(f"\nВсе проверки пройдены. Время работы: {time.time() - T_START:.1f} с")
