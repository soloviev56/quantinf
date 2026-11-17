"""
Лекция I.11. Оракульные алгоритмы: битовый и фазовый оракулы, phase kickback,
обратимое вычисление и раскомпьютация, квантовый параллелизм, алгоритмы
Дойча, Дойча–Йожи, Бернштейна–Вазирани, Саймона (с гауссовым исключением
над GF(2)); сравнение с классическими алгоритмами.

Запуск:  python примеры.py
Требуется: numpy. Все проверки оформлены как assert.
Порядок кубитов: первый множитель — старший (как в N&C и np.kron).
"""
import numpy as np
from itertools import product
from math import comb

rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (X + Z) / np.sqrt(2)
minus = np.array([1, -1]) / np.sqrt(2)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def kron(*ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def bit_oracle(f, n, m=1):
    """O_f |x>|y> = |x>|y xor f(x)>, f: {0..2^n-1} -> {0..2^m-1}."""
    D = 2 ** (n + m)
    U = np.zeros((D, D))
    for x in range(2 ** n):
        for y in range(2 ** m):
            U[(x << m) | (y ^ f(x)), (x << m) | y] = 1
    return U


def phase_oracle(f, n):
    return np.diag([(-1.0) ** f(x) for x in range(2 ** n)])


def hadamard_n(n):
    return kron(*[H] * n)


def fwht(a, axis=0):
    """Быстрое преобразование Уолша–Адамара с нормировкой 2^{-n/2} (= H^{(x)n})."""
    a = np.array(a, dtype=complex)
    a = np.moveaxis(a, axis, 0)
    N = a.shape[0]
    h = 1
    while h < N:
        a = a.reshape(N // (2 * h), 2, h, *a.shape[1:])
        a = np.concatenate([(a[:, 0] + a[:, 1])[:, None], (a[:, 0] - a[:, 1])[:, None]], axis=1)
        a = a.reshape(N, *a.shape[3:])
        h *= 2
    return np.moveaxis(a / np.sqrt(N), 0, axis)


def dot2(a, b):
    return bin(a & b).count("1") % 2


# ---------------------------------------------------------------------------
section("1. Оракулы: битовый, фазовый, phase kickback")
for n in (1, 2, 3):
    for _ in range(5):
        tt = rng.integers(0, 2, 2 ** n)
        f = lambda x: int(tt[x])
        U = bit_oracle(f, n)
        assert np.allclose(U @ U.T, np.eye(2 ** (n + 1))) and np.allclose(U @ U, np.eye(2 ** (n + 1)))
        for x in range(2 ** n):
            v = np.kron(np.eye(2 ** n)[x], minus)
            assert np.allclose(U @ v, (-1) ** f(x) * v)
        # фазовый оракул = битовый на |->
        V = U @ np.kron(np.eye(2 ** n), minus.reshape(2, 1))
        assert np.allclose(V, np.kron(phase_oracle(f, n), minus.reshape(2, 1)))
print("  O_f — перестановочная унитарная (O_f^2 = I); O_f|x>|-> = (-1)^{f(x)}|x>|->  (n = 1..3, случайные f)")
print("  так фазовый оракул получается из битового с вспомогательным кубитом в |-> (допуск ЛР6)")
# многобитовый выход
n, m = 3, 2
tt = rng.integers(0, 4, 8)
U = bit_oracle(lambda x: int(tt[x]), n, m)
assert np.allclose(U @ U.T, np.eye(32))
print(f"  многобитовый оракул n = {n}, m = {m}: унитарен; таблица истинности требует 2^n = 8 строк — неэффективно в общем случае")

# ---------------------------------------------------------------------------
section("2. Обратимое вычисление и раскомпьютация мусора")
# f(x1,x2,x3) = (x1 AND x2) XOR x3; регистры: x(3) | a(1, мусор) | y(1)
n_all = 5
def toffoli(c1, c2, t, n):
    U = np.zeros((2 ** n, 2 ** n))
    for i in range(2 ** n):
        b = [(i >> (n - 1 - k)) & 1 for k in range(n)]
        if b[c1] and b[c2]:
            b[t] ^= 1
        j = sum(bit << (n - 1 - k) for k, bit in enumerate(b))
        U[j, i] = 1
    return U
def cnot(c, t, n):
    return toffoli(c, c, t, n)
compute = toffoli(0, 1, 3, n_all)
copy = cnot(2, 4, n_all) @ cnot(3, 4, n_all)        # y ^= a, y ^= x3
with_garbage = copy @ compute
clean = compute @ copy @ compute                    # раскомпьютация: повторное вычисление a обнуляет его
f = lambda x: (((x >> 2) & 1) & ((x >> 1) & 1)) ^ (x & 1)
for x in range(8):
    for y in (0, 1):
        i = (x << 2) | y
        assert np.argmax(clean[:, i]) == (x << 2) | (y ^ f(x))        # мусорный кубит снова 0
print("  f = (x1 AND x2) XOR x3: схема compute -> copy -> uncompute даёт |x>|0>|y xor f(x)>")
# phase kickback с мусором и без
psi_in = np.kron(np.kron(np.ones(8) / np.sqrt(8), [1, 0]), minus)
out_clean = clean @ psi_in
out_garb = with_garbage @ psi_in
def rho_x(state):
    M = state.reshape(8, 4)
    return M @ M.conj().T
target = np.array([(-1) ** f(x) for x in range(8)]) / np.sqrt(8)
assert np.allclose(rho_x(out_clean), np.outer(target, target))
pur_g = np.trace(rho_x(out_garb) @ rho_x(out_garb)).real
print(f"  без мусора: регистр x в чистом состоянии sum (-1)^f(x)|x>; с мусором: чистота rho_x = {pur_g:.3f} < 1")
assert pur_g < 0.99

# ---------------------------------------------------------------------------
section("3. Квантовый параллелизм: одно измерение — одна пара (x, f(x))")
n = 4
tt = rng.integers(0, 2, 16)
U = bit_oracle(lambda x: int(tt[x]), n)
st = U @ np.kron(hadamard_n(n) @ np.eye(16)[0], [1, 0])
probs = np.abs(st) ** 2
pairs = {(i >> 1, i & 1): p for i, p in enumerate(probs) if p > 1e-12}
assert len(pairs) == 16 and all(np.isclose(p, 1 / 16) for p in pairs.values())
assert all(y == tt[x] for (x, y) in pairs)
print("  sum_x |x>|f(x)>/4: измерение даёт равновероятно одну из 16 пар (x, f(x)) — не больше, чем один классический запрос")

# ---------------------------------------------------------------------------
section("4. Алгоритм Дойча")
for tt in product([0, 1], repeat=2):
    f = lambda x: tt[x]
    U = bit_oracle(f, 1)
    st = np.kron(H, H) @ np.kron(I2, I2) @ np.array([0, 1, 0, 0], complex)   # |0>|1> -> H(x)H
    st = U @ st
    st = np.kron(H, I2) @ st
    p1 = np.sum(np.abs(st.reshape(2, 2)[1]) ** 2)
    assert np.isclose(p1, tt[0] ^ tt[1])
    print(f"  f = {tt}: P(первый кубит = 1) = {p1:.0f} = f(0) xor f(1)")

# ---------------------------------------------------------------------------
section("5. Дойч–Йожа: n = 1..8, один запрос; классические стратегии")
def dj_prob_zero(fvals):
    """Вероятность исхода 0^n: |(1/N) sum (-1)^f|^2 (через фазовый оракул и FWHT)."""
    N = len(fvals)
    st = fwht((-1.0) ** fvals / np.sqrt(N))
    return abs(st[0]) ** 2
for n in range(1, 9):
    N = 2 ** n
    for _ in range(20):
        c = rng.integers(2)
        fc = np.full(N, c)
        fb = np.zeros(N, int); fb[rng.permutation(N)[:N // 2]] = 1
        assert np.isclose(dj_prob_zero(fc), 1) and np.isclose(dj_prob_zero(fb), 0)
print("  n = 1..8, по 20 случайных постоянных и сбалансированных f: P(0^n) = 1 и 0 соответственно")
# полный унитарный прогон для n = 3 (с вспомогательным кубитом)
n = 3
fb = np.array([0, 1, 1, 0, 1, 0, 0, 1])
U = bit_oracle(lambda x: int(fb[x]), n)
st = kron(hadamard_n(n), H) @ np.kron(np.eye(8)[0], [0, 1])
st = kron(hadamard_n(n), I2) @ (U @ st)
assert np.isclose(np.sum(np.abs(st.reshape(8, 2)[0]) ** 2), 0)
print("  полная схема на n+1 = 4 кубитах для сбалансированной f = x1 xor x2 xor x3: P(000) = 0")
# классика: детерминированно 2^{n-1}+1; вероятностно k случайных запросов
print("  классический вероятностный: k случайных различных запросов, ошибка на сбалансированной f")
for n, k in [(4, 2), (4, 3), (4, 5), (10, 5), (10, 10)]:
    N = 2 ** n
    exact = 2 * comb(N // 2, k) / comb(N, k)
    sims = []
    for _ in range(20000):
        fb = np.zeros(N, int); fb[:N // 2] = 1
        q = rng.choice(N, k, replace=False)
        sims.append(len(set(fb[q])) == 1)
    print(f"    N = {N:5d}, k = {k:2d}: точно 2 C(N/2,k)/C(N,k) = {exact:.4f}, моделирование {np.mean(sims):.4f}, 2^(1-k) = {2.0 ** (1 - k):.4f}")
    assert abs(np.mean(sims) - exact) < 0.01 and exact <= 2.0 ** (1 - k) + 1e-12
print("  детерминированный классический алгоритм: в худшем случае 2^(n-1) + 1 запросов (n = 8: 129)")

# ---------------------------------------------------------------------------
section("6. Бернштейн–Вазирани: s за один запрос (n до 16)")
for n in (4, 8, 12, 16):
    N = 2 ** n
    for _ in range(5):
        s = int(rng.integers(N))
        xs = np.arange(N)
        par = np.array([bin(v).count("1") % 2 for v in (xs & s)])
        st = fwht((-1.0) ** par / np.sqrt(N))
        out = int(np.argmax(np.abs(st)))
        assert out == s and np.isclose(abs(st[out]), 1)
    print(f"  n = {n:2d}: H^n O_s H^n |0> = |s> (5 случайных s), классически нужно n = {n} запросов")
# классический алгоритм: запросы e_i
n = 10; s = int(rng.integers(2 ** n))
rec = sum(dot2(s, 1 << (n - 1 - i)) << (n - 1 - i) for i in range(n))
assert rec == s

# ---------------------------------------------------------------------------
section("7. Алгоритм Саймона: n = 3..10, гауссово исключение над GF(2)")
def simon_oracle(n, s):
    """Случайная 2-к-1 функция с f(x) = f(x xor s), значения — случайные метки."""
    N = 2 ** n
    labels = rng.permutation(N)
    f = np.empty(N, int)
    seen = {}
    for x in range(N):
        r = min(x, x ^ s)
        if r not in seen:
            seen[r] = labels[len(seen)]
        f[x] = seen[r]
    return f
def simon_sample(f, n):
    """Один запуск: H^n, O_f, измерение второго регистра, H^n, измерение — точная симуляция."""
    N = 2 ** n
    M = np.zeros((N, N), complex)
    M[np.arange(N), f] = 1 / np.sqrt(N)              # sum_x |x>|f(x)>
    M = fwht(M, axis=0)                               # H^n на первом регистре
    p = np.sum(np.abs(M) ** 2, axis=1)
    return int(rng.choice(N, p=p / p.sum())), p
def gf2_rank_insert(basis, v):
    """Добавить v в базис (словарь ведущий_бит -> вектор); True, если ранг вырос."""
    while v:
        hb = v.bit_length() - 1
        if hb in basis:
            v ^= basis[hb]
        else:
            basis[hb] = v
            return True
    return False
def solve_s(basis, n):
    """Ненулевое решение y.s = 0 для всех y из базиса ранга n-1."""
    for cand in range(1, 2 ** n):
        if all(dot2(cand, b) == 0 for b in basis.values()):
            return cand
def expected_runs(n):
    return sum(1 / (1 - 2.0 ** (k - (n - 1))) for k in range(n - 1))
print("   n   среднее число запусков (моделирование)   теория sum 1/(1-2^{k-n+1})   n + 0.607")
for n in range(3, 11):
    runs = []
    for trial in range(200 if n <= 8 else 80):
        s = int(rng.integers(1, 2 ** n))
        f = simon_oracle(n, s)
        _, p = simon_sample(f, n)                     # распределение одного запуска (точная симуляция схемы)
        for y in range(2 ** n):
            assert np.isclose(p[y], (2 / 2 ** n) if dot2(y, s) == 0 else 0)
        basis = {}
        r = 0
        while len(basis) < n - 1:
            y = int(rng.choice(2 ** n, p=p / p.sum()))   # независимый повторный запуск
            r += 1
            gf2_rank_insert(basis, y)
        assert solve_s(basis, n) == s
        runs.append(r)
    print(f"  {n:2d}          {np.mean(runs):6.2f}                          {expected_runs(n):6.3f}                {n + 0.607:6.3f}")
    assert abs(np.mean(runs) - expected_runs(n)) < 1.0
print("  распределение исходов: p(y) = 2/2^n при y.s = 0, иначе 0 — проверено; s найдено во всех запусках")
const = sum(1 / (2 ** j - 1) for j in range(1, 60))
print(f"  предел E - (n-1) = sum_j 1/(2^j - 1) = {const:.4f}")
# классический поиск коллизии
print("  классический поиск коллизии случайными запросами: среднее число запросов")
cls = []
for n in (6, 8, 10, 12, 14, 16):
    N = 2 ** n
    qs = []
    for _ in range(300):
        s = int(rng.integers(1, N))
        seen = set()
        q = 0
        for x in rng.permutation(N):
            q += 1
            r = min(int(x), int(x) ^ s)
            if r in seen:
                break
            seen.add(r)
        qs.append(q)
    cls.append(np.mean(qs))
    print(f"    n = {n:2d}: {np.mean(qs):7.1f}   sqrt(pi 2^n / 2) = {np.sqrt(np.pi * N / 2):7.1f}")
slope = np.polyfit([6, 8, 10, 12, 14, 16], np.log2(cls), 1)[0]
print(f"  наклон log2(запросов) по n = {slope:.3f} (теория 1/2: Theta(2^(n/2)))")
assert 0.45 < slope < 0.55

# ---------------------------------------------------------------------------
section("8. Ответы к задачам семинара")
# 1: AND — Toffoli, фазовый — CZ
U = bit_oracle(lambda x: (x >> 1) & x & 1, 2)
assert np.allclose(U, toffoli(0, 1, 2, 3))
assert np.allclose(phase_oracle(lambda x: (x >> 1) & x & 1, 2), np.diag([1, 1, 1, -1]))
print("  1: битовый оракул AND = Toffoli, фазовый = CZ")
# 3: f с N/4 единицами
for n in (3, 6):
    N = 2 ** n
    fq = np.zeros(N, int); fq[:N // 4] = 1
    print(f"  3: n = {n}: f с N/4 единицами: P(0^n) = {dj_prob_zero(fq):.4f}")
    assert np.isclose(dj_prob_zero(fq), 0.25)
# 4: классическая ошибка N = 16, k = 3
e = 2 * comb(8, 3) / comb(16, 3)
print(f"  4: N = 16, k = 3 без повторений: {e:.4f};  с повторениями 2^(1-k) = {2 ** (1 - 3):.4f}")
assert np.isclose(e, 0.2)
# 5: BV для s = 1011
s = 0b1011; n = 4
par = np.array([dot2(x, s) for x in range(16)])
st = fwht((-1.0) ** par / 4)
assert np.argmax(np.abs(st)) == s
print("  5: s = 1011: на выходе |1011> с вероятностью 1")
# 7: Саймон n = 3, s = 110
ys = [y for y in range(8) if dot2(y, 0b110) == 0]
print("  7: s = 110: возможные y =", [format(y, "03b") for y in ys])
assert ys == [0b000, 0b001, 0b110, 0b111]
# 8: E(10) и вероятность независимости n-1 выборок
pind = np.prod([1 - 2.0 ** (-j) for j in range(1, 10)])
print(f"  8: n = 10: E = {expected_runs(10):.3f};  P(9 выборок независимы) = prod(1-2^-j) = {pind:.4f}")
pinf = np.prod([1 - 2.0 ** (-j) for j in range(1, 80)])
print(f"     предел при n -> inf: {pinf:.4f}")
# 9: мусор = копия x: P(0^n) = 1/N для любой f
for n in (3, 5):
    N = 2 ** n
    for fv in (np.zeros(N, int), np.r_[np.ones(N // 2, int), np.zeros(N // 2, int)]):
        # состояние sum_x (-1)^f(x) |x>|x> / sqrt N, затем H^n на первом регистре
        M = np.diag((-1.0) ** fv) / np.sqrt(N)
        M = fwht(M, axis=0)
        p0 = np.sum(np.abs(M[0]) ** 2)
        assert np.isclose(p0, 1 / N)
print("  9: если мусор — копия x, то P(0^n) = 1/N и для постоянной, и для сбалансированной f")
# 10: оценка вероятности успеха классического алгоритма: (q(q-1)/2 + 1) / (2^n - 1 - q(q-1)/2)
for n, q in [(20, 100), (20, 1000)]:
    pairs = q * (q - 1) // 2
    bound = (pairs + 1) / (2 ** n - 1 - pairs)
    print(f"  10: n = {n}, q = {q}: (q(q-1)/2 + 1)/(2^n - 1 - q(q-1)/2) = {bound:.3e}")
print("\nВсе проверки пройдены.")
