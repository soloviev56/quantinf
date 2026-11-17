"""
Лекция I.13. Алгоритм Шора и задача о скрытой подгруппе: сведение факторизации
к поиску порядка, доля «хороших» оснований, оператор модульного умножения и его
собственные векторы, поиск порядка через QPE, цепные дроби, полный конвейер Шора
для малых N, дискретный логарифм как абелева HSP, «скомпилированные» схемы,
грубые ресурсные оценки.

Запуск:  python примеры.py
Требуется: numpy. Все проверки оформлены как assert.
Порядок кубитов: первый множитель — старший (как в N&C и np.kron).
"""
import numpy as np
from math import gcd, pi, log2, ceil
from fractions import Fraction
from functools import lru_cache

rng = np.random.default_rng(2026)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


@lru_cache(maxsize=None)
def factorize(n):
    f, p = {}, 2
    while p * p <= n:
        while n % p == 0:
            f[p] = f.get(p, 0) + 1
            n //= p
        p += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def lcm(a, b):
    return a * b // gcd(a, b)


@lru_cache(maxsize=None)
def carmichael(N):
    lam = 1
    for p, k in factorize(N).items():
        l = (p - 1) * p ** (k - 1)
        if p == 2 and k >= 3:
            l //= 2
        lam = lcm(lam, l)
    return lam


def order(a, N):
    """Мультипликативный порядок a по модулю N (делитель lambda(N))."""
    lam = carmichael(N)
    r = lam
    for p in factorize(lam).keys():
        while r % p == 0 and pow(a, r // p, N) == 1:
            r //= p
    return r


def order_brute(a, N):
    r, x = 1, a % N
    while x != 1:
        x = x * a % N
        r += 1
    return r


def good_base(a, N):
    """Даёт ли a (взаимно простое с N) нетривиальный делитель через порядок."""
    r = order(a, N)
    if r % 2:
        return False
    y = pow(a, r // 2, N)
    return y != N - 1


# ---------------------------------------------------------------------------
section("1. Сведение факторизации к поиску порядка")
for N in (15, 21, 33, 35, 91):
    for a in range(2, N):
        if gcd(a, N) == 1:
            assert order(a, N) == order_brute(a, N)
print("  порядок через делители lambda(N) совпадает с прямым перебором (N = 15, 21, 33, 35, 91)")
# пример ЛР7: N = 21, a = 2
r = order(2, 21)
y = pow(2, r // 2, 21)
print(f"  N = 21, a = 2: r = {r}, a^(r/2) = {y}, gcd({y}-1, 21) = {gcd(y - 1, 21)}, gcd({y}+1, 21) = {gcd(y + 1, 21)}")
assert r == 6 and {gcd(y - 1, 21), gcd(y + 1, 21)} == {7, 3}
# N = 15: порядки всех оснований
orders15 = {a: order(a, 15) for a in range(1, 15) if gcd(a, 15) == 1}
print("  N = 15: порядки", orders15)
# доля хороших оснований
print("   N    множители       доля хороших a в Z_N^*   доля по всем a in [2, N-2] (с учётом gcd > 1)   1 - 2^(1-m)")
for N in (15, 21, 33, 35, 39, 55, 91, 105, 1155):
    fac = factorize(N)
    m = len(fac)
    units = [a for a in range(1, N) if gcd(a, N) == 1]
    good = sum(good_base(a, N) for a in units)
    allg = sum((gcd(a, N) > 1) or good_base(a, N) for a in range(2, N - 1))
    frac = good / len(units)
    print(f"  {N:4d}   {str(fac):15s}        {frac:.4f}                         {allg / (N - 3):.4f}                          {1 - 2.0 ** (1 - m):.4f}")
    assert frac >= 1 - 2.0 ** (1 - m) - 1e-12
assert sum(good_base(a, 15) for a in orders15) == 6
# минимальная доля по всем нечётным N < 3000 с m различными простыми, не степеням простого
mins = {}
for N in range(9, 3000, 2):
    fac = factorize(N)
    m = len(fac)
    if m < 2:
        continue
    units = [a for a in range(1, N) if gcd(a, N) == 1]
    frac = sum(good_base(a, N) for a in units) / len(units)
    assert frac >= 1 - 2.0 ** (1 - m) - 1e-12
    if m not in mins or frac < mins[m][0]:
        mins[m] = (frac, N)
for m, (fr, N) in sorted(mins.items()):
    print(f"  нечётные N < 3000 с m = {m} простыми: минимум доли = {fr:.4f} (N = {N}), оценка 1 - 2^(1-m) = {1 - 2.0 ** (1 - m):.4f}")
assert mins[2][0] == 0.5

# ---------------------------------------------------------------------------
section("2. Оператор модульного умножения U_a и его собственные векторы")


def U_mult(a, N, n):
    """|y> -> |a y mod N> при y < N, |y> -> |y> при y >= N (перестановочная матрица 2^n x 2^n)."""
    D = 2 ** n
    U = np.zeros((D, D))
    for y in range(D):
        U[(a * y) % N if y < N else y, y] = 1
    return U


for N, a in ((15, 7), (21, 2), (35, 3)):
    n = ceil(log2(N))
    U = U_mult(a, N, n)
    assert np.allclose(U @ U.T, np.eye(2 ** n))
    r = order(a, N)
    orbit = [pow(a, k, N) for k in range(r)]
    total = np.zeros(2 ** n, complex)
    for s in range(r):
        u = np.zeros(2 ** n, complex)
        for k, y in enumerate(orbit):
            u[y] = np.exp(-2j * pi * s * k / r) / np.sqrt(r)
        assert np.allclose(U @ u, np.exp(2j * pi * s / r) * u)
        total += u
    e1 = np.zeros(2 ** n); e1[1] = 1
    assert np.allclose(total / np.sqrt(r), e1)
    print(f"  N = {N}, a = {a}, n = {n}: U_a унитарна; |u_s> — собственные с e^(2 pi i s/r), r = {r}; sum_s |u_s>/sqrt(r) = |1>")

# ---------------------------------------------------------------------------
section("3. Поиск порядка через QPE: распределение исходов")


def order_finding_full(a, N, t):
    """Точная симуляция: sum_k |k>|a^k mod N>/sqrt(T), затем F^dag на первом регистре."""
    T = 2 ** t
    vals = np.array([pow(a, k, N) for k in range(T)])
    M = np.zeros((T, N), complex)
    M[np.arange(T), vals] = 1 / np.sqrt(T)
    M = np.fft.fft(M, axis=0) / np.sqrt(T)          # F^dag x = fft(x)/sqrt(T)
    return np.sum(np.abs(M) ** 2, axis=1)


@lru_cache(maxsize=None)
def order_finding_probs(r, t):
    """То же через замкнутую формулу: зависит только от r и t (результат кэшируется)."""
    T = 2 ** t
    m = np.arange(T)
    p = np.zeros(T)
    x = pi * m * r / T
    for k0 in range(r):
        L = (T - k0 + r - 1) // r
        num = np.sin(L * x) ** 2
        den = np.sin(x) ** 2
        small = np.isclose(den, 0, atol=1e-20)
        p += np.where(small, L ** 2, num / np.where(small, 1, den)) / T ** 2
    return p


for N, a, t in ((15, 7, 8), (21, 2, 10), (21, 2, 11), (35, 3, 12)):
    pf = order_finding_full(a, N, t)
    pc = order_finding_probs(order(a, N), t)
    assert np.allclose(pf, pc)
print("  замкнутая формула совпадает с полной симуляцией (N = 15, 21, 35)")
p15 = order_finding_probs(4, 8)
peaks = np.nonzero(p15 > 1e-12)[0]
print(f"  N = 15, a = 7 (r = 4), t = 8: исходы {list(peaks)} с вероятностями {[round(x, 4) for x in p15[peaks]]}")
assert list(peaks) == [0, 64, 128, 192] and np.allclose(p15[peaks], 0.25)
t = 10
p21 = order_finding_probs(6, t)
top = np.argsort(p21)[::-1][:12]
print(f"  N = 21, a = 2 (r = 6), t = {t}: крупнейшие исходы (m: P)  — пики около s*1024/6:")
print("    " + ", ".join(f"{m}: {p21[m]:.3f}" for m in sorted(top)))
near_idx = {int(np.floor(s * 1024 / 6)) for s in range(6)} | {int(np.ceil(s * 1024 / 6)) % 1024 for s in range(6)}
near = p21[list(near_idx)].sum()
assert near >= 4 / pi ** 2
print(f"    вероятность двух ближайших к s*1024/6 точек (сумма по s) = {near:.3f}")

# ---------------------------------------------------------------------------
section("4. Цепные дроби")


def cf_expand(num, den):
    a = []
    while den:
        q = num // den
        a.append(q)
        num, den = den, num - q * den
    return a


def convergents(cf):
    h0, h1 = 0, 1
    k0, k1 = 1, 0
    out = []
    for q in cf:
        h0, h1 = h1, q * h1 + h0
        k0, k1 = k1, q * k1 + k0
        out.append(Fraction(h1, k1))
    return out


# теорема: если |x - s/r| < 1/(2 r^2), то s/r — подходящая дробь x
for _ in range(3000):
    r = int(rng.integers(2, 2000))
    s = int(rng.integers(0, r))
    t = 2 * ceil(log2(2000)) + 1
    T = 2 ** t
    m = round(s / r * T)
    if abs(m / T - s / r) < 1 / (2 * r * r):
        assert Fraction(s, r) in convergents(cf_expand(m, T)) or s == 0
print("  если |m/2^t - s/r| < 1/(2r^2), то s/r (несократимая) — среди подходящих дробей m/2^t (3000 случайных проб)")
m_ex, t_ex = 853, 10
cf = cf_expand(m_ex, 2 ** t_ex)
cv = convergents(cf)
print(f"  пример: m/2^t = {m_ex}/1024 = {m_ex / 1024:.6f}: [a0; a1, ...] = {cf}")
print("    подходящие дроби:", ", ".join(str(c) for c in cv))
assert Fraction(5, 6) in cv


def recover_order(m, t, a, N):
    """Классическая постобработка: знаменатели подходящих дробей < N, проверка a^q = 1 (и малые кратные)."""
    if m == 0:
        return None
    for c in convergents(cf_expand(m, 2 ** t)):
        q = c.denominator
        if q >= N:
            break
        for mult in (1, 2, 3):
            if pow(a, q * mult, N) == 1:
                return q * mult
    return None


def success_prob(a, N, t, mults=(1,)):
    r = order(a, N)
    p = order_finding_probs(r, t)
    ok = 0.0
    for m in np.nonzero(p > 1e-14)[0]:
        if m == 0:
            continue
        # строго: только знаменатели подходящих дробей, без кратных
        for c in convergents(cf_expand(int(m), 2 ** t)):
            q = c.denominator
            if q >= N:
                break
            if any(q * k == r for k in mults):
                ok += p[m]
                break
    return ok


print("  вероятность восстановить r за один запуск (только подходящие дроби) и с проверкой кратных 2q, 3q:")
for N, a in ((15, 7), (21, 2), (35, 3), (91, 2)):
    n = ceil(log2(N))
    for t in (2 * n, 2 * n + 1):
        r = order(a, N)
        p1 = success_prob(a, N, t)
        p3 = success_prob(a, N, t, (1, 2, 3))
        from_phi = sum(1 for s in range(r) if gcd(s, r) == 1) / r
        print(f"    N = {N:3d}, a = {a}, r = {r:2d}, t = {t:2d}: P = {p1:.3f}  (с кратными {p3:.3f});  phi(r)/r = {from_phi:.3f};  среднее число запусков {1 / p1:.2f}")
        if t == 2 * n + 1:
            assert p1 >= 0.4 * from_phi

# ---------------------------------------------------------------------------
section("5. Полный конвейер Шора для малых N")


def shor(N, max_runs=50):
    """Возвращает (делитель, число квантовых запусков, число попыток a)."""
    if N % 2 == 0:
        return 2, 0, 0
    for b in range(2, int(log2(N)) + 1):                 # степень простого / полный квадрат
        root = round(N ** (1 / b))
        for c in (root - 1, root, root + 1):
            if c > 1 and c ** b == N:
                return c, 0, 0
    n = ceil(log2(N))
    t = 2 * n + 1
    runs, tries = 0, 0
    while runs < max_runs:
        a = int(rng.integers(2, N - 1))
        tries += 1
        g = gcd(a, N)
        if g > 1:
            return g, runs, tries                       # повезло классически
        r = order(a, N)
        p = order_finding_probs(r, t)
        rr = None
        for _ in range(4):                              # несколько запусков QPE для этого a
            m = int(np.searchsorted(np.cumsum(p), rng.random() * p.sum()))
            runs += 1
            cand = recover_order(m, t, a, N)
            rr = cand if rr is None or cand is None else (lcm(rr, cand) if pow(a, lcm(rr, cand), N) == 1 else rr)
            if rr is not None and rr == r:
                break
        if rr is None or pow(a, rr, N) != 1 or rr % 2:
            continue
        y = pow(a, rr // 2, N)
        if y == N - 1:
            continue
        d = gcd(y - 1, N)
        if 1 < d < N:
            return d, runs, tries
    return None, runs, tries


print("    N     найденный делитель   среднее число запусков QPE   среднее число оснований a   (по 200 прогонов)")
for N in (15, 21, 33, 35, 39, 51, 55, 57, 65, 77, 85, 91, 143, 221):
    stats = []
    for _ in range(200):
        d, runs, tries = shor(N)
        assert d is not None and N % d == 0 and 1 < d < N
        stats.append((runs, tries))
    st = np.mean(stats, axis=0)
    print(f"  {N:4d}       {d:3d} x {N // d:3d}              {st[0]:5.2f}                       {st[1]:5.2f}")

# ---------------------------------------------------------------------------
section("6. Дискретный логарифм как абелева задача о скрытой подгруппе")
p_mod, g = 23, 5
r = p_mod - 1
assert order(g, p_mod) == r                             # 5 — первообразный корень по модулю 23
x_secret = 13
h = pow(g, x_secret, p_mod)
# f(a, b) = g^a h^b mod p постоянна на смежных классах H = {(a,b): a + x b = 0 mod r}
F = np.array([[pow(g, a, p_mod) * pow(h, b, p_mod) % p_mod for b in range(r)] for a in range(r)])
Hsub = [(a, b) for a in range(r) for b in range(r) if (a + x_secret * b) % r == 0]
assert len(Hsub) == r and all(F[a, b] == 1 for a, b in Hsub)
# состояние sum_{a,b} |a,b>|f(a,b)>, QFT на Z_r x Z_r, распределение (c, d)
M = np.zeros((r, r, p_mod), complex)
for a in range(r):
    for b in range(r):
        M[a, b, F[a, b]] = 1 / r
M = np.fft.ifft(np.fft.ifft(M, axis=0), axis=1) * r     # QFT по Z_r на обоих регистрах (знак +)
P2 = np.sum(np.abs(M) ** 2, axis=2)
support = {(c, d) for c in range(r) for d in range(r) if P2[c, d] > 1e-12}
perp = {(c, d) for c in range(r) for d in range(r) if (d - x_secret * c) % r == 0}
assert support == perp and np.allclose(P2[P2 > 1e-12], 1 / r)
print(f"  p = {p_mod}, g = {g}, h = g^x = {h}: выход Фурье-выборки — равномерно по H^perp = {{(c, d): d = x c mod {r}}}, |H^perp| = {r}")
# восстановление x
found = None
for _ in range(20):
    idx = rng.choice(r * r, p=P2.ravel() / P2.sum())
    c, d = divmod(int(idx), r)
    if gcd(c, r) == 1:
        found = d * pow(c, -1, r) % r
        break
assert found == x_secret
inv_frac = sum(1 for c in range(r) if gcd(c, r) == 1) / r
print(f"  по выборке (c, d) с обратимым c: x = d c^(-1) mod r = {found}; P(c обратимо) = phi(r)/r = {inv_frac:.3f}")

# ---------------------------------------------------------------------------
section("7. «Скомпилированные» схемы: N = 15, a = 11")
N, a = 15, 11
r = order(a, N)
U = U_mult(a, N, 4)
# на орбите {1, 11}: 0001 <-> 1011 — это X на битах 1 и 3 (считая слева с 1)
X = np.array([[0, 1], [1, 0]]); I2 = np.eye(2)
XX = np.kron(np.kron(X, I2), np.kron(X, I2))
for y in (1, 11):
    assert np.allclose(U[:, y], XX[:, y])
print(f"  N = 15, a = 11: r = {r}; на орбите {{1, 11}} U_11 действует как X (x) I (x) X (x) I — два CNOT от вспомогательного кубита")
print("  U_11^2 = I на орбите, поэтому все C-U^(2^j) при j >= 1 тривиальны: «схема» знает ответ заранее")
p_full = order_finding_full(a, N, 8)
assert np.allclose(p_full[[0, 128]], 0.5)
print("  QPE с t = 8 даёт m = 0 или 128 (s/r = 0 или 1/2) с вероятностью 1/2 — то же получает и схема из двух CNOT")

# ---------------------------------------------------------------------------
section("8. Ресурсы: грубые оценки")
print("   n (бит)   Борегар: 2n+3 кубитов   контр. умножений (2n)   ~n^3 элементарных операций")
for n in (4, 5, 8, 1024, 2048):
    print(f"   {n:6d}         {2 * n + 3:6d}                {2 * n:6d}                 {float(n) ** 3:.2e}")
assert 2 * 2048 + 3 == 4099
# Гидни–Экерå (2019): 0.3 n^3 + 0.0005 n^3 lg n Toffoli, 3n + 0.002 n lg n логических кубитов
n = 2048
tof = 0.3 * n ** 3 + 0.0005 * n ** 3 * log2(n)
lq = 3 * n + 0.002 * n * log2(n)
print(f"  формулы Гидни–Экеро для n = 2048: Toffoli ~ {tof:.2e}, логических кубитов ~ {lq:.0f}")

# ---------------------------------------------------------------------------
section("9. Ответы к задачам семинара")
print(f"  1: N = 21, a = 2: r = 6, gcd(7, 21) = 7, gcd(9, 21) = 3")
g15 = [a for a in orders15 if good_base(a, 15)]
print(f"  2: N = 15: хорошие основания {g15} — доля {len(g15)}/8")
units21 = [a for a in range(1, 21) if gcd(a, 21) == 1]
g21 = [a for a in units21 if good_base(a, 21)]
print(f"  3: N = 21: хорошие {g21} из {len(units21)}; порядки {dict((a, order(a, 21)) for a in units21)}")
assert len(g21) * 2 == len(units21)
print(f"  4: 853/1024: подходящие дроби {', '.join(str(c) for c in cv)}; r = 6 (a = 2: 2^6 mod 21 = {pow(2, 6, 21)})")
# 7: p = 11, g = 2, h = 9
assert order(2, 11) == 10 and pow(2, 6, 11) == 9
print("  7: p = 11, g = 2, h = 9: x = 6 (2^6 = 64 = 9 mod 11); H = {(a, b): a + 6b = 0 mod 10}, H^perp = {(c, d): d = 6c mod 10}")
# 10: phi(6)/6
print(f"  10: r = 6: доля s, взаимно простых с r, = {sum(1 for s in range(6) if gcd(s, 6) == 1)}/6")
print("\nВсе проверки пройдены.")
