"""
Лекция I.10. Модели вычислений и квантовая сложность: амплификация,
суммирование по путям (BQP в PSPACE), целочисленный «разрыв» для {H, Toffoli}
(BQP в PP), метод полиномов (приближённая степень OR, чётность), гибридный
аргумент BBBV, MBQC на кластерной цепочке, адиабатический поиск,
распределение Портера–Томаса и линейный XEB.

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
Порядок кубитов: первый множитель — старший (как в N&C и np.kron).
"""
import numpy as np
from itertools import product
from math import comb, exp, log, ceil
from scipy.optimize import linprog
from scipy.stats import unitary_group

rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (X + Z) / np.sqrt(2)
T = np.diag([1, np.exp(1j * np.pi / 4)])


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def P(phi):
    return np.diag([1, np.exp(1j * phi)])


def apply1(psi, U, a, n):
    psi = psi.reshape([2] * n)
    psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [a])), 0, a)
    return psi.reshape(-1)


def apply2(psi, U, a, b, n):
    psi = psi.reshape([2] * n)
    psi = np.tensordot(U.reshape(2, 2, 2, 2), psi, axes=([2, 3], [a, b]))
    psi = np.moveaxis(psi, [0, 1], [a, b])
    return psi.reshape(-1)


def toffoli_perm(n, c1, c2, t):
    idx = np.arange(2 ** n)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))) & 1
    flip = bits[:, c1] & bits[:, c2]
    return idx ^ (flip << (n - 1 - t))


# ---------------------------------------------------------------------------
section("1. Амплификация: большинство из k запусков")
def maj_error(k, p=1 / 3):
    return sum(comb(k, j) * p ** j * (1 - p) ** (k - j) for j in range((k + 1) // 2, k + 1))
print("   k    точная ошибка большинства    оценка Хёфдинга exp(-k/18)")
for k in (1, 11, 51, 101, 201, 401):
    e = maj_error(k)
    print(f"  {k:4d}        {e:.3e}                 {exp(-k / 18):.3e}")
    assert e <= exp(-2 * k * (1 / 6) ** 2) + 1e-15
k_hoef = ceil(18 * log(1e9))
k_exact = next(k for k in range(1, 2000, 2) if maj_error(k) <= 1e-9)
print(f"  ошибка <= 1e-9: по Хёфдингу k >= {k_hoef}; точный минимум (нечётное k) = {k_exact}")

# ---------------------------------------------------------------------------
section("2. Суммирование по путям Фейнмана: BQP в PSPACE")
def random_circuit(n, m, gates=("H", "T", "CX", "TOF")):
    circ = []
    for _ in range(m):
        g = gates[rng.integers(len(gates))]
        q = list(rng.permutation(n)[:3])
        circ.append((g, q))
    return circ
def statevector(circ, n):
    psi = np.zeros(2 ** n, complex); psi[0] = 1
    for g, q in circ:
        if g == "H": psi = apply1(psi, H, q[0], n)
        elif g == "T": psi = apply1(psi, T, q[0], n)
        elif g == "CX": psi = psi[toffoli_perm(n, q[0], q[0], q[1])]
        elif g == "TOF": psi = psi[toffoli_perm(n, q[0], q[1], q[2])]
    return psi
def path_sum(circ, n, y):
    """<y|C|0>: рекурсивный обход путей — память O(n * длина), время O(2^h)."""
    def rec(i, x):
        if i == len(circ):
            return 1.0 + 0j if x == y else 0j
        g, q = circ[i]
        bit = lambda x, k: (x >> (n - 1 - k)) & 1
        if g == "H":
            b = bit(x, q[0]); x0 = x & ~(1 << (n - 1 - q[0]))
            return (rec(i + 1, x0) + (-1) ** b * rec(i + 1, x0 | (1 << (n - 1 - q[0])))) / np.sqrt(2)
        if g == "T":
            return (np.exp(1j * np.pi / 4) if bit(x, q[0]) else 1) * rec(i + 1, x)
        if g == "CX":
            return rec(i + 1, x ^ (bit(x, q[0]) << (n - 1 - q[1])))
        return rec(i + 1, x ^ ((bit(x, q[0]) & bit(x, q[1])) << (n - 1 - q[2])))
    return rec(0, 0)
for trial in range(5):
    n = 4
    circ = random_circuit(n, 24)
    psi = statevector(circ, n)
    h = sum(1 for g, _ in circ if g == "H")
    for y in range(2 ** n):
        assert np.isclose(path_sum(circ, n, y), psi[y])
print("  5 случайных схем {H, T, CNOT, Toffoli} на 4 кубитах: сумма по путям = вектор состояния для всех y")
print(f"  (последняя схема: {h} гейтов H -> {2 ** h} путей; память — только текущий путь)")

# ---------------------------------------------------------------------------
section("3. {H, Toffoli}: амплитуда = целое / 2^{h/2}; P_acc = (A - B) / 2^h (BQP в PP)")
def paths_HTof(circ, n):
    """Все пути: (конечная строка, знак)."""
    out = [(0, 1)]
    for g, q in circ:
        new = []
        for x, s in out:
            if g == "H":
                b = (x >> (n - 1 - q[0])) & 1
                x0 = x & ~(1 << (n - 1 - q[0]))
                new += [(x0, s), (x0 | (1 << (n - 1 - q[0])), s * (-1) ** b)]
            else:
                b = ((x >> (n - 1 - q[0])) & (x >> (n - 1 - q[1]))) & 1
                new.append((x ^ (b << (n - 1 - q[2])), s))
        out = new
    return out
for trial in range(5):
    n = 4
    circ = random_circuit(n, 16, gates=("H", "TOF"))
    h = sum(1 for g, _ in circ if g == "H")
    psi = statevector(circ, n)
    paths = paths_HTof(circ, n)
    gap = np.zeros(2 ** n, int)
    for x, s in paths:
        gap[x] += s
    assert np.allclose(psi, gap / 2 ** (h / 2))
    acc = [y for y in range(2 ** n) if y & 1]               # «принять» = последний бит 1
    # A = # упорядоченных пар путей с одинаковым концом в acc и одинаковым знаком, B — с разными знаками
    A2 = sum(1 for x, s in paths for x2, s2 in paths if x == x2 and x in acc and s == s2)
    B2 = sum(1 for x, s in paths for x2, s2 in paths if x == x2 and x in acc and s != s2)
    pacc = np.sum(np.abs(psi[acc]) ** 2)
    assert np.isclose(pacc, (A2 - B2) / 2 ** h)
print("  5 случайных схем {H, Toffoli}: psi(y) = gap(y)/2^{h/2}, gap — целое (разность числа путей со знаками ±)")
print(f"  P_acc = (#пар путей с равными знаками - #пар с разными)/2^h; последняя схема: h = {h}, "
      f"A = {A2}, B = {B2}, P_acc = {pacc:.4f}")

# ---------------------------------------------------------------------------
section("4. Метод полиномов: приближённая степень OR_N и чётность")
def adeg_or(N, eps=1 / 3):
    """Минимальная степень d многочлена q(k), k = 0..N: |q(0)| <= eps, |q(k) - 1| <= eps (k >= 1)."""
    ks = np.arange(N + 1)
    t = 2 * ks / N - 1                                  # отображение на [-1, 1], базис Чебышёва
    for d in range(0, N + 1):
        V = np.polynomial.chebyshev.chebvander(t, d)
        target = (ks >= 1).astype(float)
        A_ub = np.vstack([V, -V]); b_ub = np.concatenate([target + eps, -(target - eps)])
        res = linprog(np.zeros(d + 1), A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * (d + 1), method="highs")
        if res.status == 0:
            return d
rows = []
for N in (4, 16, 64, 256, 1024):
    d = adeg_or(N)
    rows.append((N, d))
    print(f"  N = {N:5d}: adeg(OR_N) = {d:3d};  d/sqrt(N) = {d / np.sqrt(N):.3f};  нижняя оценка числа запросов d/2 = {d / 2:.1f};"
          f"  Гровер ~ (pi/4) sqrt(N) = {np.pi / 4 * np.sqrt(N):.1f}")
ratios = [d / np.sqrt(N) for N, d in rows[1:]]
assert max(ratios) / min(ratios) < 1.6                  # d = Theta(sqrt N)
# чётность: точная степень N (многолинейная запись), поэтому T >= N/2; Дойч достигает N/2 при N = 2
for N in (2, 3, 4, 5):
    xs = np.array(list(product([0, 1], repeat=N)))
    f = (-1.0) ** xs.sum(axis=1)
    # коэффициенты Фурье–Уолша: только старший моном
    coef = {S_: np.mean(f * (-1.0) ** (xs[:, list(S_)].sum(axis=1))) for r in range(N + 1) for S_ in __import__("itertools").combinations(range(N), r)}
    nz = [S_ for S_, c in coef.items() if abs(c) > 1e-12]
    assert nz == [tuple(range(N))]
print("  PARITY_N: в базисе (-1)^{x_S} отличен от нуля только моном степени N => deg = N, запросов >= N/2")

# ---------------------------------------------------------------------------
section("5. Гибридный аргумент BBBV: sum_x ||psi_T^x - psi_T||^2 <= 4 T^2")
def run_alg(Us, oracle_x, N):
    psi = np.zeros(N, complex); psi[0] = 1
    psi = Us[0] @ psi
    for U in Us[1:]:
        if oracle_x is not None:
            psi[oracle_x] *= -1
        psi = U @ psi
    return psi
for n in (3, 4, 5):
    N = 2 ** n
    for Tq in (1, 2, 4, 8):
        for trial in range(3):
            Us = [unitary_group.rvs(N, random_state=rng) for _ in range(Tq + 1)]
            psi0 = run_alg(Us, None, N)
            D = sum(np.linalg.norm(run_alg(Us, x, N) - psi0) ** 2 for x in range(N))
            assert D <= 4 * Tq ** 2 + 1e-9
print("  случайные алгоритмы (U_0, O_x, U_1, ..., O_x, U_T), N = 8..32, T = 1..8: неравенство выполняется")
# Гровер: насколько он близок к границе
print("     N    T_Гровер   sum_x||.||^2   4T^2    успех    sqrt(N/6)")
for n in (4, 6, 8, 10):
    N = 2 ** n
    Tg = int(np.floor(np.pi / 4 * np.sqrt(N)))
    u = np.ones(N, complex) / np.sqrt(N)
    Hn = 2 * np.outer(u, u.conj()) - np.eye(N)      # диффузия (без H^n — для экономии берём её целиком)
    def grover(x):
        psi = u.copy()
        for _ in range(Tg):
            if x is not None:
                psi[x] *= -1
            psi = Hn @ psi
        return psi
    psi0 = grover(None)
    # по симметрии все x дают одно и то же расстояние
    d0 = np.linalg.norm(grover(0) - psi0) ** 2
    succ = abs(grover(0)[0]) ** 2
    print(f"  {N:5d}   {Tg:5d}      {N * d0:9.1f}    {4 * Tg ** 2:6d}   {succ:.4f}    {np.sqrt(N / 6):6.2f}")
    assert N * d0 <= 4 * Tg ** 2 + 1e-9 and Tg >= np.sqrt(N / 6) * (1 - 2.0 / np.sqrt(N) ** 0.5)

# ---------------------------------------------------------------------------
section("6. MBQC: однобитная телепортация и кластерная цепочка")
def plus_phi(phi, m):
    return np.array([1, (-1) ** m * np.exp(1j * phi)]) / np.sqrt(2)
CZ = np.diag([1, 1, 1, -1]).astype(complex)
def eq_phase(a, b):
    return np.isclose(abs(np.vdot(a, b)), np.linalg.norm(a) * np.linalg.norm(b))
for _ in range(100):
    psi = rng.normal(size=2) + 1j * rng.normal(size=2); psi /= np.linalg.norm(psi)
    phi = rng.uniform(0, 2 * np.pi)
    st = CZ @ np.kron(psi, np.array([1, 1]) / np.sqrt(2))
    for m in (0, 1):
        out = plus_phi(phi, m).conj() @ st.reshape(2, 2)        # проекция первого кубита
        assert np.isclose(np.linalg.norm(out) ** 2, 0.5)
        assert eq_phase(out, np.linalg.matrix_power(X, m) @ H @ P(-phi) @ psi)
print("  |psi>|+>, CZ, измерение 1-го кубита в базисе (|0> ± e^{i phi}|1>)/sqrt2, исход m:")
print("  на 2-м кубите X^m H P(-phi)|psi>, p(m) = 1/2 (100 случайных psi, phi)")
# цепочка из 3 кубитов: два измерения, второй угол адаптивен
n = 3
bad_ok = bad_tot = 0
for _ in range(100):
    psi = rng.normal(size=2) + 1j * rng.normal(size=2); psi /= np.linalg.norm(psi)
    a1, a2 = rng.uniform(0, 2 * np.pi, 2)
    st = np.kron(psi, np.ones(4) / 2)
    st = apply2(st, CZ, 0, 1, n); st = apply2(st, CZ, 1, 2, n)
    for m1 in (0, 1):
        s1 = np.tensordot(plus_phi(a1, m1).conj(), st.reshape(2, 4), axes=([0], [0]))
        s1 = s1 / np.linalg.norm(s1)
        phi2 = (-1) ** m1 * a2                                   # адаптация угла
        for m2 in (0, 1):
            out = plus_phi(phi2, m2).conj() @ s1.reshape(2, 2)
            target = np.linalg.matrix_power(X, m2) @ np.linalg.matrix_power(Z, m1) @ H @ P(-a2) @ H @ P(-a1) @ psi
            assert eq_phase(out, target)
            # без адаптации — неверно (при m1 = 1, общем a2)
            if m1 == 1:
                bad = plus_phi(a2, m2).conj() @ s1.reshape(2, 2)
                bad_tot += 1
                bad_ok += eq_phase(bad, target)
assert bad_ok <= 0.05 * bad_tot
print("  цепочка из 3 кубитов: углы (phi1, (-1)^{m1} phi2) дают X^{m2} Z^{m1} H P(-phi2) H P(-phi1)|psi>;")
print("  без адаптации второго угла результат неверен — MBQC нужна классическая обратная связь")

# ---------------------------------------------------------------------------
section("7. Адиабатический поиск: щель 1/sqrt(N), линейное и локальное расписания")
def gap(s, N):
    return np.sqrt(1 - 4 * (1 - 1 / N) * s * (1 - s))
for N in (4, 64, 1024):
    u = np.ones(N) / np.sqrt(N); m = np.zeros(N); m[0] = 1
    ss = np.linspace(0, 1, 201)
    gaps = []
    for s in ss[::20]:
        Hs = (1 - s) * (np.eye(N) - np.outer(u, u)) + s * (np.eye(N) - np.outer(m, m)) if N <= 64 else None
        if Hs is not None:
            ev = np.sort(np.linalg.eigvalsh(Hs))
            assert np.isclose(ev[1] - ev[0], gap(s, N))
    print(f"  N = {N:5d}: минимальная щель (s = 1/2) = {gap(0.5, N):.5f} = 1/sqrt(N) = {1 / np.sqrt(N):.5f}")
    assert np.isclose(gap(0.5, N), 1 / np.sqrt(N))
def evolve(N, Ttot, schedule, K=2000):
    """Динамика в 2-мерном подпространстве span{|m>, |m_perp>}: K шагов
    кусочно-постоянного гамильтониана (точная экспонента 2x2 на каждом шаге)."""
    a = 1 / np.sqrt(N); b = np.sqrt(1 - 1 / N)
    uu = np.array([a, b]); mm = np.array([1.0, 0.0])
    H0 = np.eye(2) - np.outer(uu, uu); H1 = np.eye(2) - np.outer(mm, mm)
    psi = uu.astype(complex)
    dt = Ttot / K
    for k in range(K):
        sk = schedule((k + 0.5) / K, N)
        w, V = np.linalg.eigh((1 - sk) * H0 + sk * H1)
        psi = V @ (np.exp(-1j * w * dt) * (V.T @ psi))
    return abs(psi[0]) ** 2
lin = lambda tau, N: tau
def rc(tau, N):
    """Роланд–Серф: ds/dt пропорционально g(s)^2; явная формула."""
    sq = np.sqrt(N - 1)
    return 0.5 - 0.5 * np.tan((1 - 2 * tau) * np.arctan(sq)) / sq
def time_needed(N, schedule, target=0.9):
    lo, hi = 0.1, 10.0
    while evolve(N, hi, schedule) < target:
        hi *= 2
    for _ in range(18):
        mid = np.sqrt(lo * hi)
        if evolve(N, mid, schedule) >= target: hi = mid
        else: lo = mid
    return hi
print("      N     T(линейное)   T/N     T(Роланд–Серф)   T/sqrt(N)")
tl, tr = [], []
for N in (16, 64, 256, 1024):
    a_, b_ = time_needed(N, lin), time_needed(N, rc)
    tl.append(a_); tr.append(b_)
    print(f"  {N:6d}   {a_:9.1f}    {a_ / N:6.3f}     {b_:9.1f}      {b_ / np.sqrt(N):6.3f}")
sl = np.polyfit(np.log([16, 64, 256, 1024]), np.log(tl), 1)[0]
sr = np.polyfit(np.log([16, 64, 256, 1024]), np.log(tr), 1)[0]
print(f"  показатели: линейное T ~ N^{sl:.2f}, локальное T ~ N^{sr:.2f} (теория 1 и 1/2) — для верности 0,9")
assert 0.85 < sl < 1.15 and 0.4 < sr < 0.6

# ---------------------------------------------------------------------------
section("8. Квантовое превосходство: Портер–Томас и линейный XEB")
n = 12; N = 2 ** n
ps = []
for _ in range(20):
    v = rng.normal(size=N) + 1j * rng.normal(size=N); v /= np.linalg.norm(v)
    ps.append(np.abs(v) ** 2)
ps = np.array(ps)
Np = (N * ps).ravel()
print(f"  хааровские состояния, n = {n}: среднее N p = {Np.mean():.4f}, доля N p > 1 = {np.mean(Np > 1):.4f} (e^-1 = {np.exp(-1):.4f}),"
      f" доля N p > 3 = {np.mean(Np > 3):.4f} (e^-3 = {np.exp(-3):.4f})")
assert abs(np.mean(Np > 1) - np.exp(-1)) < 0.01 and abs(np.mean(Np > 3) - np.exp(-3)) < 0.005
xeb_ideal = np.mean([N * np.sum(p ** 2) - 1 for p in ps])
print(f"  XEB идеального сэмплера = N sum p^2 - 1 = {xeb_ideal:.4f} (ожидание 2N/(N+1) - 1 = {2 * N / (N + 1) - 1:.4f})")
assert abs(xeb_ideal - 1) < 0.05
for F in (1.0, 0.5, 0.002):
    vals = []
    for p in ps[:5]:
        q = F * p + (1 - F) / N                          # деполяризованный сэмплер
        x = rng.choice(N, size=200000, p=q)
        vals.append(N * np.mean(p[x]) - 1)
    print(f"  деполяризованный сэмплер с верностью F = {F:5.3f}: оценка XEB по 2e5 выборкам = {np.mean(vals):.4f}")
    assert abs(np.mean(vals) - F) < 0.02
# случайные схемы: антиконцентрация с ростом глубины
def haar4():
    return unitary_group.rvs(4, random_state=rng)
n = 10; N = 2 ** n
print("  кирпичные схемы из хааровских двухкубитных гейтов, n = 10: N sum p^2 - 1 по глубине")
for depth in (1, 2, 4, 8, 16):
    vals = []
    for _ in range(10):
        psi = np.zeros(N, complex); psi[0] = 1
        for layer in range(depth):
            for q in range(layer % 2, n - 1, 2):
                psi = apply2(psi, haar4(), q, q + 1, n)
        vals.append(N * np.sum(np.abs(psi) ** 4) - 1)
    print(f"    глубина {depth:2d}: {np.mean(vals):8.3f}")
    last = np.mean(vals)
assert abs(last - 1) < 0.15
print("  при глубине ~ n распределение выходит на Портера–Томаса (значение 1): антиконцентрация")

# ---------------------------------------------------------------------------
section("9. Ответы к задачам семинара")
# 3: <0|H T H|0>
amp = (H @ T @ H)[0, 0]
assert np.isclose(amp, (1 + np.exp(1j * np.pi / 4)) / 2)
print(f"  3: <0|HTH|0> = (1 + e^(i pi/4))/2, |.|^2 = cos^2(pi/8) = {abs(amp) ** 2:.4f}")
# 5: adeg OR_2 = 1
assert adeg_or(2) == 1
print(f"  5: adeg(OR_2) = 1 (q(k) = 2k/3); adeg(OR_16) = {adeg_or(16)}")
# 6: граница из гибридного аргумента
for N in (10 ** 4, 10 ** 6):
    Tlow = np.sqrt((2 * N / 3 - 2 * np.sqrt(2 / 3) * np.sqrt(N)) / 4)
    print(f"  6: N = {N:.0e}: T >= sqrt((2N/3 - 2 sqrt(2/3) sqrt(N))/4) = {Tlow:.1f};  sqrt(N/6) = {np.sqrt(N / 6):.1f};"
          f"  Гровер (pi/4)sqrt(N) = {np.pi / 4 * np.sqrt(N):.1f}")
# 11: оценка верности схемы и числа выборок (иллюстративные параметры)
F_est = 0.995 ** 500 * 0.98 ** 50
M = 25 / F_est ** 2
print(f"  11: F ~ 0.995^500 * 0.98^50 = {F_est:.4f};  выборок для 5 sigma: M ~ 25/F^2 = {M:.0f}")
assert np.isclose(F_est, 0.0297, atol=5e-4)
# 1: числа уже выше
print(f"  1: k_Хёфдинг = {k_hoef}, k_точно = {k_exact}")

print("\nВсе проверки пройдены.")
