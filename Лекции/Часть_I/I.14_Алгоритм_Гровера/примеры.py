"""
Лекция I.14. Квантовый поиск и усиление амплитуды: алгоритм Гровера (геометрия
двух отражений, число итераций, «перекручивание»), неизвестное число решений
(BBHT), квантовый подсчёт, нижняя оценка (гибридный аргумент), усиление и оценка
амплитуды, квадратичное ускорение Монте-Карло, модель «точки безубыточности»
для квадратичного ускорения на отказоустойчивом компьютере, влияние шума.

Запуск:  python примеры.py
Требуется: numpy. Все проверки оформлены как assert.
Порядок кубитов: первый множитель — старший (как в N&C и np.kron).
"""
import numpy as np
from math import pi, sqrt, asin, sin, cos, floor, ceil, log2

rng = np.random.default_rng(2026)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (X + Z) / np.sqrt(2)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def kron(*ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def grover_state(N, marked, k):
    """Вектор состояния после k итераций (оракул — знак, диффузия 2|s><s| - I)."""
    psi = np.full(N, 1 / np.sqrt(N))
    mask = np.zeros(N, bool); mask[list(marked)] = True
    for _ in range(k):
        psi = np.where(mask, -psi, psi)
        psi = 2 * psi.mean() - psi
    return psi


def p_succ(N, M, k):
    th = asin(sqrt(M / N))
    return sin((2 * k + 1) * th) ** 2


def k_opt(N, M):
    return floor(pi / 4 * sqrt(N / M))


# ---------------------------------------------------------------------------
section("1. Схема: оракул и диффузия; D = H^n (2|0><0| - I) H^n")
n = 3
N = 2 ** n
Hn = kron(*[H] * n)
s = np.full(N, 1 / np.sqrt(N))
D = 2 * np.outer(s, s) - np.eye(N)
R0 = 2 * np.outer(np.eye(N)[0], np.eye(N)[0]) - np.eye(N)
assert np.allclose(D, Hn @ R0 @ Hn)
# 2|0><0| - I = -(X^n)(C^{n-1}Z)(X^n) с точностью до знака
CZn = np.diag([1] * (N - 1) + [-1])
Xn = kron(*[X] * n)
assert np.allclose(R0, -Xn @ CZn @ Xn)
print("  D = 2|s><s| - I = H^n (2|0><0| - I) H^n = -H^n X^n C^(n-1)Z X^n H^n  (n = 3)")
# действие D: инверсия относительно среднего
v = rng.normal(size=N)
assert np.allclose(D @ v, 2 * v.mean() - v)
print("  D a = 2 <a> - a: «инверсия относительно среднего»")
# фазовый оракул из битового на |->: проверено в I.11; здесь O = I - 2 sum |x><x|
marked = [5]
O = np.diag([-1 if x in marked else 1 for x in range(N)])
G = D @ O
psi = s.copy()
for k in range(3):
    psi = G @ psi
assert np.allclose(psi, grover_state(N, marked, 3))

# ---------------------------------------------------------------------------
section("2. Геометрия: поворот на 2 theta в плоскости span{|alpha>, |beta>}")
for N, M in ((16, 1), (64, 3), (1024, 7), (4096, 1)):
    marked = rng.choice(N, M, replace=False)
    th = asin(sqrt(M / N))
    for k in range(0, 3 * ceil(sqrt(N))):
        psi = grover_state(N, marked, k)
        amp_b = psi[marked]
        assert np.allclose(amp_b, sin((2 * k + 1) * th) / sqrt(M))
        if k > 40:
            break
    ko = k_opt(N, M)
    print(f"  N = {N:5d}, M = {M}: sin theta = sqrt(M/N), P_k = sin^2((2k+1)theta) подтверждено;  k_opt = floor(pi/4 sqrt(N/M)) = {ko:3d}, P = {p_succ(N, M, ko):.4f}")
    assert p_succ(N, M, ko) >= 1 - M / N
# два правила выбора k: k* = round(pi/(4 theta) - 1/2) гарантирует P >= 1 - M/N;
# k = floor(pi/4 sqrt(N/M)) (упрощённое правило) — почти всегда то же, но гарантия слабее
viol, worst = 0, {0.25: 1.0, 0.1: 1.0, 0.01: 1.0}
for N in range(4, 2049):
    for M in range(1, N // 4 + 1):
        th = asin(sqrt(M / N))
        ks = round(pi / (4 * th) - 0.5)
        assert sin((2 * ks + 1) * th) ** 2 >= 1 - M / N - 1e-12
        kf = k_opt(N, M)
        pf = p_succ(N, M, kf)
        viol += pf < 1 - M / N - 1e-12
        for lim in worst:
            if M / N <= lim:
                worst[lim] = min(worst[lim], pf)
print("  k* = round(pi/(4 theta) - 1/2): P >= 1 - M/N для всех N <= 2048, M <= N/4")
print(f"  k = floor(pi/4 sqrt(N/M)): P < 1 - M/N в {viol} случаях; минимум P при M/N <= 1/4: {worst[0.25]:.3f}, <= 0.1: {worst[0.1]:.3f}, <= 0.01: {worst[0.01]:.4f}")
assert worst[0.25] > 0.81 and worst[0.01] > 0.989
# N = 4, M = 1: одна итерация — точно
assert np.isclose(np.abs(grover_state(4, [2], 1)[2]) ** 2, 1)
print("  N = 4, M = 1: theta = 30 градусов, после одной итерации P = 1")
# M = N/4: одна итерация точно
for N in (8, 64, 1024):
    assert np.isclose(p_succ(N, N // 4, 1), 1)
print("  M = N/4 (любое N): одна итерация даёт P = 1")
# M >= N/2: Гровер не нужен; при M = N/2 итерации не помогают
print(f"  M = N/2: theta = 45 градусов, P_k = sin^2((2k+1) pi/4) = 1/2 для всех k: {[round(p_succ(64, 32, k), 3) for k in range(4)]}")

# ---------------------------------------------------------------------------
section("3. Зависимость от k и «перекручивание»")
N = 1024
print("   M   k_opt   P(k_opt)   P(2 k_opt)   P(3 k_opt)   период по k ~ pi/(2 theta)")
for M in (1, 2, 4):
    ko = k_opt(N, M)
    th = asin(sqrt(M / N))
    row = [p_succ(N, M, k) for k in (ko, 2 * ko, 3 * ko)]
    print(f"   {M}    {ko:3d}     {row[0]:.4f}     {row[1]:.4f}       {row[2]:.4f}        {pi / (2 * th):.1f}")
# N = 2^20
N = 2 ** 20
ko = k_opt(N, 1)
print(f"  N = 2^20, M = 1: k_opt = {ko}, P = {p_succ(N, 1, ko):.7f}, 1 - P = {1 - p_succ(N, 1, ko):.2e}")
assert ko == 804
# N = 64, M = 1: P(k)
N = 64
Pk = [p_succ(N, 1, k) for k in range(13)]
print("  N = 64, M = 1: P(k), k = 0..12:", " ".join(f"{p:.3f}" for p in Pk))
kmin = int(np.argmin(Pk[6:])) + 6
print(f"    максимум при k = {int(np.argmax(Pk))}, затем минимум при k = {kmin} (P = {Pk[kmin]:.4f})")
# точный Гровер: можно добиться P = 1, изменив фазы на последнем шаге
# (Лонг 2001): все итерации с фазами phi, k = ceil(pi/(4 theta) - 1/2)
def long_exact(N, M):
    th = asin(sqrt(M / N))
    J = ceil(pi / (4 * th) - 0.5)
    # фаза phi из условия sin(pi/(4J+2)) = sin(theta) sin(phi/2)
    phi = 2 * asin(sin(pi / (4 * J + 2)) / sin(th))
    psi = np.zeros(2, complex); psi[:] = [cos(th), sin(th)]       # (|alpha>, |beta>)
    a, b = cos(th), sin(th)
    Ophi = np.diag([1, np.exp(1j * phi)])
    sv = np.array([a, b])
    Dphi = (1 - np.exp(1j * phi)) * np.outer(sv, sv) - np.eye(2) * (-1)      # I - (1 - e^{i phi})|s><s|
    Dphi = np.eye(2) - (1 - np.exp(1j * phi)) * np.outer(sv, sv)
    for _ in range(J):
        psi = -Dphi @ (Ophi @ psi)
    return J, abs(psi[1]) ** 2
for N, M in ((64, 1), (100, 3), (1024, 5)):
    J, P = long_exact(N, M)
    assert np.isclose(P, 1)
    print(f"  точный поиск (фазы phi вместо pi, алгоритм Лонга): N = {N}, M = {M}: {J} итераций, P = {P:.12f}")

# ---------------------------------------------------------------------------
section("4. Неизвестное число решений: BBHT")


def avg_success(m, th):
    """Средняя вероятность при k, равномерном в {0..m-1}."""
    return np.mean([sin((2 * j + 1) * th) ** 2 for j in range(m)])


for N, M in ((1024, 1), (1024, 10), (4096, 3)):
    th = asin(sqrt(M / N))
    for m in (1, 5, 20, 50):
        formula = 0.5 - sin(4 * m * th) / (4 * m * sin(2 * th))
        assert np.isclose(avg_success(m, th), formula)
    m0 = ceil(1 / sin(2 * th))
    for m in range(m0, m0 + 50):
        assert avg_success(m, th) >= 0.25 - 1e-12
print("  среднее по k in {0..m-1}: P = 1/2 - sin(4 m theta)/(4 m sin 2 theta); при m >= 1/sin(2 theta) P >= 1/4")


def bbht(N, M, lam=6 / 5):
    """Число запросов до нахождения решения (каждая итерация — 1 запрос, проверка — 1 запрос)."""
    th = asin(sqrt(M / N))
    m, q = 1.0, 0
    while True:
        j = int(rng.integers(0, max(1, int(m))))
        q += j + 1                                     # j итераций + проверка найденного
        if rng.random() < sin((2 * j + 1) * th) ** 2:
            return q
        m = min(lam * m, sqrt(N))


print("     N/M       sqrt(N/M)   среднее число запросов BBHT   отношение")
xs, ys = [], []
for N, M in ((2 ** 10, 64), (2 ** 10, 16), (2 ** 12, 16), (2 ** 12, 4), (2 ** 14, 4), (2 ** 14, 1), (2 ** 16, 1), (2 ** 18, 1)):
    qs = [bbht(N, M) for _ in range(2000)]
    r = sqrt(N / M)
    xs.append(r); ys.append(np.mean(qs))
    print(f"  {N // M:8d}     {r:7.1f}           {np.mean(qs):8.1f}             {np.mean(qs) / r:.2f}")
slope = np.polyfit(np.log(xs), np.log(ys), 1)[0]
print(f"  наклон log(запросов) по log sqrt(N/M) = {slope:.3f} (теория 1)")
assert 0.9 < slope < 1.1 and max(np.array(ys) / np.array(xs)) < 4.5

# ---------------------------------------------------------------------------
section("5. Квантовый подсчёт: QPE для итерации Гровера")
N = 64
for M in (1, 3, 10):
    marked = rng.choice(N, M, replace=False)
    O = np.diag([-1.0 if x in marked else 1.0 for x in range(N)])
    s = np.full(N, 1 / np.sqrt(N))
    G = (2 * np.outer(s, s) - np.eye(N)) @ O
    ev = np.linalg.eigvals(G)
    th = asin(sqrt(M / N))
    for e in (np.exp(2j * th), np.exp(-2j * th)):
        assert np.min(np.abs(ev - e)) < 1e-9
    print(f"  N = {N}, M = {M:2d}: у G есть собственные значения e^(+-2i theta), theta = {th:.4f}")


def qpe_probs(phi, t):
    T = 2 ** t
    delta = phi - np.arange(T) / T
    num = np.sin(pi * T * delta) ** 2
    den = (T * np.sin(pi * delta)) ** 2
    exact = np.isclose(np.abs(((delta + 0.5) % 1) - 0.5), 0, atol=1e-13)
    return np.where(exact, 1.0, num / np.where(exact, 1.0, den))


def counting_dist(N, M, t):
    """|s> = (|g+> + |g->)/sqrt2 (с фазами) в плоскости; QPE даёт phi = theta/pi или 1 - theta/pi."""
    th = asin(sqrt(M / N))
    return 0.5 * qpe_probs(th / pi, t) + 0.5 * qpe_probs(1 - th / pi, t)


# проверка через полную матрицу G и QPE с собственным разложением
N, M, t = 64, 3, 7
marked = rng.choice(N, M, replace=False)
O = np.diag([-1.0 if x in marked else 1.0 for x in range(N)])
s = np.full(N, 1 / np.sqrt(N))
G = (2 * np.outer(s, s) - np.eye(N)) @ O
w, V = np.linalg.eig(G)
c = np.linalg.solve(V, s)
phis = (np.angle(w) / (2 * pi)) % 1
p_full = sum(abs(c[i]) ** 2 * qpe_probs(phis[i], t) for i in range(N) if abs(c[i]) > 1e-12)
assert np.allclose(p_full, counting_dist(N, M, t))
print(f"  распределение QPE для G (полная матрица, t = {t}) = 1/2 P_(theta/pi) + 1/2 P_(1-theta/pi)")
T = 2 ** t
m_best = int(np.argmax(counting_dist(N, M, t)))
M_est = N * sin(pi * m_best / T) ** 2
print(f"  N = 64, M = 3, t = 7: наиболее вероятный исход m = {m_best}, оценка M = N sin^2(pi m/T) = {M_est:.3f}")
# ошибка оценки: |dM| <= 2 pi sqrt(M N)/T + pi^2 N / T^2 с вероятностью >= 8/pi^2
for N, M, t in ((256, 5, 8), (1024, 30, 10), (4096, 100, 12)):
    T = 2 ** t
    p = counting_dist(N, M, t)
    Mest = N * np.sin(pi * np.arange(T) / T) ** 2
    bound = 2 * pi * sqrt(M * (N - M)) / T + pi ** 2 * N / T ** 2
    pin = p[np.abs(Mest - M) <= bound].sum()
    print(f"  N = {N}, M = {M}, t = {t}: P(|M_est - M| <= 2 pi sqrt(M(N-M))/T + pi^2 N/T^2 = {bound:.2f}) = {pin:.3f} >= 8/pi^2")
    assert pin >= 8 / pi ** 2

# ---------------------------------------------------------------------------
section("6. Нижняя оценка: гибридный аргумент (I.10) и оптимальность Гровера")
# для Гровера: sum_x ||psi_T^x - psi_T||^2 <= 4 T^2 и требуемая величина ~ 2N
print("     N     T = k_opt   sum_x ||psi^x - psi||^2   4T^2   отношение")
for N in (64, 256, 1024):
    T = k_opt(N, 1)
    psi0 = grover_state(N, [], T)
    tot = 0.0
    for x in range(N):
        tot += np.sum((grover_state(N, [x], T) - psi0) ** 2)
    print(f"  {N:5d}     {T:3d}          {tot:9.1f}          {4 * T * T:6d}   {tot / (4 * T * T):.3f}")
    assert tot <= 4 * T * T + 1e-9
# граница Залки: вероятность успеха после T запросов не больше sin^2((2T+1) arcsin(1/sqrt N))
print("  Гровер достигает максимальной вероятности sin^2((2T+1) arcsin(1/sqrt N)) — она оптимальна (Залка, 1999)")
# с p параллельными оракулами: Гровер на блоках N/p
N = 2 ** 20
for p in (1, 4, 16, 64):
    print(f"    p = {p:2d} параллельных оракулов: глубина ~ pi/4 sqrt(N/p) = {pi / 4 * sqrt(N / p):7.1f} — выигрыш лишь sqrt(p) = {sqrt(p):.0f}")

# ---------------------------------------------------------------------------
section("7. Усиление амплитуды: произвольный алгоритм A")
d = 32
A = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))[0]
good = np.zeros(d, bool); good[[3, 17]] = True
psi = A[:, 0]
a = np.sum(np.abs(psi[good]) ** 2)
tha = asin(sqrt(a))
S_chi = np.diag(np.where(good, -1.0, 1.0))
S0 = np.eye(d); S0[0, 0] = -1                      # I - 2|0><0|
Q = -A @ S0 @ A.conj().T @ S_chi
st = psi.copy()
for k in range(1, 6):
    st = Q @ st
    pk = np.sum(np.abs(st[good]) ** 2)
    assert np.isclose(pk, sin((2 * k + 1) * tha) ** 2)
print(f"  случайный A (d = 32), a = {a:.4f}: после k применений Q = -A S_0 A^dag S_chi P = sin^2((2k+1) theta_a) (k = 1..5)")
# a = 0.01
a = 0.01
tha = asin(sqrt(a))
k = round(pi / (4 * tha) - 0.5)
print(f"  a = 0.01: theta_a = {tha:.4f}, k = round(pi/(4 theta_a) - 1/2) = {k}, P = {sin((2 * k + 1) * tha) ** 2:.4f};"
      f" запусков A: {2 * k + 1} против ~1/a = 100 классических повторений")
assert k == 7

# ---------------------------------------------------------------------------
section("8. Оценка амплитуды и квадратичное ускорение Монте-Карло")


def ae_dist(a, t):
    """Каноническая оценка амплитуды: QPE для Q; собственные фазы +-theta_a/pi (в единицах оборота)."""
    th = asin(sqrt(a))
    return 0.5 * qpe_probs(th / pi, t) + 0.5 * qpe_probs(1 - th / pi, t)


a_true = 0.3
print(f"  a = {a_true}: ошибка оценки a против числа запросов Q (AE: Q = 2^t - 1; Монте-Карло: Q выборок)")
print("  80 %-полуширина AE усреднена по 40 случайным a in [0.05, 0.95] (у одного a она ступенчатая)")
print("     Q     80% AE (сред.)   RMSE AE (a = 0.3)   RMSE МК = sqrt(a(1-a)/Q)")
a_list = rng.uniform(0.05, 0.95, 40)
Qs, w_ae, r_ae, r_mc = [], [], [], []
for t in range(4, 13):
    T = 2 ** t
    Qn = T - 1
    est = np.sin(pi * np.arange(T) / T) ** 2
    widths = []
    for aa in list(a_list) + [a_true]:
        p = ae_dist(aa, t)
        err = est - aa
        # гарантия BHMT: |a_est - a| <= 2 pi sqrt(a(1-a))/T + pi^2/T^2 с вероятностью >= 8/pi^2
        bound = 2 * pi * sqrt(aa * (1 - aa)) / T + pi ** 2 / T ** 2
        assert p[np.abs(err) <= bound].sum() >= 8 / pi ** 2
        order = np.argsort(np.abs(err))
        widths.append(np.abs(err)[order][np.searchsorted(np.cumsum(p[order]), 0.8)])
    p = ae_dist(a_true, t)
    rmse = sqrt(np.sum(p * (est - a_true) ** 2))
    mc = sqrt(a_true * (1 - a_true) / Qn)
    Qs.append(Qn); w_ae.append(np.mean(widths[:-1])); r_ae.append(rmse); r_mc.append(mc)
    print(f"  {Qn:5d}     {w_ae[-1]:.2e}          {rmse:.2e}           {mc:.2e}")
s_w = np.polyfit(np.log(Qs), np.log(w_ae), 1)[0]
s_rmse = np.polyfit(np.log(Qs), np.log(r_ae), 1)[0]
print(f"  наклоны по log Q: 80%-полуширина AE {s_w:.2f}, RMSE одиночной AE {s_rmse:.2f} (тяжёлые хвосты), Монте-Карло -0.50")
print("  гарантия BHMT |a_est - a| <= 2 pi sqrt(a(1-a))/T + pi^2/T^2 с вероятностью >= 8/pi^2 выполнена для всех a и t")
assert s_w < -0.9
# медиана нескольких запусков AE подавляет хвосты: RMSE ~ 1/Q (усреднение по тем же 40 случайным a)
print("  медиана 7 запусков AE (общее число запросов 7(2^t - 1)), RMSE по 40 случайным a:")
Q7, r7 = [], []
for t in range(4, 12):
    T = 2 ** t
    est = np.sin(pi * np.arange(T) / T) ** 2
    sq = []
    for aa in a_list:
        p = ae_dist(aa, t)
        samp = rng.choice(T, size=(3000, 7), p=p / p.sum())
        sq.append(np.mean((np.median(est[samp], axis=1) - aa) ** 2))
    rm = sqrt(np.mean(sq))
    mc = sqrt(np.mean(a_list * (1 - a_list)) / (7 * (T - 1)))
    Q7.append(7 * (T - 1)); r7.append(rm)
    print(f"    Q = {7 * (T - 1):6d}: RMSE = {rm:.2e};  Монте-Карло с тем же Q: {mc:.2e}")
s7 = np.polyfit(np.log(Q7), np.log(r7), 1)[0]
print(f"  наклон RMSE медианы 7 запусков: {s7:.2f} (теория -1)")
assert s7 < -0.9
# пример: среднее функции f: {0..2^n-1} -> [0,1] через A|0>|0> = sum_x |x> (sqrt(1-f)|0> + sqrt(f)|1>)
n = 5
fvals = rng.random(2 ** n)
amp = np.zeros((2 ** n, 2))
amp[:, 0] = np.sqrt((1 - fvals) / 2 ** n)
amp[:, 1] = np.sqrt(fvals / 2 ** n)
assert np.isclose(np.sum(amp[:, 1] ** 2), fvals.mean())
print(f"  кодирование среднего: A|0>|0> = sum_x |x>(sqrt(1-f(x))|0> + sqrt(f(x))|1>)/sqrt(2^n): a = mean f = {fvals.mean():.4f}")

# ---------------------------------------------------------------------------
section("9. Квадратичное ускорение на отказоустойчивом компьютере: точка безубыточности")
# Модель: классический перебор — N t_c; Гровер — (pi/4) sqrt(N) t_q. Безубыточность при
# N* = (pi/4)^2 (t_q/t_c)^2, время T* = N* t_c.
print("  иллюстративные параметры (не данные конкретных устройств): T* = (pi/4)^2 t_q^2 / t_c")
print("     t_c (класс. проверка)    t_q (итерация Гровера)    N*          T* (время до безубыточности)")
for tc, tq in ((1e-9, 1e-6), (1e-9, 1e-3), (1e-12, 1e-3), (1e-12, 1e-2), (1e-12, 1e-1)):
    Ns = (pi / 4) ** 2 * (tq / tc) ** 2
    Ts = Ns * tc
    assert np.isclose((pi / 4) * sqrt(Ns) * tq, Ns * tc)
    print(f"      {tc:.0e} с                 {tq:.0e} с              {Ns:.1e}     {Ts:.1e} с = {Ts / 86400:.2e} сут = {Ts / 86400 / 365.25:.2e} лет")
# для сравнения: кубическое и четвертичное ускорение — классика N t_c, квантово N^(1/3) t_q, N^(1/4) t_q
tc, tq = 1e-9, 1e-3
for power, name in ((2, "квадратичное"), (3, "кубическое"), (4, "четвертичное")):
    Ns = (tq / tc) ** (power / (power - 1))
    print(f"  {name:12s} ускорение (квантово N^(1/{power}) t_q): N* = {Ns:.1e}, время {Ns * tc:.1e} с")

# ---------------------------------------------------------------------------
section("10. Шум: Гровер с глобальной деполяризацией")


def p_noisy(N, M, k, eps):
    """После каждой итерации с вероятностью eps состояние заменяется на I/N."""
    f = (1 - eps) ** k
    return f * p_succ(N, M, k) + (1 - f) * M / N


for n in (6, 10):
    N = 2 ** n
    ko = k_opt(N, 1)
    print(f"  N = 2^{n}: k_opt = {ko}")
    for eps in (0.0, 0.01, 0.05, 0.2):
        Pk = [p_noisy(N, 1, k, eps) for k in range(0, 2 * ko)]
        kb = int(np.argmax(Pk))
        # классический перебор с тем же числом запросов kb: (kb+1)/N (kb запросов + проверка-угадывание)
        cls = (kb + 1) / N
        print(f"    eps = {eps:4.2f}: лучшее k = {kb:3d}, P = {Pk[kb]:.4f}; классика с {kb} запросами: {cls:.4f}")
# проверка модели на векторе состояния (глобальная деполяризация линейна)
N = 64; eps = 0.03
rho = np.full((N, N), 1 / N)
marked = [11]
O = np.diag([-1.0 if x in marked else 1.0 for x in range(N)])
s = np.full(N, 1 / np.sqrt(N))
G = (2 * np.outer(s, s) - np.eye(N)) @ O
for k in range(1, 7):
    rho = (1 - eps) * G @ rho @ G.T + eps * np.eye(N) / N
    assert np.isclose(rho[11, 11], p_noisy(N, 1, k, eps))
print("  формула P = (1-eps)^k sin^2((2k+1)theta) + (1 - (1-eps)^k) M/N сверена с матрицей плотности (N = 64)")

# ---------------------------------------------------------------------------
section("11. Ответы к задачам семинара")
print(f"  3: N = 2^20, M = 1: k = {k_opt(2 ** 20, 1)}, P = {p_succ(2 ** 20, 1, k_opt(2 ** 20, 1)):.7f}")
N = 64
Pk = [p_succ(N, 1, k) for k in range(13)]
print(f"  4: N = 64: k_opt = {k_opt(64, 1)}, P(6) = {Pk[6]:.4f}; P(12) = {Pk[12]:.4f}")
th = asin(sqrt(3 / 64))
print(f"  7: N = 64, M = 3: theta = {th:.4f} рад, собственные фазы G: +-2 theta = +-{2 * th:.4f}; t = 7: m = {m_best}, M_est = {M_est:.3f}")
print(f"  8: a = 0.01: k = 7, P = {sin(15 * asin(0.1)) ** 2:.4f}")
eps_t = 1e-3
print(f"  9: eps = 1e-3, a ~ 0.3: МК ~ a(1-a)/eps^2 = {0.21 / eps_t ** 2:.1e} выборок; AE ~ 2 pi sqrt(a(1-a))/eps = {2 * pi * sqrt(0.21) / eps_t:.0f} запросов, отношение {0.21 / eps_t ** 2 / (2 * pi * sqrt(0.21) / eps_t):.0f}")
for tc, tq in ((1e-9, 1e-3), (1e-12, 1e-3)):
    Ns = (pi / 4) ** 2 * (tq / tc) ** 2
    print(f"  10: t_c = {tc:.0e} с, t_q = {tq:.0e} с: N* = {Ns:.2e}, T* = {Ns * tc:.2e} с = {Ns * tc / 60:.1f} мин = {Ns * tc / 86400:.2f} сут")
print("\nВсе проверки пройдены.")
