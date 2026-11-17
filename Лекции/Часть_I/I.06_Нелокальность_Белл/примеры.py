"""
Лекция I.6. Нелокальность: ЭПР, CHSH, граница Цирельсона, PR-ящик, GHZ,
контекстуальность (Кохен–Шпеккер, Перес–Мермин), нелокальность как ресурс.
Численные иллюстрации ко всем утверждениям лекции.

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
Порядок кубитов: первый множитель — старший (как в N&C и np.kron).
"""
import numpy as np
from itertools import product
from scipy.optimize import minimize, brentq

rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = [X, Y, Z]


def dag(A):
    return A.conj().T


def proj(v):
    return np.outer(v, v.conj())


def kron(*ops):
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def obs(theta):
    """Наблюдаемая ±1 в плоскости x–z: cos(theta) Z + sin(theta) X."""
    return np.cos(theta) * Z + np.sin(theta) * X


def obs3(n):
    """Наблюдаемая n·sigma для единичного вектора n."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    return sum(c * P for c, P in zip(n, PAULI))


def ev(rho, A):
    return np.real(np.trace(rho @ A))


def chsh_operator(A0, A1, B0, B1):
    return kron(A0, B0) + kron(A0, B1) + kron(A1, B0) - kron(A1, B1)


def h2(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def werner(p):
    return p * proj(Phi_p) + (1 - p) * np.eye(4) / 4


def corr_matrix(rho):
    return np.array([[ev(rho, kron(Pi, Pj)) for Pj in PAULI] for Pi in PAULI])


def horodecki_smax(rho):
    """Критерий Городецких: S_max = 2 sqrt(t1^2 + t2^2), t^2 — собств. числа T^T T."""
    T = corr_matrix(rho)
    w = np.sort(np.linalg.eigvalsh(T.T @ T))[::-1]
    return 2 * np.sqrt(w[0] + w[1])


def random_reflection(d):
    """Случайная наблюдаемая ±1 (эрмитова, A^2 = I) размерности d."""
    M = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(M)
    k = rng.integers(1, d)
    return Q @ np.diag([1] * k + [-1] * (d - k)) @ dag(Q)


ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)
Phi_p = (kron(ket0, ket0) + kron(ket1, ket1)) / np.sqrt(2)
Psi_m = (kron(ket0, ket1) - kron(ket1, ket0)) / np.sqrt(2)
rho_Phi = proj(Phi_p)

# ---------------------------------------------------------------------------
section("1. ЭПР–Бом: совершенные корреляции синглета в любом общем базисе")
rho_s = proj(Psi_m)
for _ in range(100):
    n = rng.normal(size=3)
    assert np.isclose(ev(rho_s, kron(obs3(n), obs3(n))), -1)
for _ in range(100):
    a, b = rng.normal(size=3), rng.normal(size=3)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    assert np.isclose(ev(rho_s, kron(obs3(a), obs3(b))), -a @ b)
    assert np.isclose(ev(rho_Phi, kron(obs(a[0]), obs(b[0]))), np.cos(a[0] - b[0]))
print("  синглет: <(n.s)(n.s)> = -1 для любого n; E(a,b) = -a.b")
print("  Phi+, наблюдаемые в плоскости x-z: E(alpha,beta) = cos(alpha - beta)")

# ---------------------------------------------------------------------------
section("2. Локальные модели: 16 детерминированных стратегий, |S| <= 2")
vals = []
for a0, a1, b0, b1 in product([1, -1], repeat=4):
    s = a0 * b0 + a0 * b1 + a1 * b0 - a1 * b1
    assert s == a0 * (b0 + b1) + a1 * (b0 - b1)
    vals.append(s)
print(f"  значения S на детерминированных стратегиях: {sorted(set(vals))}")
assert max(map(abs, vals)) == 2
# случайная модель со скрытым параметром: выпуклая смесь
for _ in range(200):
    w = rng.dirichlet(np.ones(16))
    assert abs(w @ np.array(vals)) <= 2 + 1e-12
print("  любая смесь (модель с lambda) даёт |S| <= 2")

# ---------------------------------------------------------------------------
section("3. Квантовое нарушение CHSH и игра CHSH")
a_opt, b_opt = (0, np.pi / 2), (np.pi / 4, -np.pi / 4)
Bop = chsh_operator(obs(a_opt[0]), obs(a_opt[1]), obs(b_opt[0]), obs(b_opt[1]))
S_q = ev(rho_Phi, Bop)
print(f"  S(Phi+, оптимальные углы) = {S_q:.6f};  2 sqrt 2 = {2 * np.sqrt(2):.6f}")
assert np.isclose(S_q, 2 * np.sqrt(2))
# игра CHSH: выигрыш, если a XOR b = x AND y; p_win = 1/2 + S/8
p_cl = max(np.mean([((a[x] ^ b[y]) == (x & y)) for x, y in product([0, 1], repeat=2)])
           for a in product([0, 1], repeat=2) for b in product([0, 1], repeat=2))
print(f"  игра CHSH: классический максимум {p_cl}; квантовый 1/2 + sqrt2/4 = {0.5 + np.sqrt(2) / 4:.6f}"
      f" = cos^2(pi/8) = {np.cos(np.pi / 8) ** 2:.6f}")
assert np.isclose(p_cl, 0.75) and np.isclose(0.5 + S_q / 8, np.cos(np.pi / 8) ** 2)
# вероятность выигрыша напрямую из проекторов
pw = 0
for x, y in product([0, 1], repeat=2):
    Ax, By = obs(a_opt[x]), obs(b_opt[y])
    for a, b in product([0, 1], repeat=2):
        if (a ^ b) == (x & y):
            Pa = (I2 + (-1) ** a * Ax) / 2
            Pb = (I2 + (-1) ** b * By) / 2
            pw += ev(rho_Phi, kron(Pa, Pb)) / 4
assert np.isclose(pw, np.cos(np.pi / 8) ** 2)
print(f"  прямой подсчёт по проекторам: p_win = {pw:.6f}")

# ---------------------------------------------------------------------------
section("4. Граница Цирельсона: SOS-разложение и тождество Ландау")
r2 = np.sqrt(2)
for d in [2, 3, 4, 6]:
    worst = 0
    for _ in range(300):
        A0, A1, B0, B1 = (random_reflection(d) for _ in range(4))
        Bd = chsh_operator(A0, A1, B0, B1)
        Id = np.eye(d * d)
        C = kron(A0, np.eye(d)) - kron(np.eye(d), B0 + B1) / r2
        D = kron(A1, np.eye(d)) - kron(np.eye(d), B0 - B1) / r2
        assert np.allclose(2 * r2 * Id - Bd, (C @ C + D @ D) / r2)          # SOS
        comm = lambda P, Q: P @ Q - Q @ P
        assert np.allclose(Bd @ Bd, 4 * Id - kron(comm(A0, A1), comm(B0, B1)))  # Ландау
        worst = max(worst, np.max(np.abs(np.linalg.eigvalsh(Bd))))
    assert worst <= 2 * r2 + 1e-9
    print(f"  d = {d}: SOS и B^2 = 4 - [A0,A1](x)[B0,B1] выполнены; max ||B|| на 300 наборах = {worst:.4f}")
# оптимизация по углам и состояниям кубитов: максимум = 2 sqrt 2
def neg_smax(t):
    A0, A1, B0, B1 = (obs3([np.sin(t[2 * i]) * np.cos(t[2 * i + 1]),
                            np.sin(t[2 * i]) * np.sin(t[2 * i + 1]), np.cos(t[2 * i])]) for i in range(4))
    return -np.max(np.linalg.eigvalsh(chsh_operator(A0, A1, B0, B1)))
best = min(minimize(neg_smax, rng.uniform(0, np.pi, 8), method="Nelder-Mead",
                    options=dict(maxiter=4000, xatol=1e-10, fatol=1e-12)).fun for _ in range(10))
print(f"  численный максимум по кубитным наблюдаемым и состояниям: {-best:.6f}")
assert np.isclose(-best, 2 * r2, atol=1e-6)
# равенство только при антикоммутирующих наблюдаемых: при угле theta между A0 и A1
# ||[A0,A1]|| = 2 sin theta, и из тождества Ландау max ||B|| = 2 sqrt(1 + sin theta)
def best_for_A(theta):
    A0, A1 = obs(0), obs(theta)
    f = lambda t: -np.max(np.linalg.eigvalsh(chsh_operator(A0, A1, obs(t[0]), obs(t[1]))))
    return -min(minimize(f, rng.uniform(-np.pi, np.pi, 2), method="Nelder-Mead",
                         options=dict(xatol=1e-10, fatol=1e-12)).fun for _ in range(8))
for th in [0.0, 0.5, 1.0, np.pi / 2]:
    assert np.isclose(best_for_A(th), 2 * np.sqrt(1 + np.sin(th)), atol=1e-5)
    assert np.isclose(np.linalg.norm(obs(0) @ obs(th) - obs(th) @ obs(0), 2), 2 * np.sin(th))
print("  при угле theta между A0 и A1 максимум равен 2 sqrt(1 + sin theta): 2 sqrt2 только при theta = pi/2")

# ---------------------------------------------------------------------------
section("5. Критерий Городецких; состояния Вернера; теорема Гизина")
assert np.isclose(horodecki_smax(rho_Phi), 2 * r2)
for p in [0, 0.3, 0.5, 1 / r2, 0.8, 1]:
    assert np.isclose(horodecki_smax(werner(p)), 2 * r2 * p)
print("  Вернер: S_max = 2 sqrt2 p, нарушение CHSH при p > 1/sqrt2 = 0.7071")
for th in np.linspace(0.05, np.pi / 4, 6):
    psi = np.cos(th) * kron(ket0, ket0) + np.sin(th) * kron(ket1, ket1)
    assert np.isclose(horodecki_smax(proj(psi)), 2 * np.sqrt(1 + np.sin(2 * th) ** 2))
print("  cos t|00> + sin t|11>: S_max = 2 sqrt(1 + sin^2 2t) > 2 при любом t in (0, pi/2)")
# проверим критерий Городецких прямой оптимизацией для случайного смешанного состояния
G = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
rho_r = G @ dag(G); rho_r /= np.trace(rho_r)
rho_r = 0.4 * rho_r + 0.6 * rho_Phi
def neg_S_rho(t):
    ns = [[np.sin(t[2 * i]) * np.cos(t[2 * i + 1]), np.sin(t[2 * i]) * np.sin(t[2 * i + 1]), np.cos(t[2 * i])]
          for i in range(4)]
    return -ev(rho_r, chsh_operator(*[obs3(n) for n in ns]))
s_num = -min(minimize(neg_S_rho, rng.uniform(0, np.pi, 8), method="Nelder-Mead",
                      options=dict(maxiter=6000, xatol=1e-10, fatol=1e-12)).fun for _ in range(12))
print(f"  случайное состояние: прямая оптимизация {s_num:.6f}, формула Городецких {horodecki_smax(rho_r):.6f}")
assert np.isclose(s_num, horodecki_smax(rho_r), atol=1e-5)

# ---------------------------------------------------------------------------
section("6. Отсутствие сигнала; PR-ящик; L ⊂ Q ⊂ NS")
def quantum_box(rho, As, Bs):
    P = np.zeros((2, 2, 2, 2))   # P[a,b,x,y]
    for x, y, a, b in product([0, 1], repeat=4):
        Pa = (I2 + (-1) ** a * As[x]) / 2
        Pb = (I2 + (-1) ** b * Bs[y]) / 2
        P[a, b, x, y] = ev(rho, kron(Pa, Pb))
    return P
def S_of_box(P):
    E = lambda x, y: sum((-1) ** (a + b) * P[a, b, x, y] for a, b in product([0, 1], repeat=2))
    return E(0, 0) + E(0, 1) + E(1, 0) - E(1, 1)
def no_signalling(P):
    pa = P.sum(axis=1)      # [a, x, y]
    pb = P.sum(axis=0)      # [b, x, y]
    return np.allclose(pa[:, :, 0], pa[:, :, 1]) and np.allclose(pb[:, 0, :], pb[:, 1, :])
PQ = quantum_box(rho_Phi, [obs(t) for t in a_opt], [obs(t) for t in b_opt])
assert no_signalling(PQ) and np.isclose(S_of_box(PQ), 2 * r2)
PR = np.zeros((2, 2, 2, 2))
for a, b, x, y in product([0, 1], repeat=4):
    PR[a, b, x, y] = 0.5 * ((a ^ b) == (x & y))
assert no_signalling(PR) and np.isclose(S_of_box(PR), 4)
assert np.allclose(PR.sum(axis=1), 0.5)
print("  квантовый ящик: S = 2 sqrt2, сигналов нет;  PR-ящик: S = 4, маргиналы 1/2, сигналов нет")
# случайные квантовые ящики (произвольные состояния и наблюдаемые) не сигнализируют
for _ in range(100):
    G = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)); rr = G @ dag(G); rr /= np.trace(rr)
    Pq = quantum_box(rr, [obs3(rng.normal(size=3)) for _ in range(2)], [obs3(rng.normal(size=3)) for _ in range(2)])
    assert no_signalling(Pq)
print("  100 случайных квантовых ящиков: условие отсутствия сигнала выполнено")
# квантовый оптимум — это смесь PR-ящика с белым шумом при v = 1/sqrt2
assert np.allclose(PQ, PR / r2 + (1 - 1 / r2) / 4)
print("  оптимальный квантовый ящик = (1/sqrt2) PR + (1 - 1/sqrt2) * (белый шум): S = 4v = 2 sqrt2")

# ---------------------------------------------------------------------------
section("7. GHZ: теорема Белла без неравенств; неравенство Мермина")
GHZ = (kron(ket0, ket0, ket0) + kron(ket1, ket1, ket1)) / np.sqrt(2)
rG = proj(GHZ)
xxx, xyy, yxy, yyx = (ev(rG, kron(*ops)) for ops in [(X, X, X), (X, Y, Y), (Y, X, Y), (Y, Y, X)])
print(f"  <XXX> = {xxx:+.0f}, <XYY> = {xyy:+.0f}, <YXY> = {yxy:+.0f}, <YYX> = {yyx:+.0f}")
assert np.allclose([xxx, xyy, yxy, yyx], [1, -1, -1, -1])
for k, ops in enumerate([(X, X, X), (X, Y, Y), (Y, X, Y), (Y, Y, X)]):
    assert np.allclose(kron(*ops) @ GHZ, (1 if k == 0 else -1) * GHZ)   # GHZ — общий собственный вектор
# локальные значения: x_i, y_i = ±1;  (x1 y2 y3)(y1 x2 y3)(y1 y2 x3) = x1 x2 x3
M_loc = max(abs(x1 * x2 * x3 - x1 * y2 * y3 - y1 * x2 * y3 - y1 * y2 * x3)
            for x1, x2, x3, y1, y2, y3 in product([1, -1], repeat=6))
M_q = xxx - xyy - yxy - yyx
print(f"  неравенство Мермина: локальный максимум {M_loc}, GHZ даёт {M_q:.0f}")
assert M_loc == 2 and np.isclose(M_q, 4)
assert all(x1 * x2 * x3 == (x1 * y2 * y3) * (y1 * x2 * y3) * (y1 * y2 * x3)
           for x1, x2, x3, y1, y2, y3 in product([1, -1], repeat=6))

# ---------------------------------------------------------------------------
section("8. Контекстуальность: магический квадрат Переса–Мермина")
sq = [[kron(X, I2), kron(I2, X), kron(X, X)],
      [kron(I2, Z), kron(Z, I2), kron(Z, Z)],
      [-kron(X, Z), -kron(Z, X), kron(Y, Y)]]
I4 = np.eye(4)
for i in range(3):
    row = [sq[i][j] for j in range(3)]
    col = [sq[j][i] for j in range(3)]
    for grp in (row, col):
        for P, Q in product(grp, repeat=2):
            assert np.allclose(P @ Q, Q @ P)            # совместимы
    assert np.allclose(row[0] @ row[1] @ row[2], I4)    # строки: +I
    assert np.allclose(col[0] @ col[1] @ col[2], -I4)   # столбцы: -I
for i, j in product(range(3), repeat=2):
    assert np.allclose(sq[i][j] @ sq[i][j], I4) and np.allclose(sq[i][j], sq[i][j].T)
print("  в каждой строке и каждом столбце наблюдаемые коммутируют;")
print("  произведение по строке = +I, по столбцу = -I  ->  неконтекстное приписывание значений невозможно")
# неконтекстных приписываний нет: перебор 2^9
ok = 0
for v in product([1, -1], repeat=9):
    m = np.array(v).reshape(3, 3)
    if all(np.prod(m[i]) == 1 for i in range(3)) and all(np.prod(m[:, j]) == -1 for j in range(3)):
        ok += 1
assert ok == 0
print("  перебор 2^9 таблиц ±1: ни одна не удовлетворяет всем шести условиям")
# игра: Алисе строка r, Бобу столбец c; классический максимум 8/9
rows_even = [t for t in product([1, -1], repeat=3) if np.prod(t) == 1]
cols_odd = [t for t in product([1, -1], repeat=3) if np.prod(t) == -1]
best_cl = 0
for A in product(rows_even, repeat=3):
    for B in product(cols_odd, repeat=3):
        w = sum(A[r][c] == B[c][r] for r, c in product(range(3), repeat=2))
        best_cl = max(best_cl, w)
print(f"  игра «магический квадрат»: классический максимум {best_cl}/9")
assert best_cl == 8
# квантовая стратегия: две пары Phi+ (кубиты A1 A2 B1 B2); Алиса и Боб меряют одни и те же наблюдаемые
psi2 = np.kron(Phi_p, Phi_p).reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(16)   # A1 A2 B1 B2
r4 = proj(psi2)
def eig_projectors(ops, signs_list):
    """Совместные проекторы коммутирующих ±1-наблюдаемых на исходы signs."""
    out = {}
    for s in signs_list:
        P = np.eye(4)
        for O, si in zip(ops, s):
            P = P @ (np.eye(4) + si * O) / 2
        out[s] = P
    return out
p_win = 0
for r, c in product(range(3), repeat=2):
    PA = eig_projectors(sq[r], rows_even)
    PB = eig_projectors([sq[j][c] for j in range(3)], cols_odd)
    assert np.allclose(sum(PA.values()), I4) and np.allclose(sum(PB.values()), I4)
    for sa, Pa in PA.items():
        for sb, Pb in PB.items():
            if sa[c] == sb[r]:
                p_win += ev(r4, kron(Pa, Pb)) / 9
print(f"  квантовая стратегия на двух парах Phi+: p_win = {p_win:.6f}")
assert np.isclose(p_win, 1)

# ---------------------------------------------------------------------------
section("9. Лазейка детектирования: порог эффективности для CHSH")
def S_eta(eta):
    # не сработавший детектор -> исход +1; для Phi+ маргиналы нулевые
    return eta ** 2 * 2 * r2 + (1 - eta) ** 2 * 2
eta_c = brentq(lambda e: S_eta(e) - 2, 0.5, 1)
print(f"  S(eta) = 2 sqrt2 eta^2 + 2 (1-eta)^2;  порог eta* = {eta_c:.4f};  2/(1+sqrt2) = {2 / (1 + r2):.4f}")
assert np.isclose(eta_c, 2 / (1 + r2))
# проверим S(eta) прямым построением «ящика с потерями»
def lossy_box(P, eta):
    Q = np.zeros_like(P)
    for x, y in product([0, 1], repeat=2):
        pa = P[:, :, x, y].sum(axis=1); pb = P[:, :, x, y].sum(axis=0)
        for a, b in product([0, 1], repeat=2):
            Q[a, b, x, y] += eta ** 2 * P[a, b, x, y]
            Q[a, 0, x, y] += eta * (1 - eta) * pa[a] * (b == 0)
            Q[0, b, x, y] += (1 - eta) * eta * pb[b] * (a == 0)
        Q[0, 0, x, y] += (1 - eta) ** 2
    return Q
for eta in [0.7, 0.8, eta_c, 0.9, 1.0]:
    assert np.isclose(S_of_box(lossy_box(PQ, eta)), S_eta(eta))
print("  проверено прямым построением распределения с потерями")

# ---------------------------------------------------------------------------
section("10. Нелокальность как ресурс: DI-случайность и DIQKD")
def hmin_DI(S):
    """Пиронио и др. 2010: H_min >= 1 - log2(1 + sqrt(2 - S^2/4))."""
    return 1 - np.log2(1 + np.sqrt(max(0.0, 2 - S ** 2 / 4)))
def r_DIQKD(S, Q):
    """Асин и др. 2007 (коллективные атаки): r >= 1 - h(Q) - h((1 + sqrt((S/2)^2 - 1))/2)."""
    return 1 - h2(Q) - h2((1 + np.sqrt(max(0.0, (S / 2) ** 2 - 1))) / 2)
assert np.isclose(hmin_DI(2), 0) and np.isclose(hmin_DI(2 * r2), 1)
assert np.isclose(r_DIQKD(2 * r2, 0), 1)
for S in [2.0, 2.2, 2.4, 2.6, 2 * r2]:
    print(f"  S = {S:.3f}: H_min >= {hmin_DI(S):.3f} бит на раунд")
# Вернер: S = 2 sqrt2 p, ошибка ключа Q = (1-p)/2
p_star = brentq(lambda p: r_DIQKD(2 * r2 * p, (1 - p) / 2), 1 / r2 + 1e-6, 0.999)
print(f"  DIQKD на состояниях Вернера: r > 0 при p > {p_star:.4f}  (Q < {(1 - p_star) / 2:.4f}, S > {2 * r2 * p_star:.4f})")
assert 0.85 < p_star < 0.87

print("\nВсе проверки пройдены.")
