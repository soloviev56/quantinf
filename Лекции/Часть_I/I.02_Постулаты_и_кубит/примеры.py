"""
Лекция I.2. Постулаты квантовой механики; кубит.
Численные иллюстрации ко всем утверждениям лекции.

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
Конвенция: hbar = 1.
"""
import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (X + Z) / np.sqrt(2)
PAULI = [X, Y, Z]
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)


def dag(A):
    return A.conj().T


def ndot_sigma(n):
    return sum(n[i] * PAULI[i] for i in range(3))


def bloch_state(theta, phi):
    """cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>."""
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])


def bloch_vector(psi):
    """r_j = <psi|sigma_j|psi>."""
    return np.array([np.vdot(psi, S @ psi).real for S in PAULI])


def random_state(d=2):
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    return v / np.linalg.norm(v)


def R(n, theta):
    n = np.asarray(n, float) / np.linalg.norm(n)
    return expm(-1j * theta * ndot_sigma(n) / 2)


def rodrigues(n, theta):
    """Матрица поворота SO(3) на угол theta вокруг единичной оси n (правый винт)."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def section(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


# ---------------------------------------------------------------------------
section("1. Глобальная фаза ненаблюдаема, относительная — наблюдаема")
psi = random_state()
alpha = 1.234
for S in PAULI:
    assert np.isclose(np.vdot(psi, S @ psi), np.vdot(np.exp(1j * alpha) * psi, S @ (np.exp(1j * alpha) * psi)))
plus = (ket0 + ket1) / np.sqrt(2)
minus = (ket0 - ket1) / np.sqrt(2)
print("|+> и |->: P(0) в базисе Z =", abs(plus[0]) ** 2, abs(minus[0]) ** 2,
      "; <X> =", np.vdot(plus, X @ plus).real, np.vdot(minus, X @ minus).real)

# ---------------------------------------------------------------------------
section("2. Сфера Блоха: r = (sin th cos ph, sin th sin ph, cos th)")
for _ in range(1000):
    th, ph = rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
    r = bloch_vector(bloch_state(th, ph))
    assert np.allclose(r, [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
print("формула проверена на 1000 состояниях; |r| = 1 для чистых")
named = {"|0>": ket0, "|1>": ket1, "|+>": plus, "|->": minus,
         "|+i>": (ket0 + 1j * ket1) / np.sqrt(2), "|-i>": (ket0 - 1j * ket1) / np.sqrt(2)}
for k, v in named.items():
    print(f"  {k:5s} -> r = {np.round(bloch_vector(v), 12)}")
# собственный вектор H: theta = pi/4, phi = 0
v = bloch_state(np.pi / 4, 0)
assert np.allclose(H @ v, v)
print("cos(pi/8)|0> + sin(pi/8)|1>: r =", np.round(bloch_vector(v), 6), "— собственный вектор H (+1)")

# ---------------------------------------------------------------------------
section("3. |<psi|phi>|^2 = (1 + n.m)/2; ортогональные состояния — антиподы")
for _ in range(1000):
    a, b = random_state(), random_state()
    assert np.isclose(abs(np.vdot(a, b)) ** 2, (1 + bloch_vector(a) @ bloch_vector(b)) / 2)
print("проверено на 1000 парах")

# ---------------------------------------------------------------------------
section("4. U = R_n(theta) поворачивает вектор Блоха на theta вокруг n")
for _ in range(500):
    n = rng.normal(size=3)
    th = rng.uniform(0, 2 * np.pi)
    psi = random_state()
    assert np.allclose(bloch_vector(R(n, th) @ psi), rodrigues(n, th) @ bloch_vector(psi))
# сопряжение: R_z X R_z^dag = cos th X + sin th Y
th = 0.7
assert np.allclose(R([0, 0, 1], th) @ X @ dag(R([0, 0, 1], th)), np.cos(th) * X + np.sin(th) * Y)
# ядро гомоморфизма SU(2) -> SO(3): R(2pi) = -I действует тождественно
assert np.allclose(R([0, 0, 1], 2 * np.pi), -I2) and np.allclose(rodrigues([0, 0, 1], 2 * np.pi), np.eye(3))
print("проверено на 500 случайных (n, theta, psi); R(2pi) = -I, но поворот сферы тождественный")

# ---------------------------------------------------------------------------
section("5. Ларморова прецессия H = (w/2) Z из |+>:  <X> = cos wt, <Y> = sin wt")
w = 2.0
for t in np.linspace(0, 5, 11):
    psi_t = expm(-1j * (w / 2) * Z * t) @ plus
    r = bloch_vector(psi_t)
    assert np.allclose(r, [np.cos(w * t), np.sin(w * t), 0])
print("совпадает при 11 моментах времени")

section("6. Осцилляции Раби: H = (D/2) Z + (W/2) X из |0>")
for Om, De in [(1.0, 0.0), (1.0, 0.5), (1.0, 2.0)]:
    Hr = De / 2 * Z + Om / 2 * X
    Omg = np.hypot(Om, De)
    for t in np.linspace(0, 10, 21):
        p1 = abs((expm(-1j * Hr * t) @ ket0)[1]) ** 2
        assert np.isclose(p1, Om ** 2 / Omg ** 2 * np.sin(Omg * t / 2) ** 2)
    print(f"  Omega={Om}, Delta={De}: амплитуда P1 = {Om**2/Omg**2:.3f}, частота {Omg:.3f} — формула верна")

# ---------------------------------------------------------------------------
section("7. Правило Борна, измерение наблюдаемой n.sigma: p(+-) = (1 +- n.r)/2")
for _ in range(500):
    psi = random_state()
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    Pp = (I2 + ndot_sigma(n)) / 2
    assert np.isclose(np.vdot(psi, Pp @ psi).real, (1 + n @ bloch_vector(psi)) / 2)
print("проверено на 500 случайных (psi, n)")

# Измерение в произвольном базисе = U^dag, затем измерение в базисе Z
th, ph = 1.1, 0.4
U = np.column_stack([bloch_state(th, ph), bloch_state(np.pi - th, ph + np.pi)])  # столбцы: |psi>, |psi_perp>
assert np.allclose(dag(U) @ U, I2)
psi = random_state()
p_direct = [abs(np.vdot(U[:, k], psi)) ** 2 for k in range(2)]
p_circuit = np.abs(dag(U) @ psi) ** 2
assert np.allclose(p_direct, p_circuit)
print("измерение в базисе {|psi>,|psi_perp>} == U^dag + измерение Z:", np.round(p_direct, 6))

# Повторяемость (правило Людерса): повторное измерение даёт тот же исход
P0 = np.outer(ket0, ket0)
post = P0 @ psi / np.linalg.norm(P0 @ psi)
assert np.isclose(abs(np.vdot(post, P0 @ post)), 1)
print("после исхода 0 повторное измерение Z даёт 0 с вероятностью 1")

# ---------------------------------------------------------------------------
section("8. Статистика: оценка <Z> по N измерениям, ошибка sqrt((1-<Z>^2)/N)")
psi = bloch_state(1.0, 0.3)
pz0 = abs(psi[0]) ** 2
Zexp = 2 * pz0 - 1
for N in [100, 1000, 10000]:
    est = [2 * rng.binomial(N, pz0) / N - 1 for _ in range(4000)]
    print(f"  N={N:6d}: эмпир. ст. откл. = {np.std(est):.4f}, теория = {np.sqrt((1 - Zexp**2) / N):.4f}")
    assert abs(np.std(est) / np.sqrt((1 - Zexp ** 2) / N) - 1) < 0.05

# ---------------------------------------------------------------------------
section("9. Соотношение неопределённостей Робертсона: dA dB >= |<[A,B]>|/2")


def var(A, psi):
    m = np.vdot(psi, A @ psi).real
    return np.vdot(psi, A @ A @ psi).real - m ** 2


for d in [2, 3, 5]:
    for _ in range(500):
        G1 = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        G2 = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        A, B = (G1 + dag(G1)) / 2, (G2 + dag(G2)) / 2
        psi = random_state(d)
        lhs = np.sqrt(var(A, psi) * var(B, psi))
        rhs = abs(np.vdot(psi, (A @ B - B @ A) @ psi)) / 2
        assert lhs >= rhs - 1e-12
print("проверено на 1500 случайных (A, B, psi) в размерностях 2, 3, 5")
# кубит: dX dY >= |<Z>|, равенство при r_x r_y = 0
psi = bloch_state(0.8, 0.0)  # r_y = 0
r = bloch_vector(psi)
print("  состояние с r_y=0: dX dY =", np.sqrt(var(X, psi) * var(Y, psi)), " |<Z>| =", abs(r[2]))
assert np.isclose(np.sqrt(var(X, psi) * var(Y, psi)), abs(r[2]))

# Энтропийное соотношение Маассена–Уффинка для кубита: H(Z) + H(X) >= 1 бит


def h2(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


vals = []
for th in np.linspace(0, 2 * np.pi, 20001):  # чистые состояния в плоскости xz
    rx, rz = np.sin(th), np.cos(th)
    vals.append(h2((1 + rz) / 2) + h2((1 + rx) / 2))
print("  min_{psi} [H(Z) + H(X)] =", round(min(vals), 6), "бит (граница 1, достигается на собственных состояниях)")
assert min(vals) >= 1 - 1e-9

# ---------------------------------------------------------------------------
section("10. «Квантовая монета»: интерференция амплитуд")
print("  H|0>: P(0) =", round(abs((H @ ket0)[0]) ** 2, 12))
print("  H H|0>: P(0) =", round(abs((H @ H @ ket0)[0]) ** 2, 12), " (классическая монета дала бы 1/2)")
# H, измерение, H — интерференция уничтожена измерением
p = abs((H @ ket0)) ** 2
p_after = sum(p[k] * abs((H @ (ket0 if k == 0 else ket1))[0]) ** 2 for k in range(2))
print("  H, измерение, H: P(0) =", p_after)
assert np.isclose(p_after, 0.5)

# ---------------------------------------------------------------------------
section("11. Неортогональные состояния: лучшая проективная стратегия для |0> и |+>")
best = 0
for a in np.linspace(0, 2 * np.pi, 100001):
    m = np.array([np.sin(a), 0, np.cos(a)])  # ось измерения в плоскости xz
    p_succ = 0.5 * ((1 + m @ [0, 0, 1]) / 2 + (1 - m @ [1, 0, 0]) / 2)  # «+m» -> |0>, «-m» -> |+>
    best = max(best, p_succ)
print("  max P_succ =", round(best, 6), "; ожидается (1 + 1/sqrt2)/2 =", round((1 + 1 / np.sqrt(2)) / 2, 6))
assert np.isclose(best, (1 + 1 / np.sqrt(2)) / 2, atol=1e-8)

# Лемма: если <psi|P|psi> = 1, то <phi|P|phi> >= |<psi|phi>|^2
for _ in range(500):
    psi, phi = random_state(4), random_state(4)
    # случайный проектор, содержащий psi
    extra = random_state(4)
    extra -= np.vdot(psi, extra) * psi
    extra /= np.linalg.norm(extra)
    P = np.outer(psi, psi.conj()) + (rng.random() < 0.5) * np.outer(extra, extra.conj())
    assert np.vdot(phi, P @ phi).real >= abs(np.vdot(psi, phi)) ** 2 - 1e-12
print("  лемма о неразличимости проверена на 500 примерах")

# ---------------------------------------------------------------------------
section("12. Составные системы: |Phi+> не раскладывается в произведение")
bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
prod = np.kron(random_state(), random_state())
print("  ранг матрицы коэффициентов: |Phi+> ->", np.linalg.matrix_rank(bell.reshape(2, 2)),
      "; произведение ->", np.linalg.matrix_rank(prod.reshape(2, 2), tol=1e-10))
for n in [1, 2, 3, 10, 50]:
    print(f"  {n:3d} кубит(ов): размерность 2^n = {2**n:.3e}, вещественных параметров 2^(n+1)-2 = {2**(n+1)-2:.3e}")

print("\nВсе проверки пройдены.")
