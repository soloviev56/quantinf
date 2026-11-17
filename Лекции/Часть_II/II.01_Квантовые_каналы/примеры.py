"""
Лекция II.1. Квантовые каналы: положительность и полная положительность
(транспонирование), представление Крауса, матрица Чоя (соглашение курса:
ненормированная, вход — первый множитель), изометрия Стайнспринга и
дополнительный канал, унитарная свобода, гейзенбергова картина, унитальные
каналы и каналы Паули (аффинное действие на шар Блоха, тетраэдр
Фудзивары–Алгоэта), каналы, разрушающие запутанность, сжатие расстояний,
верность процесса и средняя верность, ромбовидная норма.

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
Порядок множителей: первый — старший (как в np.kron); в матрице Чоя вход — первый.
"""
import numpy as np
from math import pi, sqrt, sin, cos
from functools import reduce
from scipy.linalg import sqrtm
from scipy.optimize import minimize

rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [I2, X, Y, Z]


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def dag(A):
    return A.conj().T


def ket(i, d):
    v = np.zeros(d, complex); v[i] = 1
    return v


def ptrace(R, dims, keep):
    """Частичный след по всем множителям, кроме keep (список индексов)."""
    n = len(dims)
    R = R.reshape(dims + dims)
    idx = list(range(n))
    trace_out = [i for i in idx if i not in keep]
    for k, i in enumerate(sorted(trace_out, reverse=True)):
        R = np.trace(R, axis1=i, axis2=i + R.ndim // 2)
    dk = int(np.prod([dims[i] for i in keep]))
    return R.reshape(dk, dk)


def apply_kraus(K, rho):
    return sum(k @ rho @ dag(k) for k in K)


def choi(channel, d):
    """J = sum_ij |i><j| (x) Phi(|i><j|)  (вход — первый множитель)."""
    return sum(np.kron(np.outer(ket(i, d), ket(j, d)), channel(np.outer(ket(i, d), ket(j, d))))
               for i in range(d) for j in range(d))


def choi_kraus(K):
    d = K[0].shape[1]
    return choi(lambda r: apply_kraus(K, r), d)


def from_choi(J, X_in, d_in, d_out):
    """Phi(X) = Tr_1[(X^T (x) I) J]."""
    return ptrace(np.kron(X_in.T, np.eye(d_out)) @ J, [d_in, d_out], [1])


def kraus_from_choi(J, d_in, d_out, tol=1e-12):
    w, V = np.linalg.eigh(J)
    K = []
    for lam, v in zip(w, V.T):
        if lam > tol:
            K.append(sqrt(lam) * v.reshape(d_in, d_out).T)
    return K


def is_cptp(K, tol=1e-10):
    return np.allclose(sum(dag(k) @ k for k in K), np.eye(K[0].shape[1]), atol=tol)


def random_channel(d_in, d_out, r):
    """Случайный канал через случайную изометрию C^d_in -> C^d_out (x) C^r."""
    G = rng.normal(size=(d_out * r, d_in)) + 1j * rng.normal(size=(d_out * r, d_in))
    V, _ = np.linalg.qr(G)
    V = V.reshape(d_out, r, d_in)
    return [V[:, k, :] for k in range(r)]


def random_state(d, rank=None):
    rank = rank or d
    G = rng.normal(size=(d, rank)) + 1j * rng.normal(size=(d, rank))
    R = G @ dag(G)
    return R / np.trace(R).real


def tnorm(A):
    return np.sum(np.linalg.svd(A, compute_uv=False))


def fidelity(r, s):
    sr = sqrtm(r)
    return np.real(np.trace(sqrtm(sr @ s @ sr))) ** 2


def bloch(rho):
    return np.real([np.trace(rho @ P) for P in (X, Y, Z)])


# стандартные каналы кубита (параметризация курса, Обозначения.md, §12)
def K_dephasing(p):
    return [sqrt(1 - p) * I2, sqrt(p) * Z]


def K_depol(p):
    return [sqrt(1 - 3 * p / 4) * I2] + [sqrt(p / 4) * P for P in (X, Y, Z)]


def K_ad(g):
    return [np.array([[1, 0], [0, sqrt(1 - g)]], complex), np.array([[0, sqrt(g)], [0, 0]], complex)]


Phi_plus = (np.kron(ket(0, 2), ket(0, 2)) + np.kron(ket(1, 2), ket(1, 2))) / sqrt(2)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], complex)

# ---------------------------------------------------------------------------
section("1. Положительность против полной положительности: транспонирование")
T = lambda r: r.T
for _ in range(100):
    r = random_state(2)
    assert np.linalg.eigvalsh(T(r)).min() > -1e-12
print("  транспонирование переводит матрицы плотности в матрицы плотности (положительно)")
JT = choi(T, 2)
assert np.allclose(JT, SWAP)
print(f"  J(T) = SWAP, собственные значения {np.round(np.linalg.eigvalsh(JT), 6)} — есть -1: не вполне положительно")
out = choi(T, 2) / 2       # (id (x) T)(|Phi+><Phi+|)
print(f"  (id (x) T)(|Phi+><Phi+|) = SWAP/2: собственные значения {np.round(np.linalg.eigvalsh(out), 3)}")
assert np.isclose(np.linalg.eigvalsh(out).min(), -0.5)
# универсальное NOT r -> -r: положительно, но не вполне положительно
UNOT = lambda r: np.trace(r) * I2 - r      # (I - rho) = 1/2 (I - r.sigma) для Tr rho = 1
for _ in range(50):
    r = random_state(2)
    assert np.allclose(bloch(UNOT(r)), -bloch(r))
ev = np.linalg.eigvalsh(choi(UNOT, 2))
print(f"  «универсальное NOT» r -> -r: положительно; собственные значения Чоя {np.round(ev, 6)}")
assert ev.min() < -0.5

# ---------------------------------------------------------------------------
section("2. Представление Крауса; гейзенбергова картина")
K = random_channel(2, 2, 3)
assert is_cptp(K)
for _ in range(20):
    r = random_state(2)
    out = apply_kraus(K, r)
    assert np.isclose(np.trace(out), 1) and np.linalg.eigvalsh(out).min() > -1e-12
    O = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)); O = O + dag(O)
    adj = sum(dag(k) @ O @ k for k in K)
    assert np.isclose(np.trace(out @ O), np.trace(r @ adj))
print("  случайный канал с 3 операторами Крауса: sum K^dag K = I, выход — матрица плотности")
print("  Tr(Phi(rho) O) = Tr(rho Phi^dag(O)),  Phi^dag(O) = sum K^dag O K")
assert np.allclose(sum(dag(k) @ k for k in K), sum(dag(k) @ I2 @ k for k in K))
print("  Phi^dag(I) = I  <=>  Phi сохраняет след; Phi унитален <=> Phi^dag сохраняет след")
# частичный след, измерение с забыванием результата, приготовление — тоже каналы
Ktr = [np.kron(I2, ket(j, 2)[None, :]) for j in range(2)]      # Tr_2: C^4 -> C^2
R = random_state(4)
assert np.allclose(apply_kraus(Ktr, R), ptrace(R, [2, 2], [0]))
print("  частичный след Tr_2 = канал с операторами Крауса I (x) <j|")

# ---------------------------------------------------------------------------
section("3. Матрица Чоя: CP <=> J >= 0, TP <=> Tr_2 J = I, восстановление, Краус из Чоя")
for d_in, d_out, r in ((2, 2, 3), (2, 3, 2), (3, 2, 4)):
    K = random_channel(d_in, d_out, r)
    J = choi_kraus(K)
    assert np.linalg.eigvalsh(J).min() > -1e-12
    assert np.allclose(ptrace(J, [d_in, d_out], [0]), np.eye(d_in))
    assert np.isclose(np.trace(J).real, d_in)
    # J = (id (x) Phi)(d |Phi_d><Phi_d|)
    phid = sum(np.kron(ket(i, d_in), ket(i, d_in)) for i in range(d_in))
    big = np.kron(np.eye(d_in), np.eye(d_out))
    Jalt = sum(np.kron(np.outer(ket(i, d_in), ket(j, d_in)), apply_kraus(K, np.outer(ket(i, d_in), ket(j, d_in)))) for i in range(d_in) for j in range(d_in))
    assert np.allclose(J, Jalt)
    for _ in range(5):
        Xin = rng.normal(size=(d_in, d_in)) + 1j * rng.normal(size=(d_in, d_in))
        assert np.allclose(from_choi(J, Xin, d_in, d_out), apply_kraus(K, Xin))
    K2 = kraus_from_choi(J, d_in, d_out)
    assert is_cptp(K2) and np.allclose(choi_kraus(K2), J)
    assert len(K2) == np.linalg.matrix_rank(J, tol=1e-10) <= min(r, d_in * d_out)
    print(f"  d_in = {d_in}, d_out = {d_out}, {r} операторов Крауса: J >= 0, Tr_2 J = I, Tr J = d_in, Phi(X) = Tr_1[(X^T (x) I) J]; Краус из Чоя: {len(K2)} операторов = ранг J")
print("  ранги Чоя (минимальное число операторов Крауса):")
ranks = {}
for name, Kk in (("тождественный", [I2]), ("унитарный H", [(X + Z) / sqrt(2)]), ("дефазировка p = 0.3", K_dephasing(0.3)),
                 ("амплитудное затухание g = 0.4", K_ad(0.4)), ("деполяризующий p = 0.5", K_depol(0.5)),
                 ("полностью деполяризующий p = 1", K_depol(1.0))):
    rk = np.linalg.matrix_rank(choi_kraus(Kk), tol=1e-10)
    ranks[name] = rk
    print(f"    {name:32s}: {rk}")
assert ranks["деполяризующий p = 0.5"] == 4 and ranks["амплитудное затухание g = 0.4"] == 2
# Чой полностью деполяризующего = I (x) I/2 ... в нашей нормировке: J = I_4/2 * 2/2 = I/2? проверка
assert np.allclose(choi_kraus(K_depol(1.0)), np.eye(4) / 2)
print("  полностью деполяризующий канал: J = I (x) I/2")

# ---------------------------------------------------------------------------
section("4. Стайнспринг и дополнительный канал")
K = random_channel(2, 2, 3)
V = sum(np.kron(k, ket(j, 3)[:, None]) for j, k in enumerate(K))    # V = sum K_j (x) |j>: C^2 -> C^2 (x) C^3
assert np.allclose(dag(V) @ V, np.eye(2))
r = random_state(2)
assert np.allclose(ptrace(V @ r @ dag(V), [2, 3], [0]), apply_kraus(K, r))
print("  V = sum_j K_j (x) |j>_E — изометрия, Phi(rho) = Tr_E(V rho V^dag)")
# унитарное расширение: U (|psi> (x) |0>_E) = V |psi>
Q, _ = np.linalg.qr(np.c_[V, rng.normal(size=(6, 4)) + 1j * rng.normal(size=(6, 4))])
U = np.zeros((6, 6), complex)
cols_e0 = [0, 3]
others = [1, 2, 4, 5]
U[:, cols_e0] = V
U[:, others] = Q[:, 2:]
assert np.allclose(dag(U) @ U, np.eye(6))
rho_se = np.kron(r, np.outer(ket(0, 3), ket(0, 3)))
assert np.allclose(ptrace(U @ rho_se @ dag(U), [2, 3], [0]), apply_kraus(K, r))
print("  дополнение V до унитарного U на S (x) E: Phi(rho) = Tr_E[U (rho (x) |0><0|_E) U^dag]")


def complementary(K, rho):
    r = len(K)
    return np.array([[np.trace(K[a] @ rho @ dag(K[b])) for b in range(r)] for a in range(r)])


for g in (0.1, 0.4, 0.75):
    for _ in range(5):
        r = random_state(2)
        Vg = sum(np.kron(k, ket(j, 2)[:, None]) for j, k in enumerate(K_ad(g)))
        comp = ptrace(Vg @ r @ dag(Vg), [2, 2], [1])
        assert np.allclose(comp, complementary(K_ad(g), r))
        assert np.allclose(comp, apply_kraus(K_ad(1 - g), r))
print("  дополнительный канал к амплитудному затуханию с параметром g — амплитудное затухание с 1 - g (g = 0.1, 0.4, 0.75)")
# дополнительный к унитарному — постоянный (окружение ничего не узнаёт); к полностью деполяризующему — «всё знает»
r1, r2 = random_state(2), random_state(2)
assert np.allclose(complementary([(X + Z) / sqrt(2)], r1), complementary([(X + Z) / sqrt(2)], r2))
print("  дополнительный к унитарному каналу — постоянный: окружение не получает информации")
c1, c2 = complementary(K_depol(1.0), r1), complementary(K_depol(1.0), r2)
print(f"  дополнительный к полностью деполяризующему: D(rho1, rho2) = {0.5 * tnorm(r1 - r2):.4f}, D на выходе дополнительного = {0.5 * tnorm(c1 - c2):.4f} (сохраняется)")
assert np.isclose(0.5 * tnorm(r1 - r2), 0.5 * tnorm(c1 - c2))

# ---------------------------------------------------------------------------
section("5. Унитарная свобода в представлении Крауса")
K = random_channel(2, 2, 3)
J = choi_kraus(K)
for m in (3, 5):
    Kp = K + [np.zeros((2, 2), complex)] * (m - 3)
    G = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    u, _ = np.linalg.qr(G)
    L = [sum(u[i, j] * Kp[j] for j in range(m)) for i in range(m)]
    assert np.allclose(choi_kraus(L), J)
print("  L_i = sum_j u_ij K_j (u унитарна, K дополнены нулями до 5): матрица Чоя не меняется")
# обратно: два набора одного канала связаны унитарной матрицей
K2 = kraus_from_choi(J, 2, 2)
Kv = np.array([k.reshape(-1) for k in K])          # строки — векторизованные K_j
K2v = np.array([k.reshape(-1) for k in K2])
u_rec = K2v @ np.linalg.pinv(Kv)                    # K2 = u K
assert np.allclose(u_rec @ Kv, K2v)
assert np.allclose(u_rec @ dag(u_rec), np.eye(len(K2)))
print("  набор Крауса из спектрального разложения J и исходный набор связаны унитарной u (найдена явно)")
# пример: дефазировка в двух представлениях
p = 0.3
Ka = K_dephasing(p)
P0, P1 = np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)
lam = 2 * p   # эквивалентная запись: (1 - lam) rho + lam (P0 rho P0 + P1 rho P1)
Kb = [sqrt(1 - lam) * I2, sqrt(lam) * P0, sqrt(lam) * P1]
assert np.allclose(choi_kraus(Ka), choi_kraus(Kb))
print("  дефазировка: {sqrt(1-p) I, sqrt(p) Z} и {sqrt(1-2p) I, sqrt(2p) P0, sqrt(2p) P1} — один и тот же канал (p = 0.3)")

# ---------------------------------------------------------------------------
section("6. Каналы кубита на шаре Блоха; каналы Паули; унитальность")


def affine(K):
    """r -> M r + c."""
    c = bloch(apply_kraus(K, I2 / 2))
    M = np.array([bloch(apply_kraus(K, (I2 + P) / 2)) - c for P in (X, Y, Z)]).T
    return M, c


M, c = affine(K_depol(0.4))
assert np.allclose(M, 0.6 * np.eye(3)) and np.allclose(c, 0)
M, c = affine(K_dephasing(0.3))
assert np.allclose(M, np.diag([0.4, 0.4, 1])) and np.allclose(c, 0)
g = 0.4
M, c = affine(K_ad(g))
assert np.allclose(M, np.diag([sqrt(1 - g), sqrt(1 - g), 1 - g])) and np.allclose(c, [0, 0, g])
print("  деполяризующий: r -> (1-p) r;  дефазировка: (x, y, z) -> ((1-2p)x, (1-2p)y, z)")
print("  амплитудное затухание: (x, y, z) -> (sqrt(1-g) x, sqrt(1-g) y, (1-g) z + g); неподвижная точка |0>, канал не унитален")
# канал Паули: lambda_x = pI + pX - pY - pZ и т.д.
pv = rng.dirichlet(np.ones(4))
Kp = [sqrt(q) * P for q, P in zip(pv, PAULIS)]
M, c = affine(Kp)
lx, ly, lz = pv[0] + pv[1] - pv[2] - pv[3], pv[0] - pv[1] + pv[2] - pv[3], pv[0] - pv[1] - pv[2] + pv[3]
assert np.allclose(M, np.diag([lx, ly, lz])) and np.allclose(c, 0)
print(f"  канал Паули p = {np.round(pv, 3)}: M = diag(pI+pX-pY-pZ, pI-pX+pY-pZ, pI-pX-pY+pZ)")


def pauli_probs(l):
    lx, ly, lz = l
    return np.array([1 + lx + ly + lz, 1 + lx - ly - lz, 1 - lx + ly - lz, 1 - lx - ly + lz]) / 4


# тетраэдр Фудзивары–Алгоэта: diag(l) задаёт канал <=> все p >= 0
cases = {"(1, 1, -1) (отражение)": (1, 1, -1), "(-1, -1, -1) (универсальное NOT)": (-1, -1, -1),
         "(1, -1, -1) (сопряжение X)": (1, -1, -1), "(1/3, 1/3, 1/3)": (1 / 3, 1 / 3, 1 / 3),
         "(-1/3, -1/3, -1/3)": (-1 / 3, -1 / 3, -1 / 3), "(0.9, 0.9, -0.9)": (0.9, 0.9, -0.9)}
for name, l in cases.items():
    pp = pauli_probs(l)
    Jl = sum(q * choi_kraus([P]) for q, P in zip(pp, PAULIS))
    cp = np.linalg.eigvalsh(Jl).min() > -1e-12
    assert cp == (pp.min() > -1e-12)
    print(f"    l = {name:34s}: p = {np.round(pp, 3)} -> {'канал' if cp else 'не CP'}")
# унитальность: для кубита унитальный канал = смесь унитарных (здесь: проверка на случайном унитальном канале Паули после поворотов)
print("  каналы Паули унитальны: Phi(I) = I; амплитудное затухание — нет")
assert np.allclose(apply_kraus(Kp, I2), I2) and not np.allclose(apply_kraus(K_ad(0.4), I2), I2)

# ---------------------------------------------------------------------------
section("7. Каналы, разрушающие запутанность")


def ppt_min(R):
    Rg = R.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
    return np.linalg.eigvalsh(Rg).min()


for p in (0.5, 0.6, 2 / 3, 0.7, 0.9):
    Jp = choi_kraus(K_depol(p)) / 2           # (id (x) Delta_p)(|Phi+><Phi+|) = rho_q, q = 1 - p
    q = 1 - p
    assert np.allclose(Jp, q * np.outer(Phi_plus, Phi_plus) + (1 - q) * np.eye(4) / 4)
    print(f"    Delta_p, p = {p:.3f}: Чой/2 = rho_q с q = 1-p = {q:.3f}; min eig частичного транспонирования = {ppt_min(Jp):+.4f}")
print("  Delta_p разрушает запутанность <=> Чой сепарабелен <=> q = 1 - p <= 1/3 <=> p >= 2/3")
assert ppt_min(choi_kraus(K_depol(0.6)) / 2) < 0 and ppt_min(choi_kraus(K_depol(0.7)) / 2) >= 0
# канал «измерь и приготовь» разрушает запутанность
Kmp = [np.outer(ket(a, 2), ket(b, 2)) for a in range(2) for b in range(2) if a == b]   # измерение Z и приготовление |b>
assert ppt_min(choi_kraus(Kmp) / 2) >= -1e-12
print("  «измерь в базисе Z и приготовь результат»: Чой сепарабелен (разрушает запутанность)")

# ---------------------------------------------------------------------------
section("8. Сжатие: расстояние следа и верность не ухудшаются")
viol = 0
for _ in range(300):
    K = random_channel(3, 2, rng.integers(2, 6))
    r, s = random_state(3), random_state(3)
    D0, D1 = 0.5 * tnorm(r - s), 0.5 * tnorm(apply_kraus(K, r) - apply_kraus(K, s))
    F0, F1 = fidelity(r, s), fidelity(apply_kraus(K, r), apply_kraus(K, s))
    assert D1 <= D0 + 1e-10 and F1 >= F0 - 1e-8
print("  300 случайных каналов C^3 -> C^2: D(Phi(rho), Phi(sigma)) <= D(rho, sigma), F(Phi(rho), Phi(sigma)) >= F(rho, sigma)")
# транспонирование на половине пары — не канал: сжатие нарушается для id (x) T
r = np.outer(Phi_plus, Phi_plus).astype(complex)
s = np.diag([1, 0, 0, 0]).astype(complex)
PT = lambda R: R.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
print(f"  id (x) T: D(Phi+, |00>) = {0.5 * tnorm(r - s):.4f} -> D(PT(Phi+), PT(|00>)) = {0.5 * tnorm(PT(r) - PT(s)):.4f} (растёт: не канал)")
assert 0.5 * tnorm(PT(r) - PT(s)) > 0.5 * tnorm(r - s)

# ---------------------------------------------------------------------------
section("9. Верность процесса, средняя верность, ромбовидная норма")


def F_pro(K, Utarget=None):
    d = K[0].shape[1]
    if Utarget is not None:
        K = [dag(Utarget) @ k for k in K]
    J = choi_kraus(K)
    phi = sum(np.kron(ket(i, d), ket(i, d)) for i in range(d)) / sqrt(d)
    return np.real(np.vdot(phi, J @ phi)) / d


def F_avg_haar(K, n=20000):
    d = K[0].shape[1]
    acc = 0
    for _ in range(n):
        v = rng.normal(size=d) + 1j * rng.normal(size=d); v /= np.linalg.norm(v)
        acc += np.real(np.vdot(v, apply_kraus(K, np.outer(v, v.conj())) @ v))
    return acc / n


pv = np.array([0.9, 0.05, 0.03, 0.02])
Kp = [sqrt(q) * P for q, P in zip(pv, PAULIS)]
assert np.isclose(F_pro(Kp), pv[0])
print(f"  канал Паули: F_pro = p_I = {F_pro(Kp):.4f}")
for name, Kk in (("канал Паули", Kp), ("амплитудное затухание g = 0.2", K_ad(0.2)), ("случайный канал", random_channel(2, 2, 2))):
    fp = F_pro(Kk)
    fa_formula = (2 * fp + 1) / 3
    six = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1]) / sqrt(2), np.array([1, -1]) / sqrt(2), np.array([1, 1j]) / sqrt(2), np.array([1, -1j]) / sqrt(2)]
    fa6 = np.mean([np.real(np.vdot(v, apply_kraus(Kk, np.outer(v, v.conj())) @ v)) for v in six])
    fah = F_avg_haar(Kk, 20000)
    print(f"    {name:30s}: F_pro = {fp:.4f}, (d F_pro + 1)/(d + 1) = {fa_formula:.4f}, по 6 состояниям {fa6:.4f}, Монте-Карло (2e4) {fah:.4f}")
    assert np.isclose(fa6, fa_formula) and abs(fah - fa_formula) < 0.005
# d = 3
K3 = random_channel(3, 3, 2)
fp3 = F_pro(K3)
fa3 = F_avg_haar(K3, 20000)
assert abs(fa3 - (3 * fp3 + 1) / 4) < 0.005
print(f"    случайный канал d = 3: (d F_pro + 1)/(d + 1) = {(3 * fp3 + 1) / 4:.4f}, Монте-Карло {fa3:.4f}")


def diamond_dist(J1, J2, d_in, d_out, starts=12):
    """|| Phi1 - Phi2 ||_diamond = max_sigma || (sqrt(sigma) (x) I)(J1 - J2)(sqrt(sigma) (x) I) ||_1 (кубит на входе)."""
    dJ = J1 - J2
    assert d_in == 2

    def val(v):
        rv = v / max(1, np.linalg.norm(v))
        s = (I2 + rv[0] * X + rv[1] * Y + rv[2] * Z) / 2
        w, U = np.linalg.eigh(s)
        sq = U @ np.diag(np.sqrt(np.clip(w, 0, None))) @ dag(U)
        A = np.kron(sq, np.eye(d_out))
        return tnorm(A @ dJ @ A)
    best = val(np.zeros(3))
    for _ in range(starts):
        r = minimize(lambda v: -val(v), rng.normal(size=3) * 0.5, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 4000})
        best = max(best, -r.fun)
    return best


def no_ancilla_dist(K1, K2, d, n=4000):
    best = 0
    for _ in range(n):
        v = rng.normal(size=d) + 1j * rng.normal(size=d); v /= np.linalg.norm(v)
        r = np.outer(v, v.conj())
        best = max(best, tnorm(apply_kraus(K1, r) - apply_kraus(K2, r)))
    return best


Jid = choi_kraus([I2])
dd = diamond_dist(choi_kraus(Kp), Jid, 2, 2)
print(f"  канал Паули против тождественного: ||.||_diamond = {dd:.6f}, 2(1 - p_I) = {2 * (1 - pv[0]):.6f}")
assert abs(dd - 2 * (1 - pv[0])) < 1e-6
print("  когерентная ошибка R_x(eps) против канала Паули с той же верностью процесса:")
print("     eps       1 - F_pro    ||R_x - id||_dia   2 sin(eps/2)   Паули: 2(1 - F_pro)")
coh = []
for eps in (0.2, 0.05, 0.01):
    Rx = np.array([[cos(eps / 2), -1j * sin(eps / 2)], [-1j * sin(eps / 2), cos(eps / 2)]])
    infid = 1 - F_pro([Rx])
    assert np.isclose(infid, sin(eps / 2) ** 2)
    dRx = diamond_dist(choi_kraus([Rx]), Jid, 2, 2)
    assert abs(dRx - 2 * sin(eps / 2)) < 1e-6
    coh.append((eps, infid, dRx))
    print(f"     {eps:5.2f}    {infid:.3e}    {dRx:.4e}         {2 * sin(eps / 2):.4e}     {2 * infid:.3e}")
r_inf = 1e-4
print(f"  при 1 - F_pro = 1e-4: когерентная ошибка ||.||_dia = 2 sqrt(1e-4) = {2 * sqrt(r_inf):.0e}, канал Паули — {2 * r_inf:.0e} (в {sqrt(r_inf) ** -1:.0f} раз меньше)")
# пример, где вспомогательная система нужна: каналы Вернера–Холево, d = 2
W0 = lambda r: (np.trace(r) * I2 + r.T) / 3
W1 = lambda r: np.trace(r) * I2 - r.T
J0, J1 = choi(W0, 2), choi(W1, 2)
assert np.allclose(J0, (np.eye(4) + SWAP) / 3) and np.allclose(J1, np.eye(4) - SWAP)
assert np.linalg.eigvalsh(J0).min() > -1e-12 and np.linalg.eigvalsh(J1).min() > -1e-12
assert np.allclose(ptrace(J0, [2, 2], [0]), I2) and np.allclose(ptrace(J1, [2, 2], [0]), I2)
K0w, K1w = kraus_from_choi(J0, 2, 2), kraus_from_choi(J1, 2, 2)
dd_w = diamond_dist(J0, J1, 2, 2)
na_w = no_ancilla_dist(K0w, K1w, 2)
print(f"  каналы Вернера–Холево W0 = (I Tr + T)/3, W1 = I Tr - T: без вспомогательной системы max ||W0(rho) - W1(rho)||_1 = {na_w:.4f} (= 4/3),")
print(f"    с вспомогательной системой (вход |Psi->): ||W0 - W1||_dia = {dd_w:.4f} (= 2, различимы идеально)")
assert abs(na_w - 4 / 3) < 1e-3 and abs(dd_w - 2) < 1e-6

# ---------------------------------------------------------------------------
section("10. Ответы к задачам семинара")
print("  1: J(T) = SWAP, собственные значения +1 (x3), -1 (x1)")
print(f"  3: J(дефазировки p) = [[1,0,0,1-2p],[0,0,0,0],[0,0,0,0],[1-2p,0,0,1]]; при p = 0.3: 1 - 2p = {1 - 2 * 0.3:.1f}")
Jd = choi_kraus(K_dephasing(0.3))
assert np.allclose(Jd, np.array([[1, 0, 0, 0.4], [0, 0, 0, 0], [0, 0, 0, 0], [0.4, 0, 0, 1]]))
Jad = choi_kraus(K_ad(0.4))
print("  4: J(AD, g) = [[1,0,0,sqrt(1-g)],[0,0,0,0],[0,0,g,0],[sqrt(1-g),0,0,1-g]]; g = 0.4:")
print("     ", np.round(Jad.real, 4).tolist())
assert np.allclose(Jad, np.array([[1, 0, 0, sqrt(0.6)], [0, 0, 0, 0], [0, 0, 0.4, 0], [sqrt(0.6), 0, 0, 0.6]]))
Kr = kraus_from_choi(Jad, 2, 2)
assert len(Kr) == 2 and all(min(min(np.linalg.norm(k - s * k0) for s in (1, -1)) for k0 in K_ad(0.4)) < 1e-10 for k in Kr)
print("     Краус из спектрального разложения J(AD) совпадает с {K0, K1} с точностью до знака")
print(f"  8: Delta_p разрушает запутанность при p >= 2/3; F_pro(Delta_p) = 1 - 3p/4 = {1 - 3 * (2 / 3) / 4:.2f} при p = 2/3")
print(f"  9: Паули (0.9, 0.05, 0.03, 0.02): F_pro = 0.9, F_avg = {(2 * 0.9 + 1) / 3:.4f}, ||.||_dia = 0.2")
print(f"  10: R_x(0.01): 1 - F_pro = {sin(0.005) ** 2:.2e}, ||.||_dia = {2 * sin(0.005):.4e}")
print("\nВсе проверки пройдены.")
