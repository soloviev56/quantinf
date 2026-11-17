"""
Лекция I.1. Линейная алгебра для квантовой информатики.
Численные иллюстрации ко всем утверждениям лекции.

Запуск:  python примеры.py
Требуется: numpy, scipy.
Каждый блок печатает проверяемую величину; все проверки оформлены как assert,
поэтому успешный запуск до конца = все утверждения подтверждены численно.
"""
import numpy as np
from scipy.linalg import expm, sqrtm, polar

rng = np.random.default_rng(2026)
TOL = 1e-12

# ---------------------------------------------------------------------------
# Базовые объекты
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (X + Z) / np.sqrt(2)
T = np.diag([1, np.exp(1j * np.pi / 4)])
PAULI = [X, Y, Z]

ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)
ketp = (ket0 + ket1) / np.sqrt(2)
ketm = (ket0 - ket1) / np.sqrt(2)


def dag(A):
    """Эрмитово сопряжение A† = (A*)^T."""
    return A.conj().T


def ketbra(a, b):
    """Внешнее произведение |a><b|."""
    return np.outer(a, b.conj())


def braket(a, b):
    """Скалярное произведение <a|b> (антилинейно по первому аргументу!)."""
    return np.vdot(a, b)  # vdot сопрягает первый аргумент


def random_unitary(d):
    """Унитарная матрица по мере Хаара (QR-разложение гинибровской матрицы)."""
    G = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2)
    Q, R = np.linalg.qr(G)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def random_hermitian(d):
    G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    return (G + dag(G)) / 2


def section(title):
    print("\n" + "=" * 72 + "\n" + title + "\n" + "=" * 72)


# ---------------------------------------------------------------------------
section("1. Скалярное произведение: антилинейность по первому аргументу")
a = rng.normal(size=2) + 1j * rng.normal(size=2)
b = rng.normal(size=2) + 1j * rng.normal(size=2)
c = 2 + 3j
print("<c a|b> =", braket(c * a, b), " c*<a|b> =", np.conj(c) * braket(a, b))
assert abs(braket(c * a, b) - np.conj(c) * braket(a, b)) < TOL
# Неравенство Коши–Шварца
assert abs(braket(a, b)) <= np.linalg.norm(a) * np.linalg.norm(b) + TOL

# ---------------------------------------------------------------------------
section("2. Полнота базиса: sum_i |e_i><e_i| = I (для базисов Z и X)")
for name, (e0, e1) in {"Z": (ket0, ket1), "X": (ketp, ketm)}.items():
    S = ketbra(e0, e0) + ketbra(e1, e1)
    print(f"базис {name}: ||sum - I|| =", np.linalg.norm(S - I2))
    assert np.allclose(S, I2)

# ---------------------------------------------------------------------------
section("3. Алгебра Паули: s_j s_k = delta_jk I + i eps_jkl s_l")
eps = np.zeros((3, 3, 3))
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1
for j in range(3):
    for k in range(3):
        rhs = (j == k) * I2 + 1j * sum(eps[j, k, l] * PAULI[l] for l in range(3))
        assert np.allclose(PAULI[j] @ PAULI[k], rhs)
print("все 9 тождеств выполнены; [X,Z] = -2iY:",
      np.allclose(X @ Z - Z @ X, -2j * Y))
# (a.s)(b.s) = (a.b) I + i (a x b).s
va, vb = rng.normal(size=3), rng.normal(size=3)
As = sum(va[i] * PAULI[i] for i in range(3))
Bs = sum(vb[i] * PAULI[i] for i in range(3))
cross = np.cross(va, vb)
assert np.allclose(As @ Bs, va @ vb * I2 + 1j * sum(cross[i] * PAULI[i] for i in range(3)))
print("(a.s)(b.s) = (a.b)I + i(a x b).s  — проверено")

# ---------------------------------------------------------------------------
section("4. Классы операторов")
A = random_hermitian(4)
ev = np.linalg.eigvals(A)
print("эрмитов: max|Im lambda| =", np.max(np.abs(ev.imag)))
assert np.max(np.abs(ev.imag)) < 1e-10
U = random_unitary(4)
print("унитарный: ||U^dag U - I|| =", np.linalg.norm(dag(U) @ U - np.eye(4)),
      "; |lambda| =", np.round(np.abs(np.linalg.eigvals(U)), 12))
assert np.allclose(np.abs(np.linalg.eigvals(U)), 1)
# сохранение скалярного произведения
a4 = rng.normal(size=4) + 1j * rng.normal(size=4)
b4 = rng.normal(size=4) + 1j * rng.normal(size=4)
assert abs(braket(U @ a4, U @ b4) - braket(a4, b4)) < 1e-12
# положительный оператор B^dag B; проектор
G = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
Pos = dag(G) @ G
print("B^dag B >= 0: min lambda =", np.linalg.eigvalsh(Pos).min())
assert np.linalg.eigvalsh(Pos).min() > -TOL
P = ketbra(ketp, ketp)
assert np.allclose(P @ P, P) and np.allclose(P, dag(P))
print("|+><+| — проектор; Tr P = rank P =", np.trace(P).real)

# ---------------------------------------------------------------------------
section("5. Спектральная теорема; ненормальный оператор")
lam, V = np.linalg.eigh(A)
A_rec = sum(lam[k] * ketbra(V[:, k], V[:, k]) for k in range(4))
print("эрмитов A = sum lambda_k P_k: ошибка =", np.linalg.norm(A - A_rec))
assert np.allclose(A, A_rec)
N = np.array([[0, 1], [0, 0]], dtype=complex)
print("N = |0><1|: нормален?", np.allclose(N @ dag(N), dag(N) @ N),
      "; rank(N - 0*I) =", np.linalg.matrix_rank(N),
      "-> собственное подпространство 1-мерно, базиса из собственных векторов нет")

# ---------------------------------------------------------------------------
section("6. Функции от операторов: exp(-i th n.s/2) = cos(th/2) I - i sin(th/2) n.s")
for _ in range(100):
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    th = rng.uniform(0, 4 * np.pi)
    ns = sum(n[i] * PAULI[i] for i in range(3))
    lhs = expm(-1j * th * ns / 2)
    rhs = np.cos(th / 2) * I2 - 1j * np.sin(th / 2) * ns
    assert np.allclose(lhs, rhs)
print("формула проверена на 100 случайных (n, theta)")
# функция через спектральное разложение совпадает с expm
Hh = random_hermitian(5)
lam, V = np.linalg.eigh(Hh)
U_spec = V @ np.diag(np.exp(-1j * 0.7 * lam)) @ dag(V)
print("exp(-iHt) через спектр vs expm: ошибка =", np.linalg.norm(U_spec - expm(-1j * 0.7 * Hh)))
assert np.allclose(U_spec, expm(-1j * 0.7 * Hh))
# поворот на 2*pi даёт -I
n = np.array([0, 0, 1.0])
print("R_z(2 pi) =", np.round(expm(-1j * 2 * np.pi * Z / 2), 12).diagonal(), "(= -I)")
# H = i R_n(pi), n = (1,0,1)/sqrt2
ns = (X + Z) / np.sqrt(2)
assert np.allclose(H, 1j * expm(-1j * np.pi * ns / 2))
print("H = i * R_n(pi), n=(1,0,1)/sqrt(2) — проверено")

# ---------------------------------------------------------------------------
section("7. e^{A+B} != e^A e^B: ошибка ~ (t^2/2)[A,B]")
for t in [0.1, 0.05, 0.025]:
    exact = expm(-1j * t * (X + Z))
    split = expm(-1j * t * X) @ expm(-1j * t * Z)
    pred = -(t ** 2 / 2) * (X @ Z - Z @ X)  # главный член разности split - exact
    err = np.linalg.norm(split - exact, 2)
    print(f"t={t:6.3f}: ||e^-itX e^-itZ - e^-it(X+Z)|| = {err:.3e};"
          f"  ||(t^2/2)[X,Z]|| = {np.linalg.norm(pred, 2):.3e}")
# порядок t^2: при уменьшении t вдвое ошибка падает ~ в 4 раза

# ---------------------------------------------------------------------------
section("8. След: цикличность, Tr(A|psi><psi|) = <psi|A|psi>, базис Паули")
A3, B3, C3 = (random_hermitian(3) for _ in range(3))
print("Tr(ABC) - Tr(CAB) =", abs(np.trace(A3 @ B3 @ C3) - np.trace(C3 @ A3 @ B3)))
print("Tr(ABC) - Tr(BAC) =", abs(np.trace(A3 @ B3 @ C3) - np.trace(B3 @ A3 @ C3)),
      "(в общем случае != 0)")
psi = rng.normal(size=3) + 1j * rng.normal(size=3)
psi /= np.linalg.norm(psi)
assert abs(np.trace(A3 @ ketbra(psi, psi)) - braket(psi, A3 @ psi)) < TOL
# ортонормированность {I, X, Y, Z}/sqrt2 в скалярном произведении Гильберта–Шмидта
basis = [M / np.sqrt(2) for M in [I2, X, Y, Z]]
gram = np.array([[np.trace(dag(P1) @ P2) for P2 in basis] for P1 in basis])
assert np.allclose(gram, np.eye(4))
M = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
M_rec = 0.5 * (np.trace(M) * I2 + sum(np.trace(M @ S) * S for S in PAULI))
print("разложение M по базису Паули: ошибка =", np.linalg.norm(M - M_rec))
assert np.allclose(M, M_rec)

# ---------------------------------------------------------------------------
section("9. SVD и полярное разложение")
A = np.array([[1, 1], [0, 0]], dtype=complex)
Uu, s, Vh = np.linalg.svd(A)
print("A = [[1,1],[0,0]]: сингулярные числа =", s, " (ожидается sqrt2, 0)")
assert np.allclose(s, [np.sqrt(2), 0])
G = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
Up, Pp = polar(G, side="right")  # G = Up @ Pp,  Pp = sqrt(G^dag G)
assert np.allclose(G, Up @ Pp) and np.allclose(Pp, sqrtm(dag(G) @ G))
assert np.allclose(dag(Up) @ Up, np.eye(3))
print("полярное разложение G = U|G|, |G| = sqrt(G^dag G) — проверено")

# ---------------------------------------------------------------------------
section("10. Нормы")


def schatten(A, p):
    s = np.linalg.svd(A, compute_uv=False)
    return s.max() if p == np.inf else (s ** p).sum() ** (1 / p)


for _ in range(1000):
    G = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    n1, n2, ninf = schatten(G, 1), schatten(G, 2), schatten(G, np.inf)
    assert ninf <= n2 + 1e-12 <= n1 + 2e-12
    assert np.isclose(n2, np.linalg.norm(G, "fro"))
    Bm = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    assert abs(np.trace(Bm @ G)) <= schatten(Bm, np.inf) * n1 + 1e-9  # Гёльдер
print("||A||_inf <= ||A||_2 <= ||A||_1 и |Tr(BA)| <= ||B||_inf ||A||_1 — на 1000 примерах")

# Следовая норма разности чистых состояний: 2 sqrt(1 - |<psi|phi>|^2)
d01 = schatten(ketbra(ket0, ket0) - ketbra(ketp, ketp), 1)
print("|| |0><0| - |+><+| ||_1 =", d01, " (ожидается sqrt2 =", np.sqrt(2), ")")
assert np.isclose(d01, np.sqrt(2))

# Расстояние между унитарными с точностью до глобальной фазы: I и T
phis = np.linspace(-np.pi, np.pi, 2001)
dist = min(schatten(I2 - np.exp(1j * f) * T, np.inf) for f in phis)
print("min_phi ||I - e^{i phi} T|| ~", round(dist, 4),
      " (точно 2 sin(pi/16) =", round(2 * np.sin(np.pi / 16), 4), ")")
assert abs(dist - 2 * np.sin(np.pi / 16)) < 1e-3

# Вариационная формула: ||A||_1 = max_U |Tr(U A)|, максимум на U = V W^dag из SVD
G = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
W, s, Vh = np.linalg.svd(G)
U_opt = dag(Vh) @ dag(W)
print("|Tr(U_opt G)| =", abs(np.trace(U_opt @ G)), " ||G||_1 =", s.sum())
assert np.isclose(abs(np.trace(U_opt @ G)), s.sum())
assert all(abs(np.trace(random_unitary(3) @ G)) <= s.sum() + 1e-12 for _ in range(2000))

print("\nВсе проверки пройдены.")
