"""
Лекция I.3. Матрица плотности, обобщённые измерения, различимость.
Численные иллюстрации ко всем утверждениям лекции.

Запуск:  python примеры.py
Требуется: numpy. Все проверки оформлены как assert.
Конвенция верности: F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2  (квадрат!).
"""
import numpy as np

rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = [X, Y, Z]
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)
ketp = (ket0 + ket1) / np.sqrt(2)
ketm = (ket0 - ket1) / np.sqrt(2)


def dag(A):
    return A.conj().T


def proj(v):
    return np.outer(v, v.conj())


def rho_bloch(r):
    return 0.5 * (I2 + sum(r[i] * PAULI[i] for i in range(3)))


def bloch(rho):
    return np.array([np.trace(rho @ S).real for S in PAULI])


def random_state(d):
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    return v / np.linalg.norm(v)


def random_rho(d, rank=None):
    """Случайная матрица плотности: G G^dag / Tr (мера Гильберта–Шмидта при rank = d)."""
    k = d if rank is None else rank
    G = rng.normal(size=(d, k)) + 1j * rng.normal(size=(d, k))
    R = G @ dag(G)
    return R / np.trace(R)


def is_density(rho, tol=1e-10):
    return (np.allclose(rho, dag(rho), atol=tol) and abs(np.trace(rho) - 1) < tol
            and np.linalg.eigvalsh((rho + dag(rho)) / 2).min() > -tol)


def trace_norm(A):
    return np.linalg.svd(A, compute_uv=False).sum()


def trace_distance(r, s):
    return 0.5 * trace_norm(r - s)


def psd_sqrt(A):
    w, V = np.linalg.eigh((A + dag(A)) / 2)
    w[w < 1e-12] = 0  # иначе sqrt от шума ~1e-16 даёт ошибку ~1e-8
    return V @ np.diag(np.sqrt(w)) @ dag(V)


def fidelity(r, s):
    sr = psd_sqrt(r)
    return np.real(np.trace(psd_sqrt(sr @ s @ sr))) ** 2


def section(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


# ---------------------------------------------------------------------------
section("1. Ансамбль -> матрица плотности; Tr(P rho) даёт те же вероятности")
ens = [(0.3, random_state(3)), (0.5, random_state(3)), (0.2, random_state(3))]
rho = sum(p * proj(v) for p, v in ens)
P = proj(random_state(3))
p_ens = sum(p * np.vdot(v, P @ v).real for p, v in ens)
assert np.isclose(p_ens, np.trace(P @ rho).real) and is_density(rho)
print("p по ансамблю =", round(p_ens, 10), " Tr(P rho) =", round(np.trace(P @ rho).real, 10))

section("2. Чистота: Tr rho^2 in [1/d, 1]; = 1 только для чистых")
for d in [2, 4, 8]:
    pur = [np.trace(np.linalg.matrix_power(random_rho(d), 2)).real for _ in range(2000)]
    assert min(pur) >= 1 / d - 1e-12 and max(pur) <= 1 + 1e-12
    print(f"  d={d}: Tr rho^2 в [{min(pur):.4f}, {max(pur):.4f}], нижняя граница 1/d = {1/d:.4f}")
v = random_state(4)
assert np.isclose(np.trace(proj(v) @ proj(v)).real, 1) and np.isclose(np.trace(np.eye(4) / 4 @ np.eye(4) / 4), 1 / 4)
print("  чистое: Tr rho^2 = 1; максимально смешанное I/d: Tr rho^2 = 1/d")

section("3. Суперпозиция != смесь: |+> и I/2")
mix = 0.5 * (proj(ket0) + proj(ket1))
for name, r in [("|+><+|", proj(ketp)), ("I/2", mix)]:
    print(f"  {name:7s}: P(0 в базисе Z) = {np.trace(proj(ket0) @ r).real:.3f},"
          f"  P(+ в базисе X) = {np.trace(proj(ketp) @ r).real:.3f}")

section("4. Неединственность разложения и теорема Хьюстона–Йожи–Вуттерса")
mixX = 0.5 * (proj(ketp) + proj(ketm))
print("  1/2(|0><0|+|1><1|) == 1/2(|+><+|+|-><-|):", np.allclose(mix, mixX))
# HJW: {sqrt(p_i)|psi_i>} = U {sqrt(q_j)|phi_j>}
d, k = 3, 5
rho = random_rho(d)
w, V = np.linalg.eigh(rho)
eig_vecs = [np.sqrt(w[i]) * V[:, i] for i in range(d)] + [np.zeros(d)] * (k - d)  # дополнили нулями
Uk = np.linalg.qr(rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k)))[0]
new = [sum(Uk[i, j] * eig_vecs[j] for j in range(k)) for i in range(k)]  # ненормированные векторы
rho_new = sum(np.outer(v, v.conj()) for v in new)
probs = [np.linalg.norm(v) ** 2 for v in new]
print(f"  новый ансамбль из {k} состояний с вероятностями {np.round(probs, 3)} даёт тот же rho:",
      np.allclose(rho, rho_new))
assert np.allclose(rho, rho_new)

# ---------------------------------------------------------------------------
section("5. Шар Блоха: rho >= 0 <=> |r| <= 1; собственные значения (1 +- |r|)/2")
for _ in range(1000):
    r = rng.normal(size=3) * rng.uniform(0, 1.5) / np.sqrt(3)
    rho = rho_bloch(r)
    ev = np.sort(np.linalg.eigvalsh(rho))
    nr = np.linalg.norm(r)
    assert np.allclose(ev, [(1 - nr) / 2, (1 + nr) / 2])
    assert is_density(rho) == (nr <= 1 + 1e-12)
    assert np.isclose(np.trace(rho @ rho).real, (1 + nr ** 2) / 2)
print("  проверено на 1000 векторах r (часть — вне шара)")
A = np.array([[0.75, 0.25], [0.25, 0.25]])
print("  [[3/4,1/4],[1/4,1/4]]: r =", bloch(A), "|r| =", round(np.linalg.norm(bloch(A)), 4), "-> допустимо")
B = np.array([[0.5, (1 + 1j) / 2], [(1 - 1j) / 2, 0.5]])
print("  [[1/2,(1+i)/2],...]: r =", bloch(B), "|r| =", round(np.linalg.norm(bloch(B)), 4), "-> НЕ матрица плотности")
assert not is_density(B)
mid = 0.5 * (proj(ket0) + proj(ketp))
print("  1/2(|0><0|+|+><+|): собственные значения =", np.round(np.linalg.eigvalsh(mid), 6),
      " (1 +- 1/sqrt2)/2 =", np.round([(1 - 1 / np.sqrt(2)) / 2, (1 + 1 / np.sqrt(2)) / 2], 6))

# ---------------------------------------------------------------------------
section("6. POVM: ошибка считывания, трин, однозначное различение")
e0, e1 = 0.02, 0.05  # P(прочитали 1 | был 0), P(прочитали 0 | был 1)
E0 = (1 - e0) * proj(ket0) + e1 * proj(ket1)
E1 = I2 - E0
assert np.allclose(E0 + E1, I2) and np.linalg.eigvalsh(E0).min() >= 0
print("  ошибка считывания: E0 =", np.round(np.diag(E0).real, 3), "не проектор:", not np.allclose(E0 @ E0, E0))
# трин
# трин-состояния: векторы в R^2 под углом 60 градусов, на сфере Блоха — под 120 градусов (половинные углы)
tr = [np.array([np.cos(np.pi * k / 3), np.sin(np.pi * k / 3)], dtype=complex) for k in range(3)]
Etr = [2 / 3 * proj(v) for v in tr]
assert np.allclose(sum(Etr), I2)
Ptab = np.array([[np.trace(Etr[k] @ proj(tr[j])).real for k in range(3)] for j in range(3)])
print("  трин-POVM, p(k|j):\n", np.round(Ptab, 4))
assert np.allclose(np.diag(Ptab), 2 / 3)
assert np.allclose(Ptab[~np.eye(3, dtype=bool)], 1 / 6)

# ---------------------------------------------------------------------------
section("7. Теорема Наймарка: POVM = унитарный оператор + проективное измерение вспомогательной системы")


def naimark_unitary(E):
    """Унитарный U на H (x) C^k, U(|psi>|0>) = sum_m sqrt(E_m)|psi> (x) |m>."""
    d, k = E[0].shape[0], len(E)
    Vis = np.zeros((d * k, d), dtype=complex)  # изометрия
    for m, Em in enumerate(E):
        S = psd_sqrt(Em)
        for j in range(d):
            for i in range(d):
                Vis[i * k + m, j] = S[i, j]
    assert np.allclose(dag(Vis) @ Vis, np.eye(d))
    # столбцы U с индексами j*k + 0 задаются изометрией; остальные — дополнение до ортонормированного базиса
    U = np.zeros((d * k, d * k), dtype=complex)
    fixed = [j * k for j in range(d)]
    U[:, fixed] = Vis
    rest = [c for c in range(d * k) if c not in fixed]
    M = np.hstack([Vis, rng.normal(size=(d * k, len(rest))) + 1j * rng.normal(size=(d * k, len(rest)))])
    Q, _ = np.linalg.qr(M)
    U[:, rest] = Q[:, d:]
    return U


U = naimark_unitary(Etr)
assert np.allclose(dag(U) @ U, np.eye(6))
psi = random_state(2)
out = U @ np.kron(psi, np.array([1, 0, 0]))
p_anc = [sum(abs(out[i * 3 + m]) ** 2 for i in range(2)) for m in range(3)]
p_povm = [np.vdot(psi, Em @ psi).real for Em in Etr]
print("  p(m) через Наймарка:", np.round(p_anc, 6), "  Tr(E_m rho):", np.round(p_povm, 6))
assert np.allclose(p_anc, p_povm)

# ---------------------------------------------------------------------------
section("8. Расстояние следа")
# для кубита D = |r - s|/2
for _ in range(1000):
    r, s = [rng.normal(size=3) for _ in range(2)]
    r, s = r / max(1, np.linalg.norm(r)), s / max(1, np.linalg.norm(s))
    assert np.isclose(trace_distance(rho_bloch(r), rho_bloch(s)), np.linalg.norm(r - s) / 2)
print("  кубит: D = |r - s|/2 — проверено на 1000 парах")
# D = max_P Tr P(rho - sigma), максимум на проекторе на положительную часть
rho, sig = random_rho(4), random_rho(4)
w, V = np.linalg.eigh(rho - sig)
Popt = V[:, w > 0] @ dag(V[:, w > 0])
assert np.isclose(np.trace(Popt @ (rho - sig)).real, trace_distance(rho, sig))
for _ in range(2000):  # никакой 0 <= E <= I не даёт больше
    G = random_rho(4) * rng.uniform(0, 4)
    E = G / max(1, np.linalg.eigvalsh(G).max())
    assert np.trace(E @ (rho - sig)).real <= trace_distance(rho, sig) + 1e-12
print("  D = max_{0<=E<=I} Tr E(rho-sigma) — проверено")
# монотонность при измерении: TV(p, q) <= D, равенство для оптимального измерения
for _ in range(500):
    Vb = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    pr = [np.vdot(Vb[:, i], rho @ Vb[:, i]).real for i in range(4)]
    qs = [np.vdot(Vb[:, i], sig @ Vb[:, i]).real for i in range(4)]
    assert 0.5 * np.abs(np.subtract(pr, qs)).sum() <= trace_distance(rho, sig) + 1e-12
print("  классическое расстояние распределений исходов <= D — на 500 базисах")

# ---------------------------------------------------------------------------
section("9. Верность и неравенства Фукса–ван де Граафа")
for _ in range(2000):
    d = rng.choice([2, 3, 4])
    rho, sig = random_rho(d, rank=rng.integers(1, d + 1)), random_rho(d, rank=rng.integers(1, d + 1))
    F, D = fidelity(rho, sig), trace_distance(rho, sig)
    assert np.isclose(F, fidelity(sig, rho), atol=1e-8)
    assert 1 - np.sqrt(F) <= D + 1e-8 and D <= np.sqrt(max(0, 1 - F)) + 1e-8
print("  1 - sqrt(F) <= D <= sqrt(1 - F) — на 2000 парах (d = 2..4, разные ранги)")
a, b = random_state(3), random_state(3)
assert np.isclose(fidelity(proj(a), proj(b)), abs(np.vdot(a, b)) ** 2)
assert np.isclose(trace_distance(proj(a), proj(b)), np.sqrt(1 - abs(np.vdot(a, b)) ** 2))
print("  чистые: F = |<a|b>|^2, D = sqrt(1 - F) (правое неравенство обращается в равенство)")
sig = random_rho(3)
assert np.isclose(fidelity(proj(a), sig), np.vdot(a, sig @ a).real)
# кубит: F = Tr(rho sigma) + 2 sqrt(det rho det sigma)
for _ in range(1000):
    r1, r2 = random_rho(2), random_rho(2)
    Fq = np.trace(r1 @ r2).real + 2 * np.sqrt(max(0, np.linalg.det(r1).real * np.linalg.det(r2).real))
    assert np.isclose(fidelity(r1, r2), Fq, atol=1e-8)
print("  кубит: F = Tr(rho sigma) + 2 sqrt(det rho det sigma) — на 1000 парах")
print("  |0> против I/2: F =", round(fidelity(proj(ket0), I2 / 2), 6), " D =", round(trace_distance(proj(ket0), I2 / 2), 6))

# ---------------------------------------------------------------------------
section("10. Граница Хелстрома: P_succ = (1 + ||p0 rho0 - p1 rho1||_1)/2")


def helstrom(p0, r0, r1):
    return 0.5 * (1 + trace_norm(p0 * r0 - (1 - p0) * r1))


for _ in range(300):
    d = rng.choice([2, 3])
    r0, r1 = random_rho(d), random_rho(d)
    p0 = rng.uniform(0.05, 0.95)
    Ph = helstrom(p0, r0, r1)
    # оптимальный проектор
    w, V = np.linalg.eigh(p0 * r0 - (1 - p0) * r1)
    Eopt = V[:, w > 0] @ dag(V[:, w > 0])
    Pemp = p0 * np.trace(Eopt @ r0).real + (1 - p0) * np.trace((np.eye(d) - Eopt) @ r1).real
    assert np.isclose(Ph, Pemp)
    for _ in range(50):  # случайные двухисходные POVM не лучше
        G = random_rho(d) * rng.uniform(0, 3)
        E = G / max(1, np.linalg.eigvalsh(G).max())
        Pr = p0 * np.trace(E @ r0).real + (1 - p0) * np.trace((np.eye(d) - E) @ r1).real
        assert Pr <= Ph + 1e-12
print("  оптимальность проектора на положительную часть — на 300 задачах x 50 случайных POVM")
c = abs(np.vdot(ket0, ketp))
print("  |0> vs |+>, p0 = 1/2: P_succ =", round(helstrom(0.5, proj(ket0), proj(ketp)), 6),
      "; формула (1 + sqrt(1 - c^2))/2 =", round((1 + np.sqrt(1 - c ** 2)) / 2, 6))
for p0 in [0.5, 0.7, 0.9]:
    a, b = random_state(2), random_state(2)
    cc = abs(np.vdot(a, b))
    assert np.isclose(helstrom(p0, proj(a), proj(b)), 0.5 * (1 + np.sqrt(1 - 4 * p0 * (1 - p0) * cc ** 2)))
print("  чистые состояния, неравные априорные: P = (1 + sqrt(1 - 4 p0 p1 |<a|b>|^2))/2 — проверено")
# n копий
print("  n копий |0> и |+>: вероятность ошибки Хелстрома и приближение c^(2n)/4:")
for n in [1, 2, 4, 8, 16]:
    Perr = 0.5 * (1 - np.sqrt(1 - c ** (2 * n)))
    print(f"    n={n:2d}: P_err = {Perr:.3e},  c^(2n)/4 = {c**(2*n)/4:.3e}")

# ---------------------------------------------------------------------------
section("11. Однозначное различение (IDP): P_succ = 1 - |<psi|phi>|")
for alpha in [0.2, np.pi / 4, 1.2]:
    psi = np.array([1, 0], dtype=complex)
    phi = np.array([np.cos(alpha), np.sin(alpha)], dtype=complex)
    cc = abs(np.vdot(psi, phi))
    perp = lambda v: np.array([-np.conj(v[1]), np.conj(v[0])])
    E_psi = proj(perp(phi)) / (1 + cc)  # срабатывает только на psi
    E_phi = proj(perp(psi)) / (1 + cc)
    E_q = I2 - E_psi - E_phi
    assert np.linalg.eigvalsh(E_q).min() > -1e-12
    # никогда не ошибается
    assert np.isclose(np.vdot(phi, E_psi @ phi), 0) and np.isclose(np.vdot(psi, E_phi @ psi), 0)
    Ps = 0.5 * (np.vdot(psi, E_psi @ psi).real + np.vdot(phi, E_phi @ phi).real)
    assert np.isclose(Ps, 1 - cc)
    # коэффициент больше 1/(1+c) нарушает положительность E_?
    t = 1.01 / (1 + cc)
    assert np.linalg.eigvalsh(I2 - t * proj(perp(phi)) - t * proj(perp(psi))).min() < 0
    print(f"  overlap c = {cc:.3f}: IDP = {Ps:.4f} (1 - c), Хелстром = {(1 + np.sqrt(1 - cc**2))/2:.4f}, ошибок нет")

print("\nВсе проверки пройдены.")
