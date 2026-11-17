"""
Лекция I.4. Составные системы: тензорное произведение, частичный след,
разложение Шмидта, очищение, теорема Ульмана.
Численные иллюстрации ко всем утверждениям лекции.

Запуск:  python примеры.py
Требуется: numpy. Все проверки оформлены как assert.
Порядок подсистем: как в np.kron и в N&C — первый множитель старший (|q0 q1 ... >).
"""
import numpy as np

rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (X + Z) / np.sqrt(2)
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)


def dag(A):
    return A.conj().T


def proj(v):
    return np.outer(v, v.conj())


def kron(*ops):
    out = np.array([[1]], dtype=complex) if ops[0].ndim == 2 else np.array([1], dtype=complex)
    for o in ops:
        out = np.kron(out, o)
    return out


def random_state(d):
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    return v / np.linalg.norm(v)


def random_unitary(d):
    Q, R = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
    return Q * (np.diag(R) / abs(np.diag(R)))


def random_rho(d, rank=None):
    k = d if rank is None else rank
    G = rng.normal(size=(d, k)) + 1j * rng.normal(size=(d, k))
    R = G @ dag(G)
    return R / np.trace(R)


def partial_trace(rho, dims, keep):
    """Частичный след по всем подсистемам, кроме перечисленных в keep."""
    n = len(dims)
    keep = sorted(keep)
    t = rho.reshape(dims + dims)
    letters = "abcdefghijklmnopqrstuvwxyz"
    row = list(letters[:n])
    col = list(letters[n:2 * n])
    for i in range(n):
        if i not in keep:
            col[i] = row[i]  # свёртка по следу
    out = "".join(row[i] for i in keep) + "".join(col[i] for i in keep)
    res = np.einsum("".join(row) + "".join(col) + "->" + out, t)
    dk = int(np.prod([dims[i] for i in keep]))
    return res.reshape(dk, dk)


def entropy(rho):
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-12]
    return float(-(w * np.log2(w)).sum())


def psd_sqrt(A):
    w, V = np.linalg.eigh((A + dag(A)) / 2)
    w[w < 1e-12] = 0
    return V @ np.diag(np.sqrt(w)) @ dag(V)


def fidelity(r, s):
    sr = psd_sqrt(r)
    return np.real(np.trace(psd_sqrt(sr @ s @ sr))) ** 2


def section(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


# ---------------------------------------------------------------------------
section("1. Свойства тензорного произведения")
A, B, C, D = (rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k)) for k in (2, 3, 2, 3))
assert np.allclose(np.kron(A, B) @ np.kron(C, D), np.kron(A @ C, B @ D))
assert np.allclose(dag(np.kron(A, B)), np.kron(dag(A), dag(B)))
assert np.isclose(np.trace(np.kron(A, B)), np.trace(A) * np.trace(B))
a, b, c, d = random_state(2), random_state(3), random_state(2), random_state(3)
assert np.isclose(np.vdot(np.kron(a, b), np.kron(c, d)), np.vdot(a, c) * np.vdot(b, d))
ea, eb = np.linalg.eigvals(A), np.linalg.eigvals(B)
assert np.allclose(np.sort_complex(np.linalg.eigvals(np.kron(A, B))), np.sort_complex(np.outer(ea, eb).ravel()))
print("(A(x)B)(C(x)D) = AC(x)BD, (A(x)B)^dag, Tr, <a(x)b|c(x)d>, спектр — проверено")

section("2. Порядок кубитов: N&C/np.kron против Qiskit")
psi = np.kron(ket1, ket0)  # кубит 0 в |1>, кубит 1 в |0>  ->  |10>
print("  np.kron(|1>,|0>) = базисный вектор с индексом", int(np.argmax(abs(psi))), "(= двоичное '10')")
print("  в Qiskit тот же вектор интерпретируется как q1=1, q0=0  — строка '10', но кубит 0 справа")

section("3. H^{(x)n}|x> = 2^{-n/2} sum_y (-1)^{x.y} |y>")
n = 4
Hn = kron(*[H] * n)
for x in range(2 ** n):
    ex = np.zeros(2 ** n)
    ex[x] = 1
    pred = np.array([(-1) ** bin(x & y).count("1") for y in range(2 ** n)]) / 2 ** (n / 2)
    assert np.allclose(Hn @ ex, pred)
print(f"  проверено для всех x при n = {n}")

section("4. CNOT не является произведением A (x) B: операторный ранг Шмидта = 2")
CNOT = kron(proj(ket0), I2) + kron(proj(ket1), X)


def operator_schmidt_rank(U, dA, dB):
    T = U.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(dA * dA, dB * dB)
    return np.linalg.matrix_rank(T, tol=1e-10)


print("  CNOT:", operator_schmidt_rank(CNOT, 2, 2), "; X(x)Z:", operator_schmidt_rank(kron(X, Z), 2, 2),
      "; SWAP:", operator_schmidt_rank(np.eye(4)[[0, 2, 1, 3]], 2, 2))
bell = CNOT @ kron(H, I2) @ kron(ket0, ket0)
assert np.allclose(bell, np.array([1, 0, 0, 1]) / np.sqrt(2))
print("  CNOT (H(x)I)|00> = |Phi+>")

# ---------------------------------------------------------------------------
section("5. Частичный след: определение и единственность")
dA, dB = 3, 4
rho = random_rho(dA * dB)
rA = partial_trace(rho, [dA, dB], [0])
for _ in range(200):  # Tr(rho_A M) = Tr(rho (M (x) I)) для любых M
    M = rng.normal(size=(dA, dA)) + 1j * rng.normal(size=(dA, dA))
    assert np.isclose(np.trace(rA @ M), np.trace(rho @ np.kron(M, np.eye(dB))))
r1, r2 = random_rho(2), random_rho(3)
assert np.allclose(partial_trace(np.kron(r1, r2), [2, 3], [0]), r1)
assert np.allclose(partial_trace(np.kron(r1, r2), [2, 3], [1]), r2)
print("  Tr(rho_A M) = Tr(rho_AB (M(x)I)), Tr_B(rho(x)sigma) = rho — проверено")
rho3 = random_rho(2 * 3 * 2)
assert np.allclose(partial_trace(partial_trace(rho3, [2, 3, 2], [0, 1]), [2, 3], [0]), partial_trace(rho3, [2, 3, 2], [0]))
print("  частичные следы можно брать по очереди")

section("6. Приведённые состояния: Белл, GHZ, W")
bells = {"Phi+": [1, 0, 0, 1], "Phi-": [1, 0, 0, -1], "Psi+": [0, 1, 1, 0], "Psi-": [0, 1, -1, 0]}
for k, v in bells.items():
    v = np.array(v, dtype=complex) / np.sqrt(2)
    assert np.allclose(partial_trace(proj(v), [2, 2], [0]), I2 / 2)
print("  у всех четырёх состояний Белла rho_A = I/2")
ghz = np.zeros(8, dtype=complex)
ghz[[0, 7]] = 1 / np.sqrt(2)
w = np.zeros(8, dtype=complex)
w[[1, 2, 4]] = 1 / np.sqrt(3)
print("  GHZ: rho_A =", np.round(np.diag(partial_trace(proj(ghz), [2, 2, 2], [0])).real, 4),
      "; rho_AB диагональна:", np.allclose(partial_trace(proj(ghz), [2, 2, 2], [0, 1]),
                                           np.diag([.5, 0, 0, .5])))
print("  W:   rho_A =", np.round(np.diag(partial_trace(proj(w), [2, 2, 2], [0])).real, 4))

section("7. Отсутствие сигнала: локальные операции на B не меняют rho_A")
rho = random_rho(dA * dB)
rA = partial_trace(rho, [dA, dB], [0])
U = random_unitary(dB)
assert np.allclose(partial_trace(np.kron(np.eye(dA), U) @ rho @ dag(np.kron(np.eye(dA), U)), [dA, dB], [0]), rA)
Vb = random_unitary(dB)
meas = sum(np.kron(np.eye(dA), proj(Vb[:, m])) @ rho @ np.kron(np.eye(dA), proj(Vb[:, m])) for m in range(dB))
assert np.allclose(partial_trace(meas, [dA, dB], [0]), rA)
print("  унитарный оператор на B и измерение на B без сообщения исхода: rho_A не изменилась")

# ---------------------------------------------------------------------------
section("8. Разложение Шмидта через SVD")


def schmidt(psi, dA, dB):
    Cm = psi.reshape(dA, dB)
    Um, s, Vh = np.linalg.svd(Cm)
    return s, Um, Vh  # |a_k> = Um[:, k], |b_k> = Vh[k, :]


for _ in range(200):
    dA, dB = rng.integers(2, 5), rng.integers(2, 6)
    psi = random_state(dA * dB)
    s, Um, Vh = schmidt(psi, dA, dB)
    rec = sum(s[k] * np.kron(Um[:, k], Vh[k, :]) for k in range(len(s)))
    assert np.allclose(rec, psi) and np.isclose((s ** 2).sum(), 1)
    eA = np.sort(np.linalg.eigvalsh(partial_trace(proj(psi), [dA, dB], [0])))[::-1]
    eB = np.sort(np.linalg.eigvalsh(partial_trace(proj(psi), [dA, dB], [1])))[::-1]
    r = len(s)
    assert np.allclose(eA[:r], s ** 2) and np.allclose(eB[:r], s ** 2)
    assert np.allclose(eA[r:], 0) and np.allclose(eB[r:], 0)
print("  |psi> = sum sqrt(lambda_k)|a_k>|b_k>; спектры rho_A и rho_B совпадают с lambda_k — на 200 состояниях")
ex = np.array([1, 1, 1, 0], dtype=complex) / np.sqrt(3)
s, _, _ = schmidt(ex, 2, 2)
print("  (|00>+|01>+|10>)/sqrt3: lambda =", np.round(s ** 2, 6), " (3 +- sqrt5)/6 =",
      np.round([(3 + np.sqrt(5)) / 6, (3 - np.sqrt(5)) / 6], 6),
      "; E =", round(entropy(partial_trace(proj(ex), [2, 2], [0])), 4), "бит")
assert np.allclose(s ** 2, [(3 + np.sqrt(5)) / 6, (3 - np.sqrt(5)) / 6])
ex2 = np.array([1, 1, 1, -1], dtype=complex) / 2
print("  (|00>+|01>+|10>-|11>)/2: lambda =", np.round(schmidt(ex2, 2, 2)[0] ** 2, 6), "— максимально запутанное")
# произведение <=> rho_A чистое <=> ранг Шмидта 1
prod = np.kron(random_state(3), random_state(4))
print("  произведение: ранг Шмидта =", int((schmidt(prod, 3, 4)[0] > 1e-10).sum()),
      ", Tr rho_A^2 =", round(np.trace(np.linalg.matrix_power(partial_trace(proj(prod), [3, 4], [0]), 2)).real, 10))

section("9. Локальные унитарные операторы не меняют коэффициенты Шмидта")
psi = random_state(12)
UA, UB = random_unitary(3), random_unitary(4)
assert np.allclose(schmidt(psi, 3, 4)[0], schmidt(np.kron(UA, UB) @ psi, 3, 4)[0])
print("  проверено; спектры GHZ и W на одном кубите (1/2,1/2) и (2/3,1/3) разные -> GHZ и W не связаны LU")

# ---------------------------------------------------------------------------
section("10. Очищение и его единственность")
d = 3
rho = random_rho(d)
lam, E = np.linalg.eigh(rho)
pur = sum(np.sqrt(lam[k]) * np.kron(E[:, k], np.eye(d)[k]) for k in range(d))
assert np.allclose(partial_trace(proj(pur), [d, d], [0]), rho)
Omega = np.eye(d).reshape(d * d)  # sum_i |i>|i>, ненормированный
pur2 = np.kron(psd_sqrt(rho), np.eye(d)) @ Omega  # каноническое очищение (sqrt(rho)(x)I)|Omega>
assert np.allclose(partial_trace(proj(pur2), [d, d], [0]), rho)
# единственность: pur2 = (I (x) U) pur для некоторого унитарного U на R
Cp, Cq = pur.reshape(d, d), pur2.reshape(d, d)
# Cq = Cp U^T  =>  U^T = Cp^{-1} Cq (rho полного ранга)
UT = np.linalg.solve(Cp, Cq)
assert np.allclose(UT @ dag(UT), np.eye(d))
assert np.allclose(np.kron(np.eye(d), UT.T) @ pur, pur2)
print("  два очищения rho связаны унитарным оператором на R — найден явно и проверен")
# трюк с транспонированием
M = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
assert np.allclose(np.kron(M, np.eye(d)) @ Omega, np.kron(np.eye(d), M.T) @ Omega)
N = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
assert np.isclose(Omega @ np.kron(M, N) @ Omega, np.trace(M @ N.T))
print("  (M(x)I)|Omega> = (I(x)M^T)|Omega>,  <Omega|M(x)N|Omega> = Tr(M N^T) — проверено")

section("11. Управление ансамблем (steering): измерение R готовит разные ансамбли rho_A")
phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
for name, basis in [("Z", [ket0, ket1]), ("X", [(ket0 + ket1) / np.sqrt(2), (ket0 - ket1) / np.sqrt(2)]),
                    ("случайный", list(random_unitary(2).T))]:
    ens = []
    for f in basis:
        unnorm = np.kron(I2, f.conj()[None, :]) @ phi_plus  # (I (x) <f|)|Phi+>
        p = np.vdot(unnorm, unnorm).real
        ens.append((p, unnorm / np.sqrt(p)))
        assert np.allclose(proj(unnorm / np.sqrt(p)), proj(f.conj()))  # A оказывается в |f*>
    avg = sum(p * proj(v) for p, v in ens)
    assert np.allclose(avg, I2 / 2)
    print(f"  базис {name:9s}: вероятности {np.round([p for p, _ in ens], 3)}, среднее = I/2")

# ---------------------------------------------------------------------------
section("12. Теорема Ульмана: F = max |<psi_rho|psi_sigma>|^2 по очищениям")
for _ in range(100):
    d = rng.integers(2, 4)
    rho, sig = random_rho(d), random_rho(d)
    F = fidelity(rho, sig)
    prho = np.kron(psd_sqrt(rho), np.eye(d)) @ np.eye(d).reshape(d * d)
    best = 0
    for _ in range(300):  # случайные очищения sigma
        ps = np.kron(psd_sqrt(sig), random_unitary(d)) @ np.eye(d).reshape(d * d)
        best = max(best, abs(np.vdot(prho, ps)) ** 2)
    assert best <= F + 1e-10
    # оптимальное очищение: W из SVD sqrt(rho) sqrt(sigma)
    Wm, s, Vh = np.linalg.svd(psd_sqrt(rho) @ psd_sqrt(sig))
    Wopt = dag(Vh) @ dag(Wm)  # максимизирует |Tr(sqrt rho sqrt sigma W)|
    ps_opt = np.kron(psd_sqrt(sig), Wopt.T) @ np.eye(d).reshape(d * d)
    assert np.isclose(abs(np.vdot(prho, ps_opt)) ** 2, F)
print("  случайные очищения не превышают F; оптимальное (из SVD) достигает F — на 100 парах")
F1 = fidelity(proj(ket0), I2 / 2)
print("  F(|0><0|, I/2) =", round(F1, 6))

section("13. Монотонность верности при частичном следе: F(rho_A, sigma_A) >= F(rho_AB, sigma_AB)")
for _ in range(500):
    rho, sig = random_rho(6), random_rho(6)
    assert fidelity(partial_trace(rho, [2, 3], [0]), partial_trace(sig, [2, 3], [0])) >= fidelity(rho, sig) - 1e-9
print("  проверено на 500 парах")

# ---------------------------------------------------------------------------
section("14. Типичные состояния сильно запутаны (формула Пейджа)")
for n in [4, 6, 8, 10, 12]:
    m = n // 2
    Ss = [entropy(partial_trace(proj(random_state(2 ** n)), [2 ** m, 2 ** (n - m)], [0])) for _ in range(30)]
    dA = dB = 2 ** m
    page = (sum(1 / k for k in range(dB + 1, dA * dB + 1)) - (dA - 1) / (2 * dB)) / np.log(2)
    print(f"  n={n:2d}: <S(A)> = {np.mean(Ss):.3f} бит, формула Пейджа = {page:.3f}, максимум = {m}")
    assert abs(np.mean(Ss) - page) < 0.05

print("\nВсе проверки пройдены.")
