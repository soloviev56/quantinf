"""
Лекция I.5. Протоколы на запутанности: запрет клонирования, базис Белла,
телепортация, сверхплотное кодирование, обмен запутанностью.
Численные иллюстрации ко всем утверждениям лекции.

Запуск:  python примеры.py
Требуется: numpy. Все проверки оформлены как assert.
Порядок кубитов: первый множитель — старший (как в N&C и np.kron).
"""
import numpy as np
from itertools import product

rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (X + Z) / np.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)
P0, P1 = np.outer(ket0, ket0), np.outer(ket1, ket1)


def dag(A):
    return A.conj().T


def proj(v):
    return np.outer(v, v.conj())


def kron(*ops):
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def random_state(d=2):
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    return v / np.linalg.norm(v)


def op_on(U, k, n):
    """Однокубитный U на кубите k из n."""
    return kron(*[U if i == k else I2 for i in range(n)])


def cnot(c, t, n):
    return kron(*[P0 if i == c else I2 for i in range(n)]) + kron(*[P1 if i == c else (X if i == t else I2) for i in range(n)])


def partial_trace(rho, dims, keep):
    n = len(dims)
    t = rho.reshape(dims + dims)
    L = "abcdefghijklmnop"
    row, col = list(L[:n]), list(L[n:2 * n])
    for i in range(n):
        if i not in keep:
            col[i] = row[i]
    out = "".join(row[i] for i in keep) + "".join(col[i] for i in keep)
    dk = int(np.prod([dims[i] for i in keep]))
    return np.einsum("".join(row) + "".join(col) + "->" + out, t).reshape(dk, dk)


def fid_pure(psi, rho):
    return np.vdot(psi, rho @ psi).real


def section(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


# ---------------------------------------------------------------------------
section("1. Запрет клонирования")
# (a) аргумент через скалярное произведение: U сохраняет <.|.>, а клонирование превращает c в c^2
for c in [0.0, 0.3, 0.7, 1.0]:
    print(f"  |<psi|phi>| = {c}: после клонирования было бы {c**2:.2f} -> совместимо с унитарностью: {np.isclose(c, c**2)}")
# (b) CNOT копирует базисные состояния, но не суперпозиции
CN = cnot(0, 1, 2)
for x, v in [(0, ket0), (1, ket1)]:
    assert np.allclose(CN @ kron(v, ket0), kron(v, v))
plus = (ket0 + ket1) / np.sqrt(2)
out = CN @ kron(plus, ket0)
print("  CNOT|+>|0> == |+>|+>?", np.allclose(out, kron(plus, plus)),
      "; это |Phi+>:", np.allclose(out, np.array([1, 0, 0, 1]) / np.sqrt(2)),
      "; верность «копии» с |+>:", round(fid_pure(plus, partial_trace(proj(out), [2, 2], [1])), 3))
# (c) клонирование => сигнал: Боб различил бы ансамбли Z и X
ensZ = [ket0, ket1]
ensX = [plus, (ket0 - ket1) / np.sqrt(2)]
rZ = sum(0.5 * proj(v) for v in ensZ)
rX = sum(0.5 * proj(v) for v in ensX)
rZ2 = sum(0.5 * proj(kron(v, v)) for v in ensZ)
rX2 = sum(0.5 * proj(kron(v, v)) for v in ensX)
D1 = 0.5 * np.abs(np.linalg.eigvalsh(rZ - rX)).sum()
D2 = 0.5 * np.abs(np.linalg.eigvalsh(rZ2 - rX2)).sum()
print(f"  ансамбли Z и X у Боба: D(одна копия) = {D1:.3f}, D(две «копии») = {D2:.3f} -> клонер дал бы сигнал")
assert D1 < 1e-12 and D2 > 0.4

section("2. Оптимальный универсальный клонер Бужека–Хиллери: F = 5/6")
# |0>|Q> -> sqrt(2/3)|00>|0>_c + sqrt(1/6)(|01>+|10>)|1>_c ; |1>|Q> -> sqrt(2/3)|11>|1>_c + sqrt(1/6)(|01>+|10>)|0>_c
k00, k01, k10, k11 = (kron(a, b) for a, b in product([ket0, ket1], repeat=2))
img0 = np.sqrt(2 / 3) * kron(k00, ket0) + np.sqrt(1 / 6) * kron(k01 + k10, ket1)
img1 = np.sqrt(2 / 3) * kron(k11, ket1) + np.sqrt(1 / 6) * kron(k01 + k10, ket0)
V = np.column_stack([img0, img1])  # изометрия C^2 -> C^8
assert np.allclose(dag(V) @ V, I2)
Fs = []
for _ in range(500):
    psi = random_state()
    outv = V @ psi
    r = proj(outv)
    Fs += [fid_pure(psi, partial_trace(r, [2, 2, 2], [0])), fid_pure(psi, partial_trace(r, [2, 2, 2], [1]))]
print("  верность каждой копии: min =", round(min(Fs), 10), " max =", round(max(Fs), 10), " (5/6 =", round(5 / 6, 10), ")")
assert np.allclose(Fs, 5 / 6)

# ---------------------------------------------------------------------------
section("3. Базис Белла: |b_xy> = CNOT (H(x)I)|xy> = (Z^x X^y (x) I)|Phi+>")
Phi_p = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
bell = {}
names = {(0, 0): "Phi+", (1, 0): "Phi-", (0, 1): "Psi+", (1, 1): "Psi-"}
for x, y in product([0, 1], repeat=2):
    b = CN @ kron(H, I2) @ kron([ket0, ket1][x], [ket0, ket1][y])
    b2 = kron(np.linalg.matrix_power(Z, x) @ np.linalg.matrix_power(X, y), I2) @ Phi_p
    assert np.allclose(b, b2)
    bell[(x, y)] = b
    print(f"  b_{x}{y} = {names[(x, y)]:4s}:", np.round(b * np.sqrt(2), 3).real)
G = np.array([[np.vdot(bell[a], bell[b]) for b in bell] for a in bell])
assert np.allclose(G, np.eye(4))
# измерение в базисе Белла = обратная схема + измерение Z
Ubm = kron(H, I2) @ CN
for (x, y), b in bell.items():
    assert np.allclose(Ubm @ b, kron([ket0, ket1][x], [ket0, ket1][y]))
print("  ортонормированный базис; (H(x)I)CNOT переводит |b_xy> в |xy>")

# ---------------------------------------------------------------------------
section("4. Телепортация: разложение по базису Белла и коррекции")
# кубиты: 0 = C (вход), 1 = A (Алиса), 2 = B (Боб)
for _ in range(200):
    psi = random_state()
    full = kron(psi, Phi_p)
    for (x, y), b in bell.items():
        # (<b_xy|_{CA} (x) I_B)|psi>|Phi+>
        bob = kron(b.conj()[None, :], I2) @ full
        p = np.vdot(bob, bob).real
        assert np.isclose(p, 0.25)
        bob /= np.sqrt(p)
        corr = np.linalg.matrix_power(Z, x) @ np.linalg.matrix_power(X, y)  # сначала X^y, потом Z^x
        assert np.isclose(abs(np.vdot(psi, corr @ bob)), 1)
print("  каждый исход Белла с вероятностью 1/4; коррекция Z^x X^y восстанавливает |psi> — на 200 состояниях")

# схема N&C: CNOT(0->1), H(0), измерение 0 (m1) и 1 (m2), Боб: X^{m2}, затем Z^{m1}
circ = op_on(H, 0, 3) @ cnot(0, 1, 3)
for _ in range(100):
    psi = random_state()
    st = circ @ kron(psi, Phi_p)
    bob_states, probs = [], []
    for m1, m2 in product([0, 1], repeat=2):
        Pm = kron([P0, P1][m1], [P0, P1][m2], I2)
        v = Pm @ st
        p = np.vdot(v, v).real
        probs.append(p)
        bobv = v.reshape(2, 2, 2)[m1, m2, :] / np.sqrt(p)
        bob_states.append((p, bobv))
        fixed = np.linalg.matrix_power(Z, m1) @ np.linalg.matrix_power(X, m2) @ bobv
        assert np.isclose(abs(np.vdot(psi, fixed)), 1)
    # до получения битов Боб видит I/2
    rho_bob = sum(p * proj(v) for p, v in bob_states)
    assert np.allclose(rho_bob, I2 / 2) and np.allclose(probs, 0.25)
print("  схема CNOT, H, измерения, X^{m2} Z^{m1}: верность 1; до получения битов rho_B = I/2; исходы равновероятны")

# отложенное измерение: CX(1->2), CZ(0->2) вместо классического управления
CZ02 = kron(P0, I2, I2) + kron(P1, I2, Z)
st = cnot(1, 2, 3) @ circ @ kron(psi := random_state(), Phi_p)
st = CZ02 @ st
assert np.isclose(fid_pure(psi, partial_trace(proj(st), [2, 2, 2], [2])), 1)
print("  вариант с отложенным измерением (квантовые CX и CZ) — верность 1")

section("5. Шумный ресурс: изотропное состояние p|Phi+><Phi+| + (1-p) I/4 -> F = (1+p)/2")


def teleport_rho(psi, resource):
    """Телепортация с произвольным двухкубитным ресурсом (матрица плотности). Возвращает средн. верность."""
    rho = kron(proj(psi), resource)
    rho = circ @ rho @ dag(circ)
    out = np.zeros((2, 2), dtype=complex)
    for m1, m2 in product([0, 1], repeat=2):
        Pm = kron([P0, P1][m1], [P0, P1][m2], I2)
        r = Pm @ rho @ Pm
        rb = partial_trace(r, [2, 2, 2], [2])
        C = np.linalg.matrix_power(Z, m1) @ np.linalg.matrix_power(X, m2)
        out += C @ rb @ dag(C)
    return fid_pure(psi, out)


for p in [0, 1 / 3, 0.5, 0.8, 1]:
    res = p * proj(Phi_p) + (1 - p) * np.eye(4) / 4
    F = np.mean([teleport_rho(random_state(), res) for _ in range(300)])
    print(f"  p = {p:.3f}: средняя верность = {F:.4f}, (1+p)/2 = {(1 + p) / 2:.4f}")
    assert abs(F - (1 + p) / 2) < 1e-10  # для изотропного ресурса верность не зависит от psi

section("6. Классический предел: «измерить в Z и приготовить» даёт 2/3")
Fs = []
for _ in range(200000 // 100):
    psi = random_state()
    p0 = abs(psi[0]) ** 2
    Fs.append(p0 * p0 + (1 - p0) * (1 - p0))  # с вер. p0 готовим |0> (верность p0), иначе |1>
print("  средняя верность по мере Хаара =", round(np.mean(Fs), 4), "(точно 2/3)")
assert abs(np.mean(Fs) - 2 / 3) < 0.02

# ---------------------------------------------------------------------------
section("7. Сверхплотное кодирование: 2 бита за 1 кубит")
for p in [1.0, 0.8, 0.5]:
    res = p * proj(Phi_p) + (1 - p) * np.eye(4) / 4
    M = np.zeros((4, 4))
    for i, (x, y) in enumerate(product([0, 1], repeat=2)):
        enc = kron(np.linalg.matrix_power(Z, x) @ np.linalg.matrix_power(X, y), I2)
        r = enc @ res @ dag(enc)
        for j, (a, b) in enumerate(product([0, 1], repeat=2)):
            M[i, j] = fid_pure(bell[(a, b)], r)
    assert np.allclose(M.sum(axis=1), 1)
    pj = M.mean(axis=0)
    Imut = sum(0.25 * M[i, j] * np.log2(M[i, j] / pj[j]) for i in range(4) for j in range(4) if M[i, j] > 1e-15)
    print(f"  p = {p}: P(верно) = {np.mean(np.diag(M)):.3f}, взаимная информация = {Imut:.3f} бит")
    if p == 1.0:
        assert np.allclose(M, np.eye(4)) and np.isclose(Imut, 2)
# одиночный перехваченный кубит ничего не несёт
for (x, y) in product([0, 1], repeat=2):
    enc = kron(np.linalg.matrix_power(Z, x) @ np.linalg.matrix_power(X, y), I2) @ Phi_p
    assert np.allclose(partial_trace(proj(enc), [2, 2], [0]), I2 / 2)
print("  отправленный кубит в одиночку всегда в состоянии I/2 — перехват не даёт информации")

# ---------------------------------------------------------------------------
section("8. Обмен запутанностью: (1,2),(3,4) -> измерение Белла на (2,3) -> (1,4) запутаны")
state = kron(Phi_p, Phi_p)  # кубиты 0,1,2,3
for (x, y), b in bell.items():
    proj23 = kron(I2, b.conj()[None, :], I2)  # <b_xy| на кубитах 1,2
    v = proj23 @ state  # остаются кубиты 0 и 3
    p = np.vdot(v, v).real
    v /= np.sqrt(p)
    which = [k for k, bb in bell.items() if np.isclose(abs(np.vdot(bb, v)), 1)]
    print(f"  исход {names[(x, y)]:4s} (p = {p:.2f}) -> пара (1,4) в состоянии {names[which[0]]}")
    assert np.isclose(p, 0.25) and len(which) == 1

section("9. Обмен запутанностью с шумными парами: p1, p2 -> p1*p2")


def iso(p):
    return p * proj(Phi_p) + (1 - p) * np.eye(4) / 4


for p1, p2 in [(0.9, 0.9), (0.8, 0.7), (0.95, 0.5)]:
    rho = kron(iso(p1), iso(p2))
    out = np.zeros((4, 4), dtype=complex)
    for (x, y), b in bell.items():
        Pm = kron(I2, proj(b), I2)
        r = partial_trace(Pm @ rho @ Pm, [2, 2, 2, 2], [0, 3])
        # коррекция на кубите 4, переводящая результат в Phi+
        C = np.linalg.matrix_power(Z, x) @ np.linalg.matrix_power(X, y)
        Cf = kron(I2, C)
        r2 = Cf @ r @ dag(Cf)
        out += r2
    Fphi = fid_pure(Phi_p, out)
    p_eff = (4 * Fphi - 1) / 3
    print(f"  p1={p1}, p2={p2}: p_out = {p_eff:.4f}, p1*p2 = {p1 * p2:.4f}")
    assert np.isclose(p_eff, p1 * p2)
print("  после k звеньев p^k: верность падает экспоненциально -> нужна дистилляция")

# ---------------------------------------------------------------------------
section("10. Телепортация гейта T: ресурс (I(x)T)|Phi+>, коррекции — клиффордовы")
res_T = kron(I2, T) @ Phi_p
for _ in range(200):
    psi = random_state()
    st = circ @ kron(psi, res_T)
    for m1, m2 in product([0, 1], repeat=2):
        v = (kron([P0, P1][m1], [P0, P1][m2], I2) @ st).reshape(2, 2, 2)[m1, m2, :]
        v /= np.linalg.norm(v)
        # Боб имеет T X^{m2} Z^{m1} psi;  T X = e^{-i pi/4} S X T
        corr = np.linalg.matrix_power(Z, m1) @ np.linalg.matrix_power(dag(S @ X), m2)
        assert np.isclose(abs(np.vdot(T @ psi, corr @ v)), 1)
assert np.allclose(T @ X @ dag(T), np.exp(-1j * np.pi / 4) * S @ X)
print("  T X T^dag = e^{-i pi/4} S X; коррекции (S X)^dag и Z — из группы Клиффорда; результат T|psi> — на 200 состояниях")

print("\nВсе проверки пройдены.")
