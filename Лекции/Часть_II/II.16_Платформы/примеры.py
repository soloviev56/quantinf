"""
Лекция II.16. Физические платформы и состояние области.

Проверяется: тепловое заселение кубита (почему милликельвины); трансмон —
численная диагонализация гамильтониана куперовской коробки в зарядовом
базисе: частота перехода против sqrt(8 E_J E_C) - E_C, ангармонизм,
экспоненциальное подавление зарядовой дисперсии (сравнение с асимптотикой
Коха и др.); ионы — гейт Мёльмера–Сёренсена с явным учётом фононной моды
(усечённый базис Фока): XX-поворот, нечувствительность к тепловому
состоянию моды; нейтральные атомы — ридберговская блокада (коллективная
частота sqrt2 Omega) и CZ-протокол Якша (pi - 2pi - pi), ошибка ~ (Omega/V)^2;
спиновые кубиты — обменное взаимодействие и sqrt(SWAP); фотоника —
вероятность выживания n фотонов; бюджет ошибок схемы и «эффективный объём»;
ресурсы поверхностного кода при разных физических ошибках (II.7, (5.1)).

Соглашения (Лекции/Обозначения.md, §3, §19, §28): hbar = 1 в гамильтонианах
(энергии в единицах частоты); E_J, E_C — энергии Джозефсона и зарядовая;
Omega — частота Раби; V — ридберговский сдвиг; p — ошибка двухкубитного
гейта; p_L = A (p/p_th)^((d+1)/2), A = 0.1, p_th = 1 %.

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
from math import sqrt, pi, exp, factorial, log, ceil
import numpy as np
from scipy.linalg import expm

T_START = time.time()
HPL, KB = 6.62607015e-34, 1.380649e-23


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
I2 = np.eye(2, dtype=complex)

# ===========================================================================
section("1. Тепловое заселение: почему милликельвины")
def nbar(f, T):
    x = HPL * f / (KB * T)
    return 1 / np.expm1(x)

f = 5e9
print(f"  hf/k_B для f = 5 ГГц: {HPL * f / KB * 1e3:.1f} мК")
TH = {}
for T in (0.01, 0.02, 0.05, 0.1, 4.0, 300.0):
    TH[T] = nbar(f, T)
    print(f"    T = {T * 1e3 if T < 1 else T:>7.1f} {'мК' if T < 1 else 'К '}: n_th = {TH[T]:.2e}")
assert TH[0.02] < 1e-5 and TH[4.0] > 10
T3 = HPL * f / KB / log(1 + 1e3)
assert abs(nbar(f, T3) - 1e-3) < 1e-12
print(f"  n_th = 1e-3 при T = {T3 * 1e3:.1f} мК")
# оптический переход (ионы, атомы) при комнатной температуре
f_opt = 3e8 / 800e-9
print(f"  оптический переход 800 нм при 300 К: n_th = {nbar(f_opt, 300):.1e} — кубит «холодный» сам по себе")
assert nbar(f_opt, 300) < 1e-20

# ===========================================================================
section("2. Трансмон: гамильтониан куперовской коробки")
def transmon(EJ, EC, ng, N=40):
    n = np.arange(-N, N + 1)
    H = np.diag(4 * EC * (n - ng) ** 2) - EJ / 2 * (np.eye(2 * N + 1, k=1) + np.eye(2 * N + 1, k=-1))
    return np.linalg.eigvalsh(H)

EC = 1.0
TR = {}
print("  E_J/E_C |  f01/E_C  | sqrt(8EJEC)-EC | (f12-f01)/E_C | дисперсия e0, e1 (в E_C)     | Кох и др. e0, e1")
for r in (1, 5, 10, 20, 50, 100):
    e0, e5 = transmon(r, EC, 0.0), transmon(r, EC, 0.5)
    f01, f12 = e0[1] - e0[0], e0[2] - e0[1]
    d0, d1 = abs(e0[0] - e5[0]), abs(e0[1] - e5[1])
    koch = lambda m: EC * 2 ** (4 * m + 5) / factorial(m) * sqrt(2 / pi) * (r / 2) ** (m / 2 + 0.75) * exp(-sqrt(8 * r))
    TR[r] = (f01, f12 - f01, d0, d1, koch(0), koch(1))
    print(f"  {r:7d} | {f01:9.4f} | {sqrt(8 * r) - 1:14.4f} | {f12 - f01:13.4f} | {d0:.3e}, {d1:.3e}       | {koch(0):.3e}, {koch(1):.3e}")
# проверки: асимптотика частоты, ангармонизм -> -E_C, отношение к формуле Коха -> 1
assert abs(TR[50][0] - (sqrt(400) - 1)) / TR[50][0] < 0.005
assert -1.2 < TR[50][1] < -1.0 and -1.15 < TR[100][1] < -1.0
assert 0.9 < TR[50][2] / TR[50][4] < 1.0 and 0.9 < TR[100][2] / TR[100][4] < 1.0
assert TR[100][2] < 1e-9 and TR[1][2] > 0.1
# наклон ln(дисперсии) по sqrt(8 EJ/EC) ~ -1
sl_raw = (log(TR[100][2]) - log(TR[50][2])) / (sqrt(800) - sqrt(400))
sl = (log(TR[100][2] / 100 ** 0.75) - log(TR[50][2] / 50 ** 0.75)) / (sqrt(800) - sqrt(400))
assert -1.05 < sl < -0.95
print(f"  наклон ln(e0) по sqrt(8E_J/E_C): {sl_raw:.3f}; без степенного множителя (E_J/E_C)^(3/4): {sl:.3f} (~ -1)")
print("  зарядовая дисперсия подавлена экспоненциально, а ангармонизм убывает к -E_C лишь медленно")
ECh = 0.25   # ГГц
print(f"  пример: E_C/h = {ECh} ГГц, E_J/E_C = 50 -> f01 = {TR[50][0] * ECh:.3f} ГГц, ангармонизм {TR[50][1] * ECh * 1e3:.0f} МГц")
# импульс для подавления утечки в |2>: длительность ~ несколько 1/|alpha|
alpha_hz = abs(TR[50][1]) * ECh * 1e9
print(f"  1/|alpha| = {1 / alpha_hz * 1e9:.1f} нс — масштаб длительности однокубитного гейта без утечки")

# ===========================================================================
section("3. Ионы: гейт Мёльмера–Сёренсена")
Nf = 30
a = np.diag(np.sqrt(np.arange(1, Nf)), 1).astype(complex)
Sx = np.kron(X, I2) + np.kron(I2, X)
delta = 1.0
g = delta / 4                     # условие XX(pi/4) за одну петлю t = 2pi/delta
T = 2 * pi / delta
steps = 3000
dt = T / steps
U = np.eye(4 * Nf, dtype=complex)
for k in range(steps):
    t = (k + 0.5) * dt
    Hm = a * np.exp(-1j * delta * t)
    Hm = Hm + Hm.conj().T
    U = expm(-1j * g * np.kron(Sx, Hm) * dt) @ U
target = expm(1j * pi / 4 * np.kron(X, X))
bell = target @ np.array([1, 0, 0, 0], complex)
assert np.allclose(bell, np.array([1, 0, 0, 1j]) / sqrt(2))
MS = {}
for nb in (0.0, 0.5, 2.0):
    p = np.array([(nb / (nb + 1)) ** n / (nb + 1) for n in range(Nf)]) if nb > 0 else np.eye(Nf)[0]
    F = 0.0
    wsum = 0.0
    for n in range(16):
        if p[n] < 1e-8:
            continue
        out = U @ np.kron(np.array([1, 0, 0, 0], complex), np.eye(Nf)[n])
        r = out.reshape(4, Nf)
        rs = r @ r.conj().T
        F += p[n] * np.real(bell.conj() @ rs @ bell)
        wsum += p[n]
    MS[nb] = F / wsum
    print(f"  тепловое состояние моды n = {nb}: верность |00> -> (|00> + i|11>)/sqrt2 = {MS[nb]:.8f}")
assert all(v > 1 - 1e-6 for v in MS.values())
# ошибка при неточной отстройке: петля не замыкается
U2 = np.eye(4 * Nf, dtype=complex)
d2 = 1.05 * delta
for k in range(steps):
    t = (k + 0.5) * dt
    Hm = a * np.exp(-1j * d2 * t)
    Hm = Hm + Hm.conj().T
    U2 = expm(-1j * g * np.kron(Sx, Hm) * dt) @ U2
out = U2 @ np.kron(np.array([1, 0, 0, 0], complex), np.eye(Nf)[0])
r = out.reshape(4, Nf)
F_det = np.real(bell.conj() @ (r @ r.conj().T) @ bell)
assert 0.9 < F_det < 0.9999
print(f"  отстройка на 5 % (петля в фазовом пространстве не замкнута): верность {F_det:.4f}")

# ===========================================================================
section("4. Нейтральные атомы: ридберговская блокада и CZ")
# два атома, уровни g, r; H = Omega/2 sum (|r><g| + h.c.) + V |rr><rr|
def two_atom_H(Om, V):
    sp = np.array([[0, 0], [1, 0]], complex)          # |r><g| в базисе (g, r)
    hx = sp + sp.conj().T
    H = Om / 2 * (np.kron(hx, I2) + np.kron(I2, hx))
    H[3, 3] += V
    return H

Om, V = 1.0, 200.0
H = two_atom_H(Om, V)
ts = np.linspace(0, 2 * pi / (sqrt(2) * Om), 2001)
W = np.array([0, 1, 1, 0]) / sqrt(2)
pw = [abs(W @ expm(-1j * H * t) @ np.array([1, 0, 0, 0])) ** 2 for t in ts]
t_max = ts[int(np.argmax(pw))]
assert abs(t_max - pi / (sqrt(2) * Om)) < 2e-3 and max(pw) > 0.999
print(f"  |gg> -> симметричное W за t = {t_max:.4f} = pi/(sqrt2 Omega) = {pi / (sqrt(2) * Om):.4f}; P(rr) мала при V >> Omega")

# CZ Якша: уровни атома (0, 1, r); |1> <-> |r> с частотой Omega
def jaksch(Om, V):
    d = 3
    sp = np.zeros((3, 3), complex)
    sp[2, 1] = 1                                        # |r><1|
    hx = sp + sp.conj().T
    I3 = np.eye(3)
    rr = np.zeros((9, 9))
    rr[8, 8] = V                                        # |r r>
    Hc = Om / 2 * np.kron(hx, I3) + rr
    Ht = Om / 2 * np.kron(I3, hx) + rr
    Upi = expm(-1j * Hc * pi / Om)
    U2pi = expm(-1j * Ht * 2 * pi / Om)
    return Upi @ U2pi @ Upi

idx = {"00": 0, "01": 1, "10": 3, "11": 4}
CZ = {}
for Vv in (10.0, 30.0, 100.0, 300.0):
    Uj = jaksch(1.0, Vv)
    M = np.array([[Uj[idx[i], idx[j]] for j in ("00", "01", "10", "11")] for i in ("00", "01", "10", "11")])
    ideal = np.diag([1, -1, -1, -1]).astype(complex)
    Fg = (abs(np.trace(ideal.conj().T @ M)) ** 2 + np.trace(M.conj().T @ M).real) / 20   # средняя верность (d = 4)
    CZ[Vv] = 1 - Fg
    print(f"  V/Omega = {Vv:5.0f}: ошибка гейта diag(1,-1,-1,-1) (= CZ с точностью до Z x Z) = {CZ[Vv]:.2e}")
assert CZ[300.0] < 1e-4 and CZ[10.0] > CZ[100.0] > CZ[300.0]
sl = log(CZ[300.0] / CZ[30.0]) / log(10)
assert -2.3 < sl < -1.7
print(f"  наклон log(ошибки) / log(V/Omega) = {sl:.2f} (~ -2)")
# diag(1,-1,-1,-1) = CZ (Z x Z)
assert np.allclose(np.diag([1, -1, -1, -1]), np.diag([1, 1, 1, -1]) @ np.kron(Z, Z).real)

# ===========================================================================
section("5. Спиновые кубиты: обмен и sqrt(SWAP)")
J = 1.0
Hex = J / 4 * (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z))
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], complex)
Uh = expm(-1j * Hex * pi / (2 * J))
# с точностью до глобальной фазы Uh^2 = SWAP
ph = (Uh @ Uh)[0, 0]
assert np.allclose(Uh @ Uh, ph * SWAP)
psi = Uh @ np.array([0, 1, 0, 0], complex)            # |01> -> запутанное
rhoA = np.einsum("ab,cb->ac", psi.reshape(2, 2), psi.reshape(2, 2).conj())
pur = np.trace(rhoA @ rhoA).real
assert abs(pur - 0.5) < 1e-12
print(f"  exp(-i J t sigma.sigma/4) при J t = pi/2 — sqrt(SWAP): квадрат = SWAP; |01> -> максимально запутанное (чистота rho_A = {pur:.2f})")

# ===========================================================================
section("6. Фотоника: потери")
PH = {}
for eta in (0.99, 0.95, 0.9):
    for n in (10, 50, 100):
        PH[(eta, n)] = eta ** n
    print(f"  выживание всех n фотонов при eta = {eta}: n = 10: {PH[(eta, 10)]:.3f}, n = 50: {PH[(eta, 50)]:.3e}, n = 100: {PH[(eta, 100)]:.3e}")
assert PH[(0.9, 100)] < 1e-4

# ===========================================================================
section("7. Бюджет ошибок схемы")
def circuit_fid(n, depth, p2, p1=0.0, pm=0.0):
    # слой: n/2 двухкубитных гейтов, n однокубитных; в конце измерение всех кубитов
    return (1 - p2) ** (n // 2 * depth) * (1 - p1) ** (n * depth) * (1 - pm) ** n
BUD = {}
for p2 in (1e-2, 5e-3, 1e-3, 1e-4):
    n_gates = 1 / p2
    BUD[p2] = n_gates
    print(f"  p2 = {p2:.0e}: ~{n_gates:.0f} двухкубитных гейтов до верности 1/e; схема 100 кубитов x глубина 20: F = {circuit_fid(100, 20, p2):.2e}")
assert circuit_fid(100, 20, 1e-3) > 0.3 and circuit_fid(100, 20, 1e-2) < 1e-4
# «1000 кубитов с 99,5 %»: один слой из 500 двухкубитных гейтов
F1 = circuit_fid(1000, 1, 5e-3)
print(f"  1000 кубитов, p2 = 0,5 %: один слой из 500 двухкубитных гейтов -> F = {F1:.3f}")
assert abs(F1 - 0.995 ** 500) < 1e-12
# эффективный объём: наибольшее n, при котором схема n x n имеет F >= 1/e (грубая оценка)
EFF = {}
for p2 in (1e-2, 5e-3, 1e-3):
    n = 2
    while circuit_fid(n + 2, n + 2, p2) >= exp(-1):
        n += 2
    EFF[p2] = n
    print(f"  p2 = {p2:.0e}: квадратная схема n x n с F >= 1/e при n <= {n}")
assert EFF[1e-3] > EFF[1e-2]

# ===========================================================================
section("8. Ресурсы поверхностного кода (II.7, (5.1))")
A, pth = 0.1, 1e-2
RES = {}
for p in (5e-3, 2e-3, 1e-3, 5e-4, 1e-4):
    d = 3
    while A * (p / pth) ** ((d + 1) / 2) > 1e-12 * (1 + 1e-9):
        d += 2
    RES[p] = (d, 2 * d * d - 1)
    print(f"  p = {p:.0e} (Lambda = {pth / p:4.0f}): p_L <= 1e-12 при d = {d}, физических кубитов на логический {2 * d * d - 1}")
assert RES[1e-3] == (21, 881)
assert RES[5e-3][1] > 10 * RES[1e-3][1]

# ===========================================================================
section("9. Ответы к задачам")
print(f"  1: n_th(5 ГГц): 20 мК -> {TH[0.02]:.1e}, 4 К -> {TH[4.0]:.1f}; n_th = 1e-3 при {T3 * 1e3:.1f} мК")
print(f"  2: E_J/E_C = 50: f01 = {TR[50][0]:.3f} E_C (sqrt(8*50) - 1 = 19), alpha = {TR[50][1]:.3f} E_C")
print(f"  3: MS: g = delta/4; верность при n = 0.5: {MS[0.5]:.6f}")
print(f"  4: блокада: t = pi/(sqrt2 Omega); ошибка CZ при V/Omega = 30: {CZ[30.0]:.2e}, 100: {CZ[100.0]:.2e}")
print(f"  6: 1000 кубитов, 0.5 %: F = {F1:.3f}; p = 1e-3: {BUD[1e-3]:.0f} гейтов")
print(f"  7: d и кубиты: p = 2e-3 -> {RES[2e-3]}, p = 1e-3 -> {RES[1e-3]}, p = 1e-4 -> {RES[1e-4]}")

print(f"\nВсе проверки пройдены. Время работы: {time.time() - T_START:.1f} с")
