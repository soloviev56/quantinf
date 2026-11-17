"""
Лекция I.12. Квантовое преобразование Фурье и оценка фазы: определение QFT,
связь с ДПФ и np.fft, схема из H и управляемых фазовых гейтов, приближённое QFT,
периодические состояния, оценка фазы (QPE): распределение исходов, точность,
вероятность успеха, суперпозиция собственных векторов, итеративный QPE с одним
вспомогательным кубитом, тест Адамара и swap-тест.

Запуск:  python примеры.py
Требуется: numpy. Все проверки оформлены как assert.
Порядок кубитов: первый множитель — старший (как в N&C и np.kron).
"""
import numpy as np
from math import ceil, log2, pi, sqrt

rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (X + Z) / np.sqrt(2)
Sdg = np.diag([1, -1j])


def P(phi):
    return np.diag([1, np.exp(1j * phi)])


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def kron(*ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def haar_state(d):
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    return v / np.linalg.norm(v)


def haar_unitary(d):
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def dft_matrix(N):
    """F_N[k, j] = e^{2 pi i jk/N}/sqrt(N) — QFT|j> = sum_k F[k, j]|k>."""
    j = np.arange(N)
    return np.exp(2j * pi * np.outer(j, j) / N) / np.sqrt(N)


# ---- простейший симулятор вектора состояния (кубиты 0..n-1, кубит 0 — старший)
def apply_1q(psi, U, q, n):
    """psi: вектор (2^n,) или пачка столбцов (2^n, K)."""
    sh = psi.shape
    psi = psi.reshape([2] * n + [-1])
    psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)
    return psi.reshape(sh)


def apply_cphase(psi, phi, c, t, n):
    sh = psi.shape
    psi = psi.reshape([2] * n + [-1]).copy()
    idx = [slice(None)] * (n + 1)
    idx[c] = 1
    idx[t] = 1
    psi[tuple(idx)] *= np.exp(1j * phi)
    return psi.reshape(sh)


def apply_swap(psi, a, b, n):
    sh = psi.shape
    psi = psi.reshape([2] * n + [-1])
    return np.swapaxes(psi, a, b).reshape(sh)


def qft_gates(n, m=None, swaps=True):
    """Список гейтов схемы QFT (N&C, рис. 5.1). m — максимальное число управляемых
    поворотов на кубит (приближённое QFT: оставляем R_k при k <= m+1); m=None — точное."""
    gates = []
    for l in range(n):
        gates.append(("H", l))
        for k in range(2, n - l + 1):              # R_k: управление — кубит l+k-1
            if m is None or k <= m + 1:
                gates.append(("CP", 2 * pi / 2 ** k, l + k - 1, l))
    if swaps:
        for l in range(n // 2):
            gates.append(("SWAP", l, n - 1 - l))
    return gates


def run(gates, psi, n, inverse=False):
    seq = gates[::-1] if inverse else gates
    for g in seq:
        if g[0] == "H":
            psi = apply_1q(psi, H, g[1], n)
        elif g[0] == "CP":
            psi = apply_cphase(psi, -g[1] if inverse else g[1], g[2], g[3], n)
        elif g[0] == "SWAP":
            psi = apply_swap(psi, g[1], g[2], n)
    return psi


def circuit_matrix(gates, n, inverse=False):
    return run(gates, np.eye(2 ** n, dtype=complex), n, inverse)


# ---------------------------------------------------------------------------
section("1. QFT как матрица: унитарность, связь с np.fft, F^2, F^4, сдвиг")
for n in range(1, 8):
    N = 2 ** n
    F = dft_matrix(N)
    assert np.allclose(F.conj().T @ F, np.eye(N))
    assert np.allclose(F, np.fft.ifft(np.eye(N), axis=0) * np.sqrt(N))      # знак: ifft * sqrt(N)
    assert np.allclose(F.conj(), np.fft.fft(np.eye(N), axis=0) / np.sqrt(N))
    R = np.zeros((N, N)); R[(-np.arange(N)) % N, np.arange(N)] = 1           # |j> -> |-j mod N>
    assert np.allclose(F @ F, R) and np.allclose(np.linalg.matrix_power(F, 4), np.eye(N))
print("  F_N = sqrt(N) * ifft (а не fft!); F^dag F = I; F^2|j> = |-j mod N>; F^4 = I  (N = 2..128)")
assert np.allclose(dft_matrix(2), H)
F4 = dft_matrix(4) * 2
assert np.allclose(F4, [[1, 1, 1, 1], [1, 1j, -1, -1j], [1, -1, 1, -1], [1, -1j, -1, 1j]])
print("  N = 2: F = H;  N = 4: 2F = [[1,1,1,1],[1,i,-1,-i],[1,-1,1,-1],[1,-i,-1,i]]")
for N in (4, 8, 16, 32):
    ev = np.linalg.eigvals(dft_matrix(N))
    cnt = {lab: int(np.sum(np.isclose(ev, val))) for lab, val in (("1", 1), ("-1", -1), ("i", 1j), ("-i", -1j))}
    assert sum(cnt.values()) == N
    print(f"  N = {N:2d}: кратности собственных значений {cnt}")
# сдвиг: F^dag Shift F диагонален (теорема о свёртке)
N = 16
Sh = np.roll(np.eye(N), 1, axis=0)                  # |j> -> |j+1 mod N>
D = dft_matrix(N).conj().T @ Sh @ dft_matrix(N)
assert np.allclose(D, np.diag(np.diag(D)))
assert np.allclose(np.diag(D), np.exp(-2j * pi * np.arange(N) / N))
print("  F^dag Shift F = diag(e^{-2 pi i k/N}): сдвиг по Z_N переходит в фазу (теорема о свёртке)")

# ---------------------------------------------------------------------------
section("2. Схема QFT: H + управляемые R_k + SWAP; формула произведения")
for n in range(1, 9):
    g = qft_gates(n)
    U = circuit_matrix(g, n)
    assert np.allclose(U, dft_matrix(2 ** n))
    nH = sum(1 for x in g if x[0] == "H"); nCP = sum(1 for x in g if x[0] == "CP"); nS = sum(1 for x in g if x[0] == "SWAP")
    assert (nH, nCP, nS) == (n, n * (n - 1) // 2, n // 2)
    Unos = circuit_matrix(qft_gates(n, swaps=False), n)
    rev = np.array([int(format(j, f"0{n}b")[::-1], 2) for j in range(2 ** n)])
    assert np.allclose(Unos[rev, :], dft_matrix(2 ** n))           # без SWAP — QFT с обращённым порядком выхода
print("  n = 1..8: схема совпадает с матрицей F; гейтов: n H, n(n-1)/2 управляемых фаз, floor(n/2) SWAP")
print("  без финальных SWAP схема даёт QFT с обращённым порядком выходных кубитов")
for n in (10, 20, 50):
    print(f"    n = {n}: H = {n}, CP = {n * (n - 1) // 2}, SWAP = {n // 2}")


def product_form(j, n):
    """QFT|j_1..j_n> = (x)_{l=1}^n (|0> + e^{2 pi i 0.j_{n-l+1}..j_n}|1>)/sqrt2."""
    bits = [(j >> (n - 1 - k)) & 1 for k in range(n)]
    out = np.array([1.0 + 0j])
    for l in range(1, n + 1):
        frac = sum(bits[n - l + i] / 2 ** (i + 1) for i in range(l))        # 0.j_{n-l+1} ... j_n
        out = np.kron(out, np.array([1, np.exp(2j * pi * frac)]) / np.sqrt(2))
    return out


for n in (3, 5, 7):
    F = dft_matrix(2 ** n)
    for j in rng.integers(0, 2 ** n, 6):
        assert np.allclose(product_form(int(j), n), F[:, j])
print("  формула произведения (2.1) проверена для случайных j при n = 3, 5, 7: QFT|j> — произведение состояний")
v = product_form(5, 3) * np.sqrt(8)
print("  пример j = 101 (n = 3): множители (|0> + e^{2 pi i x}|1>) с x = 0.1 = 1/2, 0.01 = 1/4, 0.101 = 5/8")
assert np.allclose(v, np.kron(np.kron([1, np.exp(2j * pi * 0.5)], [1, np.exp(2j * pi * 0.25)]), [1, np.exp(2j * pi * 5 / 8)]))

# ---------------------------------------------------------------------------
section("3. Приближённое QFT (n = 10): ошибка и число гейтов от m")
n = 10
N = 2 ** n
F = dft_matrix(N)


def dropped_bound(n, m):
    """Сумма ||CR_k - I|| = 2 sin(pi/2^k) по отброшенным поворотам (k > m+1)."""
    return sum((n - k + 1) * 2 * np.sin(pi / 2 ** k) for k in range(m + 2, n + 1))


print("   m   CP-гейтов   ||F - A_m||_inf   оценка (3.1)   средняя верность гейта")
aqft = {}
for m in range(1, n):
    g = qft_gates(n, m)
    A = circuit_matrix(g, n)
    err = np.linalg.norm(F - A, 2)
    favg = (abs(np.trace(F.conj().T @ A)) ** 2 + N) / (N * (N + 1))
    ncp = sum(1 for x in g if x[0] == "CP")
    assert ncp == sum(min(n - 1 - l, m) for l in range(n))
    bnd = dropped_bound(n, m)
    assert err <= bnd + 1e-9
    aqft[m] = (ncp, err, bnd, favg)
    print(f"  {m:2d}     {ncp:3d}         {err:.4f}          {bnd:.4f}        {favg:.6f}")
assert aqft[9][1] < 1e-10
# асимптотика: m ~ log2(n/eps) даёт ошибку <= eps
for n_, eps in ((100, 1e-2), (1000, 1e-2), (1000, 1e-4)):
    m = next(m for m in range(1, n_) if dropped_bound(n_, m) <= eps)
    ncp = sum(min(n_ - 1 - l, m) for l in range(n_))
    print(f"  n = {n_:4d}, eps = {eps:.0e}: достаточно m = {m:2d} (log2(n/eps) = {log2(n_ / eps):.1f}), CP: {ncp} против {n_ * (n_ - 1) // 2} у точного")
    assert m <= log2(n_ / eps) + 3

# ---------------------------------------------------------------------------
section("4. QFT периодического состояния: пики около кратных N/r")
n = 8
N = 2 ** n
for r, x0 in ((8, 3), (10, 7), (6, 1)):
    xs = np.arange(x0, N, r)
    psi = np.zeros(N, complex); psi[xs] = 1 / np.sqrt(len(xs))
    p = np.abs(dft_matrix(N) @ psi) ** 2
    if N % r == 0:
        assert np.allclose(p[np.arange(0, N, N // r)].sum(), 1)
        print(f"  r = {r:2d} (делит N = {N}): вся вероятность на k*N/r — {r} пиков, сдвиг x0 ушёл в фазы")
    else:
        near = set()
        for s in range(r):
            c = s * N / r
            near |= {int(np.floor(c)) % N, int(np.ceil(c)) % N}
        pn = p[list(near)].sum()
        print(f"  r = {r:2d} (не делит N): вероятность двух ближайших к s*N/r точек = {pn:.3f} >= 4/pi^2 = {4 / pi ** 2:.3f}")
        assert pn >= 4 / pi ** 2
# классическая стоимость: FFT за O(N log N); здесь просто сверяем с np.fft
v = haar_state(N)
assert np.allclose(dft_matrix(N) @ v, np.fft.ifft(v) * np.sqrt(N))
print("  QFT вектора = sqrt(N) ifft(вектор): классически O(N log N) операций, квантово O(n^2) гейтов над амплитудами")

# ---------------------------------------------------------------------------
section("5. Оценка фазы: распределение исходов, точность, вероятность успеха")


def qpe_probs(phi, t):
    """Аналитическое распределение исходов m = 0..2^t-1 для собственной фазы phi."""
    T = 2 ** t
    delta = phi - np.arange(T) / T
    num = np.sin(pi * T * delta) ** 2
    den = (T * np.sin(pi * delta)) ** 2
    exact = np.isclose(np.abs(((delta + 0.5) % 1) - 0.5), 0, atol=1e-13)
    out = np.where(exact, 1.0, num / np.where(exact, 1.0, den))
    return out


def qpe_circuit_probs(Umat, u, t):
    """Полная схема QPE: t вспомогательных кубитов (старшие) + регистр u; распределение по первым t."""
    d = len(u)
    T = 2 ** t
    # состояние после H^t и управляемых U^{2^j}: sum_k |k> U^k |u> / sqrt(T)
    st = np.zeros((T, d), complex)
    Uk = np.eye(d, dtype=complex)
    for k in range(T):
        st[k] = Uk @ u / np.sqrt(T)
        Uk = Umat @ Uk
    # проверим, что это то же, что каскад C-U^{2^j}: k = sum_j k_j 2^{t-1-j}, кубит j управляет U^{2^{t-1-j}}
    st2 = np.zeros((T, d), complex)
    for k in range(T):
        v = u.copy()
        for j in range(t):
            if (k >> (t - 1 - j)) & 1:
                v = np.linalg.matrix_power(Umat, 2 ** (t - 1 - j)) @ v
        st2[k] = v / np.sqrt(T)
    assert np.allclose(st, st2)
    # обратное QFT на вспомогательном регистре — схема из раздела 2
    g = qft_gates(t)
    out = run(g, st, t, inverse=True)
    return np.sum(np.abs(out) ** 2, axis=1)


# (а) phi = k/2^t — детерминированно
for t in (3, 5):
    for kk in (1, 5, 2 ** t - 1):
        phi = kk / 2 ** t
        p = qpe_circuit_probs(P(2 * pi * phi), np.array([0, 1], complex), t)
        assert np.isclose(p[kk], 1)
print("  (а) phi = k/2^t: исход k с вероятностью 1 (схема с C-U^{2^j} и обратным QFT; t = 3, 5)")
p_c = qpe_circuit_probs(P(2 * pi / 3), np.array([0, 1], complex), 3)
assert np.allclose(p_c, qpe_probs(1 / 3, 3))
print("  схема и аналитическая формула (5.3) совпадают (phi = 1/3, t = 3)")
p3 = qpe_probs(1 / 3, 3)
print("  phi = 1/3, t = 3: P(m) =", " ".join(f"{x:.3f}" for x in p3), f"; лучшая оценка m = 3 (3/8 = 0.375), P = {p3[3]:.3f}")
# (б) иррациональная фаза: P(ближайшая) >= 4/pi^2
phi_irr = (np.sqrt(5) - 1) / 2 - 0.5                    # 0.1180...
print(f"  (б) phi = (sqrt5 - 1)/2 - 1/2 = {phi_irr:.6f}")
print("     t   P(ближайшая оценка)   P(две ближайшие)   ошибка |phi - m/2^t|")
for t in range(3, 11):
    p = qpe_probs(phi_irr, t)
    T = 2 ** t
    b = int(np.floor(phi_irr * T))
    best = b if phi_irr * T - b < 0.5 else b + 1
    two = p[b] + p[(b + 1) % T]
    assert p[best] >= 4 / pi ** 2 and two >= 8 / pi ** 2
    print(f"    {t:2d}        {p[best]:.4f}             {two:.4f}            {abs(phi_irr - best / T):.2e}")
# минимум P(ближайшая) по delta: достигается при delta = 1/2 * 2^{-t}, стремится к 4/pi^2
for t in (3, 6, 10):
    worst = qpe_probs(0.5 / 2 ** t, t)
    print(f"  худший случай delta = 2^-(t+1), t = {t:2d}: P(ближайшая) = {worst[0]:.4f} (предел 4/pi^2 = {4 / pi ** 2:.4f})")
    assert worst[0] >= 4 / pi ** 2
# (в) t = n + ceil(log2(2 + 1/(2 eps))) даёт n верных бит с вероятностью >= 1 - eps
print("  (в) t = n + ceil(log2(2 + 1/(2 eps))): P(|m/2^t - phi| < 2^-n) по 200 случайным phi")
for n_, eps in ((3, 0.1), (4, 0.05), (5, 0.01)):
    t = n_ + ceil(log2(2 + 1 / (2 * eps)))
    T = 2 ** t
    worst = 1.0
    for phi in rng.random(200):
        p = qpe_probs(phi, t)
        m = np.arange(T)
        dist = np.abs(((m / T - phi + 0.5) % 1) - 0.5)            # расстояние на окружности
        worst = min(worst, p[dist < 2.0 ** (-n_)].sum())
    print(f"    n = {n_}, eps = {eps}: t = {t}, минимум P = {worst:.4f} >= {1 - eps}")
    assert worst >= 1 - eps
# оценка хвоста P(|m - b| > e) <= 1/(2(e-1))
t = 8
T = 2 ** t
for phi in rng.random(50):
    p = qpe_probs(phi, t)
    b = int(np.floor(phi * T))
    for e in (2, 4, 8, 16):
        far = sum(p[m] for m in range(T) if min((m - b) % T, (b - m) % T) > e)
        assert far <= 1 / (2 * (e - 1)) + 1e-12
print("  хвостовая оценка P(|m - b| > e) <= 1/(2(e-1)) выполнена (t = 8, e = 2, 4, 8, 16, 50 случайных phi)")

# суперпозиция собственных векторов
d = 4
V = haar_unitary(d)
phis = np.array([0.125, 0.3, 0.55, 0.8])
Umat = V @ np.diag(np.exp(2j * pi * phis)) @ V.conj().T
psi = haar_state(d)
c2 = np.abs(V.conj().T @ psi) ** 2
t = 5
p_sup = qpe_circuit_probs(Umat, psi, t)
p_mix = sum(c2[i] * qpe_probs(phis[i], t) for i in range(d))
assert np.allclose(p_sup, p_mix)
print("  вход — суперпозиция собственных векторов: P(m) = sum_u |c_u|^2 P_{phi_u}(m) (проверено, t = 5, d = 4)")
print("    веса |c_u|^2 =", " ".join(f"{x:.3f}" for x in c2))

# масштабирование Гейзенберга: QPE против повторных тестов Адамара при одинаковом числе применений U
print("  точность при Q применениях U: QPE (t кубитов, Q = 2^t - 1) против Q тестов Адамара")
print("  мера точности — полуширина 80 %-го интервала |ошибки| и RMS")
print("     Q     80% QPE    80% Адамар    RMS QPE    RMS Адамар    1/Q       1/sqrt(Q)")
q80q, q80h, rmsq, Qs = [], [], [], []
for t in range(4, 11):
    Q = 2 ** t - 1
    T = 2 ** t
    p = qpe_probs(phi_irr, t)
    m = np.arange(T)
    dist = np.abs(((m / T - phi_irr + 0.5) % 1) - 0.5)
    order = np.argsort(dist)
    cq = dist[order][np.searchsorted(np.cumsum(p[order]), 0.8)]
    rq = np.sqrt(np.sum(p * dist ** 2))
    # тест Адамара: Q/2 повторений для cos и Q/2 для sin, оценка phi = atan2 / 2pi
    reps = Q // 2
    pc = (1 + np.cos(2 * pi * phi_irr)) / 2
    ps = (1 + np.sin(2 * pi * phi_irr)) / 2
    trials = 4000
    c_est = 2 * rng.binomial(reps, pc, trials) / reps - 1
    s_est = 2 * rng.binomial(reps, ps, trials) / reps - 1
    err = np.abs((((np.arctan2(s_est, c_est) / (2 * pi)) - phi_irr + 0.5) % 1) - 0.5)
    ch = np.quantile(err, 0.8)
    rh = np.sqrt(np.mean(err ** 2))
    q80q.append(cq); q80h.append(ch); rmsq.append(rq); Qs.append(Q)
    print(f"  {Q:5d}   {cq:.2e}    {ch:.2e}     {rq:.2e}   {rh:.2e}    {1 / Q:.2e}  {1 / np.sqrt(Q):.2e}")
    assert cq <= 1 / T + 1e-12
    assert rq <= 1 / (2 * np.sqrt(T))                   # задача 13: RMS <= 1/(2 sqrt(2^t))
sq = np.polyfit(np.log(Qs), np.log(q80q), 1)[0]
sh = np.polyfit(np.log(Qs), np.log(q80h), 1)[0]
sr = np.polyfit(np.log(Qs), np.log(rmsq), 1)[0]
print(f"  наклоны log(80%-полуширины) по log Q: QPE {sq:.2f}, тесты Адамара {sh:.2f};  RMS QPE: {sr:.2f} (тяжёлые хвосты)")
assert sq < -0.85 and -0.6 < sh < -0.4 and -0.65 < sr < -0.35
# медианная оценка: повторяя QPE и беря медиану, хвосты подавляются экспоненциально
t = 6; T = 2 ** t
p = qpe_probs(phi_irr, t)
m = np.arange(T); dist = np.abs(((m / T - phi_irr + 0.5) % 1) - 0.5)
good = p[dist <= 1 / T].sum()
print(f"  t = 6: P(ошибка <= 1/2^t) = {good:.3f}")
signed = ((m / T - phi_irr + 0.5) % 1) - 0.5
for k in (1, 5, 15):
    samples = rng.choice(T, size=(20000, k), p=p / p.sum())
    med = np.median(signed[samples], axis=1)
    fail = np.mean(np.abs(med) > 1 / T)
    print(f"    медиана {k:2d} повторений: P(ошибка > 1/2^t) = {fail:.4f}")
    if k == 15:
        assert fail < 0.01

# ---------------------------------------------------------------------------
section("6. Итеративный QPE с одним вспомогательным кубитом")


def iqpe_probs(phi, t):
    """Точное распределение строк исходов итеративного QPE (биты от младшего к старшему)
    с обратной связью P(-omega_k), omega_k = 2 pi * 0.0 b_{k+1} b_{k+2} ... b_t."""
    T = 2 ** t
    probs = np.zeros(T)

    def rec(k, bits_low, prob):
        # bits_low — уже измеренные b_{k+1}..b_t (список, b_{k+1} первым)
        if k == 0:
            m = sum(b << (len(bits_low) - 1 - i) for i, b in enumerate(bits_low))
            probs[m] += prob
            return
        omega = 2 * pi * sum(b / 2 ** (i + 2) for i, b in enumerate(bits_low))
        anc = np.array([1, np.exp(2j * pi * 2 ** (k - 1) * phi)]) / np.sqrt(2)   # H, C-U^{2^{k-1}}
        anc = H @ (P(-omega) @ anc)
        for b in (0, 1):
            pb = abs(anc[b]) ** 2
            if pb > 1e-15:
                rec(k - 1, [b] + bits_low, prob * pb)

    rec(t, [], 1.0)
    return probs


for t in (3, 5, 7):
    for phi in list(rng.random(5)) + [5 / 2 ** t]:
        assert np.allclose(iqpe_probs(phi, t), qpe_probs(phi, t))
print("  распределение исходов итеративного QPE совпадает со стандартным (t = 3, 5, 7; случайные phi)")
p = iqpe_probs(5 / 8, 3)
assert np.isclose(p[5], 1)
print("  phi = 0.101 (двоичная) = 5/8: биты 1, 0, 1 выдаются детерминированно (младший — первым)")
# моделирование с выборкой: одна траектория
t = 6; phi = 0.3; counts = np.zeros(2 ** t)
for _ in range(3000):
    bits = []
    for k in range(t, 0, -1):
        omega = 2 * pi * sum(b / 2 ** (i + 2) for i, b in enumerate(bits))
        anc = H @ (P(-omega) @ (np.array([1, np.exp(2j * pi * 2 ** (k - 1) * phi)]) / np.sqrt(2)))
        b = int(rng.random() < abs(anc[1]) ** 2)
        bits = [b] + bits
    counts[sum(b << (t - 1 - i) for i, b in enumerate(bits))] += 1
tv = 0.5 * np.abs(counts / 3000 - qpe_probs(phi, t)).sum()
print(f"  выборка 3000 траекторий (t = 6, phi = 0.3): расстояние полной вариации до QPE = {tv:.3f}")
assert tv < 0.06
print("  ресурсы: 1 вспомогательный кубит вместо t; t измерений с обратной связью; те же 2^t - 1 применений U")

# ---------------------------------------------------------------------------
section("7. Тест Адамара и swap-тест")
for _ in range(10):
    d = 4
    U = haar_unitary(d)
    psi = haar_state(d)
    # схема: |0>|psi> -> H -> C-U -> (S^dag) -> H -> измерить
    CU = np.block([[np.eye(d), np.zeros((d, d))], [np.zeros((d, d)), U]])
    for phase_gate, part in ((I2, "re"), (Sdg, "im")):
        st = np.kron(H @ phase_gate, np.eye(d)) @ CU @ np.kron(H, np.eye(d)) @ np.kron([1, 0], psi)
        p0 = np.sum(np.abs(st[:d]) ** 2)
        ev = np.vdot(psi, U @ psi)
        assert np.isclose(p0, (1 + (ev.real if part == "re" else ev.imag)) / 2)
print("  тест Адамара: P(0) = (1 + Re<psi|U|psi>)/2; с S^dag перед H: (1 + Im<psi|U|psi>)/2  (10 случайных U, psi)")


def swap_test_p0(rho, sigma):
    """Схема: |0><0| (x) rho (x) sigma; H; controlled-SWAP; H; P(0)."""
    d = rho.shape[0]
    SW = np.zeros((d * d, d * d))
    for a in range(d):
        for b in range(d):
            SW[b * d + a, a * d + b] = 1
    CSW = np.block([[np.eye(d * d), np.zeros((d * d, d * d))], [np.zeros((d * d, d * d)), SW]])
    U = np.kron(H, np.eye(d * d)) @ CSW @ np.kron(H, np.eye(d * d))
    full = U @ np.kron(np.diag([1, 0]), np.kron(rho, sigma)) @ U.conj().T
    return np.trace(full[:d * d, :d * d]).real


for _ in range(5):
    a, b = haar_state(8), haar_state(8)                 # три кубита
    p0 = swap_test_p0(np.outer(a, a.conj()), np.outer(b, b.conj()))
    assert np.isclose(p0, (1 + abs(np.vdot(a, b)) ** 2) / 2)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)); rho = A @ A.conj().T; rho /= np.trace(rho)
    B = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)); sig = B @ B.conj().T; sig /= np.trace(sig)
    assert np.isclose(swap_test_p0(rho, sig), (1 + np.trace(rho @ sig).real) / 2)
print("  swap-тест: P(0) = (1 + |<a|b>|^2)/2 (случайные 3-кубитные состояния); для смешанных (1 + Tr rho sigma)/2")
# вариант без вспомогательного кубита: измерение пар (a_i, b_i) в базисе Белла;
# SWAP = prod_i SWAP_i, SWAP_i = I - 2|Psi-><Psi-|, поэтому P(0) = P(чётное число синглетов)
Psim = np.array([0, 1, -1, 0]) / np.sqrt(2)
a, b = haar_state(8), haar_state(8)
st = np.kron(a, b).reshape([2] * 6)                       # a1 a2 a3 b1 b2 b3
st = np.transpose(st, (0, 3, 1, 4, 2, 5)).reshape(4, 4, 4)  # пары (a_i b_i)
Pm = np.outer(Psim, Psim)
proj = [Pm, np.eye(4) - Pm]
p_even = 0.0
for c in np.ndindex(2, 2, 2):
    v = np.einsum("ai,bj,ck,ijk->abc", proj[c[0]], proj[c[1]], proj[c[2]], st)
    if (3 - sum(c)) % 2 == 0:                                 # c = 0 — синглет
        p_even += np.vdot(v, v).real
assert np.isclose(p_even, (1 + abs(np.vdot(a, b)) ** 2) / 2)
print("  измерение пар в базисе Белла: P(чётное число синглетов) = (1 + |<a|b>|^2)/2 — swap-тест без анциллы")
# статистическая ошибка ~ 1/sqrt(shots)
a, b = haar_state(8), haar_state(8)
ov = abs(np.vdot(a, b)) ** 2
p0 = (1 + ov) / 2
print(f"  оценка |<a|b>|^2 = {ov:.4f} по выборке:  shots   RMSE     sqrt((1-ov^2)/shots)")
errs, shots_list = [], [100, 1000, 10000, 100000]
for shots in shots_list:
    est = 2 * rng.binomial(shots, p0, 2000) / shots - 1
    rmse = np.sqrt(np.mean((est - ov) ** 2))
    theo = np.sqrt(4 * p0 * (1 - p0) / shots)
    errs.append(rmse)
    print(f"      {shots:6d}   {rmse:.4f}   {theo:.4f}")
    assert abs(rmse / theo - 1) < 0.1
slope = np.polyfit(np.log(shots_list), np.log(errs), 1)[0]
print(f"  наклон log RMSE по log shots = {slope:.3f} (теория -1/2)")
assert abs(slope + 0.5) < 0.05
# тест Адамара = QPE с t = 1
phi = 0.2
assert np.isclose(qpe_probs(phi, 1)[0], (1 + np.cos(2 * pi * phi)) / 2)
print("  QPE с t = 1 есть тест Адамара: P(0) = (1 + cos 2 pi phi)/2")

# ---------------------------------------------------------------------------
section("8. Ответы к задачам семинара")
# 3: число гейтов при n = 10
g = qft_gates(10)
print(f"  3: n = 10: H = {sum(x[0] == 'H' for x in g)}, CP = {sum(x[0] == 'CP' for x in g)}, SWAP = {sum(x[0] == 'SWAP' for x in g)}")
# 4: j = 5 (n=3) — уже проверено в разделе 2
# 5: оценка ошибки AQFT для n = 10, m = 3
print(f"  5: n = 10, m = 3: оценка {dropped_bound(10, 3):.4f}, точная операторная ошибка {aqft[3][1]:.4f}, CP-гейтов {aqft[3][0]}")
# 6: phi = 1/3, t = 3
print(f"  6: phi = 1/3, t = 3: P(m = 3) = {p3[3]:.4f}, P(m = 2 или 3) = {p3[2] + p3[3]:.4f}, P(m = 5/8 = 0.101 при phi = 5/8) = 1")
# 8: t для n = 4, eps = 0.05
t8 = 4 + ceil(log2(2 + 1 / (2 * 0.05)))
print(f"  8: n = 4, eps = 0.05: t = {t8}")
assert t8 == 8
# 9: U = Z, вход |+>, t = 1
p = qpe_circuit_probs(Z, np.array([1, 1], complex) / np.sqrt(2), 1)
print(f"  9: U = Z, вход |+>, t = 1: P(0) = {p[0]:.2f}, P(1) = {p[1]:.2f}; после исхода регистр в |0> или |1>")
assert np.allclose(p, [0.5, 0.5])
# 10: число повторений теста Адамара для std 0.01
print("  10: Var(2 P0_hat - 1) = (1 - r^2)/shots <= 1/shots, при std 0.01 достаточно 10^4 повторений")
# 11: чистота через swap-тест: rho_p (вернеровское) — Tr rho^2 = (1 + 3 p^2)/4
Phi = np.array([1, 0, 0, 1]) / np.sqrt(2)
for pp in (0.0, 0.5, 1.0):
    rho = pp * np.outer(Phi, Phi) + (1 - pp) * np.eye(4) / 4
    p0 = swap_test_p0(rho, rho)
    assert np.isclose(p0, (1 + (1 + 3 * pp ** 2) / 4) / 2)
    print(f"  11: rho_p, p = {pp}: Tr rho^2 = {(1 + 3 * pp ** 2) / 4:.4f}, P(0) swap-теста = {p0:.4f}")
print("\nВсе проверки пройдены.")
