"""
Лекция II.15. Практическая QKD и квантовые сети.

Проверяется: пуассоновская статистика ослабленного лазера и доля
многофотонных импульсов; условие необнаружимости атаки с разделением числа
фотонов (PNS) и её обнаружение decoy-состояниями; модель канала (выходы
Y_n, коэффициенты усиления Q_mu, ошибки E_mu); скорость GLLP без decoy,
с бесконечным числом decoy и с двумя decoy (вакуум + слабый) — оценки
Y_1, e_1 и предельные расстояния; атака ослеплением детекторов (QBER 0 при
полном знании ключа Евой); MDI-QKD: корреляции после белловского измерения
у Чарли, линейно-оптическое белловское измерение (успех 1/2, по алгебре
операторов рождения); масштабирование η, √η и граница PLOB; DIQKD:
порог эффективности детекторов для CHSH (82,8 %) и для ключа (92,3 %);
повторители: обмен запутанностью, очищение BBPSSW (формула против явного
моделирования матриц плотности); спутник: дифракционные потери против
волокна; CV-QKD GG02 (гомодин) в канале потерь: прямое и обратное
согласование по гауссовым ковариационным матрицам, предел 3 дБ, сравнение
с PLOB.

Соглашения (Лекции/Обозначения.md, §4, §21, §25–§27): log = log2;
h — двоичная энтропия; mu — средняя интенсивность сигнала, nu — decoy;
eta — полное пропускание (канал x детектор); Y_n — выход n-фотонных
импульсов, Q_mu — коэффициент усиления, E_mu — QBER; R — бит на посылку;
в CV-части дисперсия вакуума 1/2 (II.9), V_A — дисперсия модуляции в
единицах дробового шума (SNU), т. е. в наших единицах V_A/2.

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
from math import log2, sqrt, pi, exp, factorial, log
from itertools import product
import numpy as np
from scipy.optimize import brentq

T_START = time.time()
rng = np.random.default_rng(2026)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def h(x):
    if x <= 0 or x >= 1:
        return 0.0
    return -x * log2(x) - (1 - x) * log2(1 - x)


def g(n):
    return 0.0 if n <= 1e-12 else (n + 1) * log2(n + 1) - n * log2(n)


def ket(*v):
    v = np.array(v, complex)
    return v / np.linalg.norm(v)


def proj(v):
    return np.outer(v, v.conj())


# ===========================================================================
section("1. Ослабленный лазер: статистика Пуассона")
def pois(n, mu):
    return exp(-mu) * mu ** n / factorial(n)

for mu in (0.1, 0.5, 1.0):
    p0, p1 = pois(0, mu), pois(1, mu)
    pm = 1 - p0 - p1
    frac = pm / (1 - p0)
    assert abs(sum(pois(n, mu) for n in range(60)) - 1) < 1e-12
    assert frac < mu / 2 + 1e-12
    print(f"  mu = {mu}: P(0) = {p0:.4f}, P(1) = {p1:.4f}, P(n>=2) = {pm:.4f}; доля многофотонных среди непустых = {frac:.4f} (~mu/2 = {mu / 2:.3f})")

# ===========================================================================
section("2. Модель канала и атака с разделением числа фотонов")
ALPHA, ETA_B, Y0, ED, FEC, QS = 0.21, 0.045, 1.7e-6, 0.033, 1.22, 0.5   # параметры как в Ma et al. (2005), эксперимент GYS


def eta_of(L):
    return ETA_B * 10 ** (-ALPHA * L / 10)


def Yn(n, eta):
    return 1 - (1 - Y0) * (1 - eta) ** n


def en(n, eta):
    y = Yn(n, eta)
    return (0.5 * Y0 + ED * (y - Y0)) / y


def gain_sum(mu, eta, nmax=60):
    Q = sum(Yn(n, eta) * pois(n, mu) for n in range(nmax))
    EQ = sum(en(n, eta) * Yn(n, eta) * pois(n, mu) for n in range(nmax))
    return Q, EQ / Q


def gain(mu, eta):
    Q = 1 - (1 - Y0) * exp(-eta * mu)
    return Q, (0.5 * Y0 + ED * (Q - Y0)) / Q


# замкнутые формулы против суммы по числу фотонов
for mu, L in ((0.5, 0), (0.5, 50), (0.1, 100)):
    eta = eta_of(L)
    Q, E = gain_sum(mu, eta)
    assert abs(Q - (1 - (1 - Y0) * exp(-eta * mu))) < 1e-12
    assert abs(E * Q - (0.5 * Y0 + ED * (Q - Y0))) < 1e-12
print("  Q_mu = 1 - (1 - Y0) exp(-eta mu), E_mu Q_mu = Y0/2 + e_d (Q_mu - Y0) — проверено")

# PNS: Ева в канале без потерь блокирует однофотонные импульсы и пропускает по фотону из многофотонных.
# Необнаружима (по одному Q_mu), пока Боб и так ждёт не больше кликов, чем P(n>=2).
PNS = {}
for mu in (0.1, 0.5):
    pm = 1 - pois(0, mu) - pois(1, mu)
    eta_c = brentq(lambda e: (1 - exp(-e * mu)) - pm, 1e-9, 1)
    PNS[mu] = eta_c
    print(f"  mu = {mu}: PNS необнаружима (по одному Q_mu) при полном пропускании eta < {eta_c:.4f}")
assert PNS[0.5] > ETA_B and PNS[0.1] > ETA_B
print(f"  уже эффективность детектора Боба eta_B = {ETA_B} ниже порога: без decoy нужна mu ~ eta")
# decoy обнаруживает PNS: подгоним PNS-выходы (Y_1 = 0, Y_n = y для n >= 2) под Q_mu сигнала
mu, nu, L = 0.5, 0.1, 60
eta = eta_of(L)
Qmu, _ = gain(mu, eta)
Qnu, _ = gain(nu, eta)
y_fit = (Qmu - Y0 * pois(0, mu)) / (1 - pois(0, mu) - pois(1, mu))
Qnu_pns = Y0 * pois(0, nu) + y_fit * (1 - pois(0, nu) - pois(1, nu))
assert y_fit <= 1 and Qnu_pns < 0.5 * Qnu
print(f"  L = {L} км, mu = {mu}: PNS подогнана под Q_mu = {Qmu:.3e}; для decoy nu = {nu} Боб ждёт Q_nu = {Qnu:.3e},"
      f" а при PNS получит {Qnu_pns:.3e} (в {Qnu / Qnu_pns:.1f} раза меньше) — атака видна")

# ===========================================================================
section("3. Скорость ключа: GLLP без decoy, бесконечно много decoy, вакуум + слабый decoy")
def R_inf(mu, L):
    eta = eta_of(L)
    Q, E = gain(mu, eta)
    Q1 = Yn(1, eta) * pois(1, mu)
    return QS * (-Q * FEC * h(E) + Q1 * (1 - h(en(1, eta))))


def R_nodecoy(mu, L):
    eta = eta_of(L)
    Q, E = gain(mu, eta)
    Q1 = Q - (1 - pois(0, mu) - pois(1, mu))          # худший случай: все многофотонные импульсы дошли
    if Q1 <= 0:
        return -1.0
    e1 = min(0.5, E * Q / Q1)
    return QS * (-Q * FEC * h(E) + Q1 * (1 - h(e1)))


def decoy_bounds(mu, nu, L):
    eta = eta_of(L)
    Qm, Em = gain(mu, eta)
    Qn, En = gain(nu, eta)
    Y1L = mu / (mu * nu - nu ** 2) * (Qn * exp(nu) - Qm * exp(mu) * nu ** 2 / mu ** 2 - (mu ** 2 - nu ** 2) / mu ** 2 * Y0)
    e1U = min(0.5, (En * Qn * exp(nu) - 0.5 * Y0) / (Y1L * nu)) if Y1L > 0 else 0.5
    return Y1L, e1U, Qm, Em


def R_vw(mu, L, nu=0.05):
    Y1L, e1U, Q, E = decoy_bounds(mu, nu, L)
    if Y1L <= 0:
        return -1.0
    return QS * (-Q * FEC * h(E) + Y1L * pois(1, mu) * (1 - h(e1U)))


MUS = np.logspace(-4.5, 0, 400)


def best(f, L):
    v = [f(m, L) for m in MUS]
    i = int(np.argmax(v))
    return v[i], MUS[i]


# оценки decoy консервативны
for L in (0, 50, 100):
    Y1L, e1U, _, _ = decoy_bounds(0.48, 0.05, L)
    eta = eta_of(L)
    assert Y1L <= Yn(1, eta) and e1U >= en(1, eta)
    print(f"  L = {L:3d} км: Y_1 >= {Y1L:.4e} (истинное {Yn(1, eta):.4e}), e_1 <= {e1U:.4f} (истинное {en(1, eta):.4f})")
RATES = {}
print("   L, км | без decoy (mu_opt)     | decoy, предел (mu_opt) | вакуум + слабый nu = 0.05")
for L in (0, 20, 40, 60, 100, 140):
    RATES[L] = (best(R_nodecoy, L), best(R_inf, L), best(R_vw, L))
    (a, ma), (b, mb), (c, mc) = RATES[L]
    fmt = lambda r: f"{r:.3e}" if r > 0 else "    0    "
    print(f"   {L:5d} | {fmt(a)} ({ma:.4f})    | {fmt(b)} ({mb:.3f})       | {fmt(c)} ({mc:.3f})")
def maxdist(f, Lmax=250):
    """Наибольшее L, при котором максимум скорости по mu положителен (бисекция; скорость убывает с L)."""
    lo, hi = 0.0, Lmax
    assert best(f, lo)[0] > 0 and best(f, hi)[0] <= 0
    while hi - lo > 0.1:
        mid = (lo + hi) / 2
        if best(f, mid)[0] > 0:
            lo = mid
        else:
            hi = mid
    return lo
D_no, D_inf, D_vw = maxdist(R_nodecoy, 80), maxdist(R_inf), maxdist(R_vw)
assert 30 <= D_no <= 50 and 130 <= D_inf <= 150 and D_inf - 3 <= D_vw <= D_inf
print(f"  предельное расстояние: без decoy {D_no:.1f} км, decoy (предел) {D_inf:.1f} км, вакуум + слабый {D_vw:.1f} км")
# масштабирование: без decoy R ~ eta^2, с decoy R ~ eta (наклоны на 0..20 и 20..60 км)
s_no = np.log10(RATES[20][0][0] / RATES[0][0][0]) / np.log10(eta_of(20) / eta_of(0))
s_in = np.log10(RATES[60][1][0] / RATES[20][1][0]) / np.log10(eta_of(60) / eta_of(20))
assert 1.7 < s_no < 3.0 and 0.9 < s_in < 1.15
print(f"  наклон log R / log eta: без decoy {s_no:.2f} (~2), с decoy {s_in:.2f} (~1)")
# сравнение с PLOB для того же полного пропускания (с учётом детектора Боба)
for L in (60, 100):
    eta = eta_of(L)
    plob = -log2(1 - eta)
    assert RATES[L][1][0] < plob
    print(f"  L = {L} км: R(decoy) = {RATES[L][1][0]:.2e} < PLOB(eta) = {plob:.2e} (отношение {RATES[L][1][0] / plob:.3f})")

# ===========================================================================
section("4. Атака ослеплением детекторов (faked states)")
N = 200000
a = rng.integers(0, 2, N)
al = rng.integers(0, 2, N)
be = rng.integers(0, 2, N)
eb = rng.integers(0, 2, N)                       # Ева: базис
em = np.where(eb == al, a, rng.integers(0, 2, N))   # Ева измеряет
# ослеплённый детектор Боба щёлкает, только если базис Боба совпал с базисом Евы, и выдаёт бит Евы
click = be == eb
bob = em
sift = (al == be) & click
qber = np.mean(a[sift] != bob[sift])
eve_know = np.mean(em[sift] == bob[sift])
rate_click = click.mean()
print(f"  доля кликов {rate_click:.3f} (~1/2, списывается на потери)")
# в просеянном ключе базис Алисы = базису Боба = базису Евы, поэтому Ева знает бит точно
assert np.all(em[sift] == a[sift]) and qber == 0.0 and eve_know == 1.0
print(f"  в просеянных позициях базисы Алисы, Боба и Евы совпадают: QBER = {qber:.4f}, Ева знает {eve_know:.0%} ключа")

# ===========================================================================
section("5. MDI-QKD: белловское измерение у Чарли")
k0, k1 = ket(1, 0), ket(0, 1)
kp, km = ket(1, 1), ket(1, -1)
B = {"Z": (k0, k1), "X": (kp, km)}
PSIm, PSIp = ket(0, 1, -1, 0), ket(0, 1, 1, 0)
PHIp, PHIm = ket(1, 0, 0, 1), ket(1, 0, 0, -1)
CORR = {}
for name, bell in (("Psi-", PSIm), ("Psi+", PSIp)):
    for bas in ("Z", "X"):
        pr = np.zeros((2, 2))
        for x, y in product(range(2), range(2)):
            pr[x, y] = abs(np.vdot(bell, np.kron(B[bas][x], B[bas][y]))) ** 2 / 4
        pr /= pr.sum()
        CORR[(name, bas)] = pr[0, 0] + pr[1, 1]          # доля совпадений
        print(f"  Чарли объявил {name}, базис {bas}: P(биты Алисы и Боба совпадают) = {CORR[(name, bas)]:.2f}")
assert CORR[("Psi-", "Z")] < 1e-12 and CORR[("Psi-", "X")] < 1e-12
assert CORR[("Psi+", "Z")] < 1e-12 and abs(CORR[("Psi+", "X")] - 1) < 1e-12
# если Чарли нечестен и объявляет случайный результат, ошибки видны: QBER 1/2 в базисе X
print("  Psi-: биты противоположны в Z и X; Psi+: противоположны в Z, совпадают в X -> Боб инвертирует по правилу")

# линейно-оптическое белловское измерение: поляризационное кодирование, светоделитель 50:50 + PBS
# состояние — многочлен от операторов рождения (aH, aV, bH, bV) -> (cH, cV, dH, dV)
def bs_transform(poly):
    out = {}
    for mono, amp in poly.items():
        # mono — кортеж мод (с повторениями); a -> (c + d)/sqrt2, b -> (c - d)/sqrt2, поляризация сохраняется
        terms = [((), amp)]
        for m in mono:
            port, pol = m[0], m[1]
            new = []
            for tmono, tamp in terms:
                if port == "a":
                    new += [(tmono + ("c" + pol,), tamp / sqrt(2)), (tmono + ("d" + pol,), tamp / sqrt(2))]
                else:
                    new += [(tmono + ("c" + pol,), tamp / sqrt(2)), (tmono + ("d" + pol,), -tamp / sqrt(2))]
            terms = new
        for tmono, tamp in terms:
            key = tuple(sorted(tmono))
            out[key] = out.get(key, 0) + tamp
    # амплитуда состояния Фока: коэффициент при мономе умножается на sqrt(prod n_k!)
    res = {}
    for key, amp in out.items():
        if abs(amp) < 1e-14:
            continue
        mult = np.prod([factorial(key.count(m)) for m in set(key)])
        res[key] = amp * sqrt(mult)
    return res

BELL_POLY = {"Psi+": {("aH", "bV"): 1 / sqrt(2), ("aV", "bH"): 1 / sqrt(2)},
             "Psi-": {("aH", "bV"): 1 / sqrt(2), ("aV", "bH"): -1 / sqrt(2)},
             "Phi+": {("aH", "bH"): 1 / sqrt(2), ("aV", "bV"): 1 / sqrt(2)},
             "Phi-": {("aH", "bH"): 1 / sqrt(2), ("aV", "bV"): -1 / sqrt(2)}}
PATTERNS = {}
for name, poly in BELL_POLY.items():
    out = bs_transform(poly)
    probs = {k: abs(v) ** 2 for k, v in out.items()}
    assert abs(sum(probs.values()) - 1) < 1e-12
    PATTERNS[name] = probs
    print(f"  {name}: картины отсчётов " + ", ".join(f"{'+'.join(k)}: {p:.2f}" for k, p in sorted(probs.items())))
sets = {n: set(p) for n, p in PATTERNS.items()}
assert sets["Psi+"].isdisjoint(sets["Psi-"]) and sets["Psi+"].isdisjoint(sets["Phi+"]) and sets["Psi-"].isdisjoint(sets["Phi+"])
assert sets["Phi+"] == sets["Phi-"]
print("  Psi+ и Psi- однозначно различимы (разные картины), Phi+ и Phi- дают одинаковые картины -> успех 1/2")
# HOM: два фотона одной поляризации на входах a, b -> оба в одном выходе
hom = bs_transform({("aH", "bH"): 1.0})
assert all(k.count("cH") == 2 or k.count("dH") == 2 for k in hom)
print(f"  эффект Хонга–У–Мандела: |1_H,1_H> -> " + ", ".join(f"{'+'.join(k)}: {abs(v) ** 2:.2f}" for k, v in hom.items()))

# масштабирование: прямая линия ~ eta, MDI ~ eta (Чарли посередине: sqrt(eta) * sqrt(eta)), TF ~ sqrt(eta)
print("  масштабирование с пропусканием линии eta (0.2 дБ/км):")
SC = {}
for L in (100, 300, 500):
    eta = 10 ** (-0.02 * L)
    SC[L] = (eta, -log2(1 - eta), sqrt(eta))
    print(f"    L = {L} км: eta = {eta:.1e}, PLOB = {SC[L][1]:.2e}, sqrt(eta) = {SC[L][2]:.1e}")
# при каком L sqrt(eta) * c обгоняет PLOB при c = 1e-3
c_tf = 1e-3
L_cross = brentq(lambda L: c_tf * sqrt(10 ** (-0.02 * L)) + log2(1 - 10 ** (-0.02 * L)), 10, 1000)
assert 250 < L_cross < 350
print(f"  схема с R = {c_tf:.0e} * sqrt(eta) обгоняет PLOB при L > {L_cross:.0f} км")

# ===========================================================================
section("6. DIQKD: эффективность детекторов")
def S_eta(eta):
    # пропуск отсчёта -> исход +1; для Phi+ маргиналы нулевые: E = eta^2 E_q + (1-eta)^2
    return 2 * sqrt(2) * eta ** 2 + 2 * (1 - eta) ** 2

# проверка формулы прямым расчётом с POVM «клик/нет» на изотропном Phi+
Zm = np.diag([1.0, -1.0])
Xm = np.array([[0, 1], [1, 0]], float)
def A_obs(t):
    return np.cos(t) * Zm + np.sin(t) * Xm
def S_direct(eta):
    rho = proj(PHIp).real
    def E(ta, tb):
        # исход +1 при пропуске: эффективная наблюдаемая eta*A + (1-eta)*I
        Aa = eta * A_obs(ta) + (1 - eta) * np.eye(2)
        Bb = eta * A_obs(tb) + (1 - eta) * np.eye(2)
        return np.trace(rho @ np.kron(Aa, Bb))
    return E(0, pi / 4) + E(0, -pi / 4) + E(pi / 2, pi / 4) - E(pi / 2, -pi / 4)
for e_ in (1.0, 0.9, 0.8):
    assert abs(S_eta(e_) - S_direct(e_)) < 1e-12
eta_chsh = brentq(lambda e: S_eta(e) - 2, 0.5, 1)
assert abs(eta_chsh - 2 / (1 + sqrt(2))) < 1e-10
print(f"  CHSH > 2 при eta > 2/(1 + sqrt2) = {eta_chsh:.4f}")
def r_di(eta):
    S = S_eta(eta)
    Q = eta * (1 - eta)             # ключевые измерения совпадают; ошибка, когда щёлкнул ровно один детектор
    if S <= 2:
        return -1.0
    return 1 - h(Q) - h((1 + sqrt((S / 2) ** 2 - 1)) / 2)
eta_key = brentq(r_di, 0.85, 0.999)
assert abs(eta_key - 0.923) < 1e-3
print(f"  ключ DIQKD (I.6, (9.2), без шума, пропуски -> +1): eta > {eta_key:.4f}")
for e_ in (0.95, 0.98, 0.99):
    print(f"    eta = {e_}: S = {S_eta(e_):.4f}, Q = {e_ * (1 - e_):.4f}, r >= {r_di(e_):.4f}")

# ===========================================================================
section("7. Повторители: обмен запутанностью и очищение")
def iso(F):
    """Изотропное состояние с верностью F относительно Phi+."""
    p = (4 * F - 1) / 3
    return p * proj(PHIp) + (1 - p) * np.eye(4) / 4

def bbpssw_formula(F):
    t = (1 - F) / 3
    num = F ** 2 + t ** 2
    den = F ** 2 + 2 * F * t + 5 * t ** 2
    return num / den, den

def cnot(n, c, t):
    d = 2 ** n
    U = np.zeros((d, d))
    for i in range(d):
        bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
        if bits[c]:
            bits[t] ^= 1
        j = sum(b << (n - 1 - k) for k, b in enumerate(bits))
        U[j, i] = 1
    return U

def bbpssw_sim(F):
    # кубиты: A1 B1 A2 B2 (0..3); пары (A1B1), (A2B2)
    rho = np.kron(iso(F), iso(F))
    U = cnot(4, 0, 2) @ cnot(4, 1, 3)      # двусторонний CNOT
    rho = U @ rho @ U.T
    # измеряем A2, B2 в Z, оставляем при совпадении
    keep = np.zeros(16)
    for i in range(16):
        b2, b3 = (i >> 1) & 1, i & 1
        keep[i] = 1.0 if b2 == b3 else 0.0
    P = np.diag(keep)
    r = P @ rho @ P
    ps = np.trace(r).real
    r = r / ps
    t = r.reshape(4, 4, 4, 4)                 # (A1B1)(A2B2) x (A1B1)(A2B2)
    r1 = np.einsum("ikjk->ij", t)
    return np.vdot(PHIp, r1 @ PHIp).real, ps

PUR = {}
for F in (0.6, 0.75, 0.9):
    Ff, pf = bbpssw_formula(F)
    Fs, ps = bbpssw_sim(F)
    assert abs(Ff - Fs) < 1e-12 and abs(pf - ps) < 1e-12
    PUR[F] = (Ff, pf)
    print(f"  BBPSSW: F = {F} -> F' = {Ff:.4f}, вероятность успеха {pf:.4f} (формула = моделирование)")
assert bbpssw_formula(0.5)[0] - 0.5 < 1e-12 and bbpssw_formula(0.51)[0] > 0.51
# итерации из F = 0.7
F = 0.7
seq = [F]
for _ in range(6):
    F = bbpssw_formula(F)[0]
    seq.append(F)
assert all(seq[i + 1] > seq[i] for i in range(6))
print("  итерации (с твирлингом к изотропному состоянию): " + " -> ".join(f"{x:.4f}" for x in seq))
# обмен: изотропные пары p1, p2 -> p1 p2 (I.5, §5); в верностях
for F1, F2 in ((0.95, 0.95), (0.9, 0.8)):
    p1, p2 = (4 * F1 - 1) / 3, (4 * F2 - 1) / 3
    Fo = (3 * p1 * p2 + 1) / 4
    print(f"  обмен: F1 = {F1}, F2 = {F2} -> F = {Fo:.4f}")
# цепочка из 2^k звеньев без очищения
F0 = 0.98
for k in (1, 2, 3, 4, 5):
    p = ((4 * F0 - 1) / 3) ** (2 ** k)
    print(f"    {2 ** k:2d} звеньев с F = {F0}: F = {(3 * p + 1) / 4:.4f}")

# ===========================================================================
section("8. Спутник: дифракционные потери против волокна (игрушечная оценка)")
lam, DT, DR = 850e-9, 0.3, 1.0
SAT = {}
for L in (500e3, 1200e3):
    theta = 1.22 * lam / DT                         # полуугол расходимости дифракционного пучка
    spot = 2 * theta * L
    eta_geo = min(1.0, (DR / spot) ** 2)
    fiber_db = 0.2 * L / 1e3
    SAT[L] = (spot, -10 * np.log10(eta_geo), fiber_db)
    print(f"  L = {L / 1e3:.0f} км: пятно ~{spot:.1f} м, геометрические потери {SAT[L][1]:.1f} дБ; волокно 0.2 дБ/км: {fiber_db:.0f} дБ")
assert SAT[1200e3][1] < 30 and SAT[1200e3][2] == 240
print("  (без атмосферы, турбулентности, наведения и эффективности телескопов — реальные потери больше)")

# ===========================================================================
section("9. CV-QKD (GG02, гомодин) в канале чистых потерь")
Om = np.array([[0, 1], [-1, 0]])
Zd = np.diag([1.0, -1.0])


def symp(V):
    m = V.shape[0] // 2
    O = np.kron(np.eye(m), Om)
    return np.sort(np.abs(np.linalg.eigvals(1j * O @ V)))[::2]


def Sg(V):
    return sum(g(nu - 0.5) for nu in symp(V))


def cv_keys(VA, eta):
    """VA — дисперсия модуляции в SNU. Возвращает (I_AB, K_DR, K_RR)."""
    v = (VA + 1) / 2                         # запутанный вариант: TMSV, дисперсия моды v (вакуум 1/2)
    c = sqrt(v * v - 0.25)
    gB = eta * v + (1 - eta) / 2
    G = np.block([[v * np.eye(2), sqrt(eta) * c * Zd], [sqrt(eta) * c * Zd, gB * np.eye(2)]])
    I_AB = 0.5 * log2(1 + eta * VA)
    S_AB = Sg(G)                             # = S(E): ABE чистое
    # обратное согласование: Ева о гомодинном x_B; S(E|x_B) = S(A|x_B)
    gA = np.diag([v - eta * c * c / gB, v])
    chi_BE = S_AB - g(sqrt(np.linalg.det(gA)) - 0.5)
    # прямое согласование: Ева о квадратуре x_A, модулированной Алисой; при известном x_A мода Евы гауссова
    # с дисперсиями 1/2 и 1/2 + (1 - eta) VA/2
    chi_AE = S_AB - g(sqrt(0.5 * (0.5 + (1 - eta) * VA / 2)) - 0.5)
    return I_AB, I_AB - chi_AE, I_AB - chi_BE

# проверка: при eta = 1 ключ = I_AB (Ева ничего не знает)
I1, d1, r1 = cv_keys(10.0, 1.0)
assert abs(d1 - I1) < 1e-9 and abs(r1 - I1) < 1e-9
VAS = np.logspace(-2, 5, 500)
CV = {}
for eta in (0.9, 0.6, 0.5, 0.1, 0.01):
    dr = max(cv_keys(V, eta)[1] for V in VAS)
    rr = max(cv_keys(V, eta)[2] for V in VAS)
    plob = -log2(1 - eta)
    CV[eta] = (dr, rr, plob)
    assert rr <= plob + 1e-9
    print(f"  eta = {eta:5}: прямое K = {max(dr, 0):.4f}, обратное K = {rr:.4f}, PLOB = {plob:.4f}")
assert CV[0.5][0] < 1e-5 and CV[0.6][0] > 0.2 and CV[0.01][1] > 0
eta_dr = brentq(lambda e: max(cv_keys(V, e)[1] for V in VAS) - 1e-7, 0.45, 0.7)
assert abs(eta_dr - 0.5) < 0.01
print(f"  прямое согласование работает только при eta > {eta_dr:.3f} (предел 3 дБ); обратное — при любом eta")
ratio = CV[0.01][1] / CV[0.01][2]
print(f"  при eta = 0.01 обратное согласование даёт {ratio:.2f} от PLOB")
# предел большой модуляции: K_RR -> (1/2) log2(1/(1 - eta)) — половина PLOB (гомодин видит одну квадратуру)
for eta in (0.9, 0.5, 0.1, 0.01):
    kr = cv_keys(1e7, eta)[2]
    assert abs(kr - 0.5 * (-log2(1 - eta))) < 1e-4
print("  при V_A -> бесконечности K_RR -> (1/2) log2(1/(1-eta)) = PLOB/2 — проверено для eta = 0.9; 0.5; 0.1; 0.01")

# ЛР16, задание 6: идеальный однофотонный источник, те же детекторы и шум
def R_single(L):
    eta = eta_of(L)
    Q = Yn(1, eta)
    E = en(1, eta)
    return QS * Q * (1 - h(E) - FEC * h(E))
lo, hi = 0.0, 400.0
while hi - lo > 0.05:
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if R_single(mid) > 0 else (lo, mid)
D_single = lo
eta_end = eta_of(D_single)
print(f"  однофотонный источник: ключ до L = {D_single:.1f} км; там eta = {eta_end:.2e} = {eta_end / Y0:.1f} Y0, QBER = {en(1, eta_end):.4f}")
assert D_single > D_inf and 5 < eta_end / Y0 < 50
for L in (50, 100, 150):
    print(f"    L = {L} км: R = {max(0, R_single(L)):.3e}, PLOB(eta) = {-log2(1 - eta_of(L)):.3e}")
# очищение ниже F = 1/2 ухудшает верность
assert bbpssw_formula(0.4)[0] < 0.4
print(f"  BBPSSW при F = 0.4: F' = {bbpssw_formula(0.4)[0]:.4f} < 0.4")

# ===========================================================================
section("10. Ответы к задачам")
print(f"  1: mu = 0.5: P(n>=2) = {1 - pois(0, 0.5) - pois(1, 0.5):.4f}, доля среди непустых = {(1 - pois(0, 0.5) - pois(1, 0.5)) / (1 - pois(0, 0.5)):.4f}")
print(f"  2: PNS необнаружима при eta < {PNS[0.5]:.4f} (mu = 0.5), {PNS[0.1]:.4f} (mu = 0.1)")
print(f"  3: предельные расстояния: {D_no:.1f} / {D_inf:.1f} / {D_vw:.1f} км")
print(f"  5: CHSH: eta > {eta_chsh:.4f}; ключ DI: eta > {eta_key:.4f}")
print(f"  6: BBPSSW F = 0.75 -> {PUR[0.75][0]:.4f} (p = {PUR[0.75][1]:.4f})")
print(f"  7: 1200 км: волокно {SAT[1200e3][2]:.0f} дБ, спутник (геометрия) {SAT[1200e3][1]:.1f} дБ")
print(f"  12: однофотонный BB84: до {D_single:.1f} км (eta = {eta_end / Y0:.1f} Y0)")
print(f"  8: CV: eta = 0.1, обратное согласование K = {CV[0.1][1]:.4f}, PLOB = {CV[0.1][2]:.4f}")

print(f"\nВсе проверки пройдены. Время работы: {time.time() - T_START:.1f} с")
