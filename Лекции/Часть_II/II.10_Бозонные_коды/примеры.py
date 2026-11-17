"""
Лекция II.10. Бозонные коды: условия Книлла–Лафламма в осцилляторе и
биномиальный код; восстановление «транспонированным каналом» и порядок ошибки
при потерях; двухкомпонентный кошачий кубит (подавление переворотов бита) и
четырёхкомпонентный cat-код (точка «sweet spot»); GKP-код: алгебра смещений,
приближённые состояния в базисе Фока, сжатие в децибелах, ожидания
стабилизаторов; ошибка декодирования при гауссовом шуме (точная, erfc,
Монте-Карло); конечное сжатие и потери: базис Фока против гауссовой модели;
аналоговая информация синдрома: GKP + код повторения, жёсткое и мягкое
декодирование.

Соглашения (Лекции/Обозначения.md, §21–§22): hbar = 1, x = (a + a^+)/sqrt2,
дисперсия вакуума 1/2. Квадратный GKP-код: стабилизаторы S_x = exp(-i 2 sqrt(pi) p),
S_p = exp(i 2 sqrt(pi) x); X_L = exp(-i sqrt(pi) p), Z_L = exp(i sqrt(pi) x);
приближённые состояния exp(-Delta^2 n)|mu_L>; S_dB = -10 log10(Delta^2).

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
"""
import time
from math import comb, log10, sqrt, pi
import numpy as np
from scipy.linalg import expm, sqrtm
from scipy.special import erfc, gammaln
from scipy.stats import norm
from scipy.optimize import brentq

T_START = time.time()
rng = np.random.default_rng(2026)
SP = sqrt(pi)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def ops(N):
    a = np.diag(np.sqrt(np.arange(1, N)), 1).astype(complex)
    return a, a.conj().T


def fock(n, N):
    v = np.zeros(N, complex); v[n] = 1
    return v


def coh(alpha, N):
    n = np.arange(N)
    if alpha == 0:
        return fock(0, N)
    return np.exp(-abs(alpha) ** 2 / 2 + n * np.log(alpha + 0j) - 0.5 * gammaln(n + 1))


def loss_kraus(eta, N, kmax=None):
    Ks = []
    for k in range(N if kmax is None else kmax + 1):
        K = np.zeros((N, N))
        for n in range(k, N):
            K[n - k, n] = np.sqrt(comb(n, k) * eta ** (n - k) * (1 - eta) ** k)
        Ks.append(K)
    return Ks


def hermite_functions(nmax, xs):
    xs = np.asarray(xs, dtype=float)
    out = np.zeros((nmax, xs.size))
    out[0] = np.pi ** -0.25 * np.exp(-xs ** 2 / 2)
    if nmax > 1:
        out[1] = np.sqrt(2) * xs * out[0]
    for n in range(1, nmax - 1):
        out[n + 1] = np.sqrt(2 / (n + 1)) * xs * out[n] - np.sqrt(n / (n + 1)) * out[n - 1]
    return out


def kl_matrix(code, errs):
    """Матрицы <i_L|E_a^+ E_b|j_L>; возвращает максимальное нарушение условий КЛ."""
    viol = 0.0
    for Ea in errs:
        for Eb in errs:
            M = np.array([[np.vdot(ci, Ea.conj().T @ Eb @ cj) for cj in code] for ci in code])
            viol = max(viol, abs(M[0, 1]), abs(M[1, 0]), abs(M[0, 0] - M[1, 1]))
    return viol


def transpose_recovery_fidelity(code, Ks):
    """Энтанглмент-верность кода с восстановлением транспонированным каналом (Петц):
    R(rho) = sum_k P K_k^+ N(P)^(-1/2) rho N(P)^(-1/2) K_k P."""
    N = len(code[0])
    B = np.array(code).T                    # N x 2
    P = B @ B.conj().T
    NP = sum(K @ P @ K.conj().T for K in Ks)
    w, v = np.linalg.eigh(NP)
    inv_sqrt = v @ np.diag([1 / np.sqrt(x) if x > 1e-14 else 0 for x in w]) @ v.conj().T
    # композиция R o N на логическом пространстве: операторы Крауса B^+ P K_k^+ NP^{-1/2} K_l B
    Fe = 0.0
    for Kk in Ks:
        Rk = B.conj().T @ Kk.conj().T @ inv_sqrt
        for Kl in Ks:
            M = Rk @ Kl @ B
            Fe += abs(np.trace(M)) ** 2 / 4
    return Fe.real


# ===========================================================================
section("1. Код в осцилляторе: условия Книлла–Лафламма, биномиальный код")
N = 40
a, ad = ops(N)
n_op = ad @ a
binom = [(fock(0, N) + fock(4, N)) / np.sqrt(2), fock(2, N)]
fockq = [fock(0, N), fock(1, N)]
v1 = kl_matrix(binom, [np.eye(N), a])
v2 = kl_matrix(binom, [np.eye(N), a, a @ a])
assert v1 < 1e-12 and v2 > 0.1
nbar_b = [np.vdot(c, n_op @ c).real for c in binom]
assert np.allclose(nbar_b, [2, 2])
print(f"  биномиальный код |0_L> = (|0> + |4>)/sqrt2, |1_L> = |2>: <n> = {nbar_b}; КЛ для {{I, a}}: нарушение {v1:.1e};"
      f" для {{I, a, a^2}}: {v2:.2f}")
print(f"  кубит Фока {{|0>, |1>}}: КЛ для {{I, a}}: нарушение {kl_matrix(fockq, [np.eye(N), a]):.2f} (не исправляет потерю)")
# порядок ошибки при потерях с восстановлением транспонированным каналом
GAM = [0.01, 0.02, 0.04]
FID = {"Фок": [], "биномиальный": []}
for g in GAM:
    Ks = loss_kraus(1 - g, N)
    FID["Фок"].append(1 - transpose_recovery_fidelity(fockq, Ks))
    FID["биномиальный"].append(1 - transpose_recovery_fidelity(binom, Ks))
slopes = {k: np.polyfit(np.log(GAM), np.log(v), 1)[0] for k, v in FID.items()}
for k, v in FID.items():
    print(f"  {k}: 1 - F_e при gamma = {GAM}: " + ", ".join(f"{x:.2e}" for x in v) + f"; наклон {slopes[k]:.2f}")
assert 0.9 < slopes["Фок"] < 1.1 and 1.85 < slopes["биномиальный"] < 2.15
BIN_RATIO = FID["Фок"][0] / FID["биномиальный"][0]

# ===========================================================================
section("2. Кошачьи коды")
N = 60
a, ad = ops(N)
n_op = ad @ a


def cat(alpha, sign):
    v = coh(alpha, N) + sign * coh(-alpha, N)
    return v / np.linalg.norm(v)


# двухкомпонентный кошачий кубит: |+-_L> = |C+->, |0/1_L> = (|C+> +- |C->)/sqrt2 ~ |+-alpha>
print("  alpha^2   <alpha|-alpha>   |<1_L|n|0_L>|/<n>   |<1_L|a|0_L>|/|<0_L|a|0_L>|")
CAT2 = {}
for al2 in (1.0, 2.0, 4.0):
    al = sqrt(al2)
    cp, cm = cat(al, +1), cat(al, -1)
    l0, l1 = (cp + cm) / np.sqrt(2), (cp - cm) / np.sqrt(2)
    ov = abs(np.vdot(coh(-al, N), coh(al, N)))
    assert np.isclose(ov, np.exp(-2 * al2))
    bf_n = abs(np.vdot(l1, n_op @ l0)) / np.vdot(l0, n_op @ l0).real
    bf_a = abs(np.vdot(l1, a @ l0)) / abs(np.vdot(l0, a @ l0))
    CAT2[al2] = (ov, bf_n, bf_a)
    print(f"   {al2:4.1f}     {ov:.3e}        {bf_n:.3e}            {bf_a:.3e}")
    # потеря фотона меняет чётность: a|C+> пропорционально |C->
    ac = a @ cp
    assert np.isclose(abs(np.vdot(cm, ac)) / np.linalg.norm(ac), 1)
for al2 in (2.0, 4.0):   # переворот бита подавлен как e^{-2 alpha^2}
    assert 1.5 < CAT2[al2][1] / np.exp(-2 * al2) < 2.5 and abs(CAT2[al2][2] / np.exp(-2 * al2) - 1) < 0.05
# в логическом базисе a ~ alpha Z_L: <0_L|a|0_L> ~ alpha, <1_L|a|1_L> ~ -alpha
al = 2.0
cp, cm = cat(al, +1), cat(al, -1)
l0, l1 = (cp + cm) / np.sqrt(2), (cp - cm) / np.sqrt(2)
A_L = np.array([[np.vdot(u, a @ v) for v in (l0, l1)] for u in (l0, l1)])
print(f"  alpha = 2: матрица a в базисе {{|0_L>, |1_L>}}: diag = {np.round(np.diag(A_L).real, 4)}, "
      f"внедиагональ {abs(A_L[0, 1]):.1e} -> a ~ alpha Z_L (переворот фазы)")
assert abs(A_L[0, 0] - al) < 1e-3 and abs(A_L[1, 1] + al) < 1e-3

# четырёхкомпонентный cat-код
def cat4(alpha):
    cpa, cpi = coh(alpha, N) + coh(-alpha, N), coh(1j * alpha, N) + coh(-1j * alpha, N)
    z = cpa + cpi
    o = cpa - cpi
    return z / np.linalg.norm(z), o / np.linalg.norm(o)


def nbar_diff(alpha):
    z, o = cat4(alpha)
    return np.vdot(z, n_op @ z).real - np.vdot(o, n_op @ o).real


z, o = cat4(1.5)
nz = np.nonzero(np.abs(z) > 1e-12)[0]
no = np.nonzero(np.abs(o) > 1e-12)[0]
assert np.all(nz % 4 == 0) and np.all(no % 4 == 2)
alphas = np.linspace(0.8, 3.0, 221)
dif = [nbar_diff(x) for x in alphas]
roots = [brentq(nbar_diff, alphas[i], alphas[i + 1]) for i in range(len(alphas) - 1) if dif[i] * dif[i + 1] < 0]
sweet = roots[0]
print(f"  4-cat: |0_L> — числа фотонов 0 mod 4, |1_L> — 2 mod 4; потеря фотона переводит в 3 и 1 mod 4")
print(f"  разность <n> логических состояний обращается в 0 при alpha = " + ", ".join(f"{r_:.4f}" for r_ in roots[:3])
      + f" (alpha^2 = {sweet ** 2:.4f} для первой)")
for alv in (1.2, sweet, 2.5):
    zz, oo = cat4(alv)
    print(f"    alpha = {alv:.4f}: нарушение КЛ для {{I, a}} = {kl_matrix([zz, oo], [np.eye(N), a]):.2e}")
zz, oo = cat4(sweet)
assert kl_matrix([zz, oo], [np.eye(N), a]) < 1e-9
FID4 = {}
for alv, nm in ((sweet, "sweet spot"), (1.2, "alpha = 1.2")):
    zz, oo = cat4(alv)
    FID4[nm] = [1 - transpose_recovery_fidelity([zz, oo], loss_kraus(1 - g, N)) for g in GAM]
    s_ = np.polyfit(np.log(GAM), np.log(FID4[nm]), 1)[0]
    FID4[nm + "_slope"] = s_
    print(f"  4-cat ({nm}): 1 - F_e при gamma = {GAM}: " + ", ".join(f"{x:.2e}" for x in FID4[nm]) + f"; наклон {s_:.2f}")
assert 1.8 < FID4["sweet spot_slope"] < 2.2 and FID4["alpha = 1.2_slope"] < 1.6

# ===========================================================================
section("3. GKP-код: алгебра смещений и приближённые состояния")
# соотношение Вейля: e^{iux} e^{ivp} = e^{-iuv} e^{ivp} e^{iux}; проверка на маломерной части большого пространства
NB = 300
aB, adB = ops(NB)
xB = (aB + adB) / np.sqrt(2)
pB = (aB - adB) / (1j * np.sqrt(2))
Sx = expm(-1j * 2 * SP * pB)
Sp = expm(1j * 2 * SP * xB)
XL = expm(-1j * SP * pB)
ZL = expm(1j * SP * xB)
k_ = 40
assert np.allclose((Sx @ Sp)[:k_, :k_], (Sp @ Sx)[:k_, :k_], atol=1e-8)
assert np.allclose((XL @ ZL)[:k_, :k_], -(ZL @ XL)[:k_, :k_], atol=1e-8)
assert np.allclose((XL @ Sp)[:k_, :k_], (Sp @ XL)[:k_, :k_], atol=1e-8)
assert np.allclose((XL @ XL)[:k_, :k_], Sx[:k_, :k_], atol=1e-8)
print("  S_x S_p = S_p S_x (фаза e^{-i 4 pi} = 1); X_L Z_L = -Z_L X_L (фаза e^{-i pi}); X_L^2 = S_x; X_L коммутирует с S_p")

xs = np.arange(-30, 30, 0.005)
dxs = xs[1] - xs[0]
NF = 220
PHI = hermite_functions(NF, xs)


def gkp_fock(mu, Delta, Nf=NF, smax=12):
    """Коэффициенты Фока e^{-Delta^2 n} sum_s phi_n((2s+mu) sqrt(pi)), нормированные."""
    pts = (2 * np.arange(-smax, smax + 1) + mu) * SP
    c = hermite_functions(Nf, pts).sum(1) * np.exp(-Delta ** 2 * np.arange(Nf))
    return c / np.linalg.norm(c)


def wavefunction(c):
    return c @ PHI[:len(c)]


DB_LIST = (6, 8, 10, 12)
GKP = {}
print("  дБ    Delta    <n>      1/(2D^2)-1/2   Var пика   tanh(D^2)/2   <S_p>     <S_x>     e^{-pi D^2}   P(n>=N-20)")
for dbv in DB_LIST:
    D = 10 ** (-dbv / 20)
    c0 = gkp_fock(0, D)
    psi = wavefunction(c0)
    nb = np.sum(np.arange(NF) * np.abs(c0) ** 2)
    tail = np.sum(np.abs(c0[NF - 20:]) ** 2)
    prob = np.abs(psi) ** 2
    assert abs(prob.sum() * dxs - 1) < 1e-8
    cen = np.abs(xs) < SP / 2
    var_pk = np.sum(xs[cen] ** 2 * prob[cen]) / np.sum(prob[cen])
    Sp_exp = np.sum(prob * np.exp(1j * 2 * SP * xs)) * dxs
    shift = int(round(2 * SP / dxs))
    # <psi| e^{-i 2 sqrt(pi) p} |psi> = int psi*(x) psi(x - 2 sqrt(pi)) dx; на сетке — сдвиг на ближайшее число шагов
    xs_sh = xs - 2 * SP
    psi_sh = np.interp(xs_sh, xs, psi.real)
    Sx_exp = np.sum(psi.conj() * psi_sh) * dxs
    GKP[dbv] = dict(D=D, c0=c0, nbar=nb, var=var_pk, Sp=Sp_exp.real, Sx=Sx_exp.real, tail=tail)
    print(f"  {dbv:2d}   {D:.4f}  {nb:7.3f}   {1 / (2 * D * D) - 0.5:7.3f}       {var_pk:.5f}    {np.tanh(D * D) / 2:.5f}"
          f"     {Sp_exp.real:.4f}    {Sx_exp.real:.4f}    {np.exp(-pi * D * D):.4f}       {tail:.1e}")
    assert tail < 1e-10
    assert abs(Sp_exp.imag) < 1e-8
for dbv in (10, 12):    # асимптотические формулы (малые Delta)
    g_ = GKP[dbv]
    assert abs(g_["var"] / (g_["D"] ** 2 / 2) - 1) < 0.02            # пик: Var ~ Delta^2/2
    assert abs(-10 * log10(2 * g_["var"]) - dbv) < 0.1               # дБ по ширине пика
    assert abs(g_["nbar"] - (1 / (2 * g_["D"] ** 2) - 0.5)) < 0.02    # <n> ~ 1/(2 Delta^2) - 1/2
    assert abs(g_["Sp"] - g_["Sx"]) < 0.002 and abs(g_["Sp"] / np.exp(-pi * g_["D"] ** 2) - 1) < 0.002
assert GKP[6]["nbar"] < 0.8 * (1 / (2 * GKP[6]["D"] ** 2) - 0.5)     # при 6 дБ асимптотика уже неточна
print("  при 10–12 дБ: Var пика = Delta^2/2, <n> = 1/(2 Delta^2) - 1/2, <S_p> = <S_x> = e^(-pi Delta^2) с точностью < 1 %;"
      " при 6 дБ асимптотика не работает")
# сходимость по размерности Фока (12 дБ)
c_lo = gkp_fock(0, GKP[12]["D"], Nf=120)
c_hi = gkp_fock(0, GKP[12]["D"], Nf=220)
conv = 1 - abs(np.vdot(np.pad(c_lo, (0, 100)), c_hi)) ** 2
print(f"  сходимость (12 дБ): 1 - |<psi_120|psi_220>|^2 = {conv:.1e}")
assert conv < 1e-6
# логические операторы на приближённых состояниях
D10 = GKP[10]["D"]
c0, c1 = gkp_fock(0, D10), gkp_fock(1, D10)
ov01 = abs(np.vdot(c0, c1))
print(f"  10 дБ: |<0~|1~>| = {ov01:.2e} (приближённые слова почти ортогональны)")
assert ov01 < 1e-2
x_fl = abs(np.vdot(np.pad(c1, (0, NB - NF)), XL @ np.pad(c0, (0, NB - NF)))) ** 2
z0 = np.vdot(np.pad(c0, (0, NB - NF)), ZL @ np.pad(c0, (0, NB - NF))).real
z1 = np.vdot(np.pad(c1, (0, NB - NF)), ZL @ np.pad(c1, (0, NB - NF))).real
print(f"  10 дБ: |<1~|X_L|0~>|^2 = {x_fl:.4f}, <0~|Z_L|0~> = {z0:+.4f}, <1~|Z_L|1~> = {z1:+.4f}")
# сдвиг на u даёт <D> ~ exp(-u^2 Delta^2/4): u = sqrt(pi) -> |<1~|X_L|0~>|^2 ~ e^{-pi D^2/2}, <Z_L> ~ e^{-pi D^2/4}
assert abs(x_fl - np.exp(-pi * D10 ** 2 / 2)) < 2e-3 and abs(z0 - np.exp(-pi * D10 ** 2 / 4)) < 2e-3 and abs(z1 + z0) < 1e-3, (x_fl, z0, z1)
print(f"  сравнение: e^(-pi D^2/2) = {np.exp(-pi * D10 ** 2 / 2):.4f}, e^(-pi D^2/4) = {np.exp(-pi * D10 ** 2 / 4):.4f}")

# логические клиффорды — гауссовы: H_L = e^{i pi n/2} (поворот на pi/2), S_L = e^{i x^2/2}
def gkp_raw(mu, Delta, Nf=NF, smax=12):
    pts = (2 * np.arange(-smax, smax + 1) + mu) * SP
    return hermite_functions(Nf, pts).sum(1) * np.exp(-Delta ** 2 * np.arange(Nf))


r0, r1 = gkp_raw(0, D10), gkp_raw(1, D10)
plus = (r0 + r1) / np.linalg.norm(r0 + r1)
minus = (r0 - r1) / np.linalg.norm(r0 - r1)
Fh = np.exp(1j * pi / 2 * np.arange(NF))
fH = abs(np.vdot(plus, Fh * c0)) ** 2
fH1 = abs(np.vdot(minus, Fh * c1)) ** 2
assert fH > 1 - 1e-9 and fH1 > 1 - 1e-9
psi0, psi1 = wavefunction(c0), wavefunction(c1)
ph = np.exp(1j * xs ** 2 / 2)
fS0 = abs(np.sum(psi0.conj() * ph * psi0) * dxs) ** 2
ovS1 = np.sum(psi1.conj() * ph * psi1) * dxs
print(f"  H_L = exp(i pi n/2): |<+~|H|0~>|^2 = {fH:.10f}, |<-~|H|1~>|^2 = {fH1:.10f} (затухание e^(-D^2 n) коммутирует с поворотом)")
print(f"  S_L = exp(i x^2/2): |<0~|S|0~>|^2 = {fS0:.4f}, <1~|S|1~> = {ovS1:.4f} (идеально: 1 и i)")
assert 0.75 < fS0 < 0.85 and abs(ovS1 / abs(ovS1) - 1j) < 0.01  # фаза верна, но сдвиг портит конечное сжатие

# ===========================================================================
section("4. Ошибка декодирования одномодового GKP при гауссовом шуме смещений")


def p_err_exact(sig):
    """P(округление u к ближайшему кратному sqrt(pi) даёт нечётное кратное), u ~ N(0, sig^2)."""
    k = np.arange(-60, 61)
    lo = (2 * k + 1) * SP - SP / 2
    hi = (2 * k + 1) * SP + SP / 2
    return float(np.sum(norm.cdf(hi / sig) - norm.cdf(lo / sig)))


def p_err_approx(sig):
    return erfc(SP / (2 * np.sqrt(2) * sig))


def decode_round(x):
    """Ближайшая точка решётки sqrt(pi) Z; возвращает бит (чётность кратного)."""
    return np.mod(np.rint(x / SP), 2).astype(int)


SIGS = (0.3, 0.4, 0.5, 0.6, 0.8)
P1 = {}
print("   sigma    точная        erfc          Монте-Карло (2e5)")
for s in SIGS:
    u = rng.normal(0, s, 200_000)
    mc = decode_round(u).mean()
    P1[s] = (p_err_exact(s), p_err_approx(s), mc)
    print(f"   {s:.1f}     {P1[s][0]:.4e}    {P1[s][1]:.4e}    {mc:.4e}")
    assert abs(mc - P1[s][0]) < 5 * np.sqrt(P1[s][0] / 200_000) + 1e-4
    assert P1[s][0] <= P1[s][1] + 1e-15
assert abs(P1[0.3][0] / P1[0.3][1] - 1) < 1e-6 and 1.001 < P1[0.8][1] / P1[0.8][0] < 1.01
print(f"  erfc(sqrt(pi)/(2 sqrt2 sigma)) = P(|u| > sqrt(pi)/2) завышает ошибку на P(|u| > 3 sqrt(pi)/2) и т. д. "
      f"(перескоки через период): при sigma = 0.8 отношение {P1[0.8][1] / P1[0.8][0]:.4f}")
# соответствие sigma и Delta: sigma^2 = Delta^2/2  ->  дБ = -10 log10(2 sigma^2)
for s in (0.3, 0.4, 0.5):
    print(f"    sigma = {s}: эквивалентное сжатие -10 log10(2 sigma^2) = {-10 * log10(2 * s * s):.2f} дБ")

# ===========================================================================
section("5. Конечное сжатие + шум: базис Фока против гауссовой модели")


def p_err_from_density(prob_x, sig_extra, scale=1.0):
    """Вероятность ошибочного бита при измерении x: распределение prob_x(xs), свёртка с N(0, sig_extra^2),
    затем деление на scale и округление к решётке sqrt(pi) Z; ошибка — нечётное кратное (для |0~>)."""
    if sig_extra > 0:
        ker_x = np.arange(-8 * sig_extra, 8 * sig_extra + dxs / 2, dxs)
        ker = np.exp(-ker_x ** 2 / (2 * sig_extra ** 2))
        ker /= ker.sum()
        prob_x = np.convolve(prob_x, ker, mode="same")
    bits = decode_round(xs / scale)
    return float(np.sum(prob_x[bits == 1]) * dxs)


print("   дБ   sigma   Фок (точно)    гаусс. модель sigma_eff^2 = sigma^2 + Delta^2/2   отношение")
CMP = {}
for dbv in (6, 8, 10, 12):
    D = GKP[dbv]["D"]
    prob0 = np.abs(wavefunction(GKP[dbv]["c0"])) ** 2
    for s in (0.0, 0.2, 0.3, 0.4):
        pf = p_err_from_density(prob0, s)
        pg = p_err_exact(np.sqrt(s * s + D * D / 2))
        CMP[(dbv, s)] = (pf, pg)
        print(f"   {dbv:2d}   {s:.1f}    {pf:.4e}     {pg:.4e}                                  {pf / pg:.3f}")
for dbv in (10, 12):
    for s in (0.2, 0.3, 0.4):
        pf, pg = CMP[(dbv, s)]
        assert abs(pf / pg - 1) < 0.15
print("  отношение Фок/гаусс: " + ", ".join(f"{d_} дБ, sigma = 0: {CMP[(d_, 0.0)][0] / CMP[(d_, 0.0)][1]:.3f}" for d_ in (6, 8, 10, 12)))

# потери: канал потерь в базисе Фока, затем измерение x и деление на sqrt(eta)
print("   eta    Фок (потери, затем x/sqrt(eta))   гаусс.: sigma_eff^2 = Delta^2/2 + (1-eta)/(2 eta)")
LOSS = {}
D = GKP[10]["D"]
c0 = GKP[10]["c0"]
rho0 = np.outer(c0, c0.conj())
for eta in (0.9, 0.95, 0.99):
    Ks = loss_kraus(eta, NF)
    rho = sum(K @ rho0 @ K.T for K in Ks)
    w, v = np.linalg.eigh(rho)
    prob = np.zeros_like(xs)
    for lam, vec in zip(w, v.T):
        if lam > 1e-12:
            prob += lam * np.abs(vec @ PHI) ** 2
    pf = p_err_from_density(prob, 0.0, scale=np.sqrt(eta))
    pg = p_err_exact(np.sqrt(D * D / 2 + (1 - eta) / (2 * eta)))
    LOSS[eta] = (pf, pg)
    print(f"   {eta:.2f}   {pf:.4e}                         {pg:.4e}")
    assert abs(pf / pg - 1) < 0.2
# потери + усиление 1/eta = шум смещений (1-eta)/eta (II.9, (5.6)) — для дальнейшей квантовой обработки
print("  для дальнейшей квантовой обработки (потери + усиление 1/eta): sigma^2 = (1-eta)/eta: "
      + ", ".join(f"eta = {e_}: {(1 - e_) / e_:.4f}" for e_ in (0.9, 0.95, 0.99)))

# ===========================================================================
section("6. Аналоговая информация: GKP + код повторения на 3 модах (гауссова модель)")


def bit_likelihoods(xm, sig):
    """L(b) = sum_k g(xm - (2k + b) sqrt(pi)); возвращает (L0, L1)."""
    k = np.arange(-6, 7)
    L0 = np.exp(-(xm[..., None] - 2 * k * SP) ** 2 / (2 * sig ** 2)).sum(-1)
    L1 = np.exp(-(xm[..., None] - (2 * k + 1) * SP) ** 2 / (2 * sig ** 2)).sum(-1)
    return L0, L1


SHOTS = 400_000
REP = {}
print("   sigma   одна мода     жёсткое (большинство)   мягкое (правдоподобие)   выигрыш")
for s in (0.4, 0.5, 0.6, 0.7):
    u = rng.normal(0, s, (SHOTS, 3))          # логический 0 на трёх модах, x-смещения
    bits = decode_round(u)
    hard = (bits.sum(1) >= 2).mean()
    L0, L1 = bit_likelihoods(u, s)
    soft = (np.log(L1).sum(1) > np.log(L0).sum(1)).mean()
    REP[s] = (p_err_exact(s), hard, soft)
    print(f"   {s:.1f}     {REP[s][0]:.4e}    {hard:.4e}              {soft:.4e}               {hard / soft:.2f}")
    p1 = p_err_exact(s)
    assert abs(hard - (3 * p1 ** 2 - 2 * p1 ** 3)) < 5 * np.sqrt(hard / SHOTS) + 2e-5
    assert soft < hard
assert REP[0.4][1] / REP[0.4][2] > 1.5
# простой пример: значения x1, x2, x3, на которых мягкое и жёсткое декодирование расходятся
ex = np.array([[0.80 * SP, 0.55 * SP, 0.05 * SP]])   # биты: 1, 1, 0 -> большинство 1
L0, L1 = bit_likelihoods(ex, 0.5)
soft_ex = int(np.log(L1).sum() > np.log(L0).sum())
assert decode_round(ex).tolist() == [[1, 1, 0]] and soft_ex == 0
print(f"  пример: x/sqrt(pi) = (0.80, 0.55, 0.05), sigma = 0.5: биты (1, 1, 0) -> большинство 1; "
      f"мягкое -> {soft_ex} (второй бит ненадёжен — у границы ячейки)")

# ===========================================================================
section("7. Ответы к задачам")
print(f"  2: биномиальный код: КЛ для {{I, a}} выполнены, <n> = 2 у обоих слов; при gamma = 0.01: 1 - F_e "
      f"Фок {FID['Фок'][0]:.2e}, биномиальный {FID['биномиальный'][0]:.2e}")
print(f"  4: <alpha|-alpha> = e^(-2 alpha^2): alpha^2 = 1, 2, 4: " + ", ".join(f"{CAT2[k][0]:.2e}" for k in (1.0, 2.0, 4.0)))
print(f"  5: 4-cat sweet spot alpha = {sweet:.4f} (alpha^2 = {sweet ** 2:.4f})")
print(f"  7: 10 дБ: Delta = {GKP[10]['D']:.4f}, <n> = {GKP[10]['nbar']:.3f}; 12 дБ: Delta = {GKP[12]['D']:.4f}, <n> = {GKP[12]['nbar']:.3f}")
print(f"  8: sigma = 0.5: точная {P1[0.5][0]:.4e}, erfc {P1[0.5][1]:.4e}")
print(f"  10: eta = 0.95, 10 дБ: Фок {LOSS[0.95][0]:.3e}, гаусс {LOSS[0.95][1]:.3e}")
print(f"  11: sigma = 0.5: одна мода {REP[0.5][0]:.3e}, жёсткое {REP[0.5][1]:.3e}, мягкое {REP[0.5][2]:.3e}")

NREQ = {}
for dbv in (10, 12):
    tailc = np.cumsum((np.abs(GKP[dbv]["c0"]) ** 2)[::-1])[::-1]
    NREQ[dbv] = int(np.argmax(tailc < 1e-8))
print(f"  12: уровней Фока для хвоста < 1e-8: 10 дБ — {NREQ[10]}, 12 дБ — {NREQ[12]}; "
      f"для 3 мод при 10 дБ: {NREQ[10]}^3 = {NREQ[10] ** 3:.1e}, для 49 мод: {NREQ[10]}^49 ~ 10^{49 * np.log10(NREQ[10]):.0f}")
assert 40 < NREQ[10] < 120

print(f"\nВсе проверки пройдены. Время работы: {time.time() - T_START:.1f} с")
