"""
Лекция II.9. Квантовые системы с непрерывными переменными: мода как осциллятор,
базис Фока и квадратуры; когерентные, сжатые и тепловые состояния; гауссовы
состояния и операции (вектор средних, ковариационная матрица, симплектические
матрицы, теорема Уильямсона, разложение Блоха–Мессии); функция Вигнера и её
отрицательность; гауссовы каналы (потери, усиление, аддитивный шум) и условие
полной положительности; гомодинное измерение в гауссовом формализме и сверка
с расчётом в базисе Фока.

Соглашения (Лекции/Обозначения.md, §21): hbar = 1, x = (a + a^+)/sqrt2,
p = (a - a^+)/(i sqrt2), [x, p] = i, дисперсия квадратуры вакуума 1/2 (как в ЛР15).
Вектор квадратур r = (x1, p1, x2, p2, ...); V_ij = <{dr_i, dr_j}>/2;
Omega = (+) [[0, 1], [-1, 0]]. D(alpha) = exp(alpha a^+ - alpha* a),
alpha = (x0 + i p0)/sqrt2; S(r) = exp(r (a^2 - a^+2)/2) сжимает x.
W(x, p) = (1/pi) int psi(x+y) psi*(x-y) exp(-2ipy) dy.

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
"""
import time
from math import factorial, comb, log2, log10, e as E_
import numpy as np
from scipy.linalg import expm, sqrtm
from scipy.special import eval_laguerre, gammaln

T_START = time.time()
rng = np.random.default_rng(2026)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def ops(N):
    a = np.diag(np.sqrt(np.arange(1, N)), 1).astype(complex)
    ad = a.conj().T
    x = (a + ad) / np.sqrt(2)
    p = (a - ad) / (1j * np.sqrt(2))
    return a, ad, x, p


def expect(O, psi):
    return np.vdot(psi, O @ psi)


def hermite_functions(nmax, xs):
    """phi_n(x) = <x|n>, n = 0..nmax-1 (устойчивая рекурсия)."""
    xs = np.asarray(xs, dtype=float)
    out = np.zeros((nmax, xs.size))
    out[0] = np.pi ** -0.25 * np.exp(-xs ** 2 / 2)
    if nmax > 1:
        out[1] = np.sqrt(2) * xs * out[0]
    for n in range(1, nmax - 1):
        out[n + 1] = np.sqrt(2 / (n + 1)) * xs * out[n] - np.sqrt(n / (n + 1)) * out[n - 1]
    return out


# ===========================================================================
section("1. Мода как осциллятор: базис Фока и квадратуры")
N = 80
a, ad, x, p = ops(N)
comm = x @ p - p @ x
assert np.allclose(comm[:N - 1, :N - 1], 1j * np.eye(N - 1))       # [x,p] = i вне последнего уровня
assert abs(comm[N - 1, N - 1] - 1j * (1 - N)) < 1e-9                  # артефакт обрезки
vac = np.zeros(N, complex); vac[0] = 1
assert np.isclose(expect(x @ x, vac), 0.5) and np.isclose(expect(p @ p, vac), 0.5)
Hosc = (x @ x + p @ p) / 2
assert np.allclose(Hosc[:N - 1, :N - 1], (ad @ a + 0.5 * np.eye(N))[:N - 1, :N - 1])
xs = np.linspace(-16, 16, 6401)
dx = xs[1] - xs[0]
PHI = hermite_functions(N, xs)
G = PHI @ PHI.T * dx
assert np.allclose(G, np.eye(N), atol=1e-8)
# x|n> = (sqrt(n)|n-1> + sqrt(n+1)|n+1>)/sqrt2 в координатном представлении
n0 = 5
lhs = xs * PHI[n0]
rhs = (np.sqrt(n0) * PHI[n0 - 1] + np.sqrt(n0 + 1) * PHI[n0 + 1]) / np.sqrt(2)
assert np.allclose(lhs, rhs, atol=1e-10)
print(f"  N = {N}: [x, p] = i на подпространстве n < N-1 (на последнем уровне — артефакт обрезки {comm[N-1, N-1]:.0f})")
print("  вакуум: Var x = Var p = 1/2; H = (x^2 + p^2)/2 = a^+a + 1/2; функции Эрмита ортонормированы")

# ===========================================================================
section("2. Когерентные состояния и оператор смещения")
NB = 160
aB, adB, xB, pB = ops(NB)


def D_op(alpha, Nout=N):
    return expm(alpha * adB - np.conj(alpha) * aB)[:Nout, :Nout]


def coh(alpha, Nn=N):
    n = np.arange(Nn)
    return np.exp(-abs(alpha) ** 2 / 2 + n * np.log(alpha + 0j) - 0.5 * gammaln(n + 1)) if alpha != 0 else \
        np.eye(Nn)[0].astype(complex)


alpha = 1.3 - 0.7j
ca = D_op(alpha) @ vac
assert np.allclose(ca, coh(alpha), atol=1e-10)
assert np.allclose(a @ ca, alpha * ca, atol=1e-8)                       # собственный вектор a
mx, mp = expect(x, ca).real, expect(p, ca).real
assert np.isclose(mx, np.sqrt(2) * alpha.real) and np.isclose(mp, np.sqrt(2) * alpha.imag)
vx = expect(x @ x, ca).real - mx ** 2
assert np.isclose(vx, 0.5)
pn = np.abs(ca) ** 2
nbar = np.sum(np.arange(N) * pn)
varn = np.sum(np.arange(N) ** 2 * pn) - nbar ** 2
assert np.isclose(nbar, abs(alpha) ** 2) and np.isclose(varn, abs(alpha) ** 2)
beta = -0.4 + 0.9j
ov = abs(np.vdot(coh(beta), coh(alpha))) ** 2
assert np.isclose(ov, np.exp(-abs(alpha - beta) ** 2))
DD = D_op(alpha, NB) @ D_op(beta, NB)
Dsum = np.exp((alpha * np.conj(beta) - np.conj(alpha) * beta) / 2) * D_op(alpha + beta, NB)
assert np.allclose(DD[:40, :40], Dsum[:40, :40], atol=1e-8)
# D(alpha) = exp(i(p0 x - x0 p)); D^+ x D = x + x0
x0, p0 = np.sqrt(2) * alpha.real, np.sqrt(2) * alpha.imag
assert np.allclose(expm(1j * (p0 * xB - x0 * pB))[:60, :60], D_op(alpha, 60), atol=1e-8)
Dm = expm(alpha * adB - np.conj(alpha) * aB)
assert np.allclose((Dm.conj().T @ xB @ Dm)[:60, :60], (xB + x0 * np.eye(NB))[:60, :60], atol=1e-8)
print(f"  alpha = {alpha}: <x> = sqrt2 Re alpha = {mx:.4f}, <p> = {mp:.4f}, Var x = {vx:.4f}; <n> = Var n = |alpha|^2 = {nbar:.4f}")
print(f"  |<beta|alpha>|^2 = exp(-|alpha-beta|^2) = {ov:.6f}; D(a)D(b) = e^((a b* - a* b)/2) D(a+b); D^+ x D = x + x0")

# ===========================================================================
section("3. Сжатые и тепловые состояния")


def S_op(r, Nout=N):
    return expm(r / 2 * (aB @ aB - adB @ adB))[:Nout, :Nout]


r = 0.8
sv = S_op(r) @ vac
vx = expect(x @ x, sv).real
vp = expect(p @ p, sv).real
assert np.isclose(vx, np.exp(-2 * r) / 2) and np.isclose(vp, np.exp(2 * r) / 2)
assert np.isclose(expect(ad @ a, sv).real, np.sinh(r) ** 2)
assert np.allclose(sv[1::2], 0)                                        # только чётные n
m = np.arange(N // 2)
c2m = (-np.tanh(r)) ** m * np.exp(0.5 * gammaln(2 * m + 1) - m * np.log(2) - gammaln(m + 1)) / np.sqrt(np.cosh(r))
assert np.allclose(sv[::2], c2m, atol=1e-10)
dB = lambda var: -10 * log10(2 * var)                                   # сжатие относительно вакуума
print(f"  r = {r}: Var x = e^(-2r)/2 = {vx:.5f}, Var p = e^(2r)/2 = {vp:.5f}, <n> = sinh^2 r = {np.sinh(r)**2:.5f}")
print(f"  коэффициенты Фока: c_2m = (-tanh r)^m sqrt((2m)!)/(2^m m!)/sqrt(cosh r) — совпадают")
print(f"  сжатие в дБ: S_dB = -10 log10(Var x / (1/2)) = 20 r log10 e = {20 * log10(E_):.3f} r")
for rr in (0.5, 1.0):
    print(f"    r = {rr}: {dB(np.exp(-2 * rr) / 2):.2f} дБ")
r10 = 10 / (20 * log10(E_))
r15 = 15 / (20 * log10(E_))
print(f"    10 дБ <-> r = {r10:.3f}; 15 дБ <-> r = {r15:.3f}; <n> при 10 дБ = {np.sinh(r10)**2:.3f}")
assert abs(dB(np.exp(-2) / 2) - 8.686) < 1e-3 and abs(r10 - 1.1513) < 1e-4
# тепловое состояние
nth = 1.5
pth = nth ** np.arange(N) / (nth + 1) ** (np.arange(N) + 1)
rho_th = np.diag(pth).astype(complex)
assert abs(np.trace(rho_th) - 1) < 1e-12
vth = np.trace(rho_th @ x @ x).real
assert np.isclose(vth, nth + 0.5) and np.isclose(np.trace(rho_th @ rho_th).real, 1 / (2 * nth + 1))
print(f"  тепловое n = {nth}: Var x = n + 1/2 = {vth:.4f}, чистота 1/(2n+1) = {np.trace(rho_th @ rho_th).real:.4f}")

# ===========================================================================
section("4. Гауссовы состояния: ковариационная матрица, симплектические преобразования")
Om1 = np.array([[0, 1], [-1, 0]])


def Omega(n):
    return np.kron(np.eye(n), Om1)


def moments(rho, quads):
    """Вектор средних и V_ij = <{dr_i, dr_j}>/2 для матрицы плотности rho."""
    d = np.array([np.trace(rho @ q).real for q in quads])
    n = len(quads)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            V[i, j] = 0.5 * np.trace(rho @ (quads[i] @ quads[j] + quads[j] @ quads[i])).real - d[i] * d[j]
    return d, V


def dm(psi):
    return np.outer(psi, psi.conj())


d_c, V_c = moments(dm(ca), [x, p])
assert np.allclose(V_c, np.eye(2) / 2, atol=1e-9) and np.allclose(d_c, [x0, p0])
d_s, V_s = moments(dm(sv), [x, p])
assert np.allclose(V_s, np.diag([np.exp(-2 * r), np.exp(2 * r)]) / 2, atol=1e-9)
# поворот фазы exp(-i phi n): x -> x cos phi + p sin phi
phi = 0.6
Rphi = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
Uphi = np.diag(np.exp(-1j * phi * np.arange(N)))
_, V_rot = moments(dm(Uphi @ sv), [x, p])
assert np.allclose(V_rot, Rphi @ V_s @ Rphi.T, atol=1e-9)
Ssq = np.diag([np.exp(-r), np.exp(r)])
for Sm in (Rphi, Ssq):
    assert np.allclose(Sm @ Om1 @ Sm.T, Om1)
print("  когерентное: V = I/2; сжатое: V = diag(e^-2r, e^2r)/2; поворот фазы и сжатие — симплектические матрицы")

# двухмодовые операции в базисе Фока (N2 уровней на моду)
N2 = 30
a1, ad1, x1, p1 = ops(N2)
I_ = np.eye(N2)
A = np.kron(a1, I_); B = np.kron(I_, a1)
X1, P1 = np.kron(x1, I_), np.kron(p1, I_)
X2, P2 = np.kron(I_, x1), np.kron(I_, p1)
Q2 = [X1, P1, X2, P2]
vac2 = np.zeros(N2 * N2, complex); vac2[0] = 1


def bs_sym(th):
    c, s = np.cos(th), np.sin(th)
    return np.block([[c * np.eye(2), s * np.eye(2)], [-s * np.eye(2), c * np.eye(2)]])


def tms_sym(rr):
    c, s = np.cosh(rr), np.sinh(rr)
    Zm = np.diag([1, -1])
    return np.block([[c * np.eye(2), -s * Zm], [-s * Zm, c * np.eye(2)]])


th = 0.5
UBS = expm(th * (A.conj().T @ B - A @ B.conj().T))
# подготовим некоторое гауссово состояние: сжатие моды 1 и смещение моды 2
Ssq1 = expm(0.4 / 2 * (A @ A - A.conj().T @ A.conj().T))
Dis2 = expm(0.5 * B.conj().T - 0.5 * B)
psi_in = Dis2 @ Ssq1 @ vac2
d_in, V_in = moments(dm(psi_in), Q2)
d_out, V_out = moments(dm(UBS @ psi_in), Q2)
SB = bs_sym(th)
assert np.allclose(SB @ Omega(2) @ SB.T, Omega(2))
assert np.allclose(V_out, SB @ V_in @ SB.T, atol=1e-6) and np.allclose(d_out, SB @ d_in, atol=1e-6)
rt = 0.5
UTMS = expm(rt * (A @ B - A.conj().T @ B.conj().T))
tmsv = UTMS @ vac2
_, V_tm = moments(dm(tmsv), Q2)
c2, s2 = np.cosh(2 * rt), np.sinh(2 * rt)
V_tm_th = 0.5 * np.block([[c2 * np.eye(2), -s2 * np.diag([1, -1])], [-s2 * np.diag([1, -1]), c2 * np.eye(2)]])
assert np.allclose(V_tm, V_tm_th, atol=1e-6)
assert np.allclose(tms_sym(rt) @ (np.eye(4) / 2) @ tms_sym(rt).T, V_tm_th)
# коэффициенты Фока TMSV: (1/cosh r) sum (-tanh r)^n |n,n>
cf = np.array([tmsv[n * N2 + n] for n in range(10)])
assert np.allclose(cf, (-np.tanh(rt)) ** np.arange(10) / np.cosh(rt), atol=1e-8)
print(f"  светоделитель exp(th(a^+b - ab^+)), th = {th}: V -> S V S^T, d -> S d (проверено в базисе Фока, N = {N2}^2)")
print(f"  двухмодовое сжатие exp(r(ab - a^+b^+)), r = {rt}: V = [[c I, -s Z], [-s Z, c I]]/2, c = ch 2r, s = sh 2r;"
      f" |TMSV> = sum (-th r)^n |nn>/ch r")
# 50:50 светоделитель на x- и p-сжатых входах даёт двухмодовое сжатое состояние
Vin = np.diag([np.exp(-2 * rt), np.exp(2 * rt), np.exp(2 * rt), np.exp(-2 * rt)]) / 2
Vbs = bs_sym(np.pi / 4) @ Vin @ bs_sym(np.pi / 4).T
Vepr = 0.5 * np.block([[c2 * np.eye(2), s2 * np.diag([1, -1])], [s2 * np.diag([1, -1]), c2 * np.eye(2)]])
assert np.allclose(Vbs, Vepr)
print("  50:50 светоделитель на x- и p-сжатых входах -> двухмодовое сжатое (ЭПР) состояние (с точностью до фазы)")

# принцип неопределённости, теорема Уильямсона, чистота
def symp_eigs(V):
    ev = np.linalg.eigvals(1j * Omega(V.shape[0] // 2) @ V)
    return np.sort(np.abs(ev.real))[::2]


for name, V in (("сжатое", V_s), ("тепловое", (nth + 0.5) * np.eye(2)), ("TMSV", V_tm_th),
                ("редуцированное TMSV", V_tm_th[:2, :2])):
    n_ = V.shape[0] // 2
    mineig = np.linalg.eigvalsh(V + 0.5j * Omega(n_)).min()
    nu = symp_eigs(V)
    pur = 1 / (2 ** n_ * np.sqrt(np.linalg.det(V)))
    assert mineig > -1e-9 and np.all(nu > 0.5 - 1e-9)
    print(f"  {name}: min eig(V + i Omega/2) = {mineig:+.2e}, симплектические собств. значения {np.round(nu, 4)}, "
          f"чистота 1/(2^n sqrt det V) = {pur:.4f}")
nbar_red = np.sinh(rt) ** 2
assert np.isclose(symp_eigs(V_tm_th[:2, :2])[0], nbar_red + 0.5)
E_ent = (nbar_red + 1) * log2(nbar_red + 1) - nbar_red * log2(nbar_red)
lam = np.abs(cf) ** 2
E_fock = -np.sum(lam * np.log2(lam))
print(f"  TMSV r = {rt}: мода 1 тепловая с n = sinh^2 r = {nbar_red:.4f}; энтропия запутанности "
      f"g(n) = {E_ent:.4f} бит (по Фоку {E_fock:.4f})")
assert abs(E_ent - E_fock) < 1e-3
# нарушение: V = diag(0.2, 0.5) нефизична
Vbad = np.diag([0.2, 0.5])
assert np.linalg.eigvalsh(Vbad + 0.5j * Om1).min() < 0
print(f"  V = diag(0.2, 0.5): det = {np.linalg.det(Vbad):.2f} < 1/4 -> min eig(V + i Omega/2) < 0, нефизична")

# разложение Блоха–Мессии для случайного двухмодового симплектического S
def rot_sym(ph1, ph2):
    R = lambda f: np.array([[np.cos(f), np.sin(f)], [-np.sin(f), np.cos(f)]])
    return np.block([[R(ph1), np.zeros((2, 2))], [np.zeros((2, 2)), R(ph2)]])


def sq_sym(r1, r2):
    return np.diag([np.exp(-r1), np.exp(r1), np.exp(-r2), np.exp(r2)])


Srand = rot_sym(0.3, 1.1) @ bs_sym(0.7) @ sq_sym(0.5, -0.2) @ bs_sym(0.2) @ rot_sym(-0.4, 0.9) @ sq_sym(0.1, 0.3) @ bs_sym(1.0)
assert np.allclose(Srand @ Omega(2) @ Srand.T, Omega(2))
# комплексное представление: (a1, a2, a1^+, a2^+) -> [[Ac, Bc], [Bc*, Ac*]]
M = np.zeros((4, 4), complex)  # r = M xi, xi = (a1, a2, a1^+, a2^+)
for k in range(2):
    M[2 * k, k] = M[2 * k, 2 + k] = 1 / np.sqrt(2)
    M[2 * k + 1, k] = 1 / (1j * np.sqrt(2)); M[2 * k + 1, 2 + k] = -1 / (1j * np.sqrt(2))
Sc = np.linalg.inv(M) @ Srand @ M
Ac, Bc = Sc[:2, :2], Sc[:2, 2:]
assert np.allclose(Sc[2:, 2:], Ac.conj()) and np.allclose(Sc[2:, :2], Bc.conj())
U1, ch, V1h = np.linalg.svd(Ac)
Dm_ = U1.conj().T @ Bc @ V1h.T            # должно быть диагональным
assert np.allclose(Dm_ - np.diag(np.diag(Dm_)), 0, atol=1e-9)
assert np.allclose(np.abs(np.diag(Dm_)), np.sqrt(ch ** 2 - 1), atol=1e-9)
r_bm = np.arccosh(ch)
sv_S = np.sort(np.linalg.svd(Srand, compute_uv=False))
assert np.allclose(np.sort(np.concatenate([np.exp(r_bm), np.exp(-r_bm)])), sv_S)
print(f"  Блох–Мессия: A = U ch(R) V^+, B = U sh(R) e^(i theta) V^T; сжатия r = {np.round(r_bm, 4)}; "
      f"сингулярные числа S = e^(+-r) — совпадают")

# ===========================================================================
section("5. Функция Вигнера")
h = 0.02
K = 700
zgrid = np.arange(-2 * K, 2 * K + 1) * h          # точки x + y и x - y
NW = 100
PHIz = hermite_functions(NW, zgrid)


def wigner_pure(c, xi_idx, pgrid):
    """W(x_i, p) на сетке x_i = i h (i из xi_idx), p — массив; psi = sum c_n phi_n."""
    psi = c[:NW] @ PHIz if len(c) >= NW else np.pad(c, (0, NW - len(c))) @ PHIz
    js = np.arange(-K, K + 1)
    phase = np.exp(-2j * np.outer(pgrid, js * h))
    W = np.zeros((len(xi_idx), len(pgrid)))
    for a_, i in enumerate(xi_idx):
        f = psi[2 * K + i + js] * np.conj(psi[2 * K + i - js])
        W[a_] = (phase @ f).real * h / np.pi
    return W, psi


def wigner_rho(rho, xi_idx, pgrid):
    w, v = np.linalg.eigh(rho)
    W = 0
    for lam_, vec in zip(w, v.T):
        if lam_ > 1e-12:
            W = W + lam_ * wigner_pure(vec, xi_idx, pgrid)[0]
    return W


gstep = 5
xi_idx = np.arange(-300, 301, gstep)           # x от -6 до 6 с шагом 0,1
xg = xi_idx * h
pg = np.arange(-6, 6.0001, gstep * h)
dA = (gstep * h) ** 2


def fock(n, Nn=NW):
    v = np.zeros(Nn, complex); v[n] = 1
    return v


def cat(alpha, sign, Nn=NW):
    v = coh(alpha, Nn) + sign * coh(-alpha, Nn)
    return v / np.linalg.norm(v)


W0, _ = wigner_pure(fock(0), xi_idx, pg)
XX, PP = np.meshgrid(xg, pg, indexing="ij")
assert np.allclose(W0, np.exp(-XX ** 2 - PP ** 2) / np.pi, atol=1e-10)
assert abs(W0.sum() * dA - 1) < 1e-6
RES_W = {}
for n in (1, 2, 3):
    Wn, psi_n = wigner_pure(fock(n), xi_idx, pg)
    Wn_th = (-1) ** n / np.pi * np.exp(-XX ** 2 - PP ** 2) * eval_laguerre(n, 2 * (XX ** 2 + PP ** 2))
    assert np.allclose(Wn, Wn_th, atol=1e-9)
    neg = -Wn[Wn < 0].sum() * dA
    RES_W[f"|{n}>"] = (Wn[60, 60], neg)
    assert np.isclose(Wn[60, 60], (-1) ** n / np.pi)
assert abs(RES_W["|1>"][1] - (2 / np.sqrt(E_) - 1)) < 2e-3
# точный объём отрицательной части для |n>: (int_0^inf e^-u |L_n(2u)| du - 1)/2 (радиальный интеграл)
from scipy.integrate import quad
NEG_EXACT = {}
for n in (1, 2, 3):
    roots = np.sort(np.polynomial.laguerre.lagroots([0] * n + [1])) / 2
    pts = np.concatenate([[0], roots, [60]])
    tot = sum(quad(lambda u: np.exp(-u) * abs(eval_laguerre(n, 2 * u)), pts[i], pts[i + 1])[0] for i in range(len(pts) - 1))
    NEG_EXACT[n] = (tot - 1) / 2
    assert abs(NEG_EXACT[n] - RES_W[f"|{n}>"][1]) < 2e-3
assert abs(NEG_EXACT[1] - (2 / np.sqrt(E_) - 1)) < 1e-10
# маргиналы: int W dp = |psi(x)|^2
W3, psi3 = wigner_pure(fock(3), xi_idx, pg)
marg = W3.sum(1) * gstep * h
assert np.allclose(marg, np.abs(psi3[2 * K + xi_idx]) ** 2, atol=1e-6)
print("  W вакуума = exp(-x^2 - p^2)/pi; W_n = (-1)^n/pi exp(-r^2) L_n(2 r^2); маргинал = |psi(x)|^2; нормировка 1")
for kname, (w00, neg) in RES_W.items():
    print(f"    {kname}: W(0,0) = {w00:+.4f} = (-1)^n/pi; объём отрицательной части {neg:.4f}")
print(f"    для |1>: 2/sqrt(e) - 1 = {2 / np.sqrt(E_) - 1:.4f}; точные (радиальный интеграл): "
      + ", ".join(f"|{n}>: {v:.4f}" for n, v in NEG_EXACT.items()) + " (сетка с шагом 0,1 — погрешность ~1e-3)")
# формула через смещённую чётность: W(alpha) = (1/pi) Tr[rho D(alpha) Par D(alpha)^+]
Par = np.diag((-1.0) ** np.arange(NB))
c3 = np.zeros(NB, complex); c3[3] = 1
for (xx, pp) in ((0.5, -0.3), (1.2, 0.4)):
    al = (xx + 1j * pp) / np.sqrt(2)
    Dfull = expm(al * adB - np.conj(al) * aB)
    w_par = (np.vdot(c3, Dfull @ Par @ Dfull.conj().T @ c3)).real / np.pi
    w_grid = wigner_pure(fock(3), [int(round(xx / h))], np.array([pp]))[0][0, 0]
    assert abs(w_par - w_grid) < 1e-8
print("  W(alpha) = (1/pi) Tr[rho D(alpha) Pi D(alpha)^+] — совпадает с интегральной формулой; |W| <= 1/pi")
# кошачьи состояния
for al_ in (1.0, 2.0):
    for sgn, nm in ((+1, "чётное"), (-1, "нечётное")):
        Wc, _ = wigner_pure(cat(al_, sgn), xi_idx, pg)
        assert abs(Wc.sum() * dA - 1) < 1e-5 and np.isclose(Wc[60, 60], sgn / np.pi, atol=1e-8)
        neg = -Wc[Wc < 0].sum() * dA
        RES_W[f"cat {nm} alpha={al_}"] = (Wc[60, 60], neg)
        print(f"    кошка {nm}, alpha = {al_}: W(0,0) = {Wc[60, 60]:+.4f}, объём отрицательной части {neg:.4f}")
# структура W кошки (задача 8): пики в x = +-sqrt2 alpha, полосы cos(2 sqrt2 alpha p) у x = 0
al_ = 1.5
Wc, _ = wigner_pure(cat(al_, +1), xi_idx, pg)
Nc2 = 1 / (2 * (1 + np.exp(-2 * al_ ** 2)))
Wc_th = Nc2 / np.pi * (np.exp(-(XX - np.sqrt(2) * al_) ** 2 - PP ** 2) + np.exp(-(XX + np.sqrt(2) * al_) ** 2 - PP ** 2)
                       + 2 * np.exp(-XX ** 2 - PP ** 2) * np.cos(2 * np.sqrt(2) * al_ * PP))
assert np.allclose(Wc, Wc_th, atol=1e-8)
print(f"  кошка alpha = {al_}: W = N^2/pi [пики в x = +-sqrt2 alpha + 2 e^(-x^2-p^2) cos(2 sqrt2 alpha p)] — проверено")
# гауссовы состояния: W >= 0 и формула через V
al_g = 0.8 + 0.3j
gs = D_op(al_g, NW) @ S_op(0.5, NW) @ fock(0)
Wg, _ = wigner_pure(gs, xi_idx, pg)
dg, Vg = moments(dm(gs), list(ops(NW)[2:]))
Vi = np.linalg.inv(Vg)
dXX, dPP = XX - dg[0], PP - dg[1]
Wg_th = np.exp(-0.5 * (Vi[0, 0] * dXX ** 2 + 2 * Vi[0, 1] * dXX * dPP + Vi[1, 1] * dPP ** 2)) / (2 * np.pi * np.sqrt(np.linalg.det(Vg)))
assert np.allclose(Wg, Wg_th, atol=1e-8) and Wg.min() > -1e-10
print("  сжатое когерентное: W = exp(-(r-d)^T V^-1 (r-d)/2)/(2 pi sqrt det V) >= 0 (проверено на сетке)")
# тепловое состояние через смесь
Wth = wigner_rho(np.diag(nth ** np.arange(NW) / (nth + 1) ** (np.arange(NW) + 1)).astype(complex), xi_idx, pg)
assert np.allclose(Wth, np.exp(-(XX ** 2 + PP ** 2) / (2 * nth + 1)) / (np.pi * (2 * nth + 1)), atol=1e-6)
print(f"  тепловое n = {nth}: W = exp(-r^2/(2n+1))/(pi(2n+1)) (проверено суммой по Фоку)")

# ===========================================================================
section("6. Гауссовы каналы")


def loss_kraus(eta, Nn):
    Ks = []
    for k in range(Nn):
        Kk = np.zeros((Nn, Nn))
        for n in range(k, Nn):
            Kk[n - k, n] = np.sqrt(comb(n, k) * eta ** (n - k) * (1 - eta) ** k)
        Ks.append(Kk)
    return Ks


def amp_kraus(G, Nn):
    Ks = []
    for k in range(Nn):
        Kk = np.zeros((Nn, Nn))
        for n in range(0, Nn - k):
            Kk[n + k, n] = np.sqrt(comb(n + k, k) * (1 - 1 / G) ** k) * G ** (-(n + 1) / 2)
        Ks.append(Kk)
    return Ks


def apply_channel(Ks, rho):
    return sum(K @ rho @ K.T for K in Ks)


NC = 70
aC, adC, xC, pC = ops(NC)
psiC = D_op(0.6 + 0.2j, NC) @ S_op(0.4, NC) @ fock(0, NC)
rhoC = dm(psiC)
d0, V0 = moments(rhoC, [xC, pC])
eta = 0.7
Kl = loss_kraus(eta, NC)
assert np.allclose(sum(K.T @ K for K in Kl), np.eye(NC))
d1, V1 = moments(apply_channel(Kl, rhoC), [xC, pC])
assert np.allclose(V1, eta * V0 + (1 - eta) / 2 * np.eye(2), atol=1e-8) and np.allclose(d1, np.sqrt(eta) * d0, atol=1e-8)
print(f"  потери eta = {eta}: V -> eta V + (1-eta)/2 I, d -> sqrt(eta) d (Краус в базисе Фока)")
Gm = 1.3
Ka = amp_kraus(Gm, NC)
rho_small = dm(D_op(0.3, NC) @ S_op(0.2, NC) @ fock(0, NC))
d0s, V0s = moments(rho_small, [xC, pC])
out = apply_channel(Ka, rho_small)
assert abs(np.trace(out).real - 1) < 1e-6
d2, V2 = moments(out, [xC, pC])
assert np.allclose(V2, Gm * V0s + (Gm - 1) / 2 * np.eye(2), atol=1e-5) and np.allclose(d2, np.sqrt(Gm) * d0s, atol=1e-6)
print(f"  усиление G = {Gm}: V -> G V + (G-1)/2 I, d -> sqrt(G) d (Краус в базисе Фока)")
# потери + усиление 1/eta = аддитивный гауссов шум sigma^2 = (1-eta)/eta
Vla = (1 / eta) * (eta * V0 + (1 - eta) / 2 * np.eye(2)) + (1 / eta - 1) / 2 * np.eye(2)
assert np.allclose(Vla, V0 + (1 - eta) / eta * np.eye(2))
print(f"  потери eta, затем усиление 1/eta: V -> V + (1-eta)/eta I — классический шум смещений sigma^2 = {(1 - eta) / eta:.4f}")
# классический шум смещений: случайное смещение N(0, s2) по обеим квадратурам (усреднение по сетке Гаусса–Эрмита)
s2 = 0.05
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(12)
rho_cl = np.zeros((NC, NC), complex)
for ux, wx in zip(gh_x, gh_w):
    for up, wp in zip(gh_x, gh_w):
        al = np.sqrt(s2) * (ux + 1j * up) / np.sqrt(2)
        Dd = D_op(al, NC)
        rho_cl += wx * wp * Dd @ rho_small @ Dd.conj().T
rho_cl /= (np.sqrt(2 * np.pi)) ** 2
d3, V3 = moments(rho_cl, [xC, pC])
assert np.allclose(V3, V0s + s2 * np.eye(2), atol=1e-8)
print(f"  случайные смещения N(0, sigma^2) по x и p: V -> V + sigma^2 I (sigma^2 = {s2})")
# условие полной положительности Y + i(Omega - X Omega X^T)/2 >= 0
def cp_ok(Xm, Ym):
    return np.linalg.eigvalsh(Ym + 0.5j * (Om1 - Xm @ Om1 @ Xm.T)).min() > -1e-12


assert cp_ok(np.sqrt(eta) * np.eye(2), (1 - eta) / 2 * np.eye(2))
assert cp_ok(np.sqrt(Gm) * np.eye(2), (Gm - 1) / 2 * np.eye(2))
assert not cp_ok(np.sqrt(Gm) * np.eye(2), 0 * np.eye(2))
assert not cp_ok(np.sqrt(Gm) * np.eye(2), 0.99 * (Gm - 1) / 2 * np.eye(2))
print("  условие CP: Y + i(Omega - X Omega X^T)/2 >= 0; бесшумный усилитель (Y = 0) и усилитель с шумом "
      "меньше (G-1)/2 нефизичны")
# потери на сжатом состоянии 10 дБ (задача 9)
Vsq10 = np.diag([np.exp(-2 * r10), np.exp(2 * r10)]) / 2
SQ_LOSS = {}
for et in (0.9, 0.5):
    Vl = et * Vsq10 + (1 - et) / 2 * np.eye(2)
    SQ_LOSS[et] = dB(Vl[0, 0])
    print(f"  10 дБ сжатия после потерь eta = {et}: {SQ_LOSS[et]:.2f} дБ")
assert SQ_LOSS[0.5] < 3.0103 and abs(SQ_LOSS[0.9] - (-10 * log10(0.9 * 0.1 + 0.1))) < 1e-9
# тепловой шум: потери с тепловым окружением N_env
Nenv = 0.2
Vt = eta * V0 + (1 - eta) * (Nenv + 0.5) * np.eye(2)
assert cp_ok(np.sqrt(eta) * np.eye(2), (1 - eta) * (Nenv + 0.5) * np.eye(2))

# ===========================================================================
section("7. Гомодинное измерение в гауссовом формализме (сверка с базисом Фока)")
# TMSV r = rt; измеряем x моды 2 с результатом xm; условная ковариация моды 1:
# V_A' = V_A - C (Pi V_B Pi)^MP C^T, Pi = diag(1, 0); d_A' = d_A + C (Pi V_B Pi)^MP (xm - d_B)
VA, VB, Cc = V_tm_th[:2, :2], V_tm_th[2:, 2:], V_tm_th[:2, 2:]
Pi = np.diag([1.0, 0.0])
MP = np.linalg.pinv(Pi @ VB @ Pi)
VA_cond = VA - Cc @ MP @ Cc.T
xm = 0.7
dA_cond = Cc @ MP @ np.array([xm, 0.0])
# Фок: <x = xm| на моде 2
phi_xm = hermite_functions(N2, [xm])[:, 0]
psiA = tmsv.reshape(N2, N2) @ phi_xm
psiA /= np.linalg.norm(psiA)
dA_f, VA_f = moments(dm(psiA), [x1, p1])
assert np.allclose(VA_f, VA_cond, atol=1e-6) and np.allclose(dA_f, dA_cond, atol=1e-6)
print(f"  TMSV r = {rt}, x_2 = {xm}: условная мода 1: V' = {np.round(VA_cond, 5).tolist()}, d' = {np.round(dA_cond, 5).tolist()}")
print(f"  Var x_1 после измерения = 1/(2 ch 2r) = {1 / (2 * np.cosh(2 * rt)):.5f} (без измерения ch(2r)/2 = {np.cosh(2 * rt) / 2:.5f})")
assert np.isclose(VA_cond[0, 0], 1 / (2 * np.cosh(2 * rt)))
assert np.isclose(dA_cond[0], -np.tanh(2 * rt) * xm)
# стоимость гауссового моделирования: d (2n) и V (2n x 2n) против N^n амплитуд
for n_modes in (10, 100):
    print(f"  {n_modes} мод: гауссово описание {2 * n_modes + (2 * n_modes) * (2 * n_modes + 1) // 2} чисел; "
          f"Фок с N = 10 на моду: 10^{n_modes} амплитуд")

# ===========================================================================
section("8. Ответы к задачам")
print(f"  1: r = 1: {dB(np.exp(-2) / 2):.2f} дБ, <n> = sinh^2 1 = {np.sinh(1) ** 2:.4f}; 10 дБ: r = {r10:.4f}")
nb6 = np.sinh(1.0) ** 2
E6 = (nb6 + 1) * log2(nb6 + 1) - nb6 * log2(nb6)
print(f"  6: TMSV r = 1: n = {nb6:.4f}, E = {E6:.4f} бит; r = 0.5: E = {E_ent:.4f} бит")
print(f"  7: |1>: W(0,0) = {-1 / np.pi:.4f}, объём отрицательной части 2/sqrt(e) - 1 = {2 / np.sqrt(E_) - 1:.4f}")
print(f"  8: кошка alpha = 2: объём отрицательной части чётной {RES_W['cat чётное alpha=2.0'][1]:.4f}, "
      f"нечётной {RES_W['cat нечётное alpha=2.0'][1]:.4f}")
print(f"  9: 10 дБ после eta = 0.9: {SQ_LOSS[0.9]:.2f} дБ; после eta = 0.5: {SQ_LOSS[0.5]:.2f} дБ")
r11 = 0.5 * np.arccosh(10)
assert np.isclose(dB(1 / (2 * np.cosh(2 * r11))), 10)
print(f"  11: Var x_1 | x_2 = 1/(2 ch 2r); r = 0.5: {1 / (2 * np.cosh(1)):.4f}; <x_1> = -th(2r) x_2; 10 дБ при r = {r11:.4f}")

print(f"\nВсе проверки пройдены. Время работы: {time.time() - T_START:.1f} с")
