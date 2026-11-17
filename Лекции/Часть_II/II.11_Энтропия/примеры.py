"""
Лекция II.11. Энтропия: энтропия Шеннона и типичные последовательности;
энтропия фон Неймана и её свойства (границы, вогнутость, смеси, измерение,
мажоризация); совместная, условная (отрицательная!) энтропия и взаимная
информация; изотропные состояния и хэш-граница; относительная энтропия,
неравенство Клейна, неравенство Пинскера, монотонность при действии канала,
связь со свободной энергией; субаддитивность, Араки–Либ, сильная
субаддитивность и следствия на 10^4 случайных трёхкубитных состояниях;
непрерывность (Фаннес–Ауденарт); сжатие Шумахера: комбинаторный расчёт веса
типичного подпространства и прямая проверка верности сцепленности.

Соглашения (Лекции/Обозначения.md, §1, §6, §12, §23): log = log2 (биты),
S(rho) = -Tr rho log2 rho; S(A|B) = S(AB) - S(B); I(A:B) = S(A) + S(B) - S(AB);
S(rho||sigma) = Tr rho (log rho - log sigma); D(rho, sigma) = ||rho - sigma||_1 / 2;
первый тензорный множитель — старший (A в AB).

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
"""
import time
from math import comb, log2, log, sqrt
import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp

T_START = time.time()
rng = np.random.default_rng(2026)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def H(p):
    """Энтропия Шеннона распределения (биты)."""
    p = np.asarray(p, float).ravel()
    p = p[p > 1e-15]
    return float(-(p * np.log2(p)).sum())


def h(x):
    return H([x, 1 - x])


def S(rho):
    """Энтропия фон Неймана (биты)."""
    ev = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    return H(np.clip(ev, 0, None))


def S_batch(rhos):
    ev = np.clip(np.linalg.eigvalsh(rhos), 1e-300, None)
    return -(ev * np.log2(ev)).sum(-1)


def logm2(A):
    w, V = np.linalg.eigh((A + A.conj().T) / 2)
    return (V * np.log2(w)) @ V.conj().T


def rel_ent(rho, sigma):
    """S(rho||sigma) в битах; inf, если supp rho не содержится в supp sigma."""
    wr, Vr = np.linalg.eigh(rho)
    ws, Vs = np.linalg.eigh(sigma)
    if np.any(ws < 1e-13):
        # проверка носителя: проекция rho на ядро sigma
        ker = Vs[:, ws < 1e-13]
        if np.real(np.trace(ker.conj().T @ rho @ ker)) > 1e-12:
            return np.inf
    wr_c = np.clip(wr, 1e-300, None)
    t1 = float(np.sum(wr * np.log2(wr_c) * (wr > 1e-15)))
    ws_c = np.where(ws > 1e-13, ws, 1.0)
    logsig = (Vs * np.log2(ws_c)) @ Vs.conj().T
    t2 = float(np.real(np.trace(rho @ logsig)))
    return t1 - t2


def trdist(a, b):
    return 0.5 * np.abs(np.linalg.eigvalsh(a - b)).sum()


def ptrace(rho, dims, keep):
    """Частичный след: оставить подсистемы keep (список индексов)."""
    n = len(dims)
    r = rho.reshape(dims + dims)
    idx = list(range(n))
    trace_out = [i for i in idx if i not in keep]
    letters = "abcdefghijklmnop"
    in_row = [letters[i] for i in idx]
    in_col = [letters[i] if i in trace_out else letters[i].upper() for i in idx]
    out = "".join(letters[i] for i in keep) + "".join(letters[i].upper() for i in keep)
    res = np.einsum("".join(in_row) + "".join(in_col) + "->" + out, r)
    d = int(np.prod([dims[i] for i in keep]))
    return res.reshape(d, d)


def ptrace_batch(rhos, dims, keep):
    n = len(dims)
    r = rhos.reshape((rhos.shape[0],) + tuple(dims) * 2)
    letters = "abcdefghijklmnop"
    idx = list(range(n))
    in_row = [letters[i] for i in idx]
    in_col = [letters[i] if i not in keep else letters[i].upper() for i in idx]
    out = "z" + "".join(letters[i] for i in keep) + "".join(letters[i].upper() for i in keep)
    res = np.einsum("z" + "".join(in_row) + "".join(in_col) + "->" + out, r)
    d = int(np.prod([dims[i] for i in keep]))
    return res.reshape(rhos.shape[0], d, d)


def rand_rho(d, rank=None):
    rank = d if rank is None else rank
    G = rng.normal(size=(d, rank)) + 1j * rng.normal(size=(d, rank))
    r = G @ G.conj().T
    return r / np.trace(r).real


def rand_rho_batch(N, d, rank):
    G = rng.normal(size=(N, d, rank)) + 1j * rng.normal(size=(N, d, rank))
    r = G @ np.conj(np.swapaxes(G, 1, 2))
    return r / np.trace(r, axis1=1, axis2=2).real[:, None, None]


def rand_unitary(d):
    Z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, R = np.linalg.qr(Z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def rand_channel_kraus(din, dout, nk):
    """Случайный канал через случайную изометрию din -> dout*nk."""
    nk = max(nk, -(-din // dout))
    G = rng.normal(size=(dout * nk, din)) + 1j * rng.normal(size=(dout * nk, din))
    Q, _ = np.linalg.qr(G)
    return [Q[k * dout:(k + 1) * dout, :] for k in range(nk)]


def apply(Ks, rho):
    return sum(K @ rho @ K.conj().T for K in Ks)


def ket(*bits):
    v = np.zeros(2 ** len(bits)); v[int("".join(map(str, bits)), 2)] = 1
    return v


I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
phi_plus = (ket(0, 0) + ket(1, 1)) / sqrt(2)
BELL = np.outer(phi_plus, phi_plus)

# ===========================================================================
section("1. Энтропия Шеннона и типичные последовательности (напоминание)")
p = 0.11
Hp = h(p)
print(f"  h({p}) = {Hp:.4f} бита: источник Бернулли сжимается до {Hp:.4f} бита на символ")
assert abs(Hp - 0.4999) < 1e-3
# вес и размер типичного множества: -log2 p(x)/n отличается от H не более чем на eps (N&C, §12.2)
eps = 0.05
TYP = {}
for n in (100, 1000, 10000):
    ks = np.arange(n + 1)
    lncomb = gammaln(n + 1) - gammaln(ks + 1) - gammaln(n - ks + 1)
    surprise = -(ks * np.log2(p) + (n - ks) * np.log2(1 - p)) / n
    typ = np.abs(surprise - Hp) <= eps
    weight = np.exp(lncomb[typ] + (ks[typ] * np.log(p) + (n - ks[typ]) * np.log(1 - p))).sum()
    logsize = logsumexp(lncomb[typ]) / np.log(2)
    TYP[n] = (weight, logsize / n)
    print(f"  n = {n:5d}: вес eps-типичного множества (eps = {eps}) = {weight:.4f}, log2(размер)/n = {logsize / n:.4f} "
          f"(границы H -+ eps: {Hp - eps:.4f}..{Hp + eps:.4f})")
    assert logsize / n <= Hp + eps + 1e-12
assert TYP[10000][0] > 0.99 and TYP[10000][1] > Hp - eps
# энтропия как ожидание «удивления»: H = E[-log p(x)]
assert abs(H([0.5, 0.25, 0.25]) - 1.5) < 1e-12

# ===========================================================================
section("2. Энтропия фон Неймана: свойства")
for d in (2, 3, 4):
    r = rand_rho(d)
    U = rand_unitary(d)
    assert 0 <= S(r) <= log2(d) + 1e-12
    assert abs(S(U @ r @ U.conj().T) - S(r)) < 1e-10
assert abs(S(np.eye(4) / 4) - 2) < 1e-12 and S(BELL) < 1e-10
# пример: rho = (|0><0| + |+><+|)/2 — собственные значения cos^2(pi/8), sin^2(pi/8)
plus = np.array([1, 1]) / sqrt(2)
rho_0p = 0.5 * np.diag([1, 0]) + 0.5 * np.outer(plus, plus)
S_0p = S(rho_0p)
print(f"  rho = (|0><0| + |+><+|)/2: собств. {np.linalg.eigvalsh(rho_0p).round(4)}, S = {S_0p:.4f} < H(1/2,1/2) = 1")
assert abs(S_0p - h(np.cos(np.pi / 8) ** 2)) < 1e-12
# вогнутость и граница для смесей
worst_conc, worst_mix, n_eq = 1.0, 1.0, 0
for _ in range(500):
    d = 3
    k = 3
    ps = rng.dirichlet(np.ones(k))
    rs = [rand_rho(d, rank=rng.integers(1, d + 1)) for _ in range(k)]
    mix = sum(pi * ri for pi, ri in zip(ps, rs))
    lo = sum(pi * S(ri) for pi, ri in zip(ps, rs))
    hi = H(ps) + lo
    worst_conc = min(worst_conc, S(mix) - lo)
    worst_mix = min(worst_mix, hi - S(mix))
assert worst_conc > -1e-10 and worst_mix > -1e-10
# равенство в верхней границе для ортогональных носителей
rA, rB = np.zeros((4, 4)), np.zeros((4, 4))
rA[:2, :2] = rand_rho(2).real * 0 + np.diag([0.7, 0.3]); rB[2:, 2:] = np.diag([0.5, 0.5])
mix = 0.4 * rA + 0.6 * rB
assert abs(S(mix) - (H([0.4, 0.6]) + 0.4 * S(rA) + 0.6 * S(rB))) < 1e-12
print(f"  500 случайных смесей: min[S(sum p rho) - sum p S(rho)] = {worst_conc:.2e} >= 0 (вогнутость), "
      f"min[H(p) + sum p S - S(mix)] = {worst_mix:.2e} >= 0")
# S(rho) <= H(p) для любого разложения на чистые состояния (разложения — через унитарную свободу, I.3, §1.5)
r = rand_rho(3)
w, V = np.linalg.eigh(r)
worst = 1.0
for _ in range(300):
    m = 5
    U = rand_unitary(m)[:, :3]               # изометрия 3 -> 5: |psi_i> = sum_j U_ij sqrt(w_j)|e_j>
    vecs = (U * np.sqrt(w)) @ V.T             # строки — ненормированные векторы sqrt(p_i)|psi_i>
    ps = (np.abs(vecs) ** 2).sum(1)
    assert abs(ps.sum() - 1) < 1e-10
    recon = sum(np.outer(v, v.conj()) for v in vecs)
    assert np.allclose(recon, r)
    worst = min(worst, H(ps) - S(r))
print(f"  300 разложений rho (d = 3) на 5 чистых состояний: min[H(p) - S(rho)] = {worst:.3e} >= 0; "
      f"равенство — спектральное разложение")
assert worst > -1e-10
# измерение (проективное, без отбора) не уменьшает энтропию; H(diag rho) >= S(rho) (мажоризация)
worst = 1.0
for _ in range(300):
    r = rand_rho(4)
    worst = min(worst, H(np.diag(r).real) - S(r))
assert worst > -1e-10
print(f"  300 случайных rho (d = 4): min[H(diag rho) - S(rho)] = {worst:.2e} >= 0 (дефазировка повышает энтропию)")

# ===========================================================================
section("3. Совместная и условная энтропия, взаимная информация")


def info_quantities(rho, dA, dB):
    rA = ptrace(rho, [dA, dB], [0]); rB = ptrace(rho, [dA, dB], [1])
    SAB, SA, SB = S(rho), S(rA), S(rB)
    return dict(SAB=SAB, SA=SA, SB=SB, cond=SAB - SB, I=SA + SB - SAB)


q_bell = info_quantities(BELL, 2, 2)
cl = 0.5 * (np.outer(ket(0, 0), ket(0, 0)) + np.outer(ket(1, 1), ket(1, 1)))
q_cl = info_quantities(cl, 2, 2)
q_prod = info_quantities(np.kron(np.eye(2) / 2, np.eye(2) / 2), 2, 2)
print("  состояние                      S(AB)   S(A)   S(B)   S(A|B)   I(A:B)")
for name, q in (("Белла Phi+", q_bell), ("классич. (|00><00|+|11><11|)/2", q_cl), ("I/4", q_prod)):
    print(f"  {name:30s} {q['SAB']:5.2f}  {q['SA']:5.2f}  {q['SB']:5.2f}  {q['cond']:6.2f}   {q['I']:5.2f}")
assert abs(q_bell["cond"] + 1) < 1e-10 and abs(q_bell["I"] - 2) < 1e-10
assert abs(q_cl["cond"]) < 1e-10 and abs(q_cl["I"] - 1) < 1e-10
# для сепарабельных состояний S(A|B) >= 0
worst = 10
for _ in range(400):
    k = rng.integers(1, 6)
    ps = rng.dirichlet(np.ones(k))
    rho = sum(pi * np.kron(rand_rho(2, 1 + rng.integers(2)), rand_rho(3, 1 + rng.integers(3))) for pi in ps)
    worst = min(worst, info_quantities(rho, 2, 3)["cond"], S(rho) - S(ptrace(rho, [2, 3], [0])))
print(f"  400 случайных сепарабельных состояний 2x3: min S(A|B), S(B|A) = {worst:.3e} >= 0")
assert worst > -1e-10


# изотропные состояния rho_p = p Phi+ + (1-p) I/4
def iso(pp):
    return pp * BELL + (1 - pp) * np.eye(4) / 4


def cond_iso(pp):
    return H([(1 + 3 * pp) / 4] + [(1 - pp) / 4] * 3) - 1


p_star = brentq(cond_iso, 0.5, 0.99)
for pp in (0.5, p_star, 0.9):
    assert abs(info_quantities(iso(pp), 2, 2)["cond"] - cond_iso(pp)) < 1e-10
print(f"  rho_p: S(A|B) = H((1+3p)/4, (1-p)/4 x3) - 1 < 0 при p > {p_star:.4f}; "
      f"запутанность — при p > 1/3 (I.7): отрицательность S(A|B) — лишь достаточное условие")
print(f"  хэш-граница E_D >= -S(A|B): при p = 0.9 не меньше {-cond_iso(0.9):.4f} эбита на пару")
assert 0.74 < p_star < 0.75
# условная энтропия классических переменных: H(X|Y) = sum_y p(y) H(X|Y=y) >= 0
pxy = np.array([[0.3, 0.1], [0.05, 0.55]])
HXgY = H(pxy) - H(pxy.sum(0))
direct = sum(pxy[:, y].sum() * H(pxy[:, y] / pxy[:, y].sum()) for y in range(2))
assert abs(HXgY - direct) < 1e-12 and HXgY > 0

# ===========================================================================
section("4. Относительная энтропия: Клейн, Пинскер, монотонность")
worst_klein, worst_pinsker = 1.0, 1.0
for _ in range(500):
    d = rng.integers(2, 5)
    a, b = rand_rho(d), rand_rho(d)
    s = rel_ent(a, b)
    worst_klein = min(worst_klein, s)
    worst_pinsker = min(worst_pinsker, s - 2 / log(2) * trdist(a, b) ** 2)
assert worst_klein > 0 and worst_pinsker > -1e-12
assert abs(rel_ent(rho_0p, rho_0p)) < 1e-10
print(f"  500 пар: min S(rho||sigma) = {worst_klein:.3e} > 0 (Клейн); min[S - (2/ln2) D^2] = {worst_pinsker:.3e} >= 0 (Пинскер)")
# бесконечность при несовпадении носителей; несимметричность
s01 = rel_ent(np.diag([0.5, 0.5]), np.diag([1.0, 0.0]))
s10 = rel_ent(np.diag([1.0, 0.0]), np.diag([0.5, 0.5]))
print(f"  S(I/2 || |0><0|) = {s01}, S(|0><0| || I/2) = {s10:.4f}: несимметрична, может быть бесконечной")
assert s01 == np.inf and abs(s10 - 1) < 1e-12
# формула S(rho||I/d) = log d - S(rho)
r = rand_rho(3)
assert abs(rel_ent(r, np.eye(3) / 3) - (log2(3) - S(r))) < 1e-10
# I(A:B) = S(rho_AB || rho_A x rho_B), S(A|B) = -S(rho_AB || I x rho_B)
r = rand_rho(6)
rA, rB = ptrace(r, [2, 3], [0]), ptrace(r, [2, 3], [1])
q = info_quantities(r, 2, 3)
assert abs(rel_ent(r, np.kron(rA, rB)) - q["I"]) < 1e-10
assert abs(-rel_ent(r, np.kron(np.eye(2), rB)) - q["cond"]) < 1e-10
# монотонность при действии каналов (Линдблад–Ульман)
worst = 1.0
for _ in range(300):
    din, dout = 3, int(rng.integers(2, 4))
    Ks = rand_channel_kraus(din, dout, int(rng.integers(1, 5)))
    assert np.allclose(sum(K.conj().T @ K for K in Ks), np.eye(din))
    a, b = rand_rho(din), rand_rho(din)
    worst = min(worst, rel_ent(a, b) - rel_ent(apply(Ks, a), apply(Ks, b)))
print(f"  300 случайных каналов C^3 -> C^d: min[S(rho||sigma) - S(Phi rho||Phi sigma)] = {worst:.3e} >= 0")
assert worst > -1e-10
# монотонность при частичном следе
worst = 1.0
for _ in range(300):
    a, b = rand_rho(8), rand_rho(8)
    worst = min(worst, rel_ent(a, b) - rel_ent(ptrace(a, [2, 4], [0]), ptrace(b, [2, 4], [0])))
assert worst > -1e-10
# транспонирование (положительное, но не вполне положительное) — не канал; здесь монотонность всё равно
# выполнена для самих rho (T сохраняет спектр), а для id x T — оператор может стать неположительным.
# свободная энергия: S(rho || gamma_beta) = beta (F(rho) - F_eq) / ln 2,  H_S = -(omega/2) Z (II.2)
om, beta = 1.0, 2.0
Hs = -om / 2 * Z
gam = expm(-beta * Hs); gam /= np.trace(gam).real
r = rand_rho(2)
F_r = np.trace(r @ Hs).real - S(r) * log(2) / beta
F_eq = -np.log(np.trace(expm(-beta * Hs)).real) / beta
assert abs(rel_ent(r, gam) - beta * (F_r - F_eq) / log(2)) < 1e-10
print(f"  S(rho||gamma_beta) = beta (F(rho) - F_eq)/ln2: проверено (omega = 1, beta = 2); "
      f"F_eq = {F_eq:.4f}, F(rho) = {F_r:.4f}")

# ===========================================================================
section("5. Субаддитивность, Араки–Либ, сильная субаддитивность (10^4 трёхкубитных состояний)")
NST = 10_000
ranks = rng.integers(1, 9, NST)
rhos = np.empty((NST, 8, 8), complex)
for rk in range(1, 9):
    sel = ranks == rk
    rhos[sel] = rand_rho_batch(int(sel.sum()), 8, rk)
dims = [2, 2, 2]
Sabc = S_batch(rhos)
marg = {}
for keep in ([0], [1], [2], [0, 1], [1, 2], [0, 2]):
    marg[tuple(keep)] = S_batch(ptrace_batch(rhos, dims, keep))
SA, SB, SC = marg[(0,)], marg[(1,)], marg[(2,)]
SAB, SBC, SAC = marg[(0, 1)], marg[(1, 2)], marg[(0, 2)]
checks = {
    "субаддитивность S(A)+S(B)-S(AB)": SA + SB - SAB,
    "Араки–Либ S(AB)-|S(A)-S(B)|": SAB - np.abs(SA - SB),
    "ССА S(AB)+S(BC)-S(ABC)-S(B)": SAB + SBC - Sabc - SB,
    "слабая монотонность S(AB)+S(BC)-S(A)-S(C)": SAB + SBC - SA - SC,
    "S(A|B) - S(A|BC)": (SAB - SB) - (Sabc - SBC),
    "I(A:BC) - I(A:B)": (SA + SBC - Sabc) - (SA + SB - SAB),
}
MIN5 = {}
for name, v in checks.items():
    MIN5[name] = v.min()
    print(f"  {name:45s} min = {v.min():.3e}")
    assert v.min() > -1e-9
# «классическое» неравенство S(AB) >= S(A) нарушается
frac_neg = np.mean(SAB - SA < -1e-9)
print(f"  доля состояний с S(AB) < S(A) (у классических переменных невозможно): {frac_neg:.3f}")
assert frac_neg > 0.05
# равенство в ССА: марковская цепочка (классическая) и rho_A x rho_BC
pa = rng.dirichlet(np.ones(2)); Tba = rng.dirichlet(np.ones(2), size=2); Tcb = rng.dirichlet(np.ones(2), size=2)
pabc = np.einsum("a,ab,bc->abc", pa, Tba, Tcb).ravel()
rho_m = np.diag(pabc)
SSA_gap = (S(ptrace(rho_m, dims, [0, 1])) + S(ptrace(rho_m, dims, [1, 2])) - S(rho_m) - S(ptrace(rho_m, dims, [1])))
assert abs(SSA_gap) < 1e-12
# равенство в Араки–Либе: чистые состояния (S(A) = S(B), S(AB) = 0)
psi = rng.normal(size=8) + 1j * rng.normal(size=8); psi /= np.linalg.norm(psi)
rp = np.outer(psi, psi.conj())
assert abs(S(ptrace(rp, [2, 4], [0])) - S(ptrace(rp, [2, 4], [1]))) < 1e-10
print(f"  равенство в ССА: марковская цепочка A -> B -> C (I(A:C|B) = {SSA_gap:.1e}); "
      f"в Араки–Либе: чистые состояния")
# I(A:B) <= 2 min(log dA, log dB), равенство — максимально запутанное
assert abs(q_bell["I"] - 2) < 1e-10
# GHZ: S(A|BC) = S(ABC)-S(BC) = -1, S(A|B) = 0
ghz = (ket(0, 0, 0) + ket(1, 1, 1)) / sqrt(2); rg = np.outer(ghz, ghz)
cA_B = S(ptrace(rg, dims, [0, 1])) - S(ptrace(rg, dims, [1]))
cA_BC = S(rg) - S(ptrace(rg, dims, [1, 2]))
print(f"  GHZ: S(A|B) = {cA_B:.3f}, S(A|BC) = {cA_BC:.3f} (условие уменьшает энтропию, до отрицательных значений)")
assert abs(cA_B) < 1e-10 and abs(cA_BC + 1) < 1e-10

# ===========================================================================
section("6. Непрерывность: неравенство Фаннеса–Ауденарта")
worst = 10
for _ in range(2000):
    d = int(rng.integers(2, 6))
    a = rand_rho(d, int(rng.integers(1, d + 1)))
    # близкие состояния: смесь с другим
    b = (1 - rng.uniform(0, 0.5)) * a + rng.uniform(0, 0.5) * rand_rho(d)
    b /= np.trace(b).real
    T = trdist(a, b)
    bound = T * log2(d - 1) + h(min(T, 1 - 1e-15)) if d > 2 else h(T)
    worst = min(worst, bound - abs(S(a) - S(b)))
print(f"  2000 пар: min[T log(d-1) + h(T) - |S(rho) - S(sigma)|] = {worst:.3e} >= 0")
assert worst > -1e-10
# достижимость границы: rho = |0><0|, sigma = (1-T)|0><0| + T/(d-1) sum_{k>0}|k><k|
d, T = 4, 0.2
a = np.diag([1.0, 0, 0, 0]); b = np.diag([1 - T] + [T / (d - 1)] * (d - 1))
assert abs(abs(S(a) - S(b)) - (T * log2(d - 1) + h(T))) < 1e-12

# ===========================================================================
section("7. Сжатие Шумахера: rho = (|0><0| + |+><+|)/2, S = 0.6009")
lam = np.cos(np.pi / 8) ** 2
S_src = h(lam)
print(f"  собственные значения {lam:.4f}, {1 - lam:.4f}; S(rho) = {S_src:.4f} кубита на символ")


def best_weight(n, R):
    """Максимальный вес Tr(P rho^{(x)n}) по проекторам ранга floor(2^{nR}) (Ки Фан): сумма наибольших собств. значений.
    Собственные значения rho^{(x)n} — lam^(n-k) (1-lam)^k с кратностью C(n,k), убывают с ростом k."""
    D = int(np.floor(2.0 ** (n * R)))
    w, left = 0.0, D
    for k in range(n + 1):
        m = comb(n, k)
        take = min(m, left)
        w += take * np.exp((n - k) * np.log(lam) + k * np.log(1 - lam))
        left -= take
        if left == 0:
            break
    return w


SCH = {}
Rs = (0.4, 0.5, 0.6, 0.7, 0.8)
print("    n   " + "   ".join(f"R = {R:.1f}" for R in Rs))
for n in (10, 20, 50, 100, 200):
    row = [best_weight(n, R) for R in Rs]
    SCH[n] = row
    print(f"  {n:4d}   " + "    ".join(f"{x:.4f}" for x in row))
assert SCH[200][0] < 0.01 and SCH[200][1] < 0.15 and SCH[200][3] > 0.95 and SCH[200][4] > 0.999
assert SCH[200][1] < SCH[100][1] < SCH[50][1] and SCH[200][3] > SCH[100][3]
# порог: R, при котором вес = 1/2, стремится к S(rho)
R_half = {n: brentq(lambda R: best_weight(n, R) - 0.5, 0.3, 1.0, xtol=1e-6) for n in (50, 200, 1000)}
print("  R при весе 1/2: " + ", ".join(f"n = {n}: {R_half[n]:.4f}" for n in R_half) + f"  ->  S = {S_src:.4f}")
assert abs(R_half[1000] - S_src) < 0.02
# прямая проверка: n = 8, R = 0.75 (подпространство размерности 64 из 256): верность сцепленности
n, R = 8, 0.75
Dsub = int(2 ** (n * R))
w1, V1 = np.linalg.eigh(rho_0p)
order = np.argsort(-w1)
w1, V1 = w1[order], V1[:, order]
# собственный базис rho^{x n}: произведения; выбираем Dsub с наибольшими собств. значениями
idx = sorted(range(2 ** n), key=lambda i: -np.prod([w1[(i >> (n - 1 - j)) & 1] for j in range(n)]))[:Dsub]
Vn = V1
for _ in range(n - 1):
    Vn = np.kron(Vn, V1)
P = Vn[:, idx] @ Vn[:, idx].conj().T
rho_n = rho_0p
for _ in range(n - 1):
    rho_n = np.kron(rho_n, rho_0p)
trP = np.trace(P @ rho_n).real
# схема: проекция на типичное подпространство; при неудаче — фиксированное состояние |0..0>
# верность сцепленности F_e = sum_k |Tr(rho A_k)|^2 для операторов Крауса A_0 = P, A_j = |0..0><j| (j по ортодоп.)
junk = np.zeros(2 ** n); junk[0] = 1
Pc = np.eye(2 ** n) - P
wc, Vc = np.linalg.eigh(Pc)
comp = Vc[:, wc > 0.5]
Fe = abs(np.trace(rho_n @ P)) ** 2 + sum(abs(np.vdot(comp[:, j], rho_n @ junk)) ** 2 for j in range(comp.shape[1]))
print(f"  n = 8, подпространство 2^6 = 64: Tr(P rho^n) = {trP:.4f} (комбинаторно {best_weight(8, 0.75):.4f}), "
      f"F_e = {Fe:.4f} >= (Tr P rho)^2 = {trP ** 2:.4f}")
assert abs(trP - best_weight(8, 0.75)) < 1e-10 and Fe >= trP ** 2 - 1e-12
# средняя верность по ансамблю {|0>, |+>}: <psi|D(E(psi))|psi> >= 2 Tr(P rho) - 1
states = [np.array([1.0, 0.0]), plus]
avgF = 0.0
for bits in range(2 ** n):
    v = np.array([1.0])
    for j in range(n):
        v = np.kron(v, states[(bits >> (n - 1 - j)) & 1])
    Pv = P @ v
    out = np.outer(Pv, Pv) + (1 - np.vdot(Pv, Pv).real) * np.outer(junk, junk)
    avgF += np.vdot(v, out @ v).real / 2 ** n
print(f"  средняя по 2^8 последовательностям верность = {avgF:.4f} >= 2 Tr(P rho) - 1 = {2 * trP - 1:.4f}")
assert avgF >= 2 * trP - 1 - 1e-12

# ===========================================================================
section("8. Ответы к задачам")
# 1: rho = diag(1/2,1/4,1/4)
print(f"  1: S(diag(1/2,1/4,1/4)) = {S(np.diag([0.5, 0.25, 0.25])):.4f}; rho из (|0><0|+|+><+|)/2: S = {S_0p:.4f}")
# 2: Белл и классически коррелированное
print(f"  2: Phi+: S(A|B) = {q_bell['cond']:.1f}, I = {q_bell['I']:.1f}; классич.: S(A|B) = {q_cl['cond']:.1f}, I = {q_cl['I']:.1f}")
# 3: W-состояние
wst = (ket(0, 0, 1) + ket(0, 1, 0) + ket(1, 0, 0)) / sqrt(3); rw = np.outer(wst, wst)
SA_w, SAB_w = S(ptrace(rw, dims, [0])), S(ptrace(rw, dims, [0, 1]))
IAB_w = 2 * SA_w - SAB_w
cA_B_w = SAB_w - SA_w
print(f"  3: W: S(A) = {SA_w:.4f}, S(AB) = {SAB_w:.4f} (= S(C)), I(A:B) = {IAB_w:.4f}, S(A|B) = {cA_B_w:.4f}, "
      f"S(A|BC) = {-SA_w:.4f}")
assert abs(SA_w - h(1 / 3)) < 1e-12 and abs(SAB_w - SA_w) < 1e-12
# 5: S(rho||sigma) для кубитов с векторами Блоха вдоль z: r = 0.6, s = 0.2
a = np.diag([0.8, 0.2]); b = np.diag([0.6, 0.4])
s_ab, s_ba = rel_ent(a, b), rel_ent(b, a)
print(f"  5: S(diag(.8,.2)||diag(.6,.4)) = {s_ab:.4f}, обратно {s_ba:.4f}; Пинскер: (2/ln2) D^2 = {2 / log(2) * 0.2 ** 2:.4f}")
assert abs(s_ab - (0.8 * log2(0.8 / 0.6) + 0.2 * log2(0.2 / 0.4))) < 1e-12
# 6: деполяризация Delta_p сжимает относительную энтропию
pdep = 0.5
dep = lambda r_: (1 - pdep) * r_ + pdep * np.eye(2) / 2
s_out = rel_ent(dep(a), dep(b))
print(f"  6: после Delta_0.5: S = {s_out:.4f} < {s_ab:.4f}")
assert s_out < s_ab
# 7: порог изотропного состояния
print(f"  7: S(A|B) < 0 для rho_p при p > {p_star:.4f}")
# 8: изотропное p = 0.9 — хэш-граница
print(f"  8: -S(A|B) при p = 0.9: {-cond_iso(0.9):.4f}")
# 10: сжатие: n = 100 при R = 0.7 и 0.5
print(f"  10: n = 100: вес при R = 0.5: {SCH[100][1]:.4f}, R = 0.7: {SCH[100][3]:.4f}")

print(f"\nВсе проверки пройдены. Время работы: {time.time() - T_START:.1f} с")
