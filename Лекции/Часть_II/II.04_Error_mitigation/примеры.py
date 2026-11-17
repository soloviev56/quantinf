"""
Лекция II.4. Error mitigation: смещение, дисперсия и выборочный оверхед;
подавление ошибок считывания (тензорная модель против полной матрицы);
ZNE (глобальный и локальный фолдинг, растяжение шума, экстраполяции
линейная, Ричардсона, экспоненциальная; когерентные ошибки); PEC
(квазивероятности для деполяризующего шума, gamma и оверхед gamma^2,
несмещённость, смещение при неверной модели, разреженная модель
Паули–Линдблада); CDR (обучение на почти-клиффордовых схемах); проверка
симметрий и виртуальная дистилляция; экспоненциальный рост оверхеда.

Задача-бенчмарк (как в ЛР12): троттеризованная динамика TFIM на n = 4 кубитах,
H = -J sum Z_i Z_{i+1} - h sum X_i, |0000>, наблюдаемая <Z_2 Z_3>,
d = 1..10 шагов Троттера до t = 1. Шум: деполяризация после гейтов,
амплитудное затухание, ошибки считывания.

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
Порядок множителей: первый — старший (np.kron); кубиты в коде нумеруются с 0.
"""
import time
import itertools
import numpy as np
from math import pi, sqrt, exp, log
from functools import reduce
from scipy.linalg import expm
from scipy.optimize import curve_fit

T_START = time.time()
rng = np.random.default_rng(2026)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [I2, X, Y, Z]
CNOT2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def dag(A):
    return A.conj().T


def kron(*ops):
    return reduce(np.kron, ops)


def Rz(t):
    return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])


def Rx(t):
    return expm(-1j * t / 2 * X)


N_Q = 4
DIM = 2 ** N_Q
J_, h_, T_TOT = 1.0, 1.0, 1.0


def embed1(U, q, n=N_Q):
    return kron(*[U if i == q else I2 for i in range(n)])


def embed2(U, q, n=N_Q):
    return kron(*([I2] * q + [U] + [I2] * (n - q - 2)))


# ---------------------------------------------------------------------------
# схема как список гейтов: ("1q", U, q, angle_kind, angle) или ("cx", q)
def trotter_circuit(d, n=N_Q):
    dt = T_TOT / d
    gates = []
    for _ in range(d):
        for q in range(n - 1):  # e^{+i J dt Z Z} = CNOT (I x Rz(-2 J dt)) CNOT
            gates.append(("cx", q))
            gates.append(("rz", q + 1, -2 * J_ * dt))
            gates.append(("cx", q))
        for q in range(n):     # e^{+i h dt X} = Rx(-2 h dt)
            gates.append(("rx", q, -2 * h_ * dt))
    return gates


_GATE_CACHE = {}


def gate_unitary(g):
    key = (g[0], g[1], round(g[2], 12) if len(g) > 2 else None)
    U = _GATE_CACHE.get(key)
    if U is None:
        if g[0] == "cx":
            U = embed2(CNOT2, g[1])
        elif g[0] == "rz":
            U = embed1(Rz(g[2]), g[1])
        else:
            U = embed1(Rx(g[2]), g[1])
        _GATE_CACHE[key] = U
    return U


def inverse_gate(g):
    if g[0] == "cx":
        return g
    return (g[0], g[1], -g[2])


# шумовые операторы
def dep1_kraus(p):
    return [sqrt(1 - 3 * p / 4) * I2, sqrt(p / 4) * X, sqrt(p / 4) * Y, sqrt(p / 4) * Z]


def ad_kraus(g):
    return [np.array([[1, 0], [0, sqrt(1 - g)]], complex), np.array([[0, sqrt(g)], [0, 0]], complex)]


def apply_1q_kraus(rho, K, q, n=N_Q):
    """Однокубитный канал на кубите q: тензорная свёртка с суперматрицей sum K (x) K^* (4x4)."""
    S4 = sum(np.kron(k, k.conj()) for k in K).reshape(2, 2, 2, 2)
    T = rho.reshape([2] * (2 * n))
    T = np.tensordot(S4, T, axes=([2, 3], [q, n + q]))
    T = np.moveaxis(T, [0, 1], [q, n + q])
    return T.reshape(2 ** n, 2 ** n)


class Noise:
    def __init__(self, p1=0.0, p2=0.0, gad=0.0, coh=0.0, scale=1.0):
        self.p1, self.p2, self.gad, self.coh, self.scale = p1, p2, gad, coh, scale

    def scaled(self, s):
        return Noise(self.p1, self.p2, self.gad, self.coh, self.scale * s)


def run(gates, noise=None, rho0=None):
    rho = np.zeros((DIM, DIM), complex); rho[0, 0] = 1
    if rho0 is not None:
        rho = rho0.copy()
    for g in gates:
        U = gate_unitary(g)
        if noise is not None and noise.coh and g[0] == "rx":
            # систематический пере-поворот: реализуется Rx(theta (1 + coh))
            U = embed1(Rx(g[2] * (1 + noise.coh)), g[1])
        rho = U @ rho @ dag(U)
        if noise is None:
            continue
        s = noise.scale
        if g[0] == "cx":
            for q in (g[1], g[1] + 1):
                if noise.p2:
                    rho = apply_1q_kraus(rho, dep1_kraus(min(1, s * noise.p2)), q)
                if noise.gad:
                    rho = apply_1q_kraus(rho, ad_kraus(min(1, s * noise.gad)), q)
        else:
            if noise.p1:
                rho = apply_1q_kraus(rho, dep1_kraus(min(1, s * noise.p1)), g[1])
            if noise.gad:
                rho = apply_1q_kraus(rho, ad_kraus(min(1, s * noise.gad)), g[1])
    return rho


OBS = embed1(Z, 1) @ embed1(Z, 2)  # Z_2 Z_3 (кубиты 1, 2 в нумерации с 0)
diagO = np.real(np.diag(OBS))


def expval(rho, readout=None):
    p = np.real(np.diag(rho))
    if readout is not None:
        p = readout @ p
    return float(p @ diagO)


NOISE = Noise(p1=0.002, p2=0.01, gad=0.003)
E0 = 0.02  # ошибка считывания 0 -> 1
E1 = 0.05  # ошибка считывания 1 -> 0
A1 = np.array([[1 - E0, E1], [E0, 1 - E1]])  # p_изм = A p_ист (столбец — истинный бит)
A_ro = kron(*[A1] * N_Q).real

# ---------------------------------------------------------------------------
section("0. Задача-бенчмарк: TFIM n = 4, <Z2 Z3>, d шагов Троттера до t = 1")
H_tfim = sum(-J_ * embed1(Z, q) @ embed1(Z, q + 1) for q in range(N_Q - 1)) + sum(-h_ * embed1(X, q) for q in range(N_Q))
psi0 = np.zeros(DIM, complex); psi0[0] = 1
psi_ex = expm(-1j * H_tfim * T_TOT) @ psi0
E_exact_dyn = np.real(psi_ex.conj() @ OBS @ psi_ex)
IDEAL, NOISY, NOISY_RO, NG = {}, {}, {}, {}
print("   d   гейтов  CNOT   <O> идеал. схема   <O> шум.   <O> шум.+считыв.   точная динамика")
for d in range(1, 11):
    gs = trotter_circuit(d)
    IDEAL[d] = expval(run(gs))
    rn = run(gs, NOISE)
    NOISY[d] = expval(rn)
    NOISY_RO[d] = expval(rn, A_ro)
    NG[d] = (len(gs), sum(g[0] == "cx" for g in gs))
    print(f"  {d:2d}   {NG[d][0]:4d}   {NG[d][1]:3d}      {IDEAL[d]:+.5f}        {NOISY[d]:+.5f}     {NOISY_RO[d]:+.5f}          {E_exact_dyn:+.5f}")
# проверка схемы одного шага
gs1 = trotter_circuit(1)
U1 = reduce(lambda A, B: B @ A, [gate_unitary(g) for g in gs1], np.eye(DIM))
Hzz = sum(-J_ * embed1(Z, q) @ embed1(Z, q + 1) for q in range(N_Q - 1))
Hx = sum(-h_ * embed1(X, q) for q in range(N_Q))
assert np.allclose(U1, expm(-1j * Hx * T_TOT) @ expm(-1j * Hzz * T_TOT))
print("  один шаг схемы = e^{-i H_x dt} e^{-i H_zz dt} (проверено); цель mitigation — <O> идеальной схемы")

# ---------------------------------------------------------------------------
section("1. Ошибки считывания: тензорная модель, полная матрица, дисперсия")
d_ref = 4
rho_ref = run(trotter_circuit(d_ref), NOISE)
p_true = np.real(np.diag(rho_ref))
p_meas = A_ro @ p_true
p_mit = np.linalg.solve(A_ro, p_meas)
assert np.allclose(p_mit, p_true)
A1_inv = np.linalg.inv(A1)
assert np.allclose(kron(*[A1_inv] * N_Q), np.linalg.inv(A_ro))
print(f"  d = {d_ref}: <O> без считывания {p_true @ diagO:+.5f}, со считыванием {p_meas @ diagO:+.5f}, после обращения (A1^-1)^(x4) {p_mit @ diagO:+.5f}")
# один кубит: <Z>_mit = (<Z>_raw - (E1 - E0)) / (1 - E0 - E1)
zt = 0.3
pz = np.array([(1 + zt) / 2, (1 - zt) / 2])
z_raw = (A1 @ pz) @ np.array([1, -1])
z_mit = (z_raw - (E1 - E0)) / (1 - E0 - E1)
assert np.isclose(z_mit, zt)
print(f"  один кубит: <Z>_изм = (1 - e0 - e1)<Z> + (e1 - e0); при <Z> = {zt}: {z_raw:.4f}; обращение даёт {z_mit:.4f}")
print(f"  дисперсия растёт в 1/(1 - e0 - e1)^2 = {1 / (1 - E0 - E1) ** 2:.4f} раза на кубит наблюдаемой; для Z2Z3: {1 / (1 - E0 - E1) ** 4:.4f}")
# 6 кубитов: калибровка полной матрицы (2^6 схем) против тензорной (2 схемы на кубит) при конечной статистике
n6 = 6
shots_cal = 10000
e0s = rng.uniform(0.01, 0.03, n6); e1s = rng.uniform(0.03, 0.06, n6)
A_list = [np.array([[1 - a, b], [a, 1 - b]]) for a, b in zip(e0s, e1s)]
A6 = kron(*A_list).real
# коррелированная добавка: если кубит 0 в |1>, кубит 1 чаще ошибается (перекрёстные помехи)
def A6_corr(c):
    M = A6.copy()
    for col in range(64):
        if (col >> 5) & 1:  # кубит 0 = 1
            v = np.zeros(64); v[col] = 1
            # дополнительный переворот кубита 1 с вероятностью c
            v2 = v.copy(); v2[col] = 0; v2[col ^ (1 << 4)] = 1
            M[:, col] = (1 - c) * A6[:, col] + c * A6[:, col ^ (1 << 4)]
    return M


p6 = 0.1 * np.ones(64) / 64  # 0.9 |100000> + 0.1 I/64
p6[0b100000] += 0.9
obs6 = np.array([(1 - 2 * ((i >> 3) & 1)) * (1 - 2 * ((i >> 2) & 1)) for i in range(64)])  # Z3 Z4
obs6b = np.array([(1 - 2 * ((i >> 5) & 1)) * (1 - 2 * ((i >> 4) & 1)) for i in range(64)])  # Z1 Z2
true6, true6b = p6 @ obs6, p6 @ obs6b
RO6 = {}
for corr in (0.0, 0.05):
    Atrue = A6_corr(corr) if corr else A6
    err_full, err_tens, err_raw = [], [], []
    errb_full, errb_tens = [], []
    for rep in range(10):
        # полная калибровка: 64 схемы
        Afull = np.zeros((64, 64))
        for col in range(64):
            Afull[:, col] = rng.multinomial(shots_cal, Atrue[:, col]) / shots_cal
        # тензорная: 2 схемы (все |0>, все |1>), маргиналы по кубитам
        c0 = rng.multinomial(shots_cal, Atrue[:, 0]) / shots_cal
        c1 = rng.multinomial(shots_cal, Atrue[:, 63]) / shots_cal
        Ats = []
        for q in range(n6):
            bit = np.array([(i >> (n6 - 1 - q)) & 1 for i in range(64)])
            a = c0[bit == 1].sum(); b = c1[bit == 0].sum()
            Ats.append(np.array([[1 - a, b], [a, 1 - b]]))
        Aten = kron(*Ats)
        pm = rng.multinomial(100000, Atrue @ p6) / 100000
        for arr_f, arr_t, ob, tv in ((err_full, err_tens, obs6, true6), (errb_full, errb_tens, obs6b, true6b)):
            arr_f.append(np.linalg.solve(Afull, pm) @ ob - tv)
            arr_t.append(np.linalg.solve(Aten, pm) @ ob - tv)
        err_raw.append(pm @ obs6 - true6)
    RO6[corr] = (np.sqrt(np.mean(np.square(err_raw))), np.sqrt(np.mean(np.square(err_full))), np.sqrt(np.mean(np.square(err_tens))),
                 np.sqrt(np.mean(np.square(errb_full))), np.sqrt(np.mean(np.square(errb_tens))))
    print(f"  6 кубитов (<Z3Z4> = {true6:.2f}, <Z1Z2> = {true6b:.2f}), корреляция {corr:.2f}: ср.кв. ошибка <Z3Z4>: без обращения {RO6[corr][0]:.4f}, полная матрица (64 схемы) {RO6[corr][1]:.4f}, тензорная (2 схемы) {RO6[corr][2]:.4f}")
    print(f"                           <Z1Z2>: полная {RO6[corr][3]:.4f}, тензорная {RO6[corr][4]:.4f}")
assert RO6[0.0][2] < RO6[0.0][0] / 3 and RO6[0.05][4] > 2 * RO6[0.05][3]
print(f"  число калибровочных схем: полная 2^n = {2 ** n6} (n = 20: {2 ** 20}), тензорная — 2")

# ---------------------------------------------------------------------------
section("2. ZNE: фолдинг, экстраполяция, смещение и дисперсия")


def fold_global(gates, k):
    """U (U^+ U)^k: коэффициент растяжения 2k + 1."""
    inv = [inverse_gate(g) for g in reversed(gates)]
    return gates + (inv + gates) * k


def fold_local(gates, k):
    out = []
    for g in gates:
        out += [g] + [inverse_gate(g), g] * k
    return out


def richardson_coeffs(lams):
    lams = np.asarray(lams, float)
    return np.array([np.prod([lj / (lj - li) for j, lj in enumerate(lams) if j != i]) for i, li in enumerate(lams)])


g135 = richardson_coeffs([1, 3, 5])
assert np.allclose(g135, [15 / 8, -10 / 8, 3 / 8])
g123 = richardson_coeffs([1, 2, 3])
assert np.allclose(g123, [3, -3, 1])
print(f"  коэффициенты Ричардсона: (1, 3, 5) -> {g135}, sum gamma^2 = {np.sum(g135 ** 2):.4f}; (1, 2, 3) -> {g123}, sum gamma^2 = {np.sum(g123 ** 2):.0f}")


def extrap(lams, vals, kind):
    lams = np.asarray(lams, float); vals = np.asarray(vals, float)
    if kind == "lin":
        return np.polyfit(lams, vals, 1)[1]
    if kind == "rich":
        return richardson_coeffs(lams) @ vals
    if kind == "exp":  # vals = a e^{-c lam} (наблюдаемая без смещения), лог-линейная подгонка
        if np.any(vals * np.sign(vals[0]) <= 0):
            return np.nan
        s = np.sign(vals[0])
        k, b = np.polyfit(lams, np.log(np.abs(vals)), 1)
        return s * np.exp(b)


def zne_values(d, folding, scales, noise=NOISE, readout=None):
    gs = trotter_circuit(d)
    out = []
    for lam in scales:
        if folding == "global":
            g2 = fold_global(gs, (lam - 1) // 2)
            out.append(expval(run(g2, noise), readout))
        elif folding == "local":
            g2 = fold_local(gs, (lam - 1) // 2)
            out.append(expval(run(g2, noise), readout))
        elif folding == "stretch":  # идеализированное растяжение импульсов: шум x lam
            out.append(expval(run(gs, noise.scaled(lam)), readout))
    return out


SH = 10000  # запусков на точку
ZNE = {}
print(f"  смещение |E_mit - E_ideal| (глобальный фолдинг, масштабы 1, 3, 5; без ошибок считывания) и ст. откл. при {SH} запусков на точку:")
print("   d    E_ideal    E_noisy    лин.(1,3,5)        Ричардсон          эксп.            растяжение+Ричардсон")
for d in (1, 2, 4, 6, 8, 10):
    vals = zne_values(d, "global", [1, 3, 5])
    vs = zne_values(d, "stretch", [1, 2, 3])
    sig = [sqrt(max(1 - v ** 2, 0) / SH) for v in vals]
    res = {}
    for kind in ("lin", "rich", "exp"):
        est = extrap([1, 3, 5], vals, kind)
        boots = []
        for _ in range(400):
            vb = np.array(vals) + rng.normal(size=3) * sig
            boots.append(extrap([1, 3, 5], vb, kind))
        res[kind] = (est, np.nanstd(boots))
    est_s = extrap([1, 2, 3], vs, "rich")
    ZNE[d] = (vals, res, est_s)
    print(f"  {d:2d}   {IDEAL[d]:+.4f}    {vals[0]:+.4f}   {abs(res['lin'][0] - IDEAL[d]):.4f}+-{res['lin'][1]:.4f}    {abs(res['rich'][0] - IDEAL[d]):.4f}+-{res['rich'][1]:.4f}    {abs(res['exp'][0] - IDEAL[d]):.4f}+-{res['exp'][1]:.4f}    {abs(est_s - IDEAL[d]):.4f}")
for d in (4, 6, 8, 10):
    vals, res, _ = ZNE[d]
    assert abs(res["rich"][0] - IDEAL[d]) < 0.6 * abs(vals[0] - IDEAL[d]) and abs(ZNE[d][2] - IDEAL[d]) < abs(res["rich"][0] - IDEAL[d])
    assert res["rich"][1] > 1.5 * sqrt((1 - vals[0] ** 2) / SH)
# глобальный против локального фолдинга при одинаковом коэффициенте
d = 6
vg = zne_values(d, "global", [1, 3, 5]); vl = zne_values(d, "local", [1, 3, 5])
print(f"  d = {d}: глобальный фолдинг E(1, 3, 5) = {np.round(vg, 4)}, локальный = {np.round(vl, 4)}; Ричардсон: {extrap([1, 3, 5], vg, 'rich'):+.4f} / {extrap([1, 3, 5], vl, 'rich'):+.4f} (идеал {IDEAL[d]:+.4f})")
FOLD = (vg, vl)
# дисперсия: усиление ст. откл. Ричардсона ~ sqrt(sum gamma_i^2 (1 - E_i^2)/(1 - E_1^2))
# когерентная ошибка: пере-поворот Rx на 5 % — фолдинг его не усиливает
coh_noise = Noise(coh=0.05)
d = 4
vc = zne_values(d, "global", [1, 3, 5], coh_noise)
print(f"  когерентный пере-поворот Rx(theta(1 + 0.05)), d = {d}: E(1, 3, 5) = {np.round(vc, 5)} — фолдинг G G^+ G = G не меняет ошибку")
print(f"    экстраполяция {extrap([1, 3, 5], vc, 'rich'):+.5f}, идеал {IDEAL[d]:+.5f}: смещение {abs(extrap([1, 3, 5], vc, 'rich') - IDEAL[d]):.4f} остаётся")
assert np.allclose(vc, vc[0]) and abs(vc[0] - IDEAL[d]) > 0.005
# если пере-поворот не обращается точно (G^+ реализован как Rx(-theta) без ошибки): фолдинг его усиливает частично
COH = (vc, IDEAL[4])
# ZNE со считыванием: смещение считывания не масштабируется фолдингом
v_ro = zne_values(6, "global", [1, 3, 5], readout=A_ro)
print(f"  d = 6 с ошибками считывания: Ричардсон {extrap([1, 3, 5], v_ro, 'rich'):+.4f} (идеал {IDEAL[6]:+.4f}); "
      f"считывание надо подавлять отдельно: с обращением {extrap([1, 3, 5], [v * 1 for v in zne_values(6, 'global', [1, 3, 5])], 'rich'):+.4f}")
ZNE_RO = extrap([1, 3, 5], v_ro, 'rich')

# ---------------------------------------------------------------------------
section("3. PEC: квазивероятности, gamma, несмещённость")


def dep_inverse_quasi(p):
    lam = 1 - p
    qI = (1 + 3 / lam) / 4
    qP = (1 - 1 / lam) / 4
    return np.array([qI, qP, qP, qP])


for p in (0.01, 0.002):
    q = dep_inverse_quasi(p)
    # проверка: sum q_P P Delta_p(P rho P) P = rho
    rho_t = np.array([[0.6, 0.3 - 0.2j], [0.3 + 0.2j, 0.4]])
    dp = lambda r: sum(k @ r @ dag(k) for k in dep1_kraus(p))
    rec = sum(qq * P @ dp(r) @ P for qq, P in zip(q, PAULIS) for r in [rho_t])
    assert np.allclose(rec, rho_t)
    gam = np.abs(q).sum()
    assert np.isclose(gam, (3 / (1 - p) - 1) / 2)
    print(f"  Delta_p^-1, p = {p}: q = ({q[0]:.6f}, {q[1]:.6f} x3), gamma = (3/(1-p) - 1)/2 = {gam:.6f} ~ 1 + 3p/2 = {1 + 1.5 * p:.6f}")
PEC_NOISE = Noise(p1=0.002, p2=0.01)  # модель PEC — только деполяризация


def noise_locations(gates):
    locs = []
    for i, g in enumerate(gates):
        if g[0] == "cx":
            locs += [(i, g[1], PEC_NOISE.p2), (i, g[1] + 1, PEC_NOISE.p2)]
        else:
            locs.append((i, g[1], PEC_NOISE.p1))
    return locs


def gamma_total(gates, scale=1.0):
    return np.prod([np.abs(dep_inverse_quasi(p * scale)).sum() for (_, _, p) in noise_locations(gates)])


def run_with_insertions(gates, noise, inserts):
    """inserts: словарь (индекс гейта, кубит) -> паули (0..3), применяется после шума гейта."""
    rho = np.zeros((DIM, DIM), complex); rho[0, 0] = 1
    for i, g in enumerate(gates):
        U = gate_unitary(g)
        rho = U @ rho @ dag(U)
        qs = (g[1], g[1] + 1) if g[0] == "cx" else (g[1],)
        for q in qs:
            pp = noise.p2 if g[0] == "cx" else noise.p1
            if pp:
                rho = apply_1q_kraus(rho, dep1_kraus(pp), q)
            if noise.gad:
                rho = apply_1q_kraus(rho, ad_kraus(noise.gad), q)
            k = inserts.get((i, q), 0)
            if k:
                Pf = embed1(PAULIS[k], q)
                rho = Pf @ rho @ Pf
    return rho


def pec_exact(gates, noise, p_model_scale=1.0):
    """Точное значение PEC-оценки: обратные отображения с квазивероятностями (без выборки)."""
    rho = np.zeros((DIM, DIM), complex); rho[0, 0] = 1
    for i, g in enumerate(gates):
        U = gate_unitary(g)
        rho = U @ rho @ dag(U)
        qs = (g[1], g[1] + 1) if g[0] == "cx" else (g[1],)
        for q in qs:
            pp = noise.p2 if g[0] == "cx" else noise.p1
            if pp:
                rho = apply_1q_kraus(rho, dep1_kraus(pp), q)
            if noise.gad:
                rho = apply_1q_kraus(rho, ad_kraus(noise.gad), q)
            pm = (PEC_NOISE.p2 if g[0] == "cx" else PEC_NOISE.p1) * p_model_scale
            qv = dep_inverse_quasi(pm)
            rho = sum(qv[k] * embed1(PAULIS[k], q) @ rho @ embed1(PAULIS[k], q) for k in range(4))
    return expval(rho)


print("   d   gamma_tot   gamma^2 (оверхед)   E_noisy    PEC точно (модель верна)   модель занижена на 30 %   + неучтённое ампл. затухание")
PEC = {}
for d in (1, 2, 4, 6, 8, 10):
    gs = trotter_circuit(d)
    gt = gamma_total(gs)
    en = expval(run(gs, PEC_NOISE))
    e_ok = pec_exact(gs, PEC_NOISE)
    e_low = pec_exact(gs, PEC_NOISE, 0.7)
    e_ad = pec_exact(gs, Noise(p1=0.002, p2=0.01, gad=0.003))
    PEC[d] = (gt, en, e_ok, e_low, e_ad)
    print(f"  {d:2d}   {gt:8.4f}   {gt ** 2:10.4f}          {en:+.5f}    {e_ok:+.6f} (ид. {IDEAL[d]:+.6f})       {e_low:+.5f}               {e_ad:+.5f}")
    assert abs(e_ok - IDEAL[d]) < 1e-10
assert abs(PEC[10][3] - IDEAL[10]) > 0.01
# gamma ~ exp(2 * sum eps), eps = 3p/4
gs10 = trotter_circuit(10)
eps_sum = sum(0.75 * p for (_, _, p) in noise_locations(gs10))
print(f"  d = 10: gamma = {gamma_total(gs10):.4f}, exp(2 sum eps_i) = {exp(2 * eps_sum):.4f} (eps_i = 3p_i/4 — вероятность ошибки)")
# выборочная PEC (Монте-Карло): несмещённость и дисперсия
d = 4
gs = trotter_circuit(d)
locs = noise_locations(gs)
gt = gamma_total(gs)
Nmc = 3000
ests = []
for _ in range(Nmc):
    ins = {}
    sign = 1.0
    for (i, q, p) in locs:
        qv = dep_inverse_quasi(p)
        pr = np.abs(qv) / np.abs(qv).sum()
        k = rng.choice(4, p=pr)
        sign *= np.sign(qv[k])
        if k:
            ins[(i, q)] = k
    val = expval(run_with_insertions(gs, PEC_NOISE, ins))
    shot = 1 if rng.random() < (1 + val) / 2 else -1  # один запуск на схему
    ests.append(gt * sign * shot)
ests = np.array(ests)
m, s = ests.mean(), ests.std() / sqrt(Nmc)
print(f"  Монте-Карло PEC, d = {d}: {Nmc} схем по 1 запуску: оценка {m:+.4f} +- {s:.4f} (идеал {IDEAL[d]:+.4f}); ст. откл. одного запуска {ests.std():.3f} ~ gamma = {gt:.3f}")
assert abs(m - IDEAL[d]) < 4 * s and abs(ests.std() - gt) < 0.1 * gt
PEC_MC = (m, s, ests.std(), gt)
# разреженная модель Паули–Линдблада: e^{lam (P.P - id)} = w id + (1-w) P.P, w = (1 + e^{-2 lam})/2; обратное — gamma = e^{2 lam}
lam_ = 0.004
w = (1 + exp(-2 * lam_)) / 2
winv = np.array([(1 + exp(2 * lam_)) / 2, (1 - exp(2 * lam_)) / 2])
rho_t = np.array([[0.6, 0.3 - 0.2j], [0.3 + 0.2j, 0.4]])
chan = lambda r: w * r + (1 - w) * X @ r @ X
# генератор: L(r) = lam (X r X - r); e^{L}
Lsup = lam_ * (np.kron(X, X.conj()) - np.eye(4))
assert np.allclose((expm(Lsup) @ rho_t.reshape(-1)).reshape(2, 2), chan(rho_t))
rec = winv[0] * chan(rho_t) + winv[1] * X @ chan(rho_t) @ X
assert np.allclose(rec, rho_t) and np.isclose(np.abs(winv).sum(), exp(2 * lam_))
# деполяризующий канал как модель Паули–Линдблада с lam_X = lam_Y = lam_Z = lam: 1 - p = e^{-4 lam}
p_ = 0.01
lamd = -log(1 - p_) / 4
L3 = sum(lamd * (np.kron(P, P.conj()) - np.eye(4)) for P in (X, Y, Z))
assert np.allclose((expm(L3) @ rho_t.reshape(-1)).reshape(2, 2), sum(k @ rho_t @ dag(k) for k in dep1_kraus(p_)))
print(f"  Паули–Линдблад: e^(lam(P.P - id)) обращается с gamma = e^(2 lam); Delta_p при p = {p_}: lam = -ln(1-p)/4 = {lamd:.6f},")
print(f"    gamma (через генераторы) = e^(6 lam) = {exp(6 * lamd):.6f}, оптимальное (3/(1-p) - 1)/2 = {(3 / (1 - p_) - 1) / 2:.6f}")
SPL = (lamd, exp(6 * lamd), (3 / (1 - p_) - 1) / 2)

# ---------------------------------------------------------------------------
section("4. CDR: обучение на почти-клиффордовых схемах")


def random_clifford_angle(t):
    """Один из двух ближайших кратных pi/2 углов, вероятность пропорциональна близости."""
    lo = np.floor(t / (pi / 2)) * (pi / 2)
    w = (t - lo) / (pi / 2)
    return lo + (pi / 2 if rng.random() < w else 0.0)


def cdr_training(gates, n_train, frac_keep):
    """В каждой обучающей схеме случайная доля (1 - frac_keep) неклиффордовых поворотов
    заменяется клиффордовыми (случайное округление к соседнему кратному pi/2)."""
    idx_nc = [i for i, g in enumerate(gates) if g[0] in ("rz", "rx")]
    keep = int(round(frac_keep * len(idx_nc)))
    train = []
    for _ in range(n_train):
        kept = set(rng.choice(idx_nc, size=keep, replace=False)) if keep else set()
        g2 = [g if (i in kept or g[0] == "cx") else (g[0], g[1], random_clifford_angle(g[2])) for i, g in enumerate(gates)]
        train.append(g2)
    return train


def cdr_estimate(d, n_train=20, frac_keep=0.1, noise=NOISE, readout=None):
    gs = trotter_circuit(d)
    xs, ys = [], []
    for g2 in cdr_training(gs, n_train, frac_keep):
        xs.append(expval(run(g2, noise), readout))
        ys.append(expval(run(g2)))
    a, b = np.polyfit(xs, ys, 1)
    return a * expval(run(gs, noise), readout) + b, (a, b), np.array(xs), np.array(ys)


print("   d   E_ideal    E_noisy    CDR (20 схем, 10 % неклифф.)   |смещение|   наклон a")
CDR = {}
for d in (1, 2, 4, 6, 8, 10):
    est, (a, b), xs, ys = cdr_estimate(d)
    CDR[d] = (est, a, b)
    print(f"  {d:2d}   {IDEAL[d]:+.4f}    {NOISY[d]:+.4f}    {est:+.4f}                       {abs(est - IDEAL[d]):.4f}       {a:.3f}")
assert all(abs(CDR[d][0] - IDEAL[d]) < abs(NOISY[d] - IDEAL[d]) for d in (4, 6, 8, 10))
print("  зависимость от числа обучающих схем и доли неклиффордовых гейтов (d = 6, среднее |смещение| по 5 повторам):")
CDR2 = {}
for n_tr in (5, 20):
    for fk in (0.0, 0.1, 0.3):
        errs = [abs(cdr_estimate(6, n_tr, fk)[0] - IDEAL[6]) for _ in range(5)]
        CDR2[(n_tr, fk)] = np.mean(errs)
        print(f"     схем {n_tr:2d}, неклиффордовых {fk:.1f}: {np.mean(errs):.4f}")
# с когерентной ошибкой — линейная модель «шумное -> идеальное» хуже
estc, _, _, _ = cdr_estimate(4, 20, 0.1, Noise(p1=0.002, p2=0.01, gad=0.003, coh=0.05))
noisy_c = expval(run(trotter_circuit(4), Noise(p1=0.002, p2=0.01, gad=0.003, coh=0.05)))
print(f"  d = 4 с добавленным пере-поворотом 5 %: шумное {noisy_c:+.4f}, CDR {estc:+.4f}, идеал {IDEAL[4]:+.4f}")
CDR_COH = (noisy_c, estc)

# ---------------------------------------------------------------------------
section("5. Проверка симметрий и виртуальная дистилляция")
# GHZ_4 с шумом; симметрия Z^{(x)4} = +1; наблюдаемая Z1 Z2
ghz_gates = [("h", 0)]
H1 = np.array([[1, 1], [1, -1]], complex) / sqrt(2)
rho = np.zeros((DIM, DIM), complex); rho[0, 0] = 1
Uh = embed1(H1, 0)
rho = Uh @ rho @ dag(Uh)
rho = apply_1q_kraus(rho, dep1_kraus(0.01), 0)
for q in range(N_Q - 1):
    U = embed2(CNOT2, q)
    rho = U @ rho @ dag(U)
    for qq in (q, q + 1):
        rho = apply_1q_kraus(rho, dep1_kraus(0.03), qq)
        rho = apply_1q_kraus(rho, ad_kraus(0.01), qq)
pz = np.real(np.diag(rho))
par = np.array([(-1) ** bin(i).count("1") for i in range(DIM)])
z12 = np.array([(1 - 2 * ((i >> 3) & 1)) * (1 - 2 * ((i >> 2) & 1)) for i in range(DIM)])
acc = pz[par == 1].sum()
raw = pz @ z12
post = (pz * (par == 1)) @ z12 / acc
print(f"  GHZ_4 с шумом: <Z1 Z2> = {raw:.4f} (идеал 1); после отбора по чётности Z^(x)4 = +1: {post:.4f}; доля принятых {acc:.4f}, оверхед 1/{acc:.4f} = {1 / acc:.3f}")
assert post > raw
SYM = (raw, post, acc)
# виртуальная дистилляция на TFIM: Tr(rho^M O)/Tr(rho^M)
d = 6
rn = run(trotter_circuit(d), NOISE)
ev, V = np.linalg.eigh(rn)
dom = V[:, -1]
e_dom = np.real(dom.conj() @ OBS @ dom)
vd = {}
for M in (1, 2, 3):
    rM = np.linalg.matrix_power(rn, M)
    vd[M] = np.real(np.trace(rM @ OBS) / np.trace(rM))
print(f"  VD, d = {d}: M = 1: {vd[1]:+.4f}, M = 2: {vd[2]:+.4f}, M = 3: {vd[3]:+.4f}; доминирующий собств. вектор: {e_dom:+.4f}; идеал {IDEAL[d]:+.4f}")
print(f"    чистота Tr rho^2 = {np.real(np.trace(rn @ rn)):.4f}, наибольшее собств. значение {ev[-1]:.4f}, перекрытие с идеальным = {abs(dom.conj() @ (lambda g: reduce(lambda A, B: B @ A, [gate_unitary(x) for x in g], np.eye(DIM)))(trotter_circuit(d))[:, 0]) ** 2:.4f}")
assert abs(vd[2] - IDEAL[d]) < abs(vd[1] - IDEAL[d])
VD = (vd, e_dom, np.real(np.trace(rn @ rn)), ev[-1])

# ---------------------------------------------------------------------------
section("6. Пределы: экспоненциальный рост оверхеда")
print("   d    CNOT   сигнал E_noisy/E_ideal    PEC gamma^2    запусков для |ошибки| 0.01 (PEC)   ZNE-Ричардсон: усиление дисперсии")
LIM = {}
for d in (1, 2, 4, 6, 8, 10):
    gs = trotter_circuit(d)
    g2 = gamma_total(gs) ** 2
    vals = ZNE[d][0] if d in ZNE else zne_values(d, "global", [1, 3, 5])
    amp = np.sum(g135 ** 2 * (1 - np.array(vals) ** 2)) / (1 - vals[0] ** 2)
    LIM[d] = (NOISY[d] / IDEAL[d], g2, g2 / 0.01 ** 2, amp)
    print(f"  {d:2d}    {NG[d][1]:3d}        {NOISY[d] / IDEAL[d]:.4f}               {g2:8.3f}        {g2 / 0.01 ** 2:12.0f}                       {amp:.2f}")
# глубокая схема: оверхед PEC при 1000 CNOT с p2 = 0.01
for ncx, p in ((100, 0.01), (1000, 0.01), (1000, 0.001), (5000, 0.001)):
    gq = ((3 / (1 - p) - 1) / 2) ** (2 * ncx)
    print(f"  {ncx} CNOT с деполяризацией p = {p} на каждом из двух кубитов: gamma^2 = {gq ** 2:.3e}")
print(f"  оценка e^(4 * eps * N_loc): N_loc = 2000, eps = 0.0075 -> {exp(4 * 0.0075 * 2000):.3e}")

# ---------------------------------------------------------------------------
section("7. Ответы к задачам")
print(f"  1: Ричардсон (1, 3, 5): (15/8, -5/4, 3/8); sum gamma^2 = {np.sum(g135 ** 2):.4f} -> ст. откл. x {sqrt(np.sum(g135 ** 2)):.3f}")
print(f"  3: e0 = 0.02, e1 = 0.05: <Z>_mit = (<Z>_изм - 0.03)/0.93; дисперсия x {1 / 0.93 ** 2:.4f}")
print(f"  4: p = 0.01: gamma = {(3 / 0.99 - 1) / 2:.6f}; 100 гейтов (по одному месту шума): gamma_tot = {((3 / 0.99 - 1) / 2) ** 100:.4f}, оверхед {((3 / 0.99 - 1) / 2) ** 200:.3f}")
f5, E5 = 0.8, 0.6
vals5 = [E5 * f5 ** l for l in (1, 3, 5)]
r5 = g135 @ vals5
print(f"  5: E(lam) = E0 f^lam, f = {f5}, E0 = {E5}: Ричардсон даёт {r5:.4f} (смещение {r5 - E5:+.4f}), экспоненциальная — точно {extrap([1, 3, 5], vals5, 'exp'):.4f}")
assert abs(extrap([1, 3, 5], vals5, "exp") - E5) < 1e-12
print(f"  7: Паули–Линдблад: gamma = e^(2 lam); для Delta_0.01: {SPL[1]:.6f} против {SPL[2]:.6f}")
# задача 8: клиффордовы слои + Delta_p на всех кубитах; сигнал = (1-p)^{сумма весов} (гейзенбергова картина)
H1c = np.array([[1, 1], [1, -1]], complex) / sqrt(2)
Sg = np.diag([1, 1j])
layers = []
for l in range(5):
    U = np.eye(DIM, dtype=complex)
    for q in range(N_Q):
        U = embed1([H1c, Sg, I2][rng.integers(3)], q) @ U
    for q in range(l % 2, N_Q - 1, 2):
        U = embed2(CNOT2, q) @ U
    layers.append(U)
p8 = 0.02
Cfull = reduce(lambda A, B: B @ A, layers, np.eye(DIM))
O8 = Cfull @ embed1(Z, 0) @ dag(Cfull)  # идеальное значение на |0000> равно 1
rho = np.zeros((DIM, DIM), complex); rho[0, 0] = 1
for U in layers:
    rho = U @ rho @ dag(U)
    for q in range(N_Q):
        rho = apply_1q_kraus(rho, dep1_kraus(p8), q)
sig8 = np.real(np.trace(rho @ O8))


def pauli_weight(P):
    w = 0
    for q in range(N_Q):
        # вес на кубите q: коммутирует ли P со всеми X_q и Z_q
        if not (np.allclose(embed1(X, q) @ P, P @ embed1(X, q)) and np.allclose(embed1(Z, q) @ P, P @ embed1(Z, q))):
            w += 1
    return w


wsum = 0
Ph = O8.copy()
for U in reversed(layers):
    wsum += pauli_weight(Ph)  # шум после слоя действует на текущую (гейзенбергову) наблюдаемую
    Ph = dag(U) @ Ph @ U
assert np.isclose(sig8, (1 - p8) ** wsum)
print(f"  8: 5 случайных клиффордовых слоёв, Delta_{p8} на всех кубитах: сигнал {sig8:.6f} = (1-p)^(сумма весов = {wsum}) = {(1 - p8) ** wsum:.6f}")
print(f"  9: GHZ_4: <Z1Z2> {SYM[0]:.4f} -> {SYM[1]:.4f}, принимается {SYM[2]:.4f}")
eps10 = 0.1
sig_ = np.diag([0, 1]).astype(complex)
psi_ = np.diag([1, 0]).astype(complex)
O10 = np.array([[1, 0.5], [0.5, -1]], complex)
rho10 = (1 - eps10) * psi_ + eps10 * sig_
vd10 = np.real(np.trace(rho10 @ rho10 @ O10) / np.trace(rho10 @ rho10))
print(f"  10: rho = (1-eps)|0><0| + eps|1><1|, eps = {eps10}, O = Z + X/2: <O> = {np.real(np.trace(rho10 @ O10)):.4f} (ошибка {2 * eps10:.2f}), VD: {vd10:.4f} (ошибка {1 - vd10:.4f} = 2 eps^2/((1-eps)^2 + eps^2) = {2 * eps10 ** 2 / ((1 - eps10) ** 2 + eps10 ** 2):.4f})")
assert np.isclose(1 - vd10, 2 * eps10 ** 2 / ((1 - eps10) ** 2 + eps10 ** 2))

print(f"\nВсе проверки пройдены. Время работы: {time.time() - T_START:.1f} с")
