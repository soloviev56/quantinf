"""
Лекция II.14. Квантовое распределение ключей: протоколы и безопасность.

Проверяется: эпсилон-секретность через расстояние следа и вероятность угадывания,
сложение эпсилон при композиции; эквивалентность запутанного варианта и BB84
(состояния Боба после измерения Алисы на Phi+); атака «перехват–пересылка»
(QBER 1/4, базис Брейдбарта); формула Девитака–Винтер для диагональных в
базисе Белла состояний: min H(Z_A|E) = 1 - h(Q), скорость 1 - 2h(Q), порог
11,00 %; энтропийное соотношение неопределённостей с квантовой памятью на
случайных состояниях; шестисостоянийный протокол (порог 12,62 %);
индивидуальная атака Фукса–Гизина–Гриффитса–Ниу–Переса (порог 14,6 %);
граница 25 % (сепарабельность); полное моделирование BB84: 10^6 посылок,
деполяризующий шум + доля перехватов, оценка QBER с границей Хёфдинга,
упрощённый Cascade (BINARY без обратного хода) с подсчётом раскрытых битов,
усиление приватности матрицей Тёплица, длина ключа против 1 - 2h(Q);
лемма о остаточном хешировании на игрушечном примере; E91/BBM92:
связь CHSH и QBER для изотропных состояний, порог DIQKD.

Соглашения (Лекции/Обозначения.md, §4, §5, §23, §26): log = log2;
h — двоичная энтропия; Q — QBER (доля ошибок в просеянном ключе);
e_b, e_p — доли битовых и фазовых ошибок; Bell-диагональные веса
(Phi+, Phi-, Psi+, Psi-); r — скорость ключа на просеянный бит.

Запуск:  python примеры.py
Требуется: numpy, scipy. Все проверки оформлены как assert.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
from math import log2, sqrt, pi, log, ceil, floor
import numpy as np
from scipy.optimize import brentq
from scipy.signal import fftconvolve

T_START = time.time()
rng = np.random.default_rng(2026)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def H(p):
    p = np.asarray(p, float).ravel()
    p = p[p > 1e-15]
    return float(-(p * np.log2(p)).sum())


def h(x):
    return H([x, 1 - x])


def S(rho):
    return H(np.clip(np.linalg.eigvalsh((rho + rho.conj().T) / 2), 0, None))


def tdist(a, b):
    return 0.5 * np.abs(np.linalg.eigvalsh(a - b)).sum()


def ket(*v):
    v = np.array(v, complex)
    return v / np.linalg.norm(v)


def proj(v):
    return np.outer(v, v.conj())


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
k0, k1 = ket(1, 0), ket(0, 1)
kp, km = ket(1, 1), ket(1, -1)
BASIS = {"Z": (k0, k1), "X": (kp, km)}
PHIP = ket(1, 0, 0, 1)
BELL = [ket(1, 0, 0, 1), ket(1, 0, 0, -1), ket(0, 1, 1, 0), ket(0, 1, -1, 0)]   # Phi+, Phi-, Psi+, Psi-


def ptrans_B(r):
    return r.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)


# ===========================================================================
section("1. Эпсилон-секретность: расстояние следа и вероятность угадывания")
# Ключевой бит k, у Евы — чистое состояние |e_k>, <e_0|e_1> = cos(theta)
for th in (0.0, 0.2, 0.6, pi / 2):
    e = [ket(1, 0), ket(np.cos(th), np.sin(th))]
    rKE = sum(0.5 * np.kron(proj(np.eye(2)[k]), proj(e[k])) for k in range(2))
    rE = sum(0.5 * proj(e[k]) for k in range(2))
    eps = tdist(rKE, np.kron(I2 / 2, rE))
    pg = 0.5 + tdist(0.5 * proj(e[0]), 0.5 * proj(e[1]))   # Хелстром (I.3, §5)
    assert pg <= 0.5 + eps + 1e-12
    assert abs(pg - (0.5 + eps)) < 1e-12          # для одного бита — равенство
    print(f"  theta = {th:.3f}: eps = {eps:.4f}, P_guess(Ева) = {pg:.4f} = 1/2 + eps")
# композиция: два независимых ключа с eps1, eps2 -> расстояние <= eps1 + eps2
def cq(th):
    e = [ket(1, 0), ket(np.cos(th), np.sin(th))]
    rKE = sum(0.5 * np.kron(proj(np.eye(2)[k]), proj(e[k])) for k in range(2))
    rE = sum(0.5 * proj(e[k]) for k in range(2))
    return rKE, np.kron(I2 / 2, rE)
(r1, i1), (r2, i2) = cq(0.05), cq(0.1)
e1, e2 = tdist(r1, i1), tdist(r2, i2)
e12 = tdist(np.kron(r1, r2), np.kron(i1, i2))
assert e12 <= e1 + e2 + 1e-12
print(f"  композиция: eps1 = {e1:.4f}, eps2 = {e2:.4f}, eps(пара) = {e12:.4f} <= {e1 + e2:.4f}")

# ===========================================================================
section("2. BB84 и его запутанный вариант (эквивалентность)")
# Алиса измеряет свою половину Phi+ в базисе b: Боб получает |a>_b с вероятностью 1/2
for b, (u0, u1) in BASIS.items():
    for a, u in enumerate((u0, u1)):
        M = np.kron(proj(u), I2)
        post = M @ PHIP
        pa = np.vdot(post, post).real
        rhoB = np.einsum("ab,ac->bc", post.reshape(2, 2), post.reshape(2, 2).conj()) / pa
        target = proj(u.conj())                  # для вещественных базисов u* = u
        assert abs(pa - 0.5) < 1e-12 and np.allclose(rhoB, target)
print("  измерение Алисы на Phi+ в базисе Z или X готовит у Боба |0>,|1>,|+>,|-> с вероятностями 1/2")
# средняя матрица плотности одинакова для обоих базисов — Ева не знает базис
rZ = 0.5 * (proj(k0) + proj(k1))
rX = 0.5 * (proj(kp) + proj(km))
assert np.allclose(rZ, rX) and np.allclose(rZ, I2 / 2)
print("  rho(Z-ансамбль) = rho(X-ансамбль) = I/2")

# ===========================================================================
section("3. Атака «перехват–пересылка»")
def ir_stats(eve_basis_vecs_list):
    """QBER и вероятность угадывания Евы при перехвате каждого кубита
    (Ева измеряет в базисе из списка — равновероятный выбор)."""
    q = 0.0
    pg = 0.0
    for b, (u0, u1) in BASIS.items():
        for a, u in enumerate((u0, u1)):
            for ev in eve_basis_vecs_list:
                for m, w in enumerate(ev):
                    pm = abs(np.vdot(w, u)) ** 2
                    # Боб измеряет пересланное |w> в базисе Алисы (просеянные биты)
                    perr = abs(np.vdot(BASIS[b][1 - a], w)) ** 2
                    q += 0.25 / len(eve_basis_vecs_list) * pm * perr
                    pg += 0.25 / len(eve_basis_vecs_list) * pm * (1.0 if m == a else 0.0)   # Ева называет исход
    return q, pg

q_ir, pg_ir = ir_stats([BASIS["Z"], BASIS["X"]])
assert abs(q_ir - 0.25) < 1e-12 and abs(pg_ir - 0.75) < 1e-12
print(f"  Ева в случайном базисе Z/X: QBER = {q_ir:.4f}, P_guess = {pg_ir:.4f}; I(A:E) = 1/2 бита (знает бит в половине случаев)")
# Базис Брейдбарта: между Z и X (угол pi/8 на окружности Блоха — pi/4 в плоскости xz)
c, s = np.cos(pi / 8), np.sin(pi / 8)
breid = (ket(c, s), ket(-s, c))
q_b, pg_b = ir_stats([breid])
assert abs(q_b - 0.25) < 1e-12 and abs(pg_b - c ** 2) < 1e-12
print(f"  базис Брейдбарта: QBER = {q_b:.4f}, P_guess = cos^2(pi/8) = {pg_b:.4f}, I(A:E) = 1 - h(cos^2 pi/8) = {1 - h(c ** 2):.4f}")
for f in (0.1, 0.2, 0.5):
    print(f"  доля перехватов f = {f}: QBER = f/4 = {f / 4:.4f}")

# ===========================================================================
section("4. Девитак–Винтер, Шор–Прескилл: r = 1 - 2h(Q)")
def eve_purif(lam):
    """|psi>_{ABE} = sum_i sqrt(lam_i)|b_i>_{AB}|i>_E (Bell-диагональное rho_AB)."""
    return sum(np.sqrt(max(lam[i], 0)) * np.kron(BELL[i], np.eye(4)[i]) for i in range(4)).reshape(2, 2, 4)

def HZ_given_E(psi):
    rE = np.einsum("abe,abf->ef", psi, psi.conj())
    bl = [np.einsum("be,bf->ef", psi[a], psi[a].conj()) for a in range(2)]
    pa = [np.trace(b).real for b in bl]
    return H(pa) + sum(p * S(b / p) for p, b in zip(pa, bl) if p > 1e-15) - S(rE)

def errors(lam):
    return lam[2] + lam[3], lam[1] + lam[3]           # e_b (Psi±), e_p (Phi-, Psi-)

SP = {}
for Q in (0.02, 0.05, 0.08, 0.11):
    ts = np.linspace(0, Q, 401)
    vals = [HZ_given_E(eve_purif([1 - 2 * Q + t, Q - t, Q - t, t])) for t in ts]
    i = int(np.argmin(vals))
    eb, ep = errors([1 - 2 * Q + ts[i], Q - ts[i], Q - ts[i], ts[i]])
    assert abs(eb - Q) < 1e-12 and abs(ep - Q) < 1e-12
    assert abs(vals[i] - (1 - h(Q))) < 1e-6 and abs(ts[i] - Q * Q) < Q / 400 + 1e-12
    SP[Q] = 1 - 2 * h(Q)
    print(f"  Q = {Q:.2f}: min H(Z_A|E) = {vals[i]:.6f} = 1 - h(Q) (при lam_Psi- = Q^2); r = 1 - 2h(Q) = {SP[Q]:.4f}")
Q_SP = brentq(lambda q: 1 - 2 * h(q), 0.01, 0.2)
assert abs(Q_SP - 0.110028) < 1e-5
print(f"  порог Шора–Прескилла: 1 - 2h(Q) = 0 при Q = {Q_SP:.6f}")

# явная формула: H(Z_A|E) = 1 - H(lam) + h(lam_Phi+ + lam_Phi-) на случайных lam
for _ in range(200):
    lam = rng.dirichlet(np.ones(4))
    assert abs(HZ_given_E(eve_purif(lam)) - (1 - H(lam) + h(lam[0] + lam[1]))) < 1e-9
print("  H(Z_A|E) = 1 - H(lam) + h(lam_Phi+ + lam_Phi-) — проверено на 200 случайных Bell-диагональных состояниях")

# энтропийное соотношение неопределённостей с квантовой памятью (Берта и др.):
# H(Z_A|E) + H(X_A|B) >= 1, а H(X_A|B) <= H(X_A|X_B) <= h(e_x) — на случайных состояниях
def rand_pure(d):
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    return v / np.linalg.norm(v)

def cond_meas(psi3, basis, dims, keep):
    """H(M_A | keep) для чистого psi на A x B x E; A измеряется в basis."""
    dA, dB, dE = dims
    t = psi3.reshape(dA, dB, dE)
    tot = 0.0
    blocks = []
    for u in basis:
        v = np.tensordot(u.conj(), t, axes=([0], [0]))     # (dB, dE)
        r = np.einsum("be,cf->becf", v, v.conj())          # B E B E
        r = r.reshape(dB * dE, dB * dE)
        if keep == "E":
            r = np.trace(r.reshape(dB, dE, dB, dE), axis1=0, axis2=2)
        else:
            r = np.trace(r.reshape(dB, dE, dB, dE), axis1=1, axis2=3)
        blocks.append(r)
    pa = [np.trace(b).real for b in blocks]
    rM = sum(blocks)
    return H(pa) + sum(p * S(b / p) for p, b in zip(pa, blocks) if p > 1e-15) - S(rM)

worst = 9.0
for _ in range(2000):
    psi = rand_pure(2 * 2 * 4)
    hz_e = cond_meas(psi, BASIS["Z"], (2, 2, 4), "E")
    hx_b = cond_meas(psi, BASIS["X"], (2, 2, 4), "B")
    # e_x: вероятность расхождения X_A и X_B
    t = psi.reshape(2, 2, 4)
    ex = sum(abs(np.einsum("a,b,abe->e", BASIS["X"][i].conj(), BASIS["X"][1 - i].conj(), t)) ** 2 for i in range(2)).sum()
    assert hz_e + hx_b >= 1 - 1e-9
    # H(X_A|B) <= h(e_x) (через H(X_A|X_B))
    assert hx_b <= h(ex) + 1e-9
    assert hz_e >= 1 - h(ex) - 1e-9
    worst = min(worst, hz_e + hx_b - 1)
print(f"  2000 случайных чистых состояний ABE (2x2x4): H(Z_A|E) + H(X_A|B) >= 1 (мин. запас {worst:.2e}),"
      f" H(Z_A|E) >= 1 - h(e_x) — всегда")

# шестисостоянийный протокол: известны все три доли ошибок -> lam = (1-3Q/2, Q/2, Q/2, Q/2)
r6 = lambda Q: HZ_given_E(eve_purif([1 - 1.5 * Q, Q / 2, Q / 2, Q / 2])) - h(Q)
Q6 = brentq(r6, 0.05, 0.2)
Q6b = brentq(lambda Q: 1 - H([1 - 1.5 * Q, Q / 2, Q / 2, Q / 2]), 0.05, 0.2)
assert abs(Q6 - Q6b) < 1e-9 and abs(Q6 - 0.1262) < 1e-4
print(f"  шесть состояний: r = H(Z_A|E) - h(Q) = 1 - H(1-3Q/2, Q/2, Q/2, Q/2); порог Q = {Q6:.4f}")
print(f"    при Q = 0.05: BB84 r = {1 - 2 * h(0.05):.4f}, шесть состояний r = {r6(0.05):.4f}")

# индивидуальная атака (Фукс, Гизин, Гриффитс, Ниу, Перес): P_guess = 1/2 + sqrt(Q(1-Q))
def eve_guess(lam):
    psi = eve_purif(lam)
    r = [np.einsum("be,bf->ef", psi[a], psi[a].conj()) for a in range(2)]
    return 0.5 + 0.5 * np.abs(np.linalg.eigvalsh(r[0] - r[1])).sum()

for Q in (0.05, 0.10, 0.146):
    best = max(eve_guess([1 - 2 * Q + t, Q - t, Q - t, t]) for t in np.linspace(0, Q, 401))
    assert abs(best - (0.5 + sqrt(Q * (1 - Q)))) < 1e-5
    print(f"  Q = {Q:.3f}: max P_guess(Ева, Хелстром на каждом кубите) = {best:.4f} = 1/2 + sqrt(Q(1-Q))")
r_ind = lambda Q: h(0.5 + sqrt(Q * (1 - Q))) - h(Q)    # Чисар–Кёрнер: 1 - h(Q) - (1 - h(P))
Q_ind = brentq(r_ind, 0.05, 0.2)
assert abs(Q_ind - 0.1464) < 1e-4
print(f"  индивидуальные атаки: r = h(1/2 + sqrt(Q(1-Q))) - h(Q) = 0 при Q = {Q_ind:.4f}")

# граница 25 %: при e_b = e_p = 1/4 существует сепарабельное Bell-диагональное состояние
lam25 = [0.5, 0.25, 0.25, 0.0]
rho25 = sum(l * proj(b) for l, b in zip(lam25, BELL))
assert errors(lam25) == (0.25, 0.25)
assert np.linalg.eigvalsh(ptrans_B(rho25)).min() > -1e-12
print(f"  Q = 25 %: Bell-диагональное состояние (1/2, 1/4, 1/4, 0) даёт e_b = e_p = 1/4 и PPT (сепарабельно) -> ключа нет")
# канал «перехват–пересылка» в базисах Z/X — разрушающий запутанность: Choi PPT
choi = np.zeros((4, 4), complex)
for b in ("Z", "X"):
    for u in BASIS[b]:
        # Phi(rho) = sum_b 1/2 sum_u <u|rho|u> |u><u|; Choi = (id x Phi)(Phi+)
        choi += 0.5 * np.kron(proj(u.conj()), proj(u)) / 2
assert abs(np.trace(choi) - 1) < 1e-12 and np.linalg.eigvalsh(ptrans_B(choi)).min() > -1e-12
fid = np.vdot(PHIP, choi @ PHIP).real
assert abs(fid - 0.5) < 1e-12
print(f"  матрица Чоя канала перехвата–пересылки PPT (разрушает запутанность); <Phi+|J|Phi+> = {fid:.3f}")

# ===========================================================================
section("5. Моделирование BB84: от посылок до ключа")
N = 10 ** 6
p_dep, f_ir = 0.04, 0.2
a = rng.integers(0, 2, N)
alpha = rng.integers(0, 2, N)                       # 0 = Z, 1 = X
beta = rng.integers(0, 2, N)
# перехват: Ева в случайном базисе
intercept = rng.random(N) < f_ir
gam = rng.integers(0, 2, N)
eve_bit = np.where(gam == alpha, a, rng.integers(0, 2, N))
st_bit, st_basis = np.where(intercept, eve_bit, a), np.where(intercept, gam, alpha)
# деполяризация: с вероятностью p состояние заменяется на I/2
dep = rng.random(N) < p_dep
bob = np.where(beta == st_basis, st_bit, rng.integers(0, 2, N))
bob = np.where(dep, rng.integers(0, 2, N), bob)
sift = alpha == beta
nS = int(sift.sum())
A, B = a[sift], bob[sift]
Q_true = f_ir / 4 + p_dep / 2 - 2 * (f_ir / 4) * (p_dep / 2)
Q_emp = float(np.mean(A != B))
assert abs(nS / N - 0.5) < 0.003 and abs(Q_emp - Q_true) < 0.002
print(f"  N = {N}: просеяно {nS} ({nS / N:.4f}); QBER = {Q_emp:.4f}, ожидается f/4 + p/2 - 2(f/4)(p/2) = {Q_true:.4f}")
# оценка QBER по выборке
eps_pe = 1e-10
k = nS // 10
idx = rng.permutation(nS)
samp, keyidx = idx[:k], np.sort(idx[k:])
Q_hat = float(np.mean(A[samp] != B[samp]))
delta = sqrt(log(1 / eps_pe) / (2 * k))
print(f"  выборка k = {k}: Q_hat = {Q_hat:.4f}, delta(Хёфдинг, eps = 1e-10) = {delta:.4f}")
# проверка покрытия Хёфдинга при eps = 0.01 на 4000 повторах
kk, ee = 2000, 0.01
dd = sqrt(log(1 / ee) / (2 * kk))
miss = np.mean(Q_true - rng.binomial(kk, Q_true, 4000) / kk > dd)   # односторонняя оценка
assert miss <= ee
print(f"  покрытие Хёфдинга (k = {kk}, eps = {ee}): доля Q > Q_hat + delta = {miss:.4f} <= {ee}")
KA, KB = A[keyidx].copy(), B[keyidx].copy()
n = len(KA)


def binary_search(xa, xb, ids):
    """Бинарный поиск одной ошибки в блоке с нечётной разностью чётностей; возвращает (позиция, раскрыто бит)."""
    leak = 0
    lo, hi = 0, len(ids)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        leak += 1
        if xa[ids[lo:mid]].sum() % 2 != xb[ids[lo:mid]].sum() % 2:
            hi = mid
        else:
            lo = mid
    return ids[lo], leak


def cascade_lite(xa, xb, Qest, passes=10):
    """Упрощённый Cascade: проходы BINARY по блокам со случайной перестановкой, размер блока удваивается.
    Раскрытые биты: чётность каждого блока + шаги бинарного поиска."""
    xb = xb.copy()
    leak = 0
    kb = max(4, int(0.73 / Qest))
    m = len(xa)
    for ps in range(passes):
        perm = rng.permutation(m)
        nb = int(ceil(m / kb))
        pa = np.add.reduceat(xa[perm], np.arange(0, m, kb)) % 2
        pb = np.add.reduceat(xb[perm], np.arange(0, m, kb)) % 2
        leak += nb
        for j in np.nonzero(pa != pb)[0]:
            ids = perm[j * kb:(j + 1) * kb]
            pos, l = binary_search(xa, xb, ids)
            leak += l
            xb[pos] ^= 1
        kb = min(2 * kb, m)
    return xb, leak


KB_corr, leak = cascade_lite(KA, KB, Q_hat)
resid = int((KA != KB_corr).sum())
fEC = leak / (n * h(Q_hat))
assert resid == 0
assert 1.0 < fEC < 2.0
print(f"  коррекция ошибок (упрощённый Cascade, 10 проходов): n = {n}, раскрыто {leak} бит, "
      f"f_EC = leak/(n h(Q)) = {fEC:.3f}; остаточных ошибок: {resid}")
eps_pa, eps_cor = 1e-10, 1e-10
ell = int(floor(n * (1 - h(Q_hat + delta)) - leak - 2 * log2(1 / (2 * eps_pa)) - log2(2 / eps_cor)))
ell_asym = n * (1 - 2 * h(Q_true))
assert 0 < ell < ell_asym
print(f"  длина ключа: l = n(1 - h(Q_hat + delta)) - leak - 2log(1/2eps) - log(2/eps_cor) = {ell}"
      f"  (асимптотически n(1 - 2h(Q)) = {ell_asym:.0f}; отношение {ell / ell_asym:.3f})")
print(f"    из них: n(h(Q_hat + delta) - h(Q)) = {n * (h(Q_hat + delta) - h(Q_hat)):.0f} (оценка параметров), "
      f"leak - n h(Q) = {leak - n * h(Q_hat):.0f} (неидеальная коррекция), "
      f"{2 * log2(1 / (2 * eps_pa)) + log2(2 / eps_cor):.0f} (eps)")
# усиление приватности: матрица Тёплица l x n, задаётся n + l - 1 случайными битами
t = rng.integers(0, 2, n + ell - 1)


def toeplitz_hash(x, t, ell):
    nn = len(x)
    cv = fftconvolve(t.astype(float), x.astype(float))[nn - 1:nn - 1 + ell]
    r = np.rint(cv)
    assert np.max(np.abs(cv - r)) < 0.1
    return r.astype(np.int64) % 2


# прямая проверка Тёплица на малом примере
xs_, ts_ = rng.integers(0, 2, 20), rng.integers(0, 2, 20 + 7 - 1)
Tm = np.array([[ts_[i - j + 19] for j in range(20)] for i in range(7)])
assert np.array_equal(toeplitz_hash(xs_, ts_, 7), Tm @ xs_ % 2)
KfA, KfB = toeplitz_hash(KA, t, ell), toeplitz_hash(KB_corr, t, ell)
assert np.array_equal(KfA, KfB)
bias = abs(KfA.mean() - 0.5)
assert bias < 5 / sqrt(ell)
print(f"  хеш Тёплица: итоговые ключи совпадают, длина {ell}, доля единиц {KfA.mean():.4f}")
# сколько знала Ева о просеянном ключе (перехваченные биты, где она угадала)
eve_known = np.mean(np.where(gam == alpha, 1.0, 0.5)[sift][intercept[sift]])
assert abs(eve_known - 0.75) < 0.01
print(f"  Ева угадывает перехваченный просеянный бит с вероятностью {eve_known:.4f} (теория 3/4)")
# скорость на посылку и длины ключа для разных размеров блока
print("  конечный размер (при том же Q и f_EC; доля выборки 10 %, eps = 1e-10):")
FIN = {}
for nS_ in (10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8):
    k_ = nS_ // 10
    n_ = nS_ - k_
    d_ = sqrt(log(1 / eps_pe) / (2 * k_))
    l_ = n_ * (1 - h(Q_true + d_)) - fEC * n_ * h(Q_true) - 2 * log2(1 / (2 * eps_pa)) - log2(2 / eps_cor)
    FIN[nS_] = max(0.0, l_) / nS_
    print(f"    n_sift = {nS_:.0e}: delta = {d_:.4f}, l/n_sift = {FIN[nS_]:.4f}")
lim_f = 0.9 * (1 - (1 + fEC) * h(Q_true))
print(f"    предел при n -> бесконечности с той же f_EC и выборкой 10 %: 0.9(1 - (1 + f_EC)h(Q)) = {lim_f:.4f};"
      f" идеал 1 - 2h(Q) = {1 - 2 * h(Q_true):.4f}")
assert FIN[10 ** 4] == 0.0 and abs(FIN[10 ** 8] - lim_f) < 0.01

# ===========================================================================
section("6. Лемма об остаточном хешировании (игрушечный пример)")
# X равномерно на {0,1}^n; Ева знает k линейных чётностей M x; H_min(X|E) = n - k.
def gf2_rank(Mx):
    Mx = Mx.copy() % 2
    r = 0
    rows, cols = Mx.shape
    for c in range(cols):
        piv = [i for i in range(r, rows) if Mx[i, c]]
        if not piv:
            continue
        Mx[[r, piv[0]]] = Mx[[piv[0], r]]
        for i in range(rows):
            if i != r and Mx[i, c]:
                Mx[i] ^= Mx[r]
        r += 1
        if r == rows:
            break
    return r

nL, kL = 16, 6
M = rng.integers(0, 2, (kL, nL))
while gf2_rank(M) < kL:
    M = rng.integers(0, 2, (kL, nL))
Hmin = nL - kL
print(f"  n = {nL}, Ева знает {kL} чётностей: H_min(X|E) = {Hmin}")
LHL = {}
for ellL in (2, 4, 6, 8, 10, 12):
    ds = []
    for _ in range(400):
        tt = rng.integers(0, 2, nL + ellL - 1)
        T = np.array([[tt[i - j + nL - 1] for j in range(nL)] for i in range(ellL)])
        r = gf2_rank(np.vstack([M, T])) - kL                 # размерность образа при фиксированном знании Евы
        ds.append(1 - 2.0 ** (r - ellL))                     # расстояние от равномерного на 2^l
    dav = float(np.mean(ds))
    bound = 0.5 * sqrt(2.0 ** (ellL - Hmin))
    LHL[ellL] = (dav, bound)
    assert dav <= bound + 1e-12
    print(f"    l = {ellL:2d}: средняя (по хешам) удалённость от идеального ключа = {dav:.4f}; оценка 1/2*2^((l-Hmin)/2) = {bound:.4f}")
assert LHL[12][0] >= 1 - 2.0 ** (Hmin - 12) - 1e-12   # l > H_min: ключ заведомо неидеален

# ===========================================================================
section("7. E91 / BBM92: CHSH и QBER для изотропного состояния")
def A_obs(al):
    return np.cos(al) * Z + np.sin(al) * X
def chsh(rho):
    a0, a1 = A_obs(0), A_obs(pi / 2)
    b0, b1 = A_obs(pi / 4), A_obs(-pi / 4)
    Bop = np.kron(a0, b0 + b1) + np.kron(a1, b0 - b1)
    return np.trace(rho @ Bop).real
for p in (1.0, 0.99, 0.95, 0.9, 0.78, 1 / sqrt(2)):
    rho = p * proj(PHIP) + (1 - p) * np.eye(4) / 4
    Sv = chsh(rho)
    QZ = sum(np.vdot(np.kron(BASIS["Z"][i], BASIS["Z"][1 - i]), rho @ np.kron(BASIS["Z"][i], BASIS["Z"][1 - i])).real for i in range(2))
    QX = sum(np.vdot(np.kron(BASIS["X"][i], BASIS["X"][1 - i]), rho @ np.kron(BASIS["X"][i], BASIS["X"][1 - i])).real for i in range(2))
    assert abs(Sv - 2 * sqrt(2) * p) < 1e-12 and abs(QZ - (1 - p) / 2) < 1e-12 and abs(QX - QZ) < 1e-12
    print(f"  p = {p:.4f}: S = 2sqrt2 p = {Sv:.4f}, Q = (1-p)/2 = {QZ:.4f}, r(BBM92) = {max(0, 1 - 2 * h(QZ)):.4f}")
Q_S2 = (1 - 1 / sqrt(2)) / 2
assert abs(Q_S2 - Q_ind) < 1e-9 and abs(0.5 + sqrt(Q_S2 * (1 - Q_S2)) - (1 - Q_S2)) < 1e-12
print(f"  S = 2 <=> Q = (1 - 1/sqrt2)/2 = {Q_S2:.6f} — совпадает с порогом индивидуальных атак {Q_ind:.6f}"
      f" (там P_guess = 1 - Q, т. е. h(P) = h(Q))")
p_key = 1 - 2 * Q_SP
S_key = 2 * sqrt(2) * p_key
assert abs(S_key - 2.2059) < 1e-3
print(f"  ключ BBM92 (доверенные устройства) исчезает при Q = 11,00 %: p = {p_key:.4f}, S = {S_key:.4f}")
# DIQKD (I.6, (9.2)): r >= 1 - h(Q) - h((1 + sqrt((S/2)^2 - 1))/2)
def r_di(p):
    Sv, Q = 2 * sqrt(2) * p, (1 - p) / 2
    if Sv <= 2:
        return -1.0
    return 1 - h(Q) - h((1 + sqrt((Sv / 2) ** 2 - 1)) / 2)
p_di = brentq(r_di, 0.75, 0.99)
assert abs(p_di - 0.857) < 1e-3
print(f"  DIQKD (I.6, (9.2)): порог p = {p_di:.4f}, Q = {(1 - p_di) / 2:.4f}, S = {2 * sqrt(2) * p_di:.4f}")
DIR = {}
for p in (0.99, 0.95, 0.9):
    DIR[p] = r_di(p)
    print(f"    p = {p}: r(BBM92) = {1 - 2 * h((1 - p) / 2):.4f}, r(DI) >= {DIR[p]:.4f}")
assert DIR[0.9] > 0

# ===========================================================================
section("8. Ответы к задачам")
print(f"  1: просеивание 1/2; QBER перехвата-пересылки 1/4; при f = 0.2: {0.2 / 4}")
print(f"  2: Брейдбарт: P_guess = {c ** 2:.4f}, I(A:E) = {1 - h(c ** 2):.4f}, QBER = 1/4")
print(f"  3: r(Q=0.02) = {SP[0.02]:.4f}, r(Q=0.05) = {SP[0.05]:.4f}, r(Q=0.08) = {SP[0.08]:.4f}; порог {Q_SP:.4f}")
print(f"  4: шесть состояний: порог {Q6:.4f}; r(0.05) = {r6(0.05):.4f} против BB84 {1 - 2 * h(0.05):.4f}")
print(f"  5: индивидуальные атаки: порог {Q_ind:.4f}")
print(f"  6: delta(k = 1e4, eps = 1e-10) = {sqrt(log(1e10) / 2e4):.4f}")
print(f"  7: Q = 3 %, f_EC = 1.2: r = 1 - h - 1.2 h = {1 - 2.2 * h(0.03):.4f}; идеал {1 - 2 * h(0.03):.4f}")
print(f"  8: S при Q = 11 %: {S_key:.4f}; DI-порог: Q = {(1 - p_di) / 2:.4f}")

print(f"\nВсе проверки пройдены. Время работы: {time.time() - T_START:.1f} с")
