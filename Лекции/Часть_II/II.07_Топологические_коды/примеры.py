"""
Лекция II.7. Топологические коды и декодирование: торический код Китаева
(стабилизаторы, k = 2, логические операторы — нетривиальные циклы, дистанция L,
возбуждения на концах струн); планарный и повёрнутый поверхностные коды (число
кубитов, k = 1, дистанция d); модели шума code capacity и феноменологическая,
детекторы; декодеры MWPM (паросочетание минимального веса) и union-find; точное
сравнение с ML-декодером на торе 3 x 3; пороги и подавление ошибки Lambda;
ресурсная оценка.

Соглашения (Лекции/Обозначения.md, §18–§19). Торический код: кубиты на рёбрах
решётки L x L на торе; h(r, c) — горизонтальное ребро из вершины (r, c) вправо,
v(r, c) — вертикальное ребро из вершины (r, c) вниз; звезда A_(r,c) = X на четырёх
рёбрах вершины, плакет B_(r,c) = Z на h(r,c), h(r+1,c), v(r,c), v(r,c+1).

Запуск:  python примеры.py
Требуется: numpy, scipy, networkx (паросочетание — networkx.max_weight_matching).
Все проверки оформлены как assert.
"""
import time
import itertools
import numpy as np
import networkx as nx
from math import comb, log
from scipy.optimize import brentq

T_START = time.time()
rng = np.random.default_rng(2026)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


# ---------------------------------------------------------------------------
# GF(2)
def gf2_rank(M):
    M = (np.array(M, dtype=np.uint8) % 2).copy()
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = np.nonzero(M[r:, c])[0]
        if len(piv) == 0:
            continue
        p = r + piv[0]
        M[[r, p]] = M[[p, r]]
        nz = np.nonzero(M[:, c])[0]
        for i in nz:
            if i != r:
                M[i] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def gf2_nullspace(M):
    """Базис ядра M над GF(2) (строки результата)."""
    M = (np.array(M, dtype=np.uint8) % 2).copy()
    rows, cols = M.shape
    pivots = []
    r = 0
    for c in range(cols):
        piv = np.nonzero(M[r:, c])[0] if r < rows else []
        if len(piv) == 0:
            continue
        p = r + piv[0]
        M[[r, p]] = M[[p, r]]
        for i in np.nonzero(M[:, c])[0]:
            if i != r:
                M[i] ^= M[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    free = [c for c in range(cols) if c not in pivots]
    basis = []
    for f in free:
        v = np.zeros(cols, dtype=np.uint8)
        v[f] = 1
        for i, pc in enumerate(pivots):
            if M[i, f]:
                v[pc] = 1
        basis.append(v)
    return np.array(basis)


def in_rowspace(M, v):
    return gf2_rank(np.vstack([M, v])) == gf2_rank(M)


def css_logicals(HX, HZ):
    """Логические Z (в ker HX, не в row HZ) и X (в ker HZ, не в row HX) — по одному на логический кубит."""
    k = HX.shape[1] - gf2_rank(HX) - gf2_rank(HZ)
    Zc = [v for v in gf2_nullspace(HX) if not in_rowspace(HZ, v)]
    Xc = [v for v in gf2_nullspace(HZ) if not in_rowspace(HX, v)]
    return k, Zc, Xc


def css_distance(Hchk, logicals):
    """Наименьший вес вектора e с Hchk e = 0 и нечётным перекрытием с каким-либо из logicals.
    Граф: вершины — проверки + одна «граница», рёбра — кубиты; состояние (вершина, чётности).
    Подходит, когда каждый кубит входит не более чем в две проверки (топологические коды)."""
    m, n = Hchk.shape
    B = m  # граничная вершина
    ends = []
    for q in range(n):
        ch = list(np.nonzero(Hchk[:, q])[0])
        assert len(ch) in (1, 2)
        ends.append((ch[0], ch[1]) if len(ch) == 2 else (ch[0], B))
    adj = [[] for _ in range(m + 1)]
    L = np.array(logicals, dtype=np.uint8)
    for q, (a, b) in enumerate(ends):
        mask = int("".join(str(int(x)) for x in L[:, q]), 2)
        adj[a].append((b, mask))
        adj[b].append((a, mask))
    best = None
    for s in range(m + 1):  # кратчайший замкнутый путь из (s, 0) в (s, ненулевая маска)
        dist = {(s, 0): 0}
        frontier = [(s, 0)]
        dcur = 0
        while frontier and (best is None or dcur < best):
            dcur += 1
            nxt = []
            for (u, msk) in frontier:
                for (w, qm) in adj[u]:
                    st = (w, msk ^ qm)
                    if st not in dist:
                        dist[st] = dcur
                        nxt.append(st)
                        if w == s and st[1] != 0:
                            best = dcur if best is None else min(best, dcur)
            frontier = nxt
    return best


# ===========================================================================
section("1. Торический код Китаева")


def toric(L):
    nq = 2 * L * L
    h = lambda r, c: (r % L) * L + (c % L)
    v = lambda r, c: L * L + (r % L) * L + (c % L)
    HX = np.zeros((L * L, nq), dtype=np.uint8)  # звёзды (X)
    HZ = np.zeros((L * L, nq), dtype=np.uint8)  # плакеты (Z)
    for r in range(L):
        for c in range(L):
            for q in (h(r, c), h(r, c - 1), v(r, c), v(r - 1, c)):
                HX[r * L + c, q] = 1
            for q in (h(r, c), h(r + 1, c), v(r, c), v(r, c + 1)):
                HZ[r * L + c, q] = 1
    return HX, HZ, h, v


TOR = {}
for L in (3, 4, 5):
    HX, HZ, h, v = toric(L)
    assert np.all((HX @ HZ.T) % 2 == 0)  # звёзды и плакеты коммутируют (общие рёбра — 0 или 2)
    rX, rZ = gf2_rank(HX), gf2_rank(HZ)
    k = 2 * L * L - rX - rZ
    assert rX == L * L - 1 and rZ == L * L - 1 and k == 2
    assert np.all(HX.sum(0) % 2 == 0) and np.all(HZ.sum(0) % 2 == 0)  # произведение всех звёзд (плакетов) = I
    # логические: Z на горизонтальных рёбрах строки 0 (цикл по прямой решётке), X на вертикальных рёбрах строки 0
    # (цикл по дуальной решётке) и т. д.
    Z1 = np.zeros(2 * L * L, np.uint8); Z1[[h(0, c) for c in range(L)]] = 1
    Z2 = np.zeros(2 * L * L, np.uint8); Z2[[v(r, 0) for r in range(L)]] = 1
    X1 = np.zeros(2 * L * L, np.uint8); X1[[v(0, c) for c in range(L)]] = 1
    X2 = np.zeros(2 * L * L, np.uint8); X2[[h(r, 0) for r in range(L)]] = 1
    for Zl in (Z1, Z2):
        assert np.all((HX @ Zl) % 2 == 0) and not in_rowspace(HZ, Zl)
    for Xl in (X1, X2):
        assert np.all((HZ @ Xl) % 2 == 0) and not in_rowspace(HX, Xl)
    assert (Z1 @ X2) % 2 == 1 and (Z1 @ X1) % 2 == 0 and (Z2 @ X1) % 2 == 1 and (Z2 @ X2) % 2 == 0
    dZ = css_distance(HX, [X1, X2])  # Z-ошибки: проверяют звёзды, логичность — по X1, X2
    dX = css_distance(HZ, [Z1, Z2])
    assert dZ == L and dX == L
    TOR[L] = (HX, HZ, h, v)
    print(f"  L = {L}: n = {2 * L * L} кубитов, ранги звёзд/плакетов {rX}/{rZ}, k = {k}, дистанция Z/X: {dZ}/{dX}")
# сверка дистанции перебором для L = 3
HX3, HZ3, h3, v3 = TOR[3]
_, Zc3, _ = css_logicals(HX3, HZ3)
found = None
for w in range(1, 4):
    for qs in itertools.combinations(range(18), w):
        e = np.zeros(18, np.uint8); e[list(qs)] = 1
        if np.all((HX3 @ e) % 2 == 0) and not in_rowspace(HZ3, e):
            found = w
            break
    if found:
        break
assert found == 3
print("  перебор для L = 3: наименьший вес нетривиального Z-цикла = 3 (совпадает)")
# возбуждения на концах струны Z на горизонтальных рёбрах строки 0, столбцы 0..2 (L = 5)
HX5, HZ5, h5, v5 = TOR[5]
e = np.zeros(50, np.uint8); e[[h5(0, c) for c in range(3)]] = 1
viol = np.nonzero((HX5 @ e) % 2)[0]
assert sorted(viol.tolist()) == [0 * 5 + 0, 0 * 5 + 3]
print("  струна Z на трёх рёбрах строки 0 нарушает ровно две звезды — в вершинах (0,0) и (0,3): возбуждения на концах")
# стягиваемый цикл = произведение плакетов
loop = (HZ5[6] + HZ5[7]) % 2
assert np.all((HX5 @ loop) % 2 == 0) and in_rowspace(HZ5, loop)
print("  граница двух соседних плакетов (цикл длины 6) коммутирует со звёздами и лежит в стабилизаторе")

# ===========================================================================
section("2. Планарный и повёрнутый поверхностные коды")


def planar(d):
    """Планарный код: сетка (2d-1)^2, данные в (i,j) с чётной i+j, X-проверки в (чёт, нечёт), Z — в (нечёт, чёт)."""
    N = 2 * d - 1
    data = {(i, j): t for t, (i, j) in enumerate([(i, j) for i in range(N) for j in range(N) if (i + j) % 2 == 0])}
    xs = [(i, j) for i in range(0, N, 2) for j in range(1, N, 2)]
    zs = [(i, j) for i in range(1, N, 2) for j in range(0, N, 2)]

    def rows(chks):
        M = np.zeros((len(chks), len(data)), np.uint8)
        for a, (i, j) in enumerate(chks):
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (i + di, j + dj) in data:
                    M[a, data[(i + di, j + dj)]] = 1
        return M
    return rows(xs), rows(zs)


def rotated(d):
    """Повёрнутый код: данные (i,j), 0 <= i,j < d; грань (a,b), 0 <= a,b <= d, покрывает данные (a-1..a, b-1..b).
    Внутренние грани — все; X-тип при чётной a+b. Граничные грани веса 2: сверху/снизу — только X, слева/справа — только Z."""
    X_, Z_ = [], []
    for a in range(d + 1):
        for b in range(d + 1):
            qs = [(i, j) for i in (a - 1, a) for j in (b - 1, b) if 0 <= i < d and 0 <= j < d]
            isX = (a + b) % 2 == 0
            interior = 1 <= a <= d - 1 and 1 <= b <= d - 1
            if interior:
                (X_ if isX else Z_).append(qs)
            elif a in (0, d) and 1 <= b <= d - 1 and isX:
                X_.append(qs)
            elif b in (0, d) and 1 <= a <= d - 1 and not isX:
                Z_.append(qs)

    def rows(chks):
        M = np.zeros((len(chks), d * d), np.uint8)
        for t, qs in enumerate(chks):
            for (i, j) in qs:
                M[t, i * d + j] = 1
        return M
    return rows(X_), rows(Z_)


print("    d   планарный: данные, проверки, k, d_Z, d_X   |  повёрнутый: данные, проверки, всего кубитов, k, d_Z, d_X")
SURF = {}
for d in (3, 5, 7):
    PX, PZ = planar(d)
    RX, RZ = rotated(d)
    out = []
    for HXc, HZc in ((PX, PZ), (RX, RZ)):
        assert np.all((HXc @ HZc.T) % 2 == 0)
        k, Zc, Xc = css_logicals(HXc, HZc)
        assert k == 1
        Zl = Zc[0]
        Xl = next(x for x in Xc if (x @ Zl) % 2 == 1)
        dZ = css_distance(HXc, [Xl])
        dX = css_distance(HZc, [Zl])
        out.append((HXc.shape[1], HXc.shape[0] + HZc.shape[0], k, dZ, dX))
    (pn, pc, pk, pdz, pdx), (rn, rc, rk, rdz, rdx) = out
    assert pn == d * d + (d - 1) ** 2 and pc == pn - 1 and (pdz, pdx) == (d, d)
    assert rn == d * d and rc == d * d - 1 and rn + rc == 2 * d * d - 1 and (rdz, rdx) == (d, d)
    SURF[d] = (pn, pn + pc, rn + rc)
    print(f"    {d}   {pn:4d} {pc:4d}  {pk}  {pdz}  {pdx}                       |  {rn:4d} {rc:4d}  {rn + rc:4d}   {rk}  {rdz}  {rdx}")
# максимальный вес проверки 4; у повёрнутого кода на границе — веса 2
RX5, RZ5 = rotated(5)
assert set(RX5.sum(1)) | set(RZ5.sum(1)) == {2, 4}
# логические операторы: Z на строке (от одной Z-границы до другой), X на столбце
for d in (3, 5):
    RXd, RZd = rotated(d)
    zrow = np.zeros(d * d, np.uint8); zrow[[0 * d + j for j in range(d)]] = 1
    xcol = np.zeros(d * d, np.uint8); xcol[[i * d + 0 for i in range(d)]] = 1
    assert np.all((RXd @ zrow) % 2 == 0) and not in_rowspace(RZd, zrow)
    assert np.all((RZd @ xcol) % 2 == 0) and not in_rowspace(RXd, xcol) and (zrow @ xcol) % 2 == 1
    # Z на кубите левого столбца нарушает ровно одну X-проверку: Z-струна может там заканчиваться
    e1 = np.zeros(d * d, np.uint8); e1[2 * d + 0] = 1
    assert ((RXd @ e1) % 2).sum() == 1
print("  повёрнутый код: Z_L = Z на строке, X_L = X на столбце; Z-струна может заканчиваться на левой/правой стороне")
print("  повёрнутый код: проверки весов 4 (внутри) и 2 (на границах); всего 2d^2 - 1 кубитов (допуск ЛР14)")

# ===========================================================================
section("3. Декодеры: MWPM и union-find; точное сравнение с ML на торе 3 x 3")


class DecodingGraph:
    """Граф декодирования: вершины — детекторы, рёбра — механизмы ошибок (u, v, логическая маска)."""

    def __init__(self, n_nodes, edges):
        self.n = n_nodes
        self.edges = edges  # список (u, v, mask)
        self.adj = [[] for _ in range(n_nodes)]
        for e, (u, w, _) in enumerate(edges):
            self.adj[u].append(e)
            self.adj[w].append(e)

    def uf_decode(self, defects):
        """Union-find (Дельфосс–Никерсон): рост нечётных кластеров на полребра, слияние, «очистка» леса.
        Возвращает логическую маску исправления."""
        n, adj, edges = self.n, self.adj, self.edges
        parent = list(range(n))
        parity = [0] * n
        for v in defects:
            parity[v] = 1
        boundary = {v: [v] for v in defects}
        support = {}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        active = set(defects)
        self.rounds = 0
        while active:
            self.rounds += 1
            fusion = []
            for r in list(active):
                for v in boundary[r]:
                    for e in adj[v]:
                        s = support.get(e, 0)
                        if s < 2:
                            support[e] = s + 1
                            if s + 1 == 2:
                                fusion.append(e)
            for e in fusion:
                u, w, _ = edges[e]
                ru, rw = find(u), find(w)
                if ru == rw:
                    continue
                bu = boundary.pop(ru, [u] if ru == u else [])
                bw = boundary.pop(rw, [w] if rw == w else [])
                if not bu:
                    bu = [u]
                if not bw:
                    bw = [w]
                parent[rw] = ru
                parity[ru] ^= parity[rw]
                boundary[ru] = bu + bw
            new_active = set()
            for r in set(find(a) for a in active):
                bl = [v for v in set(boundary.get(r, [])) if any(support.get(e, 0) < 2 for e in adj[v])]
                boundary[r] = bl
                if parity[r]:
                    new_active.add(r)
            active = new_active
        # очистка: остовный лес по полностью выросшим рёбрам
        grown = [e for e, s in support.items() if s == 2]
        gadj = {}
        for e in grown:
            u, w, _ = edges[e]
            gadj.setdefault(u, []).append((w, e))
            gadj.setdefault(w, []).append((u, e))
        dfl = {v: 0 for v in gadj}
        for v in defects:
            dfl[v] = 1
        seen = set()
        mask = 0
        for root in gadj:
            if root in seen:
                continue
            order, par = [root], {root: None}
            seen.add(root)
            i = 0
            while i < len(order):
                u = order[i]
                i += 1
                for (w, e) in gadj[u]:
                    if w not in seen:
                        seen.add(w)
                        par[w] = (u, e)
                        order.append(w)
            for v in reversed(order[1:]):
                if dfl[v]:
                    u, e = par[v]
                    mask ^= edges[e][2]
                    dfl[v] = 0
                    dfl[u] ^= 1
            assert dfl[root] == 0
        return mask


def toric_graph(L):
    """Код-ёмкостный граф для X-ошибок торического кода: вершины — плакеты, рёбра — кубиты.
    Маска: бит 0 — ребро h(0, c) (пересекает Z_1), бит 1 — ребро v(r, 0) (пересекает Z_2)."""
    edges = []
    for r in range(L):
        for c in range(L):  # h(r, c) лежит в плакетах (r-1, c) и (r, c)
            edges.append((((r - 1) % L) * L + c, r * L + c, 1 if r == 0 else 0))
    for r in range(L):
        for c in range(L):  # v(r, c) лежит в плакетах (r, c-1) и (r, c)
            edges.append((r * L + (c - 1) % L, r * L + c, 2 if c == 0 else 0))
    return DecodingGraph(L * L, edges)


def toric_mwpm_mask(L, defects):
    """MWPM на торе: расстояние — манхэттенское с учётом периодичности; путь — сначала по горизонтали, затем по вертикали."""
    if not defects:
        return 0
    G = nx.Graph()
    pos = [(v // L, v % L) for v in defects]
    for a in range(len(defects)):
        for b in range(a + 1, len(defects)):
            dr = abs(pos[a][0] - pos[b][0]); dc = abs(pos[a][1] - pos[b][1])
            G.add_edge(a, b, weight=-(min(dr, L - dr) + min(dc, L - dc)))
    M = nx.max_weight_matching(G, maxcardinality=True)
    mask = 0
    for a, b in M:
        (r1, c1), (r2, c2) = pos[a], pos[b]
        # горизонтальный отрезок по строке r1: пересекает v(r1, 0), если идёт через границу столбцов L-1 | 0
        dc = (c2 - c1) % L
        if dc != 0 and (dc <= L - dc):  # вправо от c1 на dc шагов
            if c1 + dc >= L:
                mask ^= 2
        elif dc != 0:  # влево на L - dc шагов
            if c1 - (L - dc) < 0:
                mask ^= 2
        dr = (r2 - r1) % L
        if dr != 0 and (dr <= L - dr):
            if r1 + dr >= L:
                mask ^= 1
        elif dr != 0:
            if r1 - (L - dr) < 0:
                mask ^= 1
    return mask


def sample_toric_cc(L, p, shots):
    """Независимые X-ошибки с вероятностью p: возвращает список (дефекты, истинная маска)."""
    g = toric_graph(L)
    E = np.array([(u, w, m) for (u, w, m) in g.edges])
    errs = rng.random((shots, len(E))) < p
    out = []
    for row in errs:
        idx = np.nonzero(row)[0]
        syn = np.zeros(L * L, np.uint8)
        np.add.at(syn, E[idx, 0], 1)
        np.add.at(syn, E[idx, 1], 1)
        mask = int(np.bitwise_xor.reduce(E[idx, 2])) if len(idx) else 0
        out.append((list(np.nonzero(syn % 2)[0]), mask))
    return out, g


# точное сравнение на L = 3: все 2^18 ошибок, решение декодера зависит только от синдрома
L = 3
g3 = toric_graph(L)
E3 = np.array(g3.edges)
allerr = ((np.arange(2 ** 18)[:, None] >> np.arange(17, -1, -1)) & 1).astype(np.uint8)
syn_all = np.zeros((2 ** 18, 9), np.int64)
for q, (u, w, m) in enumerate(g3.edges):
    syn_all[:, u] += allerr[:, q]
    syn_all[:, w] += allerr[:, q]
syn_all %= 2
syn_id = syn_all @ (2 ** np.arange(8, -1, -1))
masks = np.array([m for (_, _, m) in g3.edges])
mask_all = np.bitwise_xor.reduce(allerr * masks, axis=1)
wt_all = allerr.sum(1)
dec_mask = {}
for s in np.unique(syn_id):
    defects = [i for i in range(9) if (s >> (8 - i)) & 1]
    dec_mask[int(s)] = (toric_mwpm_mask(3, defects), g3.uf_decode(defects))
mw_arr = np.array([dec_mask[int(s)][0] for s in syn_id])
uf_arr = np.array([dec_mask[int(s)][1] for s in syn_id])
print(f"  тор 3 x 3, X-ошибки с вероятностью p (18 кубитов, {len(dec_mask)} синдромов):")
print("     p       MWPM        union-find    ML (оптимальный)")
EXACT3 = {}
for p in (0.01, 0.05, 0.1, 0.15):
    prob = p ** wt_all * (1 - p) ** (18 - wt_all)
    P = np.zeros((512, 4))
    np.add.at(P, (syn_id, mask_all), prob)
    pl_ml = 1 - P.max(1).sum()
    pl_mw = 1 - prob[mw_arr == mask_all].sum()
    pl_uf = 1 - prob[uf_arr == mask_all].sum()
    EXACT3[p] = (pl_mw, pl_uf, pl_ml)
    assert pl_ml <= pl_mw + 1e-12 and pl_ml <= pl_uf + 1e-12
    print(f"   {p:5.2f}   {pl_mw:.6f}    {pl_uf:.6f}      {pl_ml:.6f}")
# MWPM исправляет все ошибки веса 1 (d = 3)
assert np.all(mw_arr[wt_all <= 1] == mask_all[wt_all <= 1]) and np.all(uf_arr[wt_all <= 1] == mask_all[wt_all <= 1])
print("  все ошибки веса 1 исправляются обоими декодерами; ML лучше при больших p")
n2_mw = int(np.sum((mw_arr != mask_all) & (wt_all == 2)))
n2_uf = int(np.sum((uf_arr != mask_all) & (wt_all == 2)))
print(f"  ошибок веса 2 (из {comb(18, 2)}), исправленных неверно: MWPM {n2_mw}, UF {n2_uf}")
assert n2_uf > n2_mw
# задача 3 (в дуальной записи): X на v(0,1), v(0,2), v(0,3) при L = 5 -> дефекты в плакетах (0,0) и (0,3)
g5 = toric_graph(5)
err_edges = [25 + 0 * 5 + c for c in (1, 2, 3)]  # индексы рёбер v(0, c) в toric_graph
syn5 = np.zeros(25, int)
for e in err_edges:
    u, w, m = g5.edges[e]
    syn5[u] ^= 1; syn5[w] ^= 1
    assert m == 0
assert sorted(np.nonzero(syn5)[0].tolist()) == [0, 3]
assert toric_mwpm_mask(5, [0, 3]) == 2  # кратчайший путь (2 ребра) идёт через границу: вместе с ошибкой — цикл вокруг тора
print("  задача 3: три ошибки подряд при L = 5 — MWPM выбирает путь длины 2 в обход тора -> логическая ошибка")
# задача 10: union-find на цепочке из 8 вершин, дефекты в вершинах 2 и 5 (расстояние 3)
chain = DecodingGraph(8, [(i, i + 1, 1 << i) for i in range(7)])
mk = chain.uf_decode([2, 5])
used = [i for i in range(7) if (mk >> i) & 1]
assert used == [2, 3, 4] and chain.rounds == 3
print(f"  задача 10: UF на цепочке, дефекты 2 и 5: {chain.rounds} раунда роста, исправление — рёбра {used} (путь между дефектами)")


# ===========================================================================
section("4. Порог в модели code capacity (тор, независимые X-ошибки)")
P_GRID = (0.08, 0.09, 0.10, 0.11, 0.12)
L_LIST = (5, 7, 9)  # нечётные L: нет «ничьих» при L/2 ошибках
SHOTS = 3000
CC = {"MWPM": {}, "UF": {}}
for L in L_LIST:
    for p in P_GRID:
        samples, g = sample_toric_cc(L, p, SHOTS)
        f_mw = np.mean([toric_mwpm_mask(L, df) != m for df, m in samples])
        f_uf = np.mean([g.uf_decode(df) != m for df, m in samples])
        CC["MWPM"][(L, p)] = f_mw
        CC["UF"][(L, p)] = f_uf
print(f"  p_L (доля отказов, {SHOTS} испытаний; стандартная погрешность <= 0.009)")
print("     p     " + "   ".join(f"MWPM L={L}" for L in L_LIST) + "   " + "   ".join(f"UF L={L}" for L in L_LIST))
for p in P_GRID:
    print(f"   {p:4.2f}   " + "     ".join(f"{CC['MWPM'][(L, p)]:.3f}" for L in L_LIST) + "      "
          + "     ".join(f"{CC['UF'][(L, p)]:.3f}" for L in L_LIST))


def crossing(tab, L1, L2, grid):
    """Точка пересечения кривых p_L(L1) и p_L(L2) линейной интерполяцией разности."""
    diff = [tab[(L2, p)] - tab[(L1, p)] for p in grid]
    for i in range(len(grid) - 1):
        if diff[i] < 0 <= diff[i + 1]:
            return grid[i] + (grid[i + 1] - grid[i]) * (-diff[i]) / (diff[i + 1] - diff[i])
    return None


TH = {}
for dec in ("MWPM", "UF"):
    xs = [crossing(CC[dec], L_LIST[i], L_LIST[i + 1], P_GRID) for i in range(len(L_LIST) - 1)]
    TH[dec] = xs
    print(f"  {dec}: пересечения кривых L={L_LIST[0]}/{L_LIST[1]} и {L_LIST[1]}/{L_LIST[2]}: "
          + ", ".join("—" if x is None else f"{x:.3f}" for x in xs))
assert all(x is not None and 0.085 < x < 0.12 for x in TH["MWPM"])
assert all(x is not None and 0.08 < x < 0.115 for x in TH["UF"])
# ниже порога ошибка убывает с L, выше — растёт
assert CC["MWPM"][(9, 0.08)] < CC["MWPM"][(5, 0.08)] and CC["MWPM"][(9, 0.12)] > CC["MWPM"][(5, 0.12)]
assert CC["UF"][(9, 0.08)] < CC["UF"][(5, 0.08)] and CC["UF"][(9, 0.12)] > CC["UF"][(5, 0.12)]

# ===========================================================================
print(f"  [время с начала: {time.time() - T_START:.0f} с]")
section("5. Феноменологическая модель: ошибки данных и измерения синдрома (union-find)")


def pheno_graph(L, T):
    """Детекторы (t, плакет), t = 0..T; пространственные рёбра в слоях 0..T-1 (ошибка данных перед раундом t),
    временные рёбра (t, p)-(t+1, p), t = 0..T-1 (ошибка измерения в раунде t). Раунд T — идеальный."""
    base = toric_graph(L)
    edges, kinds = [], []
    for t in range(T):
        for (u, w, m) in base.edges:
            edges.append((t * L * L + u, t * L * L + w, m))
            kinds.append(0)
    for t in range(T):
        for pl in range(L * L):
            edges.append((t * L * L + pl, (t + 1) * L * L + pl, 0))
            kinds.append(1)
    return DecodingGraph((T + 1) * L * L, edges), np.array(kinds)


def run_pheno(L, p, shots):
    T = L
    g, kinds = pheno_graph(L, T)
    E = np.array(g.edges)
    fails = 0
    for _ in range(shots):
        hit = rng.random(len(E)) < p
        idx = np.nonzero(hit)[0]
        syn = np.zeros(g.n, np.uint8)
        np.add.at(syn, E[idx, 0], 1)
        np.add.at(syn, E[idx, 1], 1)
        defects = list(np.nonzero(syn % 2)[0])
        mask = int(np.bitwise_xor.reduce(E[idx, 2])) if len(idx) else 0
        fails += g.uf_decode(defects) != mask
    return fails / shots


# проверка модели детекторов: одиночная ошибка измерения даёт пару дефектов в соседних слоях
gph, kinds = pheno_graph(4, 4)
u, w, m = gph.edges[np.nonzero(kinds == 1)[0][5]]
assert w - u == 16 and m == 0
PH_GRID = (0.020, 0.025, 0.030, 0.035)
PH_L = (4, 6, 8)
PH_SHOTS = 1000
PH = {}
for L in PH_L:
    for p in PH_GRID:
        PH[(L, p)] = run_pheno(L, p, PH_SHOTS)
print(f"  тор L x L, L раундов с ошибками измерения (вероятность p) + идеальный раунд; {PH_SHOTS} испытаний")
print("     p      " + "      ".join(f"L={L}" for L in PH_L))
for p in PH_GRID:
    print(f"   {p:5.3f}   " + "   ".join(f"{PH[(L, p)]:.3f}" for L in PH_L))
xs_ph = [crossing(PH, PH_L[i], PH_L[i + 1], PH_GRID) for i in range(len(PH_L) - 1)]
print("  пересечения: " + ", ".join("—" if x is None else f"{x:.3f}" for x in xs_ph))
assert PH[(8, 0.020)] < PH[(4, 0.020)] and PH[(8, 0.035)] > PH[(4, 0.035)]
assert all(x is not None and 0.018 < x < 0.035 for x in xs_ph)

# ===========================================================================
print(f"  [время с начала: {time.time() - T_START:.0f} с]")
section("6. Подавление ошибки ниже порога и ресурсная оценка")
LAM = {}
for p in (0.03, 0.05):
    pls = {}
    for L in (3, 5, 7):
        samples, g = sample_toric_cc(L, p, 20000 if p == 0.03 else 8000)
        pls[L] = np.mean([toric_mwpm_mask(L, df) != m for df, m in samples])
    LAM[p] = pls
    print(f"  code capacity, MWPM, p = {p}: p_L(L=3,5,7) = " + ", ".join(f"{pls[L]:.2e}" for L in (3, 5, 7))
          + f";  Lambda_3/5 = {pls[3] / pls[5]:.2f}, Lambda_5/7 = {pls[5] / pls[7]:.2f}")
assert LAM[0.03][3] / LAM[0.03][5] > LAM[0.05][3] / LAM[0.05][5] > 1
# формула p_L = A (p / p_th)^((d+1)/2) с A = 0.1, p_th = 0.01 (предположение для оценки, как в ЛР14)
A_, pth_ = 0.1, 0.01
d_need = next(d for d in range(3, 101, 2) if A_ * (1e-3 / pth_) ** ((d + 1) / 2) <= 1e-12 * (1 + 1e-9))
nq = 2 * d_need ** 2 - 1
assert d_need == 21 and nq == 881
print(f"  оценка: p = 1e-3, A = 0.1, p_th = 1 %: p_L <= 1e-12 при d = {d_need}; повёрнутый код — {nq} физических кубитов")
d_need2 = next(d for d in range(3, 201, 2) if A_ * (2e-3 / pth_) ** ((d + 1) / 2) <= 1e-12 * (1 + 1e-9))
nq2 = 2 * d_need2 ** 2 - 1
print(f"  при p = 2e-3 (Lambda = 5): d = {d_need2}, {nq2} физических кубитов")
print(f"  Lambda = p_th / p = {pth_ / 1e-3:.0f} на шаг d -> d + 2 в этой модели")

# ===========================================================================
section("7. Ответы к задачам")
print("  1: 2d^2 - 1: " + ", ".join(f"d={d}: {SURF[d][2]}" for d in (3, 5, 7)))
print("  5: планарный код: данные d^2+(d-1)^2, всего (2d-1)^2: " + ", ".join(f"d={d}: {SURF[d][0]}/{SURF[d][1]}" for d in (3, 5, 7)))
pmw, puf, pml = EXACT3[0.1]
print(f"  6: тор 3x3, p = 0.1: MWPM {pmw:.4f}, UF {puf:.4f}, ML {pml:.4f}")
print(f"  8: p = 1e-3: d = {d_need}, кубитов {nq}; p = 2e-3: d = {d_need2}, кубитов {nq2}")
print(f"  11: пересечения MWPM: " + ", ".join(f"{x:.3f}" for x in TH["MWPM"]))

print(f"\nВсе проверки пройдены. Время работы: {time.time() - T_START:.1f} с")
