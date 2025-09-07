import numpy as np
from math import factorial
from scipy.special import comb
from scipy.stats import binom
from collections import defaultdict


# ----------------------------
# Core building blocks (shared)
# ----------------------------

def _poly_powers_all(base: np.ndarray, max_pow: int) -> list:
    """
    Return list dist[i] = base(x)^i as a PMF over sums (coefficients), i = 0..max_pow.
    base is a 1D array of length L with base[j] = P(step = j) for j=0..L-1.
    dist[i] is length i*(L-1)+1 (sum range 0..i*(L-1)).
    """
    dists = [np.array([1.0], dtype=float)]
    if max_pow == 0:
        return dists
    curr = dists[0]
    for i in range(1, max_pow + 1):
        curr = np.convolve(curr, base)  # full convolution
        dists.append(curr)
    return dists


def _pmf_T_marginal(l: int, N: int, Pi: float, Pd: float) -> np.ndarray:
    """
    pmf of T = sum of N block-sums D_i, where each D_i is sum of l increments in {+1,0,-1}
    with p(+1)=Pi, p(-1)=Pd, p(0)=1-Pi-Pd.
    Returns array length 2*l*N+1 indexed by (T + l*N).
    """
    pmf1 = _pmf_of_single_block(l, Pi, Pd)  # length 2*l+1, support [-l..+l]
    pmfT = pmf1.copy()
    for _ in range(1, N):
        pmfT = np.convolve(pmfT, pmf1)
    return pmfT  # support [-lN..+lN], index T + l*N


def _prob_failure_event2(l: int, N: int, lambda_depths, Pi: float, Pd: float) -> float:
    """
    Event-2 (silent-pairs) failure probability using a compact 'lambda_depths' list:
      - lambda_depths[m] is the threshold depth for |Δ| = m, for m=0..lim-1
      - outside the window (|Δ| >= lim), event-2 is ignored (never triggers)
    This matches the “ignore outside window” semantics you want.

    Args:
      l: block length
      N: number of blocks
      lambda_depths: list/1D array of length lim, with depths for |Δ|=0..lim-1
      Pi, Pd: insertion/deletion proportions (p(+1)=Pi, p(-1)=Pd)

    Returns:
      Pr_cond2_th (float): theoretical probability of event-2 failure
    """
    lambda_depths = np.asarray(lambda_depths, dtype=int)
    lim = int(len(lambda_depths))
    m_max = int(lambda_depths.max()) if lim > 0 else 0

    # --- single-block PMF (note the order: Pi for +1, Pd for -1) ---
    probs = _pmf_of_single_block(l, Pi, Pd)  # length 2l+1, support [-l..l]

    # Decompose as in your MATLAB port
    probs_neg = np.flip(probs[:l+1])     # |d| = 0..l for d<=0  (degrees 0..l)
    probs_pos = probs[l:]                # d = 0..l             (degrees 0..l)

    # Build polynomial bases aligned by degree = step size
    # For 1..l: put a zero at degree 0 so base[j] aligns with degree j
    p_neg_nozero = probs_neg[1:]                       # length l  (steps 1..l)
    p_pos_nozero = probs_pos[1:]                       # length l
    base_neg_nozero = np.concatenate(([0.0], p_neg_nozero))  # degrees 0..l (deg0=0)
    base_pos_nozero = np.concatenate(([0.0], p_pos_nozero))
    base_neg_with0  = probs_neg.copy()                 # degrees 0..l
    base_pos_with0  = probs_pos.copy()

    # Precompute polynomial power tables
    #  - nozero bases up to exponent m (we only ever need degree m <= m_max)
    #  - with0 bases up to exponent i2 in 0..N (degree up to l*i2)
    dists_neg_nozero = _poly_powers_all(base_neg_nozero, m_max)  # list of arrays
    dists_pos_nozero = _poly_powers_all(base_pos_nozero, m_max)
    dists_neg_with0  = _poly_powers_all(base_neg_with0,  N)
    dists_pos_with0  = _poly_powers_all(base_pos_with0,  N)

    # jointPMF[m, idx] where idx = Δ + offsetT, m=0..m_max
    jointPMF = np.zeros((m_max + 1, 2*l*N + 1), dtype=float)
    offsetT = l * N

    # Fill jointPMF only up to m_max (we don't need larger m for inside-window cuts)
    for m in range(0, m_max + 1):
        # Precompute p1 for i1=0..m (prob S_{i1}=m with steps in 1..l)
        p1_neg = np.zeros(m + 1, dtype=float)
        p1_pos = np.zeros(m + 1, dtype=float)
        for i1 in range(0, m + 1):
            dist_neg = dists_neg_nozero[i1]
            dist_pos = dists_pos_nozero[i1]
            p1_neg[i1] = dist_neg[m] if m <= (l * i1) else 0.0
            p1_pos[i1] = dist_pos[m] if m <= (l * i1) else 0.0

        # Loop Δ in [-lN..+lN]
        for Delta in range(-l*N, l*N + 1):
            s = 0.0
            for i1 in range(0, m + 1):
                i2 = N - i1
                wC = comb(N, i1)
                if Delta >= 0:
                    target = Delta + m
                    if 0 <= target <= l * i2:
                        p2 = dists_pos_with0[i2][target]
                        s += wC * p1_neg[i1] * p2
                else:
                    target = (-Delta) + m
                    if 0 <= target <= l * i2:
                        p2 = dists_neg_with0[i2][target]
                        s += wC * p1_pos[i1] * p2
            jointPMF[m, Delta + offsetT] = s

    # ---------- Final probability with "ignore outside window" ----------
    # Success mass = all mass for |Δ| <= lim with m ≤ lambda_depths[|Δ|]
    success = 0.0

    for Delta in range(-(lim - 1), (lim - 1) + 1):
        lam = int(lambda_depths[abs(Delta)])
        idx = Delta + offsetT
        success += jointPMF[:lam + 1, idx].sum()

    Pr_fail = 1.0 - float(success)
    # numerical safety
    if Pr_fail < 0:  Pr_fail = 0.0
    if Pr_fail > 1:  Pr_fail = 1.0
    return Pr_fail


def _pmf_of_single_block(l: int, p_minus: float, p_plus: float) -> np.ndarray:
    """
    pmfD[d + l] = P(D = d), d in [-l..l], where each of the l steps is in {+1, 0, -1}
    with probabilities p_plus, p_minus, r = 1 - p_plus - p_minus.
    """
    r = 1.0 - p_plus - p_minus
    pmf = np.zeros(2 * l + 1, dtype=float)
    # d = K - M where K=#(+1) and M=#(-1)
    for d in range(-l, l + 1):
        val = 0.0
        # K from max(0,d) to l; M = K - d
        for K in range(max(0, d), l + 1):
            M = K - d
            if M < 0 or K + M > l:
                continue
            val += comb(l, K) * comb(l - K, M) * (p_minus ** K) * (p_plus ** M) * (r ** (l - K - M))
        pmf[d + l] = val
    return pmf


# def _prob_silent_pairs_exceed_lambda(l, p, q, N, lambda_array): #computes prob of event 2 with different approach
#     maxT = l * N
#     maxNeg = l * N
#     maxPos = l * N
#     offsetT = maxT
    
#     pmfD = _pmf_of_single_block(l, p, q)
    
#     state = defaultdict(float)
#     state[(offsetT, 0, 0)] = 1.0
    
#     for n in range(N):
#         new_state = defaultdict(float)
#         for (tIdx, negIdx, posIdx), prob in state.items():
#             if prob < 1e-20:
#                 continue
#             tVal = tIdx - offsetT
#             for d in range(-l, l + 1):
#                 pD = pmfD[d + l]
#                 if pD < 1e-20:
#                     continue
#                 tNew = tVal + d
#                 negNew = negIdx + max(0, -d)
#                 posNew = posIdx + max(0, d)
#                 if abs(tNew) > maxT or negNew > maxNeg or posNew > maxPos:
#                     continue
#                 tIdxNew = tNew + offsetT
#                 new_state[(tIdxNew, negNew, posNew)] += prob * pD
#         state = new_state

#     P = 0.0
#     for (tIdx, negIdx, posIdx), prob in state.items():
#         if prob < 1e-20:
#             continue
#         tVal = tIdx - offsetT
#         lambdaT = lambda_array[tIdx]
#         if tVal > 0:
#             if negIdx > lambdaT:
#                 P += prob
#         elif tVal < 0:
#             if posIdx > lambdaT:
#                 P += prob
#         else:
#             if negIdx == posIdx and negIdx > lambdaT:
#                 P += prob
                
#     return P





def _prob_failure_event1(Pd: float, Pi: float, Ps: float, l: int, c1: int, N: int) -> float:
    """
    Python version of prob_failure_event1.
    """
    # p0: all-zero in a block of length l (no edit)
    p0 = (1.0 - (Pd + Pi + Ps)) ** l

    # p1: complementary of the 'balanced' deletion/insertion no-substitution cases up to floor(l/2)
    p1 = 0.0
    for i in range(0, (l // 2) + 1):
        p1 += factorial(l) / (factorial(i) * factorial(i) * factorial(l - 2 * i)) * (Pd ** i) * (Pi ** i) * ((1 - Pd - Pi) ** (l - 2 * i))
    p1 = 1.0 - p1

    # p2: the rest
    p2 = 1.0 - p0 - p1

    # sum over i,j then complement
    acc = 0.0
    for i in range(0, c1 + 1):
        for j in range(0, (c1 - i) // 2 + 1):
            acc += comb(N, i) * comb(N - i, j) * (p1 ** i) * (p2 ** j) * (p0 ** (N - i - j))
    return float(1.0 - acc)


# ----------------------------
# Repetition code — DP section
# ----------------------------

def _pmf_delta(k: int, n: int, p_plus: float, p_minus: float) -> float:
    """
    pmfDelta(k, n, p_plus, p_minus) from MATLAB:
    P(total offset = k) after n steps in {+1, 0, -1} with probs p_plus, p_minus, r.
    """
    if n < 0:
        return 0.0
    if n == 0:
        return 1.0 if k == 0 else 0.0
    r = 1.0 - p_plus - p_minus
    pval = 0.0
    # z is # of zeros
    for z in range(0, n + 1):
        x = (n - z + k) / 2.0  # # of +1
        y = (n - z - k) / 2.0  # # of -1
        if x >= 0 and y >= 0 and abs(x - int(x)) < 1e-12 and abs(y - int(y)) < 1e-12:
            x = int(x)
            y = int(y)
            pval += comb(n, x) * comb(n - x, y) * (p_plus ** x) * (p_minus ** y) * (r ** z)
    return float(pval)


def _compute_cdf_sj(n: int, t: int, j: int, k: int, Pd: float, Ps: float, Pi: float, a: float, b: float):
    """
    Python version of Compute_CDF_Sj with safe s-indexing:
    any transition to s'>t is capped to s'=t (we only store 0..t).
    Returns (CDF over s=0..t, PMF over s=0..t) for block j.
    """
    import numpy as np

    pNo = 1.0 - Pd - Ps - Pi
    if j == k:
        b = 0.5  # match MATLAB special case

    # DP axes: i in [0..3t], offset in [-n..n] (shifted by +n), s in [0..t]
    DP = np.zeros((3 * t + 1, 2 * n + 1, t + 1), dtype=float)

    processedBits = max((j - 2) * t, 0)
    offsetShift = n  # so position 0 -> index n

    # Base distribution at i=0 (MATLAB preloads DP(1,...))
    for i_offset in range(-processedBits, processedBits + 1):
        DP[0, i_offset + offsetShift, 0] = _pmf_delta(i_offset, processedBits, Pi, Pd)

    def add(i_idx, off_idx, s_new, val):
        """Add probability mass with s clamped into [0..t]."""
        if val == 0.0:
            return
        if s_new < 0:
            s_clamped = 0
        elif s_new > t:
            s_clamped = t
        else:
            s_clamped = s_new
        DP[i_idx, off_idx, s_clamped] += val

    # propagate i = 0..3t-1 -> i+1
    for i in range(0, 3 * t):
        maxOffset = processedBits + i
        minOffset = -processedBits - i
        for offset in range(minOffset, maxOffset + 1):
            offIdx = offset + offsetShift
            for s in range(0, min(t, i) + 1):
                probState = DP[i, offIdx, s]
                if probState < 1e-15:
                    continue

                bitIndex = processedBits + i
                inBlockJ   = ((j - 1) * t <= bitIndex <= j * t - 1)
                inBlockJm1 = ((j - 2) * t <= bitIndex <= (j - 1) * t - 1)
                inBlockJp1 = (j * t <= bitIndex <= (j + 1) * t - 1)

                # 1) Deletion: offset -> offset - 1
                newOff = offset - 1
                add(i + 1, newOff + offsetShift, s, probState * Pd)

                # Effective position with current offset:
                effPos = bitIndex + offset

                # 2) Substitution: offset unchanged
                if (j - 1) * t <= effPos <= j * t - 1:
                    if inBlockJ:
                        add(i + 1, offIdx, s, probState * Ps)
                    elif inBlockJm1:
                        add(i + 1, offIdx, s + 1, probState * Ps * (1.0 - a))
                        add(i + 1, offIdx, s,     probState * Ps * a)
                    elif inBlockJp1:
                        add(i + 1, offIdx, s + 1, probState * Ps * (1.0 - b))
                        add(i + 1, offIdx, s,     probState * Ps * b)
                    else:
                        add(i + 1, offIdx, s,     probState * Ps)
                else:
                    add(i + 1, offIdx, s,         probState * Ps)

                # 3) No error: offset unchanged
                if (j - 1) * t <= effPos <= j * t - 1:
                    if inBlockJ:
                        add(i + 1, offIdx, s + 1, probState * pNo)
                    elif inBlockJm1:
                        add(i + 1, offIdx, s + 1, probState * pNo * a)
                        add(i + 1, offIdx, s,     probState * pNo * (1.0 - a))
                    elif inBlockJp1:
                        add(i + 1, offIdx, s + 1, probState * pNo * b)
                        add(i + 1, offIdx, s,     probState * pNo * (1.0 - b))
                    else:
                        add(i + 1, offIdx, s,     probState * pNo)
                else:
                    add(i + 1, offIdx, s,         probState * pNo)

                # 4) Insertion: offset -> offset + 1
                newOff = offset + 1
                newOffIdx = newOff + offsetShift
                if (j - 1) * t <= effPos < j * t:
                    if inBlockJ:
                        add(i + 1, newOffIdx, s + 1, probState * Pi * 0.5)
                        add(i + 1, newOffIdx, s + 2, probState * Pi * 0.5)
                    elif inBlockJm1:
                        add(i + 1, newOffIdx, s + 1, probState * Pi * 0.5)
                        add(i + 1, newOffIdx, s + 2, probState * Pi * 0.5 * a)
                        add(i + 1, newOffIdx, s,     probState * Pi * 0.5 * (1.0 - a))
                    elif inBlockJp1:
                        add(i + 1, newOffIdx, s + 1, probState * Pi * 0.5)
                        add(i + 1, newOffIdx, s + 2, probState * Pi * 0.5 * b)
                        add(i + 1, newOffIdx, s,     probState * Pi * 0.5 * (1.0 - b))
                    else:
                        add(i + 1, newOffIdx, s,     probState * Pi)
                elif effPos == j * t:
                    add(i + 1, newOffIdx, s,     probState * Pi * 0.5)
                    add(i + 1, newOffIdx, s + 1, probState * Pi * 0.5)
                elif effPos == (j - 1) * t:
                    if inBlockJ:
                        add(i + 1, newOffIdx, s + 1, probState * Pi)
                    elif inBlockJm1:
                        add(i + 1, newOffIdx, s + 1, probState * Pi * a)
                        add(i + 1, newOffIdx, s,     probState * Pi * (1.0 - a))
                    elif inBlockJp1:
                        add(i + 1, newOffIdx, s + 1, probState * Pi * b)
                        add(i + 1, newOffIdx, s,     probState * Pi * (1.0 - b))
                    else:
                        add(i + 1, newOffIdx, s,     probState * Pi)
                else:
                    add(i + 1, newOffIdx, s,         probState * Pi)

    # PMF over s at i = 3*t (sum across offsets)
    pmfS = DP[3 * t].sum(axis=0)[: t + 1]
    cdfS = np.cumsum(pmfS)
    return cdfS, pmfS



def _prob_error_rep(k: int, t: int, Pd: float, Pi: float, Ps: float):
    """
    Python version of prob_error_rep.
    Returns (PrE, Prs_all, CDFs_all, PMFs_all), but we only need PrE.
    Majority threshold index is (t-1)//2 for zero-based s.
    """
    Prs = np.zeros(k, dtype=float)
    CDFs = []
    PMFs = []
    n = k * t + k  # as in MATLAB
    thr_idx = (t - 1) // 2  # zero-based index into CDF

    for j in range(1, k + 1):  # MATLAB j=1..k
        CDF00, PMF00 = _compute_cdf_sj(n, t, j, k, Pd, Ps, Pi, 0.0, 0.0)
        CDF01, PMF01 = _compute_cdf_sj(n, t, j, k, Pd, Ps, Pi, 0.0, 1.0)
        CDF10, PMF10 = _compute_cdf_sj(n, t, j, k, Pd, Ps, Pi, 1.0, 0.0)
        CDF11, PMF11 = _compute_cdf_sj(n, t, j, k, Pd, Ps, Pi, 1.0, 1.0)

        CDF = 0.25 * (CDF00 + CDF01 + CDF10 + CDF11)
        PMF = 0.25 * (PMF00 + PMF01 + PMF10 + PMF11)

        CDFs.append(CDF)
        PMFs.append(PMF)

        Prs[j - 1] = CDF[thr_idx]  # probability that a single symbol is decided wrongly

    PrE = float(Prs.sum())
    return PrE


# ---------------------------------------
# Public API: repetition & brute versions
# ---------------------------------------

def theoretical_error_rep(
    Pe: np.ndarray,
    *,
    k: int,
    l: int,
    c1: int,
    c2: int,
    N: int,
    t_rep: int,
    lambda_depths: np.ndarray,
    split=tuple,
):
    
    Pe = np.asarray(Pe, dtype=float)
    Pd_frac, Pi_frac, Ps_frac = split

    event1 = np.zeros_like(Pe)
    event2 = np.zeros_like(Pe)
    parity = np.zeros_like(Pe)

    for i, pe in enumerate(Pe):
        Pd = Pd_frac * pe
        Pi = Pi_frac * pe
        Ps = Ps_frac * pe

        event1[i] = _prob_failure_event1(Pd, Pi, Ps, l, c1, N)
        event2[i] = _prob_failure_event2(l, N, lambda_depths, Pd, Pi)
        parity[i] = _prob_error_rep(k, t_rep, Pd, Pi, Ps)

    total = event1 + event2 + parity
    return total, event1, event2, parity


def theoretical_error_brute(
    Pe: np.ndarray,
    *,
    l: int,
    c1: int,
    c2: int,
    N: int,
    d_min: int,
    n_check: int,
    lambda_depths: np.ndarray,
    split=tuple,
):

    Pe = np.asarray(Pe, dtype=float)
    Pd_frac, Pi_frac, Ps_frac = split

    event1 = np.zeros_like(Pe)
    event2 = np.zeros_like(Pe)
    parity = np.zeros_like(Pe)

    thr = (d_min - 1) // 2  # integer threshold index (matches MATLAB (d_min-1)/2)

    for i, pe in enumerate(Pe):
        Pd = Pd_frac * pe
        Pi = Pi_frac * pe
        Ps = Ps_frac * pe

        event1[i] = _prob_failure_event1(Pd, Pi, Ps, l, c1, N)
        event2[i] = _prob_failure_event2(l, N, lambda_depths, Pd, Pi) #_prob_silent_pairs_exceed_lambda(l, Pd, Pi, N, lambda_array) 
        # parity: 1 - BinomCDF(thr, n_check, pe)
        parity[i] = 1.0 - binom.cdf(thr, n_check, pe)

    total = event1 + event2 + parity
    return total, event1, event2, parity


# ----------------------------
# Example of how you'd call it
# ----------------------------
if __name__ == "__main__":
    c2 = 1
    t_rep3 = 3
    t_rep5 = 5
    l = 8
    q = 2 ** l
    k_bits = 168
    K = k_bits // l
    c1 = 8
    N = K + c1

    k_rep = c2 * l
    l = l // 2  #remove for binary, keep for DNA

    lim = 5 #|Δ|<=lim will be decoded, otherwise failure, important for capping complexity in practice
    lambda_depths = [0]*lim #initialize all depths to zero
    lambda_depths[0] = 1  #depth=1 when Δ = 0
    lambda_depths[1] = 1  #depth=1 when |Δ| = 1
    lambda_depths[2] = 0 #depth=1 when |Δ| = 2

    # Grid of Pe
    Pe = np.arange(0.001, 0.015 + 1e-12, 0.001)
    props = (0.45, 0.02, 0.53)

    # Repetition-3
    total3, event1_3, event2_3, parity3 = theoretical_error_rep(
        Pe,
        k=k_rep,
        l=l,
        c1=c1,
        c2=c2,
        N=N,
        t_rep=t_rep3,
        lambda_depths=lambda_depths,
        split=props
    )

    # Repetition-5
    total5, event1_5, event2_5, parity5 = theoretical_error_rep(
        Pe,
        k=k_rep,
        l=l,
        c1=c1,
        c2=c2,
        N=N,
        t_rep=t_rep5,
        lambda_depths=lambda_depths,
        split=props
    )

    # Brute (choose your d_min and n_check as in MATLAB)
    d_min = 5
    n_check = 12
    totalB, event1_B, event2_B, parityB = theoretical_error_brute(
        Pe,
        l=l,          
        c1=c1,
        c2=c2,
        N=N,
        d_min=d_min,
        n_check=n_check,
        lambda_depths=lambda_depths,
        split=props
    )

    # Example: rates (optional, like your legend)
    rate_rep3 = (K * l) / (N * l + c2 * 3 * l)
    rate_rep5 = (K * l) / (N * l + c2 * 5 * l)
    rate_brut = (K * l) / (N * l + n_check)

    print(f"Done. Example rates -> Rep-3: {rate_rep3:.4f}, Rep-5: {rate_rep5:.4f}, Brute: {rate_brut:.4f}")
