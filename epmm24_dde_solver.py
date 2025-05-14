import numpy as np

# -----------------------------------------------------------------------------
# 1. Generic Lagrange interpolation
# -----------------------------------------------------------------------------
def lagrange_interpolate(t_target, ts, ys, m=4):
    n = len(ts)
    if n == 0:
        raise ValueError("lagrange_interpolate: empty data.")
    if len(ys) != n:
        raise ValueError("ts and ys length mismatch.")
    if n == 1:
        return ys[0]

    idx = np.argsort(np.abs(np.array(ts) - t_target))[: min(m, n)]
    idx.sort()
    sel_t = np.array(ts)[idx]
    sel_y = np.array(ys)[idx]

    result = 0.0
    for i in range(len(sel_t)):
        term = sel_y[i]
        for j in range(len(sel_t)):
            if i == j:
                continue
            denom = sel_t[i] - sel_t[j]
            if abs(denom) < 1e-18:
                term = 0.0
                break
            term *= (t_target - sel_t[j]) / denom
        result += term
    return result


# -----------------------------------------------------------------------------
# 2. Delay accessor
# -----------------------------------------------------------------------------
class DelayAccessor:
    def __init__(self, t0, phi, tau_fun, ts_ref, ys_ref, m_interp=4):
        self.t0 = t0
        self.phi = phi
        self.tau_fun = tau_fun
        self.ts = ts_ref
        self.ys = ys_ref
        self.m_interp = m_interp

    def get(self, k, t_eval, t_for_tau, y_curr):
        tau_val = self.tau_fun(t_for_tau, y_curr)
        td = t_eval - tau_val
        if td <= self.t0 + 1e-12:
            return self.phi(td, order=k)

        if k not in self.ys or len(self.ys[k]) == 0:
            raise RuntimeError(f"No stored values for derivative order {k}.")

        valid = [i for i, t in enumerate(self.ts)
                 if t <= t_eval + 1e-12 and i < len(self.ys[k]) and self.ys[k][i] is not None]
        if not valid:
            return self.phi(td, order=k)

        ts_valid = [self.ts[i] for i in valid]
        ys_valid = [self.ys[k][i] for i in valid]
        return lagrange_interpolate(td, ts_valid, ys_valid, self.m_interp)


# -----------------------------------------------------------------------------
# 3. RK4 one-step
# -----------------------------------------------------------------------------
def rk4_dde_step(f, t0, y0, h, accessor):
    k1 = f(t0, y0, accessor.get(0, t0, t0, y0))
    k2 = f(t0 + h / 2, y0 + 0.5 * h * k1,
           accessor.get(0, t0 + h / 2, t0 + h / 2, y0 + 0.5 * h * k1))
    k3 = f(t0 + h / 2, y0 + 0.5 * h * k2,
           accessor.get(0, t0 + h / 2, t0 + h / 2, y0 + 0.5 * h * k2))
    k4 = f(t0 + h, y0 + h * k3,
           accessor.get(0, t0 + h, t0 + h, y0 + h * k3))
    return y0 + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# -----------------------------------------------------------------------------
# 4. EPMM(2,4) solver
# -----------------------------------------------------------------------------
class EPMM24Solver:
    def __init__(self, f, tau_fun, phi_fun, d2_fun, d3_fun, d4_fun,
                 t_span, H, m_interp=4):
        self.f = f
        self.tau_fun = tau_fun
        self.phi = phi_fun
        self.d2 = d2_fun
        self.d3 = d3_fun
        self.d4 = d4_fun
        self.t0, self.tF = t_span
        self.H = H
        self.m_interp = m_interp

        self.ts = []
        self.y = []
        self.dy = {1: [], 2: [], 3: []}

        self._bootstrap()

    # ---------------- bootstrap -----------------
    def _bootstrap(self):
        self.ts.append(self.t0)
        self.y.append(self.phi(self.t0, 0))
        for k in self.dy:
            self.dy[k].append(None)

        self.acc = DelayAccessor(self.t0, self.phi, self.tau_fun,
                                 self.ts, {0: self.y, **self.dy}, self.m_interp)

        self._compute_derivatives(0)

        h1 = min(self.H, self._dist_to_next_bp(self.t0), self.tF - self.t0)
        if h1 < 1e-14:
            return
        t1 = self.t0 + h1
        y1 = rk4_dde_step(self.f, self.t0, self.y[0], h1, self.acc)

        self.ts.append(t1)
        self.y.append(y1)
        for k in self.dy:
            self.dy[k].append(None)
        self._compute_derivatives(1)

    # ------------- helpers ----------------------
    def _dist_to_next_bp(self, t_now):
        tau0 = self.tau_fun(self.t0, self.y[0])
        if tau0 <= 1e-14:
            return 1e30
        k = np.floor((t_now - self.t0) / tau0) + 1
        next_bp = self.t0 + k * tau0
        gap = next_bp - t_now
        if gap < 1e-12:          # đang ở breakpoint  -> nhảy tới breakpoint kế
            next_bp += tau0
            gap += tau0
        return gap
        # return max(next_bp - t_now, 0.0)

    def _compute_derivatives(self, idx):
        t = self.ts[idx]
        yv = self.y[idx]
        y1 = self.f(t, yv, self.acc.get(0, t, t, yv))
        y2 = self.d2(t, yv, y1, self.acc)
        y3 = self.d3(t, yv, y1, y2, self.acc)
        self.dy[1][idx] = y1
        self.dy[2][idx] = y2
        self.dy[3][idx] = y3
        return y1, y2, y3

    # ------------- integration loop -------------
    def integrate(self):
        if len(self.ts) < 2:
            return self.ts, self.y

        n = 0
        while True:
            t_n = self.ts[n]
            if t_n >= self.tF - 1e-12:
                break

            h = min(self.H, self._dist_to_next_bp(t_n) / 2.0, (self.tF - t_n) / 2.0)
            if h < 1e-14:
                break

            y1, y2, y3 = self.dy[1][n], self.dy[2][n], self.dy[3][n]
            y4 = self.d4(t_n, self.y[n], y1, y2, y3, self.acc)

            y_np2 = self.y[n] + 2 * h * y1 + 2 * h**2 * y2 + (4/3) * h**3 * y3 + (2/3) * h**4 * y4
            t_np2 = t_n + 2 * h

            idx_np2 = n + 2
            idx_np1 = n + 1
            while len(self.ts) <= idx_np2:
                self.ts.append(0.0)
                self.y.append(None)
                for k in self.dy:
                    self.dy[k].append(None)

            self.ts[idx_np2] = t_np2
            self.y[idx_np2] = y_np2

            if self.dy[1][idx_np1] is None:
                self._compute_derivatives(idx_np1)
            self._compute_derivatives(idx_np2)

            n += 1 

        # remove trailing None
        while self.y and self.y[-1] is None:
            self.ts.pop()
            self.y.pop()
            for k in self.dy:
                self.dy[k].pop()
        return self.ts, self.y


# -----------------------------------------------------------------------------
# 5. Example usage
# -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     f = lambda t, y, ylag: -2.0*y - ylag
#     tau = lambda t, y: 1.0
#     phi = lambda t, order=0: 1.0 if order == 0 else 0.0

#     def y2(t, y, y1, acc):
#         return -2.0*y1 - acc.get(1, t, t, y)

#     def y3(t, y, y1, y2_, acc):
#         return -2.0*y2_ - acc.get(2, t, t, y)

#     def y4(t, y, y1, y2_, y3_, acc):
#         return -2.0*y3_ - acc.get(3, t, t, y)

#     solver = EPMM24Solver(f, tau, phi, y2, y3, y4, (0.0, 10.0), H=0.05)
#     ts, ys = solver.integrate()

#     try:
#         import matplotlib.pyplot as plt
#         plt.plot(ts, ys)
#         plt.title("Solution of y' = -2y - y(t-1)")
#         plt.xlabel("t")
#         plt.ylabel("y")
#         plt.grid()
#         plt.show()
#     except Exception:
#         print("Matplotlib not available.")