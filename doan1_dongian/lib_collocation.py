import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numdifftools as nd

# =====================================================
# 1. Các hàm tính nút nội suy cho các phương pháp collocation
# =====================================================

def get_gauss_nodes(s):
    """Tính các nút Gauss-Legendre trên [0,1]."""
    nodes, _ = np.polynomial.legendre.leggauss(s)
    c = 0.5 * (nodes + 1)
    return c

def get_radau_nodes(s):
    """Tính xấp xỉ các nút Radau trên [0,1] với c_s = 1.
       (Cách xấp xỉ đơn giản, thay đổi tùy nhu cầu)"""
    if s == 1:
        return np.array([1.0])
    else:
        c = np.linspace(0, 1, s)
        c[0] = 0.1  # tránh lấy 0, vì Radau thường không lấy 0
        c[-1] = 1.0
        return c

def get_lobatto_nodes(s):
    """Tính các nút Lobatto trên [0,1] với đầu mút 0 và 1."""
    if s == 2:
        return np.array([0.0, 1.0])
    else:
        i = np.arange(0, s)
        c = 0.5 * (1 - np.cos(np.pi * i / (s - 1)))
        return c

def get_chebyshev_nodes(s):
    """Tính các nút Chebyshev trên [0,1]."""
    i = np.arange(1, s + 1)
    c = 0.5 * (1 - np.cos((2 * i - 1) * np.pi / (2 * s)))
    return c

def get_hermite_nodes(s):
    """Ở ví dụ này, ta dùng nút Gauss làm nút cơ bản cho Hermite."""
    return get_gauss_nodes(s)

# =====================================================
# 2. Xây dựng đa thức Lagrange và bảng Butcher
# =====================================================

def compute_lagrange_basis(c, j):
    """
    Tính đa thức cơ sở Lagrange L_j(s) với các nút c:
      L_j(s) = Π_{i ≠ j} (s - c_i)/(c_j - c_i)
    Trả về một đối tượng np.poly1d.
    """
    n = len(c)
    poly = np.poly1d([1.0])
    for i in range(n):
        if i != j:
            poly = np.polymul(poly, np.poly1d([1.0, -c[i]])) / (c[j] - c[i])
    return poly

def compute_butcher_tableau(c):
    """
    Tính bảng Butcher dựa trên các nút collocation:
      a_{ij} = ∫₀^(c_i) L_j(s) ds,   b_j = ∫₀^1 L_j(s) ds.
    """
    s = len(c)
    A = np.zeros((s, s))
    b = np.zeros(s)
    for j in range(s):
        Lj = compute_lagrange_basis(c, j)
        Lj_int = np.poly1d(np.polyint(Lj.coeffs))
        b[j] = Lj_int(1) - Lj_int(0)
        for i in range(s):
            A[i, j] = Lj_int(c[i]) - Lj_int(0)
    return A, b

class CollocationMethod:
    """
    Lớp đại diện cho phương pháp collocation.
    Các phương pháp được hỗ trợ: 'gauss', 'radau', 'lobatto', 'chebyshev', 'hermite'
    """
    def __init__(self, method_name, s):
        self.method_name = method_name.lower()
        self.s = s
        if self.method_name == 'gauss':
            self.c = get_gauss_nodes(s)
        elif self.method_name == 'radau':
            self.c = get_radau_nodes(s)
        elif self.method_name == 'lobatto':
            self.c = get_lobatto_nodes(s)
        elif self.method_name == 'chebyshev':
            self.c = get_chebyshev_nodes(s)
        elif self.method_name == 'hermite':
            self.c = get_hermite_nodes(s)
        else:
            raise ValueError("Phương pháp chưa được hỗ trợ: " + method_name)
        self.A, self.b = compute_butcher_tableau(self.c)
    
    def continuous_extension(self, theta):
        """
        Tính các hệ số mở rộng liên tục: 
          b_i(theta) = ∫₀^θ L_i(s) ds, với θ ∈ [0,1].
        """
        s = self.s
        b_theta = np.zeros(s)
        for j in range(s):
            Lj = compute_lagrange_basis(self.c, j)
            Lj_int = np.poly1d(np.polyint(Lj.coeffs))
            b_theta[j] = Lj_int(theta) - Lj_int(0)
        return b_theta

# =====================================================
# 3. Hàm lấy giá trị lịch sử (History) và nội suy
# =====================================================

def get_history_value(t, history_segments, phi, t0):
    """
    Trả về giá trị y(t) dựa trên lịch sử:
      - Nếu t <= t0, trả về hàm lịch sử φ(t).
      - Nếu t nằm trong một đoạn đã tính, sử dụng phần mở rộng liên tục.
    """
    if t <= t0:
        return phi(t)
    for seg in history_segments:
        if seg['t_start'] <= t <= seg['t_end']:
            theta = (t - seg['t_start']) / seg['h']
            b_theta = seg['collocation'].continuous_extension(theta)
            return seg['y_start'] + seg['h'] * np.dot(b_theta, seg['K'])
    seg = history_segments[-1]
    return seg['y_start'] + seg['h'] * np.dot(seg['collocation'].b, seg['K'])

# =====================================================
# 4. Hàm bước collocation cho DDE (một bước từ t đến t+h)
# =====================================================

def collocation_step_dde(f, tau, get_history_value_func, phi, t, y, h, collocation, t0, history_segments):
    """
    Giải một bước DDE trên [t, t+h]:
      Tìm K_i (i=1,...,s) sao cho:
         K_i = f( t + c_i*h,  y + h*Σ_j A[i,j]*K_j,  y( t+c_i*h - τ(t+c_i*h, y_i) ) )
      với y_i = y + h*Σ_j A[i,j]*K_j.
    Giải hệ phi tuyến này bằng fsolve.
    """
    s = collocation.s
    c = collocation.c
    A = collocation.A
    
    def F(K_flat):
        K = K_flat.reshape((s,) + y.shape)
        F_val = np.zeros_like(K)
        for i in range(s):
            t_i = t + c[i] * h
            # Ước lượng y tại t_i
            y_i = y.copy()
            for j in range(s):
                y_i += h * A[i, j] * K[j]
            # Tính thời gian trễ
            t_delay = t_i - tau(t_i, y_i)
            y_delay = get_history_value_func(t_delay, history_segments, phi, t0)
            F_val[i] = K[i] - f(t_i, y_i, y_delay)
        return F_val.flatten()
    
    K0 = np.zeros((s,) + y.shape)
    K_flat = fsolve(F, K0.flatten())
    K = K_flat.reshape((s,) + y.shape)
    
    y_next = y.copy()
    for j in range(s):
        y_next += h * collocation.b[j] * K[j]
    return y_next, K

# =====================================================
# 5. Hàm lấy "order" của phương pháp (discrete order p)
# =====================================================

def get_method_order(method_name, s):
    """
    Ước lượng bậc của phương pháp collocation:
      - Gauss: p = 2*s,
      - Radau: p = 2*s - 1,
      - Lobatto: p = 2*s - 2,
      - Chebyshev: giả sử p = s,
      - Hermite: giả sử p = 2*s.
    """
    method = method_name.lower()
    if method == 'gauss':
        return 2 * s
    elif method == 'radau':
        return 2 * s - 1
    elif method == 'lobatto':
        return 2 * s - 2
    elif method == 'chebyshev':
        return s
    elif method == 'hermite':
        return 2 * s
    else:
        raise ValueError("Không xác định được bậc cho phương pháp: " + method_name)

# =====================================================
# 6. Thuật toán đa bước thích nghi (Adaptive step size) cho DDE
# =====================================================

def solve_dde_collocation(
    f, tau, phi,
    t0, t_end,
    h0,
    *,                     # all args below là keyword-only
    method_name='gauss',
    s=2,
    adaptive=True,         # nếu False thì dùng counter
    tol=1e-4,
    L_f_u=None, L_f_v=None, L_tau_y=None,
    h_min=1e-6, h_max=1.0,
    time_tol=1e-12,        # ngưỡng chấm động
    rounding_decimals=10,  # số chữ số sau dấu thập phân khi round t
):
    """
    Giải DDE y'(t)=f(t,y(t),y(t-τ(t,y))) với collocation-Gauss bậc s.
    Nếu adaptive=True (mặc định) dùng step-doubling điều khiển bước;
    nếu adaptive=False, chạy bước cố định h0 nhưng tính t bằng counter.

    Trả về (ts, ys) – ndarray thời gian & nghiệm.
    """
    collocation = CollocationMethod(method_name, s)
    p          = get_method_order(method_name, s)   # bậc của công thức
    SAFETY     = 0.9

    t = round(t0, rounding_decimals)
    y = np.atleast_1d(phi(t0))
    ts = [t]
    ys = [y.copy()]
    history_segments = []

    if not adaptive:
        # --- Fixed-step bằng counter ---
        # xác định số bước (làm tròn cho chắc)
        n_steps = int(round((t_end - t0) / h0))
        for k in range(1, n_steps + 1):
            # tính t chính xác từ công thức
            t_next = round(t0 + k * h0, rounding_decimals)
            h = t_next - t

            # một bước collocation-DDE
            y_new, K_new = collocation_step_dde(
                f, tau, get_history_value, phi,
                t, y, h, collocation, t0, history_segments
            )

            # lưu lịch sử
            history_segments.append(
                dict(t_start=t, t_end=t_next, y_start=y.copy(),
                     h=h, collocation=collocation, K=K_new)
            )

            # cập nhật
            t = t_next
            y = y_new.copy()
            ts.append(t)
            ys.append(y.copy())

        # đảm bảo điểm cuối đúng t_end
        if abs(ts[-1] - t_end) < time_tol:
            ts[-1] = t_end

    else:
        # --- Adaptive step-doubling như trước, chỉ thêm clamp bước cuối ---
        h = h0
        while t < t_end - time_tol:
            # clamp để không vượt quá t_end
            if t + h > t_end - time_tol:
                h = t_end - t

            # backup history
            history_backup = history_segments.copy()

            # 1 bước h
            y_full, K_full = collocation_step_dde(
                f, tau, get_history_value, phi,
                t, y, h, collocation, t0, history_segments
            )

            # 2 bước h/2
            y_mid, K_mid = collocation_step_dde(
                f, tau, get_history_value, phi,
                t, y, h/2, collocation, t0, history_segments
            )
            seg_mid = dict(t_start=t, t_end=t+h/2, y_start=y.copy(),
                           h=h/2, collocation=collocation, K=K_mid)
            history_half = history_segments.copy() + [seg_mid]
            y_half, K_half = collocation_step_dde(
                f, tau, get_history_value, phi,
                t + h/2, y_mid, h/2, collocation, t0, history_half
            )

            # ước tính sai số
            err = np.linalg.norm(y_half - y_full) / (2**p - 1)

            if err < tol:
                # chấp nhận bước
                t += h
                # clamp đúng t_end
                if abs(t - t_end) < time_tol:
                    t = t_end
                y = y_half.copy()
                ts.append(round(t, rounding_decimals))
                ys.append(y.copy())
                history_segments.append(
                    dict(t_start=ts[-2], t_end=t, y_start=ys[-2].copy(),
                         h=h, collocation=collocation, K=K_full)
                )
                # update h
                factor = SAFETY * (tol/err)**(1/(p+1))
                h = min(h * min(2.0, max(0.1, factor)), h_max)
            else:
                # từ chối bước: rollback, giảm h
                factor = SAFETY * (tol/err)**(1/(p+1))
                h = max(h * max(0.1, factor), h_min)
                history_segments = history_backup

            # nếu đã chạm hoặc vượt t_end thì kết thúc
            if t >= t_end - time_tol:
                break

    return np.array(ts), np.array(ys)

# =====================================================
# Chạy ví dụ chính
# =====================================================

if __name__ == "__main__":
    # Định nghĩa hàm F và dẫn hàm dF (sử dụng numdifftools)
    def F(t):
        return np.log(1 + t**2) + np.sqrt(1 + t**4)
    
    def dF(t):
        df = nd.Derivative(F)
        return df(t)
    
    # Định nghĩa hàm tau, phi và f, đảm bảo đầu ra luôn là mảng với ít nhất 1 chiều
    def tau(t, y):
        return 2.0
    
    def phi(t):
        return np.atleast_1d(F(t))
    
    def f(t, y, y_delay):
        y = np.atleast_1d(y)
        y_delay = np.atleast_1d(y_delay)
        return dF(t) * np.exp(y / F(t) - F(t - tau(t, y)) / y_delay)
    
    # Các thiết lập thời gian và bước ban đầu
    t0 = 0.1
    t_end = 3.01
    h0 = 0.06
    tol = 1e-6
    
    # Các tham số Lipschitz (dùng như nhắc nhở, chưa dùng trực tiếp)
    L_f_u = 0.0
    L_f_v = 1.0
    L_tau_y = 0.0
    
    ts, ys = solve_dde_collocation_adaptive(f, tau, phi, t0, t_end, h0,
                                              method_name='gauss', s=2, tol=tol,
                                              L_f_u=L_f_u, L_f_v=L_f_v, L_tau_y=L_tau_y,
                                              h_min=1e-6, h_max=0.1)
    
    # Vẽ đồ thị nghiệm tại các điểm bước (ở đây vẽ thành phần đầu tiên nếu y là vector)
    plt.figure(figsize=(10, 6))
    plt.plot(ts, ys[:, 0], 'o-', label='Nghiệm tại điểm bước')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title("Giải DDE")
    plt.legend()
    plt.show()
    
    # Hàm nghiệm chính xác và tính sai số
    def y_exact(t):
        return F(t)
    
    def compute_error(ts, ys):
        return np.abs(ys[:, 0] - y_exact(ts))
    
    errors = compute_error(ts, ys)
    
    def plot_error(ts, errors):
        plt.figure(figsize=(10, 6))
        plt.plot(ts, errors, 'r-', label='Sai số')
        plt.xlabel('t')
        plt.ylabel('Sai số')
        plt.yscale('log')
        plt.title("Sai số giữa nghiệm chính xác và nghiệm nội suy")
        plt.legend()
        plt.show()
    
    plot_error(ts, errors)
