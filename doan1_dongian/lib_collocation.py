import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

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
       (Ví dụ này dùng cách xấp xỉ đơn giản; thực tế cần công thức chính xác hơn.)"""
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
    i = np.arange(1, s+1)
    c = 0.5 * (1 - np.cos((2*i - 1)*np.pi/(2*s)))
    return c

def get_hermite_nodes(s):
    """Ở ví dụ này, ta dùng nút Gauss làm nút cơ bản cho Hermite (để minh họa)."""
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
    # Nếu t vượt quá vùng lịch sử, trả về giá trị của đoạn cuối
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
            t_i = t + c[i]*h
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

def solve_dde_collocation_adaptive(f, tau, phi, t0, t_end, h0, method_name='gauss', s=2, tol=1e-4,
                                   L_f_u=None, L_f_v=None, L_tau_y=None, h_min=1e-6, h_max=1.0):
    """
    Giải DDE:
        y'(t)= f(t, y(t), y(t-τ(t,y(t)))),
        y(t)= φ(t) cho t<= t0,
    với thuật toán bước thích nghi sử dụng kỹ thuật step doubling.
    
    - h0: bước khởi đầu,
    - tol: sai số chấp nhận được (local error tolerance).
    
    Các ước lượng Lipschitz (L_f_u, L_f_v, L_tau_y) được truyền vào nhằm nhắc rằng cần đảm bảo
    tính đặt chỉnh của bài toán; trong ví dụ này chưa dùng trực tiếp để điều chỉnh bước.
    
    Trả về:
      ts: mảng thời gian của các bước,
      ys: nghiệm tại các điểm bước.
    """
    collocation = CollocationMethod(method_name, s)
    t = t0
    y = phi(t0)  # giá trị ban đầu
    ts = [t0]
    ys = [y.copy()]
    history_segments = []  # dùng để lưu trữ các đoạn giải cho nội suy giá trị trễ

    h = h0
    p = get_method_order(method_name, s)  # bậc của phương pháp

    SAFETY = 0.9  # hệ số an toàn khi điều chỉnh bước

    while t < t_end:
        if t + h > t_end:
            h = t_end - t

        # Lưu lại history_segments trước bước, nếu bước sau bị loại (reject) sẽ quay lại
        history_backup = history_segments.copy()
        
        # Tính nghiệm một bước với bước h (solution full)
        y_full, K_full = collocation_step_dde(f, tau, get_history_value, phi, t, y, h, collocation, t0, history_segments)
        
        # Tính nghiệm qua hai bước với bước h/2 (solution half-step)
        # Bước 1: từ t đến t+h/2
        y_mid, K_mid = collocation_step_dde(f, tau, get_history_value, phi, t, y, h/2, collocation, t0, history_segments)
        # Lưu đoạn giữa t -> t+h/2
        seg_mid = {
            't_start': t,
            't_end': t + h/2,
            'y_start': y.copy(),
            'h': h/2,
            'collocation': collocation,
            'K': K_mid
        }
        history_half = history_segments.copy()
        history_half.append(seg_mid)
        # Bước 2: từ t+h/2 đến t+h, dùng y_mid làm giá trị khởi đầu
        y_half, K_half = collocation_step_dde(f, tau, get_history_value, phi, t + h/2, y_mid, h/2, collocation, t0, history_half)
        
        # Ước tính sai số theo kỹ thuật step doubling
        # Ta dùng hiệu sai: error ~ || y_half - y_full || / (2^p - 1)
        error_est = np.linalg.norm(y_half - y_full) / (2**p - 1)

        if error_est < tol:
            # Bước được chấp nhận: cập nhật t, y, và lưu segment toàn bước h (có thể lưu K_full)
            t += h
            y = y_half.copy()
            ts.append(t)
            ys.append(y.copy())
            seg = {
                't_start': ts[-2],
                't_end': t,
                'y_start': ys[-2].copy(),
                'h': h,
                'collocation': collocation,
                'K': K_full  # chấp nhận K từ bước full
            }
            history_segments.append(seg)
            # Điều chỉnh bước mới (tăng nếu sai số nhỏ)
            factor = SAFETY * (tol / error_est)**(1/(p+1))
            h = min(h * min(2.0, max(0.1, factor)), h_max)
        else:
            # Bước không chấp nhận: giảm bước và thử lại, không cập nhật history
            factor = SAFETY * (tol / error_est)**(1/(p+1))
            h = max(h * max(0.1, factor), h_min)
            history_segments = history_backup  # khôi phục history
            print(f"Step rejected at t={t:.4f}, error={error_est:.2e}, new h={h:.2e}")

    return np.array(ts), np.array(ys)

# =====================================================
# 7. Ví dụ minh họa cho DDE thích nghi
#     Ví dụ: y'(t) = -y(t) + y(t-1), với y(t)=1 cho t<=0.
# =====================================================

if __name__ == "__main__":
    # Hàm f: y'(t) = -y(t) + y(t-τ)
    def f(t, u, v):
        return -u + v

    # Hàm τ: độ trễ hằng số 1 (có thể mở rộng thành hàm phụ thuộc t, y)
    def tau(t, y):
        return 1.0

    # Hàm lịch sử: φ(t)=1 cho mọi t <= t0
    def phi(t):
        return np.array([1.0])
    
    t0 = 0.0
    t_end = 5.0
    h0 = 0.1
    tol = 1e-4

    # Các ước lượng Lipschitz (chỉ để nhắc nhở, có thể dùng sau này)
    L_f_u = 1.0
    L_f_v = 1.0
    L_tau_y = 0.0

    # Chọn phương pháp collocation, ví dụ: 'gauss' với 2 giai đoạn
    ts, ys = solve_dde_collocation_adaptive(f, tau, phi, t0, t_end, h0, 
                                              method_name='gauss', s=2, tol=tol,
                                              L_f_u=L_f_u, L_f_v=L_f_v, L_tau_y=L_tau_y,
                                              h_min=1e-6, h_max=0.2)
    
    # Vẽ đồ thị nghiệm tại các điểm bước
    plt.figure(figsize=(10,6))
    plt.plot(ts, ys[:,0], 'o-', label='Nghiệm tại điểm bước')
    
    # Vẽ nghiệm nội suy (trên từng đoạn) nếu cần
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title("Giải DDE thích nghi: y'(t) = -y(t) + y(t-1), y(t)=1 cho t<=0")
    plt.legend()
    plt.show()
