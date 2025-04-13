import numpy as np
import logging
from numpy.polynomial import Polynomial
from scipy.optimize import fsolve
from typing import Optional, List, Dict, Any, Callable, Tuple

# Cấu hình logging để dễ theo dõi quá trình tính toán
logger = logging.getLogger("CollocationSolver")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# =============================================================================
# 1. Các hàm tính nút nội suy theo các phương pháp: Gauss, Radau, Lobatto, Chebyshev, Hermite
# =============================================================================

def get_gauss_nodes(s: int) -> np.ndarray:
    """
    Tính các nút Gauss-Legendre trên [0, 1].
    Sử dụng np.polynomial.legendre.leggauss trên [-1, 1] rồi chuyển đổi.
    """
    nodes, _ = np.polynomial.legendre.leggauss(s)
    c = 0.5 * (nodes + 1)
    logger.debug(f"Gauss nodes (s={s}): {c}")
    return c

def get_radau_nodes(s: int) -> np.ndarray:
    """
    Tính các nút Radau trên [0, 1] với nút cuối bằng 1.
    Trong phiên bản nâng cao, ta dùng phương pháp nội suy qua linspace với một chút điều chỉnh.
    """
    if s == 1:
        return np.array([1.0])
    else:
        # Sử dụng linspace rồi điều chỉnh nút đầu không bằng 0 (Radau không có 0)
        c = np.linspace(0, 1, s)
        c[0] = 0.1  # Điều chỉnh nhỏ, có thể thay đổi theo yêu cầu
        c[-1] = 1.0
        logger.debug(f"Radau nodes (s={s}): {c}")
        return c

def get_lobatto_nodes(s: int) -> np.ndarray:
    """
    Tính các nút Lobatto trên [0, 1] với đầu mút 0 và 1.
    Sử dụng công thức Chebyshev-Gauss-Lobatto.
    """
    if s < 2:
        raise ValueError("Phương pháp Lobatto yêu cầu ít nhất 2 nút.")
    i = np.arange(s)
    c = 0.5 * (1 - np.cos(np.pi * i / (s - 1)))
    logger.debug(f"Lobatto nodes (s={s}): {c}")
    return c

def get_chebyshev_nodes(s: int) -> np.ndarray:
    """
    Tính các nút Chebyshev trên [0, 1].
    Công thức: c_i = 0.5 * (1 - cos((2i-1)*pi/(2s))) với i=1,...,s.
    """
    i = np.arange(1, s + 1)
    c = 0.5 * (1 - np.cos((2 * i - 1) * np.pi / (2 * s)))
    logger.debug(f"Chebyshev nodes (s={s}): {c}")
    return c

def get_hermite_nodes(s: int) -> np.ndarray:
    """
    Đối với Hermite, trong ví dụ này ta sử dụng các nút Gauss làm cơ sở.
    Trong ứng dụng thực, Hermite collocation yêu cầu khớp thêm đạo hàm.
    """
    c = get_gauss_nodes(s)
    logger.debug(f"Hermite nodes (s={s}): {c}")
    return c


# =============================================================================
# 2. Xây dựng đa thức Lagrange và bảng Butcher (A, b)
# =============================================================================

def compute_lagrange_basis(c: np.ndarray, j: int) -> np.poly1d:
    """
    Tính đa thức Lagrange cơ sở L_j(s) dựa trên danh sách nút c.
    L_j(s) = ∏_{i ≠ j} (s - c[i])/(c[j] - c[i])
    """
    n = len(c)
    poly = np.poly1d([1.0])
    for i in range(n):
        if i != j:
            poly = np.polymul(poly, np.poly1d([1.0, -c[i]])) / (c[j] - c[i])
    logger.debug(f"Lagrange basis L_{j}(s): {poly}")
    return poly

def compute_butcher_tableau(c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính bảng Butcher cho collocation method:
      a_{ij} = ∫_0^(c_i) L_j(s) ds,   b_j = ∫_0^1 L_j(s) ds.
    Sử dụng np.polyint để tính tích phân của đa thức.
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
    logger.debug(f"Butcher tableau A:\n{A}\n, b: {b}")
    return A, b


# =============================================================================
# 3. Lớp đại diện cho phương pháp collocation với tùy chọn custom nodes và tính toán bảng Butcher
# =============================================================================

class CollocationMethod:
    """
    Lớp đại diện cho phương pháp collocation.
    Người dùng có thể chọn tên phương pháp (gauss, radau, lobatto, chebyshev, hermite)
    hoặc truyền vào các custom_nodes tùy chỉnh.
    
    Attributes:
      method_name: tên phương pháp dưới dạng chuỗi.
      s: số stage (số nút nội suy).
      c: mảng các nút nội suy.
      A: ma trận Butcher (hệ số nội suy cho các stage).
      b: vector Butcher (trọng số của các stage).
    """
    def __init__(self, method_name: str, s: int, custom_nodes: Optional[List[float]] = None):
        self.method_name = method_name.lower()
        self.s = s
        if custom_nodes is not None:
            self.c = np.array(custom_nodes, dtype=float)
            if len(self.c) != s:
                raise ValueError("Số điểm nội suy (custom_nodes) phải bằng s")
        else:
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
        logger.info(f"Initialized CollocationMethod: {self.method_name} with s={self.s}")

    def continuous_extension(self, theta: float) -> np.ndarray:
        """
        Tính các hệ số mở rộng liên tục: b_i(theta) = ∫₀^θ L_i(s) ds.
        
        Args:
          theta: tham số nội suy trong [0, 1].
        
        Returns:
          mảng b_theta với kích thước (s,).
        """
        b_theta = np.zeros(self.s)
        for j in range(self.s):
            Lj = compute_lagrange_basis(self.c, j)
            Lj_int = np.poly1d(np.polyint(Lj.coeffs))
            b_theta[j] = Lj_int(theta) - Lj_int(0)
        logger.debug(f"Continuous extension at theta={theta}: {b_theta}")
        return b_theta


# =============================================================================
# 4. Hàm ước lượng thứ tự (order) của phương pháp collocation dựa trên method_name và s
# =============================================================================

def collocation_order(method_name: str, s: int) -> int:
    """
    Trả về thứ tự (order) discrete của phương pháp collocation dựa trên số stage s.
    Ví dụ:
      - Gauss: order = 2*s
      - Radau: order = 2*s - 1
      - Lobatto: order = 2*s - 2
      - Các phương pháp khác: mặc định order = s (có thể tùy chỉnh thêm)
    """
    m = method_name.lower()
    if m == 'gauss':
        order_val = 2 * s
    elif m == 'radau':
        order_val = 2 * s - 1
    elif m == 'lobatto':
        order_val = 2 * s - 2
    else:
        order_val = s
    logger.debug(f"Collocation order for {method_name} with s={s}: {order_val}")
    return order_val

def get_embedded_collocation(method_name: str, s: int, custom_nodes: Optional[List[float]] = None) -> Optional[CollocationMethod]:
    """
    Trả về một đối tượng CollocationMethod với số stage embedded = s - 1, dùng để ước lượng sai số.
    Nếu s <= 1, không tồn tại embedded pair.
    """
    if s <= 1:
        logger.warning("Embedded pair không khả dụng với s <= 1.")
        return None
    return CollocationMethod(method_name, s - 1, custom_nodes=custom_nodes)


# =============================================================================
# 5. Lịch sử giải (History) và hàm nội suy cho DDE
# =============================================================================

def get_history_value(t: float,
                      history_segments: List[Dict[str, Any]],
                      phi: Callable[[float], np.ndarray],
                      t0: float) -> np.ndarray:
    """
    Trả về giá trị y(t) dựa trên lịch sử giải:
      - Nếu t <= t0: trả về giá trị từ hàm lịch sử phi(t).
      - Nếu t nằm trong một đoạn giải, sử dụng continuous extension của bước đó.
      - Nếu t > thời gian đã giải, trả về giá trị cuối cùng.
    
    Args:
      t: thời điểm cần nội suy.
      history_segments: danh sách các đoạn đã giải, mỗi đoạn là dict chứa
                        't_start', 't_end', 'h', 'y_start', 'collocation', 'K'.
      phi: hàm lịch sử ban đầu cho t <= t0.
      t0: thời điểm bắt đầu giải.
    
    Returns:
      y_value: giá trị nội suy của y tại thời điểm t.
    """
    if t <= t0:
        return phi(t)
    for seg in history_segments:
        if seg['t_start'] <= t <= seg['t_end']:
            theta = (t - seg['t_start']) / seg['h']
            b_theta = seg['collocation'].continuous_extension(theta)
            y_value = seg['y_start'] + seg['h'] * np.dot(b_theta, seg['K'])
            logger.debug(f"History value at t={t} found in segment [{seg['t_start']}, {seg['t_end']}].")
            return y_value
    # Nếu không nằm trong bất kỳ segment nào, trả về giá trị cuối cùng
    seg = history_segments[-1]
    y_value = seg['y_start'] + seg['h'] * np.dot(seg['collocation'].b, seg['K'])
    logger.debug(f"History value at t={t} using final segment.")
    return y_value


# =============================================================================
# 6. Bước collocation cho DDE (phiên bản nâng cao)
# =============================================================================

def collocation_step_dde(f: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
                         tau: Callable[[float, np.ndarray], float],
                         get_history_value_func: Callable[..., np.ndarray],
                         phi: Callable[[float], np.ndarray],
                         t: float, y: np.ndarray, h: float,
                         collocation: CollocationMethod,
                         t0: float,
                         history_segments: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Giải một bước của DDE trên [t, t+h] theo phương pháp collocation:
      Tìm các stage K_i từ hệ phương trình:
         K_i = f(t_i, y_i, y(t_i - τ(t_i, y_i))),
      với:
         t_i = t + c_i * h,
         y_i = y + h * Σ_j A[i, j] * K_j.
    
    Args:
      f: hàm định nghĩa y'(t) = f(t, y(t), y(t-τ)).
      tau: hàm định nghĩa độ trễ τ(t, y).
      get_history_value_func: hàm nội suy lịch sử.
      phi: hàm lịch sử ban đầu.
      t: thời điểm bắt đầu bước.
      y: giá trị y(t) hiện tại.
      h: bước số.
      collocation: đối tượng CollocationMethod hiện hành.
      t0: thời điểm khởi đầu (cho lịch sử).
      history_segments: danh sách các đoạn giải (dùng cho nội suy giá trị trễ).
    
    Returns:
      y_next: giá trị y tại t+h.
      K: mảng các stage K (kích thước (s, dim(y))).
    """
    s = collocation.s
    c = collocation.c
    A = collocation.A

    def F(K_flat: np.ndarray) -> np.ndarray:
        K = K_flat.reshape((s,) + y.shape)
        F_val = np.zeros_like(K)
        for i in range(s):
            t_i = t + c[i] * h
            y_i = y.copy()
            for j in range(s):
                y_i += h * A[i, j] * K[j]
            # Xác định thời gian trễ: t_delay = t_i - τ(t_i, y_i)
            t_delay = t_i - tau(t_i, y_i)
            y_delay = get_history_value_func(t_delay, history_segments, phi, t0)
            F_val[i] = K[i] - f(t_i, y_i, y_delay)
        return F_val.flatten()

    # Đoán ban đầu cho K
    K0 = np.zeros((s,) + y.shape)
    logger.debug(f"Bắt đầu giải hệ nonlinear với h={h} tại t={t}.")
    K_flat = fsolve(F, K0.flatten())
    K = K_flat.reshape((s,) + y.shape)
    y_next = y.copy()
    for j in range(s):
        y_next += h * collocation.b[j] * K[j]
    logger.debug(f"Bước collocation hoàn thành: y(t+h) = {y_next}")
    return y_next, K


# =============================================================================
# 7. Adaptive bước số với embedded pair và điều chỉnh số stage
# =============================================================================

def adaptive_collocation_step_dde(f: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
                                  tau: Callable[[float, np.ndarray], float],
                                  get_history_value_func: Callable[..., np.ndarray],
                                  phi: Callable[[float], np.ndarray],
                                  t: float, y: np.ndarray, h: float,
                                  method_name: str, s: int, t0: float,
                                  history_segments: List[Dict[str, Any]],
                                  custom_nodes: Optional[List[float]] = None,
                                  tol: float = 1e-5,
                                  safety: float = 0.9,
                                  h_min: float = 1e-6,
                                  h_max: float = 1.0,
                                  max_reject: int = 10) -> Tuple[np.ndarray, np.ndarray, float, int, bool]:
    """
    Thực hiện một bước thích nghi cho DDE với collocation.
    
    Các bước:
      - Tính nghiệm đầy đủ y_full với s stage.
      - Tính nghiệm embedded y_emb với s_emb = s - 1 (nếu có).
      - Ước lượng sai số e = ||y_full - y_emb|| (dùng norm Euclid).
      - Nếu e <= tol: chấp nhận bước và ước lượng bước mới.
      - Nếu e > tol: giảm bước h theo công thức và thử lại.
      - Nếu liên tục từ chối quá nhiều lần, tăng số stage (s_current).
    
    Returns:
      y_full: nghiệm đầy đủ tại t+h.
      K_full: các stage tương ứng.
      h_new: bước số mới được đề xuất.
      s_new: số stage hiện hành (có thể tăng lên).
      accepted: True nếu bước được chấp nhận.
    """
    reject_count = 0
    s_current = s
    while True:
        logger.debug(f"Adaptive step: thử với s={s_current}, h={h}")
        colloc_full = CollocationMethod(method_name, s_current, custom_nodes=custom_nodes)
        y_full, K_full = collocation_step_dde(f, tau, get_history_value_func, phi, t, y, h, colloc_full, t0, history_segments)
        
        embedded = get_embedded_collocation(method_name, s_current, custom_nodes=custom_nodes)
        if embedded is None:
            # Sử dụng phương pháp hai bước (step-doubling) khi không có embedded pair
            y_mid, _ = collocation_step_dde(f, tau, get_history_value_func, phi, t, y, h / 2, colloc_full, t0, history_segments)
            y_two, _ = collocation_step_dde(f, tau, get_history_value_func, phi, t + h / 2, y_mid, h / 2, colloc_full, t0, history_segments)
            y_emb = y_two
            order_emb = 1  # ước lượng đơn giản
        else:
            y_emb, _ = collocation_step_dde(f, tau, get_history_value_func, phi, t, y, h, embedded, t0, history_segments)
            order_emb = collocation_order(method_name, s_current - 1)
        
        err = np.linalg.norm(y_full - y_emb)
        p_emb = order_emb
        logger.debug(f"Ước lượng sai số: {err} (tol={tol}) với p_emb={p_emb}")
        if err <= tol:
            # Chấp nhận bước, tính bước mới theo công thức điều chỉnh
            h_new = h * safety * (tol / (err + 1e-16)) ** (1 / (p_emb + 1))
            h_new = np.clip(h_new, h_min, h_max)
            logger.info(f"Bước chấp nhận với h_new = {h_new}")
            return y_full, K_full, h_new, s_current, True
        else:
            # Từ chối bước, giảm bước h và tăng reject_count
            h = h * safety * (tol / (err + 1e-16)) ** (1 / (p_emb + 1))
            h = max(h, h_min)
            reject_count += 1
            logger.warning(f"Bước từ chối (err={err}). Thử lại với h={h}")
            if reject_count >= max_reject:
                s_current += 1
                logger.warning(f"Tăng số stage lên {s_current} do từ chối quá nhiều lần.")
                reject_count = 0
            if h < h_min:
                raise RuntimeError("Bước số giảm dưới mức cho phép, không hội tụ được.")


# =============================================================================
# 8. Adaptive solver cho DDE với method of steps
# =============================================================================

def solve_dde_collocation_adaptive(f: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
                                   tau: Callable[[float, np.ndarray], float],
                                   phi: Callable[[float], np.ndarray],
                                   t0: float, t_end: float, h0: float,
                                   method_name: str = 'gauss', s0: int = 2,
                                   custom_nodes: Optional[List[float]] = None,
                                   tol: float = 1e-5,
                                   safety: float = 0.9,
                                   h_min: float = 1e-6,
                                   h_max: float = 1.0,
                                   max_reject: int = 10) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Giải DDE:
         y'(t) = f(t, y(t), y(t - τ(t, y(t)))),
         y(t) = φ(t) cho t <= t0,
    trên khoảng [t0, t_end] với bước số thích nghi bằng phương pháp method of steps.
    
    Args:
      f: hàm định nghĩa hệ DDE.
      tau: hàm định nghĩa độ trễ τ(t, y).
      phi: hàm lịch sử cho t <= t0.
      t0: thời điểm bắt đầu giải.
      t_end: thời điểm kết thúc giải.
      h0: bước số ban đầu.
      method_name: tên phương pháp collocation (gauss, radau, lobatto, chebyshev, hermite, ...).
      s0: số stage ban đầu.
      custom_nodes: nếu cung cấp, dùng để định nghĩa các điểm nội suy tùy chỉnh.
      tol: sai số cho bước thích nghi.
      safety, h_min, h_max, max_reject: các tham số điều khiển bước thích nghi.
    
    Returns:
      ts: mảng thời gian các điểm bước.
      ys: mảng nghiệm tại các điểm bước.
      history_segments: danh sách các đoạn giải (dùng cho nội suy các giá trị trễ).
    """
    t = t0
    y = phi(t0)
    ts = [t0]
    ys = [y.copy()]
    history_segments: List[Dict[str, Any]] = []
    h = h0
    s_current = s0
    
    logger.info("Bắt đầu giải DDE thích nghi...")
    while t < t_end:
        if t + h > t_end:
            h = t_end - t
        try:
            y_next, K, h_new, s_current, accepted = adaptive_collocation_step_dde(
                f, tau, get_history_value, phi, t, y, h,
                method_name, s_current, t0, history_segments,
                custom_nodes=custom_nodes, tol=tol, safety=safety,
                h_min=h_min, h_max=h_max, max_reject=max_reject
            )
        except RuntimeError as e:
            logger.error(f"Lỗi khi giải tại t={t}: {e}")
            break

        if accepted:
            colloc = CollocationMethod(method_name, s_current, custom_nodes=custom_nodes)
            segment = {
                't_start': t,
                't_end': t + h,
                'y_start': y.copy(),
                'h': h,
                'collocation': colloc,
                'K': K
            }
            history_segments.append(segment)
            t += h
            y = y_next.copy()
            ts.append(t)
            ys.append(y.copy())
            h = h_new
            logger.info(f"Bước thành công: t = {t}, h = {h}")
        else:
            h = h_new
    return np.array(ts), np.array(ys), history_segments

