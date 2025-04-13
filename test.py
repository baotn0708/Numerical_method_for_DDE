import numpy as np

class DDE_CRK_SecondClass_Adaptive:
    """
    Giải DDE với phương pháp Continuous Runge-Kutta (CRK) second class
    + phát hiện break point
    + adaptive step size (đơn giản one-step vs two-half-steps).
    
    Scheme: 5 stage (4 chính + 1 phụ), c=[0,1/3,0.5,2/3,1].
    """

    def __init__(self, f, tau, phi, t_0, y0, t_final,
                 break_points=None,   # danh sách điểm gián đoạn
                 h_init=0.1, h_min=1e-6, h_max=0.5,
                 tol=1e-4):
        """
        f, tau, phi, t_0, y0, t_final: như trước
        break_points: list các điểm t mà ta biết f, tau, hoặc đạo hàm gián đoạn
        h_init, h_min, h_max: bước ban đầu, min, max
        tol: tolerance để so sánh sai số local (one-step vs two-half-steps)
        """
        self.f = f
        self.tau = tau
        self.phi = phi
        self.t_0 = t_0
        self.y0 = y0
        self.t_final = t_final
        self.h_init = h_init
        self.h_min = h_min
        self.h_max = h_max
        self.tol = tol
        
        # Nếu break_points=None => ko có điểm gián đoạn
        if break_points is None:
            self.break_points = []
        else:
            self.break_points = sorted(break_points)
        
        # Lưu nghiệm
        self.times = [t_0]
        self.sol = [y0]
        self.K_stages_list = []
        
        # Định nghĩa scheme
        self.c = np.array([0, 1/3, 0.5, 2/3, 1.0])
        self.A = np.zeros((5,5))
        # A[1,0]=1/3; A[2,0]=0.2; A[2,1]=0.3; A[3,1]=2/3; A[4,2]=1.0
        self.A[1,0] = 1/3
        self.A[2,0] = 0.2
        self.A[2,1] = 0.3
        self.A[3,1] = 2/3
        self.A[4,2] = 1.0
        
        # b_main
        self.b_main = np.array([1/6, 1/3, 0, 1/3, 1/6])
        
        # Nút nội suy
        self.X_interp = self.c
    
    def _lagrange_basis(self, x_nodes, k, theta):
        """Tính L_k(theta) - Lagrange cơ bản."""
        L = 1.0
        xk = x_nodes[k]
        for m, xm in enumerate(x_nodes):
            if m != k:
                L *= (theta - xm)/(xk - xm)
        return L
    
    def history_solution(self, t_query):
        """
        Trả về y(t_query) qua nội suy. 
        Nếu t_query < t_0 => phi(t_query).
        Nếu t_query > self.times[-1] => sol[-1].
        """
        if t_query <= self.t_0:
            return self.phi(t_query)
        t_end = self.times[-1]
        if t_query >= t_end:
            return self.sol[-1]
        
        # Tìm đoạn
        for i in range(len(self.times)-1):
            t_n = self.times[i]
            t_np1 = self.times[i+1]
            if t_n <= t_query <= t_np1:
                h = t_np1 - t_n
                theta = (t_query - t_n)/h
                y_n = self.sol[i]
                K_stages = self.K_stages_list[i]
                
                # Xây Y_vals:
                # Y[0]=y_n
                # Y[1]= y_n + h*A[1,0]*K1
                # ...
                # Y[4]= y_{n+1}
                Y_vals = [None]*5
                Y_vals[0] = y_n
                for j in range(1,4):
                    y_stage = np.copy(y_n)
                    for m in range(j):
                        y_stage += h*self.A[j,m]*K_stages[m]
                    Y_vals[j] = y_stage
                # y_{n+1}
                y_np1 = np.copy(y_n) + h*np.sum([self.b_main[j]*K_stages[j] 
                                                 for j in range(5)], axis=0)
                Y_vals[4] = y_np1
                
                # Lagrange
                val = np.zeros_like(y_n)
                for k in range(5):
                    Lk = self._lagrange_basis(self.X_interp, k, theta)
                    val += Lk*Y_vals[k]
                return val
        return None  # không tìm thấy (hiếm khi)
    
    def _compute_stages(self, t_n, y_n, h):
        """Tính K1..K5."""
        K = []
        for j in range(5):
            t_stage = t_n + self.c[j]*h
            y_stage = np.copy(y_n)
            for m in range(j):
                y_stage += h*self.A[j,m]*K[m]
            # Tính y_delay
            delay = self.tau(t_stage, y_stage)
            t_delay = t_stage - delay
            y_delay = self.history_solution(t_delay)
            # K_j
            K_j = self.f(t_stage, y_stage, y_delay)
            K.append(K_j)
        return K
    
    def do_one_step(self, t_n, y_n, h):
        """
        Thực hiện 1 bước CRK-second-class, 
        trả về (t_{n+1}, y_{n+1}, K_stages).
        """
        K_stages = self._compute_stages(t_n, y_n, h)
        y_np1 = np.copy(y_n) + h*np.sum([self.b_main[j]*K_stages[j] 
                                        for j in range(5)], axis=0)
        return (t_n + h, y_np1, K_stages)
    
    def _attempt_step_adaptive(self, t_n, y_n, h):
        """
        Thử thực hiện 1 bước cỡ h (full_step),
        rồi 2 bước cỡ h/2 (two_half_step),
        so sánh sai số => quyết định chấp nhận hay từ chối.
        
        Trả về (ok, h_recommended, t_np1, y_np1, K_stages)
         - ok: bool, nếu True => chấp nhận
         - h_recommended: bước mới (nếu ok, có thể >h; nếu ko, <h)
         - t_np1, y_np1, K_stages: kết quả của full_step
        """
        # --- 1) Full step ---
        t_full, y_full, K_full = self.do_one_step(t_n, y_n, h)
        
        # --- 2) Two half steps ---
        #  ta cần tạm “bộ nhớ” => tạm cất, 
        #  sau đó khôi phục.
        #  Ở đây, ta làm cục bộ => ko ghi vào self.times.
        
        # Bước 1 (h/2)
        t_mid, y_mid, K_mid = self.do_one_step(t_n, y_n, h/2)
        # Bước 2 (h/2)
        t_half2, y_half2, K_half2 = self.do_one_step(t_mid, y_mid, h/2)
        
        # So sánh sai số = norm(y_full - y_half2)
        err_est = abs(y_full - y_half2)  # hoặc: np.linalg.norm(np.atleast_1d(y_full) - np.atleast_1d(y_half2), ord=np.inf)
        
        # Kiểm tra với tol
        if err_est < self.tol:
            # chấp nhận
            # => ta có thể nới h
            h_new = min(2*h, self.h_max)
            return True, h_new, t_full, y_full, K_full
        else:
            # từ chối => giảm h
            h_new = max(h/2, self.h_min)
            return False, h_new, t_full, y_full, K_full
    
    def _detect_break_point(self, t_n, h):
        """
        Kiểm tra xem có break point trong (t_n, t_n + h).
        Nếu có, cắt h để dừng đúng break point nhỏ nhất.
        Nếu ko, return h.
        """
        t_next = t_n + h
        # Tìm break point tbp: t_n < tbp <= t_next
        for tbp in self.break_points:
            if tbp > t_n and tbp <= t_next:
                # cắt h
                return tbp - t_n
        return h  # ko có break point
    
    def solve_with_adaptive(self):
        """
        Giải từ t_0 đến t_final với adaptive step 
        + phát hiện break points (cắt bước).
        """
        t_n = self.t_0
        y_n = self.y0
        h = self.h_init
        
        while t_n < self.t_final - 1e-15:
            # Kiểm tra cắt bớt h để không vượt t_final
            if t_n + h > self.t_final:
                h = self.t_final - t_n
            
            # Kiểm tra break point
            h_cut = self._detect_break_point(t_n, h)
            
            if h_cut < h:
                # Tồn tại break point => thực hiện 1 bước cỡ h_cut 
                # (ko adaptive cho bước này, 
                #  vì ta muốn dừng chính xác tại break)
                t_np1, y_np1, K_stages = self.do_one_step(t_n, y_n, h_cut)
                # Lưu
                self.times.append(t_np1)
                self.sol.append(y_np1)
                self.K_stages_list.append(K_stages)
                
                t_n = t_np1
                y_n = y_np1
                # Giảm h còn (h - h_cut) cho vòng lặp kế,
                #  nhưng ta break ra vòng while => sang vòng lặp mới
                h = h - h_cut
                if h < self.h_min:
                    h = self.h_min
                continue
            else:
                # Thử bước adaptive cỡ h
                ok, h_new, t_full, y_full, K_full = self._attempt_step_adaptive(t_n, y_n, h)
                if ok:
                    # chấp nhận
                    self.times.append(t_full)
                    self.sol.append(y_full)
                    self.K_stages_list.append(K_full)
                    
                    t_n = t_full
                    y_n = y_full
                    h = h_new
                else:
                    # từ chối => giảm h => lặp lại
                    h = h_new
                    if h <= 1e-14:
                        raise RuntimeError("Bước h quá nhỏ, không thể tiến hành.")
        
        return np.array(self.times), np.array(self.sol)

# =========================
# Ví dụ sử dụng
# =========================
if __name__ == "__main__":
    from math import log

    # 1) Định nghĩa f, tau, phi
    def f(t, u, v):
        return ((t-1)/t)*u*v  # cẩn thận t>0
    
    def tau(t, y):
        return log(t) + 1  # cẩn thận t>0
    
    def phi(t):
        # Giả sử y=1 cho t<=1
        return 1.0

    # 2) Giả sử ta biết f có gián đoạn bậc cao tại t=2.5, t=4.0 (ví dụ)
    #    Thực ra, ở code này, f ko hẳn gián đoạn. Ta chỉ DEMO logic.
    break_points = [2.5, 4.0]

    # 3) Tạo solver
    solver = DDE_CRK_SecondClass_Adaptive(
        f=f, tau=tau, phi=phi,
        t_0=1.0, y0=1.0, t_final=6.0,
        break_points=break_points,
        h_init=0.5, h_min=1e-4, h_max=1.0,
        tol=1e-3
    )

    # 4) Giải
    t_vals, y_vals = solver.solve_with_adaptive()

    # 5) In kết quả
    print("Kết quả giải (adaptive + breakpoints):")
    for (tt, yy) in zip(t_vals, y_vals):
        print(f"t = {tt:.3f}, y = {yy:.6f}")
    
    # 6) Thử nội suy
    t_test = 3.7
    y_test = solver.history_solution(t_test)
    print(f"\nNội suy tại t={t_test:.2f} => y ~ {y_test}")
    import matplotlib.pyplot as plt
    plt.plot(t_vals, y_vals, 'o-')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.show()