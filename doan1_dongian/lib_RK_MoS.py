import numpy as np

# 1) Hàm kiểm tra gần đúng tính Lipschitz và liên tục của tau
def check_conditions(f, tau, phi, t0, delta=1e-6):
    """
    Kiểm tra gần đúng các điều kiện Lipschitz của f theo y và y_delay,
    và tính liên tục của hàm delay tau.
    Chỉ dùng để cảnh báo sơ bộ.
    """
    y0 = phi(t0)  # Giá trị nghiệm tại t0
    tau0 = tau(t0, y0)
    y_delay0 = phi(t0 - tau0)
    
    f0 = f(t0, y0, y_delay0)
    
    # Sai phân hữu hạn theo y
    f_y = (f(t0, y0 + delta, y_delay0) - f0) / delta
    # Sai phân hữu hạn theo y_delay
    f_yd = (f(t0, y0, y_delay0 + delta) - f0) / delta
    
    L_y = abs(f_y)
    L_yd = abs(f_yd)
    
    # Độ liên tục của tau
    tau_cont = abs(tau(t0 + delta, y0) - tau0) / delta
    
    print("Ước lượng Lipschitz theo y: {:.4g}".format(L_y))
    print("Ước lượng Lipschitz theo y_delay: {:.4g}".format(L_yd))
    print("Độ biến thiên (gradient) delay: {:.4g}".format(tau_cont))
    print("(Nếu các giá trị này quá lớn, có thể báo hiệu vi phạm giả định Lipschitz.)\n")
    
    return L_y, L_yd, tau_cont

# 2) Nội suy Hermite bậc 3 để có nghiệm liên tục
def cubic_hermite(t, t0, t1, y0, y1, f0, f1):
    """
    Nội suy giá trị tại t ∈ [t0, t1] dùng Hermite bậc 3
    (dựa vào giá trị và đạo hàm tại 2 đầu).
    """
    s = (t - t0) / (t1 - t0)
    H00 = 2*s**3 - 3*s**2 + 1
    H10 = -2*s**3 + 3*s**2
    H01 = s**3 - 2*s**2 + s
    H11 = s**3 - s**2
    return (y0 * H00 
            + y1 * H10 
            + (t1 - t0) * (f0 * H01 + f1 * H11))

# 3) Lấy giá trị y(t_eval) bằng lịch sử + nội suy trên các nút đã tính
def get_y_at(t_eval, sol, phi, t0):
    # Nếu t_eval <= t0, dùng hàm lịch sử
    if t_eval <= t0:
        return phi(t_eval)
    # Nếu vượt ngoài phạm vi đã tính, trả về giá trị nút cuối
    if t_eval >= sol[-1]['t']:
        return sol[-1]['y']
    # Tìm đoạn [t_i, t_{i+1}] chứa t_eval
    for i in range(len(sol)-1):
        t_i = sol[i]['t']
        t_ip1 = sol[i+1]['t']
        if t_i <= t_eval <= t_ip1:
            y_i = sol[i]['y']
            y_ip1 = sol[i+1]['y']
            f_i = sol[i]['f']
            f_ip1 = sol[i+1]['f']
            return cubic_hermite(t_eval, t_i, t_ip1, y_i, y_ip1, f_i, f_ip1)
    return None

# 4) Thực hiện một bước RK4/RK3 để ước lượng sai số
def rk_step(t, y, h, sol, f, tau, phi, t0):
    # Tính y_delay cho stage 1
    y_delay = get_y_at(t - tau(t, y), sol, phi, t0)
    k1 = f(t, y, y_delay)
    
    # Stage 2
    t2 = t + h/2
    y2 = y + (h/2)*k1
    y_delay = get_y_at(t2 - tau(t2, y2), sol, phi, t0)
    k2 = f(t2, y2, y_delay)
    
    # Stage 3
    t3 = t + h/2
    y3 = y + (h/2)*k2
    y_delay = get_y_at(t3 - tau(t3, y3), sol, phi, t0)
    k3 = f(t3, y3, y_delay)
    
    # Stage 4
    t4 = t + h
    y4 = y + h*k3
    y_delay = get_y_at(t4 - tau(t4, y4), sol, phi, t0)
    k4 = f(t4, y4, y_delay)
    
    # Nghiệm bậc 4 (RK4)
    y_high = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    # Nghiệm bậc 3 (embedded)
    # (có nhiều cách xây dựng embedded RK3, ví dụ dùng Heun / classic, minh họa một biến thể đơn giản)
    y_rk3 = y + (h/4)*(k1 + 2*k2 + k3)  # hoặc 1 cách cổ điển
    # Tính sai số
    err = abs(y_high - y_rk3)
    
    return y_high, err, (k1, k2, k3, k4)

# 5) Thuật toán giải DDE bằng CRK (RK4 + Hermite) với bước tự điều chỉnh
def solve_dde_adaptive(f, tau, phi, t0, t_final, 
                       tol=1e-5, h_init=0.1, h_min=1e-8, h_max=0.5):
    """
    f, tau, phi như mô tả:
       - f(t, y, y_delay)
       - tau(t, y)
       - phi(t): lịch sử
    t0: thời điểm bắt đầu (ở đây t0=1)
    t_final: thời điểm kết thúc
    tol: ngưỡng sai số
    h_init, h_min, h_max: bước khởi tạo, nhỏ nhất, lớn nhất.
    Trả về: sol (list các dict {'t':, 'y':, 'f':})
    """
    sol = []
    # Giá trị ban đầu
    y0 = phi(t0)
    y_delay0 = phi(t0 - tau(t0, y0))  # vì t0=1 => t0 - tau(1,y0) = 1 - (log(1)+1)=0 => phi(0)=1
    f0 = f(t0, y0, y_delay0)
    sol.append({'t': t0, 'y': y0, 'f': f0})
    
    t = t0
    y = y0
    h = h_init
    
    while t < t_final:
        # Điều chỉnh để không vượt quá t_final
        if t + h > t_final:
            h = t_final - t
        
        # Tính một bước
        y_high, err, k_stages = rk_step(t, y, h, sol, f, tau, phi, t0)
        
        # Nếu sai số < tol, chấp nhận bước
        if err <= tol:
            t_new = t + h
            y_new = y_high
            # Tính f tại nút mới
            y_delay_new = get_y_at(t_new - tau(t_new, y_new), sol, phi, t0)
            f_new = f(t_new, y_new, y_delay_new)
            sol.append({'t': t_new, 'y': y_new, 'f': f_new})
            
            t = t_new
            y = y_new
            
            # Điều chỉnh bước cho lần tiếp theo (theo quy tắc tỉ lệ 1/4)
            # Hệ số an toàn 0.9
            factor = 0.9 * (tol / (err + 1e-14))**0.25
            h = min(h * factor, h_max)
        else:
            # Sai số quá lớn => giảm bước
            factor = 0.9 * (tol / (err + 1e-14))**0.25
            h = max(h * factor, h_min)
            # In cảnh báo nếu muốn
            print(f"  [Giảm bước] t={t:.5f}, h={h:.3e}, err={err:.3e}")
            
        # Nếu h quá nhỏ mà vẫn không thoả mãn, có thể dừng/báo lỗi.
        if h < 1e-14:
            print("Bước quá nhỏ, có thể bài toán không thỏa mãn giả định Lipschitz hoặc sai số quá chặt.")
            break
    
    return sol

def f_example(t, y, y_delay):
    return 2*t+y-t**2 +y_delay-(t-1)**2

def tau_example(t, y):
    return 1

def phi_example(t):
    return t**2

# t0 = -1
# t_final = 1

# # 6) Kiểm tra nhanh các điều kiện Lipschitz
# check_conditions(f_example, tau_example, phi_example, t0)

# # 7) Giải DDE
# sol_example = solve_dde_adaptive(
#     f=f_example,
#     tau=tau_example,
#     phi=phi_example,
#     t0=t0,
#     t_final=t_final,
#     tol=1e-6,      # sai số có thể giảm xuống 1e-7, 1e-8 nếu cần
#     h_init=0.1,    # bước khởi tạo
#     h_min=1e-10,   # bước nhỏ nhất
#     h_max=0.5      # bước lớn nhất
# )

# # 8) In (hoặc vẽ) kết quả
# print("\n--- KẾT QUẢ TÍNH ---")
# for node in sol_example:
#     print(f"t = {node['t']:.5f}, y = {node['y']:.8f}")
# #Vẽ đồ thị
# import matplotlib.pyplot as plt
# t_values = [node['t'] for node in sol_example]
# y_values = [node['y'] for node in sol_example]
# plt.plot(t_values, y_values)
# plt.xlabel('t')
# plt.ylabel('y')
# plt.show()
# print(len(t_values))
