import cv2
import numpy as np
from collections import deque

# ==================== Manual Normalize ====================
def normalize_0_255(img):
    mn = img.min()
    mx = img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - mn) / (mx - mn) * 255.0
    return out.astype(np.uint8)

# ==================== Frequency Domain Filters (NumPy FFT) ====================
def apply_freq_filter(img, filter_type, D0=30, n=2, W=10):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # 1) FFT -> shift
    F = np.fft.fft2(img.astype(np.float32))
    F_shift = np.fft.fftshift(F)

    # 2) Distance D(u,v)
    x, y = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - crow) ** 2 + (y - ccol) ** 2)
    dist = np.maximum(dist, 1e-5)

    # 3) Build H(u,v)
    h = np.ones((rows, cols), dtype=np.float32)

    # Low Pass
    if filter_type == "ILPF":
        h = (dist <= D0).astype(np.float32)
    elif filter_type == "BLPF":
        h = 1 / (1 + (dist / D0) ** (2 * n))
    elif filter_type == "GLPF":
        h = np.exp(-(dist ** 2) / (2 * (D0 ** 2)))

    # High Pass (1 - LPF)
    elif filter_type == "IHPF":
        h = 1 - (dist <= D0).astype(np.float32)
    elif filter_type == "BHPF":
        h = 1 - (1 / (1 + (dist / D0) ** (2 * n)))
    elif filter_type == "GHPF":
        h = 1 - np.exp(-(dist ** 2) / (2 * (D0 ** 2)))

    # Band Reject
    elif filter_type == "IBRF":
        h = np.where((dist >= D0 - W / 2) & (dist <= D0 + W / 2), 0, 1).astype(np.float32)
    elif filter_type == "BBRF":
        h = 1 / (1 + (dist * W / (dist ** 2 - D0 ** 2 + 1e-5)) ** (2 * n))
    elif filter_type == "GBRF":
        h = 1 - np.exp(-((dist ** 2 - D0 ** 2) / (dist * W + 1e-5)) ** 2)

    # Band Pass
    elif filter_type == "IBPF":
        h = 1 - np.where((dist >= D0 - W / 2) & (dist <= D0 + W / 2), 0, 1).astype(np.float32)
    elif filter_type == "BBPF":
        h = 1 - (1 / (1 + (dist * W / (dist ** 2 - D0 ** 2 + 1e-5)) ** (2 * n)))
    elif filter_type == "GBPF":
        h = np.exp(-((dist ** 2 - D0 ** 2) / (dist * W + 1e-5)) ** 2)

    # 4) Apply filter in freq domain
    G_shift = F_shift * h

    # 5) Inverse shift -> IFFT
    G = np.fft.ifftshift(G_shift)
    img_back = np.fft.ifft2(G)

    # 6) Magnitude
    img_back = np.abs(img_back)
    return normalize_0_255(img_back)

# ==================== Spatial Filters (Manual) ====================
def window_view(img, k):
    pad = k // 2
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode="edge")
    H, W = img.shape
    s0, s1 = padded.strides
    shape = (H, W, k, k)
    strides = (s0, s1, s0, s1)
    return np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

def apply_spatial_filter(img, ftype, k, Q=1.5, d=2):
    k = max(3, k if k % 2 == 1 else k + 1)
    win = window_view(img.astype(np.float32), k)
    eps = 1e-5

    if ftype == "Arithmetic":
        out = win.mean(axis=(2, 3))
    elif ftype == "Geometric":
        out = np.exp(np.mean(np.log(win + eps), axis=(2, 3)))
    elif ftype == "Harmonic":
        out = (k * k) / (np.sum(1.0 / (win + eps), axis=(2, 3)) + eps)
    elif ftype == "Contraharmonic":
        num = np.sum((win + eps) ** (Q + 1), axis=(2, 3))
        den = np.sum((win + eps) ** Q, axis=(2, 3))
        out = num / (den + eps)
    elif ftype == "Median":
        flat = win.reshape(win.shape[0], win.shape[1], -1)
        out = np.median(flat, axis=2)
    elif ftype == "Max":
        out = win.max(axis=(2, 3))
    elif ftype == "Min":
        out = win.min(axis=(2, 3))
    elif ftype == "Midpoint":
        out = (win.max(axis=(2, 3)) + win.min(axis=(2, 3))) / 2.0
    elif ftype == "Alpha-trimmed":
        flat = win.reshape(win.shape[0], win.shape[1], -1)
        flat_sorted = np.sort(flat, axis=2)
        d = int(d)
        d = max(0, min(d, k * k - 1))
        if d % 2 == 1:
            d -= 1
        trim = d // 2
        if trim == 0:
            out = flat_sorted.mean(axis=2)
        else:
            out = flat_sorted[:, :, trim:-(trim)].mean(axis=2)
    else:
        out = img.astype(np.float32)

    return np.uint8(np.clip(out, 0, 255))

# ==================== Edge Detection (Manual-ish) ====================
def noise_reduction(img, ksize=11, sigma=1):
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (k, k), sigma)

def compute_gradients(img):
    I = img.astype(np.float32)
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)
    Gx = cv2.filter2D(I, cv2.CV_32F, kx)
    Gy = cv2.filter2D(I, cv2.CV_32F, ky)
    mag = np.hypot(Gx, Gy)
    maxv = mag.max() if mag.max() != 0 else 1.0
    mag = (mag / maxv) * 255.0
    angle = np.arctan2(Gy, Gx)
    return Gx, Gy, mag, angle

def non_maximum_suppression(mag, angle):
    M, N = mag.shape
    Z = np.zeros((M, N), dtype=np.float32)
    ang = (angle * 180 / np.pi) % 180
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            a = ang[i, j]
            if (0 <= a < 22.5) or (157.5 <= a < 180):
                q, r = mag[i, j + 1], mag[i, j - 1]
            elif (22.5 <= a < 67.5):
                q, r = mag[i + 1, j - 1], mag[i - 1, j + 1]
            elif (67.5 <= a < 112.5):
                q, r = mag[i + 1, j], mag[i - 1, j]
            else:
                q, r = mag[i - 1, j - 1], mag[i + 1, j + 1]
            if mag[i, j] >= q and mag[i, j] >= r:
                Z[i, j] = mag[i, j]
    return Z

def double_threshold(img, low, high, weak=75, strong=255):
    strong_mask = img >= high
    weak_mask = (img >= low) & (img < high)
    strong_edges = np.zeros_like(img, dtype=np.uint8)
    weak_edges = np.zeros_like(img, dtype=np.uint8)
    strong_edges[strong_mask] = strong
    weak_edges[weak_mask] = weak
    return strong_edges, weak_edges

def hysteresis(strong_edges, weak_edges, weak=75, strong=255):
    M, N = strong_edges.shape
    result = strong_edges.copy()
    visited = np.zeros_like(result)
    strong_pts = list(zip(*np.where(strong_edges == strong)))
    dq = deque(strong_pts)
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]
    while dq:
        x, y = dq.popleft()
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < M and 0 <= ny < N:
                if weak_edges[nx, ny] == weak and visited[nx, ny] == 0:
                    result[nx, ny] = strong
                    visited[nx, ny] = 1
                    dq.append((nx, ny))
    result[result != strong] = 0
    return result

def sobel_alt(gray):
    Gx, Gy, mag, ang = compute_gradients(gray)
    mag_u8 = np.uint8(np.clip(mag, 0, 255))
    return Gx, Gy, mag_u8

# ==================== Helpers for Constraints ====================
def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def make_even(x):
    x = int(x)
    return x if x % 2 == 0 else x - 1

# ==================== Main Merged ====================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error!")
        return

    # Global mode
    main_mode = "FREQ"   # or "EDGES"

    # ---- Frequency state ----
    mode_f = "NORMAL"
    D0, n, W = 30, 2, 10
    ksize, Q, d_param = 3, 1.5, 2
    sub_mode_f = 0
    selected_f = None  # D0/W/N/K/Q/T

    # ---- Edges state ----
    mode_e = "none"
    low_t, high_t = 20, 135
    ksize_e = 11
    selected_e = None  # low/high/k
    step_t = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ==================== Apply current main_mode ====================
        if main_mode == "FREQ":
            # ---- Logical constraints for FREQ ----
            rows, cols = gray.shape
            max_D0 = max(5, (min(rows, cols) // 2) - 1)

            D0 = clamp(D0, 5, max_D0)
            n = clamp(int(n), 1, 20)

            # W <= 2*D0 (prevent negative band inner radius)
            W = clamp(W, 2, max(2, 2 * D0))

            ksize = clamp(ksize, 3, 31)
            if ksize % 2 == 0:
                ksize = clamp(ksize + 1, 3, 31)

            d_param = make_even(d_param)
            d_param = clamp(d_param, 0, ksize * ksize - 1)
            d_param = make_even(d_param)

            Q = clamp(Q, -5.0, 5.0)

            # ---- Apply FREQ filters ----
            if mode_f == "LPF":
                types = ["ILPF", "BLPF", "GLPF"]
                ftype = types[sub_mode_f % 3]
                result = apply_freq_filter(gray, ftype, D0, n)
                display_mode = f"{ftype} | D0={D0} n={n}"

            elif mode_f == "HPF":
                types = ["IHPF", "BHPF", "GHPF"]
                ftype = types[sub_mode_f % 3]
                result = apply_freq_filter(gray, ftype, D0, n)
                display_mode = f"{ftype} | D0={D0} n={n}"

            elif mode_f == "BR":
                types = ["IBRF", "BBRF", "GBRF"]
                ftype = types[sub_mode_f % 3]
                result = apply_freq_filter(gray, ftype, D0, n, W)
                display_mode = f"{ftype} | D0={D0} W={W} n={n}"

            elif mode_f == "BP":
                types = ["IBPF", "BBPF", "GBPF"]
                ftype = types[sub_mode_f % 3]
                result = apply_freq_filter(gray, ftype, D0, n, W)
                display_mode = f"{ftype} | D0={D0} W={W} n={n}"

            elif mode_f == "MEAN":
                types = ["Arithmetic", "Geometric", "Harmonic", "Contraharmonic"]
                ftype = types[sub_mode_f % 4]
                result = apply_spatial_filter(gray, ftype, ksize, Q=Q)
                display_mode = f"{ftype} | k={ksize} Q={Q}"

            elif mode_f == "ORDER":
                types = ["Median", "Max", "Min", "Midpoint", "Alpha-trimmed"]
                ftype = types[sub_mode_f % 5]
                result = apply_spatial_filter(gray, ftype, ksize, d=d_param)
                display_mode = f"{ftype} | k={ksize} d={d_param}"

            else:
                result = gray.copy()
                display_mode = "NORMAL"

            sel_txt = selected_f if selected_f else "None"
            help_txt = (
    " 1=LPF 2=HPF 6=BR 7=BP 9=MEAN 0=ORDER | "
    "8=cycle | Select: d=D0 w=W n=n k=ksize p=Q t=d_param |"
)

        else:
            # ---- Apply EDGES ----
            if mode_e == "canny":
                blurred = noise_reduction(gray, ksize=ksize_e)
                Gx, Gy, mag, ang = compute_gradients(blurred)
                nms = non_maximum_suppression(mag, ang)
                strong, weak = double_threshold(nms, low_t, high_t)
                edges = hysteresis(strong, weak)
                result = edges
                display_mode = f"CANNY | k={ksize_e} low={low_t} high={high_t}"

            elif mode_e == "sobel":
                Gx, Gy, mag = sobel_alt(gray)
                result = mag
                display_mode = "SOBEL MAG"

            elif mode_e == "gx":
                Gx, _, _, _ = compute_gradients(gray)
                gx_norm = cv2.normalize(Gx, None, 0, 255, cv2.NORM_MINMAX)
                result = gx_norm.astype(np.uint8)
                display_mode = "SOBEL GX"

            elif mode_e == "gy":
                _, Gy, _, _ = compute_gradients(gray)
                gy_norm = cv2.normalize(Gy, None, 0, 255, cv2.NORM_MINMAX)
                result = gy_norm.astype(np.uint8)
                display_mode = "SOBEL GY"

            elif mode_e == "mag":
                _, _, mag, _ = compute_gradients(gray)
                result = np.uint8(np.clip(mag, 0, 255))
                display_mode = "SOBEL MAG (ALT)"

            else:
                result = gray.copy()
                display_mode = "NONE (GRAY)"

            sel_txt = (selected_e.upper() if selected_e else "NONE")
            help_txt = (
    " c=Canny s=Sobel x=Gx y=Gy m=Mag | "
    "Select: l=low h=high k=ksize | +/- change | "
    "TAB -> FREQ | ESC/q exit"
)


        # ==================== Display split ====================
        left = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        split = np.hstack((left, frame))

        cv2.putText(
            split,
            f"MAIN: {main_mode} | {display_mode} | Select: {sel_txt}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
        )
        cv2.putText(
            split,
            help_txt,
            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
        )

        cv2.imshow("Merged Lab: Frequency + Edges", split)

        key = cv2.waitKey(1) & 0xFF

        # ==================== Global keys ====================
        if key == 27 or key == ord('q'):
            break

        # TAB toggle
        if key == 9:  # TAB
            main_mode = "EDGES" if main_mode == "FREQ" else "FREQ"
            continue

        # ==================== Keys for current mode ====================
        if main_mode == "FREQ":
            # modes
            if key == ord('1'):
                mode_f, sub_mode_f = "LPF", 0
            elif key == ord('2'):
                mode_f, sub_mode_f = "HPF", 0
            elif key == ord('6'):
                mode_f, sub_mode_f = "BR", 0
            elif key == ord('7'):
                mode_f, sub_mode_f = "BP", 0
            elif key == ord('9'):
                mode_f, sub_mode_f = "MEAN", 0
            elif key == ord('0'):
                mode_f, sub_mode_f = "ORDER", 0

            # cycle types
            elif key == ord('8'):
                sub_mode_f += 1

            # select parameter (p for Q)
            elif key in (ord('d'), ord('D')):
                selected_f = "D0"
            elif key in (ord('w'), ord('W')):
                selected_f = "W"
            elif key in (ord('n'), ord('N')):
                selected_f = "N"
            elif key in (ord('k'), ord('K')):
                selected_f = "K"
            elif key in (ord('p'), ord('P')):
                selected_f = "Q"
            elif key in (ord('t'), ord('T')):
                selected_f = "T"

            # +/- change selected
            elif key in (ord('+'), ord('=')):
                if selected_f == "D0":
                    D0 += 5
                elif selected_f == "W":
                    W += 2
                elif selected_f == "N":
                    n += 1
                elif selected_f == "K":
                    ksize += 2
                elif selected_f == "Q":
                    Q += 0.5
                elif selected_f == "T":
                    d_param += 2

            elif key in (ord('-'), ord('_')):
                if selected_f == "D0":
                    D0 -= 5
                elif selected_f == "W":
                    W -= 2
                elif selected_f == "N":
                    n -= 1
                elif selected_f == "K":
                    ksize -= 2
                elif selected_f == "Q":
                    Q -= 0.5
                elif selected_f == "T":
                    d_param -= 2

        else:
            # edge modes
            if key == ord('c'):
                mode_e = "canny"
            elif key == ord('s'):
                mode_e = "sobel"
            elif key == ord('x'):
                mode_e = "gx"
            elif key == ord('y'):
                mode_e = "gy"
            elif key == ord('m'):
                mode_e = "mag"

            # select param
            elif key == ord('l'):
                selected_e = "low"
            elif key == ord('h'):
                selected_e = "high"
            elif key == ord('k'):
                selected_e = "k"

            # +/- update
            elif key in (ord('+'), ord('=')):
                if selected_e == "low":
                    low_t = min(254, low_t + step_t)
                    low_t = min(low_t, high_t - 1)
                elif selected_e == "high":
                    high_t = min(255, high_t + step_t)
                    high_t = max(high_t, low_t + 1)
                elif selected_e == "k":
                    if ksize_e < 31:
                        ksize_e += 2

            elif key in (ord('-'), ord('_')):
                if selected_e == "low":
                    low_t = max(0, low_t - step_t)
                    low_t = min(low_t, high_t - 1)
                elif selected_e == "high":
                    high_t = max(1, high_t - step_t)
                    high_t = max(high_t, low_t + 1)
                elif selected_e == "k":
                    if ksize_e > 3:
                        ksize_e -= 2

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()