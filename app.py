from flask import Flask, render_template, request, redirect, url_for, session
import os
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "development-secret-key")

# Klasör ayarları
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER


# =========================================================
# Helpers
# =========================================================
def safe_basename(path: str) -> str:
    return os.path.basename(path).replace("..", "")

def cv_imread(path: str):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Görüntü okunamadı.")
    return img

def ensure_gray_cv(img):
    return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def ensure_bgr_cv(img):
    return img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def normalize_to_uint8(x):
    x = np.asarray(x).astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - mn) * (255.0 / (mx - mn))
    return np.clip(x, 0, 255).astype(np.uint8)

def freeman_chain_code(contour):
    """8 yönlü Freeman chain code (basit)."""
    pts = contour.reshape(-1, 2)
    if len(pts) < 2:
        return []
    dirs = {
        (1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
        (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7
    }
    code = []
    for i in range(1, len(pts)):
        dx = int(np.sign(pts[i][0] - pts[i-1][0]))
        dy = int(np.sign(pts[i][1] - pts[i-1][1]))
        if (dx, dy) == (0, 0):
            continue
        code.append(dirs.get((dx, dy), None))
    return code


# =========================================================
# OPENCV KODLARI BURADA (1 işlem -> OpenCV çıktı)
# =========================================================
def apply_opencv(img_bgr, action: str):
    img = img_bgr.copy()

    # 1) GRİ
    if action == "grayscale":
        return ensure_gray_cv(img)

    # 2) PARLAKLIK
    if action == "brightness":
        return cv2.convertScaleAbs(img, alpha=1.0, beta=40)

    # 3) KONTRAST
    if action == "contrast":
        return cv2.convertScaleAbs(img, alpha=1.4, beta=0)

    # 4) KONTRAST GERME (STRETCHING)
    if action == "stretch":
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # 5) NEGATİF
    if action == "negative":
        return cv2.bitwise_not(img)

    # 6) AYNALAMA (YATAY)
    if action == "flip_h":
        return cv2.flip(img, 1)

    # 7) AYNALAMA (DİKEY)
    if action == "flip_v":
        return cv2.flip(img, 0)

    # 8) MEAN
    if action == "mean":
        return cv2.blur(img, (7, 7))

    # 9) GAUSSIAN
    if action == "gaussian":
        return cv2.GaussianBlur(img, (15, 15), 0)

    # 10) MEDYAN
    if action == "median":
        return cv2.medianBlur(img, 7)

    # 11) EŞİKLEME
    if action == "threshold":
        gray = ensure_gray_cv(img)
        _, out = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return out

    # 12) HİSTOGRAM EŞİTLEME
    if action == "histogram":
        gray = ensure_gray_cv(img)
        return cv2.equalizeHist(gray)

    # 13) SOBEL
    if action == "sobel":
        gray = ensure_gray_cv(img)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
        mag = cv2.magnitude(gx, gy)
        return normalize_to_uint8(mag)

    # 14) PREWITT
    if action == "prewitt":
        gray = ensure_gray_cv(img)
        kx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], np.float32)
        ky = np.array([[ 1, 1, 1],
                       [ 0, 0, 0],
                       [-1,-1,-1]], np.float32)
        gx = cv2.filter2D(gray, -1, kx)
        gy = cv2.filter2D(gray, -1, ky)
        return cv2.addWeighted(gx, 0.5, gy, 0.5, 0)

    # 15) LAPLACIAN
    if action == "laplacian":
        gray = ensure_gray_cv(img)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        out = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        return normalize_to_uint8(np.abs(out))

    # 16-19) MORFOLOJİ
    if action in ["erosion", "dilation", "opening", "closing"]:
        gray = ensure_gray_cv(img)
        _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        if action == "erosion":
            return cv2.erode(bw, kernel, iterations=1)
        if action == "dilation":
            return cv2.dilate(bw, kernel, iterations=1)
        if action == "opening":
            return cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        if action == "closing":
            return cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    # 20) DÖNDÜR
    if action == "rotate":
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 30, 1.0)
        return cv2.warpAffine(img, M, (w, h))

    # 21) ZOOM IN
    if action == "zoom_in":
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * 1.25), int(h * 1.25)))

    # 22) ZOOM OUT
    if action == "zoom_out":
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * 0.75), int(h * 0.75)))

    # 23) PERSPEKTİF
    if action == "perspective":
        h, w = img.shape[:2]
        src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        dst = np.float32([[40, 40], [w - 60, 30], [50, h - 50], [w - 40, h - 40]])
        P = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, P, (w, h))

    # 24) KONVOLÜSYON
    if action == "convolution":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], np.float32)
        return cv2.filter2D(img, -1, kernel)

    # 25) KORELASYON (ŞABLON EŞLEŞTİRME)
    if action == "correlation":
        gray = ensure_gray_cv(img)
        h, w = gray.shape
        th, tw = max(20, h // 6), max(20, w // 6)
        y0, x0 = h // 2 - th // 2, w // 2 - tw // 2
        template = gray[y0:y0 + th, x0:x0 + tw]
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        out = img.copy()
        cv2.rectangle(out, max_loc, (max_loc[0] + tw, max_loc[1] + th), (0, 255, 0), 3)
        cv2.putText(out, f"match={max_val:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return out

    # 26) CENTROID
    if action == "centroid":
        gray = ensure_gray_cv(img)
        _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = img.copy()
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(out, (cx, cy), 8, (0, 0, 255), -1)
                cv2.drawContours(out, [c], -1, (255, 0, 0), 2)
        return out

    return img


# =========================================================
# PYTHON / PILLOW KODLARI BURADA (1 işlem -> Pillow çıktı)
# =========================================================
def apply_python(pil_img: Image.Image, action: str) -> Image.Image:
    img = pil_img.copy()

    if action == "grayscale":
        return ImageOps.grayscale(img)

    if action == "brightness":
        return ImageEnhance.Brightness(img).enhance(1.3)

    if action == "contrast":
        return ImageEnhance.Contrast(img).enhance(1.6)

    if action == "stretch":
        return ImageOps.autocontrast(img)

    if action == "negative":
        if img.mode == "RGBA":
            img = img.convert("RGB")
        return ImageOps.invert(img)

    if action == "flip_h":
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    if action == "flip_v":
        return img.transpose(Image.FLIP_TOP_BOTTOM)

    if action == "mean":
        return img.filter(ImageFilter.BoxBlur(radius=3))

    if action == "gaussian":
        return img.filter(ImageFilter.GaussianBlur(radius=5))

    if action == "median":
        return img.filter(ImageFilter.MedianFilter(size=7))

    if action == "threshold":
        g = img.convert("L")
        return g.point(lambda p: 255 if p > 128 else 0)

    if action == "histogram":
        g = img.convert("L") if img.mode != "L" else img
        return ImageOps.equalize(g)

    if action in ["sobel", "prewitt", "laplacian"]:
        g = img.convert("L")
        return g.filter(ImageFilter.FIND_EDGES)

    if action == "rotate":
        return img.rotate(30, expand=False)

    if action == "zoom_in":
        w, h = img.size
        return img.resize((int(w * 1.25), int(h * 1.25)))

    if action == "zoom_out":
        w, h = img.size
        return img.resize((int(w * 0.75), int(h * 0.75)))

    # Pillow'da zor olanlar: OpenCV ile yapıp PIL'e çeviriyoruz (yine Python sonucu olarak kaydedilecek)
    if action in ["erosion", "dilation", "opening", "closing", "perspective",
                  "convolution", "correlation", "centroid"]:
        arr = np.array(img.convert("RGB"))
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        out = apply_opencv(bgr, action)
        if len(out.shape) == 2:
            return Image.fromarray(out)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    return img


# =========================================================
# Routes
# =========================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return redirect(url_for("index"))
    f = request.files["file"]
    if f.filename == "":
        return redirect(url_for("index"))

    filename = secure_filename(f.filename)
    
    if not filename:
        return redirect(url_for("index"))
    
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(save_path)
    session["uploaded_img_path"] = save_path
    return redirect(url_for("operations"))

@app.route("/operations")
def operations():
    if "uploaded_img_path" not in session:
        return redirect(url_for("index"))
    return render_template("operations.html")

@app.route("/process/<action>")
def process(action):
    img_path = session.get("uploaded_img_path")
    if not img_path:
        return redirect(url_for("index"))

    base = safe_basename(img_path)
    name_no_ext, _ = os.path.splitext(base)

    # Orijinal gösterim için path
    original_url = f"/{img_path}"

    # 1) OpenCV sonucu
    cv_in = cv_imread(img_path)
    cv_out = apply_opencv(cv_in, action)
    opencv_name = f"opencv_{action}_{name_no_ext}.png"
    opencv_path = os.path.join(app.config["PROCESSED_FOLDER"], opencv_name)
    cv2.imwrite(opencv_path, cv_out)

    # 2) Python/Pillow sonucu
    pil_in = Image.open(img_path)
    pil_out = apply_python(pil_in, action)
    python_name = f"python_{action}_{name_no_ext}.png"
    python_path = os.path.join(app.config["PROCESSED_FOLDER"], python_name)
    pil_out.save(python_path)

    return render_template(
        "result.html",
        action=action.upper(),
        original=original_url,
        opencv=f"/{opencv_path}",
        python=f"/{python_path}",
    )


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
        port=9000
    )
