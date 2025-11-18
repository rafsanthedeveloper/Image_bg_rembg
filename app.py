import os
import logging
import io
import random
from io import BytesIO

from flask import Flask, request, jsonify, send_from_directory, url_for
from pypdf import PdfReader
from PIL import Image, ImageFilter
import numpy as np
from rembg import remove  # LOCAL AI MODEL, NO EXTERNAL API

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "uploads"
EXTRACTED_FOLDER = "images"
ALLOWED_EXTENSIONS = {"pdf"}
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB limit

# Flask JSON settings (pretty print + Unicode + no slash escaping)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True
app.config["JSON_AS_ASCII"] = False

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACTED_FOLDER, exist_ok=True)


def generate_random_number(length=30):
    """Generate a random number string of given length."""
    return "".join([str(random.randint(0, 9)) for _ in range(length)])


def extract_images_from_pdf(pdf_file_path: str, output_path: str):
    """
    Extract images from a PDF and rename first two images as
    user-img-<30digits> and sign-img-<30digits>.
    """
    try:
        reader = PdfReader(pdf_file_path)
        seen_images = set()
        extracted_files = []
        image_count = 0

        for page in reader.pages:
            for image in page.images:
                image_data = image.data
                image_hash = hash(image_data)

                if image_hash in seen_images:
                    continue

                seen_images.add(image_hash)
                ext = os.path.splitext(image.name)[1].lower()

                # Convert JP2/JPEG2000 to PNG
                if ext in [".jp2", ".jpx"]:
                    try:
                        with Image.open(io.BytesIO(image_data)) as img:
                            if img.mode in ("RGBA", "P"):
                                img = img.convert("RGB")
                            image_bytes = io.BytesIO()
                            img.save(image_bytes, format="PNG")
                            image_data = image_bytes.getvalue()
                            ext = ".png"
                    except Exception as e:
                        logging.error(f"Failed to convert JP2 to PNG: {e}", exc_info=True)
                        continue

                random_number = generate_random_number(30)
                if image_count == 0:
                    image_filename = f"user-img-{random_number}{ext}"
                elif image_count == 1:
                    image_filename = f"sign-img-{random_number}{ext}"
                else:
                    image_filename = f"{random_number}{ext}"

                file_path = os.path.join(output_path, image_filename)
                with open(file_path, "wb") as fp:
                    fp.write(image_data)

                extracted_files.append(image_filename)
                image_count += 1

        return extracted_files

    except Exception as e:
        logging.error(f"Failed to extract images from {pdf_file_path}: {e}", exc_info=True)
        return []


# ---------- JSON wrapper ----------

def make_response(data: dict, status=200):
    if "Website" in data:
        data.pop("Website")
    data["Developer"] = "Rafsan The Developer"
    data["Website"] = "rafsanjane.com"
    return jsonify(data), status


# ---------- Helpers ----------

def is_transparent_png_bytes(img_data: bytes) -> bool:
    """Check if image bytes are already a transparent PNG."""
    try:
        with Image.open(BytesIO(img_data)) as img:
            if img.format != "PNG":
                return False
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                return True
        return False
    except Exception as e:
        logging.error(f"Error checking PNG transparency: {e}", exc_info=True)
        return False


# ---------- User-photo background removal (AI + fallback) ----------

def remove_background_ai(img_data: bytes) -> bytes | None:
    """
    Use rembg (local AI) to remove background.
    Returns PNG bytes with alpha, or None on failure.
    """
    try:
        result = remove(img_data)  # rembg runs locally, no HTTP
        # Ensure it’s a proper RGBA PNG
        with Image.open(BytesIO(result)) as img:
            img = img.convert("RGBA")
            out = BytesIO()
            img.save(out, format="PNG")
            return out.getvalue()
    except Exception as e:
        logging.error(f"rembg background removal failed: {e}", exc_info=True)
        return None


def remove_background_lightweight(
    img_data: bytes,
    sample_size: int = 12,
    dist_thresh: int = 15,
    blur_radius: int = 1,
    max_width: int = 1600,
) -> bytes | None:
    """
    Simple math-based remover (used as fallback).
    """
    try:
        img = Image.open(BytesIO(img_data)).convert("RGBA")
        orig_w, orig_h = img.size
        scale = 1.0
        if orig_w > max_width:
            scale = max_width / orig_w
            small = img.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
        else:
            small = img.copy()

        arr = np.array(small).astype(np.float32)
        h, w = arr.shape[:2]

        def sample_corner(x0, y0):
            x1 = min(w, x0 + sample_size)
            y1 = min(h, y0 + sample_size)
            return arr[y0:y1, x0:x1, :3].reshape(-1, 3)

        patches = [
            sample_corner(0, 0),
            sample_corner(max(0, w - sample_size), 0),
            sample_corner(0, max(0, h - sample_size)),
            sample_corner(max(0, w - sample_size), max(0, h - sample_size)),
        ]
        bg_color = np.vstack(patches).mean(axis=0)  # [R,G,B]

        rgb = arr[..., :3]
        dist = np.linalg.norm(rgb - bg_color.reshape(1, 1, 3), axis=2)

        max_dist = max(dist.max(), 1.0)
        alpha = (dist - dist_thresh) / (max_dist - dist_thresh)
        alpha = np.clip(alpha, 0.0, 1.0)

        mask = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
        if blur_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        if scale != 1.0:
            mask = mask.resize((orig_w, orig_h), Image.LANCZOS)

        out_img = img.copy()
        out_img.putalpha(mask)

        out = BytesIO()
        out_img.save(out, format="PNG")
        return out.getvalue()
    except Exception as e:
        logging.error(f"Lightweight BG removal failed: {e}", exc_info=True)
        return None


def remove_background_smart(img_data: bytes) -> bytes:
    """
    Combined strategy for USER IMAGE:
    1) If already transparent PNG -> return as-is
    2) Try rembg (AI)
    3) If rembg fails -> fallback to lightweight method
    """
    if is_transparent_png_bytes(img_data):
        return img_data

    ai_result = remove_background_ai(img_data)
    if ai_result:
        return ai_result

    lw_result = remove_background_lightweight(img_data)
    if lw_result:
        return lw_result

    return img_data


# ---------- Signature-specific processor (border color + crop) ----------

def process_signature_image(
    img_data: bytes,
    threshold: int = 180,   # 0 = pure black, 255 = pure white
    padding: int = 5,
    blur_radius: int = 0    # you can set 1–2 for softer edges
) -> bytes:
    """
    Signature BG remover for dark signature on light background.

    - Converts to grayscale
    - Keeps only pixels darker than `threshold`
    - Crops tightly around the dark pixels
    - Builds RGBA PNG with transparent background
    """
    try:
        # 1) Open and convert to grayscale
        img = Image.open(BytesIO(img_data)).convert("L")
        arr = np.array(img)

        # 2) Mask: dark pixels are signature
        mask = arr < threshold

        if not mask.any():
            # nothing dark enough found; fallback to original image
            return img_data

        # 3) Find bounding box of the signature
        ys, xs = np.where(mask)
        y_min = max(int(ys.min()) - padding, 0)
        y_max = min(int(ys.max()) + padding, arr.shape[0] - 1)
        x_min = max(int(xs.min()) - padding, 0)
        x_max = min(int(xs.max()) + padding, arr.shape[1] - 1)

        cropped_gray = arr[y_min:y_max + 1, x_min:x_max + 1]
        cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]

        h, w = cropped_gray.shape

        # 4) Build RGBA image
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = cropped_gray        # R
        rgba[..., 1] = cropped_gray        # G
        rgba[..., 2] = cropped_gray        # B
        rgba[..., 3] = np.where(cropped_mask, 255, 0).astype(np.uint8)  # A

        alpha_img = Image.fromarray(rgba[..., 3], mode="L")
        if blur_radius > 0:
            alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            rgba[..., 3] = np.array(alpha_img, dtype=np.uint8)

        out_img = Image.fromarray(rgba, mode="RGBA")
        buf = BytesIO()
        out_img.save(buf, format="PNG")
        return buf.getvalue()

    except Exception as e:
        logging.error(f"Signature dark-on-light processing failed: {e}", exc_info=True)
        return img_data



# ---------- Routes ----------

@app.route("/")
def home():
    return make_response({"status": "Images Extractor Active"})


# Support both /images (local) AND /extract_image (for your live endpoint)
@app.route("/images", methods=["POST"])
@app.route("/extract_image", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return make_response({"error": "No file part"}, 400)

    file = request.files["file"]
    if file.filename == "":
        return make_response({"error": "No selected file"}, 400)

    # size check
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size > MAX_FILE_SIZE:
        return make_response({"error": "File size exceeds 2 MB limit"}, 400)

    # type check
    if file.filename.split(".")[-1].lower() not in ALLOWED_EXTENSIONS:
        return make_response({"error": "Invalid file type"}, 400)

    # temp save
    pdf_tmp = os.path.join(UPLOAD_FOLDER, f"{generate_random_number(30)}.pdf")
    file.save(pdf_tmp)

    extracted_images = extract_images_from_pdf(pdf_tmp, EXTRACTED_FOLDER)

    # cleanup
    try:
        os.remove(pdf_tmp)
    except Exception as e:
        logging.error(f"Failed to delete PDF: {e}", exc_info=True)

    if not extracted_images:
        return make_response({"message": "No images found in the PDF"})

    images_dict = {}

    # --- USER IMAGE (index 0): original + BG-removed ---
    if len(extracted_images) >= 1:
        user_fname = extracted_images[0]
        user_path = os.path.join(EXTRACTED_FOLDER, user_fname)
        images_dict["user-image"] = url_for(
            "download_file", filename=user_fname, _external=True
        )

        try:
            with open(user_path, "rb") as f:
                user_in = f.read()
            user_out = remove_background_smart(user_in)
            user_bg_fname = (
                f"user-img-bg-remove-{generate_random_number(30)}.png"
            )
            with open(
                os.path.join(EXTRACTED_FOLDER, user_bg_fname), "wb"
            ) as out:
                out.write(user_out)
            images_dict["user-image-bg-remove"] = url_for(
                "download_file", filename=user_bg_fname, _external=True
            )
        except Exception as e:
            logging.error(f"User image BG removal failed: {e}", exc_info=True)

    # --- SIGNATURE IMAGE (index 1): original + signature-specific BG remove ---
    if len(extracted_images) >= 2:
        sign_fname = extracted_images[1]
        sign_path = os.path.join(EXTRACTED_FOLDER, sign_fname)
        images_dict["sign-image"] = url_for(
            "download_file", filename=sign_fname, _external=True
        )

        try:
            with open(sign_path, "rb") as f:
                sign_in = f.read()
            sign_out = process_signature_image(sign_in)
            sign_bg_fname = (
                f"sign-img-bg-remove-{generate_random_number(30)}.png"
            )
            with open(
                os.path.join(EXTRACTED_FOLDER, sign_bg_fname), "wb"
            ) as out:
                out.write(sign_out)
            images_dict["sign_bg_remove"] = url_for(
                "download_file", filename=sign_bg_fname, _external=True
            )
        except Exception as e:
            logging.error(f"Signature BG removal failed: {e}", exc_info=True)

    return make_response(
        {
            "message": "Images extracted successfully",
            "ExtractedImages": str(len(extracted_images)),
            "images": images_dict,
        }
    )


@app.route("/images/<filename>")
def download_file(filename):
    """Serve extracted images."""
    return send_from_directory(EXTRACTED_FOLDER, filename)


# For cPanel / Passenger if you use it
application = app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
