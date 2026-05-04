#!/usr/bin/env python3
"""
Portrait processing pipeline:
1. Remove background (rembg)
2. Convert to grayscale
3. Normalize brightness
4. Crop to consistent face size
5. Save as PNG with alpha channel
"""

from rembg import remove
from PIL import Image
import cv2
import numpy as np
import os

def process(src_path, dst_path, target_face_pct=0.52, target_brightness=148, sharpen=False, gamma=None):
    """
    Process portrait: remove bg, convert to grayscale, normalize brightness,
    crop to consistent face size, and save as PNG with alpha.
    """
    print(f"\n{'='*70}")
    print(f"Processing: {os.path.basename(src_path)}")
    print(f"Target face %: {target_face_pct*100:.0f}%, Target brightness: {target_brightness}")
    print(f"{'='*70}")

    # Step 1: Remove background
    print("  [1/6] Removing background...")
    src_pil = Image.open(src_path).convert('RGBA')
    print(f"       Input size: {src_pil.size}")
    bg_removed = remove(src_pil)

    # Convert to numpy
    rgba = np.array(bg_removed, dtype=np.uint8)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]

    print(f"       After bg removal: {rgba.shape}")

    # Step 2: Detect face
    print("  [2/6] Detecting face...")
    gray_for_detect = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = cascade.detectMultiScale(gray_for_detect, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        faces = cascade.detectMultiScale(gray_for_detect, 1.05, 3, minSize=(40, 40))

    if len(faces) == 0:
        print(f"       WARNING: No face detected!")
        return None

    fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
    print(f"       Face detected: x={fx}, y={fy}, w={fw}, h={fh}")

    h, w = rgba.shape[:2]

    # Step 3: Compute crop size for target face %
    print("  [3/6] Computing crop dimensions...")
    crop_size = int(fh / target_face_pct)
    print(f"       Crop size: {crop_size}x{crop_size}")

    # Step 4: Upscale if needed
    if crop_size > min(h, w):
        print(f"  [4/6] Upscaling source image...")
        scale = (crop_size / min(h, w)) * 1.15
        new_w = int(w * scale)
        new_h = int(h * scale)
        print(f"       Scale factor: {scale:.3f} -> {new_w}x{new_h}")

        rgba = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        rgb = rgba[..., :3]
        alpha = rgba[..., 3]

        fx = int(fx * scale)
        fy = int(fy * scale)
        fw = int(fw * scale)
        fh = int(fh * scale)
        h, w = rgba.shape[:2]
        crop_size = int(fh / target_face_pct)
    else:
        print(f"  [4/6] No upscaling needed")

    # Step 5: Compute crop position
    print("  [5/6] Computing crop position...")
    eye_y = fy + fh * 0.40
    crop_top = int(eye_y - 0.40 * crop_size)
    face_cx = fx + fw / 2
    crop_left = int(face_cx - crop_size / 2)
    print(f"       Crop position: top={crop_top}, left={crop_left}")

    # Handle padding
    pt = max(0, -crop_top)
    pb = max(0, crop_top + crop_size - h)
    pl = max(0, -crop_left)
    pr = max(0, crop_left + crop_size - w)

    if any([pt, pb, pl, pr]):
        print(f"       Adding padding: top={pt}, bottom={pb}, left={pl}, right={pr}")
        rgba = cv2.copyMakeBorder(rgba, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
        crop_top += pt
        crop_left += pl

    # Final bounds check
    crop_top = max(0, min(crop_top, rgba.shape[0] - crop_size))
    crop_left = max(0, min(crop_left, rgba.shape[1] - crop_size))

    cropped = rgba[crop_top:crop_top+crop_size, crop_left:crop_left+crop_size]
    print(f"       Cropped to: {cropped.shape}")

    rgb_crop = cropped[..., :3].copy()
    alpha_crop = cropped[..., 3].copy()

    # Step 6: Sharpen (optional)
    if sharpen:
        print(f"  [6/6] Applying sharpening + gamma correction...")
        blurred = cv2.GaussianBlur(rgb_crop, (0, 0), 2.5)
        rgb_crop = cv2.addWeighted(rgb_crop, 1.6, blurred, -0.6, 0)
    else:
        print(f"  [6/6] Converting to grayscale...")

    # Step 7: Grayscale
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)

    # Step 8: Gamma correction
    if gamma:
        print(f"       Applying gamma {gamma}...")
        inv = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype('uint8')
        gray = cv2.LUT(gray, table)

    # Step 9: Brightness normalization
    print(f"       Normalizing brightness to {target_brightness}...")
    visible_mask = alpha_crop > 30
    visible_pixels = gray[visible_mask]
    current_mean = visible_pixels.mean() if len(visible_pixels) > 0 else 128
    beta = target_brightness - current_mean
    print(f"       Current mean: {current_mean:.1f}, Delta: {beta:+.1f}")

    normalized = cv2.convertScaleAbs(gray, alpha=1.05, beta=beta)

    # Step 10: Resize to 900x900
    print(f"       Resizing to 900x900...")
    rgb_final = cv2.resize(normalized, (900, 900), interpolation=cv2.INTER_LANCZOS4)
    alpha_final = cv2.resize(alpha_crop, (900, 900), interpolation=cv2.INTER_LANCZOS4)

    # Step 11: Build RGBA and save
    print(f"       Saving as PNG with transparency...")
    rgba_out = np.dstack([rgb_final, rgb_final, rgb_final, alpha_final])
    Image.fromarray(rgba_out, 'RGBA').save(dst_path, 'PNG', optimize=True)

    # Verify results
    alpha_resized = cv2.resize(alpha_crop, (900, 900), interpolation=cv2.INTER_LANCZOS4)
    visible_mask_final = alpha_resized > 30
    visible_pixels_final = rgb_final[visible_mask_final]
    mean_brightness_final = visible_pixels_final.mean() if len(visible_pixels_final) > 0 else 0

    face_pct_final = (fh / crop_size) * 100

    result = {
        'final_dims': rgba_out.shape,
        'mean_brightness_visible': mean_brightness_final,
        'face_pct': face_pct_final,
        'file': os.path.basename(dst_path)
    }

    print(f"\n  RESULTS:")
    print(f"    Final dimensions: {result['final_dims']}")
    print(f"    Mean brightness (visible px): {result['mean_brightness_visible']:.1f}")
    print(f"    Face % in frame: {result['face_pct']:.1f}%")
    print(f"    Saved to: {dst_path}")

    return result

if __name__ == '__main__':
    base = '/sessions/awesome-intelligent-ramanujan/mnt/WWW/SDR/img'

    print("\n" + "="*70)
    print("PORTRAIT PHOTO BATCH PROCESSING")
    print("="*70)

    results = {}
    results['cho'] = process(
        f'{base}/cho.png',
        f'{base}/cho-bw.png',
        target_face_pct=0.63,
        target_brightness=148
    )

    results['leo'] = process(
        f'{base}/Leo.png',
        f'{base}/leo-bw.png',
        target_face_pct=0.52,
        target_brightness=150
    )

    results['finn'] = process(
        f'{base}/finn.png',
        f'{base}/finn-bw.png',
        target_face_pct=0.52,
        target_brightness=180,
        sharpen=True,
        gamma=0.75
    )

    # Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    for name, result in results.items():
        if result:
            print(f"\n{name.upper()}:")
            print(f"  File: {result['file']}")
            print(f"  Dimensions: {result['final_dims'][0]}x{result['final_dims'][1]}")
            print(f"  Mean brightness: {result['mean_brightness_visible']:.1f}")
            print(f"  Face %: {result['face_pct']:.1f}%")

    print("\n" + "="*70)
    print("All files saved as PNG with transparent backgrounds (alpha channel)")
    print("="*70)
