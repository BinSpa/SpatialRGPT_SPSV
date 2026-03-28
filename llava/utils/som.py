import os
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# 可选：RLE 解码（COCO 格式）
try:
    from pycocotools import mask as mask_utils
except Exception:
    mask_utils = None


# -----------------------------
# 1) 解析工具：兼容字段是 dict 或 str（很多 HF 数据会把 dict 存成字符串）
# -----------------------------
def ensure_py_obj(x: Any) -> Any:
    """如果 x 是 dict/list 直接返回；如果是字符串（如 "{'a':1}"）尝试 ast.literal_eval。"""
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return ast.literal_eval(s)
            except Exception:
                return x
    return x


def get_image_path(ds, i: int, base_path: Union[str, Path]) -> Path:
    # """base_path + ds[i]['image_info']['file_path']"""
    image_name = ds[i]["id"]+'.png'
    image_path = os.path.join(base_path, 'images/'+image_name)
    return image_path, image_name


# -----------------------------
# 2) bbox 解析：支持 xyxy / xywh（自动猜测，也可强制）
# -----------------------------
def to_xyxy(b: List[float], fmt: str = "auto") -> Tuple[float, float, float, float]:
    """
    b: [x1,y1,x2,y2] 或 [x,y,w,h]
    fmt: 'auto' | 'xyxy' | 'xywh'
    """
    if len(b) != 4:
        raise ValueError(f"bbox 长度不是 4：{b}")

    x0, y0, x1, y1 = map(float, b)

    if fmt == "xyxy":
        return x0, y0, x1, y1
    if fmt == "xywh":
        return x0, y0, x0 + x1, y0 + y1

    # auto: 简单启发式：如果第三四个比前两个大很多，可能是 x2,y2；否则当作 w,h
    if (x1 > x0) and (y1 > y0):
        # 可能是 xyxy，也可能是 xywh(但 w,h 通常不会 > x,y 必然成立)
        # 进一步：如果 x1,y1 看起来像坐标（很大），更偏 xyxy
        return x0, y0, x1, y1
    else:
        return x0, y0, x0 + x1, y0 + y1


# -----------------------------
# 3) RLE 解码：支持 COCO RLE dict 或 list-of-dict
# -----------------------------
def decode_rle_to_mask(rle_obj: Any) -> Optional[np.ndarray]:
    """
    返回 HxW 的 uint8 mask（0/1），失败返回 None。
    需要 pycocotools。
    """
    if mask_utils is None:
        return None

    rle_obj = ensure_py_obj(rle_obj)

    # 常见情况：
    # - 单个 rle dict: {"size":[H,W], "counts": "..."}
    # - list of rle dict
    if isinstance(rle_obj, dict):
        rle = rle_obj
        # pycocotools 需要 counts 是 bytes 或 RLE 编码字符串；通常直接可用
        m = mask_utils.decode(rle)  # (H,W,1) or (H,W)
        if m.ndim == 3:
            m = m[:, :, 0]
        return (m > 0).astype(np.uint8)

    if isinstance(rle_obj, list) and len(rle_obj) > 0 and isinstance(rle_obj[0], dict):
        # 多个 mask：decode 后可能是 (H,W,N)
        m = mask_utils.decode(rle_obj)
        if m.ndim == 2:
            return (m > 0).astype(np.uint8)
        if m.ndim == 3:
            # 返回每个实例一张 mask：这里不合并，交给外面逐个取
            # 直接返回 None 表示请外层用 decode 多实例逻辑
            return None

    return None


def decode_multi_rles(rle_list: Any) -> List[np.ndarray]:
    """
    rle_list: list[dict] 或 stringified list
    返回 list of HxW mask
    """
    if mask_utils is None:
        return []

    rle_list = ensure_py_obj(rle_list)
    if not isinstance(rle_list, list) or len(rle_list) == 0:
        return []

    # 如果是 list of dict
    if isinstance(rle_list[0], dict):
        m = mask_utils.decode(rle_list)  # (H,W,N) 或 (H,W)
        if m.ndim == 2:
            return [(m > 0).astype(np.uint8)]
        if m.ndim == 3:
            return [(m[:, :, k] > 0).astype(np.uint8) for k in range(m.shape[2])]

    # 如果每个元素是 dict 字符串
    masks = []
    for r in rle_list:
        rr = ensure_py_obj(r)
        if isinstance(rr, dict):
            mm = mask_utils.decode(rr)
            if mm.ndim == 3:
                mm = mm[:, :, 0]
            masks.append((mm > 0).astype(np.uint8))
    return masks


# -----------------------------
# 4) SoM 绘制：bbox + mask + 编号
# -----------------------------
def _get_default_font(size: int = 18) -> ImageFont.FreeTypeFont:
    # 尽量用系统字体，失败就用 PIL 默认
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", size=size)
        except Exception:
            return ImageFont.load_default()

def _pick_contrast_color(img_np, mask, candidate_colors=None):
    """
    根据 mask 周围局部背景，选择对比度最高的轮廓颜色
    img_np: HxWx3, uint8
    mask:   HxW, 0/1
    return: (R, G, B)
    """
    import cv2
    import numpy as np

    if candidate_colors is None:
        candidate_colors = [
            (255, 0, 0),      # red
            (0, 255, 0),      # green
            (0, 0, 255),      # blue
            (255, 255, 0),    # yellow
            (255, 0, 255),    # magenta
            (0, 255, 255),    # cyan
            # (255, 255, 255),  # white
            # (0, 0, 0),        # black
        ]

    mask_u8 = (mask > 0).astype(np.uint8)

    # 取 mask 外围一圈邻域，作为“周围环境”
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    ring = (dilated > 0) & (mask_u8 == 0)

    # 如果周围像素太少，就退化成看整个 mask 附近
    if ring.sum() < 10:
        ring = mask_u8 > 0

    bg_pixels = img_np[ring]
    if len(bg_pixels) == 0:
        return (255, 0, 0)

    bg_mean = bg_pixels.mean(axis=0)  # RGB 均值

    def score(c):
        c = np.array(c, dtype=np.float32)
        # RGB 欧氏距离 + 亮度差
        rgb_dist = np.linalg.norm(c - bg_mean)

        bg_luma = 0.299 * bg_mean[0] + 0.587 * bg_mean[1] + 0.114 * bg_mean[2]
        c_luma = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        luma_dist = abs(c_luma - bg_luma)

        return rgb_dist + 0.8 * luma_dist

    best_color = max(candidate_colors, key=score)
    return best_color

def _remove_small_mask_regions(mask, min_area=100):
    """
    删除二值 mask 中面积小于 min_area 的小连通块
    mask: HxW, 0/1 或 uint8
    return: 清理后的 0/1 mask
    """
    import cv2
    import numpy as np

    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned = np.zeros_like(mask)
    for lab in range(1, num_labels):  # 0 是背景
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == lab] = 1

    return cleaned

def draw_onlysom_on_image(
    image: Image.Image,
    bboxes=None,       
    masks=None,
    bbox_format="auto",
    alpha=0.35,        
    line_width=2,
) -> Image.Image:
    """
    SoM 风格：
    - 先把 mask 转成 contour polygon
    - 只画轮廓
    - 编号放在 mask 中位数位置；小目标放到边上
    """
    import cv2
    from PIL import ImageDraw

    img = image.convert("RGB")
    W, H = img.size
    img_np = np.array(img).copy()
    draw = ImageDraw.Draw(img)
    font = _get_default_font(size=max(18, int(min(W, H) * 0.03)))

    masks = masks or []

    for idx, m in enumerate(masks):
        if m is None:
            continue
        if m.shape[:2] != (H, W):
            continue

        mask = (m > 0).astype(np.uint8)
        if mask.sum() == 0:
            continue

        # 先去除小噪声块
        mask = _remove_small_mask_regions(mask, min_area=100)

        if mask.sum() == 0:
            continue

        # 再向外膨胀，给轮廓预留距离
        contour_offset = max(4, line_width + 2)
        kernel_size = contour_offset * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_expand = cv2.dilate(mask, kernel, iterations=1)

        mask_c = np.ascontiguousarray(mask_expand)
        res = cv2.findContours(mask_c, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contours = res[-2]
        hierarchy = res[-1]

        if hierarchy is None or len(contours) == 0:
            continue

        color = _pick_contrast_color(img_np, mask)

        cv2.polylines(
            img_np,
            contours,
            isClosed=True,
            color=color,
            thickness=line_width,
            lineType=cv2.LINE_AA,
        )

        # -------- SoM 风格标签位置：mask 中位数；小目标放侧边 --------
        ys, xs = np.nonzero(mask)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()

        text_x, text_y = np.median(np.stack([xs, ys], axis=1), axis=0)
        text_x, text_y = int(text_x), int(text_y)

        instance_area = (x1 - x0) * (y1 - y0)
        if instance_area < 1000 or (y1 - y0) < 40:
            if y1 >= H - 5:
                text_x, text_y = int(x1), int(y0)
            else:
                text_x, text_y = int(x0), int(y1)

        # 先把 contour 画回 PIL，再画文字
        img = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img)

        label = 'Region' + str(idx)
        l, t, r, b = draw.textbbox((0, 0), label, font=font)
        tw, th = r - l, b - t
        pad = 4
        gap = 6  # label 与物体边界的间隔

        # 当前 mask 的外接框
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()

        # 候选位置：上、右、下、左（都在物体外）
        candidates = [
            (x0, y0 - th - 2 * pad - gap),          # 上
            (x1 + gap, y0),                         # 右
            (x0, y1 + gap),                         # 下
            (x0 - tw - 2 * pad - gap, y0),         # 左
        ]

        def box_valid(tx, ty, tw, th, pad, mask, W, H):
            bx0, by0 = int(tx), int(ty)
            bx1 = int(tx + tw + 2 * pad)
            by1 = int(ty + th + 2 * pad)

            # 先检查是否出界
            if bx0 < 0 or by0 < 0 or bx1 > W or by1 > H:
                return False

            # 再检查是否和 mask 相交
            region = mask[by0:by1, bx0:bx1]
            if region.size > 0 and np.any(region > 0):
                return False

            return True

        tx, ty = None, None
        for cand_x, cand_y in candidates:
            if box_valid(cand_x, cand_y, tw, th, pad, mask, W, H):
                tx, ty = int(cand_x), int(cand_y)
                break

        # 如果四个方向都不行，就退化到右下角附近，并裁剪到图像内
        if tx is None:
            tx = min(max(0, x1 + gap), W - tw - 2 * pad)
            ty = min(max(0, y1 + gap), H - th - 2 * pad)

        # 直接写文字，不画底框
        draw.text(
            (tx, ty),
            label,
            font=font,
            fill=(255, 0, 0),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

        img_np = np.array(img)

    return Image.fromarray(img_np)
    

def draw_som_on_image(
    image: Image.Image,
    bboxes: Optional[List[List[float]]] = None,
    masks: Optional[List[np.ndarray]] = None,
    bbox_format: str = "auto",
    alpha: float = 0.35,
    line_width: int = 3,
) -> Image.Image:
    """
    返回标注后的 PIL.Image（RGB）
    - bboxes: list of [x1,y1,x2,y2] 或 [x,y,w,h]
    - masks:  list of HxW uint8(0/1)
    """
    img = image.convert("RGB")
    W, H = img.size

    # 做一层 RGBA 叠加，用于半透明 mask 填充
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(img)
    font = _get_default_font(size=max(14, int(min(W, H) * 0.03)))

    bboxes = bboxes or []
    masks = masks or []

    # 如果同时有 mask 与 bbox：用 mask 为主，bbox 可选叠加
    num_regions = max(len(bboxes), len(masks))

    for idx in range(num_regions):
        # 给每个 region 一个“稳定但不指定具体颜色”的方案会更优雅
        # 但你没要求配色，这里用简单的循环颜色
        # (R,G,B,A)
        color = (
            int(50 + (idx * 70) % 205),
            int(80 + (idx * 90) % 175),
            int(60 + (idx * 110) % 195),
            int(255 * alpha),
        )
        edge = (color[0], color[1], color[2], 255)

        # 1) mask
        if idx < len(masks) and masks[idx] is not None:
            m = masks[idx]
            if m.shape[0] != H or m.shape[1] != W:
                # 尺寸不匹配：跳过或 resize（这里选择跳过并提示）
                # 你也可以改成用最近邻 resize 对齐
                # print(f"[warn] mask shape {m.shape} != image {(H,W)}, skip mask {idx}")
                pass
            else:
                ys, xs = np.where(m > 0)
                if len(xs) > 0:
                    # 半透明填充：逐点画太慢，这里用像素方式做 overlay
                    # mask_img = Image.fromarray((m * 255).astype(np.uint8), mode="L")
                    # solid = Image.new("RGBA", (W, H), edge)
                    # overlay.paste(solid, (0, 0), mask_img)

                    # 轮廓：用 bbox 近似轮廓（更快）；要精确轮廓需要找轮廓算法
                    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                    draw.rectangle([x1, y1, x2, y2], outline=edge, width=line_width)

                    # label 放到左上角
                    label_xy = (int(x1), int(y1))
                else:
                    label_xy = (10, 10)

        # 2) bbox（没有 mask 或者你也想叠加 bbox）
        if idx < len(bboxes) and bboxes[idx] is not None:
            x1, y1, x2, y2 = to_xyxy(bboxes[idx], fmt=bbox_format)
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W - 1, x2))
            y2 = max(0, min(H - 1, y2))
            draw.rectangle([x1, y1, x2, y2], outline=edge, width=line_width)
            label_xy = (int(x1), int(y1))

        # 3) 编号标签（Region [idx]）
        label = f"{idx}"
        # 画一个不透明底框提高可读性
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        pad = 2
        bx0, by0 = label_xy[0], label_xy[1]
        bx1, by1 = bx0 + tw + 2 * pad, by0 + th + 2 * pad
        draw.rectangle([bx0, by0, bx1, by1], fill=(0, 0, 0))
        draw.text((bx0 + pad, by0 + pad), label, font=font, fill=(255, 255, 255))

    # 合成 overlay
    img_rgba = img.convert("RGBA")
    img_rgba = Image.alpha_composite(img_rgba, overlay)
    return img_rgba.convert("RGB")


# -----------------------------
# 5) 主函数：从 ds 取 bbox/rle，并生成 SoM 图
# -----------------------------
def get_regions_from_ds(ds, i: int) -> Tuple[List[List[float]], List[np.ndarray]]:
    """
    你说 region 信息来自 ds['bbox'] 和 ds['rle']：
    - ds[i]['bbox'] 可能是 list[list[4]] 或 stringified
    - ds[i]['rle']  可能是 list[dict] 或 dict 或 stringified
    """
    b = ensure_py_obj(ds[i].get("bbox", []))
    r = ensure_py_obj(ds[i].get("rle", []))

    # bboxes: list of 4-float
    bboxes: List[List[float]] = []
    if isinstance(b, list):
        # 可能是 [[...],[...]] 或 单个 [...]
        if len(b) == 4 and all(isinstance(x, (int, float)) for x in b):
            bboxes = [list(map(float, b))]
        else:
            # 多 bbox
            for bb in b:
                bb2 = ensure_py_obj(bb)
                if isinstance(bb2, (list, tuple)) and len(bb2) == 4:
                    bboxes.append(list(map(float, bb2)))

    # masks: list of HxW uint8
    masks: List[np.ndarray] = []
    if mask_utils is not None:
        # r 可能是单个 dict / list[dict]
        if isinstance(r, dict):
            m = decode_rle_to_mask(r)
            if m is not None:
                masks = [m]
        elif isinstance(r, list):
            masks = decode_multi_rles(r)

    return bboxes, masks


def som_visualize_one(
    ds,
    i: int,
    base_path: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    bbox_format: str = "auto",
    prefer_mask: bool = True,
    show: bool = True,
) -> Image.Image:
    img_path, image_name = get_image_path(ds, i, base_path)
    image = Image.open(img_path).convert("RGB")

    bboxes, masks = get_regions_from_ds(ds, i)

    # 如果 prefer_mask=True 且 mask 可用，就用 mask；否则只画 bbox
    if prefer_mask and len(masks) > 0:
        som_img = draw_onlysom_on_image(image, masks=masks)
    else:
        som_img = draw_som_on_image(image, bboxes=bboxes, masks=[], bbox_format=bbox_format)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / image_name
        # som_img.save(save_path, quality=95)
        som_img.save(save_path, format="PNG", optimize=True)
        print(f"[saved] {save_path}")

    if show:
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(som_img)
        axes[1].set_title(f"SoM annotated sample #{i}")
        axes[1].axis('off')
        
        plt.tight_layout()

    return som_img