"""Phase 1: Isolated 9:16 War Room Visual Asset Builder.

Transforms a 16:9 (1920x1080) Pierre Stock War Room capture into a seamless
5-second 1080x1920 (9:16) vertical tactical HUD background video for
MoneyPrinterTurbo without modifying any file outside pierre-training-video.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
SOURCE_IMAGE_PATH = ASSETS_DIR / "war_room_source.png"
COMPOSITE_IMAGE_PATH = ASSETS_DIR / "war_room_9x16_composite.png"

LOCAL_VIDEOS_DIR = PROJECT_ROOT / "video_studio" / "storage" / "local_videos"
ARCHIVE_VIDEOS_DIR = LOCAL_VIDEOS_DIR / "archive"
OUTPUT_VIDEO_PATH = LOCAL_VIDEOS_DIR / "war_room_9x16.mp4"

FONT_PATH = (
    PROJECT_ROOT
    / "video_studio"
    / "resource"
    / "fonts"
    / "BeVietnamPro-Bold.ttf"
)

WAR_ROOM_URL = "http://127.0.0.1:3888/pages/war-room/"
WAR_ROOM_HEALTH_URL = "http://127.0.0.1:3888/api/health"
READONLY_STAGE_VIDEO = Path(
    r"C:\Anti-Gravity-Core\pierre_stock_warroom\public\assets\julie_core_stage.mp4"
)

WINGET_FFMPEG = Path(
    r"C:\Users\Pierre\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-9.0-full_build\bin\ffmpeg.exe"
)
BROWSER_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
)

FOOTER_TEXT = (
    "AUTONOMOUS QUANT TELEMETRY // "
    "EDUCATIONAL & RESEARCH ONLY — NOT FINANCIAL ADVICE"
)


@dataclass(slots=True, frozen=True)
class CropBox:
    """Immutable pixel crop rectangle (left, top, right, bottom)."""

    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        if self.left < 0 or self.top < 0:
            raise ValueError(f"Crop origin must be non-negative: {self}")
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError(f"Invalid crop dimensions: {self}")

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    def clamp(self, max_w: int, max_h: int) -> CropBox:
        return CropBox(
            left=max(0, min(self.left, max_w - 1)),
            top=max(0, min(self.top, max_h - 1)),
            right=max(1, min(self.right, max_w)),
            bottom=max(1, min(self.bottom, max_h)),
        )


@dataclass(slots=True, frozen=True)
class PanelSlot:
    """Target slot rectangle on the 1080x1920 vertical canvas."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Panel slot must have positive size: {self}")


@dataclass(slots=True, frozen=True)
class WarRoomLayoutConfig:
    """Source 16:9 War Room crop geometry and vertical 9:16 slot layout."""

    viewport_width: int = 1920
    viewport_height: int = 1080

    # 1920x1080 War Room regions (pixel-verified against war_room_source.png)
    header_brand_crop: CropBox = CropBox(100, 18, 1820, 182)
    ticker_strip_crop: CropBox = CropBox(100, 184, 1408, 219)
    comms_terminal_crop: CropBox = CropBox(100, 238, 573, 619)
    julie_hud_crop: CropBox = CropBox(589, 238, 1331, 619)
    julie_stage_viewport_crop: CropBox = CropBox(745, 282, 1317, 599)
    radar_panel_crop: CropBox = CropBox(1347, 238, 1820, 619)
    sector_ribbon_crop: CropBox = CropBox(100, 631, 1820, 676)
    cluster_cards_crop: CropBox = CropBox(967, 708, 1820, 1018)

    # Target 1080x1920 vertical split-stack zones
    header_slot: PanelSlot = PanelSlot(0, 0, 1080, 110)
    upper_center_slot: PanelSlot = PanelSlot(0, 110, 1080, 880)
    lower_center_slot: PanelSlot = PanelSlot(0, 990, 1080, 810)
    footer_slot: PanelSlot = PanelSlot(0, 1800, 1080, 120)


@dataclass(slots=True, frozen=True)
class VideoRenderConfig:
    """Output FFmpeg render specification."""

    width: int = 1080
    height: int = 1920
    fps: int = 30
    duration_sec: float = 5.0
    total_frames: int = 150
    crf: int = 18
    codec: str = "libx264"
    pix_fmt: str = "yuv420p"
    colorspace: str = "bt709"
    bg_rgb: tuple[int, int, int] = (6, 11, 20)
    card_bg_rgb: tuple[int, int, int] = (8, 14, 26)
    border_cyan_rgb: tuple[int, int, int] = (0, 229, 255)
    border_dim_rgb: tuple[int, int, int] = (18, 56, 84)


def _isolated_env() -> dict[str, str]:
    """Return process environment that leaves GPU 0 (cuda:0) untouched."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    return env


def _resolve_ffmpeg() -> Path:
    if WINGET_FFMPEG.exists():
        return WINGET_FFMPEG
    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    raise FileNotFoundError(
        f"FFmpeg executable not found at {WINGET_FFMPEG} or on PATH."
    )


def _resolve_browser() -> Path:
    for candidate in BROWSER_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Neither Microsoft Edge nor Google Chrome was found for read-only snapshot capture."
    )


def _verify_war_room_live() -> None:
    req = urllib.request.Request(WAR_ROOM_HEALTH_URL, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"War Room health check returned HTTP {resp.status}"
                )
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"War Room server at {WAR_ROOM_HEALTH_URL} is unreachable: {exc}"
        ) from exc


def _composite_julie_stage_if_dark(
    img: Image.Image, layout: WarRoomLayoutConfig, ffmpeg_bin: Path
) -> Image.Image:
    """Overlay a single read-only frame of julie_core_stage.mp4 if headless video stayed dark."""
    stage_box = layout.julie_stage_viewport_crop.clamp(img.width, img.height)
    stage_patch = img.crop(stage_box.as_tuple())
    mean_luma = sum(ImageStat.Stat(stage_patch).mean) / 3.0
    if mean_luma >= 18.0 or not READONLY_STAGE_VIDEO.exists():
        return img

    cmd = [
        str(ffmpeg_bin),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        "00:00:01.00",
        "-i",
        str(READONLY_STAGE_VIDEO),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        check=True,
        env=_isolated_env(),
        timeout=20,
    )
    frame_img = Image.open(BytesIO(proc.stdout)).convert("RGB")
    fitted_frame = _cover_resize(frame_img, stage_box.width, stage_box.height)
    blended = ImageChops.lighter(stage_patch, fitted_frame)
    updated = img.copy()
    updated.paste(blended, (stage_box.left, stage_box.top))
    return updated


def acquire_war_room_source(
    layout: WarRoomLayoutConfig, ffmpeg_bin: Path
) -> Path:
    """Load existing war_room_source.png or capture a read-only headless snapshot."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if SOURCE_IMAGE_PATH.exists():
        with Image.open(SOURCE_IMAGE_PATH) as existing:
            if existing.size == (layout.viewport_width, layout.viewport_height):
                return SOURCE_IMAGE_PATH

    _verify_war_room_live()
    browser_exe = _resolve_browser()
    tmp_profile = ASSETS_DIR / ".tmp_headless_profile"
    if tmp_profile.exists():
        shutil.rmtree(tmp_profile, ignore_errors=True)
    tmp_profile.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(browser_exe),
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--hide-scrollbars",
        "--mute-audio",
        "--autoplay-policy=no-user-gesture-required",
        f"--user-data-dir={tmp_profile}",
        f"--window-size={layout.viewport_width},{layout.viewport_height}",
        "--force-device-scale-factor=1",
        "--virtual-time-budget=6500",
        f"--screenshot={SOURCE_IMAGE_PATH}",
        WAR_ROOM_URL,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=_isolated_env(),
            timeout=45,
        )
        if proc.returncode != 0 or not SOURCE_IMAGE_PATH.exists():
            raise RuntimeError(
                f"Headless capture failed (rc={proc.returncode}): {proc.stderr.strip()}"
            )
    finally:
        shutil.rmtree(tmp_profile, ignore_errors=True)

    with Image.open(SOURCE_IMAGE_PATH) as captured:
        rgb = captured.convert("RGB")
        enriched = _composite_julie_stage_if_dark(rgb, layout, ffmpeg_bin)
        enriched.save(SOURCE_IMAGE_PATH, format="PNG", optimize=True)

    return SOURCE_IMAGE_PATH


def _cover_resize(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Scale and center-crop without distortion."""
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    resized_w = max(target_w, round(src_w * scale))
    resized_h = max(target_h, round(src_h * scale))
    resized = img.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    left = (resized_w - target_w) // 2
    top = (resized_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def _contain_resize(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Proportionally fit an image inside (max_w, max_h) without cropping."""
    src_w, src_h = img.size
    scale = min(max_w / src_w, max_h / src_h)
    out_w = max(1, round(src_w * scale))
    out_h = max(1, round(src_h * scale))
    return img.resize((out_w, out_h), Image.Resampling.LANCZOS)


def _draw_panel_frame(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    cfg: VideoRenderConfig,
    title: str | None = None,
    font: ImageFont.FreeTypeFont | None = None,
) -> None:
    """Draw obsidian card container with cyan border highlight."""
    draw.rounded_rectangle(
        box,
        radius=12,
        fill=cfg.card_bg_rgb,
        outline=cfg.border_dim_rgb,
        width=2,
    )
    x0, y0, x1, _ = box
    draw.line([(x0 + 18, y0), (x1 - 18, y0)], fill=cfg.border_cyan_rgb, width=2)
    if title and font is not None:
        draw.text((x0 + 18, y0 + 10), title, font=font, fill=cfg.border_cyan_rgb)


def _fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    start_size: int = 22,
    min_size: int = 12,
) -> ImageFont.FreeTypeFont:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Required font not found: {FONT_PATH}")
    for size in range(start_size, min_size - 1, -1):
        font = ImageFont.truetype(str(FONT_PATH), size=size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            return font
    return ImageFont.truetype(str(FONT_PATH), size=min_size)


def _paste_centered(
    canvas: Image.Image, patch: Image.Image, area: tuple[int, int, int, int]
) -> None:
    x0, y0, x1, y1 = area
    px = x0 + max(0, ((x1 - x0) - patch.width) // 2)
    py = y0 + max(0, ((y1 - y0) - patch.height) // 2)
    canvas.paste(patch, (px, py))


def build_vertical_composite(
    source_path: Path,
    layout: WarRoomLayoutConfig,
    render_cfg: VideoRenderConfig,
) -> Path:
    """Compose the 1080x1920 9:16 split-stack War Room HUD composite image."""
    with Image.open(source_path) as raw_src:
        src = raw_src.convert("RGB")

    if src.size != (layout.viewport_width, layout.viewport_height):
        src = _cover_resize(src, layout.viewport_width, layout.viewport_height)

    canvas = Image.new(
        "RGB", (render_cfg.width, render_cfg.height), render_cfg.bg_rgb
    )
    draw = ImageDraw.Draw(canvas)

    # 1. TOP HEADER BAR (1080x110): Executive Brand + Metrics + Live Ticker Strip
    hdr = layout.header_slot
    draw.rectangle((0, hdr.y, hdr.width, hdr.y + hdr.height), fill=(11, 17, 30))
    brand_crop = src.crop(layout.header_brand_crop.as_tuple())
    ticker_crop = src.crop(layout.ticker_strip_crop.as_tuple())

    brand_h = 77
    ticker_h = hdr.height - brand_h - 3
    brand_scaled = _contain_resize(brand_crop, hdr.width - 16, brand_h)
    ticker_scaled = ticker_crop.resize(
        (hdr.width, ticker_h), Image.Resampling.LANCZOS
    )
    _paste_centered(
        canvas, brand_scaled, (8, hdr.y + 1, hdr.width - 8, hdr.y + brand_h)
    )
    draw.line(
        [(0, hdr.y + brand_h), (hdr.width, hdr.y + brand_h)],
        fill=render_cfg.border_dim_rgb,
        width=1,
    )
    canvas.paste(ticker_scaled, (0, hdr.y + brand_h + 1))
    draw.line(
        [(0, hdr.y + hdr.height - 2), (hdr.width, hdr.y + hdr.height - 2)],
        fill=render_cfg.border_cyan_rgb,
        width=2,
    )

    # 2. UPPER-CENTER PANEL (1080x880): Julie-Core HUD + Radar + Active Clusters + Sector Bar
    up = layout.upper_center_slot
    julie_crop = src.crop(layout.julie_hud_crop.as_tuple())
    radar_crop = src.crop(layout.radar_panel_crop.as_tuple())
    clusters_crop = src.crop(layout.cluster_cards_crop.as_tuple())
    sector_crop = src.crop(layout.sector_ribbon_crop.as_tuple())

    pad_x = 18
    inner_w = up.width - (2 * pad_x)
    julie_card = (pad_x, up.y + 12, up.width - pad_x, up.y + 560)
    _draw_panel_frame(draw, julie_card, render_cfg)
    julie_fitted = _contain_resize(julie_crop, inner_w - 8, 538)
    _paste_centered(
        canvas,
        julie_fitted,
        (julie_card[0] + 4, julie_card[1] + 5, julie_card[2] - 4, julie_card[3] - 5),
    )

    # Telemetry Sub-Deck inside remaining Upper-Center band (y=680..982):
    # Left: 8-Cluster Radar | Right: Live Cluster Cards | Bottom: Sector Navigator Strip
    sub_top = up.y + 570
    sub_bottom = up.y + up.height - 8
    sector_h = 34
    cards_bottom = sub_bottom - sector_h - 8

    radar_card = (pad_x, sub_top, pad_x + 344, cards_bottom)
    _draw_panel_frame(draw, radar_card, render_cfg)
    radar_fitted = _contain_resize(
        radar_crop, (radar_card[2] - radar_card[0]) - 8, (cards_bottom - sub_top) - 8
    )
    _paste_centered(
        canvas,
        radar_fitted,
        (radar_card[0] + 4, radar_card[1] + 4, radar_card[2] - 4, radar_card[3] - 4),
    )

    clusters_card = (pad_x + 356, sub_top, up.width - pad_x, cards_bottom)
    _draw_panel_frame(draw, clusters_card, render_cfg)
    clusters_fitted = _contain_resize(
        clusters_crop,
        (clusters_card[2] - clusters_card[0]) - 10,
        (cards_bottom - sub_top) - 10,
    )
    _paste_centered(
        canvas,
        clusters_fitted,
        (
            clusters_card[0] + 5,
            clusters_card[1] + 5,
            clusters_card[2] - 5,
            clusters_card[3] - 5,
        ),
    )

    sector_fitted = sector_crop.resize(
        (inner_w, sector_h), Image.Resampling.LANCZOS
    )
    canvas.paste(sector_fitted, (pad_x, cards_bottom + 8))

    # 3. LOWER-CENTER PANEL (1080x810): Live Comms & Recon Terminal (VAULT RECON)
    low = layout.lower_center_slot
    comms_crop = src.crop(layout.comms_terminal_crop.as_tuple())
    comms_card = (pad_x, low.y + 10, low.width - pad_x, low.y + low.height - 10)
    _draw_panel_frame(draw, comms_card, render_cfg)
    comms_fitted = _contain_resize(
        comms_crop, inner_w - 12, (comms_card[3] - comms_card[1]) - 12
    )
    _paste_centered(
        canvas,
        comms_fitted,
        (
            comms_card[0] + 6,
            comms_card[1] + 6,
            comms_card[2] - 6,
            comms_card[3] - 6,
        ),
    )

    # 4. COMPLIANCE FOOTER BAR (1080x120): Clean cyan/navy pill with BeVietnamPro-Bold
    ftr = layout.footer_slot
    pill_box = (24, ftr.y + 18, ftr.width - 24, ftr.y + ftr.height - 18)
    draw.rounded_rectangle(
        pill_box,
        radius=16,
        fill=(8, 22, 42),
        outline=render_cfg.border_cyan_rgb,
        width=2,
    )
    draw.rounded_rectangle(
        (pill_box[0] + 3, pill_box[1] + 3, pill_box[2] - 3, pill_box[3] - 3),
        radius=13,
        outline=(0, 120, 150),
        width=1,
    )
    footer_font = _fit_font(draw, FOOTER_TEXT, max_width=980, start_size=20)
    text_bbox = draw.textbbox((0, 0), FOOTER_TEXT, font=footer_font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    text_x = (ftr.width - text_w) // 2
    text_y = pill_box[1] + ((pill_box[3] - pill_box[1] - text_h) // 2) - text_bbox[1]
    draw.text((text_x, text_y), FOOTER_TEXT, font=footer_font, fill=(210, 245, 255))

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(COMPOSITE_IMAGE_PATH, format="PNG", optimize=True)
    return COMPOSITE_IMAGE_PATH


def archive_existing_local_videos() -> list[str]:
    """Move legacy test clips out of storage/local_videos into archive/."""
    LOCAL_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    archived: list[str] = []
    for entry in sorted(LOCAL_VIDEOS_DIR.iterdir()):
        if entry.is_file() and entry.name != OUTPUT_VIDEO_PATH.name:
            dest = ARCHIVE_VIDEOS_DIR / entry.name
            shutil.move(str(entry), str(dest))
            archived.append(entry.name)
    return archived


def encode_mp4_loop(
    composite_path: Path, cfg: VideoRenderConfig, ffmpeg_bin: Path
) -> Path:
    """Encode the 1080x1920 composite into a 5.0s H.264 bt709 silent MP4 loop."""
    LOCAL_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    temp_out = LOCAL_VIDEOS_DIR / f".{OUTPUT_VIDEO_PATH.stem}.tmp.mp4"
    if temp_out.exists():
        temp_out.unlink()

    cmd = [
        str(ffmpeg_bin),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-framerate",
        str(cfg.fps),
        "-i",
        str(composite_path),
        "-t",
        f"{cfg.duration_sec:.1f}",
        "-frames:v",
        str(cfg.total_frames),
        "-vf",
        (
            f"scale={cfg.width}:{cfg.height}:out_color_matrix={cfg.colorspace},"
            f"fps={cfg.fps}"
        ),
        "-c:v",
        cfg.codec,
        "-crf",
        str(cfg.crf),
        "-pix_fmt",
        cfg.pix_fmt,
        "-colorspace",
        cfg.colorspace,
        "-color_primaries",
        cfg.colorspace,
        "-color_trc",
        cfg.colorspace,
        "-an",
        "-movflags",
        "+faststart",
        str(temp_out),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=_isolated_env(),
        timeout=60,
    )
    if proc.returncode != 0 or not temp_out.exists():
        raise RuntimeError(
            f"FFmpeg encode failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    shutil.move(str(temp_out), str(OUTPUT_VIDEO_PATH))
    return OUTPUT_VIDEO_PATH


def main() -> int:
    layout = WarRoomLayoutConfig()
    render_cfg = VideoRenderConfig()

    ffmpeg_bin = _resolve_ffmpeg()
    source_path = acquire_war_room_source(layout, ffmpeg_bin)
    composite_path = build_vertical_composite(source_path, layout, render_cfg)
    archived = archive_existing_local_videos()
    output_path = encode_mp4_loop(composite_path, render_cfg, ffmpeg_bin)

    print(f"Source screenshot : {source_path}")
    print(f"Composite PNG     : {composite_path}")
    print(f"Archived clips    : {archived if archived else 'none'}")
    print(f"Output 9:16 MP4   : {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] build_warroom_bg failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
