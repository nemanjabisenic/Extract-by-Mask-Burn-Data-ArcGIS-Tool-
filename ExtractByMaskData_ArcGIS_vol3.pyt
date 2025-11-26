# -*- coding: utf-8 -*-
"""
ExtractByMaskData_Version6.pyt

Tool: Extract By Mask to GeoTIFF (Annotated, Auto Scale Bar, WGS84)

- Clips an input raster by polygon mask (like Extract By Mask).
- Projects the clipped raster to WGS 84 (EPSG:4326).
- Saves to GeoTIFF (.tif).
- Burns into the image pixels:
    * North arrow (bottom-right, on white background)
    * Scale bar in meters (automatic "nice" length based on pixel size)
    * Text block in the bottom-right with 3 stacked lines:
          <DATETIME>
          Lat: <lat>
          Lon: <lon>
      where lat/lon are the center of the (projected) raster in WGS-84.

Requires (ArcGIS Pro environment):
    - arcpy
    - arcpy.sa
    - Pillow (PIL)  – install via `pip install Pillow` into the Pro env if needed.
"""

import math
import os
import tempfile
import numpy as np
import arcpy
from arcpy.sa import ExtractByMask

# Try Pillow (PIL) for image annotation
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


# ----------------------------------------------------------------------
# Toolbox class
# ----------------------------------------------------------------------
class Toolbox(object):
    def __init__(self):
        self.label = "ExtractByMask_Annotated_Toolbox"
        self.alias = "ExtractByMaskAnnotated"
        self.tools = [ExtractByMaskToGeoTIFF_Annotated]


# ----------------------------------------------------------------------
# Tool class
# ----------------------------------------------------------------------
class ExtractByMaskToGeoTIFF_Annotated(object):
    def __init__(self):
        self.label = "Extract By Mask to GeoTIFF (Annotated, WGS84)"
        self.description = (
            "Clip a raster by polygon mask, project output to WGS 84, save as "
            "GeoTIFF, and burn in a north arrow, automatic scale bar in meters, "
            "and a bottom-right text block with datetime + center lat/lon."
        )
        self.canRunInBackground = True

    def getParameterInfo(self):
        params = []

        # 0 – Input raster
        p0 = arcpy.Parameter(
            displayName="Input Raster",
            name="in_raster",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"
        )

        # 1 – Mask feature(s)
        p1 = arcpy.Parameter(
            displayName="Mask Feature",
            name="in_mask",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )

        # 2 – Output GeoTIFF
        p2 = arcpy.Parameter(
            displayName="Output GeoTIFF (projected to WGS 84)",
            name="out_tif",
            datatype="DEFile",
            parameterType="Required",
            direction="Output"
        )

        # 3 – Datetime text
        p3 = arcpy.Parameter(
            displayName="Date / Time text",
            name="datetime_text",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        p3.value = ""

        # 4 – (Optional) Preferred scale-bar length in meters.
        p4 = arcpy.Parameter(
            displayName="Preferred Scale Bar Length (m) (optional – auto if empty)",
            name="scale_bar_m",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        p4.value = None

        return [p0, p1, p2, p3, p4]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        out_param = parameters[2]
        if out_param.valueAsText and not out_param.valueAsText.lower().endswith(".tif"):
            out_param.setWarningMessage("It is recommended to use a .tif extension.")
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        in_raster = parameters[0].valueAsText
        in_mask = parameters[1].valueAsText
        out_tif = parameters[2].valueAsText
        datetime_text = parameters[3].valueAsText or ""
        preferred_scale_m = parameters[4].value

        if preferred_scale_m in (None, "", 0):
            preferred_scale_m = None
        else:
            preferred_scale_m = int(preferred_scale_m)

        _run_extract_by_mask_annotated(
            in_raster,
            in_mask,
            out_tif,
            datetime_text=datetime_text,
            preferred_scale_m=preferred_scale_m,
            messages=messages
        )
        return


# ----------------------------------------------------------------------
# Core processing
# ----------------------------------------------------------------------
def _run_extract_by_mask_annotated(
    in_raster,
    in_mask,
    out_tif,
    datetime_text="",
    preferred_scale_m=None,
    messages=None
):
    if messages is None:
        messages = _DummyMessages()

    if Image is None:
        messages.addErrorMessage(
            "Pillow (PIL) is not available. Install it in the ArcGIS Pro Python "
            "environment (e.g., `pip install Pillow`) and try again."
        )
        raise RuntimeError("Pillow not available")

    arcpy.env.overwriteOutput = True
    messages.addMessage("Running Extract By Mask...")

    # 1) Clip raster by mask
    arcpy.CheckOutExtension("Spatial")
    clipped = ExtractByMask(in_raster, in_mask)

    # 2) Project clipped raster to WGS 84 (EPSG:4326)
    messages.addMessage("Projecting clipped raster to WGS 84 (EPSG:4326)...")
    sr_out = arcpy.SpatialReference(4326)
    # Save directly to the final output path
    arcpy.management.ProjectRaster(
        in_raster=clipped,
        out_raster=out_tif,
        out_coor_system=sr_out,
        resampling_type="NEAREST"
    )
    temp_tif = out_tif
    messages.addMessage(f"Projected raster saved to: {temp_tif}")

    georef_info = _capture_georeference(temp_tif, messages)

    # 3) Center point lat/lon in WGS-84 (now the native CRS)
    # Mask controls the chip extent, but we still derive the center from the
    # projected raster so the annotation reflects the exact output footprint.
    lat, lon = _get_center_latlon(temp_tif, messages)
    messages.addMessage(f"Center lat/lon (WGS-84): {lat:.6f}, {lon:.6f}")

    # 4) Determine pixel size in meters (WGS84 approximate)
    pixel_size_m = _estimate_pixel_size_meters(temp_tif, messages)
    messages.addMessage(
        f"Estimated pixel size (m): {pixel_size_m if pixel_size_m else 'unknown'}"
    )

    # 5) Annotate image (north arrow, scale bar, stacked text)
    _annotate_geotiff(
        temp_tif,
        datetime_text=datetime_text,
        center_lat=lat,
        center_lon=lon,
        pixel_size_m=pixel_size_m,
        preferred_scale_m=preferred_scale_m,
        georef_info=georef_info,
        messages=messages
    )

    messages.addMessage("Done.")
    return


# ----------------------------------------------------------------------
# Helper: messages
# ----------------------------------------------------------------------
class _DummyMessages(object):
    """Fallback if ArcPy messages object isn't provided."""
    def addMessage(self, msg):
        print(msg)

    def addWarningMessage(self, msg):
        print("WARNING:", msg)

    def addErrorMessage(self, msg):
        print("ERROR:", msg)


# ----------------------------------------------------------------------
# Helper: center lat/lon
# ----------------------------------------------------------------------
def _get_center_latlon(raster_path, messages):
    """Return center point of raster in WGS-84 (lat, lon)."""
    desc = arcpy.Describe(raster_path)
    ext = desc.extent
    sr = desc.spatialReference

    center_x = (ext.XMin + ext.XMax) / 2.0
    center_y = (ext.YMin + ext.YMax) / 2.0
    pt = arcpy.Point(center_x, center_y)
    geom = arcpy.PointGeometry(pt, sr)

    # If already geographic, just read lat/lon directly
    if sr.type == "Geographic":
        geo = geom
    else:
        geo = geom.projectAs(arcpy.SpatialReference(4326))

    lat = geo.centroid.Y
    lon = geo.centroid.X
    return (lat, lon)


def _capture_georeference(raster_path, messages):
    """Capture basic georeferencing info before Pillow overwrites metadata."""
    desc = arcpy.Describe(raster_path)
    ext = desc.extent

    cell_w = float(getattr(desc, "meanCellWidth", 0.0) or 0.0)
    cell_h = float(getattr(desc, "meanCellHeight", 0.0) or 0.0)
    if not cell_w or not cell_h:
        # Fall back to extent/size calculation
        cell_w = (ext.XMax - ext.XMin) / max(desc.width, 1)
        cell_h = (ext.YMax - ext.YMin) / max(desc.height, 1)

    info = {
        "spatial_reference": desc.spatialReference,
        "x_origin": ext.XMin,
        "y_origin": ext.YMin,
        "y_max": ext.YMax,
        "x_cellsize": cell_w,
        "y_cellsize": cell_h
    }

    messages.addMessage(
        "Captured georeferencing -> Origin: ({:.6f}, {:.6f}), Cellsize: ({:.6f}, {:.6f})".format(
            info["x_origin"], info["y_origin"], info["x_cellsize"], info["y_cellsize"]
        )
    )
    return info


def _reapply_georeference(tif_path, array_data, georef_info, messages):
    """Write annotated array back to disk and reattach georeferencing tags."""
    # Preserve the original northern edge by shifting the origin when rows grow.
    new_y_origin = georef_info["y_max"] - array_data.shape[0] * georef_info["y_cellsize"]
    lower_left = arcpy.Point(georef_info["x_origin"], new_y_origin)
    band_count = 1 if array_data.ndim == 2 else array_data.shape[2]

    temp_band_paths = []
    try:
        # NumPyArrayToRaster expects a 2D array; build per-band rasters
        bands_iterable = [array_data] if band_count == 1 else [array_data[:, :, i] for i in range(band_count)]

        for idx, band_arr in enumerate(bands_iterable):
            band_raster = arcpy.NumPyArrayToRaster(
                band_arr,
                lower_left,
                georef_info["x_cellsize"],
                georef_info["y_cellsize"]
            )

            temp_band = os.path.join(tempfile.gettempdir(), f"_annot_band_{idx}.tif")
            if arcpy.Exists(temp_band):
                arcpy.management.Delete(temp_band)

            band_raster.save(temp_band)
            temp_band_paths.append(temp_band)

        if len(temp_band_paths) == 1:
            arcpy.management.CopyRaster(temp_band_paths[0], tif_path, pixel_type="8_BIT_UNSIGNED")
        else:
            arcpy.management.CompositeBands(temp_band_paths, tif_path)

        arcpy.management.DefineProjection(tif_path, georef_info["spatial_reference"])
        messages.addMessage("Reapplied georeferencing to annotated raster.")
    except Exception as ex:
        messages.addWarningMessage(f"Unable to reapply georeferencing: {ex}")
    finally:
        for path in temp_band_paths:
            if arcpy.Exists(path):
                arcpy.management.Delete(path)


# ----------------------------------------------------------------------
# Helper: pixel size in meters
# ----------------------------------------------------------------------
def _estimate_pixel_size_meters(raster_path, messages):
    """
    Try to estimate pixel size in meters.
    Works best if raster is in a projected coordinate system with meter units,
    but here we approximate from WGS84 degrees via haversine distance.
    """
    desc = arcpy.Describe(raster_path)
    sr = desc.spatialReference
    cell_w = float(getattr(desc, "meanCellWidth", 0.0) or 0.0)

    unit_name = (sr.linearUnitName or "").lower() if sr else ""
    if "metre" in unit_name or "meter" in unit_name or unit_name == "m":
        return abs(cell_w)

    try:
        ext = desc.extent
        cx = (ext.XMin + ext.XMax) / 2.0
        cy = (ext.YMin + ext.YMax) / 2.0
        w = cell_w or (ext.XMax - ext.XMin) / max(desc.width, 1)

        pt1 = arcpy.PointGeometry(arcpy.Point(cx, cy), sr)
        pt2 = arcpy.PointGeometry(arcpy.Point(cx + w, cy), sr)

        pt1_wgs = pt1.projectAs(arcpy.SpatialReference(4326))
        pt2_wgs = pt2.projectAs(arcpy.SpatialReference(4326))

        d_m = _haversine(
            pt1_wgs.centroid.X, pt1_wgs.centroid.Y,
            pt2_wgs.centroid.X, pt2_wgs.centroid.Y
        )
        return abs(d_m)
    except Exception as ex:
        messages.addWarningMessage(
            "Could not estimate pixel size in meters: {}".format(ex)
        )
        return None


def _haversine(lon1, lat1, lon2, lat2):
    """Return great-circle distance in meters between two WGS-84 points."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ----------------------------------------------------------------------
# Helper: safe text measurement
# ----------------------------------------------------------------------
def _textsize(draw, text, font):
    """Safe text measurement for multiple Pillow versions."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        # Older Pillow
        return draw.textsize(text, font=font)


# ----------------------------------------------------------------------
# Helper: annotation
# ----------------------------------------------------------------------
def _annotate_geotiff(
    tif_path,
    datetime_text,
    center_lat,
    center_lon,
    pixel_size_m,
    preferred_scale_m,
    georef_info,
    messages
):
    # Open image and force RGB for safe drawing
    base_img = Image.open(tif_path)
    if base_img.mode not in ("RGB", "RGBA"):
        base_img = base_img.convert("RGB")

    w, h = base_img.size

    # Slightly smaller fonts
    font_small, font_medium = _get_fonts()

    # Prepare text lines
    lines = []
    dt_text = datetime_text.strip()
    if not dt_text:
        dt_text = "Date/Time: N/A"
    lines.append(dt_text)
    lines.append("Lat: {:.5f}".format(center_lat))
    lines.append("Lon: {:.5f}".format(center_lon))

    # Measurements
    max_text_w = 0
    total_text_h = 0
    for line in lines:
        t_w, t_h = _textsize(ImageDraw.Draw(base_img), line, font_small)
        max_text_w = max(max_text_w, t_w)
        total_text_h += t_h

    # Layout constants
    band_pad_x = 12
    band_pad_y = 10
    line_gap = 6  # small gap between text elements

    arrow_height = 28
    arrow_half_width = 10

    # Arrow + text block sizing
    n_w, n_h = _textsize(ImageDraw.Draw(base_img), "N", font_small)
    text_block_h = total_text_h + line_gap * (len(lines) - 1)
    arrow_only_h = n_h + 2 + arrow_height
    text_block_w = max_text_w + band_pad_x * 2

    # Scale bar sizing (if applicable)
    scale_bar_block_h = 0
    scale_bar_dims = None
    if pixel_size_m:
        scale_len_m = _choose_scale_length(pixel_size_m, w, preferred_scale_m)
        scale_px = scale_len_m / pixel_size_m

        bar_height = 8
        label_text = "{} m".format(int(scale_len_m))
        label_w, label_h = _textsize(ImageDraw.Draw(base_img), label_text, font_small)

        bar_x_left = band_pad_x
        bar_x_right = int(bar_x_left + scale_px)
        if bar_x_right > w - band_pad_x:
            bar_x_right = w - band_pad_x
            scale_len_m = (bar_x_right - bar_x_left) * pixel_size_m
            scale_px = bar_x_right - bar_x_left
            label_text = "{} m".format(int(scale_len_m))
            label_w, label_h = _textsize(ImageDraw.Draw(base_img), label_text, font_small)

        scale_bar_block_h = label_h + 6 + bar_height
        scale_bar_dims = {
            "bar_x_left": bar_x_left,
            "bar_x_right": bar_x_right,
            "bar_height": bar_height,
            "label_text": label_text,
            "label_w": label_w,
            "label_h": label_h
        }
    else:
        messages.addWarningMessage(
            "Pixel size in meters unknown – scale bar omitted."
        )

    gap_after_scale = 10
    band_height = max(
        text_block_h,
        scale_bar_block_h + gap_after_scale + arrow_only_h,
        arrow_only_h
    ) + band_pad_y * 2

    # Build new canvas with white band underneath the chip
    new_img = Image.new("RGB", (w, h + band_height), (255, 255, 255))
    new_img.paste(base_img, (0, 0))
    draw = ImageDraw.Draw(new_img)

    band_top = h

    # ------------------------------------------------------------------
    # 1) Scale bar on left of the white band
    # ------------------------------------------------------------------
    if scale_bar_dims:
        bar_y = band_top + band_pad_y + scale_bar_dims["label_h"] + 4 + scale_bar_dims["bar_height"] // 2
        patch_x1 = scale_bar_dims["bar_x_left"] - 6
        patch_y1 = band_top + band_pad_y
        patch_x2 = max(scale_bar_dims["bar_x_right"], scale_bar_dims["bar_x_left"] + scale_bar_dims["label_w"]) + 6
        patch_y2 = bar_y + scale_bar_dims["bar_height"] // 2 + 6

        draw.rectangle([patch_x1, patch_y1, patch_x2, patch_y2], fill=(255, 255, 255))
        draw.rectangle(
            [scale_bar_dims["bar_x_left"], bar_y - scale_bar_dims["bar_height"] // 2,
             scale_bar_dims["bar_x_right"], bar_y + scale_bar_dims["bar_height"] // 2],
            fill=(0, 0, 0)
        )
        draw.rectangle(
            [scale_bar_dims["bar_x_left"], bar_y - scale_bar_dims["bar_height"] // 2,
             scale_bar_dims["bar_x_right"], bar_y + scale_bar_dims["bar_height"] // 2],
            outline=(0, 0, 0), width=1
        )
        label_x = scale_bar_dims["bar_x_left"] + (scale_bar_dims["bar_x_right"] - scale_bar_dims["bar_x_left"]) / 2.0 - scale_bar_dims["label_w"] / 2.0
        label_y = band_top + band_pad_y
        draw.text((label_x, label_y), scale_bar_dims["label_text"], font=font_small, fill=(0, 0, 0))

    # ------------------------------------------------------------------
    # 2) North arrow on bottom-left of the white band
    # ------------------------------------------------------------------
    arrow_bottom_y = band_top + band_height - band_pad_y
    arrow_top_y = arrow_bottom_y - arrow_height
    n_y = arrow_top_y - n_h - 2
    if scale_bar_dims:
        min_n_y = band_top + band_pad_y + scale_bar_block_h + gap_after_scale
        if n_y < min_n_y:
            shift = min_n_y - n_y
            arrow_bottom_y += shift
            arrow_top_y += shift
            n_y += shift

    arrow_center_x = band_pad_x + arrow_half_width + 6
    tip = (arrow_center_x, arrow_top_y)
    left = (arrow_center_x - arrow_half_width, arrow_bottom_y)
    right = (arrow_center_x + arrow_half_width, arrow_bottom_y)
    draw.polygon([tip, left, right], fill=(0, 0, 0))

    n_x = arrow_center_x - n_w / 2.0
    draw.text((n_x, n_y), "N", font=font_small, fill=(0, 0, 0))

    # 3) Stacked text on the right of the white band
    # ------------------------------------------------------------------
    block_x2 = w - band_pad_x
    block_x1 = block_x2 - text_block_w
    text_y = band_top + band_height - band_pad_y - text_block_h
    for idx, line in enumerate(lines):
        t_w, t_h = _textsize(draw, line, font_small)
        text_x = block_x1 + band_pad_x
        draw.text((text_x, text_y), line, font=font_small, fill=(0, 0, 0))
        if idx < len(lines) - 1:
            text_y += t_h + line_gap
        else:
            text_y += t_h

    # Save annotated TIFF (overwrite original tif_path), then restore georeference
    annotated_array = np.array(new_img)
    _reapply_georeference(tif_path, annotated_array, georef_info, messages)


def _get_fonts():
    """Return (small_font, medium_font) with safe fallbacks, slightly smaller."""
    try:
        # Slightly smaller sizes now
        small = ImageFont.truetype("arial.ttf", 11)
        medium = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        small = ImageFont.load_default()
        medium = ImageFont.load_default()

    return small, medium


def _choose_scale_length(pixel_size_m, image_width_px, preferred_scale_m):
    """
    Choose a "nice" scale-bar length in meters.

    The goal is to have the bar somewhere between ~80 and ~250 pixels
    on the final image. User's preferred value is honoured if reasonable,
    otherwise a default from a nice sequence is used.
    """
    min_px = 80.0
    max_px = 250.0

    if preferred_scale_m:
        px = preferred_scale_m / pixel_size_m
        if min_px <= px <= max_px:
            return float(preferred_scale_m)

    # If preferred value is missing or not suitable, pick from nice candidates
    candidates = [
        10, 20, 50,
        100, 200, 500,
        1000, 2000, 5000,
        10000, 20000, 50000
    ]

    best = candidates[0]
    best_px = best / pixel_size_m
    for c in candidates:
        px = c / pixel_size_m
        if min_px <= px <= max_px:
            best = c
            best_px = px
            break
        # Otherwise keep the one whose pixel length is closest to mid-range
        mid_target = (min_px + max_px) / 2.0
        if abs(px - mid_target) < abs(best_px - mid_target):
            best = c
            best_px = px

    return float(best)
