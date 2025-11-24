# -*- coding: utf-8 -*-

# Module 1_Centerline (GRASS/SHP) — SCS Toolbox
# v12 
# Ported from arcPY to QGIS Processing framework by Martin Dawson using ChatGPT November 2025
# Based on: Standalone channel shifting toolbox (SCS Toolbox)
# Author: Milos Rusnak
#   CNRS - UMR5600 Environnement Ville Societe
#   15 Parvis Rene Descartes, BP 7000, 69342 Lyon Cedex 07, France 
               
# geogmilo@savba.sk
#   Institute of geography SAS
#   Stefanikova 49, 814 73 Bratislava, Slovakia 
               
# Standalone channel shifting toolbox (SCS Toolbox) was developed as extension of the FluvialCorridor toolbox with implemented the centerline 
# extraction approach and segmentation of DGO from FluvialCorridor toolbox.
# For each use of the Channel toolbox leading to a publication, report, presentation or any other
# document, please refer also to the following articles:
#       Rusnák, M., Opravil, Š., Dunesme, S., Afzali, H., Rey, L., Parmentier, H., Piégay, H., 2025 A channel shifting GIS toolbox for exploring
#       floodplain dynamics through channel erosion and deposition. Geomorphology, 477, 109688. 
#       https://doi.org/10.1016/j.geomorph.2025.109688 

#       Roux, C., Alber, A., Bertrand, M., Vaudor, L., Piegay, H., 2015. "FluvialCorridor": A new ArcGIS 
#       package for multiscale riverscape exploration. Geomorphology, 242, 29-37.
#       https://doi.org/10.1016/j.geomorph.2014.04.018
#
# QGIS Version(tested using 3.40.11-Bratislava)
# - Generates centerlines from channel polygons using GRASS v.voronoi.skeleton
#   (with a QGIS Voronoi-clip fallback if GRASS fails).
# - Two working modes:
#   1) PER-YEAR MODE: per-date centerlines (centro_YYYY0101.gpkg)
#   2) UNION_MODE: a single “Channel Zone” SegCenterline plus union layers
# - YEAR_FIELD can be:
#       * integer (e.g. 1949)
#       * date/datetime (uses .year or leading 4 digits)
#       * string containing a 4-digit year (18xx–20xx)
# - Inputs are reprojected to TARGET_CRS only when necessary.
# - GRASS region parameter can be:
#       * set explicitly in the dialog, or
#       * auto-derived from polygon extent in _centerline_one
# - Final outputs are GPKG layers with MULTILINESTRING centerlines.

from qgis.core import (
    QgsProcessingAlgorithm, QgsProcessing, QgsProcessingException,
    QgsProcessingParameterFolderDestination, QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterString, QgsProcessingParameterBoolean,
    QgsProcessingParameterNumber, QgsProcessingParameterExtent,
    QgsProcessingParameterCrs, QgsProcessingOutputString,
    QgsVectorLayer, QgsWkbTypes, QgsApplication, QgsProject, QgsProcessingUtils
)
from qgis.PyQt.QtCore import QCoreApplication
import processing
import shutil
import glob
import os
import tempfile
import uuid
import re
from datetime import date

# ---------------- helpers ---------------- #

# Global list tracking temporary directories created by this module.
# These are cleaned up at the end of the algorithm run.
_TRACKED_DIRS = []  # temp workspace tracker


def tr(s):
    """
    Convenience wrapper for translation of UI strings.
    """
    return QCoreApplication.translate("SCS_Centerlines", s)


def _temp_path_tracked(stem="tmp_", ext=".shp"):
    """
    Create a path to a temporary file in a single tracked temp directory.

    - On first call, create a new temp directory and remember it in _TRACKED_DIRS.
    - On subsequent calls, reuse the same temp directory.
    - Return a unique filename using stem + UUID + ext.
    """
    if not _TRACKED_DIRS:
        # Create the temp workspace once per run
        tmpdir = tempfile.mkdtemp(prefix="scs_mod1_")
        _TRACKED_DIRS.append(tmpdir)
    else:
        tmpdir = _TRACKED_DIRS[0]
    return os.path.join(tmpdir, f"{stem}{uuid.uuid4().hex}{ext}")


def _cleanup_tmp_dirs(feedback=None):
    """
    Remove all tracked temporary directories.

    Any errors while cleaning up are reported to the feedback object,
    but do not raise exceptions.
    """
    for path in list(_TRACKED_DIRS):
        try:
            shutil.rmtree(path, ignore_errors=True)
            if feedback:
                feedback.pushInfo(f"[cleanup] removed {path}")
        except Exception as e:
            if feedback:
                feedback.reportError(f"[cleanup] failed to remove {path}: {e}")
        finally:
            # Make sure the path is removed from the tracker even if rmtree fails
            try:
                _TRACKED_DIRS.remove(path)
            except Exception:
                pass


def as_layer(x, name_hint="layer"):
    """
    Resolve a QgsVectorLayer from:
      - an existing QgsVectorLayer object,
      - a layer ID / map layer string,
      - or a file path.

    Raises QgsProcessingException if x cannot be resolved or is invalid.
    """
    # Case 1: already a QgsVectorLayer object
    if isinstance(x, QgsVectorLayer):
        if not x.isValid():
            raise QgsProcessingException(f"{name_hint} is invalid")
        return x

    # Case 2: string that could be a map-layer reference
    if isinstance(x, str):
        # Try interpreting as a layer ID / layer name in the current project
        try:
            lyr = QgsProcessingUtils.mapLayerFromString(x, QgsProject.instance())
            if lyr and lyr.isValid():
                return lyr
        except Exception:
            pass

    # Case 3: treat string as a data source path (OGR)
    if isinstance(x, str):
        lyr = QgsVectorLayer(x, name_hint, "ogr")
        if not lyr.isValid():
            raise QgsProcessingException(f"Failed to load {name_hint}: {x}")
        return lyr

    # Anything else is unsupported
    raise QgsProcessingException(f"Unexpected type for {name_hint}: {type(x)}")


def _resolve_field_name(layer_or_path, desired_name):
    """
    Case-insensitive lookup of desired_name in layer fields.

    Returns:
        The actual field name (with original casing) if found, else None.
    """
    lyr = as_layer(layer_or_path, "field_resolve")
    want = (desired_name or "").strip().lower()
    for f in lyr.fields():
        nm = f.name()
        if nm.strip().lower() == want:
            return nm
    return None


def runp(alg_id, params, context=None, feedback=None, tag="", expect_key="OUTPUT"):
    """
    Wrapper around processing.run that:

      - Logs algorithm ID, tag, and parameters to feedback (if provided).
      - Returns a "best guess" output path from the result dictionary,
        preferring the given expect_key (default 'OUTPUT').

    This reduces boilerplate throughout the script.
    """
    if feedback:
        try:
            pretty = "; ".join(f"{k}={v}" for k, v in params.items())
            feedback.pushInfo(f"▶ {alg_id}{' ['+tag+']' if tag else ''}: {pretty}")
        except Exception:
            feedback.pushInfo(f"▶ {alg_id}{' ['+tag+']' if tag else ''}")
    res = processing.run(alg_id, params, context=context, feedback=feedback)

    # First try the expected key if this is a dict result
    if isinstance(res, dict) and expect_key in res:
        return res[expect_key]

    # Otherwise, try a set of common output keys
    if isinstance(res, dict):
        for k in ("OUTPUT", "output", "OUTPUT_LAYER", "OUTPUTFILE", "OUTPUT_PATH", "FILE_PATH"):
            if k in res:
                return res[k]
        # If there is only one value, return that
        if len(res) == 1:
            return next(iter(res.values()))

    # Non-dict result: return as-is
    return res


def freeze_to_shp(input_ref, context=None, feedback=None, tag="freeze"):
    """
    Force any layer/id/URI to a fresh, file-backed ESRI Shapefile.

    This avoids issues with in-memory layers and provides a stable on-disk
    representation for subsequent algorithms (especially GRASS).
    """
    tmp_shp = _temp_path_tracked("fz_", ".shp")
    runp("native:savefeatures", {"INPUT": input_ref, "OUTPUT": tmp_shp},
         context, feedback, tag)
    return tmp_shp


def save_gpkg_clean(layer_or_id, gpkg_path, context=None, feedback=None,
                    layer_name=None, force_multiline=False):
    """
    Final GPKG writer.

    Workflow:
      1) Freeze the input to SHP (file-backed).
      2) Remove typical FID-like ID fields (fid, ogc_fid, id, gid, pk).
      3) Use gdal:convertformat to convert SHP -> GPKG.

    Parameters:
      - layer_or_id: input layer or ID to save.
      - gpkg_path: full path to the output GeoPackage.
      - layer_name: name of the sublayer inside the GeoPackage.
      - force_multiline: if True, force geometry type to MULTILINESTRING
                         (used for centerlines).

    Returns:
      The output gpkg_path.
    """
    # Ensure parent folder exists
    parent = os.path.dirname(gpkg_path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    # Freeze to a SHP
    tmp_shp = freeze_to_shp(layer_or_id, context, feedback, tag="freeze_before_final")
    src_layer = as_layer(tmp_shp, "final_for_droptest")

    # Identify fields to drop (common FID/ID fields)
    drop = [f.name() for f in src_layer.fields()
            if f.name().lower() in {"fid", "ogc_fid", "id", "gid", "pk"}]

    shp_for_convert = tmp_shp
    if drop:
        # Create a copy with unwanted ID fields removed
        shp_for_convert = _temp_path_tracked("final_drop_", ".shp")
        runp("native:deletecolumn",
             {"INPUT": tmp_shp, "COLUMN": drop, "OUTPUT": shp_for_convert},
             context, feedback, "drop_id")

    # Remove any existing GPKG (gdal:convertformat overwrites the file, not layers)
    if os.path.exists(gpkg_path):
        try:
            os.remove(gpkg_path)
        except Exception:
            # Ignore failure: gdal:convertformat may still overwrite it
            pass

    # Define a unique layer name if none is provided
    unique_layer = layer_name or f"layer_{uuid.uuid4().hex[:8]}"

    # Optionally force MULTILINESTRING geometry
    opts = "-nlt MULTILINESTRING" if force_multiline else ""

    # Convert SHP to GPKG
    runp("gdal:convertformat",
         {"INPUT": shp_for_convert,
          "OPTIONS": opts,
          "OUTPUT": gpkg_path,
          "LAYER_NAME": unique_layer},
         context, feedback, "shp_to_gpkg")
    return gpkg_path


def _copy_shp_tree(src_shp_path, dst_shp_path):
    """
    Copy a SHP “tree” (.shp, .shx, .dbf, .prj, etc.) from src to dst.

    SHP files consist of multiple sidecar files with the same basename
    and different extensions; this copies all of them.
    """
    src_base, _ = os.path.splitext(src_shp_path)
    dst_base, _ = os.path.splitext(dst_shp_path)
    folder = os.path.dirname(src_shp_path)
    stem = os.path.basename(src_base)
    for extpath in glob.glob(os.path.join(folder, stem + ".*")):
        ext = os.path.splitext(extpath)[1]
        shutil.copy2(extpath, dst_base + ext)


def _normalize_region(region):
    """
    Normalize a GRASS region string.

    Accepts:
      - None or empty → returns None
      - Region strings (optionally containing extra text like '[...]').

    Returns:
      A cleaned region string (without any trailing bracketed metadata).
    """
    if not region:
        return None
    s = str(region)
    # Discard anything after '[' if present (QGIS may append extra info)
    if "[" in s:
        s = s.split("[", 1)[0].strip()
    return s


def _safe_years(layer_or_path, year_field):
    """
    Scan a layer for usable year values in the given YEAR_FIELD.

    Handles:
      - integer / numeric fields (treat value as year),
      - date/datetime/timestamp (use .year or leading 4 digits),
      - generic text (search for '18xx', '19xx', or '20xx').

    Returns:
      Sorted list of unique valid years in [1800, 2100], or [] if none.
    """
    lyr = as_layer(layer_or_path, "year_scan")
    real = _resolve_field_name(lyr, year_field)
    if not real:
        return []

    idx = lyr.fields().indexFromName(real)
    fld = lyr.fields()[idx]
    tname = (fld.typeName() or "").lower()
    years = set()

    for feat in lyr.getFeatures():
        v = feat[real]
        if v is None:
            continue
        try:
            # Numeric-like types: int, integer, integer64, real, double, numeric, decimal, etc.
            if any(key in tname for key in ("int", "integer", "real", "double", "numeric", "decimal")):
                y = int(v)

            # Date / datetime / timestamp types
            elif "date" in tname or "time" in tname:
                if hasattr(v, "year"):
                    y = int(v.year())
                else:
                    y = int(str(v)[:4])

            else:
                # Generic text: regex search for a 4-digit year
                m = re.search(r"(18|19|20)\d{2}", str(v))
                if not m:
                    continue
                y = int(m.group(0))

            if 1800 <= y <= 2100:
                years.add(y)

        except Exception:
            # Skip malformed values
            continue

    return sorted(years)

def _subset_by_year(layer_or_path, year_field, yval, context, feedback):
    """
    Create a SHP subset of polygons where YEAR_FIELD == yval.

    Works for:
      - integer / numeric fields (exact equality, robust to Integer64 etc.),
      - date fields (year(<field>) == yval),
      - text fields (regex-extract 4-digit year, compare to yval).

    Returns:
      Path to SHP subset, or None if no features match.
    """
    lyr = as_layer(layer_or_path, "subset_src")
    real = _resolve_field_name(lyr, year_field)
    if not real:
        feedback.pushInfo(f"[subset_by_year] Year field '{year_field}' not found.")
        return None

    idx = lyr.fields().indexFromName(real)
    fld = lyr.fields()[idx]
    tname = (fld.typeName() or "").lower()
    y = int(yval)

    # 1) Numeric-like field (Year, rok, Integer64, etc.)
    if any(key in tname for key in ("int", "integer", "real", "double", "numeric", "decimal")):
        # Robust even if some values are stored as text but numeric
        expr = f"to_int(\"{real}\") = {y}"

    # 2) Date / datetime / timestamp field (e.g. d: Date)
    elif "date" in tname or "time" in tname:
        expr = f"year(\"{real}\") = {y}"

    # 3) Generic text field fallback
    else:
        expr = (
            f"to_int(regexp_substr(tostring(\"{real}\"),"
            f"'(18|19|20)\\\\d{{2}}')) = {y}"
        )

    subset_shp = _temp_path_tracked(f"subset_{y}_", ".shp")
    runp(
        "native:extractbyexpression",
        {"INPUT": layer_or_path, "EXPRESSION": expr, "OUTPUT": subset_shp},
        context,
        feedback,
        "subset_expr_to_file",
    )

    lyr_out = as_layer(subset_shp, f"subset_{y}")
    cnt = lyr_out.featureCount()
    feedback.pushInfo(f"[subset_by_year] year={y} count={cnt}")
    if cnt == 0:
        feedback.pushInfo(f"Year {y}: No features found. Skipping.")
        return None

    return subset_shp



def _reproject_to_target(src_layer, target_crs, context, feedback):
    """
    Reproject a layer to TARGET_CRS only when needed.

    If the layer already has the target CRS, it is simply frozen to SHP
    (no reprojection). Otherwise, native:reprojectlayer is called.
    """
    lyr = as_layer(src_layer, "reproj_src")
    try:
        if lyr.crs() == target_crs:
            # CRS already matches -> just freeze to a stable SHP
            return freeze_to_shp(lyr, context, feedback, tag="freeze_same_crs")
    except Exception:
        # If something goes wrong with CRS comparison, fall through
        pass

    # Different CRS or unknown -> reproject
    out_shp = _temp_path_tracked("reproj_", ".shp")
    runp("native:reprojectlayer",
         {"INPUT": lyr, "TARGET_CRS": target_crs, "OPERATION": "", "OUTPUT": out_shp},
         context, feedback, "reproject_to_target_shp")
    return out_shp


def add_date_field(input_id, date_value, context, feedback):
    """
    Add a 'cnt' field of DATE type and populate with the given date_value.

    NOTE: The field name 'cnt' is inherited from earlier usage.
    """
    out_shp = _temp_path_tracked("add_date_", ".shp")
    runp("native:fieldcalculator",
         {"INPUT": input_id, "FIELD_NAME": "cnt", "FIELD_TYPE": 3,
          "FIELD_LENGTH": 0, "FIELD_PRECISION": 0, "NEW_FIELD": True,
          "FORMULA": f"to_date('{date_value}')", "OUTPUT": out_shp},
         context, feedback, "add_date_to_shp")
    return out_shp


def add_length_field(input_id, context, feedback):
    """
    Add a 'length' field and populate with $length for each feature.
    """
    out_shp = _temp_path_tracked("add_length_", ".shp")
    runp("native:fieldcalculator",
         {"INPUT": input_id, "FIELD_NAME": "length", "FIELD_TYPE": 0,
          "FIELD_LENGTH": 20, "FIELD_PRECISION": 3, "NEW_FIELD": True,
          "FORMULA": "$length", "OUTPUT": out_shp},
         context, feedback, "add_length_to_shp")
    return out_shp


def add_year_field(input_id, year_value, context, feedback):
    """
    Ensure an integer 'Year' field exists and is set to year_value.

    If an existing 'Year' field is present, it is dropped and recreated
    to avoid type or content conflicts.
    """
    src = as_layer(input_id, "add_year_src")
    fields = [f.name() for f in src.fields()]
    stage = input_id

    # Remove any pre-existing 'Year' field
    if "Year" in fields:
        dropped = _temp_path_tracked("drop_year_", ".shp")
        runp("native:deletecolumn",
             {"INPUT": stage, "COLUMN": ["Year"], "OUTPUT": dropped},
             context, feedback, "drop_existing_Year")
        stage = dropped

    # Add new integer Year field with the specified value
    out_shp = _temp_path_tracked("add_year_", ".shp")
    runp("native:fieldcalculator",
         {"INPUT": stage, "FIELD_NAME": "Year", "FIELD_TYPE": 1,
          "FIELD_LENGTH": 10, "FIELD_PRECISION": 0, "NEW_FIELD": True,
          "FORMULA": str(int(year_value)), "OUTPUT": out_shp},
         context, feedback, "add_Year_int")

    # Optional debug check
    try:
        lyr = as_layer(out_shp, "year_check")
        has = "Year" in [f.name() for f in lyr.fields()]
        feedback.pushInfo(f"[add_year_field] Year={year_value} field created: {has}")
    except Exception:
        pass

    return out_shp


def _tag_and_measure(center_path, date_value, context, feedback):
    """
    Add standard centerline attributes:

      - 'Centerln' = 'centerline' (string label)
      - 'cnt' = date (from date_value)
      - 'length' = $length

    Returns:
      Path to SHP with these fields added.
    """
    # Add Centerln = 'centerline'
    tagged = _temp_path_tracked("tag_center_", ".shp")
    runp("native:fieldcalculator",
         {"INPUT": center_path, "FIELD_NAME": "Centerln", "FIELD_TYPE": 2,
          "FIELD_LENGTH": 20, "FIELD_PRECISION": 0, "NEW_FIELD": True,
          "FORMULA": "'centerline'", "OUTPUT": tagged},
         context, feedback, "tag_center_to_shp")

    # Add date and length fields in sequence
    with_date = add_date_field(tagged, date_value, context, feedback)
    return add_length_field(with_date, context, feedback)


def _clean_channel(poly_id, context, feedback):
    """
    Clean polygon geometry for centerline extraction:

      1) Freeze input polygon(s) to SHP.
      2) Run native:fixgeometries to repair invalid geometries.
      3) Run native:deleteholes to remove interior holes.

    Raises:
      QgsProcessingException if resulting layer is empty.
    """
    in_path = freeze_to_shp(poly_id, context, feedback, tag="freeze_before_fix")

    # Fix invalid geometries
    fixed = _temp_path_tracked("fixed_", ".shp")
    runp("native:fixgeometries", {"INPUT": in_path, "OUTPUT": fixed},
         context, feedback, "fixgeom_to_shp")

    # Remove all interior holes (MIN_AREA = 0)
    cleaned = _temp_path_tracked("cleaned_noholes_", ".shp")
    runp("native:deleteholes",
         {"INPUT": fixed, "MIN_AREA": 0.0, "OUTPUT": cleaned},
         context, feedback, "delholes_to_shp")

    # Safety check: ensure we still have at least one polygon
    if as_layer(cleaned, "clean_chk").featureCount() == 0:
        raise QgsProcessingException("Empty polygon after cleaning.")
    return cleaned


def _grass_skeleton_from_polygon(poly_id, context, feedback,
                                 factor=None, max_dangle=None, region=None,
                                 vin_snap=None, vin_minarea=None,
                                 flag_skeleton=True, flag_graph=False,
                                 flag_noattr=True):
    """
    Run GRASS v.voronoi.skeleton on an input polygon and clip to the polygon.

    Parameters map to GRASS options:
      - factor        → smoothness
      - max_dangle    → thin (skeleton simplification)
      - region        → GRASS region string
      - vin_snap      → v.in.ogr snap tolerance
      - vin_minarea   → v.in.ogr min area
      - flag_skeleton → -s flag (extract skeleton)
      - flag_graph    → -g flag (output as graph)
      - flag_noattr   → -t flag (no attribute table)

    Returns:
      Path to a SHP with centerline-like lines clipped to the polygon.
    """
    # Freeze input polygon to SHP for GRASS
    poly_path = freeze_to_shp(poly_id, context, feedback, tag="freeze_poly_for_grass")

    # Create a dedicated temp directory for GRASS (separate from main workspace)
    tmpdir = tempfile.mkdtemp(prefix="scs_grass_")
    _TRACKED_DIRS.append(tmpdir)

    # Copy SHP tree into the GRASS temp directory
    shp_in = os.path.join(tmpdir, "poly_in.shp")
    _copy_shp_tree(poly_path, shp_in)

    # Build GRASS parameter dict
    P = {"input": shp_in}
    if factor is not None:
        P["smoothness"] = float(factor)
    if max_dangle is not None:
        P["thin"] = float(max_dangle)
    nr = _normalize_region(region)
    if nr:
        P["GRASS_REGION_PARAMETER"] = nr
    if vin_snap is not None:
        P["GRASS_SNAP_TOLERANCE_PARAMETER"] = float(vin_snap)
    if vin_minarea is not None:
        P["GRASS_MIN_AREA_PARAMETER"] = float(vin_minarea)
    if flag_skeleton:
        P["-s"] = True
    if flag_graph:
        P["-g"] = True
    if flag_noattr:
        P["-t"] = True

    # Output SHP path (within GRASS workspace)
    shp_out = os.path.join(tmpdir, "skel.shp")
    P["output"] = shp_out

    # Pick grass8 or grass provider depending on availability
    alg_id = "grass8:v.voronoi.skeleton" if QgsApplication.processingRegistry().algorithmById(
        "grass8:v.voronoi.skeleton") else "grass:v.voronoi.skeleton"

    # Run GRASS skeleton, expecting the "output" key
    runp(alg_id, P, context, feedback, "grass_skeleton", expect_key="output")

    # Confirm output exists
    if not (isinstance(shp_out, str) and os.path.exists(shp_out)):
        raise QgsProcessingException("GRASS skeleton produced no output file.")

    # Clip skeleton to the original polygon to remove external branches
    clipped = _temp_path_tracked("skel_clip_", ".shp")
    runp("native:clip",
         {"INPUT": shp_out, "OVERLAY": poly_path, "OUTPUT": clipped},
         context, feedback, "clip_skel_to_shp")
    return clipped


def _voronoi_clip_fallback(poly_id, context, feedback):
    """
    QGIS-only fallback centerline method using Voronoi polygons:

      1) Extract polygon vertices.
      2) Build Voronoi polygons with a bounding buffer.
      3) Clip Voronoi to the polygon.
      4) Convert polygons to lines.
      5) Extract only lines fully within the polygon (PREDICATE = [7] 'within').

    Returns:
      Path to SHP with internal Voronoi edges approximating a centerline network.
    """
    # Freeze polygon to SHP
    poly_path = freeze_to_shp(poly_id, context, feedback, tag="freeze_poly_for_fallback")

    # Extract vertices from polygon(s)
    verts = _temp_path_tracked("verts_", ".shp")
    runp("native:extractvertices", {"INPUT": poly_path, "OUTPUT": verts},
         context, feedback, "verts_to_shp")

    # Require at least 2 vertices to build a Voronoi diagram
    if as_layer(verts, "verts_chk").featureCount() < 2:
        raise QgsProcessingException("Fallback: less than 2 vertices for Voronoi.")

    # Compute a margin for the Voronoi boundary (half the max extent dimension)
    ext = as_layer(poly_path, "poly_for_margin").extent()
    margin = max(ext.width(), ext.height()) * 0.5 or 1.0

    # Choose the available Voronoi polygons algorithm (native or qgis)
    vor_alg = "native:voronoipolygons" if QgsApplication.processingRegistry().algorithmById(
        "native:voronoipolygons") else "qgis:voronoipolygons"

    # Build Voronoi polygons around vertices
    vor = _temp_path_tracked("vor_", ".shp")
    runp(vor_alg, {"INPUT": verts, "BUFFER": float(margin), "OUTPUT": vor},
         context, feedback, "voronoi_to_shp")

    # Clip Voronoi polygons to the channel polygon
    clipv = _temp_path_tracked("clipv_", ".shp")
    runp("native:clip",
         {"INPUT": vor, "OVERLAY": poly_path, "OUTPUT": clipv},
         context, feedback, "clip_voronoi_to_shp")

    # Convert clipped Voronoi polygons to lines (shared edges)
    lines = _temp_path_tracked("voro_lines_", ".shp")
    runp("native:polygonstolines",
         {"INPUT": clipv, "KEEP_FIELDS": False, "OUTPUT": lines},
         context, feedback, "voro2lines_to_shp")

    # Extract only line segments that lie within the polygon (predicate 7 = 'within')
    interior = _temp_path_tracked("interior_", ".shp")
    runp("native:extractbylocation",
         {"INPUT": lines, "PREDICATE": [7], "INTERSECT": poly_path, "OUTPUT": interior},
         context, feedback, "interior_to_shp")

    return interior


def _to_single_centerline(line_src, context, feedback, tag="single_centerline"):
    """
    Merge a line network into a single (multi)line centerline.

    Strategy:
      1) Try native:mergelines (preferred).
      2) If unavailable/fails, try native:linemerge.
      3) If that fails, fall back to native:unaryunion.
      4) Dissolve all features into one.
      5) Collect them into a single multipart feature.

    Returns:
      Path to SHP with a (multi)line geometry representing the centerline.
    """
    # Step 1–3: Merge line segments into a simpler geometry
    merged = _temp_path_tracked(f"{tag}_merged_", ".shp")
    try:
        # Preferred algorithm
        runp("native:mergelines", {"INPUT": line_src, "OUTPUT": merged},
             context, feedback, f"{tag}_mergelines")
    except Exception:
        try:
            # Alternative algorithm
            merged = _temp_path_tracked(f"{tag}_linemerge_", ".shp")
            runp("native:linemerge", {"INPUT": line_src, "OUTPUT": merged},
                 context, feedback, f"{tag}_linemerge")
        except Exception:
            # Final fallback: unary union
            merged = _temp_path_tracked(f"{tag}_uunion_", ".shp")
            runp("native:unaryunion", {"INPUT": line_src, "OUTPUT": merged},
                 context, feedback, f"{tag}_unaryunion")

    # Step 4: Dissolve into a single feature (or as few as possible)
    dissolved = _temp_path_tracked(f"{tag}_dissolve_", ".shp")
    runp("native:dissolve",
         {"INPUT": merged, "FIELD": [], "SEPARATE_DISJOINT": False, "OUTPUT": dissolved},
         context, feedback, f"{tag}_dissolve")

    # Step 5: Collect all geometries into a single multi-part feature
    collected = _temp_path_tracked(f"{tag}_collect_", ".shp")
    runp("native:collect",
         {"INPUT": dissolved, "COLUMN": None, "OUTPUT": collected},
         context, feedback, f"{tag}_collect")
    return collected


def _has_area(poly_path, context, feedback):
    """
    Check if a polygon layer has any features with area > 0.

    Returns:
      True if at least one feature satisfies $area > ~0, else False.
    """
    try:
        lyr = as_layer(poly_path, "area_chk")
        if lyr.featureCount() == 0:
            return False

        # Filter by a small positive area threshold
        tmp = runp("native:extractbyexpression",
                   {"INPUT": lyr, "EXPRESSION": "$area > 0.0000001",
                    "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
                   context, feedback, "has_area_filter")
        return as_layer(tmp, "area_chk2").featureCount() > 0
    except Exception:
        return False


def _centerline_one(poly_id, context, feedback, grass_opts,
                    already_cleaned=False, name_hint=None):
    """
    Generate a single (multi)line centerline from polygon(s).

    Steps:
      1) Clean the polygon(s) (fix geometries / remove holes) unless already_cleaned.
      2) If GRASS region is not provided, derive it from polygon extent with padding.
      3) Try GRASS v.voronoi.skeleton to get a skeleton centerline.
      4) On failure, fall back to the Voronoi-clip method.
      5) Merge resulting lines into a single multi-line centerline.

    Returns:
      SHP path containing the derived centerline.
    """
    try:
        # Make a local copy of grass_opts to avoid mutating caller's dict
        opts = dict(grass_opts or {})

        # Step 1: Clean polygon unless flagged as already cleaned
        if already_cleaned:
            clean_path = (poly_id if isinstance(poly_id, str)
                          else freeze_to_shp(poly_id, context, feedback,
                                             tag="freeze_already_clean"))
        else:
            clean_path = _clean_channel(poly_id, context, feedback)

        # Step 2: Auto-fill GRASS region if none was provided
        if not opts.get("region"):
            ext = as_layer(clean_path, "Cleaned").extent()
            pad = max(ext.width(), ext.height()) * 0.02 or 1.0
            opts["region"] = f"{ext.xMinimum()-pad},{ext.xMaximum()+pad}," \
                             f"{ext.yMinimum()-pad},{ext.yMaximum()+pad}"

        # Step 3: Attempt GRASS skeleton
        center = None
        try:
            center = _grass_skeleton_from_polygon(clean_path, context, feedback, **opts)
        except Exception as e:
            # On failure, log the error and revert to Voronoi fallback
            nh = name_hint or (os.path.basename(clean_path)
                               if isinstance(clean_path, str) else str(poly_id))
            if feedback:
                feedback.reportError(
                    f"GRASS skeleton failed on input: {nh}\n{e}\nFalling back to Voronoi clip."
                )
            center = _voronoi_clip_fallback(clean_path, context, feedback)

        # Sanity check: centerline must not be empty
        if as_layer(center, "center_chk").featureCount() == 0:
            raise QgsProcessingException("Centerline is empty after skeleton/fallback.")

        # Step 5: Merge to a single centerline geometry
        center = _to_single_centerline(center, context, feedback,
                                       tag=f"single_{os.path.basename(clean_path)}")
        return center

    except Exception as e:
        # Wrap any errors in a QgsProcessingException with helpful context
        raise QgsProcessingException(
            f"Failed on input layer: {name_hint or poly_id}\nError: {e}"
        )


# ---------------- Algorithm ---------------- #

class Modul1CenterlineGrassSHP_QC(QgsProcessingAlgorithm):
    """
    QGIS Processing Algorithm implementing Module 1 (Centerline) using
    GRASS v.voronoi.skeleton with Shapefile intermediates and GPKG finals.
    """
    # Single string output listing all produced paths (semicolon-separated)
    OUTPUTS = "OUTPUTS"

    def name(self):
        """
        Algorithm ID (used internally by QGIS).
        """
        return "module1_centerline_channel_outlines"

    def displayName(self):
        """
        Human-readable algorithm name shown in the Processing Toolbox.
        """
        return "Module_1_Centerline v12 (using GRASS)"

    def group(self):
        """
        Group name for the algorithm in the Processing Toolbox.
        """
        return "SCS Toolbox"

    def groupId(self):
        """
        Group ID used internally.
        """
        return "scs_toolbox"

    def shortHelpString(self):
        """
        Short description shown in the Processing Toolbox help panel.
        """
        return (
            "Generates channel centerlines from channel polygons.\n"
            "- Inputs reprojected to TARGET_CRS only if needed\n"
            "- Per-year centerlines: centro_YYYY0101.gpkg\n"
            "- Optional MERGE_OUTPUT: centerlines_merged.gpkg\n"
            "- UNION_MODE: Channel Zone centerline (SegCenterline) "
            "and union_channel layers\n"
            "- YEAR_FIELD may be int, date, or string with 4-digit year"
        )

    def createInstance(self):
        """
        Required factory method for QGIS Processing framework.
        """
        return Modul1CenterlineGrassSHP_QC()

    def initAlgorithm(self, _config=None):
        """
        Define input parameters and outputs for the algorithm.
        """
        # Output folder where all final GPKG files will be written
        self.addParameter(QgsProcessingParameterFolderDestination(
            "OUTPUT_FOLDER", tr("Output folder")))

        # Input channel polygon layers (one or more; multi-date allowed)
        self.addParameter(QgsProcessingParameterMultipleLayers(
            "INPUT_CHANNELS", tr("Input channel polygons (one or more)"),
            layerType=QgsProcessing.TypeVectorPolygon))

        # Name of the year/date field (can be int/date/string)
        self.addParameter(QgsProcessingParameterString(
            "YEAR_FIELD", tr("Year/date field name")))

        # Target CRS for processing (default: current project CRS)
        self.addParameter(QgsProcessingParameterCrs(
            "TARGET_CRS", tr("Target CRS for processing"),
            defaultValue=QgsProject.instance().crs().authid()))

        # UNION_MODE: build a single all-years union and SegCenterline
        self.addParameter(QgsProcessingParameterBoolean(
            "UNION_MODE",
            tr("Channel Zone Centreline (union of all inputs) instead of per-year"),
            defaultValue=False))

        # MERGE_OUTPUT: additionally merge all per-year centerlines into one layer
        self.addParameter(QgsProcessingParameterBoolean(
            "MERGE_OUTPUT",
            tr("Also merge all per-year centrelines to centerlines_merged.gpkg"),
            defaultValue=False))

        # SAVE_UNION_NOHOLES: optionally save union_channel_noholes and
        # union_channel_byYear_noholes (applies in UNION_MODE only)
        self.addParameter(QgsProcessingParameterBoolean(
            "SAVE_UNION_NOHOLES",
            tr("Save union_channel_noholes and union_channel_byYear_noholes (in UNION_MODE)"),
            defaultValue=False))

        # GRASS / v.voronoi.skeleton options (all are optional)
        self.addParameter(QgsProcessingParameterNumber(
            "GRASS_FACTOR", tr("Factor for output smoothness (optional)"),
            type=QgsProcessingParameterNumber.Double, defaultValue=None, optional=True))

        self.addParameter(QgsProcessingParameterNumber(
            "GRASS_MAX_DANGLE",
            tr("Maximum dangle length; -1 = extract centerline (optional)"),
            type=QgsProcessingParameterNumber.Double, defaultValue=None, optional=True))

        # Flags for v.voronoi.skeleton
        self.addParameter(QgsProcessingParameterBoolean(
            "FLAG_SKELETON", tr("Extract skeletons (-s)"), defaultValue=True))
        self.addParameter(QgsProcessingParameterBoolean(
            "FLAG_GRAPH", tr("Output as graph (-g)"), defaultValue=False))
        self.addParameter(QgsProcessingParameterBoolean(
            "FLAG_NOATTR", tr("Do not create attribute table (-t)"), defaultValue=True))

        # Optional explicit GRASS region extent
        self.addParameter(QgsProcessingParameterExtent(
            "GRASS_REGION", tr("GRASS region extent (optional)")))

        # Optional v.in.ogr snap tolerance and min area
        self.addParameter(QgsProcessingParameterNumber(
            "VIN_SNAP", tr("v.in.ogr snap tolerance (optional)"),
            type=QgsProcessingParameterNumber.Double, defaultValue=None, optional=True))
        self.addParameter(QgsProcessingParameterNumber(
            "VIN_MINAREA", tr("v.in.ogr min area (optional)"),
            type=QgsProcessingParameterNumber.Double, defaultValue=None, optional=True))

        # Single string output listing all final output paths
        self.addOutput(QgsProcessingOutputString(self.OUTPUTS, tr("Output path(s)")))

    def _get_optional_double_parameter(self, parameters, name, context):
        """
        Helper to safely retrieve an optional double parameter.

        Returns:
          float value, or None if the parameter was not set or is invalid.
        """
        try:
            val = self.parameterAsDouble(parameters, name, context)
            raw = parameters.get(name, None)
            # Distinguish between "unset" and 0.0
            if raw in (None, "", "None"):
                return None
            return float(val)
        except Exception:
            return None

    def processAlgorithm(self, parameters, context, feedback):
        """
        Main algorithm execution entry point.

        Orchestrates:
          - input validation and reprojection,
          - UNION_MODE vs PER-YEAR mode,
          - GRASS / fallback centerline extraction,
          - writing per-year / merged / union outputs.

        Returns:
          A dict with OUTPUTS -> semicolon-separated list of output paths.
        """
        global _TRACKED_DIRS
        _TRACKED_DIRS.clear()  # reset temp directory tracker for this run
        output_paths = []

        try:
            # ------------------------------------------------------------------
            # 1) Basic input validation and setup
            # ------------------------------------------------------------------
            out_dir = self.parameterAsString(parameters, "OUTPUT_FOLDER", context)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)

            # Retrieve list of vector polygon layers
            layers = self.parameterAsLayerList(parameters, "INPUT_CHANNELS", context) or []
            polys = [
                lyr for lyr in layers
                if isinstance(lyr, QgsVectorLayer)
                and QgsWkbTypes.geometryType(lyr.wkbType()) == QgsWkbTypes.PolygonGeometry
            ]
            if not polys:
                raise QgsProcessingException("No valid polygon vector layers were provided.")

            # Year field name must not be empty
            year_field = self.parameterAsString(parameters, "YEAR_FIELD", context)
            if not year_field:
                raise QgsProcessingException("The Year/date field name cannot be empty.")

            # Check for empty layers early
            for lyr in polys:
                if lyr.featureCount() == 0:
                    raise QgsProcessingException(f"Input layer is empty: {lyr.name()}")

            # Target CRS for all processing
            target_crs = self.parameterAsCrs(parameters, "TARGET_CRS", context)

            # Optional explicit GRASS region from dialog
            region = None
            ext = self.parameterAsExtent(parameters, "GRASS_REGION", context)
            if ext and not ext.isEmpty():
                region = f"{ext.xMinimum()},{ext.xMaximum()},{ext.yMinimum()},{ext.yMaximum()}"

            # Collect GRASS options from parameters
            grass_opts = dict(
                factor=self._get_optional_double_parameter(parameters, "GRASS_FACTOR", context),
                max_dangle=self._get_optional_double_parameter(parameters, "GRASS_MAX_DANGLE", context),
                flag_skeleton=bool(self.parameterAsBoolean(parameters, "FLAG_SKELETON", context)),
                flag_graph=bool(self.parameterAsBoolean(parameters, "FLAG_GRAPH", context)),
                flag_noattr=bool(self.parameterAsBoolean(parameters, "FLAG_NOATTR", context)),
                region=region,  # may be None; _centerline_one will fill from extent if so
                vin_snap=self._get_optional_double_parameter(parameters, "VIN_SNAP", context),
                vin_minarea=self._get_optional_double_parameter(parameters, "VIN_MINAREA", context),
            )

            # Mode flags
            union_mode = self.parameterAsBoolean(parameters, "UNION_MODE", context)
            merge_output = self.parameterAsBoolean(parameters, "MERGE_OUTPUT", context)
            save_noholes = self.parameterAsBoolean(parameters, "SAVE_UNION_NOHOLES", context)

            produced = []               # list of GPKG paths produced
            centerlines_for_merge = []  # list of layer references for merged centerlines

            # ------------------------------------------------------------------
            # 2) Reproject or freeze each input polygon layer to TARGET_CRS
            # ------------------------------------------------------------------
            reproj_paths = []
            for lyr in polys:
                rp = _reproject_to_target(lyr, target_crs, context, feedback)
                reproj_paths.append((lyr.name(), rp))

            # Collect all years present across all reprojected layers
            all_years = set()
            for _, path in reproj_paths:
                for y in _safe_years(path, year_field):
                    all_years.add(y)
            all_years = sorted(all_years)

            # ------------------------------------------------------------------
            # 3) UNION MODE
            # ------------------------------------------------------------------
            if union_mode:
                feedback.pushInfo("-> UNION MODE")

                # 0A: Merge all inputs "as is" (holes preserved) for union_by_year
                merged_raw = _temp_path_tracked("merged_raw_", ".shp")
                runp("native:mergevectorlayers",
                     {"LAYERS": [p for _, p in reproj_paths],
                      "OUTPUT": merged_raw},
                     context, feedback, "merge_raw")

                # Fix geometries before dissolving by year
                merged_raw_fixed = _temp_path_tracked("merged_raw_fixed_", ".shp")
                runp("native:fixgeometries",
                     {"INPUT": merged_raw, "OUTPUT": merged_raw_fixed},
                     context, feedback, "fix_raw_for_by_year")

                # 0B: Pre-clean each input (remove holes) and merge for
                #     the all-years union_channel_noholes
                precleaned = []
                for lname, p in reproj_paths:
                    if feedback.isCanceled():
                        break
                    try:
                        pc = _clean_channel(p, context, feedback)
                        precleaned.append(pc)
                    except Exception as e_pc:
                        feedback.reportError(f"Pre-clean failed for {lname}: {e_pc}")

                if not precleaned and not feedback.isCanceled():
                    raise QgsProcessingException("All inputs failed pre-cleaning.")

                merged_clean = _temp_path_tracked("merged_clean_", ".shp")
                runp("native:mergevectorlayers",
                     {"LAYERS": precleaned, "OUTPUT": merged_clean},
                     context, feedback, "merge_clean")

                # A) union_channel: dissolve by YEAR_FIELD, keep holes
                real_year = _resolve_field_name(merged_raw_fixed, year_field) or year_field
                union_by_year = _temp_path_tracked("union_by_year_", ".shp")
                runp("native:dissolve",
                     {"INPUT": merged_raw_fixed, "FIELD": [real_year],
                      "SEPARATE_DISJOINT": False, "OUTPUT": union_by_year},
                     context, feedback, "dissolve_by_YEAR")

                union_gpkg = os.path.join(out_dir, "union_channel.gpkg")
                save_gpkg_clean(union_by_year, union_gpkg, context, feedback,
                                layer_name="union_channel")
                feedback.pushInfo(f"Wrote per-year multipart union (holes kept): {union_gpkg}")
                produced.append(union_gpkg)

                # B) union_channel_byYear_noholes (optional)
                if save_noholes:
                    try:
                        # Remove holes from per-year union
                        union_by_year_nh = _temp_path_tracked("union_by_year_noholes_", ".shp")
                        runp("native:deleteholes",
                             {"INPUT": union_by_year, "MIN_AREA": 0.0,
                              "OUTPUT": union_by_year_nh},
                             context, feedback, "delholes_by_year")
                        out_nh_by = os.path.join(out_dir, "union_channel_byYear_noholes.gpkg")
                        save_gpkg_clean(union_by_year_nh, out_nh_by, context, feedback,
                                        layer_name="union_channel_byYear_noholes")
                        feedback.pushInfo(f"Wrote: {out_nh_by}")
                        produced.append(out_nh_by)
                    except Exception as e_nh:
                        feedback.reportError(
                            f"Failed to write union_channel_byYear_noholes: {e_nh}"
                        )

                # C) All-years union_channel_noholes (for SegCenterline base)
                feedback.pushInfo("-> Building all-years union_channel_noholes surface...")

                # Dissolve all precleaned inputs into a single multi-year surface
                union_diss_all = _temp_path_tracked("union_diss_all_", ".shp")
                runp("native:dissolve",
                     {"INPUT": merged_clean, "FIELD": [],
                      "SEPARATE_DISJOINT": False, "OUTPUT": union_diss_all},
                     context, feedback, "diss_all_years")

                # Fix geometries
                fixed_all = _temp_path_tracked("fixed_all_", ".shp")
                runp("native:fixgeometries",
                     {"INPUT": union_diss_all, "OUTPUT": fixed_all},
                     context, feedback, "fixgeom_all")

                # Heal small slivers via buffer(0)
                healed_all = _temp_path_tracked("union_healed_buf0_", ".shp")
                runp("native:buffer",
                     {"INPUT": fixed_all, "DISTANCE": 0.0, "SEGMENTS": 8,
                      "END_CAP_STYLE": 0, "JOIN_STYLE": 0, "MITER_LIMIT": 2.0,
                      "DISSOLVE": True, "OUTPUT": healed_all},
                     context, feedback, "heal_buf0_all")

                # Remove holes from healed union
                noholes_all = _temp_path_tracked("union_all_noholes_", ".shp")
                runp("native:deleteholes",
                     {"INPUT": healed_all, "MIN_AREA": 0.0, "OUTPUT": noholes_all},
                     context, feedback, "delholes_all")

                # Optionally save union_channel_noholes
                if save_noholes:
                    try:
                        out_nh = os.path.join(out_dir, "union_channel_noholes.gpkg")
                        save_gpkg_clean(noholes_all, out_nh, context, feedback,
                                        layer_name="union_channel_noholes")
                        feedback.pushInfo(f"Wrote: {out_nh}")
                        produced.append(out_nh)
                    except Exception as e_nh2:
                        feedback.reportError(
                            f"Failed to write union_channel_noholes: {e_nh2}"
                        )

                # D) Decide which geometry to use for SegCenterline skeleton
                union_geom_for_skel = None
                if _has_area(noholes_all, context, feedback):
                    union_geom_for_skel = noholes_all
                    feedback.pushInfo(
                        "SegCenterline will be computed from union_channel_noholes."
                    )
                elif _has_area(healed_all, context, feedback):
                    union_geom_for_skel = healed_all
                    feedback.pushInfo(
                        "Falling back to healed buffer(0) union for GRASS."
                    )
                elif _has_area(fixed_all, context, feedback):
                    union_geom_for_skel = fixed_all
                    feedback.pushInfo(
                        "Falling back to fixed union geometry for GRASS."
                    )
                else:
                    # If everything else fails, reconstruct polygons from boundary
                    bnd = _temp_path_tracked("union_boundary_", ".shp")
                    runp("native:polygonstolines",
                         {"INPUT": fixed_all, "KEEP_FIELDS": False, "OUTPUT": bnd},
                         context, feedback, "boundary_from_fixed")
                    poly_from_bnd = _temp_path_tracked("union_poly_from_bnd_", ".shp")
                    runp("native:polygonize",
                         {"INPUT": bnd, "KEEP_FIELDS": False, "OUTPUT": poly_from_bnd},
                         context, feedback, "polygonize_boundary")
                    union_geom_for_skel = poly_from_bnd
                    feedback.pushInfo(
                        "Using polygonized boundary for GRASS SegCenterline."
                    )

                # Optional gentle buffer before skeleton to smooth tiny artefacts
                use_for_skel = union_geom_for_skel
                try:
                    buf_path = _temp_path_tracked("union_buf_", ".shp")
                    runp("native:buffer",
                         {"INPUT": union_geom_for_skel, "DISTANCE": 0.25,
                          "SEGMENTS": 8, "END_CAP_STYLE": 0,
                          "JOIN_STYLE": 0, "MITER_LIMIT": 2.0,
                          "DISSOLVE": True, "OUTPUT": buf_path},
                         context, feedback, "pre_skel_buffer")
                    if as_layer(buf_path, "buf_chk").featureCount() > 0:
                        use_for_skel = buf_path
                except Exception:
                    # If buffer fails, just use the original union geometry
                    pass

                # Compute SegCenterline from chosen union geometry
                center_path = None
                try:
                    center_path = _centerline_one(
                        use_for_skel, context, feedback, dict(grass_opts),
                        already_cleaned=True, name_hint="union_noholes"
                    )
                except Exception:
                    # If buffered geometry fails, retry with unbuffered union geometry
                    if use_for_skel != union_geom_for_skel:
                        center_path = _centerline_one(
                            union_geom_for_skel, context, feedback, dict(grass_opts),
                            already_cleaned=True, name_hint="union_noholes_raw"
                        )
                    else:
                        # Nothing else to try
                        raise

                # Tag, measure, and save SegCenterline
                if center_path:
                    # Historically, Module 1 used a dummy date;
                    # keep 1900-01-01 for compatibility with downstream modules.
                    center_final = _tag_and_measure(center_path, "1900-01-01",
                                                    context, feedback)
                    seg_out = os.path.join(out_dir, "SegCenterline.gpkg")
                    save_gpkg_clean(center_final, seg_out, context, feedback,
                        layer_name="SegCenterline", force_multiline=True)

                    produced.append(seg_out)
                    feedback.pushInfo("Wrote final SegCenterline.gpkg")
                else:
                    feedback.reportError(
                        "Final SegCenterline path is empty after all attempts."
                    )

                output_paths = produced
                return {self.OUTPUTS: ";".join(output_paths)}

            # ------------------------------------------------------------------
            # 4) PER-YEAR MODE
            # ------------------------------------------------------------------
            feedback.pushInfo("-> PER-YEAR MODE")

            # If no usable years were detected, fall back to a single merged run
            if not all_years:
                feedback.pushInfo(
                    f"No valid years found in field '{year_field}'. "
                    "Continuing with a single merged output."
                )
                # Merge all reprojected inputs into a single polygon layer
                merged_polys = _temp_path_tracked("merged_all_", ".shp")
                runp("native:mergevectorlayers",
                     {"LAYERS": [p for _, p in reproj_paths],
                      "OUTPUT": merged_polys},
                     context, feedback, "merge_all_no_year")

                # Compute centerline on the merged multi-year polygon
                center_path = _centerline_one(
                    merged_polys, context, feedback, grass_opts,
                    already_cleaned=False, name_hint="Merged_Fallback")

                # Use today's date for tagging; year for the 'Year' field
                today = date.today()
                date_str = today.isoformat()
                center_final = _tag_and_measure(center_path, date_str,
                                                context, feedback)
                center_final = add_year_field(center_final, today.year,
                                              context, feedback)

                # Save to centro_<year>0101.gpkg (historical naming pattern)
                out_name = f"centro_{today.year}0101.gpkg"
                out_layer = f"centro_{today.year}"
                out_path = os.path.join(out_dir, out_name)
                save_gpkg_clean(center_final, out_path, context, feedback,
                    layer_name=layer_name, force_multiline=True)

                produced.append(out_path)
                centerlines_for_merge.append(out_path + "|layername=" + out_layer)

            else:
                # We have at least one valid year -> process each year separately
                for y in all_years:
                    if feedback.isCanceled():
                        break
                    feedback.pushInfo(f"Processing year: {y}")

                    # Subset each input layer to this year
                    year_subs = []
                    for lname, reproj_path in reproj_paths:
                        sub_path = _subset_by_year(reproj_path, year_field,
                                                   y, context, feedback)
                        if sub_path:
                            year_subs.append(sub_path)

                    # If no polygons exist for this year across all inputs, skip
                    if not year_subs:
                        feedback.pushInfo(
                            f"Year {y}: no polygons across inputs. Skipping."
                        )
                        continue

                    # Merge all per-layer subsets for this year
                    merged_year = _temp_path_tracked(f"merged_{y}_", ".shp")
                    runp("native:mergevectorlayers",
                         {"LAYERS": year_subs, "OUTPUT": merged_year},
                         context, feedback, f"merge_year_{y}")

                    # Compute centerline for this year's channel zone
                    center_path = _centerline_one(
                        merged_year, context, feedback, grass_opts,
                        already_cleaned=False, name_hint=f"year_{y}_poly")

                    # Tag centerline with date y-01-01 and length, then add Year field
                    date_str = f"{y}-01-01"
                    center_final = _tag_and_measure(center_path, date_str,
                                                    context, feedback)
                    center_final = add_year_field(center_final, y,
                                                  context, feedback)

                    # Save to centro_<YYYY>0101.gpkg
                    out_name = f"centro_{y}0101.gpkg"
                    out_path = os.path.join(out_dir, out_name)
                    layer_name = f"centro_{y}"
                    save_gpkg_clean(center_final, out_path, context, feedback,
                        layer_name=layer_name, force_multiline=True)

                    produced.append(out_path)
                    centerlines_for_merge.append(out_path + "|layername=" + layer_name)
                    feedback.pushInfo(f"Wrote per-year centerline: {out_name}")

            # ------------------------------------------------------------------
            # 5) OPTIONAL MERGED CENTERLINES (PER-YEAR MODE ONLY)
            # ------------------------------------------------------------------
            if merge_output and centerlines_for_merge and not feedback.isCanceled():
                merge_out = os.path.join(out_dir, "centerlines_merged.gpkg")
                runp("native:mergevectorlayers",
                     {"LAYERS": centerlines_for_merge,
                      "OUTPUT": merge_out},
                     context, feedback, "merge_all_finals")
                produced.append(merge_out)
                feedback.pushInfo(f"Wrote merged output: {merge_out}")

            output_paths = produced

        finally:
            # Ensure temporary directories are cleaned up even on failure
            _cleanup_tmp_dirs(feedback)

        # Return all output paths as a single string (semicolon-separated)
        return {self.OUTPUTS: ";".join(output_paths)}
