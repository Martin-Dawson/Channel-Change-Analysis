# -*- coding: utf-8 -*-

# Module 3 — EA calculation (QGIS) — SCS Toolbox 
# v5.1.6, 2025-11-11
# 
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
"""
Module 3 — EA calculation (QGIS) — SCS Toolbox (v5.1.6, 2025-11-11)
QGIS 3.40.11-compatible; robust field/geometry handling; single GPKG output.

Inputs:
- CHANNEL_POLYGONS: list of channel polygon layers (one per date)
- CENTRELINES: list of centerline layers (one per date)
- FIELD_YEAR: polygon date field (e.g., 'Year' or date)
- CENTERLINE_YEAR_FIELD: centreline date field
- STATISTICS: optional segments layer (line or polygon) for per-segment stats
- INTERVAL: segment spacing (m) used to normalize area->meters/day where needed
- OUTPUT_FOLDER: target directory for outputs
- DELETE_TEMP: bool

Outputs:
- <OUTPUT_FOLDER>/SCS_EA_<first>_<last>.gpkg with sublayers:
    EA_processes_y1_y2
    EAsegments_y1_y2 (if stats)
    EA_rate_y1_y2 (if stats)
"""

import os
import re
import math
import shutil
import time
from datetime import datetime, date
from datetime import datetime as _dt

# NOTE: imported but not used; also shadows QGIS "context" variable name.
# Kept here for compatibility but functionally unnecessary.
from matplotlib.style import context




from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterString,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsVectorLayer,
    QgsProcessingUtils,
    QgsProject,
    QgsWkbTypes,
)
import processing


# ---------------------- Helpers: algorithms, parsing, filenames ----------------------

def _algo_exists(alg_id):
    """
    Check whether a processing algorithm exists in the current QGIS registry.
    Used to select between native/qgis variants and to avoid calling missing algs.
    """
    from qgis.core import QgsApplication
    try:
        return QgsApplication.processingRegistry().algorithmById(alg_id) is not None
    except Exception:
        return False


def _run(alg, params, context, feedback=None):
    """
    Thin wrapper around processing.run with:
    - log message to feedback
    - is_child_algorithm=True to make this safe in Processing chains
    """
    if feedback:
        feedback.pushInfo(f"Running {alg} ...")
    return processing.run(alg, params, context=context, is_child_algorithm=True)


def _as_layer(obj, context):
    """
    Resolve obj (layer, id, name, URI, path, or arbitrary INPUT param) to a valid QgsVectorLayer.
    Returns:
        QgsVectorLayer or None if resolution fails.
    Strategy:
      1) If obj already looks like a layer and isValid() → return it
      2) Try ProcessingUtils.mapLayerFromString
      3) Try QgsProject.instance().mapLayer()
      4) Try OGR provider from path or URI
      5) Last resort: save to TEMPORARY_OUTPUT via native:savefeatures then re-resolve
    """

    # 1) Already a QgsVectorLayer
    try:
        if hasattr(obj, "isValid") and obj.isValid():
            return obj
    except Exception:
        pass

    s = str(obj) if obj is not None else ""

    # 2) Try processing resolver (works for layer ids and some URIs)
    try:
        lyr = QgsProcessingUtils.mapLayerFromString(s, context)
        if lyr and lyr.isValid():
            return lyr
    except Exception:
        pass

    # 3) Try project-by-id lookup
    try:
        lyr = QgsProject.instance().mapLayer(s)
        if lyr and lyr.isValid():
            return lyr
    except Exception:
        pass

    # 4) Try loading via OGR provider (path, layername URI, PostGIS URI)
    try:
        if os.path.exists(s) or "|layername=" in s.lower() or s.lower().startswith("postgres"):
            name = os.path.splitext(os.path.basename(s))[0] if os.path.exists(s) else "layer"
            lyr = QgsVectorLayer(s, name, "ogr")
            if lyr and lyr.isValid():
                return lyr
    except Exception:
        pass

    # 5) Last resort: materialize features to TEMPORARY_OUTPUT and re-resolve
    try:
        out = processing.run(
            "native:savefeatures",
            {"INPUT": obj, "OUTPUT": "TEMPORARY_OUTPUT"},
            context=context, is_child_algorithm=True,
        )["OUTPUT"]
        lyr = _as_layer(out, context)  # recurse once
        if lyr and lyr.isValid():
            return lyr
    except Exception:
        pass

    return None


def _add_const_field(layer, name, formula, ftype, length, precision, context, feedback):
    """
    Create or update a field using Field Calculator.
    Parameters:
        layer   : input layer or reference
        name    : field name
        formula : QGIS expression (string), e.g. "'text'", "1", etc.
        ftype   : 0=float, 1=int, 2=string
        length  : field length
        precision: field precision
    Returns:
        Updated layer (as layer object or ID) if successful, else original layer.
    """
    lyr = _as_layer(layer, context)
    if lyr is None or not lyr.isValid():
        if feedback:
            feedback.reportError(f"[_add_const_field] Invalid layer for field {name}.")
        return layer

    # Determine whether the field already exists; if so we update, else we create
    field_exists = name in lyr.fields().names()
    params = {
        "INPUT": lyr,
        "FIELD_NAME": name,
        "NEW_FIELD": not field_exists,
        "FORMULA": formula,
        "OUTPUT": "TEMPORARY_OUTPUT",
    }
    if not field_exists:
        # Only need to specify type/length/precision when a new field is created
        params.update({
            "FIELD_TYPE": int(ftype),
            "FIELD_LENGTH": int(length),
            "FIELD_PRECISION": int(precision),
        })
    try:
        out = _run("native:fieldcalculator", params, context, feedback)["OUTPUT"]
        return _as_layer(out, context) or out
    except Exception as e:
        if feedback:
            feedback.reportError(f"[_add_const_field] Failed for '{name}': {e}")
        return lyr


def _strip_pk_fields(layer, context, feedback=None):
    """
    Drop fields that conflict with GPKG primary key or that are common PK clones
    (fid, ogc_fid, pkuid, pkuid_2, pkuid_3).
    This helps avoid conflicts when writing to GeoPackage.
    """
    lyr = _as_layer(layer, context)
    if lyr is None or not lyr.isValid():
        return _as_layer(layer, context) or layer

    # Candidate PK-like fields to be removed
    kill = [n for n in lyr.fields().names()
            if n.lower() in ("fid", "ogc_fid", "pkuid", "pkuid_2", "pkuid_3")]
    if not kill:
        return lyr

    try:
        out = _run(
            "native:deletecolumn",
            {"INPUT": lyr, "COLUMN": kill, "OUTPUT": "TEMPORARY_OUTPUT"},
            context, feedback
        )["OUTPUT"]
        return _as_layer(out, context) or lyr
    except Exception as e:
        if feedback:
            feedback.reportError(f"[strip_pk] deletecolumn failed: {e} (continuing without strip)")
        return lyr


def _force_2d_safe(layer, context, feedback=None):
    """
    Safely coerce geometries to 2D.
    Strategy:
      1) Try native:force2d if available.
      2) Fallback to native/qgis:dropmz (drop M and Z).
      3) If everything fails, return layer unchanged.
    """
    lyr = _as_layer(layer, context)
    if lyr is None or not lyr.isValid():
        return _as_layer(layer, context)

    # Prefer native:force2d if available
    if _algo_exists("native:force2d"):
        try:
            out = _run(
                "native:force2d",
                {"INPUT": lyr, "OUTPUT": "TEMPORARY_OUTPUT"},
                context, feedback
            )["OUTPUT"]
            o = _as_layer(out, context)
            if o and o.isValid():
                return o
        except Exception:
            pass

    # Fallback to drop M/Z using native or qgis:
    for alg in ("native:dropmz", "qgis:dropmz"):
        if _algo_exists(alg):
            try:
                out = _run(
                    alg,
                    {"INPUT": lyr, "DROP_M": True, "DROP_Z": True,
                     "OUTPUT": "TEMPORARY_OUTPUT"},
                    context, feedback
                )["OUTPUT"]
                o = _as_layer(out, context)
                if o and o.isValid():
                    return o
            except Exception:
                pass

    # Last resort: return unchanged
    return lyr


def _polys_only(layer, context, feedback):
    """
    Filter a layer down to polygon/multipolygon geometries only.
    Returns a TEMPORARY_OUTPUT vector layer.
    """
    return _run(
        "native:extractbyexpression",
        {
            "INPUT": layer,
            "EXPRESSION": "geometry_type($geometry) IN ('Polygon','MultiPolygon')",
            "OUTPUT": "TEMPORARY_OUTPUT"
        },
        context, feedback
    )["OUTPUT"]


def _ensure_polygon(layer, context, feedback, tag, tiny=0.01):
    """
    Returns a POLYGON layer from any geometry type:
      - If input is polygon: fixgeometries → 2D → return.
      - If line: linestopolygons → (if needed polygonize) → tiny buffer → dissolve+fix+2D.
      - If point: tiny buffer → fix+2D.
      - Otherwise: raise.
    This is used before union/diff/intersection to ensure consistent polygon inputs.
    """
    lyr = _as_layer(layer, context)
    if lyr is None or not lyr.isValid():
        raise QgsProcessingException(f"[{tag}] invalid layer")

    # Determine geometry type
    wkb   = _geom_type(lyr, context)
    gtype = QgsWkbTypes.geometryType(wkb)

    # Case 1: Already polygon / multipolygon
    if gtype == QgsWkbTypes.PolygonGeometry:
        try:
            cleaned = _run(
                "native:fixgeometries",
                {"INPUT": lyr, "OUTPUT": "TEMPORARY_OUTPUT"},
                context, feedback
            )["OUTPUT"]
        except Exception:
            cleaned = lyr
        return _force_2d_safe(cleaned, context, feedback)

    # Case 2: Line data -> convert to polygons
    if gtype == QgsWkbTypes.LineGeometry:
        poly = None

        # Try native:linestopolygons first
        if _algo_exists("native:linestopolygons"):
            try:
                poly = _run(
                    "native:linestopolygons",
                    {"LINES": lyr, "OUTPUT": "TEMPORARY_OUTPUT"},
                    context, feedback
                )["OUTPUT"]
            except Exception:
                poly = None

        # Fallback to polygonize if needed
        if poly is None:
            try:
                poly = _run(
                    "native:polygonize",
                    {"INPUT": lyr, "KEEP_FIELDS": False, "OUTPUT": "TEMPORARY_OUTPUT"},
                    context, feedback
                )["OUTPUT"]
            except Exception:
                poly = None

        # Last fallback: small buffer around lines to make polygons
        if poly is None:
            poly = _run(
                "native:buffer",
                {
                    "INPUT": lyr,
                    "DISTANCE": tiny,
                    "SEGMENTS": 8,
                    "END_CAP_STYLE": 0,
                    "JOIN_STYLE": 0,
                    "MITER_LIMIT": 2.0,
                    "DISSOLVE": False,
                    "OUTPUT": "TEMPORARY_OUTPUT"
                },
                context, feedback
            )["OUTPUT"]

        # Filter to polygon geometries, dissolve & fix
        try:
            poly = _run(
                "native:extractbyexpression",
                {
                    "INPUT": poly,
                    "EXPRESSION": "geometry_type($geometry) IN ('Polygon','MultiPolygon')",
                    "OUTPUT": "TEMPORARY_OUTPUT"
                },
                context, feedback
            )["OUTPUT"]
        except Exception:
            pass
        try:
            poly = _run(
                "native:dissolve",
                {"INPUT": poly, "FIELD": [], "SEPARATE_DISJOINT": False,
                 "OUTPUT": "TEMPORARY_OUTPUT"},
                context, feedback
            )["OUTPUT"]
        except Exception:
            pass
        try:
            poly = _run(
                "native:fixgeometries",
                {"INPUT": poly, "OUTPUT": "TEMPORARY_OUTPUT"},
                context, feedback
            )["OUTPUT"]
        except Exception:
            pass

        return _force_2d_safe(poly, context, feedback)

    # Case 3: Point data -> tiny buffer to polygons
    if gtype == QgsWkbTypes.PointGeometry:
        poly = _run(
            "native:buffer",
            {
                "INPUT": lyr,
                "DISTANCE": tiny,
                "SEGMENTS": 8,
                "END_CAP_STYLE": 0,
                "JOIN_STYLE": 0,
                "MITER_LIMIT": 2.0,
                "DISSOLVE": False,
                "OUTPUT": "TEMPORARY_OUTPUT"
            },
            context, feedback
        )["OUTPUT"]
        try:
            poly = _run(
                "native:fixgeometries",
                {"INPUT": poly, "OUTPUT": "TEMPORARY_OUTPUT"},
                context, feedback
            )["OUTPUT"]
        except Exception:
            pass
        return poly

    # All other geometry types are not supported for polygon-only operations
    raise QgsProcessingException(
        f"[{tag}] non-polygon geometry not supported for union/diff/intersection."
    )


def _fix_geoms(layer, context, feedback=None):
    """
    Robust geometry cleaning chain:
    1) fixgeometries
    2) makevalid (if available)
    3) buffer(0) to heal minor issues
    4) force2d or drop M/Z
    Returns a best-effort cleaned layer.
    """
    lyr = _as_layer(layer, context)
    if lyr is None or not lyr.isValid():
        if feedback:
            feedback.reportError("[fix] input not a valid layer; returning as-is.")
        return _as_layer(layer, context) or layer

    # Step 1: fixgeometries
    try:
        lyr = _run(
            "native:fixgeometries",
            {"INPUT": lyr, "OUTPUT": "TEMPORARY_OUTPUT"},
            context, feedback
        )["OUTPUT"]
        lyr = _as_layer(lyr, context) or lyr
    except Exception as e:
        if feedback:
            feedback.reportError(f"[fix] fixgeometries failed: {e}")

    # Step 2: makevalid (if available)
    if _algo_exists("native:makevalid"):
        try:
            lyr = _run(
                "native:makevalid",
                {"INPUT": lyr, "METHOD": 0, "OUTPUT": "TEMPORARY_OUTPUT"},
                context, feedback
            )["OUTPUT"]
            lyr = _as_layer(lyr, context) or lyr
        except Exception as e:
            if feedback:
                feedback.reportError(f"[fix] makevalid failed: {e}")

    # Step 3: buffer(0) to repair self-intersections etc.
    try:
        lyr = _run(
            "native:buffer",
            {
                "INPUT": lyr, "DISTANCE": 0.0, "SEGMENTS": 8,
                "END_CAP_STYLE": 0, "JOIN_STYLE": 0, "MITER_LIMIT": 2.0,
                "DISSOLVE": False, "OUTPUT": "TEMPORARY_OUTPUT"
            },
            context, feedback
        )["OUTPUT"]
        lyr = _as_layer(lyr, context) or lyr
    except Exception as e:
        if feedback:
            feedback.reportError(f"[fix] buffer(0) failed: {e}")

    # Step 4: force 2D or drop M/Z
    try:
        lyr2d = _run(
            "native:force2d",
            {"INPUT": lyr, "OUTPUT": "TEMPORARY_OUTPUT"},
            context, feedback
        )["OUTPUT"]
        lyr = _as_layer(lyr2d, context) or lyr
    except Exception:
        try:
            lyr2d = _run(
                "native:dropmz",
                {"INPUT": lyr, "DROP_M": True, "DROP_Z": True,
                 "OUTPUT": "TEMPORARY_OUTPUT"},
                context, feedback
            )["OUTPUT"]
            lyr = _as_layer(lyr2d, context) or lyr
        except Exception:
            pass

    return _as_layer(lyr, context) or lyr


def _drop_null_tiny_polys(layer, min_area, context, feedback=None):
    """
    Remove:
      - null geometries
      - non-polygon geometries
      - polygons with area <= min_area
    Returns filtered TEMPORARY_OUTPUT layer, or original if it fails.
    """
    expr = (
        "$geometry IS NOT NULL AND "
        "geometry_type($geometry) IN ('Polygon','MultiPolygon') AND "
        f"area($geometry) > {float(min_area)}"
    )
    try:
        out = _run(
            "native:extractbyexpression",
            {"INPUT": layer, "EXPRESSION": expr, "OUTPUT": "TEMPORARY_OUTPUT"},
            context, feedback
        )["OUTPUT"]
        return _as_layer(out, context)
    except Exception as e:
        if feedback:
            feedback.reportError(f"[drop_null_tiny] failed ({e}); layer passed unchanged.")
        return _as_layer(layer, context)


def _snap_to_grid(layer, grid_size, context=None, feedback=None):
    """
    Snap geometries to a uniform grid to stabilize topology and help
    intersection/union succeed.
    Strategy:
      - Prefer native:snap or native:snapgeometries if available.
      - Fallback to native:roundgeometry to reduce coord precision.
    """
    lyr = _as_layer(layer, context)
    if lyr is None or not lyr.isValid():
        if feedback:
            feedback.reportError("[snap_to_grid] Invalid layer")
        return layer

    try:
        # Preferred snap algorithm (depending on version)
        alg = "native:snap" if _algo_exists("native:snap") else "native:snapgeometries"
        out = _run(
            alg,
            {
                "INPUT": lyr,
                "REFERENCE_LAYER": lyr,
                "TOLERANCE": grid_size,
                "BEHAVIOR": 0,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )["OUTPUT"]
        return _as_layer(out, context) or out
    except Exception:
        # Fallback: round coordinates
        try:
            out = _run(
                "native:roundgeometry",
                {
                    "INPUT": lyr,
                    "DECIMALS": max(0, int(round(-1 * (math.log10(grid_size))))),
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context,
                feedback,
            )["OUTPUT"]
            return _as_layer(out, context) or out
        except Exception as e:
            if feedback:
                feedback.reportError(f"[snap_to_grid] Failed: {e}")
            return lyr


def _safe_union(A, B, grid=0.10, context=None, feedback=None, tag="union"):
    """
    Robust union of two polygon layers, with multiple fallback strategies:
      1) Try native:union with a sequence of GRID_SIZE values.
      2) If all fail or produce empty, fallback to merge→dissolve (area-only union).
    This is used heavily for EA unions and orientation mask merges.
    """
    # Ensure both inputs are polygons
    A0 = _ensure_polygon(A, context, feedback, f"{tag}_A")
    B0 = _ensure_polygon(B, context, feedback, f"{tag}_B")

    # Sequence of grids to try; the first element is the user-specified grid (if any)
    grid_seq = [grid if grid is not None else 0.10, 0.0, 0.01, 0.10, 0.50, 1.0]
    tried = []

    for gs in grid_seq:
        try:
            if feedback:
                feedback.pushInfo(f"[safe_union] native:union (GRID_SIZE={gs}) on {tag}")

            out = _run(
                "native:union",
                {
                    "INPUT": A0,
                    "OVERLAY": B0,
                    "OVERLAY_FIELDS_PREFIX": "",
                    "GRID_SIZE": float(gs),
                    "OUTPUT": "TEMPORARY_OUTPUT"
                },
                context, feedback
            )["OUTPUT"]

            # Clean PK fields, force 2D, and strip null/tiny polygons
            out = _strip_pk_fields(out, context, feedback)
            out = _force_2d_safe(out, context, feedback)
            out = _drop_null_tiny_polys(out, 1e-8, context, feedback)

            if not is_empty_layer(out, context, feedback, f"{tag}_union_result"):
                return _as_layer(out, context) or out

            tried.append(f"GRID={gs}: empty")
        except Exception as e:
            tried.append(f"GRID={gs}: {e}")

    # If we are here, all union attempts failed
    if feedback:
        feedback.reportError(
            f"[safe_union] All native:union attempts failed on {tag}: {', '.join(map(str, tried))}"
        )
        feedback.reportError(
            "[safe_union] Falling back to merge→dissolve (attribute-blind area union)."
        )

    # Fallback: merge + dissolve (attributes are not preserved faithfully)
    try:
        merged = _run(
            "native:mergevectorlayers",
            {"LAYERS": [A0, B0], "CRS": None, "OUTPUT": "TEMPORARY_OUTPUT"},
            context, feedback
        )["OUTPUT"]
        dissolved = _run(
            "native:dissolve",
            {"INPUT": merged, "FIELD": [], "SEPARATE_DISJOINT": False,
             "OUTPUT": "TEMPORARY_OUTPUT"},
            context, feedback
        )["OUTPUT"]
        dissolved = _force_2d_safe(dissolved, context, feedback)
        dissolved = _drop_null_tiny_polys(dissolved, 1e-8, context, feedback)
        return _as_layer(dissolved, context) or dissolved
    except Exception as e:
        if feedback:
            feedback.reportError(f"[safe_union] merge→dissolve fallback failed: {e}")
        return _as_layer(A0, context) or A0


def _envelope_fallback(A, B, context, feedback=None):
    """
    Build an approximate envelope: merge both layers and dissolve.
    Used as a very last fallback for clipping intersections.
    """
    merged = _merge_layers([A, B], context, feedback)
    return _dissolve_all(merged, context, feedback)


def _area_union_envelope(A, B, context, feedback):
    """
    Build a single envelope polygon from A and B:
      - Ensure both are polygons
      - Buffer(0) each to heal geometry
      - Merge + dissolve
    Returns a layer representing the union/envelope of both polygons.
    """
    A = _ensure_polygon(A, context, feedback, "area_union_A")
    B = _ensure_polygon(B, context, feedback, "area_union_B")

    # Heal geometry via buffer(0)
    try:
        A = _run(
            "native:buffer",
            {
                "INPUT": A, "DISTANCE": 0.0, "SEGMENTS": 8,
                "END_CAP_STYLE": 0, "JOIN_STYLE": 0, "MITER_LIMIT": 2.0,
                "DISSOLVE": False, "OUTPUT": "TEMPORARY_OUTPUT"
            },
            context, feedback
        )["OUTPUT"]
        B = _run(
            "native:buffer",
            {
                "INPUT": B, "DISTANCE": 0.0, "SEGMENTS": 8,
                "END_CAP_STYLE": 0, "JOIN_STYLE": 0, "MITER_LIMIT": 2.0,
                "DISSOLVE": False, "OUTPUT": "TEMPORARY_OUTPUT"
            },
            context, feedback
        )["OUTPUT"]
    except Exception:
        pass

    # Merge then dissolve all to single envelope
    merged = _run(
        "native:mergevectorlayers",
        {"LAYERS": [A, B], "CRS": None, "OUTPUT": "TEMPORARY_OUTPUT"},
        context, feedback
    )["OUTPUT"]

    out = _run(
        "native:dissolve",
        {"INPUT": merged, "FIELD": [], "SEPARATE_DISJOINT": False,
         "OUTPUT": "TEMPORARY_OUTPUT"},
        context, feedback
    )["OUTPUT"]
    return out


def _ensure_string_field(layer, name, context, feedback, length=32):
    """
    Ensure that a string field named 'name' exists.
    If it doesn't, create it with given length and NULL default.
    Returns layer ID or object.
    """
    lyr = _as_layer(layer, context)
    if lyr is None or not lyr.isValid():
        return lyr
    if lyr.fields().indexOf(name) != -1:
        return lyr

    return _run(
        "native:fieldcalculator",
        {
            "INPUT": lyr,
            "FIELD_NAME": name,
            "FIELD_TYPE": 2,
            "FIELD_LENGTH": length,
            "FIELD_PRECISION": 0,
            "NEW_FIELD": True,
            "FORMULA": "NULL",
            "OUTPUT": "TEMPORARY_OUTPUT",
        },
        context, feedback
    )["OUTPUT"]


def _retag_union_by_dates_only_nulls(unionEA, src_old, y1, src_yng, y2, context, feedback):
    """
    Fill ONLY NULL y_/T_ fields on unionEA by spatially joining per-date envelopes
    from src_old and src_yng.
    Here we set J_T = 'channel' (we do NOT import mid_channel_bar).
    Islands are handled elsewhere from envelope minus channel (STEP 2).
    """
    u = _as_layer(unionEA, context)

    # Ensure y_y1, T_y1, y_y2, T_y2 exist as string fields on unionEA
    for nm in (f"y_{y1}", f"T_{y1}", f"y_{y2}", f"T_{y2}"):
        if u.fields().indexOf(nm) == -1:
            u = _run(
                "native:fieldcalculator",
                {
                    "INPUT": u, "FIELD_NAME": nm, "FIELD_TYPE": 2,
                    "FIELD_LENGTH": 32, "FIELD_PRECISION": 0,
                    "NEW_FIELD": True, "FORMULA": "NULL",
                    "OUTPUT": "TEMPORARY_OUTPUT"
                },
                context, feedback
            )["OUTPUT"]
        u = _as_layer(u, context)

    # Slight inward buffer to avoid slivers at the envelope boundary
    u_in = _run(
        "native:buffer",
        {
            "INPUT": u, "DISTANCE": -1e-6, "SEGMENTS": 5,
            "END_CAP_STYLE": 0, "JOIN_STYLE": 0, "MITER_LIMIT": 2.0,
            "DISSOLVE": False, "OUTPUT": "TEMPORARY_OUTPUT"
        },
        context, feedback
    )["OUTPUT"]

    # Prepare date-specific source (envelope / channel) layer:
    # - Add fields J_Y, J_T where J_T is hard-coded to 'channel'
    def prep_src(src, y):
        s = _as_layer(src, context)
        # Tag year into J_Y
        s = _run(
            "native:fieldcalculator",
            {
                "INPUT": s, "FIELD_NAME": "J_Y", "FIELD_TYPE": 2, "FIELD_LENGTH": 16,
                "FIELD_PRECISION": 0, "NEW_FIELD": True, "FORMULA": f"'{y}'",
                "OUTPUT": "TEMPORARY_OUTPUT"
            },
            context, feedback
        )["OUTPUT"]

        # Force two-class semantics: treat everything as channel here
        s = _run(
            "native:fieldcalculator",
            {
                "INPUT": s, "FIELD_NAME": "J_T", "FIELD_TYPE": 2, "FIELD_LENGTH": 32,
                "FIELD_PRECISION": 0, "NEW_FIELD": True, "FORMULA": "'channel'",
                "OUTPUT": "TEMPORARY_OUTPUT"
            },
            context, feedback
        )["OUTPUT"]
        return s

    # Join only null y_/T_ values using these J_Y, J_T tags
    def join_fill(u_layer, src, y, side):
        # Predicates: Contains, Overlaps, Within, Intersects
        preds = [1, 5, 6, 0]
        joined = _run(
            "native:joinattributesbylocation",
            {
                "INPUT": u_layer, "JOIN": src, "PREDICATE": preds,
                "JOIN_FIELDS": ["J_Y", "J_T"], "METHOD": 0,
                "DISCARD_NONMATCHING": False, "PREFIX": f"J{side}_",
                "OUTPUT": "TEMPORARY_OUTPUT"
            },
            context, feedback
        )["OUTPUT"]
        # Fill y_y and T_y with existing value or J-side join value
        joined = _run(
            "native:fieldcalculator",
            {
                "INPUT": joined, "FIELD_NAME": f"y_{y}",
                "FIELD_TYPE": 2, "FIELD_LENGTH": 16, "FIELD_PRECISION": 0,
                "NEW_FIELD": False, "FORMULA": f"coalesce(\"y_{y}\", \"J{side}_J_Y\")",
                "OUTPUT": "TEMPORARY_OUTPUT"
            },
            context, feedback
        )["OUTPUT"]
        joined = _run(
            "native:fieldcalculator",
            {
                "INPUT": joined, "FIELD_NAME": f"T_{y}",
                "FIELD_TYPE": 2, "FIELD_LENGTH": 32, "FIELD_PRECISION": 0,
                "NEW_FIELD": False, "FORMULA": f"coalesce(\"T_{y}\", \"J{side}_J_T\")",
                "OUTPUT": "TEMPORARY_OUTPUT"
            },
            context, feedback
        )["OUTPUT"]
        # Drop join helper columns
        joined = _run(
            "native:deletecolumn",
            {
                "INPUT": joined, "COLUMN": [f"J{side}_J_Y", f"J{side}_J_T"],
                "OUTPUT": "TEMPORARY_OUTPUT"
            },
            context, feedback
        )["OUTPUT"]
        return joined

    S1 = prep_src(src_old, y1)
    S2 = prep_src(src_yng, y2)

    u1 = join_fill(u_in, S1, y1, "1")
    u2 = join_fill(u1,  S2, y2, "2")

    return u2


def _merge_layers(layers, context, feedback=None):
    """
    Merge a list of layers into a single TEMPORARY_OUTPUT layer.
    If merge fails, return the first layer as a best effort.
    """
    try:
        out = _run(
            "native:mergevectorlayers",
            {"LAYERS": layers, "CRS": None, "OUTPUT": "TEMPORARY_OUTPUT"},
            context, feedback
        )["OUTPUT"]
        return _as_layer(out, context)
    except Exception as e:
        if feedback:
            feedback.reportError(f"[merge_layers] failed: {e}. Returning first layer.")
        return _as_layer(layers[0], context)


def _dissolve_all(layer, context, feedback=None):
    """
    Dissolve all geometries into one multipart / multipolygon.
    Used to build envelopes and union surfaces.
    """
    try:
        out = _run(
            "native:dissolve",
            {
                "INPUT": layer, "FIELD": [], "SEPARATE_DISJOINT": False,
                "OUTPUT": "TEMPORARY_OUTPUT"
            },
            context, feedback
        )["OUTPUT"]
        return _as_layer(out, context)
    except Exception as e:
        if feedback:
            feedback.reportError(f"[dissolve_all] native:dissolve failed: {e}. Returning input as-is.")
        return _as_layer(layer, context)


def _geom_type(layer, context=None):
    """
    Return wkbType of layer (e.g. Polygon, Line, Point) as QgsWkbTypes enum.
    """
    lyr = _as_layer(layer, context)
    return lyr.wkbType()


def _union_polys(A, B, context, feedback=None, tag="union"):
    """
    Simpler union for polygon layers:
      - fix + 2D for both inputs
      - polygonize lines if needed
      - native:union with GRID_SIZE=0
    Used as a simpler fallback alternative to _safe_union in some places.
    """
    A2 = _as_layer(
        _force_2d_safe(_fix_geoms(A, context, feedback), context=context, feedback=feedback),
        context
    )
    B2 = _as_layer(
        _force_2d_safe(_fix_geoms(B, context, feedback), context=context, feedback=feedback),
        context
    )

    # If geometry type is still line, polygonize it
    if _geom_type(A2, context) == QgsWkbTypes.LineGeometry:
        A2 = _run(
            "native:polygonize",
            {"INPUT": A2, "KEEP_FIELDS": True, "OUTPUT": "TEMPORARY_OUTPUT"},
            context, feedback
        )["OUTPUT"]
        A2 = _as_layer(A2, context)
    if _geom_type(B2, context) == QgsWkbTypes.LineGeometry:
        B2 = _run(
            "native:polygonize",
            {"INPUT": B2, "KEEP_FIELDS": True, "OUTPUT": "TEMPORARY_OUTPUT"},
            context, feedback
        )["OUTPUT"]
        B2 = _as_layer(B2, context)

    out = _run(
        "native:union",
        {
            "INPUT": A2, "OVERLAY": B2,
            "OVERLAY_FIELDS_PREFIX": "", "GRID_SIZE": 0,
            "OUTPUT": "TEMPORARY_OUTPUT"
        },
        context, feedback
    )["OUTPUT"]
    return _as_layer(out, context) or out


def _prep_poly(layer, context, feedback, min_area=1e-8):
    """
    Convenience helper:
      fixgeometries + force2d + drop null/tiny polygons
    """
    lyr = _fix_geoms(layer, context, feedback)
    lyr = _force_2d_safe(lyr, context, feedback)
    return _drop_null_tiny_polys(lyr, min_area, context, feedback)


def _safe_clip(A, mask, context, feedback, grid_sequence=(0.10, 0.25, 0.50, 1.0)):
    """
    Robust clipping:
      1) Clean polygons via _prep_poly
      2) Snap both to a sequence of grid sizes and try native:intersection
      3) If all fail, build an envelope fallback and intersect with it
    Returns intersection or original A as last fallback.
    """
    A0 = _prep_poly(A,    context, feedback)
    M0 = _prep_poly(mask, context, feedback)

    # Try intersection with several grid snaps
    for g in grid_sequence:
        try:
            A1 = _snap_to_grid(A0, g, context, feedback)
            M1 = _snap_to_grid(M0, g, context, feedback)
            out = _run(
                "native:intersection",
                {
                    "INPUT": A1, "OVERLAY": M1,
                    "INPUT_FIELDS": [], "OVERLAY_FIELDS": [],
                    "OVERLAY_FIELDS_PREFIX": "", "OUTPUT": "TEMPORARY_OUTPUT"
                },
                context, feedback
            )["OUTPUT"]
            return _as_layer(out, context) or out
        except Exception as e:
            if feedback:
                feedback.reportError(f"[safe_clip] intersection @GRID={g} failed: {e}")

    # If we get here, attempt intersection with simpler envelope
    if feedback:
        feedback.reportError("[safe_clip] Falling back to merge→dissolve envelope.")
    env = _envelope_fallback(A0, M0, context, feedback)
    try:
        out = _run(
            "native:intersection",
            {
                "INPUT": A0, "OVERLAY": env,
                "INPUT_FIELDS": [], "OVERLAY_FIELDS": [],
                "OVERLAY_FIELDS_PREFIX": "", "OUTPUT": "TEMPORARY_OUTPUT"
            },
            context, feedback
        )["OUTPUT"]
        return _as_layer(out, context) or out
    except Exception:
        return _as_layer(A0, context) or A0


def _clip_polys(A, mask, context, feedback):
    """
    Public wrapper around _safe_clip for polygon clipping.
    """
    return _safe_clip(A, mask, context, feedback)


def _intersect_polys(A, B, context, feedback=None):
    """
    Simple intersection:
      - fix + 2D both inputs
      - native:intersection with no field restrictions
    """
    A2 = _as_layer(
        _force_2d_safe(_fix_geoms(A, context, feedback), context=context, feedback=feedback),
        context
    )
    B2 = _as_layer(
        _force_2d_safe(_fix_geoms(B, context, feedback), context=context, feedback=feedback),
        context
    )
    out = _run(
        "native:intersection",
        {
            "INPUT": A2, "OVERLAY": B2,
            "INPUT_FIELDS": [], "OVERLAY_FIELDS": [],
            "OVERLAY_FIELDS_PREFIX": "", "OUTPUT": "TEMPORARY_OUTPUT"
        },
        context, feedback
    )["OUTPUT"]
    return _as_layer(out, context) or out


def _safe_intersection(A, B, grid=0.10, context=None, feedback=None):
    """
    Robust intersection similar to _safe_union:
      1) fixgeoms both
      2) Snap to several grid sizes and try native:intersection
      3) If all fail, return cleaned A
    """
    A1 = _fix_geoms(A, context, feedback)
    B1 = _fix_geoms(B, context, feedback)

    for g in [grid or 0.10, 0.25, 0.50, 1.00]:
        try:
            A2 = _snap_to_grid(A1, g, context, feedback)
            B2 = _snap_to_grid(B1, g, context, feedback)
            out = _run(
                "native:intersection",
                {
                    "INPUT": A2,
                    "OVERLAY": B2,
                    "INPUT_FIELDS": [],
                    "OVERLAY_FIELDS": [],
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context, feedback
            )["OUTPUT"]
            return _as_layer(out, context) or out
        except Exception as e:
            if feedback:
                feedback.reportError(f"[safe_intersection] GRID={g} failed: {e}")

    if feedback:
        feedback.reportError("[safe_intersection] All attempts failed; returning A unchanged.")
    return _as_layer(A1, context) or A1


def _difference_polys(A, B, context, feedback=None):
    """
    Polygon difference A - B with fix + 2D pre-cleaning on both inputs.
    Used in island detection (envelope minus channel).
    """
    A2 = _as_layer(
        _force_2d_safe(_fix_geoms(A, context, feedback), context=context, feedback=feedback),
        context
    )
    B2 = _as_layer(
        _force_2d_safe(_fix_geoms(B, context, feedback), context=context, feedback=feedback),
        context
    )
    out = _run(
        "native:difference",
        {"INPUT": A2, "OVERLAY": B2, "OUTPUT": "TEMPORARY_OUTPUT"},
        context, feedback
    )["OUTPUT"]
    return _as_layer(out, context) or out


# --- DATE PARSING HELPERS (lenient) ---

def _coerce_date_yyyy_or_yyyymmdd(val, default_md="0101"):
    """
    Attempt to parse val as either:
      - full YYYYMMDD (numeric or embedded in string)
      - ISO-like YYYY-MM-DD or YYYY/MM/DD
      - bare YYYY (18xx–20xx), in which case default_md is appended.
    Returns:
      (year_int, 'YYYYMMDD', date_object)
    Raises ValueError if nothing can be parsed.
    """
    if val is None:
        raise ValueError("empty date")
    s = str(val).strip()

    # Pattern 1: direct YYYYMMDD (e.g. 19990521)
    m8 = re.search(r"\b(1[89]\d{2}|20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\b", s)
    if m8:
        y, mo, da = map(int, m8.groups())
        return y, f"{y:04d}{mo:02d}{da:02d}", date(y, mo, da)

    # Pattern 2: ISO / slash date, e.g. 1999-05-21 or 1999/05/21
    miso = re.search(
        r"\b(1[89]\d{2}|20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b",
        s
    )
    if miso:
        y, mo, da = map(int, miso.groups())
        return y, f"{y:04d}{mo:02d}{da:02d}", date(y, mo, da)

    # Pattern 3: only year, infer default month/day
    m4 = re.search(r"\b(1[89]\d{2}|20\d{2})\b", s)
    if m4:
        y = int(m4.group(1))
        return y, f"{y:04d}{default_md}", date(y, int(default_md[:2]), int(default_md[2:]))

    raise ValueError(f"Cannot parse date/year from: {s!r}")


def _yyyymmdd_from_value_lenient(val):
    """
    Wrapper around _coerce_date_yyyy_or_yyyymmdd that returns YYYYMMDD or None.
    """
    try:
        _, ymd, _ = _coerce_date_yyyy_or_yyyymmdd(val)
        return ymd
    except Exception:
        return None


def _date_object_from_value_lenient(val):
    """
    Wrapper around _coerce_date_yyyy_or_yyyymmdd that returns a date object or None.
    """
    try:
        _, _, dt = _coerce_date_yyyy_or_yyyymmdd(val)
        return dt
    except Exception:
        return None


def _ensure_folder(path):
    """
    Ensure that a folder exists; create if missing (equivalent to mkdir -p).
    """
    os.makedirs(path, exist_ok=True)


def _yyyymmdd_from_text(text):
    """
    Try to extract YYYYMMDD, or infer YYYY0101 from text.
    Returns:
      'YYYYMMDD' or None.
    """
    if not text:
        return None
    s = str(text)

    # Direct 8-digit date with year prefix
    m8 = re.search(r"\b(18|19|20)\d{6}\b", s)
    if m8:
        return m8.group(0)

    # ISO-like date with separators
    m_iso = re.search(r"\b(18|19|20)\d{2}[-/](\d{2})[-/](\d{2})\b", s)
    if m_iso:
        return f"{m_iso.group(1)}{m_iso.group(2)}{m_iso.group(3)}"

    # Bare year
    m4 = re.search(r"\b(18|19|20)\d{2}\b", s)
    if m4:
        return f"{m4.group(0)}0101"

    return None


def _layer_source(layer):
    """
    Safe accessor for layer.source(), returning empty string on error.
    """
    try:
        return layer.source()
    except Exception:
        return ""


def _layer_name(layer):
    """
    Safe accessor for layer.name(), returning empty string on error.
    """
    try:
        return layer.name()
    except Exception:
        return ""


def _yyyymmdd_from_layer_meta(layer):
    """
    Try to get YYYYMMDD from either layer's source path or its name.
    """
    return _yyyymmdd_from_text(f"{_layer_source(layer)} {_layer_name(layer)}")


def _yyyymmdd_from_value(val):
    """
    Strict-ish date parser for layer attributes:
      - QDateTime/QDate-compatible objects
      - datetime/date objects
      - numeric YYYY or YYYYMMDD
      - string formed as YYYYMMDD, YYYY-MM-DD, or bare YYYY
    Returns:
      'YYYYMMDD' or None
    """
    if val is None:
        return None

    # QGIS date/time wrapper types
    try:
        if hasattr(val, "toPyDateTime"):
            dt = val.toPyDateTime()
            return dt.strftime("%Y%m%d")
        if hasattr(val, "toPyDate"):
            d = val.toPyDate()
            return d.strftime("%Y%m%d")
    except Exception:
        pass

    # Python datetime/date objects
    if isinstance(val, datetime):
        return val.strftime("%Y%m%d")
    if isinstance(val, date):
        return val.strftime("%Y%m%d")

    # Numeric types (e.g. 1999 or 19990521)
    if isinstance(val, (int, float)):
        s = str(int(val))
        # pure YYYY
        if re.fullmatch(r"(19|20)\d{2}", s):
            return f"{s}0101"
        # full YYYYMMDD
        if re.fullmatch(r"(19|20)\d{6}", s):
            return s
        return None

    # Strings
    s = str(val).strip()
    if re.fullmatch(r"(19|20)\d{6}", s):
        return s
    if re.fullmatch(r"(19|20)\d{2}[-/]\d{2}[-/]\d{2}", s):
        return f"{s[0:4]}{s[5:7]}{s[8:10]}"
    if re.fullmatch(r"(19|20)\d{2}", s):
        return f"{s}0101"
    return None


def _date_object_from_value(val):
    """
    Convert a value to a Python datetime (YMD) if possible.
    Accepts:
      - QGIS date/time wrappers
      - Python datetime/date
      - numeric/strings recognized by _yyyymmdd_from_value
      - simple yyyy, yyyy-mm-dd, yyyy/mm/dd
    """
    if val is None:
        return None

    # QGIS wrappers
    try:
        if hasattr(val, "toPyDateTime"):
            return val.toPyDateTime()
        if hasattr(val, "toPyDate"):
            d = val.toPyDate()
            return datetime(d.year, d.month, d.day)
    except Exception:
        pass

    # Python types
    if isinstance(val, datetime):
        return val
    if isinstance(val, date):
        return datetime(val.year, val.month, val.day)

    # Numeric
    if isinstance(val, (int, float)):
        ymd = _yyyymmdd_from_value(val)
        return datetime.strptime(ymd, "%Y%m%d") if ymd else None

    # Strings with various formats
    s = str(val).strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass

    # Bare year -> assume 1 Jan
    if re.fullmatch(r"(19|20)\d{2}", s):
        return datetime.strptime(s + "0101", "%Y%m%d")
    return None


def is_empty_layer(obj, context, feedback=None, label=""):
    """
    Determine whether a layer is "empty" (0 features).
    More robust than simply featureCount() in case of errors.
    Logs debug info if feedback is provided.
    """
    lyr = _as_layer(obj, context)
    if lyr is None or not getattr(lyr, "isValid", lambda: False)():
        if feedback and label:
            feedback.reportError(f"[DEBUG] {label}: not a valid layer; treating as EMPTY.")
        return True

    try:
        cnt = lyr.featureCount()
        if feedback and label:
            feedback.pushInfo(f"[DEBUG] {label} featureCount={cnt}")
        return cnt == 0
    except Exception as e:
        # If featureCount fails, try iterating
        if feedback and label:
            feedback.reportError(f"[DEBUG] {label} featureCount() error: {e}; trying iterator.")
        try:
            next(lyr.getFeatures())
            if feedback and label:
                feedback.pushInfo(f"[DEBUG] {label} has at least 1 feature (iterator).")
            return False
        except StopIteration:
            if feedback and label:
                feedback.pushInfo(f"[DEBUG] {label} iterator found 0 features.")
            return True
        except Exception as e2:
            if feedback and label:
                feedback.reportError(
                    f"[DEBUG] {label} getFeatures() error: {e2}; assuming NOT empty."
                )
            return False


def _ensure_exists(p):
    """
    Return True if path exists and has size > 0.
    """
    try:
        return os.path.exists(p) and os.path.getsize(p) > 0
    except Exception:
        return False


def _suffix_path(path):
    """
    Return a new path adding a timestamp suffix before the extension.
    Used when we can't overwrite a locked file cleanly.
    """
    base, ext = os.path.splitext(path)
    return f"{base}_{_dt.now().strftime('%Y%m%d_%H%M%S')}{ext}"


def _move_with_retry(src, dst, feedback=None, attempts=5, sleep_s=0.4, allow_suffix=True):
    """
    Move src to dst with multiple attempts and delays in between.
    If dst is locked, we optionally write a suffixed version and log.
    Returns the final destination path (or empty string on failure).
    """
    # First attempt: remove existing dst if possible
    try:
        if os.path.exists(dst):
            os.remove(dst)
    except Exception as e:
        if feedback:
            feedback.reportError(f"[WRITE] Could not delete existing {dst}: {e}")

    # Multiple attempts to move
    for i in range(attempts):
        try:
            shutil.move(src, dst)
            return dst if _ensure_exists(dst) else ""
        except Exception as e:
            if feedback:
                feedback.reportError(f"[WRITE] move attempt {i+1}/{attempts} failed: {e}")
            time.sleep(sleep_s)

    # If that fails and suffix is allowed, try suffixed destination
    if allow_suffix:
        alt = _suffix_path(dst)
        try:
            shutil.move(src, alt)
            if feedback:
                feedback.reportError(f"[WRITE] Destination locked; wrote to {alt} instead.")
            return alt if _ensure_exists(alt) else ""
        except Exception as e:
            if feedback:
                feedback.reportError(f"[WRITE] move to suffixed path failed: {e}")

    # Last fallback: try copy
    try:
        shutil.copy2(src, dst)
        return dst if _ensure_exists(dst) else ""
    except Exception as e:
        if feedback:
            feedback.reportError(f"[WRITE] copy failed: {e}")
        return ""


def _write_gpkg(input_layer, gpkg_path, layer_name, context, overwrite=True, feedback=None):
    """
    Write a vector layer to a (possibly multi-layer) GeoPackage using PyQGIS writer.
    Strategy:
      1) Clean the input (fixgeometries, 2D, strip PK fields).
      2) Write to a temporary GPKG.
      3) If final doesn't exist → move temp to final.
      4) If final exists       → add/overwrite layer in final file.
    This is robust against locks and ensures we don't corrupt existing GPKG.
    Returns:
      Path to the final GPKG (or "" on failure).
    """
    import tempfile
    from qgis.core import QgsVectorFileWriter, QgsProject, QgsVectorLayer

    def _gpkg_layer_ok(path, name):
        """
        Internal validation: can we load 'name' from 'path' and does it have features?
        """
        try:
            v = QgsVectorLayer(f"{path}|layername={name}", name, "ogr")
            return v.isValid() and v.featureCount() > 0
        except Exception:
            return False

    # Resolve input layer
    lyr = _as_layer(input_layer, context)
    if lyr is None or not lyr.isValid():
        if feedback:
            feedback.reportError("[WRITE] Input layer invalid; skipping write.")
        return ""

    # Sanitize geometry and PK-like fields
    try:
        lyr = _run(
            "native:fixgeometries",
            {"INPUT": lyr, "OUTPUT": "TEMPORARY_OUTPUT"},
            context, feedback
        )["OUTPUT"]
        lyr = _as_layer(lyr, context) or lyr
    except Exception:
        pass
    lyr = _force_2d_safe(lyr, context, feedback)
    lyr = _strip_pk_fields(lyr, context, feedback)
    lyr = _as_layer(lyr, context) or lyr

    # Skip writing if there are no features
    try:
        if _as_layer(lyr, context).featureCount() == 0:
            if feedback:
                feedback.reportError("[WRITE] Layer has 0 features; skipping write.")
            return ""
    except Exception:
        pass

    final_gpkg = gpkg_path
    os.makedirs(os.path.dirname(final_gpkg), exist_ok=True)

    # Always write first to a temporary GPKG
    tmpdir = tempfile.mkdtemp(prefix="scs_ea_write_")
    tmp_gpkg = os.path.join(tmpdir, os.path.basename(final_gpkg))

    # Determine write options for temporary file
    opts = QgsVectorFileWriter.SaveVectorOptions()
    opts.driverName = "GPKG"
    opts.layerName = layer_name
    opts.fileEncoding = "UTF-8"
    opts.datasourceOptions = ["SPATIAL_INDEX=YES"]
    # If tmp_gpkg doesn't exist, we create/overwrite file; else create/overwrite layer
    opts.actionOnExistingFile = (
        QgsVectorFileWriter.CreateOrOverwriteFile
        if not os.path.exists(tmp_gpkg)
        else QgsVectorFileWriter.CreateOrOverwriteLayer
    )

    # Do the write into tmp_gpkg
    try:
        try:
            tctx = context.transformContext()
        except Exception:
            tctx = QgsProject.instance().transformContext()
        ret = QgsVectorFileWriter.writeAsVectorFormatV3(lyr, tmp_gpkg, tctx, opts)
        code = ret[0] if isinstance(ret, (list, tuple)) else ret
        msg  = ret[1] if isinstance(ret, (list, tuple)) and len(ret) > 1 else ""
        if code != QgsVectorFileWriter.NoError:
            raise Exception(msg or f"Writer error code: {code}")
    except Exception as e:
        if feedback:
            feedback.reportError(f"[WRITE] PyQGIS GPKG write failed: {e}")
        return ""

    # If target GPKG doesn't exist, we can move the whole temp file into place
    if not os.path.exists(final_gpkg):
        try:
            shutil.move(tmp_gpkg, final_gpkg)
            if feedback:
                feedback.pushInfo(f"[WRITE] Final GPKG created: {final_gpkg}")
            return final_gpkg
        except Exception as e:
            if feedback:
                feedback.reportError(f"[WRITE] Move to final GPKG failed: {e}")
            return ""

    # If target exists, append/overwrite just the layer in final_gpkg
    try:
        opts2 = QgsVectorFileWriter.SaveVectorOptions()
        opts2.driverName = "GPKG"
        opts2.layerName = layer_name
        opts2.fileEncoding = "UTF-8"
        opts2.datasourceOptions = ["SPATIAL_INDEX=YES"]
        opts2.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteLayer

        src_layer = _as_layer(f"{tmp_gpkg}|layername={layer_name}", context)
        ret2 = QgsVectorFileWriter.writeAsVectorFormatV3(src_layer, final_gpkg, tctx, opts2)
        code2 = ret2[0] if isinstance(ret2, (list, tuple)) else ret2
        msg2  = ret2[1] if isinstance(ret2, (list, tuple)) and len(ret2) > 1 else ""
        if code2 != QgsVectorFileWriter.NoError:
            raise Exception(msg2 or f"Writer error code: {code2}")

        # Validate written layer
        if not _gpkg_layer_ok(final_gpkg, layer_name):
            raise Exception("Layer validation failed after write.")
        if feedback:
            feedback.pushInfo(f"[WRITE] Wrote/updated layer '{layer_name}' in {final_gpkg}")
        return final_gpkg
    except Exception as e:
        if feedback:
            feedback.reportError(f"[WRITE] Layer copy into final GPKG failed: {e}")
        return ""


# ---------------------- The Processing Algorithm ----------------------
class Module3_EAcalculation_QGIS(QgsProcessingAlgorithm):
    """
    Main QGIS Processing algorithm for Module 3 (Erosion/Accumulation).
    Implements the full SCS Toolbox EA logic, including:
      - island-aware and 'stable_floodplain' vs 'inactive_floodplain'
      - LEFT/RIGHT side masks from per-date centrelines
      - direction + migration classification
      - optional per-segment rate statistics
      - all outputs in a single GPKG (one layer per date pair).
    """

    # ---- Parameter keys (used to define and retrieve parameter values) ----
    PARAM_OUTFOLDER = "OUTPUT_FOLDER"
    PARAM_CHAN = "CHANNEL_POLYGONS"
    PARAM_CEN = "CENTRELINES"
    PARAM_FIELD_YEAR = "FIELD_YEAR"
    PARAM_FIELD_CENYEAR = "CENTERLINE_YEAR_FIELD"
    PARAM_STATS = "STATISTICS"
    PARAM_INTERVAL = "INTERVAL"
    PARAM_DELETE_TEMP = "DELETE_TEMP"
    #
    # ---------- QGIS boilerplate ----------
    def name(self):
        return "module3_eacalculation_qgis_V5.1.6"

    def displayName(self):
        return "Module_3_EA calculation (QGIS) – SCS Toolbox v5.1.6"

    def group(self):
        return "SCS Toolbox"

    def groupId(self):
        return "scs_toolbox"

    def shortHelpString(self):
        """
        Short help shown in the Processing Toolbox.
        """
        return (
            "Version 5 Computes erosion/deposition (EA) between successive dates from channel polygons and centrelines.\n"
            "SCS-style island/inactive_floodplain logic, LEFT/RIGHT orientation, optional per-segment rates.\n"
            "Outputs a single GeoPackage with one layer per date-pair."
        )

    def createInstance(self):
        """
        Required factory method for QGIS Processing.
        """
        return Module3_EAcalculation_QGIS()

    def initAlgorithm(self, config=None):
        """
        Define input parameters and options for the Processing dialog.
        """
        # Output folder for final GeoPackage
        self.addParameter(
            QgsProcessingParameterFolderDestination(self.PARAM_OUTFOLDER, "Output folder")
        )

        # Channel polygon layers (one per date)
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.PARAM_CHAN, "Channel polygons (one per date)", QgsProcessing.TypeVectorPolygon
            )
        )

        # Centreline layers (one per date)
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.PARAM_CEN, "Centrelines (one per date)", QgsProcessing.TypeVectorLine
            )
        )

        # Field names for date/year
        self.addParameter(
            QgsProcessingParameterString(self.PARAM_FIELD_YEAR, "Channel year/date field (polygons)")
        )
        self.addParameter(
            QgsProcessingParameterString(self.PARAM_FIELD_CENYEAR, "Centreline year/date field (lines)")
        )

        # Optional statistics segments layer (for rates per segment/polygon)
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.PARAM_STATS, "Statistics/segments layer (optional)", optional=True
            )
        )

        # Segment spacing for normalising area-based rates to linear rates
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_INTERVAL,
                "Interval for rate normalisation",
                QgsProcessingParameterNumber.Double,
                1.0,
            )
        )
        # Digitising tolerance: buffer applied to each channel polygon before EA tagging
        self.PARAM_DIGI_TOL = "DIGITISING_TOLERANCE"
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_DIGI_TOL,
                "Digitising tolerance / mapping gap tolerance (metres)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
                maxValue=100.0
            )
        )

        # Flag to delete intermediates at the end (currently not deeply used here,
        # but kept for UI/theme compatibility)
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.PARAM_DELETE_TEMP, "Delete debug information/intermediates when done", defaultValue=True
            )
        )
        
    # ---------- MAIN ----------
    def processAlgorithm(self, parameters, context, feedback):
        """
        Main execution entry point for the EA module.
        Steps:
          0) Resolve inputs
          1) Parse years and pair polygon/centreline by year
          2) Build hole-free envelopes + island-aware EA layers
          3) For each pair of successive years, compute ArcPro-faithful EA classification
             and orientation masks, then direction/migration and per-pair outputs.
          4) (Optional) intersect with segments and compute EA rates.
        """
        delete_temp = self.parameterAsBool(parameters, self.PARAM_DELETE_TEMP, context)
        DEBUG_SAVE = not delete_temp
        feedback.pushInfo(f"[DEBUG] DEBUG_SAVE = {DEBUG_SAVE}")


        # --- Parameter extraction ---
        outdir = self.parameterAsString(parameters, self.PARAM_OUTFOLDER, context)
        field_year = self.parameterAsString(parameters, self.PARAM_FIELD_YEAR, context)
        field_cen_year = self.parameterAsString(parameters, self.PARAM_FIELD_CENYEAR, context)
        stats = self.parameterAsVectorLayer(parameters, self.PARAM_STATS, context)
        interval = self.parameterAsDouble(parameters, self.PARAM_INTERVAL, context)
        buffer_tol = self.parameterAsDouble(parameters, self.PARAM_DIGI_TOL, context)
        feedback.pushInfo(f"[PARAM] Digitising tolerance = {buffer_tol} m")
        # Read parameter value from GUI
        delete_temp = self.parameterAsBool(parameters, self.PARAM_DELETE_TEMP, context)
        # Debug mode is enabled when user turns off deletion
        DEBUG_SAVE = not delete_temp
        feedback.pushInfo(f"[DEBUG] DEBUG_SAVE = {DEBUG_SAVE}")

       

        _ensure_folder(outdir)   # Make sure output folder exists
        _written_once = set()    # Keep track of which (gpkg, layer) we've written already

        feedback.pushInfo("STEP 0: Resolving inputs...")

        # Resolve multiple polygon and centreline inputs as valid layers
        raw_chans = self.parameterAsLayerList(parameters, self.PARAM_CHAN, context)
        raw_cens  = self.parameterAsLayerList(parameters, self.PARAM_CEN,  context)

        chans, cens = [], []
        # Convert raw channel references to layers, logging result
        for idx, obj in enumerate(raw_chans):
            lyr = _as_layer(obj, context)
            feedback.pushInfo(f"[CHAN {idx}] {obj} → {'OK' if (lyr and lyr.isValid()) else 'INVALID'}")
            if lyr and lyr.isValid():
                chans.append(lyr)
        # Convert raw centreline references to layers, logging result
        for idx, obj in enumerate(raw_cens):
            lyr = _as_layer(obj, context)
            feedback.pushInfo(f"[CEN  {idx}] {obj} → {'OK' if (lyr and lyr.isValid()) else 'INVALID'}")
            if lyr and lyr.isValid():
                cens.append(lyr)

        if not chans or not cens:
            # We need at least one channel and one centreline
            raise QgsProcessingException("Provide at least one channel polygon and one centreline.")

        # ---- STEP 1: Parse years and pair by YEAR ----
        feedback.pushInfo("STEP 1: Read inputs, parse years, and pair by matching year")

        def _year_ymd_from_feat(layer, fieldname):
            """
            Read the first feature from 'layer' and try to parse its date field (fieldname).
            Returns:
                (year_int, 'YYYYMMDD') or (None, None) if unparseable.
            """
            feat = next(layer.getFeatures(), None)
            if feat is None:
                return None, None

            names = layer.fields().names()
            # 1) If specified field exists, try to parse that value
            ymd = _yyyymmdd_from_value_lenient(feat[fieldname]) if fieldname in names else None
            # 2) Fallback to parsing from layer metadata (name/source)
            if not ymd:
                ymd = _yyyymmdd_from_layer_meta(layer)
            if not ymd:
                return None, None

            # Extract year from ymd
            try:
                return int(ymd[:4]), ymd
            except Exception:
                return None, None

        # Build dictionaries keyed by year for polygons and centrelines
        poly_by_year, cen_by_year = {}, {}

        # Populate polygon dictionary
        for ch in chans:
            y, ymd = _year_ymd_from_feat(ch, field_year)
            if y is None:
                feedback.reportError(f"[POLY] No parsable year for {ch.name()}; skipping this layer.")
                continue
            poly_by_year.setdefault(y, {"layer": ch, "ymd": ymd, "name": ch.name()})

        # Populate centreline dictionary
        for ce in cens:
            y, ymd = _year_ymd_from_feat(ce, field_cen_year)
            if y is None:
                feedback.reportError(f"[CEN] No parsable year for {ce.name()}; skipping this layer.")
                continue
            cen_by_year.setdefault(y, {"layer": ce, "ymd": ymd, "name": ce.name()})

        # Years present
        years_poly = sorted(poly_by_year.keys())
        years_cen  = sorted(cen_by_year.keys())
        # Intersection of years available for both polygon and centreline
        common_years = sorted(set(years_poly).intersection(years_cen))

        # Log any mismatches between polygon and centreline years
        miss_cen = sorted(set(years_poly) - set(years_cen))
        miss_poly = sorted(set(years_cen) - set(years_poly))
        if miss_cen:
            feedback.reportError(f"Years present in polygons but missing in centrelines: {miss_cen}")
        if miss_poly:
            feedback.reportError(f"Years present in centrelines but missing in polygons: {miss_poly}")
        if len(common_years) < 2:
            # We need at least two matching years to compute changes
            raise QgsProcessingException("Need at least two matching years to compute E/A between dates.")

        feedback.pushInfo(f"Years (polys): {years_poly}")
        feedback.pushInfo(f"Years (cens) : {years_cen}")
        feedback.pushInfo(f"Common years : {common_years}")

        # ---- Build aligned lists (ascending) + TEMP copies ----
        CH_list, CEN_list = [], []             # polygon and centreline layers, aligned by year
        years_polys, years_cens, date_objs_polys = [], [], []  # ymd strings and date objects

        # For each common year, store a temporary copy of polygon and centreline
        for y in common_years:
            ch_info = poly_by_year[y]
            ce_info = cen_by_year[y]
            feedback.pushInfo(f"Pairing year {y}: {ch_info['name']} ↔ {ce_info['name']}")

            # Save polygon to TEMPORARY_OUTPUT
            ch_temp = _run(
                "native:savefeatures",
                {"INPUT": ch_info["layer"], "OUTPUT": "TEMPORARY_OUTPUT"},
                context, feedback
            )["OUTPUT"]
            # Save centreline to TEMPORARY_OUTPUT
            ce_temp = _run(
                "native:savefeatures",
                {"INPUT": ce_info["layer"], "OUTPUT": "TEMPORARY_OUTPUT"},
                context, feedback
            )["OUTPUT"]

            CH_list.append(ch_temp)
            CEN_list.append(ce_temp)
            years_polys.append(ch_info["ymd"])
            years_cens.append(ce_info["ymd"])
            date_objs_polys.append(datetime.strptime(ch_info["ymd"], "%Y%m%d"))

               # ---- STEP 2: Build hole-free envelopes + island-aware layers ----
        feedback.pushInfo("STEP 2: Build hole-free envelopes and island tags")
        POL_list, ISL_list = [], []  # List of envelopes and envelope+island "EA" base layers

        for i, ch in enumerate(CH_list):
            ch = _as_layer(ch, context)
            if ch is None or not ch.isValid():
                feedback.reportError(f"[STEP2] CH_list[{i}] invalid; skipping.")
                continue

            # Envelope (polygon union) without hollows (holes)
            ch_diss = _run(
                "native:dissolve",
                {
                    "INPUT": ch,
                    "FIELD": [],
                    "SEPARATE_DISJOINT": False,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context, feedback
            )["OUTPUT"]

            ch_env = _run(
                "native:deleteholes",
                {
                    "INPUT": ch_diss,
                    "MIN_AREA": 0.0,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context, feedback
            )["OUTPUT"]

            POL_list.append(ch_env)

            # Date for this index as 'YYYYMMDD'
            ymd = years_polys[i]

            # --------------------------------------------------------
            # Optional digitising tolerance: buffer to fill mapping gaps
            # --------------------------------------------------------
            if buffer_tol > 0.0:
                ch_buf = _run(
                    "native:buffer",
                    {
                        "INPUT": ch,
                        "DISTANCE": buffer_tol,
                        "SEGMENTS": 8,
                        "END_CAP_STYLE": 0,
                        "JOIN_STYLE": 0,
                        "MITER_LIMIT": 2.0,
                        "DISSOLVE": False,
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                    context, feedback
                )["OUTPUT"]
                feedback.pushInfo(
                    f"[STEP2] Applied {buffer_tol} m buffer to channel polygons for {ymd}."
                )
            else:
                ch_buf = ch
                feedback.pushInfo(f"[STEP2] No buffer applied for {ymd} (tolerance = 0).")

            # islands = envelope minus *buffered* channel polygons
            islands = _difference_polys(ch_env, ch_buf, context, feedback)

            # Tag buffered channel polygons with y_YYYYMMDD and T_YYYYMMDD = 'channel'
            chan_tag = _add_const_field(
                ch_buf,
                f"y_{ymd}",
                f"'{ymd}'",
                2,   # string
                16,
                0,
                context,
                feedback
            )
            chan_tag = _add_const_field(
                chan_tag,
                f"T_{ymd}",
                "'channel'",
                2,
                16,
                0,
                context,
                feedback
            )

            # Tag islands with y_YYYYMMDD and T_YYYYMMDD = 'island'
            isl_tag = _add_const_field(
                islands,
                f"y_{ymd}",
                f"'{ymd}'",
                2,
                16,
                0,
                context,
                feedback
            )
            isl_tag = _add_const_field(
                isl_tag,
                f"T_{ymd}",
                "'island'",
                2,
                16,
                0,
                context,
                feedback
            )

            # Merge buffered channel + island into a single "EA" classification source for this date
            EA_tagged = _run(
                "native:mergevectorlayers",
                {
                    "LAYERS": [chan_tag, isl_tag],
                    "CRS": None,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context, feedback
            )["OUTPUT"]

            ISL_list.append(EA_tagged)

            # --- DEBUG_SAVE: write EA_tagged per date as SHP ---
            if DEBUG_SAVE:
                debug_ea_path = os.path.join(outdir, f"EA_tagged_{ymd}.shp")

                # Clean up any old shapefile sidecars
                for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                    p = os.path.splitext(debug_ea_path)[0] + ext
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception as e:
                            feedback.reportError(f"[DEBUG_SAVE] Could not delete {p}: {e}")

                feedback.pushInfo(f"[DEBUG_SAVE] Writing EA_tagged debug: {debug_ea_path}")
                _run(
                    "native:savefeatures",
                    {"INPUT": EA_tagged, "OUTPUT": debug_ea_path},
                    context, feedback
                )


        # Reproject centrelines to match polygon CRS (so singlesidedbuffer etc. behave properly)
        def _reproj(layer, crs_authid):
            return _run(
                "native:reprojectlayer",
                {
                    "INPUT": layer,
                    "TARGET_CRS": crs_authid,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context, feedback
            )["OUTPUT"]

        ref_crs = _as_layer(POL_list[0], context).crs().authid()
        CEN_list = [
            _reproj(ce, ref_crs) if _as_layer(ce, context).crs().authid() != ref_crs else ce
            for ce in CEN_list
        ]

        # Ensure all lists are aligned to same length; truncate if needed
        L_pol, L_cen, L_yp, L_dp = len(POL_list), len(CEN_list), len(years_polys), len(date_objs_polys)
        k = min(L_pol, L_cen, L_yp, L_dp)
        if not (L_pol == L_cen == L_yp == L_dp):
            feedback.reportError(
                f"[PAIRING] Lengths differ: POL={L_pol} CEN={L_cen} years={L_yp} dates={L_dp}; truncating to {k}."
            )
        POL_list, CEN_list = POL_list[:k], CEN_list[:k]
        ISL_list = ISL_list[:k]
        years_polys, years_cens, date_objs_polys = (
            years_polys[:k],
            years_cens[:k],
            date_objs_polys[:k],
        )
        if k < 2:
            raise QgsProcessingException("Not enough aligned dates after filtering (need ≥ 2).")


        # ---- Output container name/path ----
        gpkg_name = f"SCS_EA_{years_polys[0]}_{years_polys[-1]}.gpkg"
        gpkg_path = os.path.join(outdir, gpkg_name)
        feedback.pushInfo(f"Final GeoPackage: {gpkg_path}")

        # Remove existing GPKG if present; we'll recreate it
        if os.path.exists(gpkg_path):
            try:
                os.remove(gpkg_path)
            except Exception as e:
                feedback.reportError(f"[WRITE] Could not delete existing {gpkg_path}: {e}")

        EAprocess_outputs = []  # List of EA_processes layer names written

        # ---------- STEP 3: Pairwise EA (ArcPro-faithful, including islands) ----------
        for i in range(k - 1):
            # y1,y2 are year strings 'YYYYMMDD' from polygon years list
            y1, y2 = years_polys[i], years_polys[i + 1]
            # d1,d2 are Python datetime objects for date spans
            d1, d2 = date_objs_polys[i], date_objs_polys[i + 1]
            # Envelope surfaces and centrelines for older/younger
            pol_old, pol_yng = POL_list[i], POL_list[i + 1]
            cen_old, cen_yng = CEN_list[i], CEN_list[i + 1]
            EA1 = ISL_list[i]       # has y_y1, T_y1 tags
            EA2 = ISL_list[i + 1]   # has y_y2, T_y2 tags

            feedback.pushInfo(f"STEP 3: Pair {y1} → {y2}")

            # Clean envelopes (fix+2D) for each year
            env_old = _force_2d_safe(_fix_geoms(pol_old, context, feedback), context, feedback)
            env_yng = _force_2d_safe(_fix_geoms(pol_yng, context, feedback), context, feedback)

            # If either envelope is empty, skip this pair
            if is_empty_layer(env_old, context, feedback, f"env_old {y1}") or \
               is_empty_layer(env_yng, context, feedback, f"env_yng {y2}"):
                feedback.reportError(f"[PAIRING] Empty envelope for {y1}→{y2}; skipping.")
                continue

            # Build union envelope for both dates (used for orientation, clipping etc.)
            pair_env = _as_layer(
                _union_polys(env_old, env_yng, context, feedback, tag="pair_env"),
                context
            )
            env_union = _as_layer(
                _run(
                    "native:dissolve",
                    {
                        "INPUT": pair_env,
                        "FIELD": [],
                        "SEPARATE_DISJOINT": False,
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                    context, feedback
                )["OUTPUT"],
                context
            )
            # Boundary of the combined envelope (for snapping/extension)
            env_boundary = _as_layer(
                _run(
                    "native:boundary",
                    {"INPUT": env_union, "OUTPUT": "TEMPORARY_OUTPUT"},
                    context, feedback
                )["OUTPUT"],
                context
            )

            # Extend & snap centrelines to boundary (ArcPro-like behaviour)
            def _extend_and_snap_to_boundary(line, boundary, tag):
                """
                Extend line ends by 50 units in both directions, then snap to envelope boundary.
                The aim is to ensure centrelines reach the envelope edges for side classification.
                """
                try:
                    line_ext = _run(
                        "native:extendlines",
                        {
                            "INPUT": line,
                            "START_DISTANCE": 50,
                            "END_DISTANCE": 50,
                            "OUTPUT": "TEMPORARY_OUTPUT",
                        },
                        context, feedback
                    )["OUTPUT"]
                except Exception:
                    line_ext = line

                try:
                    # Use snapgeometries if available, else fallback to snap
                    alg = "native:snapgeometries" if _algo_exists("native:snapgeometries") else "native:snap"
                    line_snap = _run(
                        alg,
                        {
                            "INPUT": line_ext,
                            "REFERENCE_LAYER": boundary,
                            "TOLERANCE": 0.5,
                            "BEHAVIOR": 0,
                            "OUTPUT": "TEMPORARY_OUTPUT",
                        },
                        context, feedback
                    )["OUTPUT"]
                except Exception:
                    if feedback:
                        feedback.pushInfo(
                            f"[snap] snap to boundary not available; using extended only for {tag}."
                        )
                    line_snap = line_ext
                return line_snap

            # Apply extend+snap to both older and younger centrelines
            cen_old = _extend_and_snap_to_boundary(cen_old, env_boundary, "old")
            cen_yng = _extend_and_snap_to_boundary(cen_yng, env_boundary, "young")

            # Prepare EA1 / EA2 as cleaned polygons clipped to env_union
            EA1c = _polys_only(
                _drop_null_tiny_polys(
                    _clip_polys(
                        _force_2d_safe(_fix_geoms(EA1, context, feedback), context, feedback),
                        env_union, context, feedback
                    ),
                    1e-6, context, feedback
                ),
                context, feedback
            )
            EA2c = _polys_only(
                _drop_null_tiny_polys(
                    _clip_polys(
                        _force_2d_safe(_fix_geoms(EA2, context, feedback), context, feedback),
                        env_union, context, feedback
                    ),
                    1e-6, context, feedback
                ),
                context, feedback
            )

            # Debug emptiness (mainly for logging)
            _ = is_empty_layer(EA1c, context, feedback, "EA1c_polys_only")
            _ = is_empty_layer(EA2c, context, feedback, "EA2c_polys_only")

            # --- ArcPro-faithful EA classification (incl. island_* + stable_floodplain) ---

            # 1) Build union of EA1c and EA2c with robust union
            try:
                unionEA = _safe_union(EA1c, EA2c, grid=0.10, context=context, feedback=feedback)
            except Exception as e:
                if feedback:
                    feedback.reportError(
                        f"[EA union] _safe_union raised: {e}; using simple union fallback."
                    )
                unionEA = _union_polys(EA1c, EA2c, context, feedback, tag="unionEA_fallback")

            if is_empty_layer(unionEA, context, feedback, "unionEA"):
                raise QgsProcessingException(
                    "[EA] unionEA is empty after union; check inputs for validity."
                )

            unionEA = _as_layer(unionEA, context)
            # --- DEBUG: save unionEA for this pair ---
            if DEBUG_SAVE:
                debug_union_path = os.path.join(outdir, f"unionEA_{y1}_{y2}.shp")
                for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                    p = os.path.splitext(debug_union_path)[0] + ext
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception as e:
                            feedback.reportError(f"[DEBUG_SAVE] Could not delete {p}: {e}")
                feedback.pushInfo(f"[DEBUG_SAVE] Writing {debug_union_path}")
                _run(
                    "native:savefeatures",
                    {"INPUT": unionEA, "OUTPUT": debug_union_path},
                    context, feedback
                )

            # Check whether EA field already exists (to know if NEW_FIELD should be False)
            has_EA = (unionEA.fields().indexOf("EA") != -1)

            # Expression implementing ArcGIS EA logic, extended with:
            #   - island_erosion / island_deposition
            #   - stable_channel / stable_floodplain
            #   - fallback inactive_floodplain
            ea_expr = f"""
                CASE
                -- 1) erosion: y_y1 != y1 AND T_y2 = 'channel'
                WHEN coalesce("y_{y1}",'') <> '{y1}'
                    AND lower(coalesce("T_{y2}",'')) = 'channel'
                THEN 'erosion'

                -- 2) deposition: y_y2 != y2 AND T_y1 = 'channel'
                WHEN coalesce("y_{y2}",'') <> '{y2}'
                    AND lower(coalesce("T_{y1}",'')) = 'channel'
                THEN 'deposition'

                -- 3) stable_floodplain (former 'hollow'): no membership at either date
                WHEN coalesce("y_{y1}",'') <> '{y1}'
                    AND coalesce("y_{y2}",'') <> '{y2}'
                THEN 'stable_floodplain'

                -- 4) island_erosion: island -> channel
                WHEN lower(coalesce("T_{y1}",'')) = 'island'
                    AND lower(coalesce("T_{y2}",'')) = 'channel'
                THEN 'island_erosion'

                -- 5) island_deposition: channel -> island
                WHEN lower(coalesce("T_{y1}",'')) = 'channel'
                    AND lower(coalesce("T_{y2}",'')) = 'island'
                THEN 'island_deposition'

                -- 6) ArcPro 'stable' bucket, split into channel vs floodplain
                WHEN (
                        lower(coalesce("T_{y1}",'')) = lower(coalesce("T_{y2}",''))
                        OR (coalesce("y_{y1}",'') <> '{y1}' AND lower(coalesce("T_{y2}",'')) = 'island')
                        OR (coalesce("y_{y2}",'') <> '{y2}' AND lower(coalesce("T_{y1}",'')) = 'island')
                )
                THEN CASE
                        WHEN lower(coalesce("T_{y1}",'')) = 'channel'
                            AND lower(coalesce("T_{y2}",'')) = 'channel'
                        THEN 'stable_channel'
                        ELSE 'stable_floodplain'
                    END

                -- 7) fallback
                ELSE 'inactive_floodplain'
                END
                """.strip()

            # Apply EA classification expression
            unionEA = _run(
               "native:fieldcalculator",
                {
                    "INPUT": unionEA,
                    "FIELD_NAME": "EA",
                    "FIELD_TYPE": 2,
                    "FIELD_LENGTH": 32,
                    "FIELD_PRECISION": 0,
                    "NEW_FIELD": not has_EA,
                    "FORMULA": ea_expr,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
               context, feedback
            )["OUTPUT"]
            unionEA = _as_layer(unionEA, context)

            # PeriodEnd field holds the "young" date in a formatted text (YYYY_MM_DD)
            def _ymd8(val):
                """
                Internal helper to coerce a value to 'YYYYMMDD' string where possible.
                """
                s = str(val)
                m = re.search(r"(18|19|20)\d{6}", s)
                if m:
                    return m.group(0)
                m = re.search(r"(18|19|20)\d{2}[-/](\d{2})[-/](\d{2})", s)
                if m:
                    return f"{m.group(1)}{m.group(2)}{m.group(3)}"
                m = re.search(r"(18|19|20)\d{2}", s)
                if m:
                    return f"{m.group(0)}0101"
                return s

            y2_ymd = _ymd8(y2)
            period_end_fmt = (
                f"{y2_ymd[0:4]}_{y2_ymd[4:6]}_{y2_ymd[6:8]}"
                if isinstance(y2_ymd, str) and len(y2_ymd) == 8
                else str(y2_ymd)
            )
            has_period_end = (unionEA.fields().indexOf("PeriodEnd") != -1)
            unionEA = _run(
                "native:fieldcalculator",
                {
                    "INPUT": unionEA,
                    "FIELD_NAME": "PeriodEnd",
                    "FIELD_TYPE": 2,
                    "FIELD_LENGTH": 16,
                    "FIELD_PRECISION": 0,
                    "NEW_FIELD": not has_period_end,
                    "FORMULA": f"'{period_end_fmt}'",
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
               context, feedback
            )["OUTPUT"]
            unionEA = _as_layer(unionEA, context)
            
                        # unionEA already has EA + PeriodEnd at this point.

            # Build true hollows: envelope minus unionEA
            hollows = _difference_polys(env_union, unionEA, context, feedback)

            # Drop tiny slivers if needed
            hollows = _drop_null_tiny_polys(hollows, 1e-6, context, feedback)

            # Tag hollows as stable_floodplain with same PeriodEnd
            hollows = _run(
                "native:fieldcalculator",
                {
                    "INPUT": hollows,
                    "FIELD_NAME": "EA",
                    "FIELD_TYPE": 2,
                    "FIELD_LENGTH": 32,
                    "FIELD_PRECISION": 0,
                    "NEW_FIELD": True,
                    "FORMULA": "'stable_floodplain'",
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context, feedback
            )["OUTPUT"]

            hollows = _run(
                "native:fieldcalculator",
                {
                    "INPUT": hollows,
                    "FIELD_NAME": "PeriodEnd",
                    "FIELD_TYPE": 2,
                    "FIELD_LENGTH": 16,
                    "FIELD_PRECISION": 0,
                    "NEW_FIELD": True,
                    "FORMULA": f"'{period_end_fmt}'",
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context, feedback
            )["OUTPUT"]

            # Merge EA polygons + hollows: this becomes the base for side masks
            baseEA = _run(
                "native:mergevectorlayers",
                {"LAYERS": [unionEA, hollows], "CRS": None, "OUTPUT": "TEMPORARY_OUTPUT"},
                context, feedback
            )["OUTPUT"]


            # ---------- Orientation masks (LEFT/RIGHT) ----------

            # Build cells by polygonizing envelope boundary + centrelines at each date
            cells_old = _run(
                "native:polygonize",
                {
                    "INPUT": _run(
                        "native:mergevectorlayers",
                        {
                            "LAYERS": [env_boundary, cen_old],
                            "CRS": None, "OUTPUT": "TEMPORARY_OUTPUT"
                        },
                        context, feedback
                    )["OUTPUT"],
                    "KEEP_FIELDS": False,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context, feedback
            )["OUTPUT"]

            cells_yng = _run(
                "native:polygonize",
                {
                    "INPUT": _run(
                        "native:mergevectorlayers",
                        {
                            "LAYERS": [env_boundary, cen_yng],
                            "CRS": None, "OUTPUT": "TEMPORARY_OUTPUT"
                        },
                        context, feedback
                    )["OUTPUT"],
                    "KEEP_FIELDS": False,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context, feedback
            )["OUTPUT"]

            # Build single-sided buffers for old centreline (LEFT, RIGHT)
            left_old = _clip_polys(
                _run(
                    "native:singlesidedbuffer",
                    {
                        "INPUT": cen_old,
                        "DISTANCE": 1.0,
                        "SEGMENTS": 8,
                        "END_CAP_STYLE": 0,
                        "JOIN_STYLE": 0,
                        "MITER_LIMIT": 2.0,
                        "SIDE": 0,  # left
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                    context, feedback
                )["OUTPUT"],
                env_union, context, feedback
            )
            right_old = _clip_polys(
                _run(
                    "native:singlesidedbuffer",
                    {
                        "INPUT": cen_old,
                        "DISTANCE": 1.0,
                        "SEGMENTS": 8,
                        "END_CAP_STYLE": 0,
                        "JOIN_STYLE": 0,
                        "MITER_LIMIT": 2.0,
                        "SIDE": 1,  # right
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                    context, feedback
                )["OUTPUT"],
                env_union, context, feedback
            )

            # Add side labels to left/right masks
            left_old  = _add_const_field(left_old,  "SIDEL", "'LEFT'",  2, 10, 0, context, feedback)
            right_old = _add_const_field(right_old, "SIDER", "'RIGHT'", 2, 10, 0, context, feedback)

            # Merge LEFT+RIGHT for "old" year
            lr_old = _run(
                "native:mergevectorlayers",
                {"LAYERS": [left_old, right_old], "CRS": None, "OUTPUT": "TEMPORARY_OUTPUT"},
                context, feedback
            )["OUTPUT"]

            # Join side info onto cells_old
            side_old = _run(
                "native:joinattributesbylocation",
                {
                    "INPUT": cells_old,
                    "JOIN": lr_old,
                    "PREDICATE": [0],  # intersects
                    "JOIN_FIELDS": [],
                    "METHOD": 0,
                    "DISCARD_NONMATCHING": False,
                    "PREFIX": "",
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context, feedback
            )["OUTPUT"]
            side_old = _add_const_field(
                side_old,
                f"S_{y1}",
                "coalesce(\"SIDEL\",'')||coalesce(\"SIDER\",'')",
                2, 20, 0, context, feedback
            )

            # Same process for young year
            left_yng = _clip_polys(
                _run(
                    "native:singlesidedbuffer",
                    {
                        "INPUT": cen_yng,
                        "DISTANCE": 1.0,
                        "SEGMENTS": 8,
                        "END_CAP_STYLE": 0,
                        "JOIN_STYLE": 0,
                        "MITER_LIMIT": 2.0,
                        "SIDE": 0,
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                    context, feedback
                )["OUTPUT"],
                env_union, context, feedback
            )
            right_yng = _clip_polys(
                _run(
                    "native:singlesidedbuffer",
                    {
                        "INPUT": cen_yng,
                        "DISTANCE": 1.0,
                        "SEGMENTS": 8,
                        "END_CAP_STYLE": 0,
                        "JOIN_STYLE": 0,
                        "MITER_LIMIT": 2.0,
                        "SIDE": 1,
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                    context, feedback
                )["OUTPUT"],
                env_union, context, feedback
            )
            left_yng  = _add_const_field(left_yng,  "SIDEL", "'LEFT'",  2, 10, 0, context, feedback)
            right_yng = _add_const_field(right_yng, "SIDER", "'RIGHT'", 2, 10, 0, context, feedback)

            lr_yng = _run(
                "native:mergevectorlayers",
                {"LAYERS": [left_yng, right_yng], "CRS": None, "OUTPUT": "TEMPORARY_OUTPUT"},
                context, feedback
            )["OUTPUT"]

            side_yng = _run(
                "native:joinattributesbylocation",
                {
                    "INPUT": cells_yng,
                    "JOIN": lr_yng,
                    "PREDICATE": [0],  # intersects
                    "JOIN_FIELDS": [],
                    "METHOD": 0,
                    "DISCARD_NONMATCHING": False,
                    "PREFIX": "",
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context, feedback
            )["OUTPUT"]
            side_yng = _add_const_field(
                side_yng,
                f"S_{y2}",
                "coalesce(\"SIDEL\",'')||coalesce(\"SIDER\",'')",
                2, 20, 0, context, feedback
            )

            # Merge side masks (old & young) and then union with EA polygons
            side_mask = _safe_union(side_old, side_yng, grid=0.10, context=context, feedback=feedback)
            unionEAmask = _safe_union(baseEA, side_mask, grid=0.10, context=context, feedback=feedback)

            um = _as_layer(unionEAmask, context)
            if um is None or not um.isValid():
                raise QgsProcessingException("[EA] Cannot seed 'working': unionEAmask invalid.")

            # IMPORTANT: do NOT re-run EA CASE here – EA is already set.
            working = _run(
                "native:savefeatures",
                {"INPUT": um, "OUTPUT": "TEMPORARY_OUTPUT"},
                context, feedback
            )["OUTPUT"]

                
           # ---------- EA classification on unionEAmask ----------
            has_EA = (um.fields().indexOf("EA") != -1)
            um = _run(
                "native:fieldcalculator",
                {
                    "INPUT": um,
                    "FIELD_NAME": "EA",
                    "FIELD_TYPE": 2,
                    "FIELD_LENGTH": 32,
                    "FIELD_PRECISION": 0,
                    "NEW_FIELD": not has_EA,
                    "FORMULA": ea_expr,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context, feedback
            )["OUTPUT"]
            um = _as_layer(um, context)

            # ---------- PeriodEnd on unionEAmask ----------
            y2_ymd = _ymd8(y2)
            period_end_fmt = f"{y2_ymd[0:4]}_{y2_ymd[4:6]}_{y2_ymd[6:8]}"
            has_period_end = (um.fields().indexOf("PeriodEnd") != -1)
            um = _run(
                "native:fieldcalculator",
                {
                    "INPUT": um,
                    "FIELD_NAME": "PeriodEnd",
                    "FIELD_TYPE": 2,
                    "FIELD_LENGTH": 16,
                    "FIELD_PRECISION": 0,
                    "NEW_FIELD": not has_period_end,
                    "FORMULA": f"'{period_end_fmt}'",
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context, feedback
            )["OUTPUT"]
            um = _as_layer(um, context)
     

            # Seed working layer (all subsequent EA operations act on "working")
            working = _run(
                "native:savefeatures",
                {"INPUT": um, "OUTPUT": "TEMPORARY_OUTPUT"},
                context, feedback
            )["OUTPUT"]

            # ---------- Direction (ArcPro-style, adapted) ----------
            # If TRUE, ties for L+R go to LEFT; else assign BOTH
            PREFER_LEFT_ON_TIE = False

            # Determine direction based on side masks:
            #   - For deposition: prefer S_y2 (new bank) then S_y1 if missing
            #   - For erosion / stable_floodplain / inactive_floodplain / island_erosion:
            #       prefer S_y1 (old bank) then S_y2
            #   - If both LEFT and RIGHT appear -> tie; map to LEFT or BOTH.
            dir_expr = (
                "with_variable('pref', "
                "  CASE "
                "    WHEN \"EA\" IN ('deposition','island_deposition') "
                f"      THEN coalesce(\"S_{y2}\", coalesce(\"S_{y1}\",'')) "
                "    WHEN \"EA\" IN ('erosion','stable_floodplain','inactive_floodplain','island_erosion') "
                f"      THEN coalesce(\"S_{y1}\", coalesce(\"S_{y2}\",'')) "
                "    ELSE '' "
                "  END, "
                "  CASE "
                "    WHEN @pref = '' THEN 'in-channel process' "
                "    WHEN regexp_match(@pref, 'LEFT') AND regexp_match(@pref, 'RIGHT') "
                f"      THEN {'\'LEFT\'' if PREFER_LEFT_ON_TIE else '\'BOTH\''} "
                "    WHEN regexp_match(@pref, 'LEFT') THEN 'LEFT' "
                "    WHEN regexp_match(@pref, 'RIGHT') THEN 'RIGHT' "
                "    ELSE 'in-channel process' "
                "  END "
                ")"
            )

            # Add or update 'direction' field
            has_dir = (_as_layer(working, context).fields().indexOf("direction") != -1)
            working = _run(
                "native:fieldcalculator",
                {
                    "INPUT": working,
                    "FIELD_NAME": "direction",
                    "FIELD_TYPE": 2,
                    "FIELD_LENGTH": 24,
                    "FIELD_PRECISION": 0,
                    "NEW_FIELD": not has_dir,
                    "FORMULA": dir_expr,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context, feedback
            )["OUTPUT"]

            # ---------- Migration ----------
            # Combine EA and direction labels into a single 'migration' descriptor
            working = _run(
                "native:fieldcalculator",
                {
                    "INPUT": working,
                    "FIELD_NAME": "migration",
                    "FIELD_TYPE": 2,
                    "FIELD_LENGTH": 32,
                    "FIELD_PRECISION": 0,
                    "NEW_FIELD": True,
                    "FORMULA": (
                        "CASE "
                        "  WHEN \"EA\"='stable_channel' OR coalesce(\"direction\",'') IN ('','in-channel process') "
                        "    THEN 'in-channel process' "
                        "  WHEN \"EA\" IN ('erosion','stable_floodplain','inactive_floodplain','island_erosion') "
                        "    THEN 'erosion_' || \"direction\" "
                        "  WHEN \"EA\" IN ('deposition','island_deposition') "
                        "    THEN 'deposition_' || \"direction\" "
                        "  ELSE 'in-channel process' "
                        "END"
                    ),
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context, feedback
            )["OUTPUT"]

            # ---------- Period & spans ----------
            # period string: 'YYYYMMDD_YYYYMMDD'
            period_str = f"{y1}_{y2}"
            # Number of days between the two dates
            span_days = max(0, (d2 - d1).days if (d1 and d2) else 0)
            # Span in (approximate) years, used for rate per year
            span_years = float(span_days) / 365.0 if span_days > 0 else 0.0

            # Add standard period fields
            working = _add_const_field(working, "period",     f"'{period_str}'", 2, 20, 0, context, feedback)
            working = _add_const_field(working, "span_days",  str(int(span_days)), 1, 10, 0, context, feedback)
            working = _add_const_field(working, "span_years", str(span_years),    0, 20, 6, context, feedback)

            # ---------- Trim fields before dissolve ----------
            # Keep only fields needed for EA_processes (everything else is dropped)
            keep_fields = [
                f"y_{y1}", f"T_{y1}",
                f"y_{y2}", f"T_{y2}",
                "EA", "direction",
                "period", "span_days", "span_years",
                "migration", "PeriodEnd",
            ]
            try:
                lyr = _as_layer(working, context)
                names = lyr.fields().names() if lyr else []
                drop_fields = [n for n in names if (n not in keep_fields and n.lower() != "fid")]
                if drop_fields:
                    working = _run(
                        "native:deletecolumn",
                        {"INPUT": working, "COLUMN": drop_fields, "OUTPUT": "TEMPORARY_OUTPUT"},
                        context, feedback
                    )["OUTPUT"]
                    feedback.pushInfo(f"[EA TRIM] Removed {drop_fields}")
            except Exception as e:
                feedback.reportError(f"[EA TRIM] Keep-list trimming skipped due to: {e}")

            # ---------- Sanity & dissolve ----------
            lyr_work = _as_layer(working, context)
            if lyr_work is None or not lyr_work.isValid():
                feedback.reportError("[EA] Working layer invalid before dissolve; skipping pair.")
                continue
            cnt = lyr_work.featureCount()
            feedback.pushInfo(f"[EA] Working layer ready for dissolve; features={cnt}")
            if cnt == 0:
                feedback.reportError("[EA] Working layer empty; skipping pair.")
                continue

            # Dissolve by EA, direction, and period/spans to get EA_processes polygons
            feedback.pushInfo("[EA] Dissolving to EA_processes…")
            eadiss = _run(
                "native:dissolve",
                {
                    "INPUT": working,
                    "FIELD": ["EA", "direction", "period", "span_days", "span_years", "migration"],
                    "SEPARATE_DISJOINT": False,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context, feedback
            )["OUTPUT"]

            # Final trim for EA_processes fields
            _keep = [
                "EA", "direction", "period", "span_days", "span_years",
                "migration", "PeriodEnd",
                f"y_{y1}", f"T_{y1}", f"y_{y2}", f"T_{y2}",
            ]
            try:
                _names = _as_layer(eadiss, context).fields().names()
                _drop = [n for n in _names if (n not in _keep and n.lower() != "fid")]
                if _drop:
                    eadiss = _run(
                        "native:deletecolumn",
                        {"INPUT": eadiss, "COLUMN": _drop, "OUTPUT": "TEMPORARY_OUTPUT"},
                        context, feedback
                    )["OUTPUT"]
                    feedback.pushInfo(f"[EA TRIM] Removed {_drop}")
            except Exception as e:
                feedback.reportError(f"[EA TRIM] Skipped due to: {e}")

            # Layer naming utilities
            def _fmt_date(val):
                """
                Format a date value into either YYYY_MM_DD (if 8-digit) or YYYY if only year.
                """
                s = str(val)
                m = re.search(r"(18|19|20)\d{6}", s)
                if m:
                    t = m.group(0)
                    return f"{t[0:4]}_{t[4:6]}_{t[6:8]}"
                m4 = re.search(r"(18|19|20)\d{2}", s)
                return m4.group(0) if m4 else s

            formatted_start = _fmt_date(y1)
            formatted_end   = _fmt_date(y2)
            layer_proc = f"EA_processes_{formatted_start}_{formatted_end}"
            feedback.pushInfo(f"[DEBUG] Attempting to write: {gpkg_path} → {layer_proc}")

            # Avoid rewriting the same pair multiple times
            key = (gpkg_path, layer_proc)
            if key in _written_once:
                feedback.pushInfo(f"[WRITE] Skipping duplicate write: {layer_proc}")
                written = gpkg_path
            else:
                written = _write_gpkg(eadiss, gpkg_path, layer_proc, context,
                                      overwrite=True, feedback=feedback)
                if not written:
                    raise QgsProcessingException(
                        f"[WRITE] Failed to write '{layer_proc}' to {gpkg_path}. "
                        "Check file locks/permissions and that the layer has features."
                    )
                _written_once.add(key)
                feedback.pushInfo(f"[WRITE] Wrote '{layer_proc}' → {gpkg_path}")
                EAprocess_outputs.append(layer_proc)

            # ---------- Segments + rates (if stats layer provided) ----------
            if stats:
                # EAsegments_y1_y2: intersection of EA_processes with a segments/stats layer
                seg = _intersect_polys(eadiss, stats, context, feedback)
                seg = _run(
                    "native:multiparttosingleparts",
                    {"INPUT": seg, "OUTPUT": "TEMPORARY_OUTPUT"},
                    context, feedback
                )["OUTPUT"]

                # Remove fid fields to avoid PK conflicts in GPKG
                names = _as_layer(seg, context).fields().names()
                drop = [n for n in names if n.lower() == "fid"]
                if drop:
                    seg = _run(
                        "native:deletecolumn",
                        {"INPUT": seg, "COLUMN": drop, "OUTPUT": "TEMPORARY_OUTPUT"},
                        context, feedback
                    )["OUTPUT"]

                feedback.pushInfo(
                    f"[DEBUG] EAsegments {y1}->{y2} count: {_as_layer(seg, context).featureCount()}"
                )
                _write_gpkg(seg, gpkg_path, f"EAsegments_{y1}_{y2}", context,
                            overwrite=True, feedback=feedback)

                # EA_rate: dissolved by period and migration, then intersected with stats
                diss = _run(
                    "native:dissolve",
                    {
                        "INPUT": eadiss,
                        "FIELD": ["span_days", "migration", "period", "span_years"],
                        "SEPARATE_DISJOINT": False,
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                    context, feedback
                )["OUTPUT"]

                eainter = _intersect_polys(diss, stats, context, feedback)

                # EA_rate_A: area / time (positive for erosion, negative for deposition)
                eainter = _run(
                    "native:fieldcalculator",
                    {
                        "INPUT": eainter,
                        "FIELD_NAME": "EA_rate_A",
                        "FIELD_TYPE": 0,
                        "FIELD_LENGTH": 20,
                        "FIELD_PRECISION": 8,
                        "NEW_FIELD": True,
                        "FORMULA": (
                            "CASE "
                            "WHEN \"migration\"='in-channel process' OR \"span_days\"=0 THEN 0 "
                            "WHEN \"migration\" LIKE 'erosion%' THEN $area / \"span_days\" "
                            "WHEN \"migration\" LIKE 'deposition%' THEN -($area / \"span_days\") "
                            "ELSE 0 END"
                        ),
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                    context, feedback
                )["OUTPUT"]

                # EA_rate_m: area-based rate normalized by interval (m per day per interval)
                eainter = _run(
                    "native:fieldcalculator",
                    {
                        "INPUT": eainter,
                        "FIELD_NAME": "EA_rate_m",
                        "FIELD_TYPE": 0,
                        "FIELD_LENGTH": 20,
                        "FIELD_PRECISION": 8,
                        "NEW_FIELD": True,
                        "FORMULA": (
                            f"CASE WHEN \"EA_rate_A\" IS NULL THEN 0 "
                            f"ELSE \"EA_rate_A\" / {float(interval)} END"
                        ),
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                    context, feedback
                )["OUTPUT"]

                # EA_disp_m: displacement over the full period (EA_rate_m * span_days)
                eainter = _run(
                    "native:fieldcalculator",
                    {
                        "INPUT": eainter,
                        "FIELD_NAME": "EA_disp_m",
                        "FIELD_TYPE": 0,
                        "FIELD_LENGTH": 20,
                        "FIELD_PRECISION": 6,
                        "NEW_FIELD": True,
                        "FORMULA": (
                            "CASE WHEN \"migration\"='in-channel process' THEN 0 "
                            "ELSE \"EA_rate_m\" * \"span_days\" END"
                        ),
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                    context, feedback
                )["OUTPUT"]

                # EA_rate_m_py: per-year rate (EA_rate_m * 365)
                eainter = _run(
                    "native:fieldcalculator",
                    {
                        "INPUT": eainter,
                        "FIELD_NAME": "EA_rate_m_py",
                        "FIELD_TYPE": 0,
                        "FIELD_LENGTH": 20,
                        "FIELD_PRECISION": 6,
                        "NEW_FIELD": True,
                        "FORMULA": (
                            "CASE WHEN \"migration\"='in-channel process' THEN 0 "
                            "ELSE \"EA_rate_m\" * 365.0 END"
                        ),
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                    context, feedback
                )["OUTPUT"]

                # Trim to keep only rate-related fields
                keep_rate = [
                    "migration", "period", "span_days", "span_years",
                    "EA_rate_A", "EA_rate_m", "EA_disp_m", "EA_rate_m_py",
                    "PeriodEnd",
                ]
                names = _as_layer(eainter, context).fields().names()
                drop = [n for n in names if (n not in keep_rate and n.lower() != "fid")]
                if drop:
                    eainter = _run(
                        "native:deletecolumn",
                        {"INPUT": eainter, "COLUMN": drop, "OUTPUT": "TEMPORARY_OUTPUT"},
                        context, feedback
                    )["OUTPUT"]

                feedback.pushInfo(
                    f"[DEBUG] EA_rate {y1}->{y2} count: {_as_layer(eainter, context).featureCount()}"
                )
                _write_gpkg(eainter, gpkg_path, f"EA_rate_{y1}_{y2}", context,
                            overwrite=True, feedback=feedback)

        # Return list of process layers written (for QGIS UI / debugging)
        return {"EA_PROCESSES": EAprocess_outputs}


# QGIS looks for this when loading the script
def classFactory():
    """
    Factory function for QGIS to instantiate the Processing algorithm.
    """
    return Module3_EAcalculation_QGIS()
