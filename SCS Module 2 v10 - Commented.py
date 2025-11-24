# -*- coding: utf-8 -*-

# Module_2_Segmentation (QGIS) — SCS Toolbox 
# v10: use-existing-union, 2025-11-11
# 
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
Module_2_Segmentation (QGIS) — SCS Toolbox (v10: use-existing-union)
QGIS 3.28+ (tested on 3.40/3.44)

What this does
--------------
- Uses *existing* Module 1 outputs:
  • union_channel_noholes (polygon)
  • SegCenterline (line)
  (prefers layers already loaded in the project; else looks in a folder)
- Optional smoothing of centerline (SIMPLIFICATION offset in metres).
- Clips/filters midpoints to union, builds Thiessen polygons, clips to union.
- Transfers ID_SEQ & Distance to segments and writes Segments_{interval}m.gpkg.

Notes
-----
- This module **does not** recreate union_channel_noholes.
- If union_channel_noholes cannot be found in the project or in CENTERLINE_DIR,
  the algorithm raises an error.
"""

from qgis.core import (
    QgsApplication, QgsProcessing, QgsProcessingAlgorithm, QgsProcessingContext,
    QgsProcessingFeedback, QgsProcessingException,
    QgsProcessingParameterFile, QgsProcessingParameterFolderDestination,
    QgsProviderRegistry, QgsProviderSublayerDetails,
    QgsProcessingParameterNumber, QgsProcessingOutputString,
    QgsVectorLayer, QgsWkbTypes
)
from qgis.PyQt.QtCore import QCoreApplication
import processing
import os, re, math


def tr(s):
    """
    Simple translation helper for UI strings.
    """
    return QCoreApplication.translate("SCS_Segmentation", s)


def runp(alg_id, params, context=None, feedback=None, tag=""):
    """
    Wrapper around processing.run:

    - Logs algorithm ID and main parameters to the Processing feedback.
    - Returns the raw `processing.run` result (dict or other).

    `tag` is just an extra label to help identify where in the workflow we are.
    """
    if feedback:
        try:
            # Try to format parameters nicely
            pretty = "; ".join(f"{k}={v}" for k, v in params.items())
            feedback.pushInfo(f"▶ {alg_id}{' ['+tag+']' if tag else ''}: {pretty}")
        except Exception:
            # Fallback if formatting fails for some reason
            feedback.pushInfo(f"▶ {alg_id}{' ['+tag+']' if tag else ''}")
    return processing.run(alg_id, params, context=context, feedback=feedback)


def as_layer(x, name_hint="layer"):
    """
    Normalize different "layer-like" inputs into a valid QgsVectorLayer.

    Accepts:
    - a QgsVectorLayer instance
    - a file path string (GPKG/SHP/etc.)

    Raises QgsProcessingException if the layer cannot be loaded or is invalid.
    """
    # Already a vector layer: just validate
    if isinstance(x, QgsVectorLayer):
        if not x.isValid():
            raise QgsProcessingException(f"{name_hint} is invalid")
        return x

    # String: treat as data source path (OGR)
    if isinstance(x, str):
        lyr = QgsVectorLayer(x, name_hint, "ogr")
        if not lyr.isValid():
            raise QgsProcessingException(f"Failed to load {name_hint}: {x}")
        return lyr

    # Any other type is unsupported
    raise QgsProcessingException(f"Unexpected type for {name_hint}: {type(x)}")


def _log_count(layer, label, feedback=None):
    """
    Convenience helper to log feature counts of intermediate layers.

    Tries to convert `layer` using as_layer() and logs the number of features.
    """
    try:
        c = as_layer(layer, label).featureCount()
        feedback and feedback.pushInfo(f"{label} features: {c}")
    except Exception as e:
        feedback and feedback.pushInfo(f"{label}: count error ({e})")


def _pick_union_noholes(centerline_dir, context, feedback):
    """
    STRICT-ish picker for 'union_channel_noholes' polygon.

    Search order:
    1) Project:
       - Prefer polygon layer with name exactly 'union_channel_noholes' (case-insensitive).
       - Fallback: polygon layer whose data source path contains 'union_channel_noholes.gpkg'.
    2) Folder:
       - Look ONLY for union_channel_noholes.gpkg or union_channel_noholes.shp in CENTERLINE_DIR.
       - For GPKG:
           * First try polygon sublayer named exactly 'union_channel_noholes'.
           * Then polygon sublayer whose name includes both 'union' and 'nohole'.
           * Finally, if there is a single polygon sublayer, use that as last resort.

    Does NOT scan other files (e.g. union_channel_byYear_noholes.gpkg), so it won't
    accidentally pick variants.
    """
    from qgis.core import (
        QgsProject, QgsProviderRegistry, QgsProviderSublayerDetails,
        QgsVectorLayer, QgsWkbTypes, QgsProcessingException
    )
    import os

    target_name = "union_channel_noholes"

    # ---- 1) Project search ----
    exact = None
    by_path = None

    for lyr in QgsProject.instance().mapLayers().values():
        try:
            if not isinstance(lyr, QgsVectorLayer):
                continue
            if QgsWkbTypes.geometryType(lyr.wkbType()) != QgsWkbTypes.PolygonGeometry:
                continue

            nm = lyr.name().lower()
            src = lyr.dataProvider().dataSourceUri().lower()

            # (a) exact name match
            if nm == target_name:
                exact = lyr
                break

            # (b) data source contains union_channel_noholes.gpkg
            if "union_channel_noholes.gpkg" in src:
                by_path = lyr
        except Exception:
            continue

    if exact:
        feedback.pushInfo(f"Using project polygon (exact name): {exact.name()}")
        return exact
    if by_path:
        feedback.pushInfo(f"Using project polygon (by data source): {by_path.name()}")
        return by_path

    # ---- 2) Folder search ----
    if not centerline_dir or not os.path.isdir(centerline_dir):
        raise QgsProcessingException(
            "union_channel_noholes not found in project, and CENTERLINE_DIR is missing/invalid."
        )

    tried = []

    def _iter_poly_sublayers_strict(path):
        """
        From union_channel_noholes.gpkg (or .shp), return best polygon sublayer:

        Priority:
        1) name == union_channel_noholes
        2) name contains 'union' and 'nohole'
        3) single polygon sublayer, if that's all we have
        """
        layers = []
        exact = []
        named = []

        # Multi-layer container: use provider metadata
        try:
            md = QgsProviderRegistry.instance().providerMetadata("ogr")
            details = md.querySublayers(path) or []
            for d in details:
                try:
                    if d.type() != QgsProviderSublayerDetails.Vector:
                        continue
                    if QgsWkbTypes.geometryType(d.wkbType()) != QgsWkbTypes.PolygonGeometry:
                        continue

                    nm = d.name().lower()
                    lyr = QgsVectorLayer(d.uri(), d.name(), d.providerKey())
                    if not lyr.isValid():
                        continue

                    layers.append(lyr)
                    if nm == target_name:
                        exact.append(lyr)
                    elif "union" in nm and "nohole" in nm:
                        named.append(lyr)
                except Exception:
                    continue
        except Exception:
            pass

        # SHP: single polygon layer, but enforce filename
        if not layers and path.lower().endswith(".shp"):
            lyr = QgsVectorLayer(path, os.path.basename(path), "ogr")
            if lyr.isValid() and QgsWkbTypes.geometryType(lyr.wkbType()) == QgsWkbTypes.PolygonGeometry:
                layers.append(lyr)
                nm_file = os.path.splitext(os.path.basename(path))[0].lower()
                if nm_file == target_name:
                    exact.append(lyr)

        # Priority resolution
        if exact:
            return exact[0]
        if named:
            return named[0]
        if len(layers) == 1:
            # last-resort: only one polygon layer in union_channel_noholes.gpkg
            return layers[0]
        return None

    # Only look at union_channel_noholes.gpkg / .shp in the folder
    for fn in ("union_channel_noholes.gpkg", "union_channel_noholes.shp"):
        p = os.path.join(centerline_dir, fn)
        if not os.path.exists(p):
            continue
        tried.append(p)
        chosen = _iter_poly_sublayers_strict(p)
        if chosen is not None:
            feedback.pushInfo(f"Using folder polygon: {os.path.basename(p)} | layer: {chosen.name()}")
            return chosen

    # If we reach here, we looked at the right file(s) but found no acceptable layer
    raise QgsProcessingException(
        "Could not find polygon layer 'union_channel_noholes' in project or folder.\n"
        f"Tried files: {', '.join(os.path.basename(t) for t in tried) if tried else 'none'}"
    )


    def _iter_poly_sublayers(path):
        """Return polygon sublayers named exactly union_channel_noholes."""
        layers = []

        # GPKG / multi-layer containers: use provider metadata
        try:
            md = QgsProviderRegistry.instance().providerMetadata("ogr")
            details = md.querySublayers(path) or []
            for d in details:
                try:
                    if (d.type() == QgsProviderSublayerDetails.Vector and
                        QgsWkbTypes.geometryType(d.wkbType()) == QgsWkbTypes.PolygonGeometry and
                        d.name().lower() == target_name):
                        lyr = QgsVectorLayer(d.uri(), d.name(), d.providerKey())
                        if lyr.isValid():
                            layers.append(lyr)
                except Exception:
                    continue
        except Exception:
            pass

        # SHP: single polygon layer, but enforce exact name
        if not layers and path.lower().endswith(".shp"):
            lyr = QgsVectorLayer(path, os.path.basename(path), "ogr")
            if (lyr.isValid() and
                QgsWkbTypes.geometryType(lyr.wkbType()) == QgsWkbTypes.PolygonGeometry and
                os.path.splitext(os.path.basename(path))[0].lower() == target_name):
                layers.append(lyr)

        return layers

    # Explicit filenames only
    for fn in ("union_channel_noholes.gpkg", "union_channel_noholes.shp"):
        p = os.path.join(centerline_dir, fn)
        if not os.path.exists(p):
            continue
        tried.append(p)
        polys = _iter_poly_sublayers(p)
        if polys:
            chosen = polys[0]
            feedback.pushInfo(f"Using folder polygon: {os.path.basename(p)} | layer: {chosen.name()}")
            return chosen

    # If we reach here, nothing matched exactly union_channel_noholes
    raise QgsProcessingException(
        "Could not find polygon layer 'union_channel_noholes' in project or folder.\n"
        f"Tried files: {', '.join(os.path.basename(t) for t in tried) if tried else 'none'}"
    )

def _pick_centerline(centerline_dir, union_noholes, context, feedback):
    """
    Find a suitable centerline for segmentation (SegCenterline or centro_*).

    Search order:
    1) Project:
       - Prefer a line layer whose name contains 'segcenterline'.
       - Check extent overlap with union_noholes (after reprojecting union if needed).
    2) Folder:
       - Use _pick_centerline_from_folder() if no project match is found.

    Returns:
        A single-parts line layer (multipart -> singleparts done inside).
    """
    from qgis.core import QgsProject

    # --- 1) Look for a suitable centerline already loaded in the project ---
    proj_cands = []
    for lyr in QgsProject.instance().mapLayers().values():
        try:
            # Candidate: line layer with name containing 'segcenterline'
            if (isinstance(lyr, QgsVectorLayer)
                and QgsWkbTypes.geometryType(lyr.wkbType()) == QgsWkbTypes.LineGeometry
                and "segcenterline" in lyr.name().lower()):
                proj_cands.append(lyr)
        except Exception:
            pass

    chosen = None
    if proj_cands:
        # Prefer any candidate whose extent intersects the union polygon
        for c in proj_cands:
            u = union_noholes
            try:
                # Reproject union if needed to compare extents
                if u.crs() != c.crs():
                    u = runp("native:reprojectlayer",
                             {"INPUT": u, "TARGET_CRS": c.crs(),
                              "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
                             context, feedback, "reproj_union_for_choice")["OUTPUT"]
                    u = as_layer(u, "u_reproj_choice")
                if c.extent().intersects(u.extent()):
                    chosen = c
                    break
            except Exception:
                continue
        # If no candidate intersects the union extent, just pick the first by name
        if chosen is None:
            chosen = proj_cands[0]
            feedback.pushWarning(f"No SegCenterline overlapped union; using first by name: {chosen.name()}")

    # --- 2) If no project centerline found, search folder ---
    if chosen is None:
        feedback.pushWarning("No SegCenterline in project — searching folder.")
        chosen = _pick_centerline_from_folder(centerline_dir, union_noholes, context, feedback)

    # Log chosen centerline
    feedback.pushInfo(f"Using centerline: {chosen.name()} (CRS {chosen.crs().authid()})")

    # Ensure single-part lines for subsequent processing
    single = runp("native:multiparttosingleparts",
                  {"INPUT": chosen, "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
                  context, feedback, "to_single_centerline")["OUTPUT"]
    return as_layer(single, "CenterlineSingle")


def _pick_centerline_from_folder(centerline_dir, union_noholes, context, feedback):
    """
    Search CENTERLINE_DIR for a suitable centerline file.

    Priority patterns:
    - SegCenterline*.gpkg / .shp
    - centro_*.gpkg / .shp
    - any *.gpkg / *.shp as fallbacks

    Within files, prefers:
    - names containing 'segcenterline'
    - names starting 'centro_' or containing 'centerline'

    Additionally checks whether the candidate overlaps union_noholes extent.
    """
    import glob

    # File search patterns, ordered by relevance
    pats = [
        os.path.join(centerline_dir, "SegCenterline*.gpkg"),
        os.path.join(centerline_dir, "SegCenterline*.shp"),
        os.path.join(centerline_dir, "centro_*.gpkg"),
        os.path.join(centerline_dir, "centro_*.shp"),
        os.path.join(centerline_dir, "*.gpkg"),
        os.path.join(centerline_dir, "*.shp"),
    ]

    files = []
    # Collect unique files matching the patterns
    for p in pats:
        files.extend(glob.glob(p))
    files = list(dict.fromkeys(files))  # de-duplicate while preserving order

    def _iter_line_layers(path):
        """
        Enumerate line sublayers from a container file (GPKG) or SHP.

        Attempts to use dataProvider().subLayers(), otherwise falls back to treating
        a SHP as a single-layer line file.
        """
        layers = []
        probe = QgsVectorLayer(path, os.path.basename(path), "ogr")
        if not probe.isValid():
            return layers
        try:
            # For multi-layer containers: parse subLayers() strings
            for sl in probe.dataProvider().subLayers():
                subname = None
                if "table='" in sl:
                    subname = sl.split("table='", 1)[1].split("'", 1)[0]
                elif "layername=" in sl:
                    subname = sl.split("layername=", 1)[1].split(" ", 1)[0]
                if not subname:
                    continue
                uri = f"{path}|layername={subname}"
                lyr = QgsVectorLayer(uri, subname, "ogr")
                if lyr.isValid() and QgsWkbTypes.geometryType(lyr.wkbType()) == QgsWkbTypes.LineGeometry:
                    layers.append(lyr)
        except Exception:
            # Legacy fallback: if it's a SHP, treat it as a single line layer
            if path.lower().endswith(".shp"):
                lyr = QgsVectorLayer(path, os.path.basename(path), "ogr")
                if lyr.isValid() and QgsWkbTypes.geometryType(lyr.wkbType()) == QgsWkbTypes.LineGeometry:
                    layers.append(lyr)
        return layers

    # Collect candidates with a simple priority (lower is better)
    cands = []
    for f in files:
        for lyr in _iter_line_layers(f):
            nm = lyr.name().lower()
            pr = 2  # default (lowest) priority
            if "segcenterline" in nm:
                pr = 0  # best: explicit SegCenterline
            elif nm.startswith("centro_") or "centerline" in nm:
                pr = 1  # second best: centro_ or generic centerline
            cands.append((pr, lyr))

    if not cands:
        raise QgsProcessingException(f"No line layers found in CENTERLINE_DIR: {centerline_dir}")

    # Sort by priority (0, then 1, then 2)
    cands.sort(key=lambda t: t[0])
    chosen = None

    # Prefer candidates whose extent overlaps union_noholes
    for _, lyr in cands:
        try:
            u = union_noholes
            if u.crs() != lyr.crs():
                u = runp("native:reprojectlayer",
                         {"INPUT": u, "TARGET_CRS": lyr.crs(), "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
                         context, feedback, "reproj_union_for_folder_pick")["OUTPUT"]
                u = as_layer(u, "u_for_folder_pick")
            if lyr.extent().intersects(u.extent()):
                chosen = lyr
                break
        except Exception:
            continue

    # If none overlaps, take the "best named" candidate
    if chosen is None:
        chosen = cands[0][1]
        feedback.pushWarning(f"No folder centerline overlapped union extent; using best name match: {chosen.name()}")

    return chosen


# ---------- Geometry helpers ----------

def _optional_smooth(center, simplification, context, feedback):
    """
    Optionally smooth the centerline geometry.

    Uses native:smoothgeometry and native:chaikinsmoothing with fallbacks to
    handle QGIS 3.40 quirks around OFFSET.

    Parameters
    ----------
    center : QgsVectorLayer
        Input line layer.
    simplification : float
        Smoothing offset in map units (metres in our context).
    """
    # No smoothing requested or zero/negative offset
    if not simplification or float(simplification) <= 0:
        return as_layer(center, "Centerline")

    from qgis.core import QgsProcessing

    s = float(simplification)
    # Base parameter set (OFFSET will be added as needed)
    params_base = {
        "INPUT": center,
        "ITERATIONS": 1,
        "MAX_ANGLE": 180.0,
        "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
    }

    # Try 1 — native:smoothgeometry with simple OFFSET
    try:
        out = runp(
            "native:smoothgeometry",
            dict(params_base, **{"OFFSET": s}),
            context,
            feedback,
            "smooth",
        )["OUTPUT"]
        return as_layer(out, "CenterlineSmoothed")
    except Exception as e1:
        feedback.pushWarning(f"native:smoothgeometry (plain) failed: {e1}")

    # Try 2 — same algo but with explicit OFFSET_UNIT parameter (used by newer QGIS)
    try:
        out = runp(
            "native:smoothgeometry",
            dict(params_base, **{"OFFSET": s, "OFFSET_UNIT": 0}),
            context,
            feedback,
            "smooth_unit",
        )["OUTPUT"]
        return as_layer(out, "CenterlineSmoothed")
    except Exception as e2:
        feedback.pushWarning(f"native:smoothgeometry (with OFFSET_UNIT) failed: {e2}")

    # Try 3 — Chaikin smoothing via native:chaikinsmoothing (QGIS 3.40+)
    try:
        out = runp(
            "native:chaikinsmoothing",
            {
                "INPUT": center,
                "ITERATIONS": 1,
                "OFFSET": s,
                "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
            },
            context,
            feedback,
            "chaikin",
        )["OUTPUT"]
        return as_layer(out, "CenterlineSmoothed")
    except Exception as e3:
        feedback.reportError(
            f"All smoothing fallbacks failed; proceeding without smoothing. Last error: {e3}"
        )

    # Absolute fallback — return the original centerline
    return as_layer(center, "Centerline")



def _ensure_lines_projected(center, context, feedback):
    """
    Ensure centerline is:
    - In a projected CRS (units in metres)
    - Purely line geometry
    - Single-part

    Raises QgsProcessingException if CRS is geographic or invalid.
    """
    # Must be projected (not lat/long degrees)
    if center.crs().isGeographic() or not center.crs().isValid():
        raise QgsProcessingException(
            f"Centerline CRS must be projected in metres, got {center.crs().authid() or 'unknown'}"
        )

    # If the source has mixed geometry, extract only line geometries
    if QgsWkbTypes.geometryType(center.wkbType()) != QgsWkbTypes.LineGeometry:
        only_lines = runp(
            "native:extractbygeometrytype",
            {"INPUT": center, "GEOMETRY": 1, "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
            context, feedback, "only_lines"
        )["OUTPUT"]
        center = as_layer(only_lines, "CenterlineLines")

    # Convert multipart lines to single-part
    single = runp(
        "native:multiparttosingleparts",
        {"INPUT": center, "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
        context, feedback, "to_single_again"
    )["OUTPUT"]
    return as_layer(single, "CenterlineSingle")


def _points_mid_interval(center_for_points, interval, context, feedback, tag="points_along"):
    """
    Generate mid-interval points along the centerline:

    - Uses native:pointsalonglines with a start offset = interval / 2.
    - Adds:
        SEQ: auto-increment index
        ID_SEQ: reversed index (to match ArcGIS convention: higher ID upstream)
        Distance: constant distance in metres (segment length)

    Returns:
        QgsVectorLayer of points with attributes {SEQ, ID_SEQ, Distance}.
    """
    # Create points along centerline
    pts = runp(
        "native:pointsalonglines",
        {"INPUT": center_for_points, "DISTANCE": float(interval),
         "START_OFFSET": float(interval) / 2.0, "END_OFFSET": 0.0,
         "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
        context, feedback, tag
    )["OUTPUT"]

    # Add SEQ auto-increment
    pts_seq = runp(
        "native:addautoincrementalfield",
        {"INPUT": pts, "FIELD_NAME": "SEQ", "START": 0,
         "GROUP_FIELDS": [], "SORT_EXPRESSION": "", "SORT_ASCENDING": True,
         "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
        context, feedback, f"{tag}_seq"
    )["OUTPUT"]

    # Create ID_SEQ as (maximum SEQ - SEQ) to reverse the index direction
    pts_id = runp(
        "native:fieldcalculator",
        {"INPUT": pts_seq, "FIELD_NAME": "ID_SEQ", "FIELD_TYPE": 1,
         "FIELD_LENGTH": 10, "FIELD_PRECISION": 0, "NEW_FIELD": True,
         "FORMULA": 'maximum("SEQ") - "SEQ"',
         "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
        context, feedback, f"{tag}_idseq"
    )["OUTPUT"]

    # Distance field, integer approximation of the requested interval
    pts_dist = runp(
        "native:fieldcalculator",
        {"INPUT": pts_id, "FIELD_NAME": "Distance", "FIELD_TYPE": 1,
         "FIELD_LENGTH": 10, "FIELD_PRECISION": 0, "NEW_FIELD": True,
         "FORMULA": int(round(float(interval))),
         "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
        context, feedback, f"{tag}_dist"
    )["OUTPUT"]

    return as_layer(pts_dist, "Midpoints")


def _prep_points_against_union(midpts, union_poly, interval, context, feedback):
    """
    Filter midpoint points so only those within a modest buffer around the union
    polygon (channel zone) are kept.

    Steps:
    - Reproject midpoints to union CRS if required.
    - Build a small buffer around union (distance = 0.25 * interval, constrained).
    - Keep only midpoints that INTERSECT this buffered union.
    - Build spatial indexes where possible.
    - Return filtered midpoints and a debug dict summarizing CRS and counts.
    """
    u = as_layer(union_poly, "u")
    p = as_layer(midpts, "p")

    # Track the original CRS of points for debug info
    crs_before = p.crs().authid() or "unknown"

    # Reproject points to union CRS if needed
    if p.crs() != u.crs():
        p = as_layer(
            runp("native:reprojectlayer",
                 {"INPUT": p, "TARGET_CRS": u.crs(), "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
                 context, feedback, "midpts_to_union_crs")["OUTPUT"],
            "midpts_union_crs"
        )

    # Determine a reasonable buffer distance around union
    try:
        buf_dist = max(0.5, min(10.0, float(interval) * 0.25))
    except Exception:
        buf_dist = 2.0

    # Small buffer around union polygon
    ubuf = runp(
        "native:buffer",
        {"INPUT": u, "DISTANCE": float(buf_dist), "SEGMENTS": 5, "DISSOLVE": True,
         "END_CAP_STYLE": 0, "JOIN_STYLE": 0, "MITER_LIMIT": 2.0,
         "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
        context, feedback, "union_small_plus_buffer"
    )["OUTPUT"]

    # Keep only midpoints in (INTERSECTS) this buffer
    pin = runp(
        "native:extractbylocation",
        {"INPUT": p, "PREDICATE": [0], "INTERSECT": ubuf, "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
        context, feedback, "midpts_in_union_buf"
    )["OUTPUT"]

    # Try to build spatial indexes (non-fatal if it fails)
    try:
        runp("native:createspatialindex", {"INPUT": u}, context, feedback, "sidx_union")
        runp("native:createspatialindex", {"INPUT": p}, context, feedback, "sidx_midpts_all")
        runp("native:createspatialindex", {"INPUT": pin}, context, feedback, "sidx_midpts_in")
    except Exception:
        pass

    # Prepare debug info
    dbg = {
        "crs_u": u.crs().authid() or "unknown",
        "crs_p_before": crs_before,
        "crs_p_after": p.crs().authid() or "unknown",
        "count_midpts_all": as_layer(p, "p_chk").featureCount(),
        "count_midpts_in": as_layer(pin, "pin_chk").featureCount(),
        "buf_dist": buf_dist,
    }

    return as_layer(pin, "MidpointsIn"), dbg


def _add_guard_points(points, union_poly, context, feedback):
    """
    Add four "guard" points far outside the union extent to ensure Voronoi
    cells close off properly around the study area.

    Guards are placed at the corners of an expanded bounding box around the union.
    Then, a merged point layer (midpoints + guards) is returned.
    """
    from qgis.core import QgsField, QgsFeature, QgsGeometry, QgsPointXY, QgsVectorLayer
    from qgis.PyQt.QtCore import QVariant

    u = as_layer(union_poly, "u")
    p = as_layer(points, "pts")

    # Reproject points to union CRS if necessary
    if p.crs() != u.crs():
        p = as_layer(
            runp("native:reprojectlayer",
                 {"INPUT": p, "TARGET_CRS": u.crs(), "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
                 context, feedback, "reproj_pts_to_union"
            )["OUTPUT"],
            "pts_reproj"
        )

    # Expand the union extent by 50% in each direction to place guard points
    ext = u.extent()
    dx, dy = ext.width(), ext.height()
    pad = max(dx, dy) * 0.5
    guards = [
        (ext.xMinimum() - pad, ext.yMinimum() - pad),
        (ext.xMinimum() - pad, ext.yMaximum() + pad),
        (ext.xMaximum() + pad, ext.yMinimum() - pad),
        (ext.xMaximum() + pad, ext.yMaximum() + pad),
    ]

    # Memory point layer to store guards
    mem_uri = f"Point?crs={u.crs().authid()}" if u.crs().authid() else "Point"
    guard_layer = QgsVectorLayer(mem_uri, "guards", "memory")
    if not guard_layer.isValid():
        raise QgsProcessingException("Failed to create guard points layer.")

    # Add attribute 'g' for guard ID
    pr = guard_layer.dataProvider()
    pr.addAttributes([QgsField("g", QVariant.Int)])
    guard_layer.updateFields()

    # Add all guard points to memory layer
    feats = []
    for i, (x, y) in enumerate(guards):
        f = QgsFeature(guard_layer.fields())
        f.setAttributes([i])
        f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
        feats.append(f)
    pr.addFeatures(feats)
    guard_layer.updateExtents()

    # Merge midpoints and guard points into one layer
    merged_pts = runp(
        "native:mergevectorlayers",
        {"LAYERS": [p, guard_layer], "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
        context, feedback, "guards_merge"
    )["OUTPUT"]
    return as_layer(merged_pts, "PointsWithGuards")


def _thiessen_from_points(points, union_poly, context, feedback):
    """
    Build Thiessen (Voronoi) polygons from points and clip them to union_poly.

    Steps:
    - Reproject points to union CRS if needed.
    - Calculate an appropriate buffer margin around union for Voronoi generation.
    - Run qgis:voronoipolygons on all points (midpoints + guards).
    - Zero-buffer the union to heal small geometry issues.
    - Clip Voronoi polygons to the healed union polygon.

    Returns:
        QgsVectorLayer of clipped Voronoi polygons (segments).
    """
    u = as_layer(union_poly, "u")
    p = as_layer(points, "pts")

    # CRS alignment
    if p.crs() != u.crs():
        p = as_layer(
            runp("native:reprojectlayer",
                 {"INPUT": p, "TARGET_CRS": u.crs(), "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
                 context, feedback, "reproj_pts_to_union2"
            )["OUTPUT"],
            "pts_reproj2"
        )

    # Compute a reasonable margin based on union extent or point extent
    ext = u.extent()
    mx = max(ext.width(), ext.height())
    if not (math.isfinite(mx) and mx > 0):
        pext = p.extent()
        diag = math.hypot(pext.width(), pext.height())
        mx = diag if (math.isfinite(diag) and diag > 0) else 1000.0
    margin = max(1000.0, mx * 1.5)
    feedback and feedback.pushInfo(f"Voronoi margin = {margin:.2f} (units: {u.crs().authid() or 'unknown'})")

    # Build Voronoi polygons from points
    vor = runp("qgis:voronoipolygons",
               {"INPUT": p, "BUFFER": float(margin), "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
               context, feedback, "voronoi")["OUTPUT"]

    # Attempt to build spatial index on Voronoi polygons (non-critical)
    try:
        runp("native:createspatialindex", {"INPUT": vor}, context, feedback, "sidx_vor")
    except Exception:
        pass

    # Heal union polygon via buffer(0) to avoid slivers / self-intersections
    union_healed = runp(
        "native:buffer",
        {"INPUT": u, "DISTANCE": 0.0, "SEGMENTS": 5, "DISSOLVE": True,
         "END_CAP_STYLE": 0, "JOIN_STYLE": 0, "MITER_LIMIT": 2.0,
         "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
        context, feedback, "zero_buffer_union"
    )["OUTPUT"]

    # Clip Voronoi polygons to healed union polygon
    clipped = runp(
        "native:clip",
        {"INPUT": vor, "OVERLAY": union_healed, "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
        context, feedback, "clip_voronoi"
    )["OUTPUT"]

    return as_layer(clipped, "SegmentsRaw")


def _save_gpkg(layer, out_path, context, feedback):
    """
    Save the given layer to a GeoPackage at `out_path` with:

    - layer name: "Segments"
    - spatial index enabled

    Existing files at out_path are removed if possible; if not, QGIS still
    attempts to overwrite.
    """
    parent = os.path.dirname(out_path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    lyr = as_layer(layer, "final")

    # Try to remove existing file to avoid stale layers
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
    except Exception as e:
        feedback and feedback.pushInfo(f"Could not remove existing file (will try to overwrite): {e}")

    return runp(
        "native:savefeatures",
        {
            "INPUT": lyr,
            "OUTPUT": out_path,
            "LAYER_NAME": "Segments",                 # GPKG sublayer name
            "DATASOURCE_OPTIONS": [],                 # extra GDAL ds options (none)
            "LAYER_OPTIONS": ["SPATIAL_INDEX=YES"],   # ensure spatial index
        },
        context, feedback, "save_final"
    )["OUTPUT"]


# ---------- Algorithm ----------

class Module2SegmentationQGIS_UseExistingUnion(QgsProcessingAlgorithm):
    """
    QGIS Processing algorithm implementing Module 2 (Segmentation) using:

    - union_channel_noholes (polygon) from Module 1
    - SegCenterline (or equivalent) from Module 1

    It:
    - generates regularly spaced midpoints along the centerline,
    - builds Voronoi (Thiessen) polygons from those points,
    - clips them to union_channel_noholes,
    - and transfers ID_SEQ and Distance attributes to create Segments.
    """

    # Basic metadata / identifiers
    def name(self): return "module2_segmentation_qgis_v10_existing_union"
    def displayName(self): return "Module_2_Segmentation (QGIS) — v10 (use existing union)"
    def group(self): return "SCS Toolbox"
    def groupId(self): return "scs_toolbox"

    def shortHelpString(self):
        """
        One-line help string shown in the Processing Toolbox.
        """
        return tr(
            "Uses existing Module 1 outputs (union_channel_noholes + SegCenterline) to build segments.\n"
            "Search order: project layers first, then CENTERLINE_DIR. No union rebuild."
        )

    def createInstance(self):
        """
        Required factory method so QGIS can create a new instance of this algorithm.
        """
        return Module2SegmentationQGIS_UseExistingUnion()

    def initAlgorithm(self, _config=None):
        """
        Define input parameters and outputs for the Processing framework.
        """
        # Folder containing union_channel_noholes + SegCenterline / centro_* outputs
        self.addParameter(QgsProcessingParameterFile(
            "CENTERLINE_DIR", tr("Folder containing Module 1 outputs"),
            behavior=QgsProcessingParameterFile.Folder, defaultValue=""
        ))

        # Requested segment length in map units (metres)
        self.addParameter(QgsProcessingParameterNumber(
            "INTERVAL", tr("Segment length (m)"),
            type=QgsProcessingParameterNumber.Double, defaultValue=50.0, minValue=0.1
        ))

        # Optional smoothing offset for the centerline
        self.addParameter(QgsProcessingParameterNumber(
            "SIMPLIFICATION", tr("Optional centerline smoothing offset (m)"),
            type=QgsProcessingParameterNumber.Double, defaultValue=0.0, minValue=0.0
        ))

        # Output folder for Segments_Xm.gpkg
        self.addParameter(QgsProcessingParameterFolderDestination(
            "OUTPUT_FOLDER", tr("Output folder")
        ))

        # String output: path to final segments GPKG
        self.addOutput(QgsProcessingOutputString("OUTPUT", tr("Segments output path")))

    def processAlgorithm(self, parameters, context, feedback):
        """
        Main algorithm body.

        Workflow:
        1) Find union_channel_noholes (polygon) via project or folder.
        2) Find SegCenterline / centro_ (line) via project or folder.
        3) Optionally smooth centerline and ensure projected single-part lines.
        4) Clip centerline to small buffer of union and generate midpoints.
        5) Filter midpoints to union buffer.
        6) Add guard points and build Voronoi polygons; clip to union.
        7) Join midpoints (ID_SEQ, Distance) to polygons.
        8) Keep only Distance and ID_SEQ fields.
        9) Save results to Segments_{interval}m.gpkg.
        """
        # ----- Parse parameters -----
        cdir     = self.parameterAsFile(parameters, "CENTERLINE_DIR", context) or ""
        out_dir  = self.parameterAsString(parameters, "OUTPUT_FOLDER", context) or ""
        interval = float(self.parameterAsDouble(parameters, "INTERVAL", context))
        simpl    = float(self.parameterAsDouble(parameters, "SIMPLIFICATION", context))

        feedback.pushInfo(f"Segment interval = {interval} (map units)")

        # 1) Get existing union_channel_noholes (polygon) from project or folder
        union_noholes = _pick_union_noholes(cdir, context, feedback)
        if QgsWkbTypes.geometryType(union_noholes.wkbType()) != QgsWkbTypes.PolygonGeometry:
            raise QgsProcessingException("union_channel_noholes is not a polygon layer.")

        # 2) Get SegCenterline (or equivalent) → optional smoothing → ensure projected/lines-only
        center = _pick_centerline(cdir, union_noholes, context, feedback)
        center = _optional_smooth(center, simpl, context, feedback)
        center = _ensure_lines_projected(center, context, feedback)

        # Align union CRS to centerline CRS (centerline CRS is authoritative)
        if union_noholes.crs() != center.crs():
            union_noholes = as_layer(
                runp("native:reprojectlayer",
                     {"INPUT": union_noholes, "TARGET_CRS": center.crs(),
                      "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
                     context, feedback, "reproject_union")["OUTPUT"],
                "UnionNoHoles_reproj"
            )

        # 3) Clip centerline to a small buffer around union to avoid stray segments
        try:
            # Buffer distance is 10% of interval, clipped to [0.25, 5] units
            eps_clip = max(0.25, min(5.0, float(interval) * 0.10))
        except Exception:
            eps_clip = 2.0

        union_eps = runp(
            "native:buffer",
            {"INPUT": union_noholes, "DISTANCE": float(eps_clip), "SEGMENTS": 5,
             "DISSOLVE": True, "END_CAP_STYLE": 0, "JOIN_STYLE": 0, "MITER_LIMIT": 2.0,
             "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
            context, feedback, "union_eps_for_center_clip"
        )["OUTPUT"]

        center_on_union = runp(
            "native:extractbylocation",
            {"INPUT": center, "PREDICATE": [0], "INTERSECT": union_eps,
             "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
            context, feedback, "center_on_union"
        )["OUTPUT"]
        cnt_co = as_layer(center_on_union, "center_on_union_chk").featureCount()
        feedback.pushInfo(f"Centerline-on-union features: {cnt_co}")

        # 4) Midpoints: either from clipped centerline or from full centerline (fallback)
        if cnt_co == 0:
            feedback.pushWarning(
                "No centerline overlap with union (even with tolerant buffer). "
                "Continuing with FULL centerline → midpoints → filter by union buffer."
            )
            center_for_points = as_layer(center, "center_full_for_points")
            midpts = _points_mid_interval(center_for_points, interval, context, feedback,
                                          tag="points_along_FORCED")
        else:
            feedback.pushInfo("Using *clipped* centerline for midpoint generation.")
            center_for_points = as_layer(center_on_union, "center_on_union_for_points")
            midpts = _points_mid_interval(center_for_points, interval, context, feedback,
                                          tag="points_along_CLIPPED")

        _log_count(midpts, "Midpoints", feedback)

        # 5) Filter midpoints by union buffer (to ensure they are actually near channel zone)
        midpts_in, dbg = _prep_points_against_union(midpts, union_noholes, interval, context, feedback)
        feedback.pushInfo(
            "Midpoints kept after union-buffer filter: "
            f"{dbg.get('count_midpts_in', 'NA')} (buffer {dbg.get('buf_dist', 'NA')} in {dbg.get('crs_u', 'unknown')}; "
            f"points CRS {dbg.get('crs_p_before', 'unknown')}→{dbg.get('crs_p_after', 'unknown')})"
        )
        if dbg.get("count_midpts_in", 0) == 0:
            raise QgsProcessingException(
                "No midpoints fall within (or near) the union polygon. Verify inputs and CRS."
            )

        # 6) Add guard points, build Voronoi polygons, and clip to union (segments)
        midpts_wg = _add_guard_points(midpts_in, union_noholes, context, feedback)
        seg_raw = _thiessen_from_points(midpts_wg, union_noholes, context, feedback)
        _log_count(seg_raw, "Voronoi_clipped", feedback)
        if as_layer(seg_raw, "seg_raw_chk").featureCount() == 0:
            raise QgsProcessingException(
                "Voronoi clipped to union produced 0 features. "
                "Midpoints likely outside union. Verify inputs/CRS."
            )

        # 7) Join ID_SEQ & Distance from midpoints to Voronoi polygons
        seg_joined = runp(
            "native:joinattributesbylocation",
            {"INPUT": seg_raw, "PREDICATE": [1],  # CONTAINS (segments contain midpoints)
             "JOIN": midpts_in, "JOIN_FIELDS": ["ID_SEQ", "Distance"],
             "METHOD": 0, "DISCARD_NONMATCHING": False, "PREFIX": "",
             "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
            context, feedback, "join_seq_contains"
        )["OUTPUT"]

        # If the join produced no features (or no attributes), try buffered fallback joins
        if as_layer(seg_joined, "chk_join_contains").featureCount() == 0:
            # Tiny buffer around midpoints to relax spatial join
            try:
                buf_dist = max(0.25, min(5.0, float(interval) * 0.02))
            except Exception:
                buf_dist = 1.0

            midpts_buf = runp(
                "native:buffer",
                {"INPUT": midpts_in, "DISTANCE": float(buf_dist), "SEGMENTS": 5, "DISSOLVE": False,
                 "END_CAP_STYLE": 0, "JOIN_STYLE": 0, "MITER_LIMIT": 2.0,
                 "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
                context, feedback, "midpts_buffer"
            )["OUTPUT"]

            joined_ok = False
            # Attempt several predicates in order of likely usefulness
            for pred, tag in [([0], "intersects"), ([6], "within"), ([1], "contains")]:
                try:
                    seg_try = runp(
                        "native:joinattributesbylocation",
                        {"INPUT": seg_raw, "PREDICATE": pred, "JOIN": midpts_buf,
                         "JOIN_FIELDS": ["ID_SEQ", "Distance"], "METHOD": 0,
                         "DISCARD_NONMATCHING": False, "PREFIX": "",
                         "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
                        context, feedback, f"join_seq_fallback_{tag}"
                    )["OUTPUT"]
                    if as_layer(seg_try, f"chk_{tag}").featureCount() > 0:
                        seg_joined = seg_try
                        joined_ok = True
                        break
                except Exception:
                    continue

            if not joined_ok:
                raise QgsProcessingException(
                    "Could not transfer ID_SEQ/Distance from midpoints to segments via spatial join."
                )

        # 8) Keep only Distance and ID_SEQ (match ArcGIS attribute scheme)
        final_fields = runp(
            "native:refactorfields",
            {
                "INPUT": seg_joined,
                "FIELDS_MAPPING": [
                    # Distance: integer, length 10, precision 0
                    {"expression": '"Distance"', "name": "Distance", "type": 1, "length": 10, "precision": 0},
                    # ID_SEQ: integer, length 10, precision 0
                    {"expression": '"ID_SEQ"',  "name": "ID_SEQ",  "type": 1, "length": 10, "precision": 0},
                ],
                "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT
            },
            context, feedback, "refactor_keep_arcgis_fields"
        )["OUTPUT"]

        # 9) Save final layer to Segments_{interval}m.gpkg in OUTPUT_FOLDER (or CENTERLINE_DIR)
        if not out_dir:
            # Fallback: if no OUTPUT_FOLDER, try using CENTERLINE_DIR (or home)
            out_dir = cdir if cdir else os.path.expanduser("~")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"Segments_{int(round(interval))}m.gpkg")

        _save_gpkg(final_fields, out_path, context, feedback)

        # Return path to final segments GPKG
        return {"OUTPUT": out_path}
