# -*- coding: utf-8 -*-
# Module 4a — Floodplain Assemblage + HACH/CHM Stats (QGIS / SCS Toolbox)
# 
# 
# Based on Module 4 from SCS Toolbox v2.3 (ArcGIS version) by Milos Rusnak et al.
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
Module 4a — Floodplain Assemblage + HACH/CHM Stats (QGIS / SCS Toolbox)
v2.3 — 2025-11-20

Author: Martin Dawson using ChatGPT-5.1 (QGIS Processing framework).
Based on: SCS Toolbox by Milos Rusnak (ArcGIS original).

What this module does
---------------------
INPUTS:
- Up to 15 EA polygon layers (Module 3 outputs), one per period.
- Each EA layer has:
    • a class field (e.g. 'EA' or 'EA_Class')
    • a period field (e.g. '20190101_20210101' or an end-date)
- Optionally:
    • DEM (for HACH & CHM)
    • DSM (for CHM)
    • Flow path line (thalweg/centreline)
    • Segments polygon layer (Module 2 segmentation)

PART 1 — Floodplain Assemblage (EA)
-----------------------------------
1. Parse and order EA layers by **end date** (oldest → newest).
2. For each layer i, build a minimal schema:
      EA_Class_i   (string)
      Period_i     (string 'yyyy_mm_dd')
3. Use a "newest-wins" overwrite:
      - difference(older, newer) + merge(older_minus_newer, newer)
   so later periods replace earlier ones spatially.
4. Guarantee newest period is fully retained (merge in any missing slivers).
5. Compute:
      EA_Class_final = coalesce(EA_Class_n..EA_Class_1)
      Period_final   = coalesce(Period_n..Period_1)
   and add:
      Year = to_int(substr(Period_final, 1, 4))
6. Save composite as a GeoPackage with a single layer:
      <output>.gpkg | layername="Depositional_Composition"

PART 2 — HACH (height above channel) & CHM (canopy height)
----------------------------------------------------------
If DEM is provided, the module:

A. Builds AOI & clips DEM
   - Buffer around the EA assemblage (≈10× cell size).
   - Clip DEM to AOI → DEM_clipped_m5.tif.

B. If DEM + FLOW present: HACH (DED)
   - Sample DEM_clipped along FLOW at ≈5× pixel spacing.
   - Build IDW trend surface with gdal_grid (CLI if available; provider+warp fallback).
   - Compute DED_m5.tif = DEM_clipped_m5.tif – trend_idw_m5.tif

C. If DEM + DSM + FLOW present: CHM
   - CHM_raw_m5 = DSM – DEM_clipped_m5.
   - veget_CHM_m5.tif = CHM_raw_m5 with values <= 0 set to -9999.

PART 3 — Segments × Assemblage statistics (optional)
----------------------------------------------------
If Segments are provided:
1. Intersect EA assemblage (Depositional_Composition) with Segments.
2. Multipart-to-singleparts.
3. If DED_m5.tif exists:
     - Add zonal statistics (e_ prefix) from HACH raster.
4. If veget_CHM_m5.tif exists:
     - Add zonal statistics (v_ prefix) from CHM raster.
5. Save stats as:
   - M5statistics_all.gpkg   (if both DED & CHM)
   - M5statistics_hach.gpkg  (if DED only)
   - M5statistics_FAMlike.gpkg (if neither, but you still want age+segments)

NOTE:
- The algorithm itself only *returns* the composite EA GPKG path as its Processing
  output. The DED/CHM/stat GPKGs/GeoTIFFs are additional side products written
  alongside that GPKG in the same folder.
  Changes vs v2.1
----------------
- Restores v2.1 EA overwrite-chain logic (trusted classification behaviour).
- Uses SHP for all "frozen" intermediates to avoid GPKG PK/fid collisions.
- Keeps robust _safe_polygon_difference() with polygon-only I/O and try/except
  around native:difference to avoid MultiLineString→MultiPolygon write errors.
- HACH/CHM block cleaned so DEM_clip layer is constructed before reprojection
  and trend surface creation (avoids earlier duplicate reproject logic).
"""

import os
import re
import math
import shutil
import subprocess
import tempfile
import uuid

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterField,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterVectorDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsWkbTypes,
    QgsVectorLayer,
    QgsRasterLayer,
    QgsProcessingUtils,
)
import processing


class ChronologicalOverwriteEA(QgsProcessingAlgorithm):
    """
    QGIS Processing algorithm for "newest wins" mosaicing of multiple EA layers,
    extended with HACH & CHM statistics like Module 4.
    """

    # Parameter / output keys
    PARAM_LAYERS = "EA_LAYERS"
    PARAM_CLASS_FIELD = "CLASS_FIELD"
    PARAM_PERIOD_FIELD = "PERIOD_FIELD"
    PARAM_DISSOLVE = "DISSOLVE_BY_FINAL"
    PARAM_STABLE_CARRY = "STABLE_CARRY"
    PARAM_OUTPUT = "OUTPUT"

    # HACH / CHM / stats inputs
    PARAM_DEM = "DEM"
    PARAM_DSM = "DSM"
    PARAM_FLOW = "FLOW_PATH"
    PARAM_SEGMENTS = "SEGMENTS"

    # ---------------- basics / metadata ---------------- #

    def tr(self, text):
        return QCoreApplication.translate("SCS Module 4a Floodplain Assemblage v2.3", text)

    def createInstance(self):
        return ChronologicalOverwriteEA()

    def name(self):
        return "modul5_floodplain_assemblage"

    def displayName(self):
        return self.tr("Module 4a — Floodplain Assemblage + HACH/CHM (v2.3)")

    def group(self):
        return self.tr("SCS Toolbox")

    def groupId(self):
        return "scs_ea_utils"

    def shortHelpString(self):
        return self.tr(
            "Build a newest-wins EA assemblage from up to 15 EA polygon layers and, "
            "optionally, compute HACH (DED) and CHM rasters and per-segment statistics.\n"
            "- Layers are ordered oldest→newest using the END date parsed from the period field.\n"
            "- Outputs EA_Class_final and Period_final; optional dissolve by those fields.\n"
            "- Output is always written as a GeoPackage. HACH/CHM/stats are written alongside."
        )

    # ---------------- UI parameter definitions ---------------- #

    def initAlgorithm(self, config=None):
        # EA layers
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.PARAM_LAYERS,
                self.tr("EA polygon layers (one per period; order not required)"),
                layerType=QgsProcessing.TypeVectorPolygon,
            )
        )

        # Class field
        self.addParameter(
            QgsProcessingParameterField(
                self.PARAM_CLASS_FIELD,
                self.tr("Class field (e.g., EA or EA_Class)"),
                parentLayerParameterName=self.PARAM_LAYERS,
            )
        )

        # Period field
        self.addParameter(
            QgsProcessingParameterField(
                self.PARAM_PERIOD_FIELD,
                self.tr("Period field (e.g., '20190101_20210101' or end year)"),
                parentLayerParameterName=self.PARAM_LAYERS,
            )
        )

        # Dissolve
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.PARAM_DISSOLVE,
                self.tr("Dissolve by EA_Class_final + Period_final"),
                defaultValue=True,
            )
        )

        # Behaviour: stable_floodplain carry-forward vs overwrite
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.PARAM_STABLE_CARRY,
                "Treat stable_floodplain as carry-forward (do not overwrite older classes)",
                defaultValue=False,
            )
        )

        # Output EA composite
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.PARAM_OUTPUT,
                self.tr("Composite EA output (GeoPackage)"),
                type=QgsProcessing.TypeVectorPolygon,
            )
        )

        # DEM (optional, for HACH & CHM)
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.PARAM_DEM,
                self.tr("DEM for HACH/CHM (optional but needed for HACH/CHM)"),
                optional=True,
            )
        )

        # DSM (optional, for CHM)
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.PARAM_DSM,
                self.tr("DSM for canopy height model (optional)"),
                optional=True,
            )
        )

        # Flow path
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.PARAM_FLOW,
                self.tr("Flow path (line; required for HACH/CHM)"),
                [QgsProcessing.TypeVectorLine],
                optional=True,
            )
        )

        # Segments
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.PARAM_SEGMENTS,
                self.tr("Segments from Module 2 (polygon; optional for statistics)"),
                [QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )

    # ---------------- helper methods ---------------- #

    def _run_raw(self, alg_id, params, context, feedback, tag):
        """
        Thin wrapper around processing.run with context/feedback routed correctly.
        Use this when you explicitly *know* what you're doing (e.g. in helpers).
        """
        return processing.run(
            alg_id,
            params,
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )

    def _run(self, alg_id, params, context, feedback, tag):
        """
        Guarded runner:
        - Disallows direct 'native:difference' calls from the main code.
        All differences must go through _safe_polygon_difference.
        - Everything else passes through to _run_raw.
        """
        if alg_id == "native:difference":
            # If this ever fires, there is a stray native:difference call somewhere.
            raise QgsProcessingException(
                "[Guard] Direct call to native:difference is not allowed. "
                "Use _safe_polygon_difference(...) instead."
            )

        return self._run_raw(alg_id, params, context, feedback, tag)

    def _freeze_to_shp(self, layer, context, feedback, tag="freeze"):
        """
        "Freezer" that writes a layer to a fresh Shapefile (unique file).

        Purpose:
        - Ensure each intermediate is file-backed (no transient in-memory layers).
        - Drop PK-like attribute fields ('fid', 'ogc_fid', 'OBJECTID', etc.)
          to avoid UNIQUE constraint issues when copying between backends.

        Input:
        - layer: QgsVectorLayer or a data source string/URI or a temporary
                 Processing layer id (e.g. 'Refactored_...').

        Output:
        - A string "C:\\temp\\tag_xxxxxxxx.shp" for the newly written layer.
        """
        import tempfile
        import uuid
        import os

        # ---- 1. Resolve whatever we got into a real QgsVectorLayer ----
        if isinstance(layer, QgsVectorLayer):
            v = layer
        else:
            v = None

            # a) Try to resolve as a Processing temporary layer (id) via ProcessingUtils
            try:
                v = QgsProcessingUtils.mapLayerFromString(str(layer), context)
            except Exception:
                v = None

            # b) Try the context registry directly
            if v is None or not v.isValid():
                try:
                    v = context.getMapLayer(str(layer))
                except Exception:
                    v = None

            # c) Fallback: assume it's a path/URI
            if v is None or not v.isValid():
                v = QgsVectorLayer(str(layer), "freeze_src", "ogr")

            if v is None or not v.isValid():
                raise QgsProcessingException(
                    f"[freeze] Cannot load layer in _freeze_to_shp from: {layer}"
                )

        # ---- 2. New SHP in system temp ----
        shp = os.path.join(tempfile.gettempdir(), f"{tag}_{uuid.uuid4().hex[:8]}.shp")

        # ---- 3. Detect PK-like fields that MUST be removed ----
        drop_names = set()
        try:
            flds = v.fields()
            for name in ("fid", "FID", "ogc_fid", "OGC_FID", "OBJECTID", "objectid"):
                if flds.indexFromName(name) != -1:
                    drop_names.add(name)
        except Exception:
            drop_names = set()

        # ---- 4. If we have PK-like fields, refactor them away first ----
        if drop_names:
            mapping = []
            for f in v.fields():
                if f.name() in drop_names:
                    continue
                mapping.append(
                    {
                        "name": f.name(),
                        "type": f.type(),
                        "length": f.length() if hasattr(f, "length") else 254,
                        "precision": f.precision() if hasattr(f, "precision") else 0,
                        "expression": f'"{f.name()}"',
                    }
                )
            v_ref = self._run(
                "native:refactorfields",
                {
                    "INPUT": v,
                    "FIELDS_MAPPING": mapping,
                    "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
                },
                context,
                feedback,
                f"{tag}_dropfid",
            )["OUTPUT"]
        else:
            v_ref = v

        # ---- 5. Save to the new SHP ----
        res = self._run(
            "native:savefeatures",
            {"INPUT": v_ref, "OUTPUT": shp},
            context,
            feedback,
            f"{tag}_save",
        )
        return res["OUTPUT"]

    def _safe_polygon_difference(self, input_src, overlay_src, context, feedback, tag):
        """
        Robust polygon difference:
        - Drop non-polygon geometries from both inputs
        - Fix geometries for both filtered inputs
        - Drop any non-polygons produced by fixgeometries
        - Run native:difference (wrapped in try/except)
        - Keep only Polygon / MultiPolygon geometries in the result

        input_src / overlay_src can be QgsVectorLayer or a data source string.
        Returns a TEMPORARY_OUTPUT layer id, which can then be frozen to SHP.
        """

        # 0) Ensure we only work with polygons on both sides (before fixing)
        in_poly = self._run_raw(
            "native:extractbyexpression",
            {
                "INPUT": input_src,
                "EXPRESSION": "geometry_type($geometry) IN ('Polygon','MultiPolygon')",
                "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
            },
            context,
            feedback,
            f"{tag}_in_poly0",
        )["OUTPUT"]

        ov_poly = self._run_raw(
            "native:extractbyexpression",
            {
                "INPUT": overlay_src,
                "EXPRESSION": "geometry_type($geometry) IN ('Polygon','MultiPolygon')",
                "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
            },
            context,
            feedback,
            f"{tag}_ov_poly0",
        )["OUTPUT"]

        # 1) Fix geometries on both polygon-only inputs
        fixed_in = self._run_raw(
            "native:fixgeometries",
            {
                "INPUT": in_poly,
                "METHOD": 1,
                "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
            },
            context,
            feedback,
            f"{tag}_fix_in",
        )["OUTPUT"]

        fixed_ov = self._run_raw(
            "native:fixgeometries",
            {
                "INPUT": ov_poly,
                "METHOD": 1,
                "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
            },
            context,
            feedback,
            f"{tag}_fix_ov",
        )["OUTPUT"]

        # 1b) Filter again to drop any line/mixed geometries created by fixgeometries
        fixed_in_poly = self._run_raw(
            "native:extractbyexpression",
            {
                "INPUT": fixed_in,
                "EXPRESSION": "geometry_type($geometry) IN ('Polygon','MultiPolygon')",
                "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
            },
            context,
            feedback,
            f"{tag}_in_poly1",
        )["OUTPUT"]

        fixed_ov_poly = self._run_raw(
            "native:extractbyexpression",
            {
                "INPUT": fixed_ov,
                "EXPRESSION": "geometry_type($geometry) IN ('Polygon','MultiPolygon')",
                "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
            },
            context,
            feedback,
            f"{tag}_ov_poly1",
        )["OUTPUT"]

        # 2) Difference on polygon-only, fixed inputs
        try:
            diff_tmp = self._run_raw(
                "native:difference",
                {
                    "INPUT": fixed_in_poly,
                    "OVERLAY": fixed_ov_poly,
                    "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
                },
                context,
                feedback,
                f"{tag}_diff",
            )["OUTPUT"]
        except Exception as e:
            # If the provider chokes on MultiLineString vs MultiPolygon internally,
            # fall back to returning the input polygons unchanged instead of failing.
            feedback.reportError(
                f"[Diff] native:difference failed for tag={tag}; "
                f"returning input polygons unchanged. Error: {e}"
            )
            return fixed_in_poly

        # 3) Final safety: ensure only polygons in result
        diff_poly = self._run_raw(
            "native:extractbyexpression",
            {
                "INPUT": diff_tmp,
                "EXPRESSION": "geometry_type($geometry) IN ('Polygon','MultiPolygon')",
                "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
            },
            context,
            feedback,
            f"{tag}_poly",
        )["OUTPUT"]

        return diff_poly

    def _layer_period_value(self, vlyr, period_field, feedback):
        """Return integer YYYYMMDD for PeriodEnd, using strict yyyy_mm_dd in the field."""
        pat = re.compile(r"^((?:18|19|20)\d{2})_(\d{2})_(\d{2})$")

        idx = vlyr.fields().indexFromName(period_field)
        if idx == -1:
            raise QgsProcessingException(
                f"Field '{period_field}' not found in layer '{vlyr.name()}'"
            )

        vals = []
        bad = 0
        for f in vlyr.getFeatures():
            raw = f[period_field]
            if raw is None or str(raw).strip() == "":
                continue
            s = str(raw).strip()
            m = pat.match(s)
            if not m:
                bad += 1
                continue
            vals.append(int(f"{m.group(1)}{m.group(2)}{m.group(3)}"))

        if not vals:
            raise QgsProcessingException(
                f"No valid yyyy_mm_dd dates found in '{period_field}' for layer '{vlyr.name()}'"
            )
        if bad:
            feedback.reportError(
                f"[Warn] {bad} feature(s) in '{vlyr.name()}' had non-matching PeriodEnd "
                f"(expected yyyy_mm_dd). Only matching values were used for sorting."
            )
        return max(vals)

    def _drop_fid_fields(self, layer, context, feedback, tag="dropfid"):
        """
        Remove any attribute named 'fid'/'FID'/ogc_fid/OBJECTID from a layer.

        Input:
        - layer: QgsVectorLayer or path/URI or a temporary Processing layer id.

        Returns:
        - Either the original layer identifier (if nothing to drop or resolution failed),
          or a TEMPORARY_OUTPUT id where those fields have been removed.
        """
        # ---- 1. Resolve 'layer' to a real QgsVectorLayer ----
        if isinstance(layer, QgsVectorLayer):
            vl = layer
        else:
            vl = None
            # a) Try Processing temporary layer registry
            try:
                vl = QgsProcessingUtils.mapLayerFromString(str(layer), context)
            except Exception:
                vl = None
            # b) Try context registry directly
            if vl is None or not vl.isValid():
                try:
                    vl = context.getMapLayer(str(layer))
                except Exception:
                    vl = None
            # c) Fallback: assume it's a path/URI
            if vl is None or not vl.isValid():
                vl = QgsVectorLayer(str(layer), "dropfid_src", "ogr")

        if vl is None or not vl.isValid():
            feedback.reportError(f"[dropfid] Cannot resolve layer from {layer}; skipping fid-drop.")
            return layer

        fields = vl.fields()

        # Determine which PK-like fields are present
        drop = [
            n
            for n in ("fid", "FID", "ogc_fid", "OGC_FID", "OBJECTID", "objectid")
            if fields.indexFromName(n) != -1
        ]
        if not drop:
            return layer  # nothing to do

        # Build mapping for all non-dropped fields
        mapping = []
        for f in fields:
            if f.name() in drop:
                continue
            mapping.append(
                {
                    "name": f.name(),
                    "type": f.type(),
                    "length": getattr(f, "length", lambda: 254)(),
                    "precision": getattr(f, "precision", lambda: 0)(),
                    "expression": f'"{f.name()}"',
                }
            )

        out = self._run(
            "native:refactorfields",
            {
                "INPUT": vl,
                "FIELDS_MAPPING": mapping,
                "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
            },
            context,
            feedback,
            f"{tag}_refactor",
        )["OUTPUT"]

        return out

    def _as_layer(self, src):
        if isinstance(src, QgsVectorLayer):
            return src
        lyr = QgsVectorLayer(src, "tmp", "ogr")
        if not lyr.isValid():
            raise QgsProcessingException(f"Cannot load layer from: {src}")
        return lyr

    def _safe_delete_gpkg(self, path):
        for p in (path, path + "-shm", path + "-wal"):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    @staticmethod
    def _cellsize(rl: QgsRasterLayer):
        """
        Get raster cell size (assuming square pixels) from a QgsRasterLayer.

        Returns:
        - pixel size in map units (float), or 1.0 as a fallback.
        """
        try:
            res = rl.rasterUnitsPerPixelX()
            if res and res > 0:
                return float(res)
        except Exception:
            pass
        return 1.0

    # ---------------- main processAlgorithm ---------------- #

    def processAlgorithm(self, parameters, context, feedback):
        # ---- read core EA parameters ----
        layers = self.parameterAsLayerList(parameters, self.PARAM_LAYERS, context)
        class_field = self.parameterAsString(parameters, self.PARAM_CLASS_FIELD, context)
        period_field = self.parameterAsString(parameters, self.PARAM_PERIOD_FIELD, context)
        do_dissolve = self.parameterAsBoolean(parameters, self.PARAM_DISSOLVE, context)
        # behaviour flag
        use_stable_carry = self.parameterAsBoolean(
            parameters, self.PARAM_STABLE_CARRY, context
        )

        # HACH/CHM/stats inputs
        dem = self.parameterAsRasterLayer(parameters, self.PARAM_DEM, context)
        dsm = self.parameterAsRasterLayer(parameters, self.PARAM_DSM, context)
        flow = self.parameterAsVectorLayer(parameters, self.PARAM_FLOW, context)
        segs = self.parameterAsVectorLayer(parameters, self.PARAM_SEGMENTS, context)

        dem_ok = dem is not None and dem.isValid()
        dsm_ok = dsm is not None and dsm.isValid()
        flow_ok = flow is not None and flow.isValid()
        segs_ok = segs is not None and segs.isValid()

        # ---- normalize output path to GPKG ----
        raw_out = self.parameterAsOutputLayer(parameters, self.PARAM_OUTPUT, context)
        base_path = raw_out.split("|")[0]
        root, ext = os.path.splitext(base_path)
        if ext.lower() != ".gpkg":
            base_path = root + ".gpkg"
        os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
        layername = "Depositional_Composition"

        # ---- basic checks ----
        if not layers or len(layers) < 1:
            raise QgsProcessingException("Provide at least one polygon EA layer.")
        if len(layers) > 15:
            raise QgsProcessingException("This tool supports up to 15 layers.")
        for v in layers:
            if QgsWkbTypes.geometryType(v.wkbType()) != QgsWkbTypes.PolygonGeometry:
                raise QgsProcessingException(f"Layer {v.name()} is not a polygon layer.")

        # ---- sort by period (oldest → newest) ----
        scored = []
        for v in layers:
            pval = self._layer_period_value(v, period_field, feedback)
            scored.append((pval, v))

        scored.sort(key=lambda t: t[0])
        feedback.pushInfo(
            "Order (oldest→newest): "
            + "  <  ".join([f"{v.name()}[{p}]" for p, v in scored])
        )
        feedback.pushInfo(f"[Check] Selected layers count: {len(scored)}")

        # ---- canonical PeriodEnd text per layer (yyyy_mm_dd or yyyy_01_01) ----
        layer_period_txt = {}
        for _, lyr in scored:
            vals = []
            for ft in lyr.getFeatures():
                val = ft[period_field]
                if val is None:
                    continue
                s = str(val).strip()
                if re.match(r"^(18|19|20)\d{2}([_\-]?\d{2})([_\-]?\d{2})$", s):
                    y = s[0:4]
                    m = s[5:7] if len(s) >= 7 and not s[4:5].isdigit() else s[4:6]
                    d = s[-2:]
                    vals.append(f"{y}_{m}_{d}")
                elif re.match(r"^(18|19|20)\d{2}$", s):
                    vals.append(f"{s}_01_01")

            if vals:
                layer_period_txt[lyr.id()] = sorted(vals)[-1]
            else:
                layer_period_txt[lyr.id()] = "0000_00_00"
                feedback.reportError(
                    f"[Warn] No valid {period_field} found in {lyr.name()}, set to 0000_00_00"
                )

        # ---- prep each EA layer: fix → refactor EA_Class_i + Period_i → freeze ----
        prepped = []
        for i, (pval, v) in enumerate(scored, start=1):
            feedback.pushInfo(f"[Prep] {v.name()} — fix geometries")

            fixed = self._run(
                "native:fixgeometries",
                {"INPUT": v, "METHOD": 1, "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT},
                context,
                feedback,
                f"fix_{i}",
            )["OUTPUT"]

            class_new = f"EA_Class_{i}"
            period_new = f"Period_{i}"

            # We will look for EA in one of these fields, in this order:
            #   1) 'EA'
            #   2) 'EA_Class'
            #   3) the user-selected class_field (from the dialog)
            src_field_names = [f.name() for f in v.fields()]
            class_candidates = ["EA", "EA_Class", class_field]

            # Keep only those candidates that actually exist on this layer
            avail = [nm for nm in class_candidates if nm and nm in src_field_names]

            if not avail:
                # No usable EA field on this layer → EA_Class_i will be NULL
                feedback.reportError(
                    f"[Prep] No EA class field found in {v.name()} among {class_candidates}; "
                    f"EA_Class_{i} will be NULL for this layer."
                )
                expr_class = "NULL"
            else:
                # If exactly one candidate exists, just use it directly
                if len(avail) == 1:
                    base_expr = f'"{avail[0]}"'
                else:
                    # If several exist, coalesce them in a fixed order
                    base_expr = "coalesce(" + ", ".join([f'"{nm}"' for nm in avail]) + ")"
                # Treat empty string as NULL
                expr_class = f"nullif({base_expr}, '')"

            try:
                period_text = layer_period_txt[v.id()]
                expr_period = f"'{period_text}'"
            except Exception:
                expr_period = f"nullif(attribute(@feature, '{period_field}'), '')"

            ref = self._run(
                "native:refactorfields",
                {
                    "INPUT": fixed,
                    "FIELDS_MAPPING": [
                        {
                            "expression": expr_class,
                            "length": 254,
                            "name": class_new,
                            "precision": 0,
                            "type": 10,
                        },
                        {
                            "expression": expr_period,
                            "length": 254,
                            "name": period_new,
                            "precision": 0,
                            "type": 10,
                        },
                    ],
                    "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
                },
                context,
                feedback,
                f"refactor_{i}",
            )["OUTPUT"]

            ref = self._freeze_to_shp(ref, context, feedback, f"prepped_{i}")
            prepped.append(ref)

        feedback.pushInfo(f"[Check] Period layers prepared: {len(prepped)}")

        # ---- newest-wins overwrite (optionally with stable_floodplain = carry-forward) ----
        working = prepped[0]
        total = len(prepped) - 1

        if use_stable_carry:
            feedback.pushInfo(
                f"[Overwrite] {total} chronological steps (oldest → newest) "
                "(stable_floodplain = carry-forward, SHP intermediates)"
            )

            for idx, nxt in enumerate(prepped[1:], start=1):
                feedback.pushInfo(
                    f"[Overwrite {idx}/{total}] Removing overlap + adding newer period "
                    "(stable_floodplain = carry-forward)"
                )

                # Always freeze to shapefile for stability
                working_safe = self._freeze_to_shp(working, context, feedback, f"work_{idx}")
                nxt_safe = self._freeze_to_shp(nxt, context, feedback, f"next_{idx}")

                # This next layer has fields EA_Class_{idx+1}, Period_{idx+1}
                class_i = f"EA_Class_{idx+1}"

                # 1) Split the new period into stable_floodplain vs everything else
                stable_expr = (
                    f"lower(coalesce(\"{class_i}\",'')) IN "
                    "('stable_floodplain','stable floodplain')"
                )
                nonstable_expr = (
                    f"coalesce(\"{class_i}\",'') = '' OR "
                    f"lower(coalesce(\"{class_i}\",'')) NOT IN "
                    "('stable_floodplain','stable floodplain')"
                )

                nxt_stable = self._run(
                    "native:extractbyexpression",
                    {
                        "INPUT": nxt_safe,
                        "EXPRESSION": stable_expr,
                        "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
                    },
                    context,
                    feedback,
                    f"stable_{idx}",
                )["OUTPUT"]
                nxt_stable = self._freeze_to_shp(nxt_stable, context, feedback, f"stable_f_{idx}")

                nxt_nonstable = self._run(
                    "native:extractbyexpression",
                    {
                        "INPUT": nxt_safe,
                        "EXPRESSION": nonstable_expr,
                        "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
                    },
                    context,
                    feedback,
                    f"nonstable_{idx}",
                )["OUTPUT"]
                nxt_nonstable = self._freeze_to_shp(
                    nxt_nonstable, context, feedback, f"nonstable_f_{idx}"
                )

                # 2) Newest-wins ONLY for non-stable_floodplain classes
                older_minus_newer_tmp = self._safe_polygon_difference(
                    working_safe,
                    nxt_nonstable,
                    context,
                    feedback,
                    f"diff_nonstable_{idx}",
                )
                older_minus_newer = self._freeze_to_shp(
                    older_minus_newer_tmp, context, feedback, f"diffout_nonstable_{idx}"
                )

                # 3) stable_floodplain polygons are only kept where there is NO previous coverage
                stable_new_only_tmp = self._safe_polygon_difference(
                    nxt_stable,
                    working_safe,
                    context,
                    feedback,
                    f"stable_minus_work_{idx}",
                )
                stable_new_only = self._freeze_to_shp(
                    stable_new_only_tmp, context, feedback, f"stable_newonly_{idx}"
                )

                # 4) Merge: keep older-minus-new, plus new nonstable, plus new-only stable
                working = self._run(
                    "native:mergevectorlayers",
                    {
                        "LAYERS": [older_minus_newer, nxt_nonstable, stable_new_only],
                        "CRS": None,
                        "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
                    },
                    context,
                    feedback,
                    f"merge_{idx}",
                )["OUTPUT"]

        else:
            # Original behaviour: simple newest-wins overwrite, everything overwrites
            feedback.pushInfo(
                f"[Overwrite] {total} chronological steps (oldest → newest, SHP intermediates)"
            )

            for idx, nxt in enumerate(prepped[1:], start=1):
                feedback.pushInfo(
                    f"[Overwrite {idx}/{total}] Removing overlap + adding newer period"
                )

                working_safe = self._freeze_to_shp(working, context, feedback, f"work_{idx}")
                nxt_safe = self._freeze_to_shp(nxt, context, feedback, f"next_{idx}")

                older_minus_newer_tmp = self._safe_polygon_difference(
                    working_safe,
                    nxt_safe,
                    context,
                    feedback,
                    f"diff_{idx}",
                )

                older_minus_newer = self._freeze_to_shp(
                    older_minus_newer_tmp, context, feedback, f"diffout_{idx}"
                )

                working = self._run(
                    "native:mergevectorlayers",
                    {
                        "LAYERS": [older_minus_newer, nxt_safe],
                        "CRS": None,
                        "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
                    },
                    context,
                    feedback,
                    f"merge_{idx}",
                )["OUTPUT"]

        union_path = working

        # ---- retain any missing slivers of the newest period ----
        latest = prepped[-1]

        miss_tmp = self._safe_polygon_difference(
            latest,
            union_path,
            context,
            feedback,
            "latest_minus_union",
        )

        miss_frozen = self._freeze_to_shp(miss_tmp, context, feedback, "retain_miss")

        def _has_features(src):
            lyr = src if isinstance(src, QgsVectorLayer) else QgsVectorLayer(src, "chk", "ogr")
            return lyr.isValid() and lyr.featureCount() > 0

        if _has_features(miss_frozen):
            working2 = self._run(
                "native:mergevectorlayers",
                {
                    "LAYERS": [union_path, miss_frozen],
                    "CRS": None,
                    "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
                },
                context,
                feedback,
                "merge_missing_latest",
            )["OUTPUT"]
            union_path = working2
        else:
            feedback.pushInfo("[Retain] No missing pieces from newest period; skipping merge.")

        # ---- add EA_Class_final / Period_final / Year ----
        n = len(prepped)

        # Resolve union_path (which may be a temp layer id like 'Merged_...') to a real layer
        if isinstance(union_path, QgsVectorLayer):
            vlayer = union_path
        else:
            # 1) Try temporary Processing layer registry
            vlayer = QgsProcessingUtils.mapLayerFromString(str(union_path), context)
            # 2) Try context registry directly
            if vlayer is None or not vlayer.isValid():
                try:
                    vlayer = context.getMapLayer(str(union_path))
                except Exception:
                    vlayer = None
            # 3) Fallback: assume union_path is a file/URI
            if vlayer is None or not vlayer.isValid():
                vlayer = QgsVectorLayer(str(union_path), "union_temp", "ogr")

        if vlayer is None or not vlayer.isValid():
            raise QgsProcessingException("Union layer became invalid before final coalesce.")

        field_names = [f.name() for f in vlayer.fields()]
        class_fields = [f"EA_Class_{k}" for k in range(n, 0, -1) if f"EA_Class_{k}" in field_names]
        period_fields = [f"Period_{k}" for k in range(n, 0, -1) if f"Period_{k}" in field_names]
        if not class_fields or not period_fields:
            raise QgsProcessingException(
                "[EA] Missing expected EA_Class_i / Period_i fields — found only "
                f"{field_names}"
            )

        feedback.pushInfo(
            f"[Final] Coalescing {len(class_fields)} EA_Class_i and {len(period_fields)} Period_i fields"
        )

        # Use only the fields that actually exist, newest→oldest
        class_parts = [f'nullif("{fn}", \'\')' for fn in class_fields]
        period_parts = [f'nullif("{fn}", \'\')' for fn in period_fields]
        expr_class_final = "coalesce(" + ", ".join(class_parts) + ")"
        expr_period_final = "coalesce(" + ", ".join(period_parts) + ")"

        with_class = self._run(
            "native:fieldcalculator",
            {
                "INPUT": union_path,
                "FIELD_NAME": "EA_Class_final",
                "FIELD_TYPE": 2,
                "FIELD_LENGTH": 254,
                "FIELD_PRECISION": 0,
                "FORMULA": expr_class_final,
                "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
            },
            context,
            feedback,
            "final_class",
        )["OUTPUT"]

        with_period = self._run(
            "native:fieldcalculator",
            {
                "INPUT": with_class,
                "FIELD_NAME": "Period_final",
                "FIELD_TYPE": 2,
                "FIELD_LENGTH": 64,
                "FIELD_PRECISION": 0,
                "FORMULA": expr_period_final,
                "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
            },
            context,
            feedback,
            "final_period",
        )["OUTPUT"]

        expr_year = 'to_int(substr("Period_final", 1, 4))'

        if do_dissolve:
            feedback.pushInfo("[Dissolve] By EA_Class_final + Period_final")
            dissolved_tmp = self._run(
                "native:dissolve",
                {
                    "INPUT": with_period,
                    "FIELD": ["EA_Class_final", "Period_final"],
                    "SEPARATE_DISJOINT": False,
                    "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
                },
                context,
                feedback,
                "dissolve",
            )["OUTPUT"]

            with_year = self._run(
                "native:fieldcalculator",
                {
                    "INPUT": dissolved_tmp,
                    "FIELD_NAME": "Year",
                    "FIELD_TYPE": 1,
                    "FIELD_LENGTH": 10,
                    "FIELD_PRECISION": 0,
                    "FORMULA": expr_year,
                    "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
                },
                context,
                feedback,
                "add_year",
            )["OUTPUT"]

            cleaned = self._drop_fid_fields(with_year, context, feedback, "final")
        else:
            feedback.pushInfo("[Save] Without dissolve")
            with_year = self._run(
                "native:fieldcalculator",
                {
                    "INPUT": with_period,
                    "FIELD_NAME": "Year",
                    "FIELD_TYPE": 1,
                    "FIELD_LENGTH": 10,
                    "FIELD_PRECISION": 0,
                    "FORMULA": expr_year,
                    "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
                },
                context,
                feedback,
                "add_year",
            )["OUTPUT"]

            cleaned = self._drop_fid_fields(with_year, context, feedback, "final")

        # ---- save EA assemblage to GPKG ----
        self._safe_delete_gpkg(base_path)
        params = {"INPUT": cleaned, "OUTPUT": base_path, "LAYER_NAME": layername}
        params["OVERWRITE"] = True

        final_path = self._run(
            "native:savefeatures", params, context, feedback, "save_gpkg"
        )["OUTPUT"]

        # ===================== HACH / CHM / STATS (Module 4-style) ===================== #

        out_dir = os.path.dirname(base_path) or os.getcwd()

        ded_path = None
        chm_path = None

        if not dem_ok:
            feedback.pushInfo("[HACH/CHM] DEM not provided or invalid — skipping HACH/CHM/stats.")
        else:
            feedback.pushInfo("[HACH] Building AOI from EA assemblage and clipping DEM.")
            dem_pix = self._cellsize(dem)
            buffer_dist = max(10.0 * dem_pix, dem_pix)

            aoi = self._run(
                "native:buffer",
                {
                    "INPUT": cleaned,
                    "DISTANCE": buffer_dist,
                    "SEGMENTS": 5,
                    "END_CAP_STYLE": 0,
                    "JOIN_STYLE": 0,
                    "MITER_LIMIT": 2,
                    "DISSOLVE": True,
                    "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
                },
                context,
                feedback,
                "aoi_buffer",
            )["OUTPUT"]

            dem_clip = os.path.join(out_dir, "DEM_clipped_m5.tif")
            if os.path.exists(dem_clip):
                try:
                    os.remove(dem_clip)
                except Exception:
                    pass

            self._run(
                "gdal:cliprasterbymasklayer",
                {
                    "INPUT": dem,
                    "MASK": aoi,
                    "SOURCE_CRS": None,
                    "TARGET_CRS": None,
                    "NODATA": None,
                    "ALPHA_BAND": False,
                    "CROP_TO_CUTLINE": True,
                    "KEEP_RESOLUTION": True,
                    "SET_RESOLUTION": False,
                    "X_RESOLUTION": None,
                    "Y_RESOLUTION": None,
                    "MULTITHREADING": True,
                    "OPTIONS": "",
                    "DATA_TYPE": 0,
                    "EXTRA": "",
                    "OUTPUT": dem_clip,
                },
                context,
                feedback,
                "dem_clip",
            )

        # ---------- HACH (DED_m5) ---------- #
        if dem_ok and flow_ok:
            feedback.pushInfo("[HACH] Computing DED from DEM_clipped_m5 and flow-path.")

            # 1) Points along flow at a spacing of ~5× pixel size
            dem_clip_rl = QgsRasterLayer(dem_clip, "dem_clip_m5")
            if not dem_clip_rl.isValid():
                raise QgsProcessingException("DEM_clipped_m5 could not be opened.")

            ext = dem_clip_rl.extent()
            px = self._cellsize(dem_clip_rl)
            width = int(round((ext.xMaximum() - ext.xMinimum()) / px))
            height = int(round((ext.yMaximum() - ext.yMinimum()) / px))
            if width <= 0 or height <= 0:
                raise QgsProcessingException(
                    "Target grid size for HACH is not positive; check DEM/extent."
                )

            feedback.pushInfo(f"[HACH] DEM_clip_m5 size: {width} x {height} (px={px})")

            distance = max(px * 5.0, px)
            pts_along = os.path.join(out_dir, "flow_pts_m5.gpkg")

            # Ensure we can overwrite the points-along file
            if os.path.exists(pts_along):
                try:
                    os.remove(pts_along)
                except Exception:
                    feedback.reportError(
                        f"[HACH] Could not remove existing {pts_along}; overwrite may fail."
                    )

            self._run(
                "native:pointsalonglines",
                {
                    "INPUT": flow,
                    "DISTANCE": distance,
                    "START_OFFSET": 0,
                    "END_OFFSET": 0,
                    "OUTPUT": pts_along,
                },
                context,
                feedback,
                "flow_pts",
            )

            # 2) Sample DEM_clip at these points → new attribute dem_*
            pts_samp = os.path.join(out_dir, "flow_pts_dem_m5.gpkg")

            # Ensure we can overwrite the rastersampling output
            if os.path.exists(pts_samp):
                try:
                    os.remove(pts_samp)
                except Exception:
                    feedback.reportError(
                        f"[HACH] Could not remove existing {pts_samp}; overwrite may fail."
                    )

            self._run(
                "qgis:rastersampling",
                {
                    "INPUT": pts_along,
                    "RASTERCOPY": dem_clip,
                    "COLUMN_PREFIX": "dem_",
                    "OUTPUT": pts_samp,
                },
                context,
                feedback,
                "flow_pts_dem",
            )

            # 2b) Reproject sampled points to DEM CRS so GDAL sees correct coords
            pts_reproj = os.path.join(out_dir, "flow_pts_dem_m5_reproj.gpkg")
            if os.path.exists(pts_reproj):
                try:
                    os.remove(pts_reproj)
                except Exception:
                    feedback.reportError(
                        f"[HACH] Could not remove existing {pts_reproj}; overwrite may fail."
                    )

            self._run(
                "native:reprojectlayer",
                {
                    "INPUT": pts_samp,
                    "TARGET_CRS": dem_clip_rl.crs().authid(),
                    "OUTPUT": pts_reproj,
                },
                context,
                feedback,
                "flow_pts_dem_reproj",
            )

            pts_layer = QgsVectorLayer(pts_reproj, "pts_dem_m5", "ogr")
            if not pts_layer or not pts_layer.isValid():
                raise QgsProcessingException("flow_pts_dem_m5_reproj could not be opened.")

            # Sanity: ensure we actually have points
            if pts_layer.featureCount() == 0:
                raise QgsProcessingException(
                    "[HACH] flow_pts_dem_m5_reproj has 0 features – cannot build trend surface. "
                    "Check that flow intersects DEM_clipped_m5 and pointsalonglines succeeded."
                )

            elev_field = "dem_1"
            field_names = pts_layer.fields().names()
            if elev_field not in field_names:
                all_fields = ", ".join(field_names)
                raise QgsProcessingException(
                    f"Expected field '{elev_field}' not found on flow_pts_dem_m5_reproj. "
                    f"Fields: {all_fields}"
                )

            # Optional: quick check for all-NULL dem_1
            has_valid_z = False
            for f in pts_layer.getFeatures():
                z = f[elev_field]
                if z is not None:
                    has_valid_z = True
                    break
            if not has_valid_z:
                raise QgsProcessingException(
                    "[HACH] All dem_1 values on flow_pts_dem_m5_reproj are NULL – "
                    "DEM sampling failed, cannot build trend surface."
                )

            trend_path = os.path.join(out_dir, "trend_idw_m5.tif")
            if os.path.exists(trend_path):
                try:
                    os.remove(trend_path)
                except Exception:
                    pass

            # Prepare a big radius for provider fallback
            dx = ext.xMaximum() - ext.xMinimum()
            dy = ext.yMaximum() - ext.yMinimum()
            big_radius = max(1.0, math.hypot(dx, dy))

            def run_gdal_grid_cli():
                exe = shutil.which("gdal_grid") or "gdal_grid"

                # For GPKG, use the actual layer name seen by OGR
                pts_l = QgsVectorLayer(pts_reproj, "", "ogr")
                if not pts_l.isValid():
                    raise QgsProcessingException(
                        "[HACH] pts_reproj could not be opened by OGR for gdal_grid."
                    )
                layer_name = pts_l.name() or "flow_pts_dem_m5_reproj"

                alg_str = (
                    f"invdistnn:power=2.0:smoothing=0.0:radius={big_radius}:"
                    f"max_points=0:min_points=1"
                )
                args = [
                    exe,
                    "-l",
                    layer_name,
                    "-zfield",
                    elev_field,
                    "-a",
                    alg_str,
                    "-ot",
                    "Float32",
                    "-of",
                    "GTiff",
                    "-txe",
                    str(ext.xMinimum()),
                    str(ext.xMaximum()),
                    "-tye",
                    str(ext.yMinimum()),
                    str(ext.yMaximum()),
                    "-outsize",
                    str(width),
                    str(height),
                    pts_reproj,
                    trend_path,
                ]
                feedback.pushInfo("gdal_grid command:\n" + " ".join(args))
                proc = subprocess.run(args, capture_output=True, text=True)
                feedback.pushInfo("gdal_grid stdout:\n" + (proc.stdout or "").strip())
                if proc.returncode != 0:
                    err = (proc.stderr or "").strip()
                    raise QgsProcessingException(
                        f"[HACH] gdal_grid failed with code {proc.returncode}: {err}"
                    )

            try:
                if shutil.which("gdal_grid"):
                    feedback.pushInfo("[HACH] Using gdal_grid CLI for trend surface.")
                    run_gdal_grid_cli()
                else:
                    feedback.pushInfo(
                        "[HACH] gdal_grid not found in PATH; using gdal:gridinversedistancenearestneighbor."
                    )
                    trend_tmp = os.path.join(out_dir, "trend_idw_tmp_m5.tif")
                    if os.path.exists(trend_tmp):
                        try:
                            os.remove(trend_tmp)
                        except Exception:
                            pass

                    # Provider-based IDW: use a large search ellipse and require ≥1 point
                    self._run(
                        "gdal:gridinversedistancenearestneighbor",
                        {
                            "INPUT": pts_layer,
                            "Z_FIELD": elev_field,
                            "POWER": 2.0,
                            "SMOOTHING": 0.0,
                            "RADIUS_1": big_radius,
                            "RADIUS_2": big_radius,
                            "ANGLE": 0.0,
                            "MAX_POINTS": 0,      # no upper limit
                            "MIN_POINTS": 1,      # at least one point required
                            "NODATA": -9999.0,    # explicit nodata marker
                            "DATA_TYPE": 5,       # Float32
                            "OPTIONS": "",
                            "EXTRA": "",
                            "OUTPUT": trend_tmp,
                        },
                        context,
                        feedback,
                        "trend_provider",
                    )

                    # Warp to DEM grid (extent + resolution), carrying nodata
                    self._run(
                        "gdal:warpreproject",
                        {
                            "INPUT": trend_tmp,
                            "SOURCE_CRS": None,
                            "TARGET_CRS": dem_clip_rl.crs().authid() or None,
                            "RESAMPLING": 0,
                            "NODATA": -9999.0,
                            "TARGET_RESOLUTION": True,
                            "X_RESOLUTION": px,
                            "Y_RESOLUTION": px,
                            "TARGET_EXTENT": f"{ext.xMinimum()},{ext.yMinimum()},"
                                             f"{ext.xMaximum()},{ext.yMaximum()}",
                            "TARGET_EXTENT_CRS": dem_clip_rl.crs().authid() or None,
                            "MULTITHREADING": True,
                            "OPTIONS": "",
                            "DATA_TYPE": 5,
                            "EXTRA": "",
                            "OUTPUT": trend_path,
                        },
                        context,
                        feedback,
                        "trend_warp",
                    )

            except Exception as e:
                raise QgsProcessingException(f"HACH trend surface creation failed: {e}")

            # 5) Compute DED_m5 = DEM_clipped_m5 - trend_idw_m5
            ded_path = os.path.join(out_dir, "DED_m5.tif")
            if os.path.exists(ded_path):
                try:
                    os.remove(ded_path)
                except Exception:
                    pass

            self._run(
                "gdal:rastercalculator",
                {
                    "INPUT_A": dem_clip,
                    "BAND_A": 1,
                    "INPUT_B": trend_path,
                    "BAND_B": 1,
                    "FORMULA": "A-B",
                    "OUTPUT": ded_path,
                    "RTYPE": 6,
                    "NO_DATA": None,
                    "EXTRA": "",
                    "OPTIONS": "",
                },
                context,
                feedback,
                "ded_calc",
            )
            feedback.pushInfo(f"[HACH] DED_m5 (HACH) written: {ded_path}")
        else:
            if dem_ok and not flow_ok:
                feedback.pushInfo("[HACH] Flow path not provided/valid — skipping DED_m5.")
            # if no DEM, we already logged a message above

        # ---------- CHM ---------- #
        if dem_ok and dsm_ok and flow_ok:
            feedback.pushInfo("[CHM] Computing CHM = DSM - DEM_clipped_m5.")
            raw_chm = os.path.join(out_dir, "CHM_raw_m5.tif")
            if os.path.exists(raw_chm):
                try:
                    os.remove(raw_chm)
                except Exception:
                    pass

            self._run(
                "gdal:rastercalculator",
                {
                    "INPUT_A": dsm,
                    "BAND_A": 1,
                    "INPUT_B": dem_clip,
                    "BAND_B": 1,
                    "FORMULA": "A-B",
                    "OUTPUT": raw_chm,
                    "RTYPE": 6,
                    "NO_DATA": None,
                    "EXTRA": "",
                    "OPTIONS": "",
                },
                context,
                feedback,
                "chm_raw",
            )

            chm_path = os.path.join(out_dir, "veget_CHM_m5.tif")
            if os.path.exists(chm_path):
                try:
                    os.remove(chm_path)
                except Exception:
                    pass

            self._run(
                "gdal:rastercalculator",
                {
                    "INPUT_A": raw_chm,
                    "BAND_A": 1,
                    "FORMULA": "(A>0)*A + (A<=0)*(-9999)",
                    "OUTPUT": chm_path,
                    "RTYPE": 6,
                    "NO_DATA": -9999,
                    "EXTRA": "",
                    "OPTIONS": "",
                },
                context,
                feedback,
                "chm_mask",
            )
            feedback.pushInfo(f"[CHM] veget_CHM_m5 written: {chm_path}")
        else:
            if dsm_ok and not flow_ok:
                feedback.pushInfo("[CHM] DSM present but flow missing — CHM skipped.")
            elif flow_ok and not dsm_ok:
                feedback.pushInfo("[CHM] Flow present but DSM missing — CHM skipped.")
            else:
                feedback.pushInfo("[CHM] DSM/flow both missing or invalid — CHM skipped.")

        # ---------- Stats (EA × Segments) ---------- #
        if segs_ok:
            feedback.pushInfo("[Stats] Building stats from EA assemblage × segments.")

            inter = self._run(
                "native:intersection",
                {
                    "INPUT": cleaned,
                    "OVERLAY": segs,
                    "INPUT_FIELDS_PREFIX": "",
                    "OVERLAY_FIELDS_PREFIX": "",
                    "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
                },
                context,
                feedback,
                "stats_intersect",
            )["OUTPUT"]

            inter_single = self._run(
                "native:multiparttosingleparts",
                {
                    "INPUT": inter,
                    "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
                },
                context,
                feedback,
                "stats_single",
            )["OUTPUT"]

            # HACH stats
            if ded_path and os.path.exists(ded_path):
                feedback.pushInfo("[Stats] Adding HACH zonal statistics (e_*) from DED_m5.")
                self._run(
                    "qgis:zonalstatistics",
                    {
                        "INPUT_VECTOR": inter_single,
                        "INPUT_RASTER": ded_path,
                        "RASTER_BAND": 1,
                        "COLUMN_PREFIX": "e_",
                        "STATISTICS": [2, 3, 4, 5, 7, 8],
                    },
                    context,
                    feedback,
                    "stats_hach",
                )

            # CHM stats
            if chm_path and os.path.exists(chm_path):
                feedback.pushInfo("[Stats] Adding CHM zonal statistics (v_*) from veget_CHM_m5.")
                self._run(
                    "qgis:zonalstatistics",
                    {
                        "INPUT_VECTOR": inter_single,
                        "INPUT_RASTER": chm_path,
                        "RASTER_BAND": 1,
                        "COLUMN_PREFIX": "v_",
                        "STATISTICS": [2, 3, 4, 5, 7, 8],
                    },
                    context,
                    feedback,
                    "stats_chm",
                )

            if ded_path and chm_path:
                stats_name = "M5statistics_all"
            elif ded_path:
                stats_name = "M5statistics_hach"
            else:
                stats_name = "M5statistics_FAMlike"

            stats_path = os.path.join(out_dir, f"{stats_name}.gpkg")
            if os.path.exists(stats_path):
                try:
                    os.remove(stats_path)
                except Exception:
                    pass

            self._run(
                "native:savefeatures",
                {
                    "INPUT": inter_single,
                    "OUTPUT": stats_path,
                    "LAYER_NAME": stats_name,
                },
                context,
                feedback,
                "save_stats",
            )
            feedback.pushInfo(f"[Stats] {stats_name} written: {stats_path}")
        else:
            feedback.pushInfo("[Stats] Segments not provided/valid — stats layer skipped.")

        # Final Processing output: EA composite path
        return {self.PARAM_OUTPUT: final_path}
