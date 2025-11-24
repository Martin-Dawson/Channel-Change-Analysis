# -*- coding: utf-8 -*-
from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFeatureSink,
    QgsFeature,
    QgsFields,
    QgsField,
    QgsWkbTypes,
    QgsPointXY,
    QgsGeometry,
    QgsFeatureSink,
    QgsProcessingException,
)

import numpy as np
from scipy.signal import savgol_filter


class SGSmoothAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    WINDOW = "WINDOW"
    ORDER = "ORDER"
    OUTPUT_POINTS = "OUTPUT_POINTS"
    OUTPUT_LINE = "OUTPUT_LINE"

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                "Input Line Layer",
                [QgsProcessing.TypeVectorLine],
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.WINDOW,
                "Smoothing Window Length (odd ≥ 3)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=25,
                minValue=3,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.ORDER,
                "Polynomial Order",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_POINTS,
                "Smoothed Points",
                type=QgsProcessing.TypeVectorPoint,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_LINE,
                "Smoothed Line",
                type=QgsProcessing.TypeVectorLine,
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        window = self.parameterAsInt(parameters, self.WINDOW, context)
        order = self.parameterAsInt(parameters, self.ORDER, context)

        if source is None:
            raise QgsProcessingException("Invalid input source.")

        # Global sanity check
        if window < 3:
            raise QgsProcessingException(
                "Smoothing window must be at least 3."
            )
        if order < 1:
            raise QgsProcessingException(
                "Polynomial order must be at least 1."
            )
        if window <= order:
            raise QgsProcessingException(
                "Smoothing window must be greater than polynomial order."
            )

        # Prepare fields for points (carry original FID as well)
        point_fields = QgsFields()
        point_fields.append(QgsField("src_fid", QVariant.LongLong))
        point_fields.append(QgsField("pt_id", QVariant.Int))

        (point_sink, point_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT_POINTS,
            context,
            point_fields,
            QgsWkbTypes.Point,
            source.sourceCrs(),
        )

        (line_sink, line_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT_LINE,
            context,
            QgsFields(),  # no attributes on the line by default
            QgsWkbTypes.LineString,
            source.sourceCrs(),
        )

        if point_sink is None or line_sink is None:
            raise QgsProcessingException("Could not create output sinks.")

        features = list(source.getFeatures())
        if not features:
            raise QgsProcessingException("Input layer has no features.")

        total = len(features)
        for i, f in enumerate(features):
            if feedback.isCanceled():
                break

            geom = f.geometry()
            if geom is None or geom.isEmpty():
                continue
            if geom.type() != QgsWkbTypes.LineGeometry:
                continue

            # Handle single- and multi-part lines
            lines = []
            if geom.isMultipart():
                lines = geom.asMultiPolyline()
            else:
                pts = geom.asPolyline()
                if pts:
                    lines = [pts]

            if not lines:
                continue

            for line in lines:
                n = len(line)
                if n < 3:
                    # Too few points to smooth meaningfully
                    continue

                # Determine an effective window for this line:
                # - must be odd
                # - at least 3
                # - <= n
                # - > order
                eff_window = min(window, n)
                if eff_window % 2 == 0:
                    eff_window -= 1
                if eff_window < 3:
                    continue
                if eff_window <= order:
                    # Either reduce order or skip. Here we skip short lines.
                    continue

                x = np.array([p.x() for p in line], dtype=float)
                y = np.array([p.y() for p in line], dtype=float)

                try:
                    x_smooth = savgol_filter(
                        x, window_length=eff_window, polyorder=order
                    )
                    y_smooth = savgol_filter(
                        y, window_length=eff_window, polyorder=order
                    )
                except Exception as e:
                    # If SciPy complains, skip this line but continue overall
                    feedback.reportError(
                        f"Skipping feature {f.id()} due to Savitzky-Golay error: {e}"
                    )
                    continue

                smoothed_pts = [
                    QgsPointXY(xs, ys)
                    for xs, ys in zip(x_smooth, y_smooth)
                ]

                # Add smoothed points
                for j, pt in enumerate(smoothed_pts):
                    feat = QgsFeature(point_fields)
                    feat.setAttribute("src_fid", int(f.id()))
                    feat.setAttribute("pt_id", j)
                    feat.setGeometry(QgsGeometry.fromPointXY(pt))
                    point_sink.addFeature(
                        feat, QgsFeatureSink.FastInsert
                    )

                # Add smoothed line
                line_feat = QgsFeature()
                line_feat.setGeometry(
                    QgsGeometry.fromPolylineXY(smoothed_pts)
                )
                line_sink.addFeature(
                    line_feat, QgsFeatureSink.FastInsert
                )

            feedback.setProgress(int(100.0 * i / total))

        return {
            self.OUTPUT_POINTS: point_id,
            self.OUTPUT_LINE: line_id,
        }

    def name(self):
        return "sg_smooth"

    def displayName(self):
        return "Savitzky-Golay Smooth Line"

    def group(self):
        return "Custom Scripts"

    def groupId(self):
        return "customscripts"

    def createInstance(self):
        return SGSmoothAlgorithm()
