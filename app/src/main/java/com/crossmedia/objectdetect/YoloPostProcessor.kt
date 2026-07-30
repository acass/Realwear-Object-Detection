package com.crossmedia.objectdetect

import android.graphics.RectF

/**
 * Decodes raw YOLOv8 output into [Detection]s: best class per box, confidence
 * threshold, then non-max suppression.
 *
 * Split out of [Detector] because the TensorFlow Lite AAR ships Android-ABI
 * natives only — anything sitting behind an `Interpreter` cannot run on the
 * desktop JVM, so the geometry would be untestable in place.
 *
 * [out] is laid out `[channels][numBoxes]`: channels 0..3 are cx, cy, w, h and
 * the next [numClasses] are per-class scores. Emitted boxes are normalized 0..1
 * relative to the square model input.
 *
 * Anything after the class scores is ignored - a segmentation export carries 32
 * mask coefficients there, and they are not class scores however much the argmax
 * would like them to be.
 */
class YoloPostProcessor(
    private val numClasses: Int,
    private val numBoxes: Int,
    private val inputSize: Int,
    private val labels: List<String>,
    private val confidenceThreshold: Float,
    private val iouThreshold: Float,
) {

    /**
     * When [targetLabel] is set only that class survives, and it is applied
     * before [maxDetections] - a guided procedure's target must keep its place
     * even in a scene where it is not among the strongest few detections.
     *
     * [nms] emits in descending confidence, so the cap is a plain take: the
     * detections kept are the strongest ones, in the order the overlay wants
     * them for placement priority.
     */
    fun process(
        out: Array<FloatArray>,
        targetLabel: String? = null,
        maxDetections: Int = Int.MAX_VALUE,
    ): List<Detection> {
        val candidates = ArrayList<Detection>()
        // Ultralytics TFLite exports normalize coords to 0..1; guard for pixel-space models.
        var coordMax = 0f
        for (b in 0 until numBoxes) {
            if (out[2][b] > coordMax) coordMax = out[2][b]
        }
        val coordDiv = if (coordMax > 1.5f) inputSize.toFloat() else 1f

        for (b in 0 until numBoxes) {
            var bestClass = -1
            var bestScore = 0f
            for (c in 4 until 4 + numClasses) {
                val s = out[c][b]
                if (s > bestScore) {
                    bestScore = s
                    bestClass = c - 4
                }
            }
            if (bestScore < confidenceThreshold) continue

            val cx = out[0][b] / coordDiv
            val cy = out[1][b] / coordDiv
            val w = out[2][b] / coordDiv
            val h = out[3][b] / coordDiv
            candidates.add(
                Detection(
                    RectF(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                    labels.getOrElse(bestClass) { "class $bestClass" },
                    bestScore,
                    classId = bestClass,
                    boxIndex = b,
                )
            )
        }
        var kept = nms(candidates)
        if (targetLabel != null) kept = kept.filter { it.label == targetLabel }
        return if (kept.size > maxDetections) kept.subList(0, maxDetections) else kept
    }

    internal fun nms(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val kept = ArrayList<Detection>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            kept.add(best)
            sorted.removeAll { iou(best.box, it.box) > iouThreshold }
        }
        return kept
    }

    internal fun iou(a: RectF, b: RectF): Float {
        val left = maxOf(a.left, b.left)
        val top = maxOf(a.top, b.top)
        val right = minOf(a.right, b.right)
        val bottom = minOf(a.bottom, b.bottom)
        if (right <= left || bottom <= top) return 0f
        val inter = (right - left) * (bottom - top)
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }
}
