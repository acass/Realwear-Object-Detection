package com.crossmedia.objectdetect

import android.graphics.RectF

/**
 * Decodes raw YOLO pose output into [Pose]s: confidence threshold, then non-max
 * suppression on the person boxes.
 *
 * Split out of [Detector] because the TensorFlow Lite AAR ships Android-ABI
 * natives only — anything sitting behind an `Interpreter` cannot run on the
 * desktop JVM, so the geometry would be untestable in place.
 *
 * [out] is laid out `[numChannels][numBoxes]`: channels 0..3 are cx, cy, w, h,
 * channel 4 is the person score, and 5.. are [KEYPOINT_COUNT] triples of
 * (x, y, confidence). Emitted keypoints are normalized 0..1 relative to the
 * square model input. The box is decoded only to drive suppression and is then
 * discarded — nothing downstream draws it.
 */
class YoloPostProcessor(
    private val numChannels: Int,
    private val numBoxes: Int,
    private val inputSize: Int,
    private val confidenceThreshold: Float,
    private val iouThreshold: Float,
) {

    companion object {
        /** COCO keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles. */
        const val KEYPOINT_COUNT = 17
        private const val KEYPOINT_OFFSET = 5
    }

    /** A decoded person, with the box kept only until suppression is done. */
    internal class Candidate(val box: RectF, val pose: Pose)

    fun process(out: Array<FloatArray>): List<Pose> {
        // Ultralytics exports normalize coordinates to 0..1, but not every export
        // does, and box and keypoint spaces have not always agreed. Guard each
        // independently so a mixed export still decodes.
        val boxDiv = coordDiv(maxOf(channelMax(out, 2), channelMax(out, 3)))
        var keypointMax = 0f
        for (i in 0 until KEYPOINT_COUNT) {
            keypointMax = maxOf(
                keypointMax,
                channelMax(out, KEYPOINT_OFFSET + i * 3),
                channelMax(out, KEYPOINT_OFFSET + i * 3 + 1),
            )
        }
        val keypointDiv = coordDiv(keypointMax)

        val candidates = ArrayList<Candidate>()
        for (b in 0 until numBoxes) {
            val score = out[4][b]
            if (score < confidenceThreshold) continue

            val cx = out[0][b] / boxDiv
            val cy = out[1][b] / boxDiv
            val w = out[2][b] / boxDiv
            val h = out[3][b] / boxDiv

            val keypoints = FloatArray(KEYPOINT_COUNT * 3)
            for (i in 0 until KEYPOINT_COUNT) {
                val c = KEYPOINT_OFFSET + i * 3
                keypoints[i * 3] = out[c][b] / keypointDiv
                keypoints[i * 3 + 1] = out[c + 1][b] / keypointDiv
                keypoints[i * 3 + 2] = out[c + 2][b]      // confidence, not a coordinate
            }

            candidates.add(
                Candidate(
                    RectF(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                    Pose(keypoints, score)
                )
            )
        }
        return nms(candidates).map { it.pose }
    }

    private fun channelMax(out: Array<FloatArray>, channel: Int): Float {
        var max = 0f
        for (b in 0 until numBoxes) if (out[channel][b] > max) max = out[channel][b]
        return max
    }

    private fun coordDiv(max: Float) = if (max > 1.5f) inputSize.toFloat() else 1f

    internal fun nms(candidates: List<Candidate>): List<Candidate> {
        val sorted = candidates.sortedByDescending { it.pose.score }.toMutableList()
        val kept = ArrayList<Candidate>()
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
