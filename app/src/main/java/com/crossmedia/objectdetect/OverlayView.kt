package com.crossmedia.objectdetect

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.util.TypedValue
import android.view.View

/**
 * Draws a skeleton per detected person over the camera preview. Keypoints use
 * normalized 0..1 coordinates over the square center-crop of the preview; this
 * view is laid out to exactly cover that crop region, so mapping is a simple
 * scale.
 *
 * A joint below [Detector.KEYPOINT_THRESHOLD] is not drawn, and a bone is drawn
 * only when both of its ends clear the threshold — an occluded wrist reported at
 * low confidence would otherwise drag a limb across the display.
 */
class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private val jointRadius = dp(5f)

    private var poses: List<Pose> = emptyList()

    private val bonePaint = Paint().apply {
        isAntiAlias = true
        style = Paint.Style.STROKE
        strokeWidth = dp(4f)
        strokeCap = Paint.Cap.ROUND
        color = Color.WHITE
    }
    private val jointPaint = Paint().apply {
        isAntiAlias = true
        style = Paint.Style.FILL
        color = Color.GREEN
    }

    // Reused by point() so onDraw stays allocation-free.
    private val from = FloatArray(2)
    private val to = FloatArray(2)

    fun setPoses(list: List<Pose>) {
        poses = list
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()

        for (pose in poses) {
            val k = pose.keypoints

            var e = 0
            while (e < SKELETON.size) {
                if (point(k, SKELETON[e], w, h, from) && point(k, SKELETON[e + 1], w, h, to)) {
                    canvas.drawLine(from[0], from[1], to[0], to[1], bonePaint)
                }
                e += 2
            }

            for (i in 0 until YoloPostProcessor.KEYPOINT_COUNT) {
                if (point(k, i, w, h, from)) {
                    canvas.drawCircle(from[0], from[1], jointRadius, jointPaint)
                }
            }
        }
    }

    /**
     * Writes keypoint [i] of [keypoints] into [out] as view coordinates, and
     * reports whether it is confident enough to draw. [out] is left untouched
     * when it is not.
     */
    internal fun point(keypoints: FloatArray, i: Int, w: Float, h: Float, out: FloatArray): Boolean {
        val at = i * 3
        if (keypoints[at + 2] < Detector.KEYPOINT_THRESHOLD) return false
        out[0] = keypoints[at] * w
        out[1] = keypoints[at + 1] * h
        return true
    }

    private fun dp(v: Float) =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics)

    internal companion object {
        /**
         * The standard COCO 17-keypoint skeleton as flat index pairs: legs, hips,
         * torso, arms, then the face.
         */
        val SKELETON = intArrayOf(
            15, 13, 13, 11, 16, 14, 14, 12, 11, 12,
            5, 11, 6, 12, 5, 6, 5, 7, 6, 8, 7, 9, 8, 10,
            1, 2, 0, 1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6,
        )
    }
}
