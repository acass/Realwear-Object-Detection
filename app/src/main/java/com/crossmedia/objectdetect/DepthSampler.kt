package com.crossmedia.objectdetect

/**
 * Reduces a depth map region to one distance for a detection box.
 *
 * No Android types, so the logic is testable on the desktop JVM like
 * [YoloPostProcessor]. Boxes arrive normalized 0..1 against the same square crop the
 * depth map was produced from, so indexing is a plain multiply -- that shared 320px
 * crop is why the detector and the depth model run at the same input size.
 */
class DepthSampler(
    private val width: Int,
    private val height: Int,
    private val maxRangeMetres: Float = MAX_RANGE_METRES,
    private val centreFraction: Float = CENTRE_FRACTION,
) {

    companion object {
        /**
         * Beyond this a monocular nano model at 320px is guessing, so show no number
         * rather than a confident wrong one. The head's own ceiling is exp(5) = 148 m.
         */
        const val MAX_RANGE_METRES = 20f

        /**
         * Fraction of the box, about its centre, that is actually sampled.
         *
         * A detection box for a person is mostly background seen between the limbs; its
         * middle half is torso. Sampling the whole box biases far on anything thin or
         * gappy -- people, ladders, railings.
         */
        const val CENTRE_FRACTION = 0.5f
    }

    /** Scratch for the median. Grown on demand, reused across frames and boxes. */
    private var scratch = FloatArray(64)

    /**
     * Median depth over the central [centreFraction] of [box], in metres, or null when
     * the region is empty or the result is beyond [maxRangeMetres].
     *
     * @param depth row-major [width] x [height] depth map in metres.
     */
    fun sample(depth: FloatArray, left: Float, top: Float, right: Float, bottom: Float): Float? {
        val inset = (1f - centreFraction) / 2f
        val bw = right - left
        val bh = bottom - top

        // Round outward from the centre so a box thinner than one pixel still samples
        // the pixel it sits on rather than collapsing to an empty range.
        var x0 = ((left + bw * inset) * width).toInt()
        var x1 = ((right - bw * inset) * width).toInt()
        var y0 = ((top + bh * inset) * height).toInt()
        var y1 = ((bottom - bh * inset) * height).toInt()
        if (x1 <= x0) x1 = x0 + 1
        if (y1 <= y0) y1 = y0 + 1

        x0 = x0.coerceIn(0, width - 1)
        y0 = y0.coerceIn(0, height - 1)
        x1 = x1.coerceIn(x0 + 1, width)
        y1 = y1.coerceIn(y0 + 1, height)

        val count = (x1 - x0) * (y1 - y0)
        if (count <= 0) return null
        if (scratch.size < count) scratch = FloatArray(count)

        var n = 0
        for (y in y0 until y1) {
            val row = y * width
            for (x in x0 until x1) scratch[n++] = depth[row + x]
        }

        java.util.Arrays.sort(scratch, 0, n)
        val median = if (n % 2 == 1) scratch[n / 2]
        else (scratch[n / 2 - 1] + scratch[n / 2]) / 2f

        return if (median > maxRangeMetres || median.isNaN()) null else median
    }
}
