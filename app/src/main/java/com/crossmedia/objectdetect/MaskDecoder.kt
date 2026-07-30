package com.crossmedia.objectdetect

import android.graphics.RectF
import java.nio.FloatBuffer
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.ln

/**
 * One instance mask in prototype space, already cropped to its detection's box.
 *
 * [alpha] is row-major over [width] x [height], one byte per pixel, 0 or 255 -
 * exactly the layout an ALPHA_8 bitmap wants. [box] is where to draw it,
 * normalized 0..1 over the square model input and snapped to prototype pixel
 * edges, so it is the crop's true extent rather than the detection's raw box.
 */
class MaskBits(
    val width: Int,
    val height: Int,
    val alpha: ByteArray,
    val box: RectF,
)

/**
 * Synthesizes YOLOv8-seg instance masks: `sigmoid(coeffs . proto)` thresholded,
 * then cropped to the detection's box.
 *
 * Split out of [Detector] for the same reason as [YoloPostProcessor] - the
 * TensorFlow Lite AAR ships Android-ABI natives only, so anything sitting behind
 * an `Interpreter` cannot run on the desktop JVM, and this is index arithmetic
 * that badly wants testing.
 *
 * [proto] is read as channels-first `[c][y][x]`, matching the `[1, C, H, W]`
 * prototype tensor this export produces - Ultralytics builds TFLite through
 * LiteRT from PyTorch now, so the tensors come out channels-first rather than
 * the channels-last layout older TFLite exports had.
 *
 * The box crop is not an optimization detail - an uncropped mask bleeds onto
 * neighbouring objects. It is also what keeps this affordable: only the pixels
 * under the box are ever touched, so a quarter-frame box costs a sixteenth of a
 * full-frame synthesis.
 */
class MaskDecoder(
    private val protoW: Int,
    private val protoH: Int,
    private val protoC: Int,
    threshold: Float = 0.5f,
) {

    private val planeStride = protoW * protoH

    /**
     * `sigmoid(x) > t` is `x > ln(t / (1 - t))`, so the comparison happens on the
     * raw dot product and the per-pixel exp() never runs.
     */
    private val logitThreshold = ln(threshold / (1f - threshold))

    /**
     * Returns null when [box] covers no whole prototype pixel - off-frame,
     * inverted, or too small to survive the snap to the prototype grid.
     * [coeffs] must hold [protoC] mask coefficients for this detection.
     */
    fun decode(proto: FloatBuffer, coeffs: FloatArray, box: RectF): MaskBits? {
        // Snap outwards so the crop covers the box completely; the threshold
        // decides the actual silhouette edge within it.
        val x0 = floor(box.left * protoW).toInt().coerceIn(0, protoW)
        val y0 = floor(box.top * protoH).toInt().coerceIn(0, protoH)
        val x1 = ceil(box.right * protoW).toInt().coerceIn(0, protoW)
        val y1 = ceil(box.bottom * protoH).toInt().coerceIn(0, protoH)

        val w = x1 - x0
        val h = y1 - y0
        if (w <= 0 || h <= 0) return null

        // Channel-outer: each prototype plane is walked in row order, so the
        // crop's rows are contiguous reads. Pixel-outer would stride a whole
        // plane between every channel of every pixel.
        val acc = FloatArray(w * h)
        for (c in 0 until protoC) {
            val k = coeffs[c]
            if (k == 0f) continue
            val plane = c * planeStride
            var i = 0
            for (y in y0 until y1) {
                var p = plane + y * protoW + x0
                repeat(w) { acc[i++] += k * proto.get(p++) }
            }
        }

        val alpha = ByteArray(w * h)
        for (i in acc.indices) if (acc[i] > logitThreshold) alpha[i] = ON

        return MaskBits(
            w, h, alpha,
            RectF(
                x0.toFloat() / protoW,
                y0.toFloat() / protoH,
                x1.toFloat() / protoW,
                y1.toFloat() / protoH,
            ),
        )
    }

    private companion object {
        /** Opaque. ALPHA_8 reads this byte unsigned, so the sign here is irrelevant. */
        const val ON: Byte = 255.toByte()
    }
}
