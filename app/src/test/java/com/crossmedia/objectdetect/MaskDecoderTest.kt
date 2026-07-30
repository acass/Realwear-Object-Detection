package com.crossmedia.objectdetect

import android.graphics.RectF
import java.nio.FloatBuffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * A 4x4x2 prototype tensor rather than the real 160x160x32: the synthesis is
 * per-pixel over a dot product, so a handful of hand-checkable values exercise
 * every branch.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class MaskDecoderTest {

    private val protoW = 4
    private val protoH = 4
    private val protoC = 2

    private fun decoder(threshold: Float = 0.5f) =
        MaskDecoder(protoW, protoH, protoC, threshold)

    /**
     * [values] is one FloatArray per channel, each laid out row-major over
     * protoH x protoW. Concatenated into the flat channels-first [c][y][x]
     * layout the decoder reads.
     */
    private fun proto(vararg values: FloatArray): FloatBuffer {
        val flat = FloatArray(protoW * protoH * protoC)
        for (c in values.indices) {
            values[c].copyInto(flat, c * protoW * protoH)
        }
        return FloatBuffer.wrap(flat)
    }

    /** Every proto pixel set to [v] on channel 0, zero on channel 1. */
    private fun uniform(v: Float) = proto(FloatArray(protoW * protoH) { v }, FloatArray(protoW * protoH))

    private val ON = 255.toByte()
    private val OFF = 0.toByte()

    private fun full() = RectF(0f, 0f, 1f, 1f)

    @Test
    fun `positive logit over the whole frame fills the mask`() {
        val m = decoder().decode(uniform(1f), floatArrayOf(1f, 0f), full())!!

        assertEquals(protoW, m.width)
        assertEquals(protoH, m.height)
        assertEquals(protoW * protoH, m.alpha.size)
        m.alpha.forEach { assertEquals(ON, it) }
    }

    @Test
    fun `negative logit over the whole frame empties the mask`() {
        val m = decoder().decode(uniform(-1f), floatArrayOf(1f, 0f), full())!!
        m.alpha.forEach { assertEquals(OFF, it) }
    }

    @Test
    fun `coefficient sign flips the mask`() {
        val m = decoder().decode(uniform(1f), floatArrayOf(-1f, 0f), full())!!
        m.alpha.forEach { assertEquals(OFF, it) }
    }

    @Test
    fun `sums across channels before thresholding`() {
        // 0.5 * 2 + (-1) * 1 = 0 exactly, which is sigmoid 0.5 - not above threshold.
        val p = proto(
            FloatArray(protoW * protoH) { 2f },
            FloatArray(protoW * protoH) { 1f },
        )
        assertEquals(OFF, decoder().decode(p, floatArrayOf(0.5f, -1f), full())!!.alpha[0])
        // Nudge one coefficient up and the same pixel crosses.
        assertEquals(ON, decoder().decode(p, floatArrayOf(0.6f, -1f), full())!!.alpha[0])
    }

    @Test
    fun `threshold boundary sits either side of sigmoid one half`() {
        // Raw logit 0 is exactly sigmoid 0.5. Strictly-greater, so it is off.
        assertEquals(OFF, decoder().decode(uniform(0f), floatArrayOf(1f, 0f), full())!!.alpha[0])
        assertEquals(ON, decoder().decode(uniform(0.01f), floatArrayOf(1f, 0f), full())!!.alpha[0])
        assertEquals(OFF, decoder().decode(uniform(-0.01f), floatArrayOf(1f, 0f), full())!!.alpha[0])
    }

    @Test
    fun `a higher threshold demands a stronger logit`() {
        // sigmoid(1) is about 0.731, so it clears 0.5 but not 0.9.
        assertEquals(ON, decoder(0.5f).decode(uniform(1f), floatArrayOf(1f, 0f), full())!!.alpha[0])
        assertEquals(OFF, decoder(0.9f).decode(uniform(1f), floatArrayOf(1f, 0f), full())!!.alpha[0])
        assertEquals(ON, decoder(0.9f).decode(uniform(3f), floatArrayOf(1f, 0f), full())!!.alpha[0])
    }

    @Test
    fun `mask covers only the box, in proto pixels`() {
        // Right half of a 4-wide proto: x 2..3.
        val m = decoder().decode(uniform(1f), floatArrayOf(1f, 0f), RectF(0.5f, 0f, 1f, 1f))!!
        assertEquals(2, m.width)
        assertEquals(4, m.height)
        assertEquals(8, m.alpha.size)
    }

    @Test
    fun `reads the pixels under the box and not its neighbours`() {
        // Channel 0 positive only in the top-left proto pixel.
        val ch0 = FloatArray(protoW * protoH) { -1f }
        ch0[0] = 1f
        val p = proto(ch0, FloatArray(protoW * protoH))

        // A box over just that pixel is on; the pixel to its right is off.
        val topLeft = decoder().decode(p, floatArrayOf(1f, 0f), RectF(0f, 0f, 0.25f, 0.25f))!!
        assertEquals(1, topLeft.width)
        assertEquals(1, topLeft.height)
        assertEquals(ON, topLeft.alpha[0])

        val next = decoder().decode(p, floatArrayOf(1f, 0f), RectF(0.25f, 0f, 0.5f, 0.25f))!!
        assertEquals(OFF, next.alpha[0])
    }

    @Test
    fun `row major order follows the crop, not the whole frame`() {
        // Positive only at proto (x=3, y=1), inside a box covering x 2..3, y 0..1.
        val ch0 = FloatArray(protoW * protoH) { -1f }
        ch0[1 * protoW + 3] = 1f
        val p = proto(ch0, FloatArray(protoW * protoH))

        val m = decoder().decode(p, floatArrayOf(1f, 0f), RectF(0.5f, 0f, 1f, 0.5f))!!
        assertEquals(2, m.width)
        assertEquals(2, m.height)
        // Crop-local (x=1, y=1) is the last entry of a 2x2.
        assertEquals(OFF, m.alpha[0])
        assertEquals(OFF, m.alpha[1])
        assertEquals(OFF, m.alpha[2])
        assertEquals(ON, m.alpha[3])
    }

    @Test
    fun `returned box snaps to proto pixel edges`() {
        // 0.3 * 4 = 1.2 floors to 1; 0.6 * 4 = 2.4 ceils to 3.
        val m = decoder().decode(uniform(1f), floatArrayOf(1f, 0f), RectF(0.3f, 0.3f, 0.6f, 0.6f))!!

        assertEquals(2, m.width)
        assertEquals(2, m.height)
        assertEquals(0.25f, m.box.left, 1e-6f)
        assertEquals(0.25f, m.box.top, 1e-6f)
        assertEquals(0.75f, m.box.right, 1e-6f)
        assertEquals(0.75f, m.box.bottom, 1e-6f)
    }

    @Test
    fun `a box running off the frame is clamped, not wrapped`() {
        val m = decoder().decode(uniform(1f), floatArrayOf(1f, 0f), RectF(-0.5f, -0.5f, 1.5f, 1.5f))!!

        assertEquals(protoW, m.width)
        assertEquals(protoH, m.height)
        assertEquals(0f, m.box.left, 1e-6f)
        assertEquals(1f, m.box.right, 1e-6f)
    }

    @Test
    fun `a box partly off the frame keeps only the visible part`() {
        val m = decoder().decode(uniform(1f), floatArrayOf(1f, 0f), RectF(-0.5f, 0f, 0.25f, 1f))!!

        assertEquals(1, m.width)
        assertEquals(0f, m.box.left, 1e-6f)
        assertEquals(0.25f, m.box.right, 1e-6f)
    }

    @Test
    fun `a zero area box yields no mask`() {
        assertNull(decoder().decode(uniform(1f), floatArrayOf(1f, 0f), RectF(0.5f, 0.5f, 0.5f, 0.5f)))
    }

    @Test
    fun `a box entirely outside the frame yields no mask`() {
        assertNull(decoder().decode(uniform(1f), floatArrayOf(1f, 0f), RectF(1.5f, 0f, 2f, 1f)))
        assertNull(decoder().decode(uniform(1f), floatArrayOf(1f, 0f), RectF(-2f, 0f, -1.5f, 1f)))
    }

    @Test
    fun `an inverted box yields no mask`() {
        assertNull(decoder().decode(uniform(1f), floatArrayOf(1f, 0f), RectF(0.75f, 0f, 0.25f, 1f)))
    }
}
