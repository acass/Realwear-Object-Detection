package com.crossmedia.objectdetect

import android.graphics.Matrix
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlin.math.max
import kotlin.math.min

/**
 * The fused rotate + centre-crop + scale replaced three chained Bitmap copies, so
 * the property that used to be structural now has to be asserted: the centred
 * square of the rotated frame must land exactly on the model input, at every
 * rotation the camera can report.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class CropMatrixTest {

    private val size = 320
    private val matrix = Matrix()

    /** Maps a source point through the transform under test. */
    private fun map(srcW: Int, srcH: Int, rotation: Int, x: Float, y: Float): Pair<Float, Float> {
        MainActivity.cropMatrix(srcW, srcH, rotation, size, matrix)
        val p = floatArrayOf(x, y)
        matrix.mapPoints(p)
        return p[0] to p[1]
    }

    @Test
    fun `frame centre maps to the centre of the model input at every rotation`() {
        for (rotation in listOf(0, 90, 180, 270)) {
            val (x, y) = map(640, 480, rotation, 320f, 240f)
            assertEquals("rotation $rotation x", 160f, x, 1e-3f)
            assertEquals("rotation $rotation y", 160f, y, 1e-3f)
        }
    }

    @Test
    fun `the centred square fills the model input exactly at every rotation`() {
        val srcW = 640
        val srcH = 480
        for (rotation in listOf(0, 90, 180, 270)) {
            val rotated = rotation % 180 != 0
            val rotW = if (rotated) srcH else srcW
            val rotH = if (rotated) srcW else srcH
            val side = min(rotW, rotH)

            // Corners of the centred square, expressed back in source coordinates.
            val halfW = if (rotated) side / 2f else side / 2f
            val halfH = halfW
            val corners = listOf(
                srcW / 2f - halfW to srcH / 2f - halfH,
                srcW / 2f + halfW to srcH / 2f + halfH,
            )

            val mapped = corners.map { (x, y) -> map(srcW, srcH, rotation, x, y) }
            val xs = mapped.map { it.first }
            val ys = mapped.map { it.second }

            assertEquals("rotation $rotation left", 0f, xs.min(), 1e-3f)
            assertEquals("rotation $rotation right", size.toFloat(), xs.max(), 1e-3f)
            assertEquals("rotation $rotation top", 0f, ys.min(), 1e-3f)
            assertEquals("rotation $rotation bottom", size.toFloat(), ys.max(), 1e-3f)
        }
    }

    @Test
    fun `the long axis overflows the canvas and is clipped, never letterboxed`() {
        // 640x480 at rotation 0: the 480 axis fits, the 640 axis must overflow.
        val (left, _) = map(640, 480, 0, 0f, 240f)
        val (right, _) = map(640, 480, 0, 640f, 240f)

        assert(left < 0f) { "left edge should overflow, was $left" }
        assert(right > size) { "right edge should overflow, was $right" }
        // Symmetric overflow means the crop is centred.
        assertEquals(-left, right - size, 1e-3f)
    }

    @Test
    fun `a square frame maps corner to corner with no crop`() {
        val (x0, y0) = map(480, 480, 0, 0f, 0f)
        val (x1, y1) = map(480, 480, 0, 480f, 480f)

        assertEquals(0f, x0, 1e-3f)
        assertEquals(0f, y0, 1e-3f)
        assertEquals(size.toFloat(), x1, 1e-3f)
        assertEquals(size.toFloat(), y1, 1e-3f)
    }

    @Test
    fun `rotation by 90 sends the source top edge to the model right edge`() {
        // A point on the top edge, centred horizontally, rotates to the right edge.
        val (x, y) = map(480, 480, 90, 240f, 0f)

        assertEquals(size.toFloat(), x, 1e-3f)
        assertEquals(size / 2f, y, 1e-3f)
    }

    @Test
    fun `portrait frames crop the height rather than the width`() {
        val srcW = 480
        val srcH = 640
        val (top, bottom) = listOf(0f, 640f).map { map(srcW, srcH, 0, 240f, it).second }
        val span = max(top, bottom) - min(top, bottom)

        assert(span > size) { "the 640 axis should overflow, span was $span" }
        val (left, right) = listOf(0f, 480f).map { map(srcW, srcH, 0, it, 320f).first }
        assertEquals(0f, min(left, right), 1e-3f)
        assertEquals(size.toFloat(), max(left, right), 1e-3f)
    }
}
