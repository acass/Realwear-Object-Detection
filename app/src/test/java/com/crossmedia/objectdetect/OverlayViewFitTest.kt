package com.crossmedia.objectdetect

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

/**
 * Robolectric supplies a real display metrics density, so the keypoint mapping
 * runs unmodified on the JVM.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class OverlayViewFitTest {

    private lateinit var overlay: OverlayView
    private val out = FloatArray(2)

    @Before
    fun setUp() {
        overlay = OverlayView(RuntimeEnvironment.getApplication(), null)
    }

    /** All 17 keypoints at ([x], [y]) with confidence [conf]. */
    private fun keypoints(x: Float, y: Float, conf: Float): FloatArray {
        val k = FloatArray(YoloPostProcessor.KEYPOINT_COUNT * 3)
        for (i in 0 until YoloPostProcessor.KEYPOINT_COUNT) {
            k[i * 3] = x; k[i * 3 + 1] = y; k[i * 3 + 2] = conf
        }
        return k
    }

    @Test
    fun `scales a normalized keypoint to view coordinates`() {
        assertTrue(overlay.point(keypoints(0.25f, 0.75f, 0.9f), 0, 800f, 400f, out))

        assertEquals(200f, out[0], 1e-4f)
        assertEquals(300f, out[1], 1e-4f)
    }

    @Test
    fun `maps each keypoint index to its own triple`() {
        val k = keypoints(0f, 0f, 0.9f)
        k[16 * 3] = 1f
        k[16 * 3 + 1] = 0.5f

        assertTrue(overlay.point(k, 16, 100f, 100f, out))
        assertEquals(100f, out[0], 1e-4f)
        assertEquals(50f, out[1], 1e-4f)
    }

    @Test
    fun `skips a keypoint below the confidence threshold and leaves out untouched`() {
        val stale = floatArrayOf(-1f, -1f)
        val below = Detector.KEYPOINT_THRESHOLD - 0.01f

        assertFalse(overlay.point(keypoints(0.5f, 0.5f, below), 0, 100f, 100f, stale))
        assertArrayEquals(floatArrayOf(-1f, -1f), stale, 1e-6f)
    }

    @Test
    fun `draws a keypoint exactly at the threshold`() {
        assertTrue(
            overlay.point(keypoints(0.5f, 0.5f, Detector.KEYPOINT_THRESHOLD), 0, 100f, 100f, out)
        )
    }

    @Test
    fun `skeleton is whole pairs of in-range keypoint indices`() {
        assertEquals(0, OverlayView.SKELETON.size % 2)
        for (i in OverlayView.SKELETON) {
            assertTrue("index $i out of range", i in 0 until YoloPostProcessor.KEYPOINT_COUNT)
        }
    }

    @Test
    fun `skeleton connects every keypoint at least once`() {
        // A joint no bone reaches would render as a floating dot.
        val connected = OverlayView.SKELETON.toSet()
        for (i in 0 until YoloPostProcessor.KEYPOINT_COUNT) {
            assertTrue("keypoint $i has no bone", i in connected)
        }
    }
}
