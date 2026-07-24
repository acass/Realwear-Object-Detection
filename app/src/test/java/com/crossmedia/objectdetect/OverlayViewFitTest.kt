package com.crossmedia.objectdetect

import android.graphics.Rect
import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

/**
 * Robolectric supplies real RectF/Rect and a display metrics density, so the
 * callout placement geometry runs unmodified on the JVM.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class OverlayViewFitTest {

    private lateinit var overlay: OverlayView
    private val screen = Rect(0, 0, 1000, 1000)

    @Before
    fun setUp() {
        overlay = OverlayView(RuntimeEnvironment.getApplication(), null)
    }

    @Test
    fun `prefers the up-right quadrant when nothing blocks it`() {
        val pill = overlay.fit(500f, 500f, 100f, 40f, screen, emptyList())

        assertNotNull(pill)
        assertTrue("pill should sit right of the anchor", pill!!.left > 500f)
        assertTrue("pill should sit above the anchor", pill.bottom < 500f)
    }

    @Test
    fun `falls to another quadrant when up-right leaves the visible rect`() {
        // Anchor near the top edge: the up-right slot would run off the top.
        val pill = overlay.fit(500f, 10f, 100f, 40f, screen, emptyList())

        assertNotNull(pill)
        assertTrue("pill must stay inside the visible rect", pill!!.top >= screen.top)
        assertTrue(pill.bottom <= screen.bottom)
        assertTrue(pill.left >= screen.left)
        assertTrue(pill.right <= screen.right)
    }

    @Test
    fun `never returns a pill outside the visible rect`() {
        for (x in listOf(5f, 500f, 995f)) {
            for (y in listOf(5f, 500f, 995f)) {
                val pill = overlay.fit(x, y, 100f, 40f, screen, emptyList()) ?: continue
                assertTrue(
                    "pill at ($x,$y) escaped: $pill",
                    pill.left >= screen.left && pill.top >= screen.top &&
                        pill.right <= screen.right && pill.bottom <= screen.bottom
                )
            }
        }
    }

    @Test
    fun `does not overlap a pill already placed this frame`() {
        val first = overlay.fit(500f, 500f, 100f, 40f, screen, emptyList())!!
        val second = overlay.fit(500f, 500f, 100f, 40f, screen, listOf(first))

        assertNotNull(second)
        assertTrue("second pill overlaps the first", !RectF.intersects(first, second!!))
    }

    @Test
    fun `returns null when all four quadrants are blocked`() {
        // A pill wider than the visible rect cannot land anywhere.
        assertNull(overlay.fit(500f, 500f, 2000f, 40f, screen, emptyList()))
    }

    @Test
    fun `selects the five strongest detections in descending confidence`() {
        val input = (1..8).map { Detection(RectF(0f, 0f, 1f, 1f), "d$it", it / 10f) }
        overlay.selectStrongest(input)
        val got = overlay.strongestForTest()

        assertEquals(5, got.size)
        assertEquals(listOf("d8", "d7", "d6", "d5", "d4"), got.map { it.label })
    }

    @Test
    fun `selection handles fewer detections than the cap`() {
        val input = listOf(
            Detection(RectF(0f, 0f, 1f, 1f), "low", 0.2f),
            Detection(RectF(0f, 0f, 1f, 1f), "high", 0.9f),
        )
        overlay.selectStrongest(input)

        assertEquals(listOf("high", "low"), overlay.strongestForTest().map { it.label })
    }

    @Test
    fun `selection clears state between frames`() {
        overlay.selectStrongest(listOf(Detection(RectF(0f, 0f, 1f, 1f), "stale", 0.9f)))
        overlay.selectStrongest(emptyList())

        assertEquals(0, overlay.strongestForTest().size)
    }

    @Test
    fun `clipped visible rect is respected, not the view bounds`() {
        // Landscape case: the square overlay is taller than the screen, so the
        // visible band is a horizontal slice.
        val band = Rect(0, 400, 1000, 600)
        val pill = overlay.fit(500f, 500f, 100f, 40f, band, emptyList())

        if (pill != null) {
            assertTrue(pill.top >= band.top)
            assertTrue(pill.bottom <= band.bottom)
        }
    }
}
