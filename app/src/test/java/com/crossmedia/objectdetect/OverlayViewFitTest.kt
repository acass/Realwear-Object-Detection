package com.crossmedia.objectdetect

import android.graphics.Color
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
    fun `each class gets its own colour`() {
        assertTrue(overlay.colourFor(0) != overlay.colourFor(1))
        assertTrue(overlay.colourFor(0) != overlay.colourFor(40))
    }

    @Test
    fun `the same class always gets the same colour`() {
        assertEquals(overlay.colourFor(17), overlay.colourFor(17))
    }

    @Test
    fun `masks are translucent so the object stays visible`() {
        for (id in listOf(0, 1, 79)) {
            val alpha = Color.alpha(overlay.colourFor(id))
            assertTrue("class $id alpha $alpha should be translucent", alpha in 1..200)
        }
    }

    @Test
    fun `an out of range or missing class id still yields a colour`() {
        // A model with more classes than the table, or a detection built without
        // an id, must not take the overlay down.
        assertEquals(overlay.colourFor(0), overlay.colourFor(-1))
        assertEquals(overlay.colourFor(0), overlay.colourFor(80))
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
