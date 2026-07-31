package com.crossmedia.objectdetect

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * DepthSampler holds no Android types, so these run on the plain JVM.
 *
 * The load-bearing case is [centreHalfIgnoresBorderBackground]: it fails if the sampler
 * ever reverts to averaging the whole box, which is the bias that makes a person read as
 * the wall behind them.
 */
class DepthSamplerTest {

    private val w = 100
    private val h = 100

    /** Depth map where [f] maps (x, y) to metres. */
    private fun map(f: (Int, Int) -> Float) =
        FloatArray(w * h) { f(it % w, it / w) }

    @Test
    fun uniformRegionReturnsThatDepth() {
        val s = DepthSampler(w, h)
        assertEquals(3.5f, s.sample(map { _, _ -> 3.5f }, 0.2f, 0.2f, 0.8f, 0.8f)!!, 1e-4f)
    }

    @Test
    fun centreHalfIgnoresBorderBackground() {
        // Box spans 20..80 in both axes. Its central half is 35..65. Everything inside
        // 35..65 is near; the surrounding ring inside the box is far background and
        // outnumbers it, so a full-box median would return the far value.
        val depth = map { x, y -> if (x in 35..64 && y in 35..64) 2f else 40f }
        val s = DepthSampler(w, h)
        assertEquals(2f, s.sample(depth, 0.2f, 0.2f, 0.8f, 0.8f)!!, 1e-4f)
    }

    @Test
    fun fullBoxWouldReturnTheBackgroundInstead() {
        // Guards the test above by proving it discriminates: the identical map sampled
        // over the whole box returns the far background, which is precisely the bias the
        // centre crop exists to remove. If centreFraction ever regresses to 1.0, the
        // previous test fails and this one is the reason why.
        val depth = map { x, y -> if (x in 35..64 && y in 35..64) 2f else 40f }
        val s = DepthSampler(w, h, maxRangeMetres = 100f, centreFraction = 1f)
        assertEquals(40f, s.sample(depth, 0.2f, 0.2f, 0.8f, 0.8f)!!, 1e-4f)
    }

    @Test
    fun medianRejectsOutliers() {
        // A quarter of the sampled region is absurdly near; the median must ignore it.
        val depth = map { x, y -> if (x < 50 && y < 50) 0.05f else 6f }
        val s = DepthSampler(w, h)
        assertEquals(6f, s.sample(depth, 0.25f, 0.25f, 0.95f, 0.95f)!!, 1e-4f)
    }

    @Test
    fun beyondMaxRangeReturnsNull() {
        val s = DepthSampler(w, h, maxRangeMetres = 20f)
        assertNull(s.sample(map { _, _ -> 20.01f }, 0.2f, 0.2f, 0.8f, 0.8f))
    }

    @Test
    fun atMaxRangeStillReturnsValue() {
        val s = DepthSampler(w, h, maxRangeMetres = 20f)
        assertEquals(20f, s.sample(map { _, _ -> 20f }, 0.2f, 0.2f, 0.8f, 0.8f)!!, 1e-4f)
    }

    @Test
    fun degenerateBoxStillSamplesOnePixel() {
        // A box thinner than a pixel must not collapse to an empty range and return null.
        val depth = map { _, _ -> 4f }
        val s = DepthSampler(w, h)
        assertEquals(4f, s.sample(depth, 0.5f, 0.5f, 0.5f, 0.5f)!!, 1e-4f)
    }

    @Test
    fun boxOutsideBoundsIsClamped() {
        val depth = map { _, _ -> 7f }
        val s = DepthSampler(w, h)
        assertEquals(7f, s.sample(depth, -0.5f, -0.5f, 1.5f, 1.5f)!!, 1e-4f)
    }

    @Test
    fun nanRegionReturnsNull() {
        val s = DepthSampler(w, h)
        assertNull(s.sample(map { _, _ -> Float.NaN }, 0.2f, 0.2f, 0.8f, 0.8f))
    }

    @Test
    fun scratchGrowsAcrossCallsWithoutCorrupting() {
        // Small box first, then a large one: the reused scratch array must not leave
        // stale values from the earlier, smaller sample inside the median window.
        val s = DepthSampler(w, h)
        val near = map { _, _ -> 1f }
        val far = map { _, _ -> 9f }
        assertEquals(1f, s.sample(near, 0.45f, 0.45f, 0.55f, 0.55f)!!, 1e-4f)
        assertEquals(9f, s.sample(far, 0.0f, 0.0f, 1.0f, 1.0f)!!, 1e-4f)
    }
}
