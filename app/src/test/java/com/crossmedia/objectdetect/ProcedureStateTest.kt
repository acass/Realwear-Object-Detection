package com.crossmedia.objectdetect

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * No Robolectric runner here, unlike the rest of the suite: [ProcedureState] has
 * no Android dependencies, so it runs on the plain JVM.
 */
class ProcedureStateTest {

    private fun state() = ProcedureState(listOf("cup", "keyboard", "laptop", "bottle", "cell phone"))

    @Test
    fun `starts idle with no target`() {
        val p = state()
        assertEquals(Mode.IDLE, p.mode)
        assertNull(p.currentTarget)
    }

    @Test
    fun `start enters the first step`() {
        val p = state()
        p.start()
        assertEquals(Mode.RUNNING, p.mode)
        assertEquals(0, p.stepIndex)
        assertEquals("cup", p.currentTarget)
    }

    @Test
    fun `verdicts are recorded against the step that was open`() {
        val p = state()
        p.start()
        p.answer(true)   // cup
        p.answer(false)  // keyboard
        assertEquals(true, p.results[0].foundByOperator)
        assertEquals(false, p.results[1].foundByOperator)
        assertNull(p.results[2].foundByOperator)
        assertEquals("laptop", p.currentTarget)
    }

    @Test
    fun `answering the last step ends at the summary, not a sixth step`() {
        val p = state()
        p.start()
        repeat(5) { p.answer(true) }
        assertEquals(Mode.SUMMARY, p.mode)
        assertEquals(4, p.stepIndex)
        assertNull(p.currentTarget)
        assertEquals(5, p.foundCount)
    }

    @Test
    fun `answering is ignored once the run is over`() {
        val p = state()
        p.start()
        repeat(5) { p.answer(true) }
        p.answer(false)
        assertEquals(Mode.SUMMARY, p.mode)
        assertEquals(5, p.foundCount)
    }

    @Test
    fun `go back on the first step does nothing`() {
        val p = state()
        p.start()
        p.observe(0.8f)
        p.goBack()
        assertEquals(0, p.stepIndex)
        assertEquals(0.8f, p.results[0].peakConfidence, 1e-6f)
    }

    @Test
    fun `go back clears the verdict and the measurement of the step it returns to`() {
        val p = state()
        p.start()
        p.observe(0.9f)
        p.answer(true)
        p.goBack()

        assertEquals(0, p.stepIndex)
        assertNull(p.results[0].foundByOperator)
        assertEquals(0f, p.results[0].peakConfidence, 1e-6f)
    }

    @Test
    fun `observe keeps the maximum, not the most recent`() {
        val p = state()
        p.start()
        p.observe(0.9f)
        p.observe(0.6f)
        p.observe(0f)
        assertEquals(0.9f, p.results[0].peakConfidence, 1e-6f)
    }

    @Test
    fun `observe only touches the open step`() {
        val p = state()
        p.start()
        p.observe(0.7f)
        p.answer(true)
        p.observe(0.8f)

        assertEquals(0.7f, p.results[0].peakConfidence, 1e-6f)
        assertEquals(0.8f, p.results[1].peakConfidence, 1e-6f)
    }

    @Test
    fun `observe is ignored outside a run`() {
        val p = state()
        p.observe(0.9f)
        assertEquals(0f, p.results[0].peakConfidence, 1e-6f)

        p.start()
        repeat(5) { p.answer(false) }
        p.observe(0.9f)
        assertEquals(0f, p.results[4].peakConfidence, 1e-6f)
    }

    @Test
    fun `restarting discards the previous run entirely`() {
        val p = state()
        p.start()
        p.observe(0.9f)
        p.answer(true)
        p.answer(true)

        p.start()

        assertEquals(0, p.stepIndex)
        assertEquals(0, p.foundCount)
        assertTrue(p.results.all { it.foundByOperator == null && it.peakConfidence == 0f })
    }

    @Test
    fun `exit returns to free-running detection`() {
        val p = state()
        p.start()
        repeat(5) { p.answer(true) }
        p.exit()

        assertEquals(Mode.IDLE, p.mode)
        assertEquals(0, p.stepIndex)
        assertNull(p.currentTarget)
        // Verdicts survive the exit; only start() clears them.
        assertEquals(5, p.foundCount)
    }

    @Test
    fun `default targets are the five demo classes in order`() {
        assertEquals(
            listOf("cup", "keyboard", "laptop", "bottle", "cell phone"),
            ProcedureState.DEFAULT_TARGETS
        )
    }
}
