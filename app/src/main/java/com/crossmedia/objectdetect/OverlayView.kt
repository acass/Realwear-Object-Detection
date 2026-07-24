package com.crossmedia.objectdetect

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.util.TypedValue
import android.view.View
import kotlin.math.sqrt

/**
 * Draws detection callouts over the camera preview. Detections use normalized
 * 0..1 coordinates over the square center-crop of the preview; this view is
 * laid out to exactly cover that crop region, so mapping is a simple scale.
 *
 * Each callout is a white ring on the object's center, a 45-degree white leader
 * line, and an opaque white pill carrying the label in black.
 *
 * At most [MAX_CALLOUTS] are drawn, strongest first.
 */
class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    // Retuning on the headset should only need TEXT_SP; everything else derives from it.
    private val textSize = sp(32f)
    private val leaderRun = dp(64f) / sqrt(2f)   // 45 degrees, so dx == dy
    private val ringRadius = dp(6f)
    private val strokeWidth = dp(3f)

    private var detections: List<Detection> = emptyList()

    private val chromePaint = Paint().apply {
        isAntiAlias = true
        style = Paint.Style.STROKE
        strokeWidth = this@OverlayView.strokeWidth
        color = Color.WHITE
    }
    private val pillPaint = Paint().apply {
        isAntiAlias = true
        style = Paint.Style.FILL
        color = Color.WHITE
    }
    private val textPaint = Paint().apply {
        isAntiAlias = true
        style = Paint.Style.FILL
        color = Color.BLACK
        textSize = this@OverlayView.textSize
        isFakeBoldText = true
    }

    private val padX = textSize * 0.6f
    private val padY = textSize * 0.35f

    /**
     * Reused across frames. Both stay within their initial capacity, so onDraw
     * allocates only the label strings.
     */
    private val visible = Rect()
    private val placed = ArrayList<RectF>(MAX_CALLOUTS)
    private val strongest = ArrayList<Detection>(MAX_CALLOUTS + 1)

    fun setDetections(list: List<Detection>) {
        detections = list
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // The overlay square is larger than the screen in landscape, so its top and
        // bottom are clipped by the parent. Fit against the visible part, not the
        // view bounds, or pills get placed into the clipped region.
        if (!getLocalVisibleRect(visible)) return

        val w = width.toFloat()
        val h = height.toFloat()
        val fm = textPaint.fontMetrics
        val pillH = (fm.descent - fm.ascent) + padY * 2f

        placed.clear()
        selectStrongest(detections)

        // Confidence order: the strongest detection gets first claim on the up-right slot.
        for (i in strongest.indices) {
            val d = strongest[i]
            val cx = (d.box.left + d.box.right) / 2f * w
            val cy = (d.box.top + d.box.bottom) / 2f * h

            val label = "${d.label} ${(d.confidence * 100).toInt()}%".uppercase()
            val pillW = textPaint.measureText(label) + padX * 2f

            canvas.drawCircle(cx, cy, ringRadius, chromePaint)

            val pill = fit(cx, cy, pillW, pillH, visible, placed) ?: continue
            placed.add(pill)

            // Leader ends on the pill corner nearest the anchor.
            val endX = if (pill.left > cx) pill.left else pill.right
            val endY = if (pill.top > cy) pill.top else pill.bottom
            canvas.drawLine(cx, cy, endX, endY, chromePaint)

            val radius = pillH / 2f
            canvas.drawRoundRect(pill, radius, radius, pillPaint)
            canvas.drawText(label, pill.left + padX, pill.top + padY - fm.ascent, textPaint)
        }
    }

    /**
     * Fills [strongest] with the top [MAX_CALLOUTS] detections, highest confidence
     * first, by insertion into a list that never exceeds its initial capacity.
     * A sort-then-take would allocate a new list on every frame.
     */
    internal fun selectStrongest(from: List<Detection>) {
        strongest.clear()
        for (i in from.indices) {
            val d = from[i]
            var at = 0
            while (at < strongest.size && strongest[at].confidence >= d.confidence) at++
            if (at >= MAX_CALLOUTS) continue
            strongest.add(at, d)
            if (strongest.size > MAX_CALLOUTS) strongest.removeAt(strongest.size - 1)
        }
    }

    internal fun strongestForTest(): List<Detection> = strongest

    /**
     * Places the pill in the first quadrant where it lands fully inside [visible] and
     * clear of the pills already [placed] this frame. Null when all four are blocked.
     * Pure — [visible] and [placed] are passed rather than read from fields so this
     * can be exercised without driving a real draw pass.
     */
    internal fun fit(
        cx: Float, cy: Float, pillW: Float, pillH: Float, visible: Rect, placed: List<RectF>
    ): RectF? {
        for ((sx, sy) in QUADRANTS) {
            val endX = cx + leaderRun * sx
            val endY = cy + leaderRun * sy
            // The leader lands on the pill corner facing the anchor, so the pill
            // extends away from it on both axes.
            val left = if (sx > 0) endX else endX - pillW
            val top = if (sy > 0) endY else endY - pillH
            val pill = RectF(left, top, left + pillW, top + pillH)

            val onScreen = pill.left >= visible.left && pill.top >= visible.top &&
                pill.right <= visible.right && pill.bottom <= visible.bottom
            if (onScreen && placed.none { RectF.intersects(it, pill) }) return pill
        }
        return null
    }

    private fun sp(v: Float) =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, v, resources.displayMetrics)

    private fun dp(v: Float) =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics)

    private companion object {
        const val MAX_CALLOUTS = 5

        // Preference order: up-right, down-right, up-left, down-left.
        val QUADRANTS = arrayOf(
            1f to -1f,
            1f to 1f,
            -1f to -1f,
            -1f to 1f,
        )
    }
}
