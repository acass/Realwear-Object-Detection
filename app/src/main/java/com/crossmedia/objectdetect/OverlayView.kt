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
 * Draws segmentation masks and detection callouts over the camera preview.
 * Detections use normalized 0..1 coordinates over the square center-crop of the
 * preview; this view is laid out to exactly cover that crop region, so mapping
 * is a simple scale.
 *
 * Each detection gets a translucent mask tinted by its class, and over that a
 * white ring on the object's center, a 45-degree white leader line, and an
 * opaque white pill carrying the label in black. Masks go down first so the
 * chrome stays readable on top of them.
 *
 * The detector caps and orders what it hands over - strongest first - so
 * everything here is drawn.
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

    /**
     * ALPHA_8 masks carry coverage only, so the paint supplies the colour. Full
     * saturation and value keep the classes apart on a small display; the alpha
     * keeps the object underneath visible, which is the whole point of showing
     * the operator a mask rather than a filled box.
     */
    private val maskPaint = Paint().apply { isAntiAlias = false }
    private val classColours = IntArray(CLASS_COLOURS) { i ->
        Color.HSVToColor(MASK_ALPHA, floatArrayOf(i * 360f / CLASS_COLOURS, 1f, 1f))
    }

    private val padX = textSize * 0.6f
    private val padY = textSize * 0.35f

    /**
     * Reused across frames. Both stay within their initial capacity, so onDraw
     * allocates only the label strings.
     */
    private val visible = Rect()
    private val placed = ArrayList<RectF>(EXPECTED_DETECTIONS)
    private val maskDst = RectF()

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

        // Masks first, all of them, so no callout ends up under a later mask.
        for (i in detections.indices) {
            val mask = detections[i].mask ?: continue
            maskDst.set(
                mask.box.left * w, mask.box.top * h,
                mask.box.right * w, mask.box.bottom * h,
            )
            maskPaint.color = colourFor(detections[i].classId)
            canvas.drawBitmap(mask.bitmap, null, maskDst, maskPaint)
        }

        // Confidence order: the strongest detection gets first claim on the up-right slot.
        for (i in detections.indices) {
            val d = detections[i]
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
     * Hue by class, so the same class keeps its colour frame to frame and two
     * overlapping classes read as two objects. Unknown ids fall back to the
     * first hue rather than crashing on a model whose class count outruns the
     * table.
     */
    internal fun colourFor(classId: Int): Int =
        classColours[if (classId < 0) 0 else classId % CLASS_COLOURS]

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
        /** Only sizes a reused list - the detector owns the actual cap. */
        const val EXPECTED_DETECTIONS = 5

        /** COCO's 80 classes, spread evenly around the hue wheel. */
        const val CLASS_COLOURS = 80

        /** ~35%: enough to read as a silhouette, sheer enough to see through. */
        const val MASK_ALPHA = 89

        // Preference order: up-right, down-right, up-left, down-left.
        val QUADRANTS = arrayOf(
            1f to -1f,
            1f to 1f,
            -1f to -1f,
            -1f to 1f,
        )
    }
}
