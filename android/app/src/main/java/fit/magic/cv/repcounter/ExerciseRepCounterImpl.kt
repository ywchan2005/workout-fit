// Copyright (c) 2024 Magic Tech Ltd

package fit.magic.cv.repcounter

import com.chaquo.python.Python
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import fit.magic.cv.PoseLandmarkerHelper
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.system.measureTimeMillis

class ExerciseRepCounterImpl : ExerciseRepCounter() {

    data class Point(var x: Float, var y: Float, var z: Float, val v: Boolean, val p: Boolean)

    private var poses: FloatArray = floatArrayOf()
    private val smoothPosesSize = 30
    private var poseWarmup = 40
    private val poseBeta = .05f
    private var inputs = floatArrayOf()

    private val workoutPoses = intArrayOf(0, 1, 2, 1, 0, 1, 2, 1, 0)
    private val numWorkoutPoses = workoutPoses.size - 1
    private val workoutPosesGranularity = 2

    private var progress: Float = 0f
    private val progressEndThreshold = .005
    private val progressBeta = .8f

    private var cycle = 0f
    private val cycleEndThreshold = 1f / workoutPosesGranularity

    private val xgboostBatch = 4

    private val py = Python.getInstance()
    private val pyPredict = py.getModule("predict")
    private val pyTsUtil = py.getModule("ts_util")

    private var landmarkNormalizationTime = 0L
    private var inputPreparationTime = 0L
    private var poseIdentificationTime = 0L
    private var posesSmoothingTime = 0L
    private var poseSequenceRecognitionTime = 0L
    private var timeCount = 0

    private val isCaptureMode = false
    private var frameCount = 0
    private var capturedCount = 0

    override fun setResults(resultBundle: PoseLandmarkerHelper.ResultBundle) {
        // process pose data in resultBundle
        //
        // use functions in base class incrementRepCount(), sendProgressUpdate(),
        // and sendFeedbackMessage() to update the UI

        if (isCaptureMode) {
            capture(resultBundle)
        } else {
            track(resultBundle)
        }
    }

    // Function to track the workout poses
    private fun track(resultBundle: PoseLandmarkerHelper.ResultBundle) {
        for (result in resultBundle.results) { // always has a []
            if (result.landmarks().isNotEmpty()) {
                for (landmarks in result.landmarks()) {
                    var normalisedLandmarks: List<Point>
                    landmarkNormalizationTime += measureTimeMillis {
                        normalisedLandmarks = normaliseLandmarks(landmarks)
                    }
                    var input: FloatArray
                    inputPreparationTime += measureTimeMillis {
                        val xs = normalisedLandmarks.map { p -> p.x }.toFloatArray()
                        val ys = normalisedLandmarks.map { p -> p.y }.toFloatArray()
                        val zs = normalisedLandmarks.map { p -> p.z }.toFloatArray()
                        input = xs + ys + zs
                        inputs += input
                    }
                    poseIdentificationTime += measureTimeMillis {
                        if (inputs.size / input.size >= xgboostBatch) {
                            val predictedPoses = pyPredict.callAttr("predict", inputs).toJava(FloatArray::class.java)
                            if (poses.size < poseWarmup) {
                                for (p in predictedPoses) {
                                    poses += p * poseBeta + (if (poses.isNotEmpty()) poses.last() else 0f) * (1 - poseBeta)
                                }
                            } else {
                                poses += predictedPoses
                            }
                            inputs = floatArrayOf()
                        }
                    }
                    timeCount += 1
                }
            }
        }

        if (cycle >= numWorkoutPoses - cycleEndThreshold) {
            cycle = numWorkoutPoses.toFloat()
            if (progress >= cycle / numWorkoutPoses - progressEndThreshold) {
                cycle = 0f
                progress = 0f
                incrementRepCount()
                sendProgressUpdate(progress)
                poses = floatArrayOf()
            }
            println("cycle: ${"%.4f".format(cycle)}, progress: ${"%.4f".format(progress)}, poses: ${poses.size}")
        } else  if (poses.isNotEmpty() && inputs.isEmpty()) {
            var q: FloatArray
            posesSmoothingTime += measureTimeMillis {
                val granularity = max(1, poses.size / smoothPosesSize)
                q = averagePoses(poses, granularity)
            }
            poseSequenceRecognitionTime += measureTimeMillis {
                val numPredictions = 1
                val alignments = pyTsUtil.callAttr("align", q, workoutPoses, numPredictions, workoutPosesGranularity).toJava(FloatArray::class.java)
                for ((j, alignment) in alignments.withIndex()) {
                    val poseIndex = q.size - numPredictions + j
                    cycle = max(
                        cycle,
                        alignment,
                    )
                }
            }
        }

        if (progress < cycle / numWorkoutPoses) {
            progress = min(
                progress * progressBeta + cycle / numWorkoutPoses * (1 - progressBeta),
                1f,
            )
            sendProgressUpdate(progress)
        }

        if (cycle >= numWorkoutPoses) {
            println("[time] ------------------------------")
            println("[time] landmark normalization:    ${"%.4f".format(landmarkNormalizationTime.toFloat() / timeCount)} ms")
            println("[time] input preparation:         ${"%.4f".format(inputPreparationTime.toFloat() / timeCount)} ms")
            println("[time] pose identification:       ${"%.4f".format(poseIdentificationTime.toFloat() / timeCount)} ms")
            println("[time] poses smoothing:           ${"%.4f".format(posesSmoothingTime.toFloat() / timeCount)} ms")
            println("[time] pose sequence recognition: ${"%.4f".format(poseSequenceRecognitionTime.toFloat() / timeCount)} ms")
            println("[time] ------------------------------")
            val totalTime = landmarkNormalizationTime + inputPreparationTime + poseIdentificationTime + posesSmoothingTime + poseSequenceRecognitionTime
            println("[time] landmark normalization:    ${"%.2f".format(landmarkNormalizationTime.toFloat() * 100 / totalTime)}%")
            println("[time] input preparation:         ${"%.2f".format(inputPreparationTime.toFloat() * 100 / totalTime)}%")
            println("[time] pose identification:       ${"%.2f".format(poseIdentificationTime.toFloat() * 100 / totalTime)}%")
            println("[time] poses smoothing:           ${"%.2f".format(posesSmoothingTime.toFloat() * 100 / totalTime)}%")
            println("[time] pose sequence recognition: ${"%.2f".format(poseSequenceRecognitionTime.toFloat() * 100 / totalTime)}%")
            println("[time] ------------------------------")
        }
    }

    // Function to capture landmarks
    private fun capture(resultBundle: PoseLandmarkerHelper.ResultBundle) {
        for (result in resultBundle.results) { // always has a []
            if (result.landmarks().isNotEmpty()) {
                for (landmarks in result.landmarks()) {
                    if (frameCount % 20 == 0) {
                        val normalisedLandmarks = normaliseLandmarks(landmarks)
                        for ((j, landmark) in normalisedLandmarks.withIndex()) {
                            println("[csv] 8,${capturedCount},${j},${landmark.x},${landmark.y},${landmark.z},${landmark.v},${landmark.p}")
                        }
                        capturedCount += 1
                        sendFeedbackMessage("captured: ${capturedCount}")
                    }
                    frameCount += 1
                }
            }
        }
    }

    // Function to calculate the average of every several items in the whole pose array
    private fun averagePoses(poses: FloatArray, granularity: Int): FloatArray {
        var q = floatArrayOf()
        val r = poses.reversed()
        for (i in r.indices step granularity) {
            val end = if (i + granularity > r.size) r.size else i + granularity
            val segment = r.slice(i until end)
            val average = segment.average().toFloat()
            q += average
        }
        return q.reversed().toFloatArray()
    }

    // Function to normalize a list of landmarks to a standardized scale and orientation
    private fun normaliseLandmarks(landmarks: List<NormalizedLandmark>): List<Point> {
        // Get the center point between the landmarks for the ears and hip
        val ear = getLandmarkCentre(landmarks, intArrayOf(7, 8))
        val hip = getLandmarkCentre(landmarks, intArrayOf(23, 24))

        // Calculate the scale based on the distance between the ear and hip points
        val scale = 1 / norm(Point(ear.x - hip.x, hip.y - ear.y, 0f, true, true))

        // Calculate the range of z-values
        val minZ = landmarks.minBy { landmark -> landmark.z() }.z()
        val maxZ = landmarks.maxBy { landmark -> landmark.z() }.z()
        val dz = maxZ - minZ

        // Normalize each landmark by adjusting z, scaling, and rotating it based on the hip position
        val normalisedLandmarks = landmarks.map { landmark -> rotate(Point(
            landmark.x(),
            landmark.y(),
            (landmark.z() - minZ) / dz,
            landmark.visibility().isPresent,
            landmark.presence().isPresent,
        ), hip, scale) }
        return normalisedLandmarks
    }

//    private fun f(v: Point): String {
//        return "(${"%.4f".format(v.x)},${"%.4f".format(v.y)})"
//    }

    // Function to calculate the norm (magnitude) of a 2D point/vector
    private fun norm(v: Point): Float {
        return sqrt(v.x * v.x + v.y * v.y)
    }

    // Function to get the center point of specified landmarks
    private fun getLandmarkCentre(landmarks: List<NormalizedLandmark>, indices: IntArray): Point {
        // Filter the landmarks list based on the provided indices
        val desiredLandmarks = landmarks.filterIndexed { index, _ -> indices.contains(index) }

        // Calculate the average coordinates from the filtered landmarks
        val x: Float = desiredLandmarks.map { landmark -> landmark.x() }.average().toFloat()
        val y: Float = desiredLandmarks.map { landmark -> landmark.y() }.average().toFloat()
        val z: Float = desiredLandmarks.map { landmark -> landmark.z() }.average().toFloat()
        return Point(x, y, z, true, true)
    }

    // Function to rotate a point `v` around a center point `c`, scaling the result by `scale`
    private fun rotate(v: Point, c: Point, scale: Float): Point {
        // Translate the point `v` to the origin using `c` as the center
        val p = Point(v.x - c.x, v.y - c.y, v.z, v.v, v.p)

        // Calculate the angle of rotation based on the new point's position
        val angle = atan2(p.x, -p.y)
        val cosA = cos(-angle)
        val sinA = sin(-angle)

        // Apply the rotation matrix to the point and scale the result
        return Point(
            (p.x * cosA - p.y * sinA) * scale,
            -(p.x * sinA + p.y * cosA) * scale,
            p.z,
            p.v,
            p.p,
        )
    }

}
