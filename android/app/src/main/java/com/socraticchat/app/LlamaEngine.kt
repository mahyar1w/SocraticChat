package com.socraticchat.app

import android.content.Context
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.ServerSocket
import java.net.URL
import java.util.ArrayDeque

class LlamaEngine(
    private val process: Process,
    val baseUrl: String,
) {
    fun stop() {
        try {
            process.destroy()
            if (process.isAlive) {
                process.destroyForcibly()
            }
        } catch (_: Exception) {
        }
    }

    companion object {
        const val DEFAULT_GGUF_NAME = "Qwen3.5-0.8B-Socratic-Q8_0.gguf"

        fun findModel(context: Context, preferred: String): File? {
            val candidates = mutableListOf<File>()
            if (preferred.isNotBlank()) {
                candidates.add(File(preferred.trim()))
            }
            candidates.add(File(context.filesDir, DEFAULT_GGUF_NAME))
            context.getExternalFilesDir(null)?.let {
                candidates.add(File(it, DEFAULT_GGUF_NAME))
            }
            return candidates.firstOrNull { it.isFile && it.length() > 0L }
        }

        fun start(context: Context, modelFile: File): Result<LlamaEngine> {
            val libDir = File(context.applicationInfo.nativeLibraryDir)
            val serverBin = File(libDir, "libllama_server.so")
            if (!serverBin.isFile) {
                return Result.failure(IllegalStateException("Bundled llama-server not found in ${libDir.path}"))
            }
            if (!modelFile.isFile) {
                return Result.failure(IllegalStateException("GGUF model not found at ${modelFile.path}"))
            }

            val port = ServerSocket(0).use { it.localPort }
            val pb = ProcessBuilder(
                serverBin.absolutePath,
                "-m", modelFile.absolutePath,
                "--host", "127.0.0.1",
                "--port", port.toString(),
                "--ctx-size", "2048",
                "--no-webui",
                "--reasoning", "off",
                "--reasoning-budget", "0",
            )
            val env = pb.environment()
            val prevLd = env["LD_LIBRARY_PATH"].orEmpty()
            env["LD_LIBRARY_PATH"] = if (prevLd.isEmpty()) libDir.absolutePath else "${libDir.absolutePath}:$prevLd"
            pb.redirectErrorStream(true)

            val proc = try {
                pb.start()
            } catch (e: Exception) {
                return Result.failure(IllegalStateException("Failed to launch llama-server: ${e.message}"))
            }

            // Continuously drain output so the 64 KB pipe buffer never fills and stalls llama-server.
            val tail = ArrayDeque<String>(12)
            Thread({
                try {
                    BufferedReader(InputStreamReader(proc.inputStream)).use { reader ->
                        var line = reader.readLine()
                        while (line != null) {
                            synchronized(tail) {
                                if (tail.size >= 12) tail.removeFirst()
                                tail.addLast(line)
                            }
                            line = reader.readLine()
                        }
                    }
                } catch (_: Exception) {
                }
            }, "llama-stderr-drain").apply { isDaemon = true }.start()

            val healthUrl = "http://127.0.0.1:$port/health"
            val baseUrl = "http://127.0.0.1:$port/v1"
            val deadline = System.currentTimeMillis() + 60_000L

            while (System.currentTimeMillis() < deadline) {
                if (!proc.isAlive) {
                    Thread.sleep(50)
                    val logs = synchronized(tail) { tail.joinToString("\n") }
                    return Result.failure(
                        IllegalStateException("llama-server exited (${proc.exitValue()}):\n$logs")
                    )
                }
                if (checkHealth(healthUrl)) {
                    return Result.success(LlamaEngine(proc, baseUrl))
                }
                Thread.sleep(150)
            }

            proc.destroyForcibly()
            return Result.failure(IllegalStateException("Timed out waiting for local llama-server"))
        }

        private fun checkHealth(urlStr: String): Boolean {
            return try {
                val conn = (URL(urlStr).openConnection() as HttpURLConnection).apply {
                    connectTimeout = 1500
                    readTimeout = 1500
                    requestMethod = "GET"
                }
                try {
                    conn.responseCode == 200
                } finally {
                    conn.disconnect()
                }
            } catch (_: Exception) {
                false
            }
        }
    }
}
