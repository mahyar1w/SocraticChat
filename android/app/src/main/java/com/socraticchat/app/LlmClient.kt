package com.socraticchat.app

import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL

data class ChatMessage(val role: String, val content: String)

data class RequestConfig(
    val useTeacher: Boolean,
    val teacherUrl: String,
    val teacherModel: String,
    val teacherKey: String,
    val studentUrl: String,
    val studentModel: String = "local-socratic",
)

object LlmClient {
    private const val SOCRATIC_SYSTEM =
        "You are a warm, witty conversationalist and Socratic mentor. " +
            "In casual chat, greetings, venting, humor, or everyday recommendations, reply naturally " +
            "and directly with your own personable twist. When helping the user learn, debug, or solve " +
            "a problem, guide them step by step with one focused question or hint without spoiling the answer."

    private const val CHAT_TWIST_SYSTEM =
        "You are a warm, witty conversational companion. Take the reference reply and put your own " +
            "natural, lively, human spin on it—keep all concrete ideas, recommendations, and punchlines " +
            "intact, and never turn casual chat or jokes into a quiz question."

    private const val TUTOR_TWIST_SYSTEM =
        "You are a warm Socratic mentor. Use the reference answer for factual accuracy, then put " +
            "your own Socratic twist on it: guide the user step by step with one focused question or " +
            "small hint without giving away the final answer."

    private const val TEACHER_SYSTEM =
        "Prefix your reply with [CHAT] if the user is casually chatting (greetings, smalltalk, venting, " +
            "jokes, opinions, brainstorming, or everyday recommendations), or [TUTOR] if they are asking a " +
            "math, coding, debugging, science, or conceptual 'how/why' question (or answering a tutoring step). " +
            "Then answer directly, accurately, and concisely in 1-4 sentences."

    private val THINK_REGEX = Regex("<think>.*?(?:</think>|$)", setOf(RegexOption.DOT_MATCHES_ALL))

    fun stripThink(text: String): String {
        var cleaned = THINK_REGEX.replace(text, "")
        val closeIdx = cleaned.lastIndexOf("</think>")
        if (closeIdx >= 0) {
            cleaned = cleaned.substring(closeIdx + "</think>".length)
        }
        return cleaned.trim()
    }

    fun splitTeacherMode(rawRef: String): Pair<String, String> {
        val trimmed = rawRef.trimStart()
        if (trimmed.length >= 6 && trimmed.substring(0, 6).equals("[chat]", ignoreCase = true)) {
            return "CHAT" to trimmed.substring(6).trim()
        }
        if (trimmed.length >= 7 && trimmed.substring(0, 7).equals("[tutor]", ignoreCase = true)) {
            return "TUTOR" to trimmed.substring(7).trim()
        }
        return "AUTO" to trimmed.trim()
    }

    private fun formatUserWithReference(userText: String, reference: String?, mode: String): String {
        val r = reference?.trim().orEmpty()
        if (r.isEmpty()) return userText.trim()
        val header = when (mode) {
            "CHAT" -> "[Reference — rewrite naturally in your own warm, witty voice; keep key details/punchlines, no quiz]"
            "TUTOR" -> "[Reference — use for accuracy; guide with one Socratic question or hint without spoiling the answer]"
            else -> "[Reference — use these facts/ideas and put your own natural chat or Socratic twist on them]"
        }
        return "$header\n$r\n\n[User]\n${userText.trim()}"
    }

    private fun injectReference(history: List<ChatMessage>, reference: String?, mode: String): List<ChatMessage> {
        val body = history.filter { it.role != "system" }.toMutableList()
        if (body.isNotEmpty() && body.last().role == "user") {
            val last = body.last()
            body[body.lastIndex] = ChatMessage("user", formatUserWithReference(last.content, reference, mode))
        }
        val sys = when (mode) {
            "CHAT" -> CHAT_TWIST_SYSTEM
            "TUTOR" -> TUTOR_TWIST_SYSTEM
            else -> SOCRATIC_SYSTEM
        }
        return listOf(ChatMessage("system", sys)) + body
    }

    private fun resolveEndpoint(baseUrl: String): Pair<String, Boolean> {
        var url = baseUrl.trim().trimEnd('/')
        require(url.isNotEmpty()) { "Endpoint URL must not be empty" }
        if (!url.contains("://")) url = "http://$url"
        val scheme = url.substringBefore("://").lowercase()
        require(scheme == "http" || scheme == "https") { "Unsupported URL scheme: $scheme" }
        val after = url.substringAfter("://")
        val path = if ('/' in after) after.substring(after.indexOf('/')).trimEnd('/') else ""
        return when {
            path.endsWith("/api/chat") -> url to true
            path.isEmpty() -> "$url/v1/chat/completions" to false
            !path.endsWith("/chat/completions") -> "$url/chat/completions" to false
            else -> url to false
        }
    }

    private fun extractContent(value: Any?): String {
        return when (value) {
            is String -> value
            is JSONArray -> buildString {
                for (i in 0 until value.length()) {
                    val item = value.opt(i)
                    when (item) {
                        is JSONObject -> append(item.optString("text", ""))
                        is String -> append(item)
                    }
                }
            }
            else -> ""
        }
    }

    private fun parseSse(rawBody: String): String? {
        val sb = StringBuilder()
        var sawEvent = false
        for (line in rawBody.lineSequence()) {
            val trimmed = line.trim()
            if (!trimmed.startsWith("data:")) continue
            val payload = trimmed.removePrefix("data:").trim()
            if (payload == "[DONE]") {
                sawEvent = true
                continue
            }
            val obj = try { JSONObject(payload) } catch (_: Exception) { continue }
            sawEvent = true
            when {
                obj.has("message") -> sb.append(extractContent(obj.optJSONObject("message")?.opt("content")))
                obj.has("choices") -> {
                    val c0 = obj.optJSONArray("choices")?.optJSONObject(0)
                    val holder = c0?.optJSONObject("delta") ?: c0?.optJSONObject("message")
                    sb.append(extractContent(holder?.opt("content")))
                }
            }
        }
        return if (sawEvent) sb.toString() else null
    }

    private fun chatCompletion(
        baseUrl: String,
        model: String,
        messages: List<ChatMessage>,
        apiKey: String,
        temperature: Double,
        maxTokens: Int,
    ): String {
        require(model.isNotBlank()) { "Model name must not be empty" }
        val (url, isOllama) = resolveEndpoint(baseUrl)
        val msgArray = JSONArray().apply {
            for (m in messages) {
                put(JSONObject().put("role", m.role).put("content", m.content))
            }
        }
        val payload = JSONObject().apply {
            put("model", model)
            put("messages", msgArray)
            put("stream", false)
            if (isOllama) {
                put("options", JSONObject().put("temperature", temperature).put("num_predict", maxTokens))
            } else {
                put("temperature", temperature)
                put("max_tokens", maxTokens)
            }
        }

        val conn = (URL(url).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 15_000
            readTimeout = 120_000
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("Accept", "application/json")
            setRequestProperty("User-Agent", "socratic-chat-android/0.1")
            if (apiKey.isNotBlank()) {
                setRequestProperty("Authorization", "Bearer ${apiKey.trim()}")
            }
        }

        try {
            conn.outputStream.use { it.write(payload.toString().toByteArray(Charsets.UTF_8)) }
            val code = conn.responseCode
            val stream = if (code >= 400) conn.errorStream else conn.inputStream
            val rawBody = if (stream != null) {
                BufferedReader(InputStreamReader(stream, Charsets.UTF_8)).use { it.readText() }.trim()
            } else {
                ""
            }
            if (code >= 400) {
                throw IllegalStateException("HTTP $code from $url: ${rawBody.take(220)}")
            }
            if (rawBody.isEmpty()) {
                throw IllegalStateException("Empty response from $url")
            }

            val data = try {
                JSONObject(rawBody)
            } catch (_: Exception) {
                val sse = parseSse(rawBody)
                if (sse != null) return stripThink(sse)
                throw IllegalStateException("Non-JSON response from $url: ${rawBody.take(160)}")
            }

            val err = data.opt("error")
            if (err != null && err != JSONObject.NULL) {
                throw IllegalStateException("API error from $url: $err")
            }

            val raw = when {
                data.has("message") -> extractContent(data.optJSONObject("message")?.opt("content"))
                data.has("choices") -> extractContent(
                    data.optJSONArray("choices")?.optJSONObject(0)?.optJSONObject("message")?.opt("content")
                )
                else -> ""
            }
            return stripThink(raw)
        } finally {
            conn.disconnect()
        }
    }

    fun generateReply(history: List<ChatMessage>, cfg: RequestConfig): String {
        var mode = "AUTO"
        var reference = ""

        if (cfg.useTeacher && cfg.teacherUrl.isNotBlank() && cfg.teacherModel.isNotBlank()) {
            val teacherMsgs = listOf(ChatMessage("system", TEACHER_SYSTEM)) + history
            val rawRef = try {
                chatCompletion(cfg.teacherUrl, cfg.teacherModel, teacherMsgs, cfg.teacherKey, 0.3, 1536)
            } catch (e: Exception) {
                throw IllegalStateException("Teacher step failed — ${e.message}", e)
            }
            val split = splitTeacherMode(rawRef)
            mode = split.first
            reference = split.second
        }

        val refOpt = reference.ifEmpty { null }
        val studentMsgs = injectReference(history, refOpt, if (refOpt != null) mode else "AUTO")

        var twist = try {
            chatCompletion(cfg.studentUrl, cfg.studentModel, studentMsgs, "", 0.6, 512)
        } catch (e: Exception) {
            throw IllegalStateException("Local llama.cpp step failed — ${e.message}", e)
        }

        if (twist.isEmpty()) {
            twist = try {
                chatCompletion(cfg.studentUrl, cfg.studentModel, studentMsgs, "", 0.4, 768)
            } catch (e: Exception) {
                throw IllegalStateException("Local llama.cpp retry failed — ${e.message}", e)
            }
        }

        if (twist.isEmpty()) {
            if (mode == "CHAT" && reference.isNotEmpty()) return reference
            throw IllegalStateException("Local 0.8B model returned an empty reply")
        }
        return twist
    }
}
