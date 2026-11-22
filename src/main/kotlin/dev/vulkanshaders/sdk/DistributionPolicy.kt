package dev.vulkanshaders.sdk

import java.net.URI
import java.util.Properties

/**
 * Distribution channel compliance policy.
 *
 * - [MODRINTH] and [GITHUB]: User-initiated opt-in downloads from allowlisted official HTTPS vendor
 *   URLs are permitted (never silent, never bundled in the mod JAR; declared under Modrinth's
 *   External System Interactions disclosure).
 * - [CURSEFORGE]: Automated external binary downloads are strictly disabled to comply with
 *   CurseForge Moderation Policies prohibiting third-party external downloads; all downloadable
 *   entries gracefully downgrade to [AcquisitionPolicy.MANUAL_INSTALL].
 */
enum class DistributionChannel(val id: String, val externalDownloadsAllowed: Boolean) {
    MODRINTH("modrinth", true),
    CURSEFORGE("curseforge", false),
    GITHUB("github", true),
    ;

    companion object {
        private const val RESOURCE_PATH = "/vulkanshaders-distribution.properties"
        private const val SYS_PROP = "vulkanshaders.distribution"

        val ALLOWED_DOWNLOAD_HOSTS: Set<String> = setOf(
            "github.com",
            "raw.githubusercontent.com",
            "objects.githubusercontent.com",
            "release-assets.githubusercontent.com",
            "codeload.github.com",
        )

        @JvmStatic
        fun parse(raw: String?): DistributionChannel {
            val normalized = raw?.trim()?.lowercase() ?: return MODRINTH
            return entries.firstOrNull { it.id == normalized } ?: MODRINTH
        }

        @JvmStatic
        fun detect(): DistributionChannel {
            System.getProperty(SYS_PROP)?.takeIf { it.isNotBlank() }?.let {
                return parse(it)
            }
            val stream = DistributionChannel::class.java.getResourceAsStream(RESOURCE_PATH)
                ?: return MODRINTH
            return runCatching {
                stream.use { input ->
                    val props = Properties()
                    props.load(input)
                    parse(props.getProperty("channel"))
                }
            }.getOrDefault(MODRINTH)
        }

        @JvmStatic
        fun isAllowedOfficialHttpsUrl(rawUrl: String): Boolean {
            val uri = runCatching { URI.create(rawUrl) }.getOrNull() ?: return false
            if (!uri.scheme.equals("https", ignoreCase = true)) return false
            if (uri.userInfo != null) return false
            if (uri.port != -1 && uri.port != 443) return false
            val host = uri.host?.lowercase() ?: return false
            return host in ALLOWED_DOWNLOAD_HOSTS
        }
    }
}
