import java.io.File
import java.security.MessageDigest

plugins {
    id("net.fabricmc.fabric-loom") version "1.19.4"
    kotlin("jvm") version "2.4.21"
    `maven-publish`
}

val minecraftVersion = property("minecraft_version") as String
val loaderVersion = property("loader_version") as String
val fabricKotlinVersion = property("fabric_kotlin_version") as String
val nativePlatformFilter = providers.gradleProperty("vs.nativePlatform").orElse("all")

group = property("maven_group") as String
version = property("mod_version") as String
base {
    val platformSuffix = nativePlatformFilter.get().trim().lowercase()
    val baseName = property("archives_base_name") as String
    archivesName = if (platformSuffix == "all" || platformSuffix.isEmpty()) {
        baseName
    } else {
        "$baseName-$platformSuffix"
    }
}

repositories {
    maven {
        name = "Fabric"
        url = uri("https://maven.fabricmc.net/")
    }
    mavenCentral()
}

dependencies {
    minecraft("com.mojang:minecraft:$minecraftVersion")
    implementation("net.fabricmc:fabric-loader:$loaderVersion")

    implementation("net.fabricmc:fabric-language-kotlin:$fabricKotlinVersion")
    include("net.fabricmc:fabric-language-kotlin:$fabricKotlinVersion")

    val resourceLoader = "net.fabricmc.fabric-api:fabric-resource-loader-v1:3.1.0+5af7744b5d"
    runtimeOnly(resourceLoader)
    include(resourceLoader)
    include("net.fabricmc.fabric-api:fabric-api-base:2.0.6+fcdff87f5d")

    // Compile against sibling VulkanTurbo (Pyroclast) jar in either ../Pyroclast/build/libs, ../VulkanTurbo/build/libs, or ../vt/build/libs
    val turboLibs = files(
        fileTree("../Pyroclast/build/libs") { include("*.jar"); exclude("*-sources.jar") },
        fileTree("../VulkanTurbo/build/libs") { include("*.jar"); exclude("*-sources.jar") },
        fileTree("../vt/build/libs") { include("*.jar"); exclude("*-sources.jar") },
    )
    if (!turboLibs.isEmpty) {
        compileOnly(turboLibs)
        testRuntimeOnly(turboLibs)
    }

    testImplementation(kotlin("test-junit5"))
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(25)
    }
    withSourcesJar()
}

kotlin {
    jvmToolchain(25)
    compilerOptions {
        freeCompilerArgs.add("-Xjdk-release=25")
    }
}

tasks.withType<JavaCompile>().configureEach {
    options.encoding = "UTF-8"
    options.release = 25
}

tasks.test {
    useJUnitPlatform()
}

val generatedNativesDir = layout.buildDirectory.dir("generated/natives")

data class NativeTargetSpec(
    val platformDir: String,
    val libraryName: String,
    val rustTriples: List<String>,
)

val nativeTargetSpecs = listOf(
    NativeTargetSpec(
        platformDir = "win32-x86-64",
        libraryName = "vulkanshaders_core.dll",
        rustTriples = listOf("x86_64-pc-windows-gnu", "x86_64-pc-windows-msvc"),
    ),
    NativeTargetSpec(
        platformDir = "linux-x86-64",
        libraryName = "libvulkanshaders_core.so",
        rustTriples = listOf("x86_64-unknown-linux-gnu"),
    ),
    NativeTargetSpec(
        platformDir = "linux-aarch64",
        libraryName = "libvulkanshaders_core.so",
        rustTriples = listOf("aarch64-unknown-linux-gnu"),
    ),
    NativeTargetSpec(
        platformDir = "macos-aarch64",
        libraryName = "libvulkanshaders_core.dylib",
        rustTriples = listOf("aarch64-apple-darwin"),
    ),
)

val hostOsName = System.getProperty("os.name").lowercase()
val hostOsArch = System.getProperty("os.arch").lowercase()
val isHostArm64 = hostOsArch.contains("aarch64") || hostOsArch.contains("arm64")
val hostPlatformDir = when {
    hostOsName.contains("win") -> "win32-x86-64"
    hostOsName.contains("mac") -> if (isHostArm64) "macos-aarch64" else "macos-x86-64"
    else -> if (isHostArm64) "linux-aarch64" else "linux-x86-64"
}
val hostLibraryName = when {
    hostOsName.contains("win") -> "vulkanshaders_core.dll"
    hostOsName.contains("mac") -> "libvulkanshaders_core.dylib"
    else -> "libvulkanshaders_core.so"
}

val cargoBuild = tasks.register<Exec>("cargoBuild") {
    val cargoExplicitProp = providers.gradleProperty("vs.cargo")
    val projectDir = layout.projectDirectory

    group = "build"
    description = "Compiles the Rust vulkanshaders-core cdylib for the host platform"
    workingDir = projectDir.asFile
    commandLine("cargo", "build", "--release", "-p", "vulkanshaders-core")
    onlyIf {
        val explicit = cargoExplicitProp.orNull
        if (explicit != null) {
            explicit.equals("true", ignoreCase = true)
        } else {
            val pathEnv = System.getenv("PATH").orEmpty()
            pathEnv.split(File.pathSeparatorChar).any { dir: String ->
                File(dir, "cargo").canExecute() || File(dir, "cargo.exe").canExecute()
            }
        }
    }
}

val stageNativesAndChecksums by tasks.registering {
    dependsOn(cargoBuild)
    val projectDir = layout.projectDirectory
    val outNativesRoot = generatedNativesDir.map { it.dir("natives") }
    val targetSpecs = nativeTargetSpecs
    val currentHostPlatform = hostPlatformDir
    val currentHostLib = hostLibraryName
    val platformFilterProp = nativePlatformFilter

    group = "build"
    description = "Stages native libraries under natives/<platform>/ and generates SHA-256 integrity manifests"
    inputs.property("nativePlatformFilter", platformFilterProp)
    // cargoBuild has no declared source inputs/outputs, so it always runs. Track its cdylib here:
    // without this, a rebuilt .so leaves this task's inputs/filter and output dir unchanged, so it
    // stays UP-TO-DATE and a stale library propagates into processResources / build/resources/main.
    inputs.file(projectDir.file("target/release/$hostLibraryName"))
    outputs.dir(outNativesRoot)

    doLast {
        fun sha256Hex(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buf = ByteArray(8192)
                while (true) {
                    val read = input.read(buf)
                    if (read <= 0) break
                    digest.update(buf, 0, read)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }

        val outDir = outNativesRoot.get().asFile
        if (outDir.exists()) {
            outDir.deleteRecursively()
        }
        outDir.mkdirs()

        val filter = platformFilterProp.get().trim().lowercase()
        val activeSpecs = if (filter == "all" || filter.isEmpty()) {
            targetSpecs
        } else {
            targetSpecs.filter { it.platformDir.equals(filter, ignoreCase = true) }
        }

        val prebuiltRoot = projectDir.dir("prebuilt-natives").asFile
        val targetRelease = projectDir.dir("target/release").asFile

        // Stage prebuilt and cross-compiled target libraries
        for (spec in activeSpecs) {
            val platformOut = outDir.resolve(spec.platformDir)
            val prebuiltFile = prebuiltRoot.resolve(spec.platformDir).resolve(spec.libraryName)
            if (prebuiltFile.isFile) {
                platformOut.mkdirs()
                prebuiltFile.copyTo(platformOut.resolve(spec.libraryName), overwrite = true)
            }
            for (triple in spec.rustTriples) {
                val crossBuilt = projectDir.dir("target/$triple/release").asFile.resolve(spec.libraryName)
                if (crossBuilt.isFile) {
                    platformOut.mkdirs()
                    crossBuilt.copyTo(platformOut.resolve(spec.libraryName), overwrite = true)
                }
            }
        }

        // Stage locally built host cdylib when host platform matches the active filter
        val includeHost = filter == "all" || filter.isEmpty() || filter == currentHostPlatform
        val hostBuilt = targetRelease.resolve(currentHostLib)
        if (includeHost && hostBuilt.isFile) {
            val hostPlatformOut = outDir.resolve(currentHostPlatform)
            hostPlatformOut.mkdirs()
            hostBuilt.copyTo(hostPlatformOut.resolve(currentHostLib), overwrite = true)
            hostBuilt.copyTo(outDir.resolve(currentHostLib), overwrite = true)
        }

        // Generate per-native .sha256 sidecars and a consolidated natives.sha256 manifest
        // so runtime extraction into <config>/vulkanshaders/natives can verify integrity.
        val manifestLines = mutableListOf<String>()
        val stagedBinaries = outDir.walkTopDown()
            .filter { it.isFile && !it.name.endsWith(".sha256") }
            .sortedBy { it.relativeTo(outDir).invariantSeparatorsPath }
            .toList()

        for (binary in stagedBinaries) {
            val hash = sha256Hex(binary)
            val relPath = binary.relativeTo(outDir).invariantSeparatorsPath
            binary.parentFile.resolve("${binary.name}.sha256")
                .writeText("$hash  ${binary.name}\n")
            manifestLines.add("$hash  $relPath")
        }

        outDir.resolve("natives.sha256").writeText(
            if (manifestLines.isEmpty()) "" else manifestLines.joinToString("\n", postfix = "\n"),
        )
    }
}

sourceSets.main {
    resources.srcDir(generatedNativesDir)
}

val distributionChannel = providers.gradleProperty("vs.distribution").orElse("modrinth")
val generatedDistributionDir = layout.buildDirectory.dir("generated/distribution")

val generateDistributionMetadata by tasks.registering {
    val channelProp = distributionChannel
    val outDir = generatedDistributionDir
    inputs.property("distributionChannel", channelProp)
    outputs.dir(outDir)
    doLast {
        val dir = outDir.get().asFile
        dir.mkdirs()
        dir.resolve("vulkanshaders-distribution.properties")
            .writeText("channel=${channelProp.get().trim().lowercase()}\n")
    }
}

tasks.processResources {
    dependsOn(stageNativesAndChecksums, generateDistributionMetadata)
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE

    val modVersion = project.version
    inputs.property("version", modVersion)
    filesMatching("fabric.mod.json") {
        expand("version" to modVersion)
    }
    from(generatedDistributionDir)
}

tasks.named<Jar>("sourcesJar") {
    dependsOn(stageNativesAndChecksums)
}

tasks.jar {
    val projectName = project.name
    inputs.property("projectName", projectName)
    from(listOf("LICENSE", "LICENSE-EXCEPTION.md")) {
        rename { "${it}_$projectName" }
    }
}
