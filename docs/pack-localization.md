# Shader Pack Localization

Hyaloclast supports multi-language localization for `.vsp` shader packs (both unpacked directories and `.vsp` / `.zip` archives) using standard Minecraft locale codes.

## File Layout

Place UTF-8 JSON translation files under `lang/` inside your shader pack root:

```text
my-pack.vsp/
  pack.toml
  lang/
    en_us.json
    ru_ru.json
    fr.json
    fr_ca.json
    ja_jp.json
```

## Locale Normalization & Fallback Chain

Minecraft's active language code is normalized to lowercase with underscores (`fr-CA` -> `fr_ca`). Keys are merged in order of increasing specificity:

1. `lang/en_us.json` (default base fallback)
2. `lang/<lang>.json` (base language fallback when `<lang>_<region>` is selected, e.g. `fr.json` for `fr_ca`)
3. `lang/<lang>_<region>.json` (exact regional locale, e.g. `fr_ca.json`)
4. Raw strings from `pack.toml` (final fallback when a key is omitted or blank)

If a pack uses `extends = "parent-pack-id"` in `pack.toml`, parent translations are loaded first and child translations override them cleanly.

## Supported Translation Keys

Both flat dot-separated keys and nested JSON objects are supported (nested objects are flattened with `.`). Individual translation files are capped at 1 MiB.

| Key Pattern | Target in UI |
| :--- | :--- |
| `pack.name` | Shader pack display name in `ShaderPacksScreen` and `PackSettingsScreen` |
| `pack.description` | Shader pack tooltip description |
| `categories.<category_id>` | Shared category header/filter label (lowercase snake_case of `category` in `pack.toml`, e.g. `categories.lighting`, `categories.atmosphere`) |
| `settings.<key>.label` | Setting display label |
| `settings.<key>.category` | Per-setting category override (takes precedence over `categories.<id>`) |
| `settings.<key>.description` | Setting hover tooltip description |
| `settings.<key>.options.<value>` | Display label for `type = "enum"` option values and `Default:` tooltip hint |
| `presets.<id>.label` | Quality preset display label (overrides built-in `low`/`medium`/`high`/`ultra` labels when provided) |
| `presets.<id>.description` | Quality preset tooltip description |

## Example (`lang/ru_ru.json`)

```json
{
  "pack": {
    "name": "My Shader Pack",
    "description": "Custom Vulkan shader pack with localized settings."
  },
  "categories": {
    "lighting": "Освещение",
    "atmosphere": "Атмосфера",
    "water": "Вода"
  },
  "settings": {
    "sun_intensity": {
      "label": "Интенсивность солнца",
      "description": "Множитель прямого солнечного света."
    },
    "tone_mapping": {
      "label": "Тональная компрессия",
      "options": {
        "agx": "AgX",
        "aces": "ACES",
        "pbr_neutral": "PBR Neutral"
      }
    }
  },
  "presets": {
    "high": {
      "label": "Высокий",
      "description": "Рекомендуемый баланс качества и производительности."
    }
  }
}
```

## Live Language Switching

Translations are cached per `(packRoot, locale, lastModifiedTime)` and automatically refresh when the player switches language in Minecraft's Language screen or reloads shader packs (`R`).
