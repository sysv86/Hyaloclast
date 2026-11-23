# Shader pack localization

Pack metadata and settings can be localized with optional JSON files under `lang/` in a `.vsp` directory or archive. The active Minecraft locale is loaded first, then missing keys fall back to `lang/en_us.json`, and finally to the original `pack.toml` text.

Translation files use nested JSON objects. Supported keys are `pack.name`, `pack.description`, `settings.<key>.label`, `settings.<key>.category`, `settings.<key>.description`, `settings.<key>.options.<value>`, `presets.<id>.label`, and `presets.<id>.description`.

```json
{
  "pack": {
    "name": "Sunlit Vanilla"
  },
  "settings": {
    "sun_intensity": {
      "label": "Sun Intensity",
      "description": "Direct solar irradiance multiplier.",
      "options": {
        "auto": "Automatic"
      }
    }
  }
}
```

Setting option values remain stable identifiers in `pack.toml`; translations change only their displayed labels.
