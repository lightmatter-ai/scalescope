# Hot Interconnects 2025 hardware assumptions

`hardware_library.yaml` freezes the hardware assumptions used by the published
artifact. It intentionally does not inherit from
`hardware_specs/hardware_library.yaml`, because that shared library continues
to evolve.

Use this library with configurations under `hot_interconnects_2025`:

```text
--hardware-lib hardware_specs/hot_interconnects_2025/hardware_library.yaml
```
