# Alternative saliency features (non-thesis, pluggable)

The dissertation model is the default everywhere. On top of it, the framework
can plug in saliency features that were **not** part of the thesis — either
whole new bottom-up operators from the later literature, or (in future)
alternative implementations of the thesis features. This is the first step of a
broader "configure/plug what the thesis never had" direction (roadmap M9).

Because features are a registry keyed by a `type` string and are instantiated
only when a config names them, an alternative is purely additive: it gets a new
key, and nothing about the default / `thesis` / `modern` profiles changes.

## The operators

| Config key | Model | Idea | Notes |
|---|---|---|---|
| `spectral-residual` | Hou & Zhang, CVPR 2007 | The log-amplitude spectrum of natural images is statistically smooth; saliency is what departs from that. Take the "spectral residual" (log spectrum minus its local average) and inverse-transform it with the original phase. | Computed on a small (~64 px) copy; fast, parameter-light. Frequency-domain. Mirrored by the Python eval model `spectral-residual`. |
| `frequency-tuned` | Achanta et al., CVPR 2009 | Distance, in CIELab, of each (lightly blurred) pixel from the image's mean colour. Blur kills fine texture; mean-subtraction kills the smooth background — a band-pass. | Full resolution, crisp boundaries. An alternative colour/contrast channel. |
| `boolean-map` | Zhang & Sclaroff, ICCV 2013 | Threshold each Lab channel at many levels into binary "boolean maps"; a region is *figure* only if it is surrounded (its connected component does not touch the image border). Average the figure maps. | Topological figure-ground, not centre-surround. Strong on compact objects. |
| `phase-spectrum` | Guo, Ma & Zhang, CVPR 2008 | Reconstruct from the Fourier *phase* alone (unit amplitude): locally distinct, aperiodic structure survives; homogeneous regions cancel. Runs on intensity + colour opponency (RG, BY), and on temporal streams adds a motion channel (change against the previous frame). | Per-channel PFT formulation of the PQFT model (the paper reports PFT on par with the quaternion transform). The only alternative operator that is stream-aware. |
| `image-signature` | Hou, Harel & Koch, TPAMI 2012 | The "image signature" is `sign(DCT(image))`: reconstructing from the coefficient signs alone concentrates energy on a spatially sparse foreground against a spectrally sparse background. Smoothed, squared, per Lab channel. | DCT sibling of `spectral-residual` with a sparsity story instead of a spectrum-statistics one. Very fast. |
| `minimum-barrier` | Zhang et al., ICCV 2015 | For each pixel, the minimum *barrier* distance to the image border — the smallest (max − min) intensity span over any path out. Background flows to the border through near-constant paths; a distinct object is walled in by its own contrast. FastMBD raster-scan approximation, per Lab channel. | Boundary prior from the salient-*object*-detection line of work; complements the contrast/frequency operators. Core MBD only (no MB+ backgroundness cue / post-processing). |

All six are self-contained (OpenCV core + imgproc only — no Gabor bank,
pyramid, or contrib module) and degrade sensibly on grayscale input
(`frequency-tuned` → intensity deviation from the mean; `boolean-map`,
`image-signature`, `minimum-barrier` → a single intensity channel;
`phase-spectrum` → intensity ± motion; `spectral-residual` is grayscale by
construction).

**Caveat on `spectral-residual`:** it keys on departures from a *textured*
background spectrum, so it is at its best on natural images. A single blob on a
perfectly flat field is degenerate for it (the whole spectrum is near-DC).

## Running them

Standalone (the thesis features disabled) — `configs/alternative.yaml`:

```bash
./build/attention --config configs/alternative.yaml data/samples/images/soccer.jpg --no-display
# (argument order is free: "attention <image> --config <yaml>" also works)
```

Fused with the thesis features — drop the `enabled: false` block from
`configs/alternative.yaml`, or add the alternative keys to any profile:

```yaml
features:
  color: {weight: 1.0}          # thesis
  eccentricity: {weight: 1.0}   # thesis
  symmetry: {weight: 1.0}       # thesis
  spectral-residual: {weight: 0.5}   # alternative, down-weighted
```

## Parameters

Each accepts an optional `params:` block (defaults reproduce the paper):

```yaml
spectral-residual:
  weight: 1.0
  params:
    input_size: 64        # longer side of the working image
    avg_filter_size: 3    # box filter over the log spectrum (h_n)
    gaussian_sigma: 3.0   # smoothing of the reconstructed map

frequency-tuned:
  params:
    blur_size: 5          # Gaussian kernel suppressing fine texture (odd)

boolean-map:
  params:
    working_size: 256     # longer side of the working image
    threshold_step: 20    # threshold spacing over [0, 255] (smaller = more maps)
    blur_sigma: 3.0       # smoothing of the accumulated map

phase-spectrum:
  params:
    input_size: 64        # longer side of the working image
    gaussian_sigma: 3.0   # smoothing of the summed response
    use_motion: true      # motion channel on temporal streams (needs previous frame)

image-signature:
  params:
    input_size: 64        # longer side of the working image (rounded to even for DCT)
    gaussian_sigma: 3.0   # smoothing of the squared reconstruction

minimum-barrier:
  params:
    working_size: 256     # longer side of the working image
    num_passes: 3         # alternating FastMBD raster sweeps
    blur_sigma: 3.0       # smoothing of the summed map
```

## Where the code lives

- `include/attention/features/{spectral_residual,frequency_tuned,boolean_map,phase_spectrum,image_signature,minimum_barrier}_feature.h`
  and `src/features/*.cpp`
- Registered in `src/features/feature_registry.cpp`
- Tests: `tests/test_alt_features.cpp`, plus the `alt_features_profile` CLI test
