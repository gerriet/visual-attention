# Sample data

Example imagery and a video for exercising the attention system, downloaded
from the [OpenCV `samples/data`](https://github.com/opencv/opencv/tree/4.x/samples/data)
collection (OpenCV is licensed under **Apache License 2.0**; this data is
redistributed under those terms). These are widely-used computer-vision test
assets — suitable stand-ins for attention/saliency experiments and for the
streaming modes without a webcam.

For formal saliency benchmarking against human fixations, use the dataset
adapters instead (`eval/datasets/mit1003.py`, `eval/datasets/middlebury.py`),
which document how to download MIT1003 / Middlebury; those corpora are not
redistributed here.

## Contents

| File | What it is | Good for |
|---|---|---|
| `video/vtest.avi` | Surveillance clip of people walking (the standard OpenCV test video) | `--live`, `--attend`, `--sequence` (motion/onset, object tracking) |
| `stereo/aloe_left.jpg`, `stereo/aloe_right.jpg` | The **Aloe** rectified stereo pair (Middlebury Stereo Datasets, Scharstein & Szeliski / Hirschmüller & Scharstein 2007, via OpenCV) | `--stereo` (disparity/depth feature) |
| `images/fruits.jpg` | Colourful still life | colour feature, multiple salient regions |
| `images/butterfly.jpg` | Butterfly on foliage | figure–ground, single salient object |
| `images/baboon.jpg` | Mandrill (classic vision test image) | colour/texture, face |
| `images/soccer.jpg` | A footballer on a pitch | a clear salient person / focus of attention |

## Try it

```bash
# Real stereo pair through the depth feature (near surfaces pop out)
./build/attention --stereo data/samples/stereo/aloe_left.jpg data/samples/stereo/aloe_right.jpg \
    --config configs/stereo.yaml --no-display

# The video instead of a webcam — live overlay with object-file plugins
./build/attention --live data/samples/video/vtest.avi --config configs/live.yaml \
    --processors region-descriptor            # add --no-display --frames 20 --output results/live to run headless

# Object files + scanpath over the video (cap frames for a quick run)
./build/attention --attend data/samples/video/vtest.avi --output results/attend --emit-scanpath results/scan.json

# A single natural image
./build/attention data/samples/images/soccer.jpg --no-display
```

Note: `--attend`/`--sequence` on the full video process every frame; pass a
short clip or interrupt when you have enough. `--live` supports `--frames N`.
