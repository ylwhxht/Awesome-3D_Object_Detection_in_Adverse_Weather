# Real camera footage

The homepage camera inset contains recorded front-camera data. Its weather
category follows the automatic Clear / Fog / Rain / Snow cycle. The animated
point cloud below it is a separate illustration, not a synchronized sensor
recording or the output of a detector applied to these clips.

## Clear, rain and snow — Boreas

Credit: Keenan Burnett, David J. Yoon, Yuchen Wu, Andrew Z. Li, Haowei Zhang,
Shichen Lu, Jingxing Qian, Wei-Kang Tseng, Andrew Lambert, Keith Y.K. Leung,
Angela P. Schoellig and Timothy D. Barfoot. University of Toronto Institute
for Aerospace Studies. *Boreas: A Multi-Season Autonomous Driving Dataset*,
The International Journal of Robotics Research, 2023.

- Dataset and license: https://www.boreas.utias.utoronto.ca/#/boreas
- Sequence catalogue and weather tags: https://www.boreas.utias.utoronto.ca/#/download
- Data format: https://github.com/utiasASRL/pyboreas/blob/master/DATA_REFERENCE.md
- License: **Creative Commons Attribution 4.0 (CC BY 4.0)**,
  https://creativecommons.org/licenses/by/4.0/

| Local file | Original sequence | Publisher's weather tags | Original camera video |
| --- | --- | --- | --- |
| `camera-clear.mp4` | `boreas-2021-03-09-14-23` | Sun | https://boreas.s3.amazonaws.com/boreas-2021-03-09-14-23/video.mp4 |
| `camera-rain.mp4` | `boreas-2021-06-29-18-53` | Overcast, Rain | https://boreas.s3.amazonaws.com/boreas-2021-06-29-18-53/video.mp4 |
| `camera-snow.mp4` | `boreas-2021-01-26-11-22` | Overcast, Snow, Snowing | https://boreas.s3.amazonaws.com/boreas-2021-01-26-11-22/video.mp4 |

Adaptations: each clip is the 60–68 second excerpt of its published camera
preview. The 2448 × 2048 image is cropped to the central 2448 × 1376 area
(x=0, y=336), resized to 640 × 360, and encoded as silent H.264 at 24 fps.
The matching JPEG is the first frame. Playback speed is unchanged from the
published preview; this does not assert the timing of the raw camera capture.
The adapted clips and posters retain CC BY 4.0. No weather effects are added.

## Fog — RADIATE

Credit: Marcel Sheeny, Emanuele De Pellegrin, Saptarshi Mukherjee,
Alireza Ahrabian, Sen Wang and Andrew Wallace. Heriot-Watt University.
*RADIATE: A Radar Dataset for Automotive Perception in Bad Weather*, ICRA 2021.

- Dataset: https://pro.hw.ac.uk/radiate/
- Source frames: https://github.com/marcelsheeny/radiate_sdk/tree/master/data/radiate/tiny_foggy/zed_left
- Source metadata: https://github.com/marcelsheeny/radiate_sdk/blob/master/data/radiate/tiny_foggy/meta.json
- License statement: https://pro.hw.ac.uk/radiate/downloads/
- License: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  (CC BY-NC-SA 4.0)**, https://creativecommons.org/licenses/by-nc-sa/4.0/

`camera-fog.mp4` uses all 50 left-camera frames, `000001.png`–`000050.png`,
from the public `tiny_foggy` sample. The source metadata identifies the
sequence as `fog_6_0` and its weather type as `fog`.

Adaptations: resized from 672 × 376 to 640 × 360, encoded as silent H.264,
and resampled to 30 fps using the source `zed_left.txt` frame timestamps.
Frames are repeated as needed; no intermediate image content is generated.
The short sample repeats during its eight-second weather slot. The matching
JPEG is the first frame. These adapted fog assets retain **CC BY-NC-SA 4.0**.
No weather effects, dehazing or scene alterations are added.

Retrieved and prepared on 12 October 2026. The original providers do not
endorse this research index. These media licenses apply to their respective
clips and posters; they do not change the license of the website code.
