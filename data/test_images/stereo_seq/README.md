# stereo_seq — a four-frame stereo sequence for the stream tests

Smoothed noise background at disparity 2, a red disk at disparity 7 moving 10 px
per frame to the right. Generated, not captured: it exists so that
`--attend --right` (a stereo stream, which the depth feature needs) and
`--emit-trace` have something to run on in CTest. Deliberately small.
