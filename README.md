# cv_structlight
Structured Light

This code will reconstruct a scene from multiple structured light scannings of it.

1. Calibrate projector with the “easy” method. Use ray-plane intersection. Get 2D-3D correspondence and use stereo calibration

2. Correlate code with (x,y) position - there is a "codebook" from binary code -> (x,y)

3. With 2D-2D correspondence, perform stereo triangulation (existing function) to get a depth map

