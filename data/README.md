# Test Data

## Test Images

Place test images in `test_images/` directory.

Good test cases:
- Faces (attention should focus on faces)
- Text on background (attention should focus on text)
- Isolated objects (attention should focus on object)
- Symmetric objects (test symmetry feature)
- Cluttered scenes (test integration)

## Expected Outputs

`expected_outputs/` contains reference results from the dissertation.
Use these to verify new implementation produces similar results.

## Stereo Pairs

If testing stereo features, place matching left/right images:
- `stereo/left_001.jpg`
- `stereo/right_001.jpg`

## Standard Datasets

For comparison with modern methods, consider:
- MIT1003 (eye-tracking)
- PASCAL VOC (object detection)
- Middlebury Stereo (stereo matching)

Download these separately and place in appropriate subdirectories.
