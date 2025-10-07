// Main entry point for the Attention Framework
// Phase 1, Session 1-2: Basic image loading and core data structures

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// Phase 1, Session 2: Core data structures
#include "attention/core/frame.h"
#include "attention/core/feature_map.h"
#include "attention/core/saliency_map.h"

// Visualization helpers
#include "attention/visualization/visualizer.h"

int main(int argc, char** argv) {
    // Check command line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path> [--no-display]" << std::endl;
        std::cerr << "Example: " << argv[0] << " ../data/test.jpg" << std::endl;
        std::cerr << "  --no-display: Load and verify image without displaying (for testing)" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    bool display = true;

    // Check for --no-display flag
    if (argc >= 3 && std::string(argv[2]) == "--no-display") {
        display = false;
    }

    // Load the image
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    // Check if image was loaded successfully
    if (image.empty()) {
        std::cerr << "Error: Could not load image from: " << image_path << std::endl;
        std::cerr << "Please check that:" << std::endl;
        std::cerr << "  - The file exists" << std::endl;
        std::cerr << "  - The path is correct" << std::endl;
        std::cerr << "  - OpenCV supports this image format" << std::endl;
        return 1;
    }

    // Display image information
    std::cout << "Successfully loaded image: " << image_path << std::endl;
    std::cout << "  Size: " << image.cols << "x" << image.rows << std::endl;
    std::cout << "  Channels: " << image.channels() << std::endl;
    std::cout << "  Type: " << image.type() << std::endl;

    // Test core data structures (Session 2)
    attention::core::Frame frame(image, image_path);
    std::cout << "\nFrame created:" << std::endl;
    std::cout << "  Frame size: " << frame.width() << "x" << frame.height() << std::endl;
    std::cout << "  Source: " << frame.source_path << std::endl;

    // Test FeatureMap with dummy data (create a simple gradient for visualization)
    cv::Mat dummy_feature(image.rows, image.cols, CV_32F);
    for (int y = 0; y < dummy_feature.rows; ++y) {
        for (int x = 0; x < dummy_feature.cols; ++x) {
            // Create a radial gradient pattern
            float dx = x - dummy_feature.cols / 2.0f;
            float dy = y - dummy_feature.rows / 2.0f;
            float dist = std::sqrt(dx * dx + dy * dy);
            float max_dist = std::sqrt(dummy_feature.cols * dummy_feature.cols / 4.0f +
                                      dummy_feature.rows * dummy_feature.rows / 4.0f);
            dummy_feature.at<float>(y, x) = 1.0f - (dist / max_dist);
        }
    }

    attention::core::FeatureMap test_feature("test_gradient", dummy_feature, 0.8f);
    std::cout << "\nFeatureMap created:" << std::endl;
    std::cout << "  Name: " << test_feature.name << std::endl;
    std::cout << "  Valid: " << (test_feature.is_valid() ? "yes" : "no") << std::endl;
    std::cout << "  Confidence: " << test_feature.confidence << std::endl;

    // Test SaliencyMap with peak detection
    attention::core::SaliencyMap saliency(dummy_feature);

    // Add a few test peaks
    saliency.peaks.push_back(attention::core::Peak(cv::Point(image.cols/2, image.rows/2), 1.0f));
    saliency.peaks.push_back(attention::core::Peak(cv::Point(image.cols/4, image.rows/4), 0.7f));
    saliency.peaks.push_back(attention::core::Peak(cv::Point(3*image.cols/4, image.rows/4), 0.6f));

    std::cout << "\nSaliencyMap created:" << std::endl;
    std::cout << "  Valid: " << (saliency.is_valid() ? "yes" : "no") << std::endl;
    std::cout << "  Size: " << saliency.size() << std::endl;
    std::cout << "  Peaks: " << saliency.peaks.size() << std::endl;

    std::cout << "\n✓ All core data structures working!" << std::endl;

    // Test visualization (always generate, display or save based on mode)
    std::cout << "\nGenerating visualizations..." << std::endl;

    // Visualize feature map
    cv::Mat feature_vis = attention::visualization::visualize_feature_map(
        test_feature, display ? "Feature Map (Gradient)" : "", false);

    // Visualize saliency map without overlay
    cv::Mat saliency_vis = attention::visualization::visualize_saliency_map(
        saliency, cv::Mat(), display ? "Saliency Map (Heatmap)" : "", true, false);

    // Visualize saliency map with overlay on original
    cv::Mat overlay_vis = attention::visualization::visualize_saliency_map(
        saliency, image, display ? "Saliency Overlay" : "", true, false);

    // Side-by-side comparison
    cv::Mat image_display;
    if (image.channels() == 3) {
        image_display = image;
    } else {
        cv::cvtColor(image, image_display, cv::COLOR_GRAY2BGR);
    }

    cv::Mat feature_bgr;
    cv::cvtColor(feature_vis, feature_bgr, cv::COLOR_GRAY2BGR);

    std::vector<cv::Mat> comparison = {image_display, feature_bgr, overlay_vis};
    std::vector<std::string> comp_labels = {"Original", "Feature", "Saliency+Peaks"};

    cv::Mat comparison_vis = attention::visualization::visualize_side_by_side(
        comparison, comp_labels, display ? "Attention Framework - Visualization Test" : "", false);

    if (!display) {
        // Save visualizations to files for inspection
        std::string output_dir = "../results/";
        attention::visualization::save_visualization(feature_vis, output_dir + "feature_map.png");
        attention::visualization::save_visualization(overlay_vis, output_dir + "saliency_overlay.png");
        attention::visualization::save_visualization(comparison_vis, output_dir + "comparison.png");
    }

    std::cout << "✓ Visualization working!" << std::endl;
    if (display) {
        std::cout << "  Press any key in any window to continue..." << std::endl;
    }

    if (display) {
        // Display the image
        const std::string window_name = "Attention Framework - Image Display";
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
        cv::imshow(window_name, image);

        std::cout << "\nPress any key in the image window to exit..." << std::endl;
        cv::waitKey(0);

        // Cleanup
        cv::destroyAllWindows();
    } else {
        std::cout << "\nImage loaded and verified successfully (--no-display mode)" << std::endl;
    }

    return 0;
}
