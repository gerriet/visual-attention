// Main entry point for the Attention Framework
// Phase 1, Session 1: Basic image loading and display

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

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
