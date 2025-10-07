#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

namespace attention {
namespace core {

/**
 * Frame represents a single image with optional metadata.
 * Supports move semantics for efficient transfer of image data.
 */
struct Frame {
    // Image data
    cv::Mat image;

    // Optional metadata
    std::string source_path;
    int frame_number = -1;
    std::chrono::system_clock::time_point timestamp;

    // Default constructor
    Frame() = default;

    // Constructor from cv::Mat
    explicit Frame(const cv::Mat& img)
        : image(img)
        , timestamp(std::chrono::system_clock::now())
    {}

    // Constructor from cv::Mat with source path
    Frame(const cv::Mat& img, const std::string& path)
        : image(img)
        , source_path(path)
        , timestamp(std::chrono::system_clock::now())
    {}

    // Move constructor
    Frame(Frame&& other) noexcept = default;

    // Move assignment
    Frame& operator=(Frame&& other) noexcept = default;

    // Copy constructor (explicit to avoid accidental copies)
    Frame(const Frame& other) = default;

    // Copy assignment
    Frame& operator=(const Frame& other) = default;

    // Query methods
    bool empty() const { return image.empty(); }
    int width() const { return image.cols; }
    int height() const { return image.rows; }
    int channels() const { return image.channels(); }
    cv::Size size() const { return image.size(); }
};

} // namespace core
} // namespace attention
