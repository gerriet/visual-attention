// Simple test of AttentionPipeline
// Compile with: g++ -std=c++17 -I../include pipeline_test.cpp ../src/pipeline/attention_pipeline.cpp -lopencv_core
// -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui

#include "attention/pipeline/attention_pipeline.h"
#include <iostream>

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
    return 1;
  }

  try
  {
    attention::pipeline::AttentionPipeline pipeline;

    std::cout << "Loading image: " << argv[1] << std::endl;
    pipeline.load_image(argv[1]);

    std::cout << "Processing..." << std::endl;
    pipeline.process();

    std::cout << "Pipeline processed successfully!" << std::endl;
    std::cout << "  Features extracted: " << pipeline.get_features().size() << std::endl;
    std::cout << "  Saliency map size: " << pipeline.get_saliency_map().size() << std::endl;
    std::cout << "  Peaks detected: " << pipeline.get_saliency_map().peaks.size() << std::endl;

    std::cout << "\nGenerating visualization..." << std::endl;
    cv::Mat visualization = pipeline.visualize();
    std::cout << "  Visualization size: " << visualization.size() << std::endl;

    // Save result
    std::string output = "../results/pipeline_test_output.png";
    cv::imwrite(output, visualization);
    std::cout << "  Saved to: " << output << std::endl;

    std::cout << "\n✓ Pipeline test successful!" << std::endl;
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
