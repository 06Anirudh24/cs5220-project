#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <stdexcept>

std::vector<float> load_float_bin(const std::string& path, size_t count) {
    std::vector<float> data(count);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open " + path);
    }
    f.read(reinterpret_cast<char*>(data.data()), count * sizeof(float));
    if (!f) {
        throw std::runtime_error("Failed to read full float data from " + path);
    }
    return data;
}

std::vector<uint8_t> load_u8_bin(const std::string& path, size_t count) {
    std::vector<uint8_t> data(count);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open " + path);
    }
    f.read(reinterpret_cast<char*>(data.data()), count * sizeof(uint8_t));
    if (!f) {
        throw std::runtime_error("Failed to read full label data from " + path);
    }
    return data;
}

int main() {
    const std::string base = "/pscratch/sd/a/anirudh6/cs5220/project/data/mnist/processed/";

    const size_t train_n = 60000;
    const size_t test_n  = 10000;
    const size_t dim     = 784;

    try {
        auto train_images = load_float_bin(base + "train_images.bin", train_n * dim);
        auto train_labels = load_u8_bin(base + "train_labels.bin", train_n);
        auto test_images  = load_float_bin(base + "test_images.bin", test_n * dim);
        auto test_labels  = load_u8_bin(base + "test_labels.bin", test_n);

        std::cout << "Loaded MNIST successfully\n";
        std::cout << "train_images elements = " << train_images.size() << "\n";
        std::cout << "train_labels elements = " << train_labels.size() << "\n";
        std::cout << "test_images elements  = " << test_images.size() << "\n";
        std::cout << "test_labels elements  = " << test_labels.size() << "\n";

        std::cout << "First train label = " << static_cast<int>(train_labels[0]) << "\n";
        std::cout << "First pixel of first image = " << train_images[0] << "\n";
        std::cout << "Last pixel of first image = " << train_images[783] << "\n";


        float pixel_sum = 0.0f;
        int nonzero_count = 0;

        std::cout << "First 20 nonzero pixels in first image:\n";
        for (size_t i = 0; i < dim; i++) {
            float v = train_images[i];   // first image = indices [0..783]
            pixel_sum += v;
            if (v != 0.0f) {
                nonzero_count++;
                if (nonzero_count <= 20) {
                    std::cout << "  idx " << i << " = " << v << "\n";
                }
            }
        }

        std::cout << "Pixel sum of first image = " << pixel_sum << "\n";
        std::cout << "Nonzero pixel count = " << nonzero_count << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}