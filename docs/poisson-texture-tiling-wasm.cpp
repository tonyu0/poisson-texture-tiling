#include <emscripten.h>
#include <cstdlib> // for malloc, free
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>

extern "C" // Make C++ func name directly usable from JS
{
    // Memory allocation and release is on the C++ side. Returns the "start pointer" of memory allocated to JS
    EMSCRIPTEN_KEEPALIVE
    uint8_t *allocate_buffer(int width, int height)
    {
        // RGBA 4 channels
        return (uint8_t *)malloc(width * height * 4);
    }
    EMSCRIPTEN_KEEPALIVE
    void free_buffer(uint8_t *ptr)
    {
        free(ptr);
    }

    // Copied from src/main.cpp
    EMSCRIPTEN_KEEPALIVE
    void PoissonSolver_SimplicialLDLT(cv::Mat &src)
    {
        int w = src.cols;
        int h = src.rows;
        int n = w * h;
        // 1. Compute the gradient-based Laplacian from the input image
        // 2. Construct a sparse Laplacian matrix with periodic boundary conditions
        auto b = Eigen::VectorXd(n);
        b.setZero();
        // this is a sparse matrix since this process only see 4-adjacent
        static constexpr int numDir = 4;
        static constexpr int dx[numDir]{0, -1, 1, 0};
        static constexpr int dy[numDir]{-1, 0, 0, 1};

        std::vector<Eigen::Triplet<double>> _A;
        _A.reserve(n * (numDir + 1)); // each pixel, 4 directions + themselves = number of nun-zero elements

        std::vector<double> _b(n);
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                int id = y * w + x;
                _A.push_back(Eigen::Triplet<double>(id, id, 4.));

                double fq = 0.;
                double vpq = 0.;
                for (int i = 0; i < numDir; ++i)
                {
                    int ny = y + dy[i];
                    int nx = x + dx[i];

                    if (0 <= ny && ny < h && 0 <= nx && nx < w)
                    {
                        vpq += src.at<double>(y, x) - src.at<double>(ny, nx);
                    }
                    else
                    {
                        // periodic boundary conditions
                        // and let the gradient between (y, x) and (ny, nx) be 0
                        ny = (ny + h) % h;
                        nx = (nx + w) % w;
                    }
                    int nid = ny * w + nx;
                    _A.push_back(Eigen::Triplet<double>(id, nid, -1.));

                    _b[id] = vpq;
                }
            }
        }
        // Load all the data into a matrix at once (maybe auto sorting)
        Eigen::SparseMatrix<double> A(n, n);
        A.setFromTriplets(_A.begin(), _A.end());

        // Use before solve() for memory saving
        A.makeCompressed();

        b = Eigen::Map<Eigen::VectorXd>(&_b[0], _b.size());

        // 3. Solve the sparse linear system (Ax = b)
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A);
        if (solver.info() != Eigen::Success)
        {
            std::cerr << "decomposition failed" << std::endl;
        }
        Eigen::VectorXd x = solver.solve(b);
        if (solver.info() != Eigen::Success)
        {
            std::cerr << "solving failed" << std::endl;
        }

        // 4. Adjust the global color offset
        // Since the above linear system determines only relative pixel differences, not the absolute color level.
        // Just as determining a particular solution from a general solution, it needs to determine the global color.
        // This step restores the global color while preserving the seamless property.

        double mean_diff = 0.;
        for (int i = 0; i < h; ++i)
        {
            double *ptr = src.ptr<double>(i);
            int base = i * w;
            for (int j = 0; j < w; ++j)
            {
                mean_diff += ptr[j] - x[base + j];
                ptr[j] = x[base + j];
            }
        }
        mean_diff /= h * w;
        src += cv::Scalar(mean_diff);
    }

    EMSCRIPTEN_KEEPALIVE
    void GenerateSeamlessImage(cv::Mat &src)
    {
        // TODO: separate the process building matrix

        cv::Mat channels[3];
        cv::split(src, channels);
        for (cv::Mat &ch : channels)
        {
            ch.convertTo(ch, CV_64F);
            // PoissonSolver_SOR(ch);
            PoissonSolver_SimplicialLDLT(ch);
            ch.convertTo(ch, CV_8U);
        }
        cv::merge(channels, 3, src);
    }

    // Main process
    EMSCRIPTEN_KEEPALIVE
    void process_image(uint8_t *ptr, int width, int height)
    {
        int total_pixels = width * height;

        // for (int i = 0; i < total_pixels; ++i)
        // {
        //     // Image data arrangement on memory is like: RGBARGBARGBA...
        //     ptr[i * 4] = 255; // maximize R channel
        // }

        // 1. Create cv::Mat from data on memory shared with JS
        cv::Mat canvasMat(height, width, CV_8UC4, ptr);

        // 2. Change cv::Mat interpretation for main process
        cv::Mat workMat;
        cv::cvtColor(canvasMat, workMat, cv::COLOR_RGBA2RGB); // RGBA -> RGB

        GenerateSeamlessImage(workMat);

        // 3. Restore cv::Mat interpretation to make readable from JS (Canvas requires RGBA)
        cv::cvtColor(workMat, canvasMat, cv::COLOR_RGB2RGBA);
    }
}