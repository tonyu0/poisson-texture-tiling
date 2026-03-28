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

    // ===== Copied from src/main.cpp =====
    EMSCRIPTEN_KEEPALIVE
    void PoissonSolver_SimplicialLDLT(Eigen::SparseMatrix<float> &A, cv::Mat &_b, cv::Mat &x)
    {
        int h = x.rows;
        int w = x.cols;
        int n = w * h;
        Eigen::Map<Eigen::VectorXf> b(_b.ptr<float>(), _b.total());
        // 3. Solve the sparse linear system (Ax = b)
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver(A);
        if (solver.info() != Eigen::Success)
        {
            std::cerr << "decomposition failed" << std::endl;
        }
        Eigen::VectorXf _x = solver.solve(b);
        if (solver.info() != Eigen::Success)
        {
            std::cerr << "solving failed" << std::endl;
        }

        // 4. Adjust the global color offset
        // Since the above linear system determines only relative pixel differences, not the absolute color level.
        // Just as determining a particular solution from a general solution, it needs to determine the global color.
        // This step restores the global color while preserving the seamless property.

        float mean_diff = 0.;
        for (int i = 0; i < h; ++i)
        {
            float *ptr = x.ptr<float>(i);
            int base = i * w;
            for (int j = 0; j < w; ++j)
            {
                mean_diff += ptr[j] - _x[base + j];
                ptr[j] = _x[base + j];
            }
        }
        mean_diff /= h * w;
        x += cv::Scalar(mean_diff);
    }

    EMSCRIPTEN_KEEPALIVE
    void PoissonSolver_FFT(cv::Mat &_b, cv::Mat &x)
    {
        CV_Assert(x.type() == CV_32F);

        cv::Mat b = _b.reshape(0, x.rows);

        int h = b.rows;
        int w = b.cols;

        // --- FFT ---
        cv::Mat b_complex;
        cv::dft(b, b_complex, cv::DFT_COMPLEX_OUTPUT);

        // --- Solve the Poisson equation in Fourier Space ---
        // Memo:
        // Δx{i,j}​= 4x{i,j} - x{i+1,j} ​- x{i−1,j}​ - x{i,j+1} ​- x{i,j−1}
        // と、Aの構築時には出していた
        // x{i,j}=ei^(kx*​i + ky*​j)
        // これはフーリエ空間上のx{i,j}
        // Δx=(4 - e^(i*kx)​ - e^(−i*kx)​ - e^(i*ky)​ - e^(−i*ky)​) * x
        // これをオイラーの公式でまとめると
        // λ(kx​,ky​)=4 - 2cos(kx​) - 2cos(ky​)

        cv::Mat x_complex = cv::Mat::zeros(h, w, CV_32FC2);

        for (int i = 0; i < h; ++i)
        {
            float ky = 2.0f * CV_PI * i / h;

            for (int j = 0; j < w; ++j)
            {
                float kx = 2.0f * CV_PI * j / w;

                float lambda = 4.f - 2.f * cos(kx) - 2.f * cos(ky);

                cv::Vec2f b_val = b_complex.at<cv::Vec2f>(i, j);

                if (fabs(lambda) > 1e-5f) // Limit the lower lambda to prevent divergence
                {
                    x_complex.at<cv::Vec2f>(i, j)[0] = b_val[0] / lambda;
                    x_complex.at<cv::Vec2f>(i, j)[1] = b_val[1] / lambda;
                }
            }
        }

        // --- Inverse FFT ---
        cv::Mat x_real;
        cv::dft(x_complex, x_real, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

        // --- Adjust mean value ---
        float mean_diff = 0.;
        for (int i = 0; i < h; ++i)
        {
            for (int j = 0; j < w; ++j)
            {
                mean_diff += x.at<float>(i, j) - x_real.at<float>(i, j);
                x.at<float>(i, j) = x_real.at<float>(i, j);
            }
        }
        mean_diff /= h * w;
        x += cv::Scalar(mean_diff);
    }

    EMSCRIPTEN_KEEPALIVE
    void MakeLaplacian(cv::Mat &src, Eigen::SparseMatrix<float> &A, cv::Mat &b)
    {
        int w = src.cols;
        int h = src.rows;
        int n = w * h;
        // 1. Compute the gradient-based Laplacian from the input image
        // 2. Construct a sparse Laplacian matrix with periodic boundary conditions
        // this is a sparse matrix since this process only see 4-adjacent
        static constexpr int numDir = 4;
        static constexpr int dx[numDir]{0, -1, 1, 0};
        static constexpr int dy[numDir]{-1, 0, 0, 1};

        std::vector<Eigen::Triplet<float>> _A;
        _A.reserve(n * (numDir + 1)); // each pixel, 4 directions + themselves = number of nun-zero elements
        b = cv::Mat(1, n, CV_32F);

        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                int id = y * w + x;
                _A.push_back(Eigen::Triplet<float>(id, id, 4.));

                float fq = 0.;
                float vpq = 0.;
                for (int i = 0; i < numDir; ++i)
                {
                    int ny = y + dy[i];
                    int nx = x + dx[i];

                    if (0 <= ny && ny < h && 0 <= nx && nx < w)
                    {
                        vpq += src.at<float>(y, x) - src.at<float>(ny, nx);
                    }
                    else
                    {
                        // periodic boundary conditions
                        // and let the gradient between (y, x) and (ny, nx) be 0
                        ny = (ny + h) % h;
                        nx = (nx + w) % w;
                    }
                    int nid = ny * w + nx;
                    _A.push_back(Eigen::Triplet<float>(id, nid, -1.));

                    b.at<float>(0, id) = vpq;
                }
            }
        }
        // Load all the data into a matrix at once (maybe auto sorting)
        A = Eigen::SparseMatrix<float>(n, n);
        A.setFromTriplets(_A.begin(), _A.end());

        // Use before solve() for memory saving
        A.makeCompressed();
    }

    EMSCRIPTEN_KEEPALIVE
    void GenerateSeamlessImage(cv::Mat &src, int calc_type)
    {
        // Solve Ax = b
        Eigen::SparseMatrix<float> A;
        cv::Mat b;

        cv::Mat channels[3];
        cv::split(src, channels);
        for (cv::Mat &ch : channels)
        {
            ch.convertTo(ch, CV_32F);

            MakeLaplacian(ch, A, b);
            // PoissonSolver_SOR(ch);
            if (calc_type == 0)
            {
                PoissonSolver_FFT(b, ch);
            }
            else if (calc_type == 1)
            {
                PoissonSolver_SimplicialLDLT(A, b, ch);
            }
            ch.convertTo(ch, CV_8U);
        }
        cv::merge(channels, 3, src);
    }
    // ===== Copied from src/main.cpp =====

    // Main process
    EMSCRIPTEN_KEEPALIVE
    void process_image(uint8_t *ptr, int width, int height, int calc_type)
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

        GenerateSeamlessImage(workMat, calc_type);

        // 3. Restore cv::Mat interpretation to make readable from JS (Canvas requires RGBA)
        cv::cvtColor(workMat, canvasMat, cv::COLOR_RGB2RGBA);
    }
}