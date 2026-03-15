#include <cstdio>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
// #include <opencv2/cudawarping.hpp>
// #include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
// #ifndef WITH_CUDA
// #define WITH_CUDA
// #endif

void SetupCUDA();
void AffineTransform(cv::Mat &src, cv::Mat &dst, float a, float b, float c, float d, float tx, float ty);
void TileScaledImage(cv::Mat &imgScaled, cv::Mat &tiledOutput, int imgWidth, int imgHeight);
void SeamlessTiling(cv::Mat &src);

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <image_file_name>" << std::endl;
		return 1;
	}

	const char *filename = argv[1];
	double scale = 0.5f;
	bool variable_size = false;
	// if false, the size will be original width * height, otherwise, the size will be nearest power of two of resulted image.
	// TODO: vertical and horizontal repeat setting from argv
	// int horizontal_repeat = atoi(argv[2]);
	// int vertical_repeat = atoi(argv[3]);

	cv::useOptimized();
#ifdef WITH_CUDA
	SetupCUDA();
#endif

	// Load the image from the specified file path
	cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);

	// Check if the image was loaded successfully
	if (img.empty())
	{
		std::cerr << "Error: Could not read the image file or the file does not exist." << std::endl;
		return -1;
	}

	int width = img.rows;
	int height = img.cols;
	int channel = img.channels();
	std::cout << "Width: " << width << ", Height: " << height << ", Channel: " << channel << std::endl;

	// scaling -> tiling -> cropping
	cv::Mat imgScaled, tiledOutput, finalOutput, finalOutputCropped;
	if (variable_size)
	{
		width *= scale, height *= scale;
		TileScaledImage(img, tiledOutput, width, height);
	}
	else
	{
		AffineTransform(img, imgScaled, 1. / scale, 0, 0, 1. / scale, 0, 0);
		TileScaledImage(imgScaled, tiledOutput, width, height);
	}

	if (variable_size)
	{
		int adjustedSize = 1;
		// find nearest power of two.
		while (adjustedSize < width)
		{
			adjustedSize <<= 1;
		}
		if (adjustedSize - width > width - (adjustedSize >> 1))
		{
			adjustedSize >>= 1;
		}
		cv::resize(tiledOutput, finalOutputCropped, cv::Size(adjustedSize, adjustedSize));
		height = adjustedSize;
		width = adjustedSize;
	}
	else
	{
		finalOutputCropped = tiledOutput;
	}
	SeamlessTiling(finalOutputCropped);

	cv::namedWindow("Test Result", cv::WINDOW_AUTOSIZE);
	cv::imshow("Test Result", tiledOutput);
	cv::imshow("Final Result Cropped", finalOutputCropped);

	cv::Mat one[3], two[3], three;
	for (int i = 0; i < 3; ++i)
	{
		one[i] = finalOutputCropped;
	}
	for (int i = 0; i < 3; ++i)
	{
		hconcat(one, 3, two[i]);
	}
	vconcat(two, 3, three);
	cv::imshow("Tiled Result", three);

	cv::waitKey(0);
	cv::imwrite("affine.png", finalOutputCropped);
	cv::imwrite("tiled_poisson.png", three);
	cv::destroyAllWindows();

	return 0;
}

#ifdef WITH_CUDA
void SetupCUDA()
{
	int device = cv::cuda::getCudaEnabledDeviceCount();
	printf_s("Number of CUDA devices: %d\n", device);
	int getD = cv::cuda::getDevice();
	cv::cuda::setDevice(getD);
}
#endif

void AffineTransform(cv::Mat &src, cv::Mat &dst, float a, float b, float c, float d, float tx, float ty)
{
	int width = src.cols;
	int height = src.rows;
	int channel = src.channels();
	int resized_width = (int)width * a;
	int resized_height = (int)height * d;

	dst = cv::Mat::zeros(resized_height, resized_width, CV_MAKE_TYPE(8U, channel));
	cv::Mat M = (cv::Mat_<float>(2, 3) << a, b, tx, c, d, ty);
#ifdef WITH_CUDA
	cv::cuda::GpuMat gSrc, gDst;
	// upload src and dst matrix
	gSrc.upload(src);
	gDst.upload(dst);
	cv::cuda::warpAffine(gSrc, gDst, M, gDst.size());
	gDst.download(dst);
#else
	cv::warpAffine(src, dst, M, dst.size());
#endif
}

void TileScaledImage(cv::Mat &imgScaled, cv::Mat &tiledOutput, int imgWidth, int imgHeight)
{
	const int maxScale = 10;
	int tileCount = (imgWidth + imgScaled.cols - 1) / imgScaled.cols; // calc minimum tiling count to cover original image size.
	cv::Mat finalResult, resT[maxScale], T[maxScale];
	for (int i = 0; i < tileCount; ++i)
	{
		T[i] = imgScaled;
	}
	for (int i = 0; i < tileCount; ++i)
	{
		cv::hconcat(T, tileCount, resT[i]);
	}
	cv::vconcat(resT, tileCount, finalResult);
	const cv::Rect roi(0, 0, imgWidth, imgHeight);
	tiledOutput = finalResult(roi).clone();
}

void PoissonSolver(cv::Mat &src)
{
	// SOR method (improved gauss seidel method)
	int col = src.cols;
	int row = src.rows;
	cv::Mat dst = src.clone();
	// this is a sparse matrix since this process only see 4-adjacent
	static constexpr double EPS = 1e-5;
	static constexpr double PI = 3.1415926535897932384626;
	static constexpr int iteration = 20000;
	static constexpr int dx[4]{0, -1, 0, 1};
	static constexpr int dy[4]{1, 0, -1, 0};
	// left and right boundary conditions
	for (int y = 0; y < row; ++y)
	{
		dst.at<double>(y, 0) = (src.at<double>(y, 0) + src.at<double>(y, col - 1)) / 2;
		dst.at<double>(y, col - 1) = dst.at<double>(y, 0);
	}
	// top and bottom boundary conditions
	for (int x = 0; x < col; ++x)
	{
		dst.at<double>(0, x) = (src.at<double>(0, x) + src.at<double>(row - 1, x)) / 2;
		dst.at<double>(row - 1, x) = dst.at<double>(0, x);
	}

	double rect = 0.5 * (cos(PI / src.rows) + cos(PI / src.cols));
	double omega = 2. / (1 + sqrt(1 - rect * rect));
	// optimum omega: https://www.sciencedirect.com/science/article/pii/S0893965908001523
	for (int it = 0; it < iteration; ++it)
	{
		bool ok = true;
		for (int y = 1; y < row - 1; ++y)
		{
			for (int x = 1; x < col - 1; ++x)
			{
				double fq = 0.0, vpq = 0.0;
				for (int i = 0; i < 4; ++i)
				{
					int ny = y + dy[i], nx = x + dx[i];
					fq += dst.at<double>(ny, nx);
					vpq += src.at<double>(y, x) - src.at<double>(ny, nx); // guided vector depend on what to do
				}
				double fp = (fq + vpq) / 4.;
				double err = fabs(fp - dst.at<double>(y, x));
				if (err > EPS)
				{
					ok = false;
				}
				dst.at<double>(y, x) = (1. - omega) * dst.at<double>(y, x) + omega * fp;
			}
		}
		if (ok)
		{
			break;
		}
	}
	src = dst.clone();
}

// build SparseMatrix + Cholesky decomposition to solve Ax = b
void BuildMatrixAndSolve(cv::Mat &src)
{
	int col = src.cols;
	int row = src.rows;
	int n = col * row;
	// solve n consecutive linear equations
	auto A = Eigen::SparseMatrix<double>(n, n);
	auto b = Eigen::VectorXd(n);
	// this is a sparse matrix since this process only see 4-adjacent
	static constexpr int dx[5]{0, -1, 0, 1, 0};
	static constexpr int dy[5]{-1, 0, 0, 0, 1};
	A.reserve(n * 5); // each pixel, 4 directions
	std::vector<double> _b(n);
	for (int y = 0; y < row; ++y)
	{
		for (int x = 0; x < col; ++x)
		{
			int id = y * col + x;
			A.startVec(id);
			double fq = 0.0, vpq = 0.0;
			for (int i = 0; i < 5; ++i)
			{
				int ny = y + dy[i], nx = x + dx[i];
				int nid = ny * col + nx;
				double coef = (id == nid ? 4.0 : -1.0); // diagonal: 4.0
				if (ny == -1 || ny == row)
				{
					// top or bottom boundary condition
					// TODO: improve this condition to get better result.
					vpq += (src.at<double>(0, x) + src.at<double>(row - 1, x)) / 2;
				}
				else if (nx == -1 || nx == col)
				{
					// left or right boundary condition
					// TODO: improve this condition to get better result.
					vpq += (src.at<double>(y, 0) + src.at<double>(y, col - 1)) / 2;
				}
				else
				{
					vpq += src.at<double>(y, x) - src.at<double>(ny, nx);
					A.insertBack(nid, id) = coef;
				}
			}
			_b[id] = vpq;
		}
	}
	A.finalize();
	b = Eigen::Map<Eigen::VectorXd>(&_b[0], _b.size());

	// solve Ax = b
	Eigen::VectorXd x;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A);
	if (solver.info() != Eigen::Success)
	{
		std::cerr << "decomposition failed" << std::endl;
	}
	x = solver.solve(b);
	if (solver.info() != Eigen::Success)
	{
		std::cerr << "solving failed" << std::endl;
	}

	// copy result to pixel
	for (int i = 0; i < row; ++i)
	{
		double *ptr = src.ptr<double>(i);
		int base = i * col;
		for (int j = 0; j < col; ++j)
		{
			ptr[j] = x[base + j];
		}
	}
}

void SeamlessTiling(cv::Mat &src)
{
	cv::Mat channels[3];
	cv::split(src, channels);
	for (cv::Mat &ch : channels)
	{
		ch.convertTo(ch, CV_64F);
		BuildMatrixAndSolve(ch);
		// PoissonSolver(ch)
		ch.convertTo(ch, CV_8U);
	}
	cv::merge(channels, 3, src);
}