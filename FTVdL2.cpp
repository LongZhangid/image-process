#define cimg_use_png

#include <CImg.h>
#include <Eigen/Dense>
#include <fftw3.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <complex>
#include <stdexcept>
#include <algorithm>
#include <matplot/matplot.h>


using namespace cimg_library;
using namespace Eigen;
using namespace std;

namespace plot = matplot;


VectorXcd fftimg(const MatrixXd& img);
MatrixXd ifftimg(const VectorXcd& freq, int width, int height);

class FTVd {
private:
    MatrixXd F;          // Input image
    MatrixXd H;          // Blur kernel
    MatrixXd ref;        // Reference image
    double mu;               // Regularization parameter
    double beta1;            // ADMM parameter
    double gamma;            // ADMM parameter
    int max_iter;            // Maximum iterations
    double tol;              // Convergence tolerance

    // Variables for FFT operations
    VectorXcd fHtF, fDx, fDy, fHtH;

    // ADMM variables
    MatrixXd X;
    vector<MatrixXd> lambda;
    vector<MatrixXd> Y;  // Y[0] = Y_x, Y[1] = Y_y
    vector<double> snrl;

    // Image dimensions
    int width, height;

    // padding
    MatrixXd pad_kernel_to_image_size(const MatrixXd& kernel, int target_width, int target_height) {
        MatrixXd padded = MatrixXd::Zero(target_width, target_height);

        int k_width = kernel.rows();
        int k_height = kernel.cols();

        // Calculate center point
        int center_x = target_width / 2;
        int center_y = target_height / 2;
        int start_x = center_x - k_width / 2;
        int start_y = center_y - k_height / 2;

        // Addition kernel to center
        for (int i = 0; i < k_height; i++) {
            for (int j = 0; j < k_width; j++) {
                int x = (start_x + j) % target_width;
                int y = (start_y + i) % target_height;
                padded(x, y) = kernel(j, i);
            }
        }

        return padded;
    }

    // FFT helper functions
    VectorXcd fft2(const MatrixXd& img) {
        int N = img.rows() * img.cols();
        fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
        fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

        // Fill input
        for (int i = 0; i < img.cols(); i++) {
            for (int j = 0; j < img.rows(); j++) {
                int idx = i * img.rows() + j;
                in[idx][0] = img(j, i);
                in[idx][1] = 0;
            }
        }

        // Create plan and execute FFT
        fftw_plan plan = fftw_plan_dft_2d(img.cols(), img.rows(), in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);

        // Convert to complex vector
        VectorXcd result(N);
        for (int i = 0; i < N; i++) {
            result(i) = complex<double>(out[i][0], out[i][1]);
        }

        // Clean up
        fftw_destroy_plan(plan);
        fftw_free(in);
        fftw_free(out);

        return result;
    }

    MatrixXd ifft2(const VectorXcd& freq) {
        int N = width * height;
        fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
        fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

        // Fill input
        for (int i = 0; i < N; i++) {
            in[i][0] = freq(i).real();
            in[i][1] = freq(i).imag();
        }

        // Create plan and execute inverse FFT
        fftw_plan plan = fftw_plan_dft_2d(height, width, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(plan);

        // Convert to image (normalize by N)
        MatrixXd result = MatrixXd::Zero(width, height);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                result(j, i) = out[idx][0] / N;
            }
        }

        // Clean up
        fftw_destroy_plan(plan);
        fftw_free(in);
        fftw_free(out);

        return result;
    }

    // Helper function for element-wise multiplication of frequency domain arrays
    VectorXcd freq_mult(const VectorXcd& a, const VectorXcd& b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("Frequency arrays size mismatch in freq_mult");
        }

        return a.array() * b.array();
    }

    // Helper function for element-wise division of frequency domain arrays
    VectorXcd freq_div(const VectorXcd& a, const VectorXcd& b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("Frequency arrays size mismatch in freq_div");
        }

        VectorXcd result(a.size());
        for (int i = 0; i < a.size(); i++) {
            // Avoid division by zero
            if (abs(b(i)) < 1e-12) {
                result(i) = a(i) / complex<double>(1e-12, 0);
            }
            else {
                result(i) = a(i) / b(i);
            }
        }
        return result;
    }

    // Helper function for element-wise addition of frequency domain arrays
    VectorXcd freq_add(const VectorXcd& a, const VectorXcd& b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("Frequency arrays size mismatch in freq_add");
        }

        return a + b;
    }

    // Helper function for element-wise multiplication with scalar
    VectorXcd freq_scale(const VectorXcd& a, double scale) {
        return a * scale;
    }

    // Helper function for element-wise conjugation
    VectorXcd freq_conj(const VectorXcd& a) {
        VectorXcd result(a.size());
        for (int i = 0; i < a.size(); i++) {
            result(i) = conj(a(i));
        }
        return result;
    }

    // Helper function for element-wise absolute square
    VectorXcd freq_abs2(const VectorXcd& a) {
        VectorXcd result(a.size());
        for (int i = 0; i < a.size(); i++) {
            double mag = abs(a(i));
            result(i) = complex<double>(mag * mag, 0.0);
        }
        return result;
    }

    // Differential operators
    vector<MatrixXd> dif(const MatrixXd& img) {
        MatrixXd dx = MatrixXd::Zero(width, height);
        MatrixXd dy = MatrixXd::Zero(width, height);

        // Calculate differential along x direction
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width - 1; j++) {
                dx(j, i) = img(j + 1, i) - img(j, i);
            }
            // Circular
            dx(width - 1, i) = img(0, i) - img(width - 1, i);
        }

        // Calculate differential along x direction
        for (int j = 0; j < width; j++) {
            for (int i = 0; i < height - 1; i++) {
                dy(j, i) = img(j, i + 1) - img(j, i);
            }
            // Circular
            dy(j, height - 1) = img(j, 0) - img(j, height - 1);
        }

        return { dx, dy };
    }

    MatrixXd div(const MatrixXd& X, const MatrixXd& Y) {
        MatrixXd result = MatrixXd::Zero(width, height);

        // Calculate negative gradient
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                // x direction
                int prev_j = (j == 0) ? width - 1 : j - 1;
                double dx = X(j, i) - X(prev_j, i);

                // y direction
                int prev_i = (i == 0) ? height - 1 : i - 1;
                double dy = Y(j, i) - Y(j, prev_i);

                result(j, i) = -dx - dy;
            }
        }

        return result;
    }

    // SNR calculation
    double snr(const MatrixXd& X, const MatrixXd& ref) {
        double signal = ref.array().square().sum();
        double noise = (X - ref).array().square().sum();

        if (noise < 1e-10) {
            return 100.0;
        }

        return 10 * log10(signal / noise);
    }

    // Update functions for ADMM
    void update_Y() {
        vector<MatrixXd> gradX = dif(X);
        MatrixXd Z_x = gradX[0] + lambda[0] / beta1;
        MatrixXd Z_y = gradX[1] + lambda[1] / beta1;

        MatrixXd v = (Z_x.array().square() + Z_y.array().square()).sqrt();

        // avoid division zero error
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (v(j, i) == 0) v(j, i) = 1;
            }
        }

        MatrixXd shrink = (v.array() - 1.0 / beta1).max(0.0) / v.array();
        Y[0] = Z_x.array() * shrink.array();
        Y[1] = Z_y.array() * shrink.array();
    }

    void updateL2_X() {
        // Numerator
        VectorXcd numerator = freq_scale(fHtF, mu);

        MatrixXd divY = div(Y[0] - lambda[0] / beta1, Y[1] - lambda[1] / beta1);
        VectorXcd fftDivY = fft2(divY);
        VectorXcd temp = freq_scale(fftDivY, beta1);
        numerator = freq_add(numerator, temp);

        // Denominator
        VectorXcd denominator = freq_scale(fHtH, mu);

        // Calculate |fDx|^2 + |fDy|^2
        VectorXcd abs2_Dx = freq_abs2(fDx);
        VectorXcd abs2_Dy = freq_abs2(fDy);
        VectorXcd temp2 = abs2_Dx + abs2_Dy;

        temp2 = freq_scale(temp2, beta1);
        denominator = freq_add(denominator, temp2);

        // Solve X
        VectorXcd X_freq = freq_div(numerator, denominator);
        X = ifft2(X_freq);
    }

    void update_lambda() {
        vector<MatrixXd> gradX = dif(X);
        lambda[0] = lambda[0] + gamma * beta1 * (gradX[0] - Y[0]);
        lambda[1] = lambda[1] + gamma * beta1 * (gradX[1] - Y[1]);
    }

public:
    // Constructor
    FTVd(const MatrixXd& F, const MatrixXd& H, const MatrixXd& ref,
        double mu, double beta1, double gamma,
        int max_iter = 5000, double tol = 1e-8)
        : F(F), H(H), ref(ref), mu(mu), beta1(beta1), gamma(gamma),
        max_iter(max_iter), tol(tol) {

        width = F.rows();
        height = F.cols();

        // Check size
        if (ref.rows() != width || ref.cols() != height) {
            throw std::runtime_error("Reference image size mismatch");
        }

        // Initialization variable
        X = F;

        lambda.resize(2);
        lambda[0] = MatrixXd::Zero(width, height);
        lambda[1] = MatrixXd::Zero(width, height);
        Y.resize(2);
        Y[0] = MatrixXd::Zero(width, height);
        Y[1] = MatrixXd::Zero(width, height);

        // Pre-calculate FFT
        try {
            // found differential generator Dx ºÍ Dy (2x2)
            MatrixXd Dx_small(2, 2), Dy_small(2, 2);
            Dx_small << 1, -1, 0, 0;
            Dy_small << 1, 0, -1, 0;

            // padding Dx and Dy to figure size
            MatrixXd Dx_padded = pad_kernel_to_image_size(Dx_small, width, height);
            MatrixXd Dy_padded = pad_kernel_to_image_size(Dy_small, width, height);

            // padding H kernel to figure size
            MatrixXd H_padded = pad_kernel_to_image_size(H, width, height);

            // Calculate frequency
            fHtF = freq_mult(freq_conj(fft2(H_padded)), fft2(F));
            fDx = fft2(Dx_padded);
            fDy = fft2(Dy_padded);
            fHtH = freq_abs2(fft2(H_padded));
        }
        catch (const std::exception& e) {
            throw std::runtime_error(std::string("FFT initialization failed: ") + e.what());
        }
    }

    // Main solver function
    MatrixXd solver() {
        cout << "Starting L2 solver..." << endl;

        for (int i = 0; i < max_iter; i++) {
            MatrixXd X_old = X;

            try {
                update_Y();
                updateL2_X();
                update_lambda();
            }
            catch (const std::exception& e) {
                cerr << "Error in iteration " << i << ": " << e.what() << endl;
                break;
            }

            snrl.push_back(snr(X, ref));

            // Check convergence
            double norm_diff = (X - X_old).norm();
            double norm_X = X.norm();

            if (i % 100 == 0) {
                cout << "Iteration " << i << ", SNR: " << snrl.back()
                    << ", Relative change: " << norm_diff / max(norm_X, 1e-10) << endl;
            }

            if (norm_diff / max(norm_X, 1e-10) < tol) {
                cout << "Converged after " << i + 1 << " iterations." << endl;
                break;
            }

            if (i == max_iter - 1) {
                cout << "Maximum iterations reached without convergence." << endl;
            }
        }

        return X;
    }

    // Get SNR history
    vector<double> getSNRHistory() const {
        return snrl;
    }
};

// Example usage
int main() {
    try {
        // Load images
        CImg<double> ref_img("colors.png");

        if (ref_img.is_empty()) {
            cerr << "Failed to load image 'colors.png'" << endl;
            return 1;
        }

        // Convert to grayscale
        if (ref_img.spectrum() > 1) {
            ref_img = ref_img.get_channel(0);
        }

        // Display original image
        CImgDisplay orig_disp(ref_img, "Original Image", 0);
        ref_img.save("original_image.bmp");

        // Convert CImg to Eigen Matrix
        int width = ref_img.width();
        int height = ref_img.height();
        MatrixXd ref(width, height);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                ref(j, i) = ref_img(j, i);
            }
        }

        // Create blur kernel (5x5 Gaussian)
        MatrixXd H(5, 5);
        double sigma = 1.0;
        double sum = 0.0;
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                double x = i - 2;
                double y = j - 2;
                H(j, i) = exp(-(x * x + y * y) / (2 * sigma * sigma));
                sum += H(j, i);
            }
        }
        H /= sum;  // Normalize kernel

        // Generate blur image using FFT
        MatrixXd H_padded = MatrixXd::Zero(width, height);

        int center_x = width / 2;
        int center_y = height / 2;
        int h_width = H.rows();
        int h_height = H.cols();
        int start_x = center_x - h_width / 2;
        int start_y = center_y - h_height / 2;

        // Copy H to H_padded's center
        for (int i = 0; i < h_height; i++) {
            for (int j = 0; j < h_width; j++) {
                int x = (start_x + j + width) % width;
                int y = (start_y + i + height) % height;
                H_padded(x, y) = H(j, i);
            }
        }

        // FFT to H_padded
        VectorXcd H_freq = fftimg(H_padded);

        // FFT to reference image
        VectorXcd ref_freq = fftimg(ref);

        // multiply in frequency domain
        VectorXcd F_freq = ref_freq.array() * H_freq.array();

        // IFFT to get blur image
        MatrixXd F = ifftimg(F_freq, width, height);

        // Convert Eigen Matrix to CImg for display
        CImg<double> blur_img(width, height, 1, 1, 0);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                blur_img(j, i) = F(j, i);
            }
        }
        // Display blur image
        CImgDisplay blur_disp(blur_img, "Blurred Image", 0);
        blur_img.save("blurred_image.bmp");

        // Create FTVd object and solve
        FTVd ftvd(F, H, ref, 100, 20, 1.618);
        MatrixXd result = ftvd.solver();

        // Convert Eigen Matrix to CImg for displaying
        CImg<double> result_img(width, height, 1, 1, 0);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                result_img(j, i) = result(j, i);
            }
        }

        // Display result
        CImgDisplay Result_disp(result_img, "Restored Image", 0);
        result_img.save("restored_image.bmp");

        // Save SNR history
        vector<double> snr_history = ftvd.getSNRHistory();
        if (!snr_history.empty()) {
            cout << "Final SNR: " << snr_history.back() << " dB" << endl;

            // Plot SNR curve
            vector<double> iterations(snr_history.size());
            for (size_t i = 0; i < iterations.size(); i++) {
                iterations[i] = i;
            }

            plot::figure();
            plot::plot(iterations, snr_history);
            plot::title("SNR Convergence");
            plot::xlabel("Iteration");
            plot::ylabel("SNR (dB)");
            plot::grid(true);
            plot::save("snr_convergence.png");
            plot::show();
        }

    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}

// FFT helper functions
VectorXcd fftimg(const MatrixXd& img) {
    int width = img.rows();
    int height = img.cols();
    int N = width * height;
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

    // Fill input
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            in[idx][0] = img(j, i);
            in[idx][1] = 0;
        }
    }

    // Create plan and execute FFT
    fftw_plan plan = fftw_plan_dft_2d(height, width, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Convert to complex vector
    VectorXcd result(N);
    for (int i = 0; i < N; i++) {
        result(i) = complex<double>(out[i][0], out[i][1]);
    }

    // Clean up
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return result;
}

MatrixXd ifftimg(const VectorXcd& freq, int width, int height) {
    int N = width * height;
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

    // Fill input
    for (int i = 0; i < N; i++) {
        in[i][0] = freq(i).real();
        in[i][1] = freq(i).imag();
    }

    // Create plan and execute inverse FFT
    fftw_plan plan = fftw_plan_dft_2d(height, width, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Convert to image (normalize by N)
    MatrixXd result = MatrixXd::Zero(width, height);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            result(j, i) = out[idx][0] / N;
        }
    }

    // Clean up
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return result;
}
