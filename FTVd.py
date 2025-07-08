"""
By Long Zhang in CIOMP
2025/7/7
ref: Tao, Min, Junfeng Yang, and Bingsheng He. 
    "Alternating direction algorithms for total variation 
    deconvolution in image reconstruction." TR0918, Department 
    of Mathematics, Nan**g University (2009).
"""


import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

class FTVd:
    """
    Image restoration using the ADMM algorithm.
    Problem:
        min_X || DX ||_2 + μ / 2 || K * X - F ||_2^2
        min_X || DX ||_2 + μ || K * X - F ||_1
    Method:
        ADMM
    """
    def __init__(self, F, H, ref, mu, beta1, gamma, beta2=20, opt='L2', max_iter=1000, tol=1e-6):
        self.m, self.n = F.shape
        self.H = H
        self.F = F
        self.ref = ref
        self.mu = mu
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.opt = opt  # Optimization method: constraint is 'L2' or 'L1'
        self.max_iter = max_iter
        self.tol = tol
        self.Dx = np.zeros((2, 2))
        self.Dx[0, 0] = 1
        self.Dx[0, 1] = -1
        self.Dy = self.Dx.T

    def _solver(self):
        """
        Solve the image restoration problem using ADMM.
        """
        if self.opt == 'L2':
            return self._L2solver()
        elif self.opt == 'L1':
            return self._L1solver()
        else:
            raise ValueError("Invalid optimization method specified.")

    def _L2solver(self):
        """
        Solve the image restoration problem for the L2 constrait using ADMM.
        """
        # Initialize variables
        self.fHtF = np.conj(self._FFT(self.H, self.F.shape)) * self._FFT(self.F)
        self.fDx = np.conj(self._FFT(self.Dx, self.F.shape))
        self.fDy = np.conj(self._FFT(self.Dy, self.F.shape))
        self.fHtH = np.abs(self._FFT(self.H, self.F.shape))**2 
        self.X = self.F.copy()
        self.λ = [np.zeros(self.F.shape), np.zeros(self.F.shape)]  # Dual variables
        self.snrl = []
        for i in range(self.max_iter):
            X_old = self.X.copy()
            self._update_Y()
            self._updateL2_X()
            self._update_lambda()
            self.snrl.append(snr(self.X, self.ref))
            # Check convergence
            if np.linalg.norm(self.X - X_old)/np.linalg.norm(self.X) < self.tol:
                print(f"Converged after {i+1} iterations.")
                break
        else:
            print("Maximum iterations reached without convergence.")
        
        return self.X
    
    def _L1solver(self):
        """
        Solve the image restoration problem for the L1 constraint using ADMM.
        """
        # Initialize variables
        self.fHtF = np.conj(self._FFT(self.H, self.F.shape)) * self._FFT(self.F)
        self.fDx = np.conj(self._FFT(self.Dx, self.F.shape))
        self.fDy = np.conj(self._FFT(self.Dy, self.F.shape))
        self.fH = np.conj(self._FFT(self.H, self.F.shape))
        self.fHtH = np.abs(self.fH)**2 
        self.X = self.F.copy()
        self.λ = [np.zeros(self.F.shape), np.zeros(self.F.shape), np.zeros(self.F.shape)]  # Dual variables
        self.snrl = []
        for i in range(self.max_iter):
            X_old = self.X.copy()
            self._update_Y()
            self._update_Z()
            self._updateL1_X()
            self._update_lambda()
            self._update_lambda3()
            self.snrl.append(snr(self.X, self.ref))
            # Check convergence
            if np.linalg.norm(self.X - X_old)/np.linalg.norm(self.X) < self.tol:
                print(f"Converged after {i+1} iterations.")
                break
        else:
            print("Maximum iterations reached without convergence.")
        
        return self.X

    def _dif(self, X):
        """
        Compute the differential of the image.
        """
        dx = np.diff(X, axis=1, append=X[:, 0:1])
        dy = np.diff(X, axis=0, append=X[0:1, :])
        return [dx, dy]
    
    def _div(self, X, Y):
        """
        Calculate the nagative divergent of the image.
        """
        dx = np.diff(X, axis=1, prepend=X[:, -1:])
        dy = np.diff(Y, axis=0, prepend=Y[-1:, :])
        return -dx-dy
    
    def _FFT(self, X, shape=None):
        """
        Compute the FFT of the image.
        """
        if shape is None:
            return np.fft.fft2(X)
        else:
            return np.fft.fft2(X, s=shape)
    
    def _IFFT(self, X):
        """
        Compute the inverse FFT of the image.
        """
        return np.fft.ifft2(X).real
    
    def _update_Y(self):
        """
        Update the Y subproblem.
        """
        Z_x = self._dif(self.X)[0] + self.λ[0] / self.beta1
        Z_y = self._dif(self.X)[1] + self.λ[1] / self.beta1

        v = np.sqrt(Z_x ** 2 + Z_y ** 2)
        v[v == 0] = 1
        # threshold shrink
        shrink = np.maximum(v - 1/self.beta1, 0) / v
        Y_x = Z_x  * shrink
        Y_y = Z_y  * shrink
        self.Y = (Y_x, Y_y)

    def _update_Z(self):
        """
        Update the Z subproblem for the L2 constraint.
        """
        V3 = self._IFFT(self._FFT(self.H, self.F.shape) * self._FFT(self.X)) - \
                self.F + self.λ[-1] / self.beta2
        self.Z = np.sign(V3) * np.maximum(np.abs(V3) - self.mu / self.beta2, 0)
        
    def _updateL2_X(self):
        """
        Update the X subproblem for the L2 constraint.
        """
        moleculer = self.mu * self.fHtF
        moleculer += self.beta1 * self._FFT(self._div((self.Y[0] - self.λ[0] / self.beta1),
                                                     (self.Y[1] - self.λ[1] / self.beta1)))
        denominator = self.mu * self.fHtH
        denominator += self.beta1 * (np.abs(self.fDx)**2 + np.abs(self.fDy)**2)
        self.X = self._IFFT(moleculer / denominator).real

    def _updateL1_X(self):
        """
        Update the X subproblem for the L1 constraint.
        """
        moleculer = self.beta2 * self.fHtF
        moleculer += self.beta1 * self._FFT(self._div((self.Y[0] - self.λ[0] / self.beta1),
                                                     (self.Y[1] - self.λ[1] / self.beta1)))
        moleculer += self.beta2 * self.fH * self._FFT(self.Z - self.λ[-1] / self.beta2)
        denominator = self.beta2 * self.fHtH
        denominator += self.beta1 * (np.abs(self.fDx)**2 + np.abs(self.fDy)**2)
        self.X = self._IFFT(moleculer / denominator).real
    
    def _update_lambda(self):
        """
        Update the λ variable.
        """
        self.λ[0] = self.λ[0] + self.gamma * self.beta1 * (self._dif(self.X)[0] - self.Y[0])
        self.λ[1] = self.λ[1] + self.gamma * self.beta1 * (self._dif(self.X)[1] - self.Y[1])

    def _update_lambda3(self):
        """
        Update the λ variable for the L1 constraint.
        """
        self.λ[-1] = self.λ[-1] + self.gamma * self.beta2 * \
              (self._IFFT(self._FFT(self.H, self.F.shape) * self._FFT(self.X)) - self.F - self.Z)

def snr(img, ref):
    """
    Calculate signal noise ratio.
    """
    return 10 * np.log10(np.max(ref ** 2)/np.mean((img-ref)**2))

def blur_image(img: np.array, H: np.array) -> np.array:
    """
    Generate blury image
    """
    nimg = np.fft.fft2(img) * np.fft.fft2(H, s=img.shape)
    return np.fft.ifft2(nimg).real

def restore_image(F, H, ref, mu=100, beta=20, gamma=1.618, opt='L1', max_iter=10000, tol=1e-8):
    """
    Restore an image using the FTVd algorithm.
    
    Parameters:
    - F: Input image (numpy array).
    - H: Blur kernel (numpy array).
    - ref: Reference image (numpy array).
    - mu: Regularization parameter.
    - beta1, beta2: Penalty parameter.
    - gamma: Step size for the ADMM update.
    - opt: Optimization method ('L2' or 'L1').
    - max_iter: Maximum number of iterations.
    - tol: Tolerance for convergence.
    
    Returns:
    - Restored image (numpy array).
    - snrl: Signal noise ratio list (List).
    """
    ftvd = FTVd(F, H, ref, mu, beta1=beta, gamma=gamma, opt=opt, max_iter=max_iter, tol=tol)
    out = ftvd._solver()
    snrl = ftvd.snrl
    return out, snrl


if __name__ == "__main__":
    # Load an image
    img = Image.open('./restory_image/colors.png').convert('L')
    X = np.array(img, dtype=float)
    X /= np.amax(X)

    # Define a blur kernel (e.g., Gaussian)
    H = np.array([[1, 4, 4, 4, 1],
                  [4, 6, 6, 6, 4],
                  [4, 6, 8, 6, 4],
                  [4, 6, 6, 6, 4],
                  [1, 4, 4, 4, 1]], dtype=float)
    H /= np.sum(H, dtype=float)

    # Noise the image
    F = blur_image(X, H)
    # Restore the image
    restored_image, snrl = restore_image(F, H, X)

    # Display the original and restored images
    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(X, cmap='gray')
    plt.axis(False)

    plt.subplot(1, 4, 2)
    plt.title('Noisy Image')
    plt.imshow(F, cmap='gray')
    plt.axis(False)

    plt.subplot(1, 4, 3)
    plt.title('Restored Image')
    plt.imshow(restored_image, cmap='gray')
    plt.axis(False)

    plt.tight_layout()
    plt.subplot(1, 4, 4)
    plt.plot(snrl)
    plt.title('snr convolution')
    plt.grid(True)
    plt.show()