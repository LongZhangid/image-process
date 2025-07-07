This code solve the inverse problem by ADMM:
  min_X || DX ||_2 + μ / 2 || K * X - F || ^ 2
Description:
  Optical system imaging by beyond fomula:
  F = K * X + noise
  Here, K is the point spread function of optical system, X is ideal image and noise is Gauss noise of system, thereby, F is imaging by optical system.
  and symbel * is convolution operator.
    
![example](https://github.com/user-attachments/assets/2c3ae67d-bd11-4176-bafd-5b21748f8b5b)
