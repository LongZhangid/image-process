This code (FTVd) solve the inverse problem by ADMM:

  min_X || DX ||_2 + μ / 2 || K * X - F ||_F ^ 2
  
Description:

  Optical system imaging by beyond fomula:
  
  F = K * X + noise
  
  Here, K is the point spread function of optical system, X is ideal image and noise is Gauss noise of system, thereby, F is imaging by optical system.
  and symbel * is convolution operator.
    
![example](https://github.com/user-attachments/assets/2c3ae67d-bd11-4176-bafd-5b21748f8b5b)

Now, I updata this code that can solve not the L2 constraint but the L1 constraint:

  min_X || DX ||_2 + μ || K * X - F ||_1

We can use parameter opt = 'L2'  to choose the L2 constraint, or opt = 'L1' to choose the L1 constraint.

![example2](https://github.com/user-attachments/assets/f7176ae3-96ad-4366-9401-5f18071e6188)

For details introduction, please read the theory.pdf file.

2025/07/16 update:

Now, we update this code (FMTVd) that can solve multichanneel information fusion image:

<img width="1200" height="600" alt="example3" src="https://github.com/user-attachments/assets/80fc5e82-09bf-46b6-a935-21546714fcb8" />

2025/08/22 update:
Now, we addition a c++ file that achieve a part of the python file, we can use it to solve that L2 sub-problem.

<img width="560" height="420" alt="snr" src="https://github.com/user-attachments/assets/6cad807c-75ee-49a7-934b-98e4e78eccd2" />
