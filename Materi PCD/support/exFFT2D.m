function exFFT2D

close all
% load image
g = imread('fruit.bmp');

%tampilkan image asal
figure %1
imshow(g)

%lakukan fft & shift hasilnya
fft_g = fft2(g);
fs = fftshift(fft_g);

%tampilkan image hasil fft
figure %2
imshow(log(abs(fft_g)),[])

%kembalikan hasilnya ke image asal
balik = ifft2(fft_g);
figure %3
imshow(uint8(balik))

% -----------------------------------
%coba lihat apa yang terjadi jika hasil fft dimodifikasi

fs1 = fs;
fs1(150:250,150:250) = 0;
%tampilkan image hasil fft
figure %4
imshow(log(abs(fs1)),[])

%kembalikan hasilnya ke image asal
balik = ifft2(ifftshift(fs1));
figure %5
imshow(uint8(balik))

mask = zeros(size(g));
mask(150:250,150:250) = 1;

hasil = mask.*fs;
%tampilkan image hasil fft
figure %6
imshow(log(abs(hasil)),[])

%kembalikan hasilnya ke image asal
balik = ifft2(ifftshift(hasil));
figure %7
imshow(uint8(balik))
