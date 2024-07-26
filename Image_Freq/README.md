#### 이미지 푸리에 변환

- 일반적으로 이미지는 공간 영역(Spatial Domain)에 표현되며, 각 픽셀 값은 해당 위치의 밝기나 색상 정보를 나타냄.

- 오디오는 신호가 시간도메인에 표현되지만 이미지는 공간도메인에 표현 됌.
그래서 푸리에 변환된 결과 주파수를 공간주파수(Spatial Frequency)라고 부름

- 공간 영역에서는 픽셀 값으로 이미지를 표현하며 주파수 영역에서는 계수로 표현됌

- 대표적인 이미지의 주파수 변환 방법은 DCT(Discrete Cosine Transform), DFT(Discrete Fourier Transform) 가 있음