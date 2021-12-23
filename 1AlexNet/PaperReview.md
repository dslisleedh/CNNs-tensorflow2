# Alexnet

![https://github.com/dslisleedh/CNNs-tensorflow2/blob/main/1AlexNet/src/Model.jpg?raw=true](https://github.com/dslisleedh/CNNs-tensorflow2/blob/main/1AlexNet/src/Model.jpg?raw=true)

# 요약

 처음으로 CNN을 사용해 다른 방법들을 제치고 ImageNet 대회에서 엄청난 차이로 우승함. 

### ReLU

보통 사용한 Sigmoid($(1 + e^-x)^-1$)나 tahn($(e^x – e^{-x}) / (e^x + e^{-x})$)와 같은 Saturating nonlinear 활성화 함수들은 느림. 이에 Nonlinear 활성화 함수인 ReLU($max(0, x)$)를 사용함

### Local Response Normalization

인접한 커널의 값들로 정규화 하는 것. 즉 하나의 커널이 강하게 활성화된다면, 주변 커널들은 상대적으로 영향력이 낮아지게됨. 실제 뉴런에서 영감을 받았다함.

### Overlapping Pooling

전통적으론 pool size$(k , k)$와 stride$(s,s)$의 크기를 같게($k = s$)만들어서 각각의 pooling unit이 겹치지 않게 했지만, AlexNet에서는 pool size를 stride보다 크게($k > s$)만들어서 겹치는 부분이 생기게함.