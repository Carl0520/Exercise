# PowerArena Exercise
1. Split data into random train and valid test subsets at the ratio of 3:1:1
2. Pretrained neural network Resnet, vgg16 and vgg19 are used to train the model,
but the pretrained classifer is replaced with customized one.
# Environmet
pytorch v1.0

python 3.6
# Resnet50
Loss:
![image](https://github.com/Carl0520/power/blob/master/resnet50/loss.png)
Final result:
![image](https://github.com/Carl0520/power/blob/master/resnet50/result.png)

# Vgg16
Loss:
![image](https://github.com/Carl0520/power/blob/master/vgg16/loss.png)
Final result:
![image](https://github.com/Carl0520/power/blob/master/vgg16/result.png)

# Vgg19
Loss:
![image](https://github.com/Carl0520/power/blob/master/vgg19/loss.png)
Final result:
![image](https://github.com/Carl0520/power/blob/master/vgg19/result.png)


# Conclusion
Performance: Vgg16 >= Vgg19 > Resnet

It is recommended that you should choose Vgg16 to train the customized classifier.
