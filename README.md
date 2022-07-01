# DCGAN in gesture recognition
Deep Convolutional Generative Adversarial Network (DCGAN) used to produce the artificial signatures of gesture, based on samples collected from AWR1642 automotive radar. This dataset was used in Radar-Based Recognition of Multiple Hand Gestures Using Long Short-Term Memory (LSTM) Neural Network ([MDPI publication](https://www.mdpi.com/2079-9292/11/5/787),
[code repository](https://github.com/piotrgrobelny/AWR1642-hand-gesture-recognition)). The bottleneck for radar deep neural networks (DNN) is the lack of large amounts of data that are necessary for proper training. Creating radar datasets is time-consuming and expensive. To overcome this, i'm working on the idea of artificialy created samples by DCGAN.

## Gesture samples
`Data_radar` directory consist of 4600 samples for 12 different gestures. The AWR1642 board processes data using onboard DSP, which extracts parameters of the targets from raw radar signatures. The information are sent via USB interface in packages. Each package corresponds to a single frame with three columns,
one for each parameter respectively: Doppler velocity (m/s), x and y position (m). Thus, the dimensions of the single frame are 3×N, where N stands for the number of
detected objects. Hence, each gesture has dimensions of M ×3×N, where M stands for the number of collected frames. The 3D representation of the samples is shown
in figure below. 

3d plot

The gesture sample was fixed by zero padding to 80 frames × 3 parameters × 80 targets size. To fit the data into the convolutional layer,
the samples were reshaped into 3x80x80 size. Now, radar signatures can also be represented as 2D image with three RGB layers.

data processing path

## DCGAN
Block diagram for botch Discriminator and generator:

The generator creates fake samples from the latent space vector z. Input vector can be treated as a noise which G eventually transform into matrix with the same
shape 3x80x80 as the gesture sample. The input data are passed through convolutional, batch norm and uppsample layers with LeakyRelu activation function. The
batch norm layer after convolutional is a critical contribution of the original DCGAN paper. 


![Progress](https://github.com/piotrgrobelny/DCGAN-Radar-signatures-AWR1642/blob/main/Example_images/progress.gif)

![Loss of dicriminator and generator](https://github.com/piotrgrobelny/DCGAN-Radar-signatures-AWR1642/blob/main/Example_images/loss.png)

![Real vs fake in 2D space](https://github.com/piotrgrobelny/DCGAN-Radar-signatures-AWR1642/blob/main/Example_images/comparision_1.png)

![Real vs fake in 3D space](https://github.com/piotrgrobelny/DCGAN-Radar-signatures-AWR1642/blob/main/Example_images/comparision_2.png)
