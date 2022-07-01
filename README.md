# DCGAN in gesture recognition
Deep Convolutional Generative Adversarial Network (DCGAN) used to produce the artificial signatures of gesture, based on samples collected from AWR1642 automotive radar. This dataset was used in Radar-Based Recognition of Multiple Hand Gestures Using Long Short-Term Memory (LSTM) Neural Network ([MDPI publication](https://www.mdpi.com/2079-9292/11/5/787),
[code repository](https://github.com/piotrgrobelny/AWR1642-hand-gesture-recognition)). The bottleneck for radar deep neural networks (DNN) is the lack of large amounts of data that are necessary for proper training. Creating radar datasets is time-consuming and expensive. To overcome this, i'm working on the idea of artificialy created samples by DCGAN.

## Gesture samples
`Data_radar` directory consist of 4600 samples for 12 different gestures. Gesture types:

<img src="https://github.com/piotrgrobelny/DCGAN-Radar-signatures-AWR1642/blob/main/Example_images/gestures.png" width="300"/>

The AWR1642 board processes data using onboard DSP, which extracts parameters of the targets from raw radar signatures. The information are sent via USB interface in packages. Each package corresponds to a single frame with three columns,
one for each parameter respectively: Doppler velocity (m/s), x and y position (m). Thus, the dimensions of the single frame are 3×N, where N stands for the number of
detected objects. Hence, each gesture has dimensions of M ×3×N, where M stands for the number of collected frames. The 3D representation of the "hand away" (G3) gesture samples are shown below:

<img src="https://github.com/piotrgrobelny/DCGAN-Radar-signatures-AWR1642/blob/main/Example_images/hand%20away.png" width="800"/>

The gesture sample was fixed by zero padding to 80 frames × 3 parameters × 80 targets size. To fit the data into the convolutional layer,
the samples were reshaped into 3x80x80 size. Now, radar signatures can also be represented as 2D image with three RGB layers.

<img src="https://github.com/piotrgrobelny/DCGAN-Radar-signatures-AWR1642/blob/main/Example_images/data_proc.png" width="800"/>

## DCGAN
Block diagram for botch Discriminator and generator:

<img src="https://github.com/piotrgrobelny/DCGAN-Radar-signatures-AWR1642/blob/main/Example_images/dcgan.png" width="400"/>

The generator creates fake samples from the latent space vector z. Input vector can be treated as a noise which G eventually transform into matrix with the same
shape 3x80x80 as the gesture sample. The input data are passed through convolutional, batch norm and uppsample layers with LeakyRelu activation function. The
batch norm layer after convolutional is a critical contribution of the original DCGAN paper. 

## Training
Run `DCGAN_gestures_generator.py` file. All parameters of training are described here:
```
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=80, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
parser.add_argument("--ngpu", type=int, default=1, help="number of used GPU")
opt = parser.parse_args()
```

Progress of generator looks like this in 2D form:


<img src="https://github.com/piotrgrobelny/DCGAN-Radar-signatures-AWR1642/blob/main/Example_images/progress.gif" width="400"/>

## Results
Generated samples in 3D:


<img src="https://github.com/piotrgrobelny/DCGAN-Radar-signatures-AWR1642/blob/main/Example_images/fake%20samples.png" width="800"/>


While the DCGAN algorithm was able to create images resembling images of real datasets, their use as radar training samples was not beneficial to the overall system even for the case of single class of the hand-away gesture.Possible cause is that Deep Convolutional Neural Networks don’t operate with time-sequence data. After reshaping a sample dimensions from 80×3×80 to, 3×80×80 it temporarily lost the information about the relationships between successive frames. It takes the whole gesture as an image, and it treats it in the same way. DCGAN extract features from reshaped gesture that are not important because they contain information about the relationship between pixels, not frames. It should be also added, that the quantitative aspect of radar sensing is more restrictive for this application than the qualitative image generation, i.e. the radar systems measures the exact position and speed of targets, as opposed to the image with shapes.

