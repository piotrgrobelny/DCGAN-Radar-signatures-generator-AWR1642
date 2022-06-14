import argparse
import os
import numpy as np
import random
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.autograd import Variable
import torch.nn as nn
import torch
import torchvision.utils as vutils
from IPython.display import HTML

# Root directory for dataset

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
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
parser.add_argument("--load_dataset", type=int, default=0, help="0 if you want to load CSV sample files, 1 if you already did it")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Gestures():

    #Directories
    ARM_TO_LEFT = "arm_to_left"
    ARM_TO_RIGHT = "arm_to_right"
    FIST_HORIZONTALLY = "close_fist_horizontally"
    FIST_PERPENDICULARLY = "close_fist_perpendicularly"
    HAND_CLOSE = "hand_closer"
    HAND_AWAY = "hand_away"
    HAND_LEFT = "hand_to_left"
    HAND_RIGHT = "hand_to_right"
    HAND_DOWN = "hand_down"
    HAND_UP = "hand_up"
    PALM_DOWN = "hand_rotation_palm_down"
    PALM_UP = "hand_rotation_palm_up"
    STOP_GESTURE = "stop_gesture"

    LABELS = {
        HAND_AWAY: 0, HAND_AWAY: 0, ARM_TO_RIGHT: 0, ARM_TO_LEFT: 0, HAND_LEFT: 0, HAND_RIGHT: 0,
        FIST_PERPENDICULARLY: 0, FIST_PERPENDICULARLY: 0, HAND_DOWN: 0, HAND_UP: 0,
        PALM_UP: 0, PALM_DOWN: 0,
    }

    global class_number
    class_number = len(LABELS)
    training_data = []

    def ReadDatabase(self, dir):
        dataset = []
        dirpath = "data_radar/" + dir + "/"

        for gesture in os.listdir(dirpath):
            path = dirpath + gesture
            data = np.loadtxt(path, delimiter=",", skiprows=2)  # skip header and null point

            FrameNumber = 1   # counter for frames
            pointlenght = opt.img_size  # maximum number of points in array
            framelenght = opt.img_size  # maximum number of frames in array
            frame_parameters = opt.channels
            datalenght = int(len(data))
            gesturedata = np.zeros((framelenght, frame_parameters, pointlenght))
            counter = 0

            while counter < datalenght:
                velocity = np.zeros(pointlenght)
                peak_val = np.zeros(pointlenght)
                x_pos = np.zeros(pointlenght)
                y_pos = np.zeros(pointlenght)
                object_number = np.zeros(pointlenght)
                iterator = 0

                try:
                    while data[counter][0] == FrameNumber:
                        object_number = data[counter][1]
                        range = data[counter][2]
                        velocity[iterator] = round(data[counter][3],3)
                        peak_val[iterator] = data[counter][4]
                        x_pos[iterator] = round(data[counter][5],3)
                        y_pos[iterator] = round(data[counter][6],3)
                        iterator += 1
                        counter += 1
                except:
                    pass

                # Parameters you want to extract from CSV file
                framedata = np.array([x_pos, y_pos, velocity])


                try:
                    gesturedata[FrameNumber - 1] = framedata
                except:
                    pass
                FrameNumber += 1

            dataset.append(gesturedata)
            number_of_samples = len(dataset)

        return dataset, number_of_samples

    def SaveData(self, path):
        total = 0
        for label in self.LABELS:
            trainset, number_of_samples = self.ReadDatabase(label)

            for data in trainset:
                self.training_data.append([np.array(data),np.array([self.LABELS[label]])]) #save data and assign label

            total = total + number_of_samples
            print(label, number_of_samples)

        print("Total number of samples:", total)

        np.random.shuffle(self.training_data)
        np.save(path, self.training_data)
        print("Data saved in ",path)

    def LoadData(self, path, batch_size):
        print("Load data from ", path, "file")

        # Load data from .npy
        dataset = np.load(path, allow_pickle=True)

        # Convert from numpy to torch
        data = torch.Tensor([i[0] for i in dataset])
        label = torch.Tensor([i[1] for i in dataset])

        # Change dimensions to 4600x3x80x80 instead of 4600x80x3xd80
        epoch_size = int(data.shape[0]/batch_size)
        data = data.reshape(epoch_size, batch_size, opt.channels, 80, 80)
        label = label.reshape(epoch_size,batch_size)

        return data, label

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        gst = self.conv_blocks(out)
        return gst


neurons_num = 32
frame_parameters = 3
class_number = 12

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, gst):
        out = self.model(gst)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def plot3d(gesture_batch):
    # 3D plot of generator output
    fig = plt.figure(figsize=plt.figaspect(0.5))
    for z in range(6):
        # Copy to CPU and reshape to 80x3x80
        fake_output = gesture_batch[z]

        fake_output = fake_output.reshape(opt.img_size, opt.channels, opt.img_size)

        fake_output = fake_output.cpu().detach().numpy()

        ax = fig.add_subplot(2, 3, z+1, projection="3d")
        # matrix with number of frames
        frames = np.zeros((opt.img_size, opt.img_size))
        for i in range(opt.img_size):
            frames[:, i] = i

        x = fake_output[:, 0, :]
        y = fake_output[:, 1, :]

        for i in range(40):
            ax.scatter(frames[:, i], x[i, :], y[i, :])

        # set labels
        ax.set_xlabel("frames")
        ax.set_ylabel('x')
        ax.set_zlabel('y')

    plt.show()

def main():
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu")

    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Loss function
    criterion = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    print(generator)
    print(discriminator)

    if cuda:
        generator.cuda().to(device)
        discriminator.cuda().to(device)
        criterion.cuda().to(device)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Create dataset
    print("Create dataset")
    path = "GAN_dataset.npy"
    gestures = Gestures()
    if opt.load_dataset == 0:
        gestures.SaveData(path) # Save data from csv to path numpy file
    dataloader, labels = gestures.LoadData(path,opt.batch_size) # Load dataset from numpy file

    real_data = dataloader[0] # Save for ploting comparision
    save_image(real_data[:25], "real_gesture_samples.png", nrow=5, normalize=True) # Save real samples as image

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Information about CUDA usage
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    G_losses = []
    D_losses = []
    img_list = []

    # Training
    for epoch in range(opt.n_epochs):
        for i, gesture in enumerate(dataloader):

            # Label 1 for real gestures and 0 for fake
            valid = Variable(Tensor(gesture.shape[0], 1).fill_(1.0).to(device), requires_grad=False)
            fake = Variable(Tensor(gesture.shape[0], 1).fill_(0.0).to(device), requires_grad=False)

            # Configure input
            real_sample = Variable(gesture.type(Tensor)).to(device)

            #Discriminator training
            discriminator.zero_grad()
            output = discriminator(real_sample)

            #Calculate discriminator loss for real samples
            D_real_loss = criterion(output, valid)
            D_real_loss.backward(retain_graph=True)

            D_x = output.mean().item()

            # Generator training
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (gesture.shape[0], opt.latent_dim))).to(device))

            # Generate a batch of images
            gen_sample = generator(z)

            # Loss measures generator's ability to fool the discriminator
            D_fake_loss = criterion(discriminator(gen_sample), fake)

            D_fake_loss.backward(retain_graph=True)
            D_loss = D_real_loss + D_fake_loss
            optimizer_D.step()

            #Update
            generator.zero_grad()
            G_loss = criterion(discriminator(gen_sample), valid)
            G_loss.backward(retain_graph=True)
            optimizer_G.step()


            # Print loss
            if i % 100 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D(x): %.4f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), D_loss.item(), D_fake_loss.item(), D_x))
            G_losses.append(D_fake_loss.item())
            D_losses.append(D_loss.item())

            batches_done = epoch * len(dataloader) + i
            # Save images
            if batches_done % opt.sample_interval == 0:
                save_image(gen_sample.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                img_list.append(vutils.make_grid(gen_sample.data[:64].detach().cpu(), padding=2, normalize=True))

    # Comparision between real and fake samples in 2D shape
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real gestures")
    plt.imshow(
        np.transpose(vutils.make_grid(real_data[:64], normalize=True), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake gestures")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()

    #Plot progress
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot progress in time as changing output
    fig = plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.title("Progress")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())

    # Save model with weights and shape
    pytorch_model_path = 'DCGAN_model.pth'
    print("Pytorch model saved as: ",pytorch_model_path)
    torch.save(Generator, pytorch_model_path)

    # Print output arrays
    z = Variable(Tensor(np.random.normal(0, 1, (gesture.shape[0], opt.latent_dim))).to(device))

    # Generate a batch of images
    gen_sample = generator(z)
    plot3d(gen_sample)

    # Comparision of one real and fake sample in 3D form

    # Take one sample from batch and convert to numpy
    fake_output = gen_sample[10]
    fake_output = fake_output.reshape(opt.img_size, opt.channels, opt.img_size)
    fake_output = fake_output.cpu().detach().numpy()

    # List with number of frames
    frames = np.zeros(opt.img_size)
    for i in range(opt.img_size):
        frames[i] = i

    # 3D plot of generator output
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # Fake sample
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    x = fake_output[:, 0, :]
    y = fake_output[:, 1, :]


    for i in range(40):
        ax.scatter(frames[i], x[i, :], y[i, :])

    # set labels and title
    ax.set_xlabel("frames")
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    plt.title("Fake sample")

    #Real sample

    # Copy to CPU and reshape to 80x3x80
    real_output = real_data[12]
    real_output  = real_output.reshape(opt.img_size,opt.channels, opt.img_size)
    real_output  = real_output.cpu().detach().numpy()

    ax = fig.add_subplot(1, 2, 2, projection="3d")

    x = real_output[:, 0, :]
    y = real_output[:, 1, :]
    vel = real_output[:, 2, :]

    for i in range(40):
        ax.scatter(frames[i], x[i, :], y[i, :])

    # set labels and title
    ax.set_xlabel("frames")
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    plt.title("Real sample")

    plt.show()

    # Generate fake samples
    fake_samples = []
    for i in range(10):
        z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))).to(device))
        gen_sample = generator(z)
        fake_samples.append(gen_sample)

if __name__ == "__main__":
    main()