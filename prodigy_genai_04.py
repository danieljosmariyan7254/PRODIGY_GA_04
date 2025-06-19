import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

class FacadesDataset(Dataset):
    def __init__(self, root, mode="train", transform=None):
        self.transform = transform
        self.image_dir = os.path.join(root, mode)
        self.image_filenames = sorted(os.listdir(self.image_dir))  
        
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        
        image = Image.open(image_path).convert("RGB")  
        width, height = image.size

        input_image = image.crop((0, 0, width // 2, height))  
        target_image = image.crop((width // 2, 0, width, height)) 

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = FacadesDataset(root="data/facades", mode="train", transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


for i, (input_image, target_image) in enumerate(dataloader):
    print(f"Batch {i+1} - Input Shape: {input_image.shape}, Target Shape: {target_image.shape}")
    break  


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.model(torch.cat((x, y), dim=1))


device = torch.device("cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)

adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss()
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

for epoch in range(5):
    for i, (input_image, target_image) in enumerate(dataloader):
        input_image, target_image = input_image.to(device), target_image.to(device)

    
        gen_optimizer.zero_grad()
        fake_image = generator(input_image)

        fake_pred = discriminator(input_image, fake_image)
        gen_loss = adversarial_loss(fake_pred, torch.ones_like(fake_pred)) + l1_loss(fake_image, target_image)
        gen_loss.backward()
        gen_optimizer.step()

        disc_optimizer.zero_grad()
        real_pred = discriminator(input_image, target_image)
        fake_pred = discriminator(input_image, fake_image.detach())
        disc_loss = adversarial_loss(real_pred, torch.ones_like(real_pred)) + adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
        disc_loss.backward()
        disc_optimizer.step()

        print(f"Epoch {epoch+1}, Step {i+1}, Gen Loss: {gen_loss.item():.4f}, Disc Loss: {disc_loss.item():.4f}")

import matplotlib.pyplot as plt

def generate_image(input_image):
    generator.eval()  
    with torch.no_grad():
        fake_image = generator(input_image)
    return fake_image.squeeze().permute(1, 2, 0)  

sample_image, _ = dataset[0]
generated_image = generate_image(sample_image.unsqueeze(0))

plt.subplot(1, 2, 1)
plt.imshow(sample_image.permute(1, 2, 0))
plt.title("Input Image")

plt.subplot(1, 2, 2)
plt.imshow(generated_image)
plt.title("Generated Image")

plt.show() 