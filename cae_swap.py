import torch
import torchvision.utils

from cae.trainer import trainer as cae_trainer
from torchvision import transforms
from PIL import Image
from pathlib import Path

# Transform a abnormal image to normal

device = 'cuda:0'


def denormalize(tensor):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(device)
    return tensor * std + mean


gen_model_path = Path('path_for_your_cae_gen')

model_name = 'gen_00100000.pt'
input_gen_model_path = gen_model_path / model_name

trainer = cae_trainer(device=device)
state_dict_gen = torch.load(input_gen_model_path, map_location='cpu')
trainer.gen.load_state_dict(state_dict_gen['ab'])
trainer.to(device)
trainer.eval()
encode = trainer.gen.encode
decode = trainer.gen.decode

transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_list = [transforms.Resize((256, 256))] + transform_list
transform = transforms.Compose(transform_list)

image_1 = Path('/DATASET/OCT_2017/test/DME-8426419-2.png')  # Path for image A
image_2 = Path('/DATASET/OCT_2017/test/NORMAL-2131138-1.png')  # Path for image B

with torch.no_grad():
    image_tensor_1 = transform(Image.open(image_1).convert('RGB')).unsqueeze(0).to(device)
    image_tensor_2 = transform(Image.open(image_2).convert('RGB')).unsqueeze(0).to(device)

    p_code_1, c_code_1 = encode(image_tensor_1)
    p_code_2, c_code_2 = encode(image_tensor_2)

    swap_image = decode(p_code_1, c_code_2)

    swap_image = denormalize(swap_image)
    image_tensor_1 = denormalize(image_tensor_1)

    torchvision.utils.save_image(swap_image, f'normal_{image_1.name}')
