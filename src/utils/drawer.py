import io
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor

def imshow(src):
    """Create a pyplot plot and save to buffer."""
    img = src.cpu().detach().numpy()
    plt.figure()
    plt.imshow(img, cmap='gray')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    return image

