from PIL import Image
from torchvision import transforms

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

def preprocess_image(image: Image.Image):
    """
    Converts PIL image inot model-ready tensor
    """

    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = transformer(image)

    return image.unsqueeze(0)

