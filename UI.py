import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Define the CNN model class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # Use log_softmax for numerical stability

# Load the model and move it to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
print(device)
model_path = '/mnt/d/projects/MNISTdigits/models/mnist_cnn_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set the model to evaluation mode

def predict_digit(image_dict):
    if isinstance(image_dict, dict) and 'composite' in image_dict:
        image = image_dict["composite"]
    else:
        raise KeyError("The provided input does not contain the 'composite' key.")

    # Resize the image to 28x28 pixels using the LANCZOS filter
    image = image.resize((28, 28), Image.LANCZOS)

    # Convert the image to a numpy array and normalize
    image = np.array(image).astype(np.float32) / 255.0

    # Ensure the image is reshaped correctly: [28, 28] to [1, 1, 28, 28]
    image = image.reshape(1, 1, 28, 28)

    # Convert the image to a tensor and move it to the appropriate device
    tensor_image = torch.from_numpy(image).float().to(device)

    # Print the preprocessed input image tensor
    print("Preprocessed image tensor:", tensor_image)

    with torch.no_grad():
        output = model(tensor_image)

        # Print the raw output logits
        print("Model output logits:", output)

        #prediction = output.argmax(dim=1, keepdim=True).item()
        prediction = {str(i): float(output[0][i]) for i in range(10)}

    return prediction

# Define the Gradio interface
iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Paint(image_mode="L", type = "pil"),
    outputs=gr.Label(num_top_classes = 10),
    live=True,
    title="MNIST Digit Classifier",
    description="Draw a digit and see the prediction"
)

if __name__ == "__main__":
    iface.launch()
