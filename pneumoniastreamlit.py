from torchvision import transforms
import streamlit as st
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)), preds
# Define your PyTorch model class
    # Your model implementation...
class PneumoniaModelBase(nn.Module):

    # this is for loading the batch of train image and outputting its loss, accuracy
    # & predictions
    def training_step(self, batch, weight):
        images,labels = batch
        out = self(images)                                      # generate predictions
        loss = F.cross_entropy(out, labels, weight=weight)      # weighted compute loss
        acc,preds = accuracy(out, labels)                       # calculate accuracy

        return {'train_loss': loss, 'train_acc':acc}

    # this is for computing the train average loss and acc for each epoch
    def train_epoch_end(self, outputs):
        batch_losses = [x['train_loss'] for x in outputs]       # get all the batches loss
        epoch_loss = torch.stack(batch_losses).mean()           # combine losses
        batch_accs = [x['train_acc'] for x in outputs]          # get all the batches acc
        epoch_acc = torch.stack(batch_accs).mean()              # combine accuracies

        return {'train_loss': epoch_loss.item(), 'train_acc': epoch_acc.item()}

    # this is for loading the batch of val/test image and outputting its loss, accuracy,
    # predictions & labels
    def validation_step(self, batch):
        images,labels = batch
        out = self(images)                                      # generate predictions
        loss = F.cross_entropy(out, labels)                     # compute loss
        acc,preds = accuracy(out, labels)                       # calculate acc & get preds

        return {'val_loss': loss.detach(), 'val_acc':acc.detach(),
                'preds':preds.detach(), 'labels':labels.detach()}
    # detach extracts only the needed number, or other numbers will crowd memory

    # this is for computing the validation average loss and acc for each epoch
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]         # get all the batches loss
        epoch_loss = torch.stack(batch_losses).mean()           # combine losses
        batch_accs = [x['val_acc'] for x in outputs]            # get all the batches acc
        epoch_acc = torch.stack(batch_accs).mean()              # combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    # this is for printing out the results after each epoch
    def epoch_end(self, epoch, train_result, val_result):
        print('Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.
              format(epoch+1, train_result['train_loss'], train_result['train_acc'],
                     val_result['val_loss'], val_result['val_acc']))

    # this is for using on the test set, it outputs the average loss and acc,
    # and outputs the predictions
    def test_prediction(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()           # combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()              # combine accuracies
        # combine predictions
        batch_preds = [pred for x in outputs for pred in x['preds'].tolist()]
        # combine labels
        batch_labels = [lab for x in outputs for lab in x['labels'].tolist()]

        return {'test_loss': epoch_loss.item(), 'test_acc': epoch_acc.item(),
                'test_preds': batch_preds, 'test_labels': batch_labels}
    
class PneumoniaResnet(PneumoniaModelBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Freeze training for all layers before classifier
        for param in self.network.fc.parameters():
            param.require_grad = False
        num_features = self.network.fc.in_features # get number of in features of last layer
        self.network.fc = nn.Linear(num_features, 2) # replace model classifier

    def forward(self, xb):
        return self.network(xb)

# Inisialisasi model dan memindahkan ke 'cpu'
#model = FariqPneumoniaResnet.pth # model_path
#device = torch.device('cpu')  # Set device to CPU explicitly
#model.to(device)
    
model_path = 'PneumoniaResnet.pth'  # Path file model yang ingin dimuat
device = torch.device('cpu')  # Set device to CPU secara eksplisit

# Inisialisasi objek model dari kelas FariqPneumoniaResnet
model = PneumoniaResnet()
model = torch.load(model_path, map_location=torch.device('cpu')) # Muat model dari file .pth
model.to(device)  # Pindahkan model ke perangkat CPU

# UI Streamlit
st.title("Prediksi Penyakit Pneumonia Melalui Citra Rontgen Thorax")
uploaded_file = st.file_uploader("Upload an X-ray Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Upload Image', use_column_width=True)

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    Image = transform(image).unsqueeze(0) # Add batch dimension

    if model is not None:
        # Perform prediction
        with torch.no_grad():
            prediction = model(Image)
        
        # Process preiction result
        # (for example, applying softmax, and obtaining predicted class probabilities)
        probabilities = F.softmax(prediction[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()

        # Membuat label prediksi khusus
        class_labels = ["Normal", "Pneumonia"]
        predicted_label = class_labels[predicted_class]

        # Display prediction results
        st.write(f"Predicted Class: {predicted_label}")
        st.write(f"Predicted Probability: {probabilities[predicted_class].item():.4f}")