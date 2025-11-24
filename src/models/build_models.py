#### Include the model architecture

from torch import nn
import torch


class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """

    # 2. Initialize the class with appropriate variables
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embedding_dim: int = 768):
        super().__init__()

        self.patch_size = patch_size

        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2,  # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    # 5. Define the forward method
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, (f"Input image size must be divisible by patch size,"
                                                         f" image shape: {image_resolution},"
                                                         f" patch size: {self.patch_size}")

        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        # 6. Make sure the output shape has the right order
        return x_flattened.permute(0, 2, 1)


class core_module(nn.Module):
    def __init__(self,
                 model_parameters):  
        super().__init__()

        # Assert image size is divisible by patch size
        assert model_parameters['IMAGE_SIZE'] % model_parameters['PATCH_SIZE'] == 0, "Image size must be divisible by patch size."

        # 1. Create patch embedding
        self.patch_embedding = PatchEmbedding(in_channels=model_parameters['SEQUENCE_LEN'],
                                              patch_size=model_parameters['PATCH_SIZE'],
                                              embedding_dim=model_parameters['EMBEDDING_DIM'])

        # 2. Create positional embedding
        num_patches = (model_parameters['IMAGE_SIZE'] * model_parameters['IMAGE_SIZE']  ) // model_parameters['PATCH_SIZE'] ** 2  # N = HW/P^2
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, model_parameters['EMBEDDING_DIM']))  # +1 for the class

        # 3. Create patch + position embedding dropout
        self.embedding_dropout = nn.Dropout(p=model_parameters['DROPOUT'])

        # 4. Create stack Transformer Encoder layers (stacked single layers)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=model_parameters['EMBEDDING_DIM'],
                                                     nhead=model_parameters['NUM_HEADS'],
                                                     dim_feedforward=model_parameters['MLP_SIZE'],
                                                     activation="gelu",
                                                     batch_first=True,
                                                     norm_first=True),
            num_layers=model_parameters['NUM_TRANSFORMER_LAYERES'],
            enable_nested_tensor=False
            )  # Stack it N times

        # 5. Align the parameter matrix according to the embedding dimension
        self.linear_layer = nn.Linear(in_features=model_parameters['SEQUENCE_LEN'] * 2,
                                      out_features=model_parameters['EMBEDDING_DIM'])

        # 6. Create the convolutional layers after the transformers - add batch normalization

        self.conv_layers = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Up-sample to target spatial dimensions (126, 126)
            nn.Upsample(size=(model_parameters['IMAGE_SIZE'], model_parameters['IMAGE_SIZE']), mode='bilinear', align_corners=False),

            # Convolutional layer to get to 4 channels
            nn.Conv2d(in_channels=64, out_channels=model_parameters['SEQUENCE_LEN'], kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # 10
        )

    def forward(self, x, y):
        x = x.permute(0, 3, 1, 2)

        # Create the patch embedding
        x = self.patch_embedding(x)

        # Add the positional embedding to patch embedding with class token
        x = self.positional_embedding + x

        # Dropout on patch + positional embedding
        x = self.embedding_dropout(x)

        # Creating the parameters embedding and concatenating to the data embedding
        y = self.linear_layer(y)
        x = torch.cat((x, y), dim=1)

        # Pass embedding through Transformer Encoder stack
        x = self.transformer_encoder(x).unsqueeze(dim=1)
        x = self.conv_layers(x)

        return x.permute(0, 2, 3, 1)  # (n, 144, 144, k)


def quantification_module(model_parameters):
    model = core_module(model_parameters)

    model.conv_layers[9] = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
    model.conv_layers[10] = nn.BatchNorm2d(num_features=32)
    model.conv_layers.append(nn.ReLU())

    model.conv_layers.append(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1))
    model.conv_layers.append(nn.BatchNorm2d(num_features=16))
    model.conv_layers.append(nn.ReLU())

    model.conv_layers.append(nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1))
    model.conv_layers.append(nn.BatchNorm2d(num_features=8))
    model.conv_layers.append(nn.ReLU())

    model.conv_layers.append(nn.Conv2d(in_channels=8, out_channels=6, kernel_size=3, stride=1, padding=1))
    model.conv_layers.append(nn.Sigmoid())

    return model
