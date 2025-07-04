import torch
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)
from torchvision.models.vision_transformer import vit_b_16


class ImageProjModel(torch.nn.Module):
    """Audio Projection Model

    This class defines an audio projection model that takes audio embeddings as input
    and produces context tokens as output. The model is based on the ModelMixin class
    and consists of multiple linear layers and activation functions. It can be used
    for various audio processing tasks.

    Attributes:
        seq_len (int): The length of the audio sequence.
        blocks (int): The number of blocks in the audio projection model.
        channels (int): The number of channels in the audio projection model.
        intermediate_dim (int): The intermediate dimension of the model.
        context_tokens (int): The number of context tokens in the output.
        output_dim (int): The output dimension of the context tokens.

    Methods:
        __init__(self, seq_len=5, blocks=12, channels=768, intermediate_dim=512, context_tokens=32, output_dim=768):
            Initializes the AudioProjModel with the given parameters.
        forward(self, audio_embeds):
            Defines the forward pass for the AudioProjModel.
            Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, blocks, channels).
            Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, video_length, context_tokens, output_dim).

    """

    def __init__(
        self,
        seq_len=5,
        channels=768,
        intermediate_dim=512,
        output_dim=768,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.channels = channels
        self.input_dim = seq_len * channels  # update input_dim to be the product of blocks and channels.
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = torch.nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = torch.nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = torch.nn.Linear(intermediate_dim, output_dim)

        self.norm = torch.nn.LayerNorm(output_dim)

    def forward(self, img_embeds):
        """
        Defines the forward pass for the AudioProjModel.

        Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, channels).

        Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, output_dim).
        """
        batch_size, window_size, channels = img_embeds.shape
        img_embeds = img_embeds.reshape(batch_size, window_size * channels)

        img_embeds = torch.relu(self.proj1(img_embeds))
        img_embeds = torch.relu(self.proj2(img_embeds))

        tokens = self.proj3(img_embeds).reshape(batch_size, self.output_dim)

        tokens = self.norm(tokens)
        return tokens


class ImageEncoder(torch.nn.Module):
    def __init__(self, batch=5, output_dim=512, name=None, image_size=None):
        super(ImageEncoder, self).__init__()

        # Load a pre-trained ResNet34 model
        self.batch = batch
        self.output_dim = output_dim
        if name == "resnet34":
            self.pretrained_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            input_channels = 512
        elif name == "resnet18":
            self.pretrained_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            input_channels = 512
        elif name == "resnet50":
            self.pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            input_channels = 2048
        elif name == "vit_b_16":
            self.pretrained_model = vit_b_16(image_size=image_size)  # hardcoded image size for now
            self.pretrained_model.heads = torch.nn.Identity()
            input_channels = 768
        else:
            raise NotImplementedError(f"{name} not implemented . Try resnet18, 34 or 50")

        # # Remove the final fully connected layer
        self.pretrained_model.fc = torch.nn.Identity()
        self.projection = ImageProjModel(batch, input_channels, 512, output_dim=output_dim)

    def forward(self, x):
        """
        we will get the image input as (batch_size, N_images, N_channels, h, w)
        """
        batch_size = x.shape[0]
        assert self.batch == x.shape[1], "Number of images is not same"
        x = torch.cat(torch.split(x, 1, dim=1), 0).squeeze(1)
        x = self.pretrained_model(x)
        # we basically get 20, 512 # we will make it batch_size, 5, 512
        # then use a projection layer
        x = torch.flatten(x, 1)
        x = torch.stack(torch.split(x, batch_size, dim=0)).permute((1, 0, 2))
        projection = self.projection(x)
        return projection


class AudioProjModel(torch.nn.Module):
    """Audio Projection Model

    This class defines an audio projection model that takes audio embeddings as input
    and produces context tokens as output. The model is based on the ModelMixin class
    and consists of multiple linear layers and activation functions. It can be used
    for various audio processing tasks.

    Attributes:
        seq_len (int): The length of the audio sequence.
        blocks (int): The number of blocks in the audio projection model.
        channels (int): The number of channels in the audio projection model.
        intermediate_dim (int): The intermediate dimension of the model.
        context_tokens (int): The number of context tokens in the output.
        output_dim (int): The output dimension of the context tokens.

    Methods:
        __init__(self, seq_len=5, blocks=12, channels=768, intermediate_dim=512, context_tokens=32, output_dim=768):
            Initializes the AudioProjModel with the given parameters.
        forward(self, audio_embeds):
            Defines the forward pass for the AudioProjModel.
            Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, blocks, channels).
            Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, video_length, context_tokens, output_dim).

    """

    def __init__(
        self,
        seq_len=5,
        context=1,
        blocks=12,  # add a new parameter blocks
        channels=768,  # add a new parameter channels
        intermediate_dim=512,
        output_dim=768,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.context = context
        self.input_dim = (
            context * seq_len * blocks * channels
        )  # update input_dim to be the product of blocks and channels.
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = torch.nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = torch.nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = torch.nn.Linear(intermediate_dim, output_dim)

        self.norm = torch.nn.LayerNorm(output_dim)

    def forward(self, audio_embeds):
        """
        Defines the forward pass for the AudioProjModel.

        Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, blocks, channels).

        Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, video_length, context_tokens, output_dim).
        """
        if self.context == 1:
            batch_size, window_size, blocks, channels = audio_embeds.shape
            audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)
        else:
            batch_size, context, window_size, blocks, channels = audio_embeds.shape
            audio_embeds = audio_embeds.view(batch_size, context * window_size * blocks * channels)
        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(batch_size, self.output_dim)

        context_tokens = self.norm(context_tokens)
        #         context_tokens = rearrange(
        #             context_tokens, "(bz f) m c -> bz f m c", f=video_length
        #         )

        return context_tokens


class SyncNetMulti(torch.nn.Module):
    def __init__(
        self,
        img_length=5,
        audio_length=9,
        audio_context=1,
        out_dim=512,
        backbone_name=None,
        image_size=None,
    ):
        super(SyncNetMulti, self).__init__()
        self.out_dim = out_dim
        self.face_encoder = ImageEncoder(
            batch=img_length,
            output_dim=self.out_dim,
            name=backbone_name,
            image_size=image_size,
        )
        self.audio_encoder = AudioProjModel(seq_len=audio_length, context=audio_context, output_dim=self.out_dim)

    def forward(self, img, audio):
        image_features = self.face_encoder(img)
        audio_features = self.audio_encoder(audio)
        return image_features, audio_features


if __name__ == "__main__":
    print("Loading audio model")
    model = AudioProjModel(5, context=5, output_dim=512).cuda()
    audio_embed = torch.randn((4, 5, 5, 12, 768)).cuda()
    x = model(audio_embed)
    print(x.shape)

    audio_params = 0
    for _, param in model.named_parameters():
        audio_params += param.numel()
    print(audio_params)

    model = ImageEncoder(5, 512, name="resnet34", image_size=128).cuda()
    params = 0
    for _, param in model.named_parameters():
        params += param.numel()
    print(params)
    out = model(torch.randn((4, 5, 3, 128, 128)).cuda())
    print(out.shape)

    print(audio_params + params)

    model = SyncNetMulti(audio_length=7, backbone_name="resnet34", image_size=128).cuda()
    img, audio = model(torch.randn((4, 5, 3, 128, 128)).cuda(), torch.randn((4, 7, 12, 768)).cuda())
    print(img.shape, audio.shape)