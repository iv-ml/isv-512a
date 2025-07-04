data = [
    dict(
        data_root="/data/prakash_lipsync",
        train_json="/data/prakash_lipsync/v1/hdtf/train_hallo.json",
        val_json="/data/prakash_lipsync/v1/hdtf/val_hallo.json",
        ds_name="hdtf",
        video_folder="video_hallo_512",
        audio_features_name="fe_wav2vec12",
        silences=True,
    ),
    dict(
        data_root="/data/prakash_lipsync",
        train_json="/data/prakash_lipsync/v1/iv_recording/train_hallo.json",
        val_json="/data/prakash_lipsync/v1/iv_recording/val_hallo.json",
        ds_name="iv_recording",
        video_folder="video_hallo_512",
        audio_features_name="fe_wav2vec12",
        silences=True,
    ),
    dict(
        data_root="/data/prakash_lipsync",
        train_json="/data/prakash_lipsync/v1/iv_recording_v2/train_hallo.json",
        val_json="/data/prakash_lipsync/v1/iv_recording_v2/val_hallo.json",
        ds_name="iv_recording_v2",
        video_folder="video_hallo_512",
        audio_features_name="fe_wav2vec12",
        silences=True,
    ),
]


stage = 1
augment_num = [32, 8, 4, 100][stage]

# Model
source_channel = 3
image_length = 5
audio_seq_len = 5  # DINet audio length
ref_channel = 15
upscale = 1
reference_frames_process = "channel"
seg_face = True

# Identity loss
identity_loss = False
identity_loss_path = "weights/arcface.pt"

# ref frames "random", "bin_random"
mouth_region_size = [64, 128, 256, 256][stage]

# train
accelarator = "gpu"
devices = 8
nodes = 1
batch_size = 8
num_workers = 32
check_val_every_n_epoch = 1


# losses
clip_gradients = True
gen_clip_value = 0.1
disc_clip_value = 0.05
lambda_perception = 2
lambda_syncnet_perception = 0.5


# optimizers
scheduler = "lambda"
lr_g = 0.0001
lr_dI = 0.00001  # reduced by a factor of 10 not to overpower the discrimintor
lr_lip_dI = 0.00001

non_decay = 300
decay = 300
pretrained_syncnet_path = (
    "weights/v1/e1_256_e148-s16539-v194-t176-l1_iv.ckpt"  # "weights/v1/e1_306_e34077-v184-t149-l1_iv.ckpt"
)
pretrained_frame_DINet_path = None  # "weights/v1/epoch=38-step=83187-val_ep_loss=0.951-train_ep_loss=0.982-val_ep_sync_loss=0.187-train_ep_sync_loss=0.200.ckpt"
resume_from_checkpoint = None

# Discriminator
D_num_blocks = 6
D_block_expansion = 128
D_max_features = 512

# LIP Discrimintor
LipD_num_blocks = 6
LipD_block_expansion = 128
LipD_max_features = 512

##wandb
name = "exp1"
project = "lipsync_v2_l1_iv"
save_dir = "lightning_logs/"
on_wandb = False

# wandb_clip
clip_name = name + "_clip"
clip_project = project + "_clip"
