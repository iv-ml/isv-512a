# Random transform of reference frames

data = [
    dict(
        data_root="/data/lipsync_768_data",
        train_json="/data/lipsync_768_data/v1/hdtf/train_hallo.json",
        val_json="/data/lipsync_768_data/v1/hdtf/val_hallo.json",
        ds_name="hdtf",
        video_folder="video_hallo_512",
        audio_features_name="fe_wav2vec12",
        silences=True,
        reference_frames_transforms=False,
    ),
    dict(
        data_root="/data/lipsync_768_data",
        train_json="/data/lipsync_768_data/v1/iv_recording/train_hallo.json",
        val_json="/data/lipsync_768_data/v1/iv_recording/val_hallo.json",
        ds_name="iv_recording",
        video_folder="video_hallo_512",
        audio_features_name="fe_wav2vec12",
        reference_frames_transforms=False,
        silences=True,
    ),
    dict(
        data_root="/data/lipsync_768_data",
        train_json="/data/lipsync_768_data/v1/iv_recording_v2/train_hallo.json",
        val_json="/data/lipsync_768_data/v1/iv_recording_v2/val_hallo.json",
        ds_name="iv_recording_v2",
        video_folder="video_hallo_512",
        audio_features_name="fe_wav2vec12",
        silences=True,
        reference_frames_transforms=False,
    ),
]


stage = 2
augment_num = [32, 8, 8, 100][stage]

# Model
source_channel = 3
image_length = 5
audio_seq_len = 5  # DINet audio length
ref_channel = 15
upscale = 3
reference_frames_process = "channel"
seg_face = True
identity_loss = False

# ref frames "random", "bin_random"
mouth_region_size = [64, 128, 256, 256][stage]

# train
accelarator = "gpu"
devices = 8
nodes = 2
batch_size = 2
num_workers = 32
check_val_every_n_epoch = 1


# losses
clip_gradients = True
gen_clip_value = 0.05
disc_clip_value = 0.025
lambda_perception = 2
lambda_syncnet_perception = 0.5


# optimizers
scheduler = "lambda"
lr_g = 0.00005
lr_dI = 0.000005  # reduced by a factor of 10 not to overpower the discrimintor
lr_lip_dI = 0.000005
lr_dV = 0.000005
lr_lip_dV = 0.000005

non_decay = 100
decay = 100
pretrained_syncnet_path = "weights/v1/e1_256_e148-s16539-v194-t176-l1_iv.ckpt"
pretrained_frame_DINet_path = "/disk/prakash_training_weights/weights/v11/epoch=30-step=132246-val_ep_loss=1.120-train_ep_loss=1.025-val_ep_sync_loss=0.106-train_ep_sync_loss=0.154.ckpt"
resume_from_checkpoint = None  # "/home/anup/iv/lipsync/lightning_logs/lipsync_v2_l1_iv_768_clip/59fl1odr/checkpoints/epoch=11-step=40500-val_ep_loss=1.383-train_ep_loss=1.495-val_ep_sync_loss=0.152-train_ep_sync_loss=0.211.ckpt"

# Discriminator
D_num_blocks = 6
D_block_expansion = 128
D_max_features = 512

# LIP Discrimintor
LipD_num_blocks = 6
LipD_block_expansion = 128
LipD_max_features = 512

##wandb
name = "exp1_v2_data_v11_repeat_4_rt_dv"
project = "lipsync_v2_l1_iv_768"
save_dir = "lightning_logs/"
no_wandb = False


# wandb_clip
clip_name = None
clip_project = project + "_clip"
