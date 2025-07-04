# Random transform of reference frames

data = [
    dict(
        data_root="/data/prakash_lipsync",
        train_json="/data/prakash_lipsync/v2/hdtf/train_hallo.json",
        val_json="/data/prakash_lipsync/v2/hdtf/val_hallo.json",
        ds_name="hdtf",
        video_folder="video_hallo_512",
        audio_features_name="fe_wav2vec12",
        silences=True,
        reference_frames_transforms=False,
    ),
    dict(
        data_root="/data/prakash_lipsync",
        train_json="/data/prakash_lipsync/v2/iv_recording/train_hallo.json",
        val_json="/data/prakash_lipsync/v2/iv_recording/val_hallo.json",
        ds_name="iv_recording",
        video_folder="video_hallo_512",
        audio_features_name="fe_wav2vec12",
        reference_frames_transforms=False,
        silences=True,
    ),
    dict(
        data_root="/data/prakash_lipsync",
        train_json="/data/prakash_lipsync/v2/iv_recording_v2/train_hallo.json",
        val_json="/data/prakash_lipsync/v2/iv_recording_v2/val_hallo.json",
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
upscale = 2
reference_frames_process = "channel"
seg_face = True
identity_loss = False

# ref frames "random", "bin_random"
mouth_region_size = [64, 128, 256, 256][stage]

# train
accelarator = "gpu"
devices = 8
nodes = 4
batch_size = 4
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
lr_dV = 0.00001
lr_lip_dV = 0.00001

non_decay = 100
decay = 100
pretrained_syncnet_path = "weights/v1/e1_256_e148-s16539-v194-t176-l1_iv.ckpt"
pretrained_frame_DINet_path = "weights/v9/epoch=52-step=150732-val_ep_loss=1.226-train_ep_loss=1.244-val_ep_sync_loss=0.109-train_ep_sync_loss=0.161.ckpt"
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
name = "exp1_v2_data_v11_repeat_4_rt_dv"
project = "lipsync_v2_l1_iv_512"
save_dir = "lightning_logs/"
no_wandb = False


# wandb_clip
clip_name = name + "_clip"
clip_project = project + "_clip"
