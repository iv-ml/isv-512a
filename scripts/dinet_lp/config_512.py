data = [
    dict(
        data_root="/data/prakash_lipsync",
        train_json="/data/prakash_lipsync/v1/hdtf/train_hallo_512_lm.json",
        val_json="/data/prakash_lipsync/v1/hdtf/val_hallo_512_lm.json",
        ds_name="hdtf",
        video_folder="video_hallo_512",
        video_folder_lp="video_hallo_512_lp",
        audio_features_name="fe_wav2vec12",
        silences=False,
    ),
    dict(
        data_root="/data/prakash_lipsync",
        train_json="/data/prakash_lipsync/v1/iv_recording/train_hallo_512_lm.json",
        val_json="/data/prakash_lipsync/v1/iv_recording/val_hallo_512_lm.json",
        ds_name="iv_recording",
        video_folder="video_hallo_512",
        video_folder_lp="video_hallo_512_lp",
        audio_features_name="fe_wav2vec12",
        silences=False,
    ),
    # dict(
    #     data_root="/data/prakash_lipsync",
    #     train_json="/data/prakash_lipsync/v1/iv_recording_v2/train_hallo_512_lm_lp.json",
    #     val_json="/data/prakash_lipsync/v1/iv_recording_v2/val_hallo_512_lm_lp.json",
    #     ds_name="iv_recording_v2",
    #     video_folder="video_hallo_512",
    #     video_folder_lp="video_hallo_512_lp",
    #     audio_features_name="fe_wav2vec12",
    #     silences=False,
    # ),
]


stage = 1
augment_num = [32, 4, 4, 100][stage]

# Model
source_channel = 3
image_length = 5
audio_seq_len = 5  # DINet audio length
ref_channel = 15
upscale = 2
seg_face = True
mouth_region_size = [64, 128, 256, 256][stage]

# train
accelarator = "gpu"
devices = 1
nodes = 1
batch_size = 4
num_workers = 4
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
pretrained_syncnet_path = "weights/v1/e1_256_e148-s16539-v194-t176-l1_iv.ckpt"
pretrained_frame_DINet_path = "lightning_logs/lipsync_v2_l1_iv_clip/lb0pzcbt/checkpoints/epoch=22-step=160563-val_ep_loss=1.525-train_ep_loss=1.864-val_ep_sync_loss=0.415-train_ep_sync_loss=0.406.ckpt"  # "weights/v4_512/epoch=125-step=477792-val_ep_loss=2.147-train_ep_loss=2.897-val_ep_sync_loss=0.164-train_ep_sync_loss=0.136.ckpt"
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
name = "exp2_md_v5_512"
project = "lipsync_v2_l1_iv"
save_dir = "lightning_logs/"
no_wandb = False


# wandb_clip
clip_name = name + "_clip"
clip_project = project + "_clip"
