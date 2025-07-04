validate_only = False
data = [
    # hdtf
    # dict(
    #     data_root="/data/.pipeliner_cache/",
    #     train_json="/data/.pipeliner_cache/jsons/hdtf/train_silence_filtered.json",
    #     val_json="/data/.pipeliner_cache/jsons/hdtf/val_silence_filtered.json",
    #     ds_name="hdtf",
    #     video_folder="video",
    #     audio_features_name="fe_wav2vec12",
    #     silences=True,
    #     reference_frames_transforms=False,
    # ),
    # iv_recording
    dict(
        data_root="/data/wider_crop_data_512/",
        train_json="/data/wider_crop_data_512/jsons/iv_recording/train_silence_filtered_new.json",
        val_json="/data/wider_crop_data_512/jsons/iv_recording/val_silence_filtered_new.json",
        ds_name="iv_recording",
        video_folder="video",
        audio_features_name="fe_wav2vec12",
        silences=True,
        reference_frames_transforms=False,
        landmarks_folder="/data/.pipeliner_cache/mp_lms",
        silence_encoding="data/silence.safetensors",
    ),
    # iv_recording_v2
    dict(
        data_root="/data/wider_crop_data_512/",
        train_json="/data/wider_crop_data_512/jsons/iv_recording_v2/train_silence_filtered_new.json",
        val_json="/data/wider_crop_data_512/jsons/iv_recording_v2/val_silence_filtered_new.json",
        ds_name="iv_recording_v2",
        video_folder="video",
        audio_features_name="fe_wav2vec12",
        silences=True,
        reference_frames_transforms=False,
        landmarks_folder="/data/.pipeliner_cache/mp_lms",
        silence_encoding="data/silence.safetensors",
    ),
    # ai_avatars
    # dict(
    #     data_root="/data/.pipeliner_cache/",
    #     train_json="/data/.pipeliner_cache/jsons/ai_avatars/train_silence_filtered.json",
    #     val_json="/data/.pipeliner_cache/jsons/ai_avatars/val_silence_filtered.json",
    #     ds_name="ai_avatars",
    #     video_folder="video",
    #     audio_features_name="fe_wav2vec12",
    #     silences=True,
    #     reference_frames_transforms=False,
    # ),
    # # ai_avatars_v2
    # dict(
    #     data_root="/data/.pipeliner_cache/",
    #     train_json="/data/.pipeliner_cache/jsons/ai_avatars_v2/train_silence_filtered.json",
    #     val_json="/data/.pipeliner_cache/jsons/ai_avatars_v2/val_silence_filtered.json",
    #     ds_name="ai_avatars_v2",
    #     video_folder="video",
    #     audio_features_name="fe_wav2vec12",
    #     silences=True,
    #     reference_frames_transforms=False,
    # ),
]
if validate_only:
    data = data[:1]


stage = 2
augment_num = [32, 4, 4, 100][stage]

# Model
source_channel = 3
image_length = 5
audio_seq_len = 5  # DINet audio length
ref_channel = 15
upscale = 1

# ref frames "random", "bin_random"
mouth_region_size = [64, 128, 256, 256][stage]
image_size = 512
# train
accelarator = "gpu"
devices = [0] if validate_only else [0, 1, 2, 3, 4, 5, 6, 7]
nodes = 1
batch_size = 2
num_workers = 32
reference_frames_process = "channel"
seg_face = True
identity_loss = False
use_attention = False
check_val_every_n_epoch = 1


# losses
clip_gradients = True
gen_clip_value = 0.1
disc_clip_value = 0.05
lambda_perception = 2
lambda_perception_lip = 10
lambda_syncnet_perception = 0.5


# optimizers
scheduler = "lambda"
lr_g = 1e-5
lr_dI = 5e-7  # reduced by a factor of 10 not to overpower the discrimintor
lr_lip_dI = 5e-7

non_decay = 300
decay = 300
pretrained_syncnet_path = "/data/anup/e1_256_e148-s16539-v194-t176-l1_iv.ckpt"
pretrained_frame_DINet_path = "/data/anup/256-1st-frame-focus.ckpt"
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
name = ""
project = "finetune_512-noisy-first-frame"
save_dir = "/scratch/anup/lightning_logs/"
no_wandb = False

# wandb_clip
clip_name = name + "_clip" if len(name) > 0 else None
clip_project = project + "_clip"
