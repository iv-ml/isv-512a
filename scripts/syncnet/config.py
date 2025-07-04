# lightning_logs/version_2
data = [
    dict(
        data_root="/data/prakash_lipsync/",
        ds_name="hdtf",
        audio_features_name="fe_wav2vec12",
        train_json="/data/prakash_lipsync/v1/hdtf/train_hallo.json",
        val_json="/data/prakash_lipsync/v1/hdtf/val_hallo.json",
    ),
    dict(
        data_root="/data/prakash_lipsync/",
        ds_name="iv_recording",
        audio_features_name="fe_wav2vec12",
        train_json="/data/prakash_lipsync/v1/iv_recording/train_hallo.json",
        val_json="/data/prakash_lipsync/v1/iv_recording/val_hallo.json",
    ),
    dict(
        data_root="/data/prakash_lipsync/",
        ds_name="iv_recording_v2",
        audio_features_name="fe_wav2vec12",
        train_json="/data/prakash_lipsync/v1/iv_recording_v2/train_hallo.json",
        val_json="/data/prakash_lipsync/v1/iv_recording_v2/val_hallo.json",
    ),
    dict(
        data_root="/data/prakash_lipsync/",
        ds_name="th_1kh_512",
        audio_features_name="fe_wav2vec12",
        train_json="/data/prakash_lipsync/v1/th_1kh_512/train_clips_hallo_filtered.json",
        val_json=None,
    ),
]

bad_clips = "scripts/syncnet/bad_clips.txt"
stage = 1
mouth_region_size = [64, 128, 256]
augment_num = [10, 10, 10]

loss = "cosine_loss"
normalize_embeddings = True  # relu activated and L2 norm applied
wav = 12
img_length = 5
audio_length = 9
out_dim = 512
backbone_name = "resnet34"
loss_type = "bce"

# train
accelarator = "gpu"
devices = 8
batch_size = 256
num_workers = 4
scheduler = "cosine"  # only consine or None works for now

# optimizers
lr = 1e-4  # * devices * (batch_size / 32)
epochs = 600
no_wandb = False

##wandb
name = "exp3_i5_a9_res34"
project = "syncnet" + str(mouth_region_size[stage]) + "_non_silence_1"
save_dir = "lightning_logs/"
