## Adding new datasets "vfhq", "th_1kh_512", "curated", "mead"


data = [
    dict(
        data_root = "/data/prakash_lipsync/", 
        ds_name = "hdtf", 
        audio_features_name = "fe_wav2vec12", 
        train_json = "/data/prakash_lipsync/v1/hdtf/train.json", 
        val_json = "/data/prakash_lipsync/v1/hdtf/val.json", 
    ), 
    dict(
        data_root = "/data/prakash_lipsync/", 
        ds_name = "iv_recording", 
        audio_features_name = "fe_wav2vec12", 
        train_json = "/data/prakash_lipsync/v1/iv_recording/train.json", 
        val_json = "/data/prakash_lipsync/v1/iv_recording/val.json", 
    ),
    dict(
        data_root = "/data/prakash_lipsync/", 
        ds_name = "iv_recording_v2", 
        audio_features_name = "fe_wav2vec12", 
        train_json = "/data/prakash_lipsync/v1/iv_recording_v2/train.json", 
        val_json = "/data/prakash_lipsync/v1/iv_recording_v2/val.json", 
    ), 
    dict(
        data_root = "/data/prakash_lipsync/", 
        ds_name = "vfhq", 
        audio_features_name = "fe_wav2vec12", 
        train_json = "/data/prakash_lipsync/v1/vfhq/train.json", 
        val_json = "/data/prakash_lipsync/v1/vfhq/val.json", 
    ),
    dict(
        data_root = "/data/prakash_lipsync/", 
        ds_name = "th_1kh_512", 
        audio_features_name = "fe_wav2vec12", 
        train_json = "/data/prakash_lipsync/v1/th_1kh_512/train.json", 
        val_json = "/data/prakash_lipsync/v1/th_1kh_512/val.json", 
    ), 
    dict(
        data_root = "/data/prakash_lipsync/", 
        ds_name = "curated", 
        audio_features_name = "fe_wav2vec12", 
        train_json = "/data/prakash_lipsync/v1/curated/train.json", 
        val_json = "/data/prakash_lipsync/v1/curated/val.json", 
    ), 
    dict(
        data_root = "/data/prakash_lipsync/", 
        ds_name = "mead_domain1", 
        audio_features_name = "fe_wav2vec12", 
        train_json = "/data/prakash_lipsync/v1/mead_domain1/train.json", 
        val_json = "/data/prakash_lipsync/v1/mead_domain1/val.json", 
    )
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
project = "syncnet" + str(mouth_region_size[stage]) + "_fixed_silence_2"
save_dir = "lightning_logs/"
