# lipsync
syncing lip to audio in a video

## Installation

For detailed installation instructions, please see [INSTALLATION.md](INSTALLATION.md).

## Quick Start

- Run the setup `bin/run remote.setup`
- Install dependencies `bin/run deps`
- copy data to servers `for i in 7 8 9 10 11 12 13 14; do rsync -azP <folder> root@10.144.118.$i:<folder> ; done`

## Updates 
- loss functions 
    - lip and face L1 loss [done]
    - identity loss (insightface) 
    - lip distance loss (livepotrait) 
    - LPIPS loss - face and lip [Done]
    - discriminator loss - face and lip [Done]
    - sync loss [Done]
- Add spade decoder to the model. [Done]
- Add live portrait discriminator to the model. we need to extract two layers and measure the loss 


## Notes on Computer vision 
- If you are enlarging the image, you should prefer to use INTER_LINEAR or INTER_CUBIC interpolation. If you are shrinking the image, you should prefer to use INTER_AREA interpolation.

## Optimizations 
- In dataloader do the following to get 1.45x speedups compared to torchvision.io.read_image

```python
source_image_data = torchvision.io.read_image(
                source_loc, mode=torchvision.io.ImageReadMode.RGB
            )

transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Lambda(
                    lambda x: torchvision.transforms.functional.resize(
                        x, (self.img_h, self.img_w)
                    )
                ),  # Resize
                torchvision.transforms.Lambda(lambda x: x.float() / 255.0),
            ]
        )
source_image_data = transform(source_image_data)
```
- read about your gpu. make use of tensor cores by setting torch.set_float32_matmul_precision ("high") to get 8x improvements on gpu. by default when we use "highest" Tensorcores are not used. read about it more here https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html . we will go from 67 TFlops (float32) to 989 TFlops (tensor float32)
- with torch.autocast(device_type=device, dtype=torch.bfloat16) needs to be wrapped around model and loss calculation and leave backward pass outside it. bfloat16 will make us move from 989Tflops previously achieved to 1980 TFlops . 2x speed
- use torch.compile on model while training too . I m not sure why I thought this is used only for inference but this is basically eager mode in tensorflow and does kernel fusion of different layers. This should give some speedups
- always powers of 2 for batch size and across different layers.


## Workings
- preprocssing data and storing them. follow instructions in [link](https://github.com/iv-ml/DINet_internal/issues/36)
- to launch clip model training do `poetry runpython scripts/dinet/train.py --config_path="scripts/dinet/config.py"`

## Copy data in servers
we have two clusters one starts from `7 8 9 10 11 12 13 14` and the other one from `22 23 24 25 26 27 28 29`


> copy code 
``` bash
for i in 7 9 10 11 12 13 14; do rsync -azP /root/prakash/DINet_internal root@10.144.118.$i:/root/prakash/; done
```

> copy data 
```bash
for i in 11 12 13 14; do rsync -azP /data/prakash_dinet_data root@10.144.118.$i:/data/; done
```

> copy data from m1 m2 to cluster. 
```bash
rsync -az --info=progress2 /root/prakash/DINet_internal/weights c5:/root/prakash/DINet_internal
rsync -az --info=progress2 /root/prakash/data c16:/home/prakash
```

> check which nodes are free using
```
for i in $(seq 1 16); do echo $i $(ssh c$i 'nvidia-smi | head -14 | tail -4') & done | cut -d'|' -f1,11 | sed 's#/.*##' | sort -nk1
```

## Launch distributed training  

```bash
NODE_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500 WORLD_SIZE=32 python scripts/train_clip_ench2_only_sync.py --config_path="scripts/config/v11/exp1_128c_audio_only_sync_from_scratch_new_server.py"
NODE_RANK=1 MASTER_ADDR=10.144.118.22 MASTER_PORT=29500 WORLD_SIZE=32 python scripts/train_clip_ench2_only_sync.py --config_path="scripts/config/v11/exp1_128c_audio_only_sync_from_scratch_new_server.py"
NODE_RANK=2 MASTER_ADDR=10.144.118.22 MASTER_PORT=29500 WORLD_SIZE=32 python scripts/train_clip_ench2_only_sync.py --config_path="scripts/config/v11/exp1_128c_audio_only_sync_from_scratch_new_server.py"
NODE_RANK=3 MASTER_ADDR=10.144.118.22 MASTER_PORT=29500 WORLD_SIZE=32 python scripts/train_clip_ench2_only_sync.py --config_path="scripts/config/v11/exp1_128c_audio_only_sync_from_scratch_new_server.py"
```

```bash
NODE_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500 python scripts/dinet_lp/train.py --config_path="scripts/dinet_lp/config_512_static.py"
NODE_RANK=1 MASTER_ADDR=10.5.120.8 MASTER_PORT=29500 python scripts/dinet/train.py --config_path="scripts/dinet/config.py"
```

```bash
for i in m1 c2 c3 c4 c5 c6 c7 c8; do rsync -azP /data/prakash_lipsync/v2 prakash@$i:/data/prakash_lipsync/; done

for i in c2; do rsync -azP /data/prakash_lipsync/video_hallo_512_lp prakash@$i:/data/prakash_lipsync/; done

for i in 8 9 10 11 12 13 14; do rsync -azP /home/prakash/lipsync prakash@10.5.120.$i:/home/prakash/; done

for i in m1 c2 c3 c4 c5 c6 c7 c8; do rsync -azP --exclude=.venv/ --exclude=nbs/ --exclude=temp/ --exclude=lightning_logs/ /home/prakash/lipsync prakash@$i:/home/prakash/; done
```


## Inference

```bash
bin/run python lipsync/inference/infer.py <ckpt> <video> <audio> <dst>
```
Example:
```bash
bin/run python lipsync/inference/infer.py checkpoints/dinet_wider_mask.ckpt anshul3-30s.mp4 anshul-driving-audio-realistic.mp3 temp/
```

## ssh-keys distribution on cluster
- create a new key `ssh-keygen -t ed25519 -C "prakash@ivml.ai"` on local as `prakash_cluster`
- copy the private key to all the servers. 
- add .pub key to authorized_keys in all the servers. 
- Add config file to ~/.ssh/config with the following content
```bash
Host c*
    HostName 10.5.120.7
    User prakash
    IdentityFile ~/.ssh/prakash_cluster
    ForwardAgent yes
    AddKeysToAgent yes
```

## Resources 
- [Awesome talking faces](https://github.com/JosephPai/Awesome-Talking-Face)

# Running v10

```bash
bin/run wandb online # offline
poetry run python scripts/free_gpu_memory.py
mkdir wandb


# https://github.com/iv-ml/lipsync/issues/18
bin/run prod poetry run python src/clips/generate_wider_crop_without_padding.py --dst /data/prakash_lipsync/video_hallo_512/ --dataset hdtf --clip-txt "/data/prakash_lipsync/txt_files/hdtf/train.txt"
bin/run prod poetry run python src/clips/generate_wider_crop_without_padding.py --dst /data/prakash_lipsync/video_hallo_512/ --dataset hdtf --clip-txt "/data/prakash_lipsync/txt_files/hdtf/val.txt"

bin/run prod poetry run python src/clips/generate_dinet_clips.py --dst /data/prakash_lipsync/ --dataset "hdtf" --clip-txt "/data/prakash_lipsync/txt_files/hdtf/train.txt" --audio-only
bin/run prod poetry run python src/clips/generate_dinet_clips.py --dst /data/prakash_lipsync/ --dataset "hdtf" --clip-txt "/data/prakash_lipsync/txt_files/hdtf/val.txt" --audio-only

bin/run prod poetry run python scripts/classify_silence_bins.py /data/prakash_lipsync/video_hallo_512/bins/ --output-csv "/data/prakash_lipsync/silence.csv"

poetry run python src/clips/create_json.py --dst /data/prakash_lipsync/v1/ --clip-txt "/data/prakash_lipsync/txt_files/hdtf/train.txt" --data-dir /data/prakash_lipsync/ --dataset "hdtf" --silences /data/prakash_lipsync/silence.csv
poetry run python src/clips/create_json.py --dst /data/prakash_lipsync/v1/ --clip-txt "/data/prakash_lipsync/txt_files/hdtf/val.txt" --data-dir /data/prakash_lipsync/ --dataset "hdtf" --silences /data/prakash_lipsync/silence.csv


WANDB_DIR=wandb && WANDB_CACHE_DIR=wandb && WANDB_CONFIG_DIR=wandb && poetry run python scripts/dinet/train.py --config_path="scripts/dinet/config_512_v10.py"
```
