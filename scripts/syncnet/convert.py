import torch


def main(loc, save_loc):
    x = torch.load(loc)
    y = {}
    y["params"] = x["hyper_parameters"]["opt"].to_dict()
    y["state_dict"] = {k.replace("syncnet.", ""): v for k, v in x["state_dict"].items()}
    torch.save(y, save_loc)


if __name__ == "__main__":
    weights = "weights/v9_syncnet/epoch=134-step=17415-val_ep_loss=0.194-train_ep_loss=0.192.ckpt"
    main(
        weights,
        "weights/v9/e134_256_e134-s17415-v194-t192-hdtf_iv_th1kh512.ckpt",
    )
