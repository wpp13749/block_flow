# block_flow
# train cifar-10

```bash
python train_reverse_img_ddp.py --gpu 0 --dir ./runs/cifar10-beta1/ --weight_prior 1 --learning_rate 2e-4 --dataset cifar10 --warmup_steps 5000 --optimizer adam --batchsize 128 --iterations 500000 --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json

# Acknowledgements
Thanks to [fast_ode](https://github.com/sangyun884/fast-ode) and [EDM](https://github.com/nvlabs/edm) for providing their implementations, which have significantly contributed to this codebase.
