
export CUDA_VISIBLE_DEVICES=0

python ./neural-style/neural_style.py --loss-style=10000 \
  --loss-feature=1 \
  --loss-interval=100 \
  --pretrained-model="./models/vgg19-d01eb7cb-255.pth" \
  --use-cuda \
  --styled-prefix="dam_starry_night" \
  --snapshot-prefix="dam_starry_night" \
  --snapshot-interval=5000 \
  --styled-interval=100 \
  --niter=50000 \
  --lr=0.01 \
  --size="512,512" 2>&1 | tee style.log
