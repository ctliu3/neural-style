export CUDA_VISIBLE_DEVICES=0
python ./neural-style/fast.py \
  --style-image="./images/themuse.jpg" \
  --content-image="./images/robot.jpg" \
  --loss-style=5 \
  --loss-feature=1 \
  --loss-interval=100 \
  --pretrained-model="./models/vgg16-00b39a1b-255.pth" \
  --use-cuda \
  --styled-prefix="fast_robot_muse" \
  --snapshot-prefix="fast_robot_muse" \
  --snapshot-interval=5000 \
  --styled-interval=100 \
  --epoch=4 \
  --lr=0.001 \
  --size="256,256" \
  --batch-size=4 \
  --dataset="/temp-hdd/liuchundian/dataset/coco/" 2>&1 | tee fast_robot_muse.log
