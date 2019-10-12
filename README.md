<p align="center"><img src="outputs/ap_kunkun.gif" width="70%" alt="" /></p>


## Usage

```
指定输入视频(input_video)   --viz-video
指定输入视频(the keypoints of input_video)  --input-npz
指定输出视频名称(the name of output video) --viz-output
指定输出的帧数(the frame number of output video)  --viz-limit
```

<br>

## handle video by hrnet
`python tools/hrnet_video.py --viz-output output.mp4 --viz-video /path/to/video.mp4`

<br>

## handle video by alphapose
`python tools/alphapose_video.py --viz-output output.mp4 --viz-video /path/to/video.mp4`

<br>

## handle video with every frame keypoints
`python tools/wild_video.py --viz-output output.mp4 --viz-video /path/to/video.mp4 --input-npz /path/to/input_name.npz`


## Model download

### Alphapose

- Download **yolov3-spp.weights** from ([Google Drive](https://drive.google.com/open?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC) | [Baidu pan](https://pan.baidu.com/s/1Zb2REEIk8tcahDa8KacPNA)),
  place to `./joints_detectors/Alphapose/models/yolo`

- Download **duc_se.pth** from ([Google Drive](https://drive.google.com/open?id=1OPORTWB2cwd5YTVBX-NE8fsauZJWsrtW) | [Baidu pan](https://pan.baidu.com/s/15jbRNKuslzm5wRSgUVytrA)),
  place to `./joints_detectors/Alphapose/models/sppe`

### HR-Net

- Download **pose_hrnet*** from [Google Drive](https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA), 
  place to `./joints_detectors/hrnet/models/pytorch/pose_coco/`

- Download **yolov3.weights** from [here](https://pjreddie.com/media/files/yolov3.weights),
  place to `./joints_detectors/hrnet/lib/detector/yolo`

### 3D Joint Detector

- Download **pretrained_h36m_detectron_coco.bin** from [here](https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin),
  place it into `./checkpoint` folder

## paper traslation 论文翻译
https://github.com/lxy5513/videopose/blob/master/doc/translate.md


