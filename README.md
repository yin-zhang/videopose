



<p align="center"><img src="outputs/op_girl.gif" width="70%" alt="" /></p>


## envrionment configture
I use torch1.0.1 in conda
`conda env create -f env_info_file.yml`


<br>


## DEMO



`python demo.py`

```
指定输入视频(input_video)   --viz-video
指定输入视频(the keypoints of input_video)  --input-npz
指定输出视频名称(the name of output video) --viz-output
指定输出的帧数(the frame number of output video)  --viz-limit
```

<br>

## RealTime WebCam Demo

https://github.com/lxy5513/videopose/blob/master/outputs/README.md


<br>

## handle live video by hrnet
`updata at 2019-04-17`
```
cd joints_detectors/hrnet/lib/
make
cd -
python tools/hrnet_realtime.py -video /path/to/video.mp4
```

## handle video by hrnet
`python tools/hrnet_video.py --viz-output output.mp4 --viz-video /path/to/video.mp4`


add hrnet 2D keypoints detection module, to realize the end to end 3D reconstruction
添加hrnet 2D关键点检测模块,实现更高精读的3D video 重构   [`hrnet`](https://github.com/lxy5513/hrnet)

<br>

## handle video by alphapose
`python tools/alphapose_video.py --viz-output output.mp4 --viz-video /path/to/video.mp4`

<br>
or
<br>

`python tools/aphapose_video.py --viz-output output.gif --viz-video /path/to/video.mp4 --viz-limit 180`
need download model





<br>
<br>

## handle video with every frame keypoints
`python tools/wild_video.py --viz-output output.mp4 --viz-video /path/to/video.mp4 --input-npz /path/to/input_name.npz`


<br>
<br>

## `TODO`
- [ ] add `FPN-DCN` huamn detector for hrnet to realize better accuracy.





<br>
<br>


## 相关模型下载related model download

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


