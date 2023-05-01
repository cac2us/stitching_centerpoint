# [스티칭] 한양대 물체검출 모델 세팅

## 환경세팅

**Requirements**

- Python 3.6+ (python 3.6.9)
- Pytorch 1.1 or higher (1.8.1)
- CUDA 10.0 or higher (11.1)
- spconv

괄호 안은 도커에 설치되어있는 환경 버전

```bash
docker pull yckimm/stitching:v1.1 # (도커 공유)
```

## 디렉토리 세팅

HDD로 전달받은 파일 중 필요한 파일만 저장시키기 위해 기존 데이터 폴더에서 생성한 samples폴더로 옮기는 작업을 수행

**폴더 구조**

gen_stitch_data_format_lidar.py, gen_stitch_data_format_cam_radar.py, gen_stitch_data_to_folder.py, gen_stitch_data_rot_15.py를 stitch 데이터 폴더 안에 넣고 실행

```bash
stitch
|— data # 기존 데이터 폴더
|    |— pole_1
|    |— pole_2
|— annotation # 전달받은 annotation 파일
|    |— keyframe_lidar
|    |— scene_001
|— **gen_stitch_data_format_lidar.py (파일첨부)**
|— **gen_stitch_data_format_cam_radar.py (파일첨부)**
|— **gen_stitch_data_to_folder.py (파일첨부)**
|— **gen_stitch_data_rot_15.py (파일첨부)**
|— annotated_data # 코드로 자동생성
|— annotated_data_rot15 # 코드로 자동생성
|— samples # 코드로 자동생성
```

**실행해야할 명령어**

```bash
python3 gen_stitch_data_format_lidar.py # (라이다 데이터에 대해 생성)
python3 gen_stitch_data_format_cam_radar.py # (카메라, 레이더 데이터에 대해 생성)
python3 gen_stitch_data_rot_15.py # (keyframe을 scene으로 변환, pcd 15도 회전)
python3 gen_stitch_data_to_folder.py # (samples 폴더로 필요한 pcd 파일 옮겨줌)
```

실행 시 annotated_data 폴더에 물체검출모델 training 및 inference를 위한 파일이 저장되게 됨

## 프로젝트 세팅

**코드 다운 및 폴더 세팅**

```bash
git clone https://github.com/rasd3/stitching_centerpoint.git
cd stitching_centerpoint
bash setup.sh
mkdir data && mkdir data/stitch
```

**데이터셋 구조 세팅**

```bash
# For nuScenes Dataset         
└── stitching_centerpoint/data/stitch/
       ├── samples       <-- (ln -s stitch/samples ./)
       ├── v0.5-stitch   <-- metadata (파일첨부)
       ├── maps          <-- (파일첨부)
```

samples는 디렉토리 세팅 단계에서 제작한 samples를 심볼링 링크로 가져와서 사용, 동봉한 
v0.5-stitch 압축파일을 해당 위치에 복사

**Create data**

```bash
# stitching_centerpoint/
python tools/create_data.py stitch_data_prep ./data/stitch v0.5-stitch
```

**Inference**

```bash
python tools/dist_test.py ./configs/nusc/voxelnet/stitch/stitch_centerpoint_voxelnet_01voxel.py --work_dir ./work_dir/test --checkpoint ./stitching_centerpoint_test.pth
```

det3d/models/detectors/voxelnet.py L90: if False → True로 바꿔서 visualize 가능