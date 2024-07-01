python3 tools/simple_inference_stitch.py ./configs/nusc/voxelnet/stitch_centerpoint_voxelnet_01voxel_pcr2.py --checkpoint ./stitching_centerpoint_test.pth --input_data_dir ./stitching_dir/
python3 tools/stitch_tracking/pub_test.py --work_dir track --checkpoint prediction.json
cd stitching_prediction/
sh ./rm.sh
sh ./script.sh
cd ../
python3 overall_demomaker.py
