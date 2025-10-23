#! /bin/bash

run_comparison() {
    hands_repo="$HOME/Documents/projects/mecka-ai/hands"
    mmpretrain_repo="$HOME/Documents/src/python/mmpretrain"

    video_dir=$1
    video_path=$2

    video_base_name=$(basename $video_path .mp4)
    frame_dir="$video_dir/frames"

    wilor_hamer_json="$video_dir/wilor_hamer.json"
    wilor_hamer_svm_json="$video_dir/wilor_hamer_svm.json"

    hamer_data_list_json="$video_dir/hamer_data_list.json"
    hamer_data_list_svm_json="$video_dir/hamer_data_list_svm.json"

    infer_json="$video_dir/infer.json"
    pred_json="$video_dir/pred.json"

    pred_vid="$video_dir/pred.mp4"
    svm_vid="$video_dir/svm.mp4"
    compare_vid="$video_dir/compare.mp4"


    cd $hands_repo && conda activate hands
    # direct wilor+hamer detection
    python lib/hand_joints_extractor.py --input_video $video_path --output_json $wilor_hamer_json --config config_none_filter.yaml > log
    # svm filter output
    python lib/hand_joints_extractor.py --input_video $video_path --output_json $wilor_hamer_svm_json --config config.yaml > log


    cd $mmpretrain_repo && conda activate mmpretrain

    # extract frames from video
    python ego_hands/utils/extract_video_frames.py $video_dir --output $video_dir
    mv $video_dir/$video_base_name $frame_dir
    # hamer data list from direct wilor+hamer detection
    python ego_hands/runners/create_hamer_data_list.py --input $wilor_hamer_json --frame-folder $frame_dir --output-file $hamer_data_list_json
    # hamer data list from svm filter output
    python ego_hands/runners/create_hamer_data_list.py --input $wilor_hamer_svm_json --frame-folder $frame_dir --output-file $hamer_data_list_svm_json

    # create inference data list
    python ego_hands/runners/create_ego_hand_infer_data.py --hamer-data-list $hamer_data_list_json --output-file $infer_json
    # inference using classifier model
    python demo/ego_hand_inference.py configs/ego_hand/ego_classifier_cfg.py work_dirs/ego_hand/best_accuracy_top1_epoch_59.pth $infer_json --output $pred_json

    # visualize pred
    python ego_hands/runners/visualize_ego_hands.py --hamer-data-list $pred_json --video-output $pred_vid --show-wes false --quiet
    # visualize svm
    python ego_hands/runners/visualize_ego_hands.py --hamer-data-list $hamer_data_list_svm_json --video-output $svm_vid --show-wes false --quiet

    # concat videos for comparison
    python ego_hands/utils/concat_vids.py $pred_vid $svm_vid --out $compare_vid
}

video_dir="data/ultrawide_video/ultrawide_video_mop"
video_path="$video_dir/ultrawide_video_mop.mp4"

run_comparison $video_dir $video_path