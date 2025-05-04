# Set destination directory
DEST=/home/zye52/scr4_hlee283/zye52/MSR-VTT
mkdir -p $DEST && cd $DEST

# Download all files
cd /home/zye52/scr4_hlee283/zye52/MSR-VTT

# Correct URLs (remove redundant "msr-vtt")
wget -c https://huggingface.co/datasets/AlexZigma/msr-vtt/resolve/main/data/MSR-VTT.ZIP
wget -c https://huggingface.co/datasets/AlexZigma/msr-vtt/resolve/main/data/test_videodatainfo.json.zip
wget -c https://huggingface.co/datasets/AlexZigma/msr-vtt/resolve/main/data/test_videos.zip
wget -c https://huggingface.co/datasets/AlexZigma/msr-vtt/resolve/main/data/train-00000-of-00001-60e50ff5fbbd1bb5.parquet
wget -c https://huggingface.co/datasets/AlexZigma/msr-vtt/resolve/main/data/val-00000-of-00001-01bacdd7064306bc.parquet


# Unzip ZIP files only
unzip -n MSR-VTT.ZIP
unzip -n test_videodatainfo.json.zip
unzip -n test_videos.zip
