# TODO: create shell script for Problem 2
wget "https://www.dropbox.com/s/ls93u2yjl7jx014/model_p2_v2_19.pth?dl=1" -O ./p2/model_p2_v2_19.pth
python3 model_p2_batch.py $1 $2 $3