# TODO: create shell script for Problem 3
wget "https://www.dropbox.com/s/d8u70mi0p9z0rft/model_p2_v1_16_57.pth?dl=1" -O ./p3/model_p2_v1_16_57.pth
python3 p3_inference.py $1 $2