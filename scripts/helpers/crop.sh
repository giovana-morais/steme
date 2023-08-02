#!/bin/sh

# get 30 s of audio
for i in *_augmented.wav; do
	newfile="${i%.*}_cropped.wav";
	echo "cropping $i to $newfile";
	ffmpeg -i $i -ss 00:00:5 -to 00:00:35 -acodec copy $newfile;
done
