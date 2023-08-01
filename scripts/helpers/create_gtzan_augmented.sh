#!/bin/zsh

foldername=gtzan_augmented
audiofolder=$foldername/audio
tempofolder=$foldername/annotations/tempo
rm -rf foldername

# create folders
mkdir $audiofolder -p
mkdir $tempofolder -p

basefolder=gtzan_genre/gtzan_genre/genres

# copy audio content content there
echo "copying audio data to $audiofolder";
for i in gtzan_genre/gtzan_genre/genres/*; do
	# echo "copying $i to $audiofolder"
	cp $i/* $audiofolder;
done

# copy annotations
echo "copying bpm data to $tempofolder"
for i in gtzan_genre/gtzan_tempo_beat-main/tempo/*; do
	cp $i $tempofolder;
done

cd $tempofolder;

# format names
echo "formatting audio data"
for i in gtzan_*; do
	remove_gtzan=${i:6};
	new_name=${remove_gtzan/_/.};
	mv $i $new_name;
done
