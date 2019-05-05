echo "Loading preprocessed Riedel-NYT 2010 Dataset with GloVe embeddings ..."

echo "Loading data from Google Drive"
mkdir data
wget --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UD86c_6O_NSBn2DYirk6ygaHy_fTL-hN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UD86c_6O_NSBn2DYirk6ygaHy_fTL-hN" \
    -O data/riedel_preprocessed.zip
rm -rf /tmp/cookies.txt

echo "Extracting files"
unzip data/riedel_preprocessed.zip -d data
mv data/data/* data
rm data/riedel_preprocessed.zip
rm -rf data/data