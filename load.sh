echo "Loading and extracting Riedel-NYT 2010 Dataset..."

echo "Loading data from Google Drive"
wget --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1D7bZPvrSAbIPaFSG7ZswYQcPA3tmouCw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1D7bZPvrSAbIPaFSG7ZswYQcPA3tmouCw" \
    -O data/riedel_raw.zip

rm -rf /tmp/cookies.txt

echo "Extracting train.json and test.json"
unzip data/riedel_raw.zip -d data
mv data/riedel_data/* data
rm -rf data/riedel_data*