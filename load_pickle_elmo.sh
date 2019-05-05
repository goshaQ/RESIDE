echo "Loading preprocessed Riedel-NYT 2010 Dataset with ELMo embeddings ..."

echo "Loading data from Google Drive"
wget --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XT60gSEoTpb-Y0MHxRbrp_PGCa3nl1zK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XT60gSEoTpb-Y0MHxRbrp_PGCa3nl1zK" \
    -O data/riedel_preprocessed_elmo.zip
rm -rf /tmp/cookies.txt

echo "Extracting files"
unzip data/riedel_preprocessed_elmo.zip -d data
rm data/riedel_preprocessed_elmo.zip