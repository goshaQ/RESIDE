echo "Preprocessing Riedel-NYT 2010 Dataset..."

echo "\nMaking bags: python ./preproc/make_bags.py -data riedel"
python ./preproc/make_bags.py -data riedel

echo "\nGenerating pickle File: python ./preproc/generate_pickle.py -data riedel"
python ./preproc/generate_pickle.py -data riedel