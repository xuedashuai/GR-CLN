Use the preprocess_for_learning.py to preprocess raw datasets.

Then, you need to create file folds as provided "1-MA" which contains three sub folds: "train", "val", and "test".

If you are going to running continual training, please firstly process other datasets here. (to fit the default direction in codes, please use "2-FT", "3-ZS", "4-EP", and "5-SR" to name your file folds.)

After completing the preprocessing, you can run gen_temp.py to generate temporary files, which speeds up training. (This step is optional.)
