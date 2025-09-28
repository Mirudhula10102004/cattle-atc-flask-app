from extract_measurements_and_save import run_and_save_results
import joblib
import pandas as pd

def classify_new_images(model_checkpoint, images_folder, classifier_model_path):
    # Extract measurements (for simplicity, pretend images_folder is dataset root)
    run_and_save_results(model_checkpoint, 'new_animal_measurements.csv')

    data = pd.read_csv('new_animal_measurements.csv').dropna(subset=['body_length','height_at_withers','chest_width'])
    X = data[['body_length','height_at_withers','chest_width']]

    clf = joblib.load(classifier_model_path)
    preds = clf.predict(X)
    data['predicted_ATC'] = preds

    print(data[['image_name', 'predicted_ATC']])
    data.to_csv('new_animal_predictions.csv', index=False)

if __name__ == '__main__':
    classify_new_images('model_checkpoint_resnet_epoch20.pth', '.', 'atc_classifier.joblib')
