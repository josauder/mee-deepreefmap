import wandb
from PIL import Image 
import numpy as np
from time import time 
import torch
from collections import defaultdict
import utils
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse
from model import BaselineExperiment

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="dataset")

def evaluate(model, logger, epoch):
    total_time = 0
    correctly_classified_pixels = []
    annotated_pixels = []
    per_class_correct = defaultdict(list)
    per_class_annotated_pixels = defaultdict(list)
    per_class_intersection = defaultdict(list)
    per_class_union = defaultdict(list)
    per_class_correct_polygons = defaultdict(list)
    per_class_correct50_polygons = defaultdict(list)
    per_class_correct90_polygons = defaultdict(list)
    pixels_confusion_matrix = np.zeros((len(classes), len(classes)))
    polygons_confusion_matrix = np.zeros((len(classes), len(classes)))


    for i, (image_file, label_file, polygon_file) in enumerate(zip(test_images, test_labels, test_polygon_files)):
        #  Load Image and Label 
        image = Image.open(image_file)
        label = np.load(label_file)
        polygons = np.load(polygon_file)

        # Predict, while timing
        t = time()
        prediction = model.predict(image)
        total_time += time() - t
        
        if i == epoch % len(test_images) :
            log_dict = {
                "eval/imgs/image": wandb.Image(image, caption="Image"),
                "eval/imgs/label": wandb.Image(0.4 * np.array(image)/255. + 0.6 * utils.color_rgb_image(label, classes, colors), caption="Label"),
                "eval/imgs/prediction": wandb.Image(0.4 * np.array(image)/255. + 0.6 * utils.color_rgb_image(prediction, classes, colors), caption="Prediction"),
                "eval/imgs/correct": wandb.Image(0.4 * np.array(image)/255. + 0.6 * utils.color_by_correctness(prediction, label), caption="Prediction")
            }

        # Compute metrics
        mask = label != 0
        correctly_classified_pixels.append((label[mask]==prediction[mask]).sum())
        annotated_pixels.append(mask.sum())
        cmat = confusion_matrix(label[mask], prediction[mask], labels=list(class_labels))
        pixels_confusion_matrix += cmat


        for class_name, class_label in classes.items():
            per_class_mask = label == class_label
            per_class_correct[class_name].append((label[per_class_mask]==prediction[per_class_mask]).sum())
            per_class_annotated_pixels[class_name].append(per_class_mask.sum())

            per_class_intersection[class_name].append((label==prediction)[per_class_mask].sum())
            per_class_union[class_name].append(np.logical_or(prediction[mask]==class_label, label[mask]==class_label).sum())


        for polygon in np.unique(polygons):
            if polygon>0:
                mask = polygon==polygons
                polygon_mask = label[mask]
                polygon_value = polygon_mask[0]
                if polygon_value > 0:
                    class_name = classes_inverse[polygon_value]
                    
                    polygon_prediction = prediction[mask]
                    polygons_confusion_matrix[polygon_value-1, polygon_prediction-1] += 1
                    per_class_correct_polygons[class_name].append((np.bincount(polygon_prediction.flatten()).argmax()==polygon_value))
                    per_class_correct50_polygons[class_name].append((polygon_prediction==polygon_value).mean() > 0.5)
                    per_class_correct90_polygons[class_name].append((polygon_prediction==polygon_value).mean() > 0.9)
    total_accuracy = np.array(correctly_classified_pixels).sum() / np.array(annotated_pixels).sum()
    per_class_accuracy = {class_label: np.array(correct).sum() / np.array(annotated).sum() for class_label, correct, annotated in zip(class_labels, per_class_correct.values(), per_class_annotated_pixels.values())}
    per_class_iou = {class_label: np.array(intersection).sum() / np.array(union).sum() for class_label, intersection, union in zip(class_labels, per_class_intersection.values(), per_class_union.values())}
    miou = np.array([iou for iou in per_class_iou.values() if not np.isnan(iou)]).mean()

    per_class_correct_polygons = {class_label: np.mean(per_class_correct_polygons[class_name]) for class_name, class_label in classes.items()}
    per_class_correct50_polygons = {class_label: np.mean(per_class_correct50_polygons[class_name]) for class_name, class_label in classes.items()}
    per_class_correct90_polygons = {class_label: np.mean(per_class_correct90_polygons[class_name]) for class_name, class_label in classes.items()}

    # Confusion Matrix
    # Sorted class names by class label
    display_labels = [class_name for class_name,class_label in sorted(classes.items(), key=lambda x: x[1])]
    disp = ConfusionMatrixDisplay(confusion_matrix=pixels_confusion_matrix/1000, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(15, 15))
    disp.plot(cmap='viridis', ax=ax, values_format='.0f', xticks_rotation=90)
    plt.savefig('pixel_confusion.png')
    plt.close()
    log_dict["eval/pixel_confusion_matrix"] = wandb.Image('pixel_confusion.png')

    disp = ConfusionMatrixDisplay(confusion_matrix=polygons_confusion_matrix, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(15, 15))
    disp.plot(cmap='viridis', ax=ax, values_format='.0f',xticks_rotation=90)
    plt.savefig('polygon_confusion.png')
    plt.close()
    log_dict["eval/polygon_confusion_matrix"] = wandb.Image('polygon_confusion.png')

    log_dict["eval/general/total_accuracy"] = total_accuracy
    log_dict["eval/general/miou"] = miou
    log_dict["eval/general/total_time"]= total_time
    log_dict['eval/general/per_polygon_accuracy'] = np.mean([acc for acc in per_class_correct_polygons.values() if not np.isnan(acc)])

    for class_name, class_label in classes.items():
        log_dict[f"eval/per_class/per_class_accuracy/{class_name.replace('/','_')}"] = per_class_accuracy[class_label]
        log_dict[f"eval/per_class/per_class_iou/{class_name.replace('/','_')}"] = per_class_iou[class_label]
        log_dict[f"eval/per_class/per_class_correct_polygons/{class_name.replace('/','_')}"] = per_class_correct_polygons[class_label]
        log_dict[f"eval/per_class/per_class_polygon_accuracy50/{class_name.replace('/','_')}"] = per_class_correct50_polygons[class_label]
        log_dict[f"eval/per_class/per_class_polygon_accuracy90/{class_name.replace('/','_')}"] = per_class_correct90_polygons[class_label]

    logger.log(log_dict, step=epoch)

def launch_experiment(experiment, name):

    logger = wandb.init(
        project="coral-segmentation",
        name=name,
    )

    for epoch in range(experiment.epochs):
        experiment.train_epoch(epoch, logger)

        if epoch % 10 == 0 or epoch == experiment.epochs-1:
            evaluate(experiment, logger, epoch)
            torch.save(experiment.model.state_dict(), f"{name}.pth")

if __name__ == '__main__':
    args = parser.parse_args()
    classes = json.loads(open(args.base_dir+"/classes.json").read())
    print(classes)
    classes_inverse = {v: k for k, v in classes.items()}
    class_labels = list(classes.values())
    class_names = list(classes.keys())
    colors = json.loads(open(args.base_dir+"/colors.json").read())

    train_images, test_images, train_labels, test_labels, test_polygon_files, counts = utils.load_files(args.base_dir, test_splits=["gabi_split_test"], ignore_splits=[])
    print("Number of training images:", len(train_images) ," number of test images ", len(test_images))


    launch_experiment(
        experiment=Experiment(classes, train_images, train_labels, counts, classes, colors, output_size=(352*2, 608*2)),
        name=args.name
    )
