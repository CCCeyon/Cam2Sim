import csv

import cv2
import numpy as np
import tensorflow as tf
import scipy
from natsort import natsorted
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
from sklearn.metrics import normalized_mutual_info_score
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.applications.resnet50 import preprocess_input
from scipy.linalg import sqrtm
import glob
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
import warnings
from sklearn.neighbors import KernelDensity
from tensorflow.keras.metrics import MeanSquaredError

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import os

from tqdm import tqdm

from config import OUTPUT_FOLDER_NAME
from utils.argparser import parse_validation_args

args = parse_validation_args()


def calculate_kid_kitti(real_images, generated_images, image_size=(64, 64), batch_size=100):
    def calculate_gram_matrix(data):
        num_samples = tf.shape(data)[0]
        data = tf.image.resize(data, image_size)
        data = tf.reshape(data, [num_samples, -1])
        gram_matrix = tf.matmul(data, data, transpose_a=True)
        gram_matrix /= tf.cast(num_samples, tf.float32)
        return gram_matrix

    num_samples_real = len(real_images)
    num_samples_generated = len(generated_images)

    real_batches = [real_images[i:i+batch_size] for i in range(0, num_samples_real, batch_size)]
    generated_batches = [generated_images[i:i+batch_size] for i in range(0, num_samples_generated, batch_size)]

    mmd2 = 0.0

    for real_batch, generated_batch in zip(real_batches, generated_batches):
        real_batch = tf.convert_to_tensor(real_batch, dtype=tf.float32)
        generated_batch = tf.convert_to_tensor(generated_batch, dtype=tf.float32)
        real_batch = real_batch / 255.0
        generated_batch = generated_batch / 255.0
        gram_real = calculate_gram_matrix(real_batch)
        gram_generated = calculate_gram_matrix(generated_batch)
        batch_mmd2 = (
            tf.reduce_sum(gram_real) / (num_samples_real * (num_samples_real - 1)) +
            tf.reduce_sum(gram_generated) / (num_samples_generated * (num_samples_generated - 1)) -
            2 * tf.reduce_sum(tf.matmul(real_batch, generated_batch, transpose_a=True)) /
            (num_samples_real * num_samples_generated)
        )
        mmd2 += batch_mmd2
        print(mmd2)
    mmd2 /= len(real_batches)
    return mmd2


def calculate_inception_score(image_set1, image_set2, batch_size=32):
    inception_model = InceptionV3(weights='imagenet', include_top=False)
    inception_model = Model(inputs=inception_model.input, outputs=inception_model.layers[-2].output)
    def _get_predictions(images):
        n_batches = len(images) // batch_size
        preds = []
        for i in range(n_batches):
            batch = images[i * batch_size:(i + 1) * batch_size]
            batch = preprocess_input(batch)
            pred = inception_model.predict(batch)
            preds.append(pred)
        preds = np.concatenate(preds, axis=0)
        return preds

    preds_set1 = _get_predictions(image_set1)
    preds_set2 = _get_predictions(image_set2)
    p_yx_set1 = np.mean(preds_set1, axis=0)
    p_yx_set2 = np.mean(preds_set2, axis=0)
    epsilon = 1e-10
    p_yx_set1 = p_yx_set1 / np.sum(p_yx_set1)
    p_yx_set2 = p_yx_set2 / np.sum(p_yx_set2)
    p_yx_set1 += epsilon
    p_yx_set2 += epsilon
    kl_divergence = np.sum(p_yx_set1 * np.log(p_yx_set1 / p_yx_set2))
    score = np.exp(kl_divergence)

    return score


def calculate_fid(real_images, generated_images, batch_size=32, downsample_factor=4):
    inception_model = InceptionV3(weights='imagenet', include_top=False)
    inception_model = Model(inputs=inception_model.input, outputs=inception_model.layers[-2].output)
    def _get_activations(images):
        n_batches = len(images) // batch_size
        activations = []
        for i in range(n_batches):
            print("batch ",i," of ",n_batches)
            batch = images[i * batch_size:(i + 1) * batch_size]
            batch = preprocess_input(batch)
            batch_activations = inception_model.predict(batch)
            new_height = max(1, batch_activations.shape[1] // downsample_factor)
            new_width = max(1, batch_activations.shape[2] // downsample_factor)
            batch_activations = tf.image.resize(batch_activations, (new_height, new_width))
            activations.append(batch_activations)
            #tf.keras.backend.clear_session()
        activations = np.concatenate(activations, axis=0)
        return activations

    real_activations = _get_activations(real_images)
    generated_activations = _get_activations(generated_images)
    real_activations = real_activations.reshape(real_activations.shape[0], -1)
    generated_activations = generated_activations.reshape(generated_activations.shape[0], -1)
    mean_real_activations = np.mean(real_activations, axis=0)
    cov_real_activations = np.cov(real_activations, rowvar=False)
    mean_generated_activations = np.mean(generated_activations, axis=0)
    cov_generated_activations = np.cov(generated_activations, rowvar=False)
    mu1, sigma1 = mean_real_activations, cov_real_activations
    mu2, sigma2 = mean_generated_activations, cov_generated_activations
    warnings.filterwarnings("ignore")
    ssd = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssd + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def load_images_from_folder(folder_path):
    images = []
    paths = []
    for img_path in sorted(glob.glob(folder_path+'*.png')):
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                paths.append(img_path)
    return np.array(images),np.array(paths)


output_dir = os.path.join(OUTPUT_FOLDER_NAME,args.output_dir)

images1, _ = load_images_from_folder( os.path.join(output_dir,"carla") )

sim_folders = [f.path for f in os.scandir(output_dir) if
               f.is_dir() and f.name not in ["seg", "canny", "depth", "old", "carla"]]

#list_is = []
#list_fid = []
#list_kid = []
global_results = []

for simulated_folder in tqdm(natsorted(sim_folders), desc="Processing Folders"):
    images2, _ = load_images_from_folder( simulated_folder )


    print("Calculating Inception Score...")
    #inception_score = calculate_inception_score(images1, images2)
    inception_score = 0
    #list_is.append(inception_score)

    print("Calculating FID...")
    fid_score = calculate_fid(images1, images2)
    #list_fid.append(fid_score)

    print("Calculating Kernel Inception Distance (KID)...")
    kid_score = calculate_kid_kitti(images1, images2)
    #list_kid.append(kid_score.numpy())

    entry = [simulated_folder, inception_score, fid_score, kid_score.numpy()]
    global_results.append(entry)
    print(entry)

csv_filename = os.path.join(output_dir, "distribution_validation.csv")

with open(csv_filename, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Type", "IS", "FID", "KID"])
    writer.writerows(global_results)