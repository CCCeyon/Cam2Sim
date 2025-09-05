# Distribution level results
python -m validation.get_latex_figure --output_dir "fortuna_val_real" --value "KID_mean" --file "distribution.csv" --name "Avg. Kernel Inception Dist." --pgf "fortunaval_distlevel_kid"
python -m validation.get_latex_figure --output_dir "fortuna_val_real" --value "IS_mean" --file "distribution.csv" --name "Avg. Inception Score" --pgf "fortunaval_distlevel_is"
python -m validation.get_latex_figure --output_dir "fortuna_val_real" --value "FID" --file "distribution.csv" --name "Avg. Fréchet Inception Dist." --pgf "fortunaval_distlevel_fid"

python -m validation.get_latex_figure --output_dir "tesla_val_real" --value "KID_mean" --file "distribution.csv" --name "Avg. Kernel Inception Dist." --pgf "tesla3val_distlevel_kid"
python -m validation.get_latex_figure --output_dir "tesla_val_real" --value "IS_mean" --file "distribution.csv" --name "Avg. Inception Score" --pgf "tesla3_distlevel_is"
python -m validation.get_latex_figure --output_dir "tesla_val_real" --value "FID" --file "distribution.csv" --name "Avg. Fréchet Inception Dist." --pgf "tesla3val_distlevel_fid"

python -m validation.get_latex_figure --output_dir "fortuna_carla_sim" --value "KID_mean" --file "distribution.csv" --name "Avg. Kernel Inception Dist." --pgf "fortunacarlasim_distlevel_kid"
python -m validation.get_latex_figure --output_dir "fortuna_carla_sim" --value "IS_mean" --file "distribution.csv" --name "Avg. Inception Score" --pgf "fortunacarlasim_distlevel_is"
python -m validation.get_latex_figure --output_dir "fortuna_carla_sim" --value "FID" --file "distribution.csv" --name "Avg. Fréchet Inception Dist." --pgf "fortunacarlasim_distlevel_fid"

python -m validation.get_latex_figure --output_dir "tesla_carla_sim" --value "KID_mean" --file "distribution.csv" --name "Avg. Kernel Inception Dist." --pgf "tesla3carlasim_distlevel_kid"
python -m validation.get_latex_figure --output_dir "tesla_carla_sim" --value "IS_mean" --file "distribution.csv" --name "Avg. Inception Score" --pgf "tesla3carlasim_distlevel_is"
python -m validation.get_latex_figure --output_dir "tesla_carla_sim" --value "FID" --file "distribution.csv" --name "Avg. Fréchet Inception Dist." --pgf "tesla3carlasim_distlevel_fid"


# Semantic Segmentation level results
python -m validation.get_latex_figure --output_dir "z_valtest_real" --value "SSS" --file "results.csv" --name "Avg. Sem. Segmentation Score" --pgf "tesla3val_seglevel_sss"
python -m validation.get_latex_figure --output_dir "fortuna_val_real" --value "SSS" --file "results.csv" --name "Avg. Sem. Segmentation Score" --pgf "fortunaval_seglevel_sss"

python -m validation.get_latex_figure --output_dir "tesla_carla_sim" --value "SSS" --file "results.csv" --name "Avg. Sem. Segmentation Score" --pgf "tesla3carlasim_seglevel_sss"
python -m validation.get_latex_figure --output_dir "fortuna_carla_sim" --value "SSS" --file "results.csv" --name "Avg. Sem. Segmentation Score" --pgf "fortunacarlasim_seglevel_sss"

python -m validation.get_latex_figure --output_dir "z_valtest_real" --value "SSS" --file "results.csv" --name "Avg. Sem. Segmentation Score" --pgf "tesla3carlaest_seglevel_sss"
python -m validation.get_latex_figure --output_dir "fortuna_carla_real" --value "SSS" --file "results.csv" --name "Avg. Sem. Segmentation Score" --pgf "fortunacarlaest_seglevel_sss"


# single image metrics
python -m validation.get_latex_figure --output_dir "fortunavalidationset" --value "MSE" --file "results.csv" --name "Avg. Mean Squared Error" --pgf fortunaval_simetrics_mse
python -m validation.get_latex_figure --output_dir "z_valtest_real" --value "MSE" --file "results.csv" --name "Avg. Mean Squared Error" --pgf tesla3val_simetrics_mse
python -m validation.get_latex_figure --output_dir "fortunavalidationset" --value "SSIM" --file "results.csv" --name "Avg. Structural Similarity Index" --pgf fortunaval_simetrics_ssim
python -m validation.get_latex_figure --output_dir "z_valtest_real" --value "SSIM" --file "results.csv" --name "Avg. Structural Similarity Index" --pgf tesla3val_simetrics_ssim
python -m validation.get_latex_figure --output_dir "fortunavalidationset" --value "PSNR" --file "results.csv" --name "Avg. Peak Signal-to-Noise Ratio" --pgf fortunaval_simetrics_psnr
python -m validation.get_latex_figure --output_dir "z_valtest_real" --value "PSNR" --file "results.csv" --name "Avg. Peak Signal-to-Noise Ratio" --pgf tesla3val_simetrics_psnr


# temp consistency results
python -m validation.get_latex_figure --output_dir "fortuna_val_real" --value "MSE" --file "results_tempconsistency.csv" --name "Avg. Mean Squared Error" --pgf fortunaval_tempcons_mse


python -m validation.get_latex_figure --output_dir "fortuna_carla_sim" --value "MSE" --file "results_tempconsistency.csv" --name "Avg. Mean Squared Error" --pgf fortunacarlasim_tempcons_mse
python -m validation.get_latex_figure --output_dir "tesla_carla_sim" --value "MSE" --file "results_tempconsistency.csv" --name "Avg. Mean Squared Error" --pgf tesla3carlasim_tempcons_mse

python -m validation.get_latex_figure --output_dir "fortuna_carla_real" --value "MSE" --file "results_tempconsistency.csv" --name "Avg. Mean Squared Error" --pgf fortunacarlaest_tempcons_mse
python -m validation.get_latex_figure --output_dir "tesla_carla_real" --value "MSE" --file "results_tempconsistency.csv" --name "Avg. Mean Squared Error" --pgf tesla3carlaest_tempcons_mse

python -m validation.get_latex_figure --output_dir "fortuna_carla_sim" --value "SSIM" --file "results_tempconsistency.csv" --name "Avg. Structural Similarity Index" --pgf fortunacarlasim_tempcons_ssim
python -m validation.get_latex_figure --output_dir "tesla_carla_sim" --value "SSIM" --file "results_tempconsistency.csv" --name "Avg. Structural Similarity Index" --pgf tesla3carlasim_tempcons_ssim

python -m validation.get_latex_figure --output_dir "fortunavalidationset" --value "SSIM" --file "results_tempconsistency.csv" --name "Avg. Structural Similarity Index" --pgf fortunaval_tempcons_ssim
python -m validation.get_latex_figure --output_dir "fortuna_val_real" --value "SSIM" --file "results_tempconsistency.csv" --name "Avg. Structural Similarity Index" --pgf tesla3val_tempcons_ssim

python -m validation.get_latex_figure --output_dir "fortuna_carla_real" --value "SSIM" --file "results_tempconsistency.csv" --name "Avg. Structural Similarity Index" --pgf fortunacarlaest_tempcons_ssim
python -m validation.get_latex_figure --output_dir "tesla_carla_real" --value "SSIM" --file "results_tempconsistency.csv" --name "Avg. Structural Similarity Index" --pgf tesla3carlaest_tempcons_ssim


# yolo results
python -m validation.get_latex_figure --output_dir "fortuna_val_real" --value "Veh offset mean" --file "veh_results.csv" --name "Avg. Vehicle Offset" --pgf fortunaval_vehmetrics_vehoffset
python -m validation.get_latex_figure --output_dir "z_valtest_real" --value "Veh offset mean" --file "veh_results.csv" --name "Avg. Vehicle Offset" --pgf tesla3val_vehmetrics_vehoffset

python -m validation.get_latex_figure --output_dir "fortuna_val_real" --value "Recalls" --file "veh_results.csv" --name "Avg. Vehicle Recalls" --pgf fortunaval_vehmetrics_recalls
python -m validation.get_latex_figure --output_dir "z_valtest_real" --value "Recalls" --file "veh_results.csv" --name "Avg. Vehicle Recalls" --pgf tesla3val_vehmetrics_recalls

python -m validation.get_latex_figure --output_dir "fortuna_val_real" --value "Precisions" --file "veh_results.csv" --name "Avg. Vehicle Precision" --pgf fortunaval_vehmetrics_precision
python -m validation.get_latex_figure --output_dir "z_valtest_real" --value "Precisions" --file "veh_results.csv" --name "Avg. Vehicle Precision" --pgf tesla3val_vehmetrics_precision

python -m validation.get_latex_figure --output_dir "fortuna_val_real" --value "Conf Diff Avg NoMean" --file "veh_results.csv" --name "Avg. Confidence Difference" --pgf fortunaval_vehmetrics_confdiff
python -m validation.get_latex_figure --output_dir "z_valtest_real" --value "Conf Diff Avg NoMean" --file "veh_results.csv" --name "Avg. Confidence Difference" --pgf tesla3val_vehmetrics_confdiff

python -m validation.get_latex_figure --output_dir "fortuna_val_real" --value "Avg. IoU" --file "veh_results.csv" --name "Avg. Intersection over Union" --pgf fortunaval_vehmetrics_iou
python -m validation.get_latex_figure --output_dir "z_valtest_real" --value "Avg. IoU" --file "veh_results.csv" --name "Avg. Intersection over Union" --pgf tesla3val_vehmetrics_iou


python -m validation.get_latex_figure --output_dir "fortuna_carla_sim" --value "Veh offset mean" --file "veh_results.csv" --name "Avg. Vehicle Offset" --pgf fortunacarlasim_vehmetrics_vehoffset
python -m validation.get_latex_figure --output_dir "tesla_carla_sim" --value "Veh offset mean" --file "veh_results.csv" --name "Avg. Vehicle Offset" --pgf tesla3carlasim_vehmetrics_vehoffset

python -m validation.get_latex_figure --output_dir "fortuna_carla_sim" --value "Recalls" --file "veh_results.csv" --name "Avg. Vehicle Recalls" --pgf fortunacarlasim_vehmetrics_recalls
python -m validation.get_latex_figure --output_dir "tesla_carla_sim" --value "Recalls" --file "veh_results.csv" --name "Avg. Vehicle Recalls" --pgf tesla3carlasim_vehmetrics_recalls

python -m validation.get_latex_figure --output_dir "fortuna_carla_sim" --value "Precisions" --file "veh_results.csv" --name "Avg. Vehicle Precision" --pgf fortunacarlasim_vehmetrics_precision
python -m validation.get_latex_figure --output_dir "tesla_carla_sim" --value "Precisions" --file "veh_results.csv" --name "Avg. Vehicle Precision" --pgf tesla3carlasim_vehmetrics_precision

python -m validation.get_latex_figure --output_dir "fortuna_carla_sim" --value "Conf Diff Avg NoMean" --file "veh_results.csv" --name "Avg. Confidence Difference" --pgf fortunacarlasim_vehmetrics_confdiff
python -m validation.get_latex_figure --output_dir "tesla_carla_sim" --value "Conf Diff Avg NoMean" --file "veh_results.csv" --name "Avg. Confidence Difference" --pgf tesla3carlasim_vehmetrics_confdiff

python -m validation.get_latex_figure --output_dir "fortuna_carla_sim" --value "Avg. IoU" --file "veh_results.csv" --name "Avg. Intersection over Union" --pgf fortunacarlasim_vehmetrics_iou
python -m validation.get_latex_figure --output_dir "tesla_carla_sim" --value "Avg. IoU" --file "veh_results.csv" --name "Avg. Intersection over Union" --pgf tesla3carlasim_vehmetrics_iou
