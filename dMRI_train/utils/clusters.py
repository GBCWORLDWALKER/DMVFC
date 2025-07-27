import numpy
import pandas
import os
from joblib import Parallel, delayed
import wandb

import utils.fibers as fiber_distance
import utils
from utils import utils

# def outlier_removal(num_clusters, num_std, preds_final, probs_final, x_arrays, fiber_surfs_dk, subID, input_pds, name, args):
def outlier_removal(params, pred_labels, pred_probs):

    num_clusters = params["num_clusters"]
    outlier_num_std = params["ro_std"]

    rejected_fibers = []
    for c_idx in range(num_clusters):
        cluster_fiber_indices = numpy.where(pred_labels == c_idx)[0]
        if cluster_fiber_indices.shape[0] == 0:
            utils.print_both(params["log"], '  ** Warning: cluster %d is empty' % (c_idx+1))
        cluster_probs = pred_probs[cluster_fiber_indices]

        if len(cluster_probs) > 0:
            cluster_rej_indices = numpy.where((cluster_probs.mean() - cluster_probs) > outlier_num_std * cluster_probs.std())[0]
            if len(cluster_rej_indices) > 0:
                rejected_fibers.extend(cluster_fiber_indices[cluster_rej_indices])

    rejected_fibers = numpy.array(rejected_fibers)

    utils.print_both(params["log"], '%d / %f fibers are removed:' % (rejected_fibers.shape[0], len(pred_labels)))

    return rejected_fibers


def metrics_calculation(params, fiber_array, pred_labels, pred_probs, outlier_fibers=None, output=True, verbose=False):

    num_clusters = params["num_clusters"]

    fiber_array_ras = fiber_array.fiber_array_ras
    fiber_array_endpoints = fiber_array.fiber_endpoint
    fiber_subID = fiber_array.fiber_subID
    fiber_Uratio = fiber_array.fiber_Uratio
    fiber_length = fiber_array.fiber_length.astype(numpy.float64)
    fiber_hemispheres = fiber_array.fiber_hemispheres

    if outlier_fibers is not None:
        temp = numpy.ones(len(pred_labels))
        temp[outlier_fibers] = 0
        mask = temp > 0

        fiber_array_ras = fiber_array_ras[mask]
        fiber_array_endpoints = fiber_array_endpoints[mask]
        fiber_subID = fiber_subID[mask]

        pred_labels = pred_labels[mask]
        pred_probs = pred_probs[mask]

        fiber_Uratio = fiber_Uratio[mask]
        fiber_length = fiber_length[mask]
        fiber_hemispheres = fiber_hemispheres[mask]

    if verbose: print(" - compute cluster centroids ....")
    cluster_centroids, cluster_reordered_fibers, cluster_alphas = \
        compute_cluster_centroids(num_clusters, pred_labels, fiber_array_ras)

    if verbose: print(" - compute cluster basics ....")
    cluster_NoF, cluster_length, cluster_Uratio, cluster_hemishpere, cluster_prob = \
        cluster_basics_calculation_SWM(num_clusters, pred_labels, fiber_Uratio, fiber_length, fiber_hemispheres, pred_probs)

    if verbose: print(" - compute cluster DB index and Alpha....")
    db_mean, db_std, db_all = db_calculation_SWM(num_clusters, pred_labels, cluster_centroids, cluster_alphas)
    alpha_mean = numpy.nanmean(cluster_alphas)
    alpha_std = numpy.nanstd(cluster_alphas)

    if verbose: print(" - compute cluster WMPG ....")
    wmpg_mean, wmpg_std, wmpg_all = wmpg_calculation_SWM(num_clusters, pred_labels, fiber_subID)

    if verbose: print(" - compute cluster TSPC ....")
    tspc_mean, tspc_std, tspc_all, eps_all = tspc_calculation_SWM(num_clusters, pred_labels, fiber_array_endpoints, cluster_reordered_fibers)

    # all togehter
    db_all = ["%0.3f"%v if v>=0 else "nan" for v in db_all]
    wmpg_all = ["%0.3f"%v if v>=0 else "nan" for v in wmpg_all]
    tspc_all = ["%0.3f"%v if v>=0 else "nan" for v in tspc_all]
    cluster_alphas = ["%0.3f"%v if v!=0 else "nan" for v in cluster_alphas]
    data = numpy.array((cluster_NoF, cluster_length, cluster_hemishpere, cluster_Uratio, cluster_prob, db_all, cluster_alphas, tspc_all, eps_all, wmpg_all), dtype=object).transpose()
    df_stats = pandas.DataFrame(data=data, columns=["NoFiber", "F-Len(mm)", "Hemi(R/L)", "Uratio", "Pred-Prob", "DB-score", "Alpha", "TSPC", "EP-ROIs", "WMPG"])
    df_stats.index = ['cluster_{0:05d}'.format(idx + 1) for idx in df_stats.index]

    if verbose:
        utils.print_both(params["log"], df_stats.to_string())

    tmp = '\nDB: %0.3f (%0.3f),\tAlpha: %0.3f (%0.3f),\tTSPC: %0.3f (%0.3f),\tWMPG: %0.3f (%0.3f)' %\
          (db_mean, db_std, alpha_mean, alpha_std, tspc_mean, tspc_std, wmpg_mean, wmpg_std)
    utils.print_both(params["log"], tmp)
    if params["wandb_group"] is not None:
        wandb.init(group=params["wandb_group"])
        os.environ["WANDB_RUN_GROUP"] = params["wandb_group"]
        wandb.summary.update({"alpha": alpha_mean})
    # save stats
    if output:
        output_name = params['output_name']
        output_dir = params['output_dir']

        # output stats csv file
        stat_csv = os.path.join(output_dir, output_name+".stats.csv")
        if outlier_fibers is not None:
            stat_csv = os.path.join(output_dir, output_name+"_ro.stats.csv")
        df_stats.to_csv(stat_csv)
        utils.print_both(params["log"], "Save stats file at:\t %s.\n"%stat_csv)
        

    return df_stats, cluster_centroids, cluster_reordered_fibers


def compute_cluster_centroids(num_clusters, predicted, x_arrays):

    def compute(c_idx):

        x_array_orig = x_arrays[predicted == c_idx]
        x_array_quiv = numpy.flip(x_array_orig, axis=1)
        num_fibers = x_array_orig.shape[0]

        if num_fibers == 0:
            centroid = None
            reordered_fibers = numpy.array([])
            alpha = numpy.nan
        elif num_fibers == 1:
            centroid = x_array_orig
            reordered_fibers = numpy.array([0.0])
            alpha = numpy.float64(0.0)
        else:
            dis_sum = numpy.zeros(num_fibers)
            dis_arg_list = numpy.zeros((num_fibers, num_fibers))
            dis_pairwise = numpy.zeros((num_fibers, num_fibers))
            for f_idx in range(num_fibers):

                fiber_array = x_array_orig[f_idx, :]

                dis_orig = fiber_distance._fiber_distance_internal_use(
                    fiber_array[:, 0], fiber_array[:, 1], fiber_array[:, 2], x_array_orig)
                dis_quiv = fiber_distance._fiber_distance_internal_use(
                    fiber_array[:, 0], fiber_array[:, 1], fiber_array[:, 2], x_array_quiv)

                dis_tmp = numpy.stack((dis_orig, dis_quiv), axis=0)
                dis_min = numpy.min(dis_tmp, axis=0)
                dis_arg = numpy.argmin(dis_tmp, axis=0)

                dis_arg_list[f_idx, :] = dis_arg
                dis_sum[f_idx] = numpy.sum(dis_min)
                dis_pairwise[f_idx] = dis_min

            # for DB index computation
            tmp = dis_pairwise.flatten()
            tmp = tmp[tmp > 0]
            alpha = tmp.mean()

            center_fiber_idx = numpy.argmin(dis_sum)
            reordered_fibers = dis_arg_list[center_fiber_idx, :]

            x_array_orig_ = x_array_orig[numpy.where(reordered_fibers == 0)]
            x_array_quiv_ = x_array_quiv[numpy.where(reordered_fibers == 1)]

            x_array_reodered = numpy.concatenate((x_array_orig_, x_array_quiv_))

            centroid = numpy.mean(x_array_reodered, axis=0)

        return centroid, reordered_fibers, alpha

    # for c_idx in range(num_clusters):
    #
    #     centroid, reordered_fibers, alpha = compute(c_idx)
    #
    #     cluster_centroids[c_idx, :] = centroid
    #     cluster_reordered_fibers.append(reordered_fibers)
    #     cluster_alphas.append(alpha)

    pp_results = Parallel(n_jobs=1, verbose=0)(
        delayed(compute)(c_idx)
        for c_idx in range(num_clusters))

    num_points = x_arrays.shape[1]
    cluster_centroids = numpy.zeros((num_clusters, num_points, 3))
    cluster_reordered_fibers = []
    cluster_alphas = []

    for c_idx in range(num_clusters):

        print('centroid', c_idx)

        cluster_fiber_indices = predicted == c_idx
        num_fibers = numpy.sum(cluster_fiber_indices)

        centroid = pp_results[c_idx][0]
        reordered_fibers = pp_results[c_idx][1]
        alpha = pp_results[c_idx][2]

        if num_fibers > 0:
            cluster_centroids[c_idx, :] = centroid
        cluster_reordered_fibers.append(reordered_fibers)
        cluster_alphas.append(alpha)

    return cluster_centroids, cluster_reordered_fibers, cluster_alphas


def cluster_basics_calculation_SWM(num_clusters, pred_labels, fiber_Uratio, fiber_length, fiber_hemispheres, pred_probs):

    cluster_NoF = []
    cluster_length = []
    cluster_Uratio = []
    cluster_hemishpere = []
    cluster_prob = []
    for c_idx in range(num_clusters):
        cluster_fiber_indices = pred_labels == c_idx
        num_fibers = numpy.sum(cluster_fiber_indices)
        cluster_NoF.append("%d" % num_fibers)

        if num_fibers > 0:
            len_mean = numpy.mean(fiber_length[cluster_fiber_indices])
            len_std = numpy.std(fiber_length[cluster_fiber_indices])
            cluster_length.append("%0.3f" % (len_mean))
            # cluster_length.append("%0.3f" % (len_mean, len_std))

            U_mean = numpy.mean(fiber_Uratio[cluster_fiber_indices])
            U_std = numpy.std(fiber_Uratio[cluster_fiber_indices])
            cluster_Uratio.append("%0.3f" % (U_mean))

            cluster_hemishpere.append("%d/%d" % (numpy.sum(fiber_hemispheres[cluster_fiber_indices]==0),
                                                 numpy.sum(fiber_hemispheres[cluster_fiber_indices]==1)))

            prob_mean = numpy.mean(pred_probs[cluster_fiber_indices])
            prob_std = numpy.std(pred_probs[cluster_fiber_indices])
            cluster_prob.append("%0.3f" % (prob_mean))
        else:
            cluster_length.append("nan")
            cluster_Uratio.append("nan")
            cluster_hemishpere.append("nan")
            cluster_prob.append("nan")

    return cluster_NoF, cluster_length, cluster_Uratio, cluster_hemishpere, cluster_prob


def tspc_calculation_SWM(num_clusters, predicted, x_surf, cluster_reordered_fibers):

    roi_map = numpy.load('/data06/jinwang/isbi/src/DFC/SWM_DFC-main/resources/relabel_map_hHOA.npy')
    x_surf_tmp = numpy.zeros_like(x_surf)
    for idx, roi in enumerate(roi_map[0]):
        surf_label = roi_map[1][idx]
        x_surf_tmp[x_surf == roi] = surf_label
    x_surf = x_surf_tmp

    tspc_all = []
    eps_all = []
    for c_idx in range(num_clusters):

        cluster_fiber_indices = predicted == c_idx
        num_fibers = numpy.sum(cluster_fiber_indices)

        if num_fibers > 0:
            cluster_x_surf = x_surf[cluster_fiber_indices, :]
            cluster_reorder_fibers = cluster_reordered_fibers[c_idx]

            cluster_x_surf_orig = cluster_x_surf[cluster_reorder_fibers == 0, :]
            cluster_x_surf_flip = cluster_x_surf[cluster_reorder_fibers == 1, :]
            cluster_x_surf_flip = numpy.flip(cluster_x_surf_flip)

            cluster_x_surf = numpy.concatenate((cluster_x_surf_orig, cluster_x_surf_flip))

            cluster_x_surf_ep1 = cluster_x_surf[:, 0]
            cluster_x_surf_ep2 = cluster_x_surf[:, 1]

            values, counts = numpy.unique(cluster_x_surf_ep1, return_counts=True)
            arg = numpy.argsort(counts)
            arg = numpy.flip(arg)
            values = values[arg]
            ep1 = values[0]

            values, counts = numpy.unique(cluster_x_surf_ep2, return_counts=True)
            arg = numpy.argsort(counts)
            arg = numpy.flip(arg)
            values = values[arg]
            ep2 = values[0]

            if ep1 == ep2:
                cluster_tspc = numpy.sum(numpy.logical_and(cluster_x_surf_ep1 == ep1, cluster_x_surf_ep2 == ep2)) / cluster_x_surf_ep1.shape[0]
            else:
                cluster_tspc_1 = numpy.sum(numpy.logical_and(cluster_x_surf_ep1 == ep1, cluster_x_surf_ep2 == ep2)) / cluster_x_surf_ep1.shape[0]
                cluster_tspc_2 = numpy.sum(numpy.logical_and(cluster_x_surf_ep1 == ep2, cluster_x_surf_ep2 == ep1)) / cluster_x_surf_ep1.shape[0]
                cluster_tspc = cluster_tspc_1 + cluster_tspc_2
        else:
            cluster_tspc = numpy.nan
            ep1 = 0
            ep2 = 0

        tspc_all.append(cluster_tspc)
        eps_all.append(sorted([ep1, ep2]))

    tspc_mean = numpy.nanmean(tspc_all)
    tspc_std = numpy.nanstd(tspc_all)

    return tspc_mean, tspc_std, tspc_all, eps_all


def db_calculation_SWM(num_clusters, predicted, cluster_centroids, cluster_alphas, verbose=False):

    db_all = []
    for c_i_idx in range(num_clusters):

        cluster_fiber_indices = predicted == c_i_idx
        num_fibers = numpy.sum(cluster_fiber_indices)

        if num_fibers > 0:

            cluster_i_centroid = cluster_centroids[c_i_idx, :]

            d_i = fiber_distance.fiber_distance(cluster_i_centroid, cluster_centroids)
            d_i[c_i_idx] = 1000000 # excluding the cluster itself

            alpha_i_j = cluster_alphas[c_i_idx] + cluster_alphas
            db_i_j = alpha_i_j / d_i

            db_i = numpy.nanmax(db_i_j)

            if verbose:
                print("cluster %05d: alpha = %s, db_i_j = %s" % (c_i_idx, alpha_i_j, db_i_j))
        else:
            db_i = numpy.nan

        db_all.append(db_i)

    db_mean = numpy.nanmean(db_all)
    db_std  = numpy.nanstd(db_all)

    return db_mean, db_std, db_all


def wmpg_calculation_SWM(num_clusters, preds, subnum):

    sub_cluster = numpy.zeros(num_clusters)
    for c_idx in range(num_clusters):

        cluster_fiber_indices = preds == c_idx
        num_fibers = numpy.sum(cluster_fiber_indices)

        if num_fibers > 0:
            t = subnum[preds == c_idx]
            t = numpy.round(t)
            sub_cluster[c_idx] = numpy.unique(t).shape[0]
        else:
            sub_cluster[c_idx] = numpy.nan

    wmpg_all = sub_cluster / numpy.max((subnum + 1))

    wmpg_mean = numpy.nanmean(wmpg_all)
    wmpg_std  = numpy.nanstd(wmpg_all)

    return wmpg_mean, wmpg_std, wmpg_all

def metrics_subject_calculation(params, fiber_array, preds_final, probs_final, rejected_fibers, df_stats):

    num_subjects = fiber_array.fiber_subID.max() + 1
    num_clusters = params["num_clusters"]

    df_sub_stats_list = []
    sub_NoF_list = numpy.full((num_clusters, num_subjects), numpy.nan)
    sub_Uratio_list = numpy.full((num_clusters, num_subjects), numpy.nan)
    sub_Prob_list = numpy.full((num_clusters, num_subjects), numpy.nan)
    sub_DB_list = numpy.full((num_clusters, num_subjects), numpy.nan)
    sub_alpha_list = numpy.full((num_clusters, num_subjects), numpy.nan)
    sub_TSPC_list = numpy.full((num_clusters, num_subjects), numpy.nan)

    for s_idx in range(num_subjects):
        utils.print_both(params["log"], "* Subject %04d " % (s_idx+1))
        subject_fibers_not = numpy.union1d(numpy.where(fiber_array.fiber_subID != s_idx), rejected_fibers)

        if subject_fibers_not.size == fiber_array.fiber_subID.size:
            print(" - not cluster info")
        else:
            df_stats_sub, _, _ = metrics_calculation(params, fiber_array, preds_final, probs_final,
                                                     outlier_fibers=subject_fibers_not, output=False, verbose=False)
            sub_NoF = [float(v) for v in df_stats_sub["NoFiber"].values.tolist()]
            sub_Uratio = [float(v[:5]) for v in df_stats_sub["Uratio"].values.tolist()]
            sub_Prob = [float(v[:5]) for v in df_stats_sub["Pred-Prob"].values.tolist()]
            sub_DB = [float(v) for v in df_stats_sub["DB-score"].values.tolist()]
            sub_alpha = [float(v) for v in df_stats_sub["Alpha"].values.tolist()]
            sub_TSPC = [float(v) for v in df_stats_sub["TSPC"].values.tolist()]

            sub_NoF_list[:, s_idx] = sub_NoF
            sub_Uratio_list[:, s_idx] = sub_Uratio
            sub_Prob_list[:, s_idx] = sub_Prob
            sub_DB_list[:, s_idx] = sub_DB
            sub_alpha_list[:, s_idx] = sub_alpha
            sub_TSPC_list[:, s_idx] = sub_TSPC

            df_stats_sub.drop(columns=["WMPG"])
            df_stats_sub.rename(columns=lambda x: "S%04d:" % (s_idx + 1) + x, inplace=True)

            df_sub_stats_list.append(df_stats_sub)

    cluster_sub_NoF_ = [numpy.nanmean(sub_NoF_list, axis=1), numpy.nanstd(sub_NoF_list, axis=1)]
    cluster_sub_Uratio_ = [numpy.nanmean(sub_Uratio_list, axis=1), numpy.nanstd(sub_Uratio_list, axis=1)]
    cluster_sub_Prob_ = [numpy.nanmean(sub_Prob_list, axis=1), numpy.nanstd(sub_Prob_list, axis=1)]
    cluster_sub_DB_ = [numpy.nanmean(sub_DB_list, axis=1), numpy.nanstd(sub_DB_list, axis=1)]
    cluster_sub_alpha_ = [numpy.nanmean(sub_alpha_list, axis=1), numpy.nanstd(sub_alpha_list, axis=1)]
    cluster_sub_TSPC_ = [numpy.nanmean(sub_TSPC_list, axis=1), numpy.nanstd(sub_TSPC_list, axis=1)]

    cluster_sub_NoF = []
    cluster_sub_Uratio = []
    cluster_sub_Prob = []
    cluster_sub_DB = []
    cluster_sub_alpha = []
    cluster_sub_TSPC = []
    for c_idx in range(num_clusters):
        nof_mean = cluster_sub_NoF_[0][c_idx]
        nof_std = cluster_sub_NoF_[1][c_idx]
        cluster_sub_NoF.append("%0.3f(%0.3f)" % (nof_mean, nof_std))

        Uratio_mean = cluster_sub_Uratio_[0][c_idx]
        Uratio_std = cluster_sub_Uratio_[1][c_idx]
        cluster_sub_Uratio.append("%0.3f(%0.3f)" % (Uratio_mean, Uratio_std) if not numpy.isnan(Uratio_mean) else "nan" )

        Prob_mean = cluster_sub_Prob_[0][c_idx]
        Prob_std = cluster_sub_Prob_[1][c_idx]
        cluster_sub_Prob.append("%0.3f(%0.3f)" % (Prob_mean, Prob_std) if not numpy.isnan(Prob_mean) else "nan")

        DB_mean = cluster_sub_DB_[0][c_idx]
        DB_std = cluster_sub_DB_[1][c_idx]
        cluster_sub_DB.append("%0.3f(%0.3f)" % (DB_mean, DB_std) if not numpy.isnan(DB_mean) else "nan")

        alpha_mean = cluster_sub_alpha_[0][c_idx]
        alpha_std = cluster_sub_alpha_[1][c_idx]
        cluster_sub_alpha.append("%0.3f(%0.3f)" % (alpha_mean, alpha_std) if not numpy.isnan(alpha_mean) else "nan")

        tspc_mean = cluster_sub_TSPC_[0][c_idx]
        tspc_std = cluster_sub_TSPC_[1][c_idx]
        cluster_sub_TSPC.append("%0.3f(%0.3f)" % (tspc_mean, tspc_std) if not numpy.isnan(tspc_mean) else "nan")

    data = numpy.array((cluster_sub_NoF, cluster_sub_Uratio, cluster_sub_Prob, cluster_sub_DB, cluster_sub_alpha,
                        cluster_sub_TSPC), dtype=object).transpose()
    df_avg_sub_stats = pandas.DataFrame(data=data,
                                        columns=["Sub-NoFiber", "Sub-Uratio", "Sub-Pred-Prob", "Sub-DB-score", "Sub-Alpha", "Sub-TSPC"])
    df_avg_sub_stats.index = ['cluster_{0:05d}'.format(idx + 1) for idx in df_avg_sub_stats.index]

    # appending all together
    df_sub_stats_list.insert(0, df_avg_sub_stats)
    df_sub_stats_list.insert(0, df_stats)

    df_stats_subjects = pandas.concat(df_sub_stats_list, axis=1)

    output_name = params['output_name']
    output_dir = params['output_dir']

    # output stats csv file
    stat_csv = os.path.join(output_dir, output_name + "_subjects.stats.csv")
    df_stats_subjects.to_csv(stat_csv)

    return df_stats_subjects