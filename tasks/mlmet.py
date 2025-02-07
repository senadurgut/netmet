# coding: utf-8

"""
Preprocessing tasks.
"""

__all__ = []


import abc
import contextlib
import itertools
from collections import OrderedDict, defaultdict
import os
import sys
from subprocess import call
import json
import numpy as np
import pandas as pd

import law
import luigi

from analysis_tools.utils import join_root_selection as jrs
from analysis_tools.utils import import_root, create_file_dir

from cmt.base_tasks.base import (
    Task, ConfigTask, InputData, HTCondorWorkflow, SGEWorkflow, SlurmWorkflow
)

from cmt.util import parse_workflow_file

class MLTrainingTask(ConfigTask):
    # training_config_name = luigi.Parameter(default="default", description="the name of the "
        # "training configuration, default: default")
    # training_category_name = luigi.Parameter(default="baseline_os_odd", description="the name of "
        # "a category whose selection rules are applied to data to train on, default: "
        # "baseline_os_odd")
    feature_tag = luigi.Parameter(default="default", description="the tag of features to use for "
        "the training, default: default")
    architecture = luigi.Parameter(default="256_64_16:relu", description="a string "
        "describing the network architecture, default: dense:256_64_16:relu")
    loss_name = luigi.Parameter(default="mean_absolute_percentage_error", description="the name of "
        "the loss function, default: mean_absolute_percentage_error")
    # l2_norm = luigi.FloatParameter(default=1e-3, description="the l2 regularization factor, "
        # "default: 1e-3")
    epochs = luigi.IntParameter(default=10, description="number of epochs for training, "
        "default: 10")
    learning_rate = luigi.FloatParameter(default=0.01, description="the learning rate, default: "
        "0.01")
    dropout_rate = luigi.FloatParameter(default=0.1, description="the dropout rate, default: 0.")
    batch_norm = luigi.BoolParameter(default=True, description="whether batch normalization should "
        "be applied, default: True")
    batch_size = luigi.IntParameter(default=500, description="the training batch size, default: "
        "500")
    random_seed = luigi.IntParameter(default=1, description="random seed for weight "
        "initialization, 0 means non-deterministic, default: 1")
    min_feature_score = luigi.FloatParameter(default=law.NO_FLOAT, description="minimum score "
        "for filtering used features, requires the FeatureRanking when not empty, default: empty")
    signal_dataset_name = luigi.Parameter(default="signal_nopum_new", description="name of the signal dataset, "
        "default: signal_nopum_new")

    training_id = luigi.IntParameter(default=law.NO_INT, description="when given, overwrite "
        "training parameters from the training with this branch in the training_workflow_file, "
        "default: empty")
    training_workflow_file = luigi.Parameter(description="filename with training parameters",
        default="hyperopt")
    use_gpu = luigi.BoolParameter(default=False, significant=False, description="whether to launch "
        "jobs to a gpu, default: False")

    training_hash_params = [
        "feature_tag", "architecture",
        "loss_name", "epochs", "learning_rate", "dropout_rate", "batch_norm",
        "batch_size", "random_seed", "min_feature_score",
        "signal_dataset_name",
        # "training_category_name", "training_config_name", "l2_norm", "event_weights",
    ]

    @classmethod
    def modify_param_values(cls, params):
        if "training_workflow_file" not in params or "training_id" not in params:
            return params
        if params["training_id"] == law.NO_INT:
            return params

        branch_map = parse_workflow_file(
            self.retrieve_file(f"config/{self.training_workflow_file}.yaml"))[1]

        param_names = cls.get_param_names()
        for name, value in branch_map[params["training_id"]].items():
            if name in param_names:
                params[name] = value

        params["training_id"] = law.NO_INT

        return params

    @classmethod
    def create_training_hash(cls, **kwargs):
        def fmt(key, prefix, fn):
            if key not in kwargs:
                return None
            value = fn(kwargs[key])
            if value in ("", None):
                return None
            return prefix + "_" + str(value)
        def num(n, tmpl="{}", skip_empty=False):
            if skip_empty and n in (law.NO_INT, law.NO_FLOAT):
                return None
            else:
                return tmpl.format(n).replace(".", "p")
        parts = [
            #fmt("training_category_name", "TC", str),
            fmt("feature_tag", "FT", lambda v: v.replace("*", "X").replace("?", "Y")),
            fmt("architecture", "AR", lambda v: v.replace(":", "_")),
            fmt("loss_name", "LN", str),
            # fmt("l2_norm", "L2", lambda v: num(v, "{:.2e}")),
            fmt("epochs", "EP", int),
            fmt("learning_rate", "LR", lambda v: num(v, "{:.2e}")),
            fmt("dropout_rate", "DO", num),
            fmt("batch_norm", "BN", int),
            fmt("batch_size", "BS", num),
            fmt("random_seed", "RS", str),
            fmt("min_feature_score", "MF", lambda v: num(v, skip_empty=True)),
            fmt("signal_dataset_name", "SIG", str),
        ]
        return "__".join(part for part in parts if part)

    def __init__(self, *args, **kwargs):
        super(MLTrainingTask, self).__init__(*args, **kwargs)

        # save training features, without minimum feature score filtering applied
        self.training_features = [
            feature for feature in self.config.features
            if feature.has_tag(self.feature_tag)
        ]

        # compute the storage hash
        self.training_hash = self.create_training_hash(**self.get_training_hash_data())

    def get_training_hash_data(self):
        return {p: getattr(self, p) for p in self.training_hash_params}

    def store_parts(self):
        parts = super(MLTrainingTask, self).store_parts()
        parts["training_hash"] = self.training_hash
        return parts



class MLTraining(MLTrainingTask):
    def __init__(self, *args, **kwargs):
        super(MLTraining, self).__init__(*args, **kwargs)
        self.signal_dataset = self.config.datasets.get(self.signal_dataset_name)
        self.device = "GPU: 0" if self.use_gpu else "CPU: 0"

    def requires(self):
        return InputData.req(self, dataset_name=self.signal_dataset_name, file_index=0)
        # I don't really need it, it's just a placeholder

    def output(self):
        return {
            "model": self.local_target("model.h5"),
            "loss": self.local_target("loss.pdf"),
            "acc": self.local_target("accuracy.pdf")
        }

    def generate_model(self, X_train):
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Normalization
        from keras.optimizers import Adam
        print('Training input shape:', X_train.shape)

        model = Sequential()
        if self.batch_norm:
            normalizer = Normalization(input_shape=[X_train.shape[1],], axis=-1)
            normalizer.adapt(X_train)
            model.add(normalizer)
        arch, act = self.architecture.split(":")
        for i, nneur in enumerate(arch.split("_")):
            if i == 0:
                model.add(Dense(nneur, input_shape=(X_train.shape[1],), activation=act))
            else:
                model.add(Dense(nneur, activation=act))
            if self.dropout_rate > 0.:
                model.add(Dropout(self.dropout_rate))
        model.add(Dense(1))
        model.compile(
            loss=self.loss_name,
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['mean_squared_error']
        )

        return model

    def get_data(self, dataset=None, output_y=True, output_df=False, nfiles=-1, min_puppi_pt=-1,
            remove_saturated=False, useEmu=None, remove_methf = False, sort_jets = False, compute_mjjj=False):
        import utils.tools as tools

        if not dataset:
            dataset = self.signal_dataset

        feature_params = self.config.training_feature_groups()[self.feature_tag]
        inputs = feature_params.get("inputs", [])
        inputSums = feature_params.get("inputSums", [])
        nObj = feature_params.get("nObj", 4)
        if useEmu == None:
            useEmu = feature_params.get("useEmu", False)
        useMP = feature_params.get("useMP", False)
        keepStruct = feature_params.get("keepStruct", False)

        branches = tools.getBranches(inputs, useEmu, useMP)

        if nfiles != -1:
            dataset_files = dataset.get_files(
                os.path.expandvars("$CMT_TMP_DIR/%s/" % self.config_name), check_empty=True)[0:nfiles]
        else:
            dataset_files = dataset.get_files(
                os.path.expandvars("$CMT_TMP_DIR/%s/" % self.config_name), check_empty=True)
            # os.path.expandvars("$CMT_TMP_DIR/%s/" % self.config_name), check_empty=True)[0:2]
        data = tools.getArrays(dataset_files, branches, len(dataset_files), None)

        if output_y:
            puppiMET, puppiMET_noMu = tools.getPUPPIMET(data)
            if min_puppi_pt > -1:
                data, puppiMET_noMu = tools.apply_pt_cut(data, puppiMET_noMu, min_puppi_pt)
            puppiMETNoMu_df = tools.arrayToDataframe(puppiMET_noMu, 'puppiMET_noMu', None)
            collections = tools.getCollections(data, inputSums, inputs)
            df = tools.makeDataframe(collections, None, nObj, keepStruct)

            if remove_saturated:
                df, puppiMET_noMu = tools.remove_saturated(df, puppiMET_noMu)
            return df.copy(), puppiMETNoMu_df.copy()

            if remove_methf:
                df, puppiMET_noMu = tools.remove_methf(df, puppiMET_noMu)
            return df.copy(), puppiMETNoMu_df.copy()

            if sort_jets:
                df, puppiMET_noMu = tools.sort_jets(df, puppiMET_noMu)
            return df.copy(), puppiMETNoMu_df.copy()
            
            if compute_mjjj:
                df, puppiMET_noMu = tools.compute_mjjj(df, puppiMET_noMu)
            return df.copy(), puppiMETNoMu_df.copy()

        else:
            collections = tools.getCollections(data, inputSums, inputs)
            df = tools.makeDataframe(collections, None, nObj, keepStruct)
            return df.copy(), None

    def run(self):
        from sklearn.preprocessing import StandardScaler
        import tensorflow as tf
        from matplotlib import pyplot as plt

        feature_params = self.config.training_feature_groups()[self.feature_tag]
        scaleData = feature_params.get("scaleData", False)
        trainFrac = feature_params.get("trainFrac", 0.5)
        min_puppi_pt = feature_params.get("min_puppi_pt", -1)
        remove_saturated = feature_params.get("remove_saturated", False)
        remove_methf = feature_params.get("remove_methf", True)
        sort_jets = feature_params.get("sort_jets", False)
        compute_mjjj = feature_params.get("compute_mjjj", False)

        # X, Y = self.get_data(nfiles=-1, min_puppi_pt=min_puppi_pt)
        X, Y = self.get_data(nfiles=200, min_puppi_pt=min_puppi_pt, remove_saturated=remove_saturated, compute_mjjj = compute_mjjj)

        scaler = StandardScaler()
        if scaleData:
            X[X.columns] = pd.DataFrame(scaler.fit_transform(X))

        X_train = X.sample(frac=trainFrac, random_state=3).dropna()
        Y_train = Y.loc[X_train.index]

        with tf.device(self.device):
            model = self.generate_model(X_train)
            history = model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size,
                validation_split=0.3, verbose=1)
            model.save(create_file_dir(self.output()["model"].path))

            # loss plotting
            plt.plot(history.history['loss'], label="Training loss")
            plt.plot(history.history['val_loss'], label="Validation loss")
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.savefig(create_file_dir(self.output()["loss"].path))
            plt.close()

            # acc plotting
            plt.plot(history.history['mean_squared_error'], label="Training accuracy")
            plt.plot(history.history['val_mean_squared_error'], label="Validation accuracy")
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            plt.savefig(create_file_dir(self.output()["acc"].path))
            print(create_file_dir(self.output()["loss"].path))



class MLTrainingWorkflowBase(ConfigTask, law.LocalWorkflow, HTCondorWorkflow, SGEWorkflow, SlurmWorkflow):
    training_workflow_file = MLTrainingTask.training_workflow_file
    output_collection_cls = law.FileCollection

    def __init__(self, *args, **kwargs):
        super(MLTrainingWorkflowBase, self).__init__(*args, **kwargs)

        self.workflow_data, self._file_branch_map = parse_workflow_file(
            self.retrieve_file(f"config/{self.training_workflow_file}.yaml"))

    def create_branch_map(self):
        return self._file_branch_map

    def workflow_requires(self):
        return {
            b: self.trace_branch_requires(self.as_branch(b).requires())
            for b in self.workflow_data["require_branches"]
        }

    def trace_branch_requires(self, branch_req):
        return branch_req.requires()

    def output(self):
        return self.requires().output()

    def matching_branch_data(self, task_cls):
        assert(self.is_branch())
        param_names = task_cls.get_param_names()
        return {
            key: value for key, value in self.branch_data.items()
            if key in param_names
        }


class MLTrainingWorkflow(MLTrainingWorkflowBase):
    def requires(self):
        return MLTraining.vreq(self, **self.matching_branch_data(MLTraining))

    def output(self):
        return self.requires().output()

    def run(self):
        pass


class BaseValidationTask():

    def get_output_postfix(self):
        return ""

    def output(self):
        postfix = self.get_output_postfix()
        return {
            "rate": self.local_target(f"rate_comparison{postfix}.pdf"),
            "x_rate": self.local_target(f"x_rate{postfix}.npy"),
            "y_rate": self.local_target(f"y_rate{postfix}.npy"),
            "resolution": self.local_target(f"resolution_comparison{postfix}.pdf"),
            "x_resolution": self.local_target(f"x_resolution{postfix}.npy"),
            "y_resolution": self.local_target(f"y_resolution{postfix}.npy"),
            "dist": self.local_target(f"distributions{postfix}.pdf"),
            "x_dist": self.local_target(f"x_distributions{postfix}.pdf"),
            "y_dist": self.local_target(f"y_distributions{postfix}.pdf"),
            "netmet_met": self.local_target(f"l1netmet_vs_l1met__puppimet_{self.puppi_met_cut}{postfix}.pdf"),
            "netmet_met_log": self.local_target(f"l1netmet_vs_l1met__puppimet_{self.puppi_met_cut}_log{postfix}.pdf"),
            "netmet_puppi": self.local_target(f"l1netmet_vs_puppimet__puppimet_{self.puppi_met_cut}{postfix}.pdf"),
            "netmet_puppi_log": self.local_target(f"l1netmet_vs_puppimet__puppimet_{self.puppi_met_cut}_log{postfix}.pdf"),
            "efficiency": self.local_target(f"efficiency{postfix}.pdf"),
            "x_efficiency": self.local_target(f"x_efficiency{postfix}.npy"),
            "y_efficiency": self.local_target(f"y_efficiency{postfix}.npy"),
            "y_error_efficiency": self.local_target(f"y_error_efficiency{postfix}.npy"),
            "netMET_thresh": self.local_target(f"netMET_thresh{postfix}.txt"),

            "json": self.local_target(f"plotting_data{postfix}.json")
        }

    def get_performance_plots(self, X, Y, X_train, Y_train, Xp, Yp, Xp_bkg, Yp_bkg, X_bkg_ref=None):
        import utils.plotting as plotting
        from matplotlib import pyplot as plt
        from matplotlib.colors import LogNorm, NoNorm

        plotting_data = {}

        ##################################
        # L1 MET rate vs L1 NET MET rate #
        ##################################

        l1NetMET_bkg = Yp_bkg.flatten()
        # l1MET_bkg = Xp_bkg['methf_0_pt']
        if type(X_bkg_ref) == str:
            l1MET_bkg = Xp_bkg['methf_0_hwPt'] / 2.
        else:
            l1MET_bkg = X_bkg_ref['methf_0_hwPt'] / 2.

        ax = plt.subplot()
        # rate plots must be in bins of GeV
        xrange = [0, 200]
        bins = xrange[1]

        rateHist = plt.hist(l1MET_bkg, bins=bins, range=xrange, histtype='step', label='L1 MET Rate', cumulative=-1, log=True,
            weights=[1./l1MET_bkg.shape[0] for i in range(l1MET_bkg.shape[0])])
        rateHist_netMET = plt.hist(l1NetMET_bkg, bins=bins, range=xrange, histtype='step', label='L1 NET MET Rate', cumulative=-1, log=True, 
            weights=[1./l1NetMET_bkg.shape[0] for i in range(l1NetMET_bkg.shape[0])])

        with open(create_file_dir(self.output()["x_rate"].path), "wb+") as f:
            np.save(f, rateHist_netMET[1])
        with open(create_file_dir(self.output()["y_rate"].path), "wb+") as f:
            np.save(f, rateHist_netMET[0])

        plt.legend()
        plt.savefig(create_file_dir(self.output()["rate"].path))
        plt.close('all')

        #save the values to a json file 

        plotting_data['rate'] = {
            'l1MET_bkg': l1MET_bkg.tolist(),
            "l1NetMET_bkg": l1NetMET_bkg.tolist()
        }

        # get rate at threshold
        l1MET_fixed_rate = rateHist[0][int(self.l1_met_threshold) * int((xrange[1] / bins))]
        netMET_thresh = plotting.getThreshForRate(rateHist_netMET[0], bins, l1MET_fixed_rate)

        plotting_data["rate"]["l1MET_fixed_rate"] = l1MET_fixed_rate
        plotting_data["rate"]["netMET_thresh"] = netMET_thresh



        ##################
        # MET Resolution #
        ##################

        l1MET = X['methf_0_hwPt'] / 2
        l1NetMET = Yp.flatten()
        l1MET_test = l1MET.drop(X_train.index)
        puppiMETNoMu_df_test = Y.drop(X_train.index)

        l1MET_diff = l1MET_test - puppiMETNoMu_df_test['PuppiMET_pt']
        l1NetMET_diff = l1NetMET - puppiMETNoMu_df_test['PuppiMET_pt']
        plt.hist(l1MET_diff, bins=80, range=[-100, 100], label="L1 MET Diff", histtype='step')
        result = plt.hist(l1NetMET_diff, bins=80, range=[-100, 100], label="L1 NET MET Diff", histtype='step')

        with open(create_file_dir(self.output()["x_resolution"].path), "wb+") as f:
            np.save(f, result[1])
        with open(create_file_dir(self.output()["y_resolution"].path), "wb+") as f:
            np.save(f, result[0])

        plt.legend()
        plt.savefig(create_file_dir(self.output()["resolution"].path))
        plt.close('all')

        plotting_data['resolution'] = {
            'l1MET_diff': l1MET_diff.tolist(),
            'l1NetMET_diff': l1NetMET_diff.tolist()
        }

        #################
        # Distributions #
        #################

        puppiMET_hist = plt.hist(puppiMETNoMu_df_test['PuppiMET_pt'], bins=100, range=[0, 200], histtype='step', log=True, label="PUPPI MET NoMu")
        l1MET_hist = plt.hist(l1MET_test, bins=100, range=[0, 200], histtype='step', label="L1MET")
        l1NetMET_hist = plt.hist(l1NetMET, bins=100, range=[0, 200], histtype='step', label="L1 Net MET ")

        with open(create_file_dir(self.output()["x_dist"].path), "wb+") as f:
            np.save(f, l1NetMET_hist[1])
        with open(create_file_dir(self.output()["y_dist"].path), "wb+") as f:
            np.save(f, l1NetMET_hist[0])

        plt.legend(fontsize=16)
        plt.savefig(create_file_dir(self.output()["dist"].path))
        plt.close('all')

        plotting_data["distribution"] = {
            "puppiMETNoMu": puppiMET_hist[0].tolist(),
            "l1MET": l1MET_hist[0].tolist(),
            "l1NetMET": l1NetMET_hist[0].tolist()
        }
        #######################
        # Additional 2D Plots #
        #######################

        import awkward as ak
        fig = plt.figure()
        ax = plt.subplot()

        try:
            Yp_flat = ak.flatten(Yp)
        except:
            Yp_flat = Yp

        hist2d_netmet_met, x_edges, y_edges, img = plt.hist2d(ak.to_numpy(Yp_flat), l1MET_test, bins=[50, 50],
            range=[[self.puppi_met_cut, 200 + self.puppi_met_cut],
                [self.puppi_met_cut, 200 + self.puppi_met_cut]])
        cbar = fig.colorbar(img)
        ax.set_xlabel('L1 NET MET [GeV]')
        ax.set_ylabel('L1 MET [GeV]')
        plt.savefig(create_file_dir(self.output()["netmet_met"].path))
        plt.close('all')

        plotting_data["l1netmet_vs_l1met"] = {
            "hist2d": hist2d_netmet_met.tolist(),
            "x_edges": x_edges.tolist(),
            "y_edges": y_edges.tolist()
        }

        fig = plt.figure()
        ax = plt.subplot()
        img = plt.hist2d(ak.to_numpy(Yp_flat), l1MET_test, bins=[50, 50],
            range=[[self.puppi_met_cut, 200 + self.puppi_met_cut],
                [self.puppi_met_cut, 200 + self.puppi_met_cut]],
            norm=LogNorm()
        )
        cbar = fig.colorbar(img[3])
        ax.set_xlabel('L1 NET MET [GeV]')
        ax.set_ylabel('L1 MET [GeV]')
        plt.savefig(create_file_dir(self.output()["netmet_met_log"].path))
        plt.close('all')

        fig = plt.figure()
        ax = plt.subplot()
        plt.hist2d(ak.to_numpy(Yp_flat), ak.to_numpy(puppiMETNoMu_df_test['PuppiMET_pt']),
            bins=[50, 50],
            range=[[self.puppi_met_cut, 200 + self.puppi_met_cut],
                [self.puppi_met_cut, 200 + self.puppi_met_cut]])
        cbar = fig.colorbar(img[3])
        ax.set_xlabel('L1 NET MET [GeV]')
        ax.set_ylabel('PUPPI MET No Mu [GeV]')
        plt.savefig(create_file_dir(self.output()["netmet_puppi"].path))
        plt.close('all')

        fig = plt.figure()
        ax = plt.subplot()
        plt.hist2d(ak.to_numpy(Yp_flat), ak.to_numpy(puppiMETNoMu_df_test['PuppiMET_pt']),
            bins=[50, 50],
            range=[[self.puppi_met_cut, 200 + self.puppi_met_cut],
                [self.puppi_met_cut, 200 + self.puppi_met_cut]],
            norm=LogNorm()
        )
        cbar = fig.colorbar(img[3])
        ax.set_xlabel('L1 NET MET [GeV]')
        ax.set_ylabel('PUPPI MET No Mu [GeV]')
        plt.savefig(create_file_dir(self.output()["netmet_puppi_log"].path))
        plt.close('all')
  
        json_output_path = create_file_dir(self.output()["json"].path)
        with open(json_output_path, "w") as json_file:
            json.dump(plotting_data, json_file, indent=4)

        ##################
        # MET Efficiency #
        ##################

        fig = plt.figure()
        ax = plt.subplot()
        eff_data, xvals, eff_errors = plotting.efficiency(l1MET, Y['PuppiMET_pt'],
            self.l1_met_threshold, 10, 400)
        netMET_eff_data, _, netMET_eff_errors = plotting.efficiency(l1NetMET,
            puppiMETNoMu_df_test['PuppiMET_pt'],
            netMET_thresh, 10, 400)
        plt.axhline(0.95, linestyle='--', color='black')
        plt.errorbar(xvals, eff_data, eff_errors, label="L1 MET > " + str(self.l1_met_threshold),
            marker='o', capsize=7, linestyle='none')
        plt.errorbar(xvals, netMET_eff_data, netMET_eff_errors, label="L1 NETMET > " + str(netMET_thresh),
            marker='o', capsize=7, linestyle='none')

        with open(create_file_dir(self.output()["x_efficiency"].path), "wb+") as f:
            np.save(f, xvals)
        with open(create_file_dir(self.output()["y_efficiency"].path), "wb+") as f:
            np.save(f, np.array(netMET_eff_data))
        with open(create_file_dir(self.output()["y_error_efficiency"].path), "wb+") as f:
            np.save(f, np.array(netMET_eff_errors))
        with open(create_file_dir(self.output()["netMET_thresh"].path), "w+") as f:
            f.write(str(netMET_thresh))

        ax.set_xlabel('PUPPI MET No Mu [GeV]')
        ax.set_ylabel('Efficiency')
        plt.legend()
        plt.savefig(create_file_dir(self.output()["efficiency"].path))
        plt.close('all')


class MLValidation(BaseValidationTask, MLTraining):
    background_dataset_name = luigi.Parameter(default="background_nopum_new", description="name of the "
        "background dataset, default: background_nopum_new")
    l1_met_threshold = luigi.IntParameter(default=80, description="value of the MET threshold, "
        "default: 50")
    puppi_met_cut = luigi.IntParameter(default=0, description="minimum cut on the PuppiMET value, "
        "default: 0")

    def __init__(self, *args, **kwargs):
        super(MLValidation, self).__init__(*args, **kwargs)
        self.background_dataset = self.config.datasets.get(self.background_dataset_name)

    def requires(self):
        return MLTraining.vreq(self)

    def run(self):
        import utils.plotting as plotting
        from sklearn.preprocessing import StandardScaler
        import tensorflow as tf
        from matplotlib import pyplot as plt
        from matplotlib.colors import LogNorm, NoNorm
        import keras

        feature_params = self.config.training_feature_groups()[self.feature_tag]
        scaleData = feature_params.get("scaleData", False)
        trainFrac = feature_params.get("trainFrac", 0.5)
        remove_saturated = feature_params.get("remove_saturated", False)
        remove_methf = feature_params.get("remove_methf", False)
        sort_jets = feature_params.get("sort_jets", False)
        compute_mjjj = feature_params.get("compute_mjjj", True)

        X, Y = self.get_data(nfiles=200, remove_saturated=remove_saturated, remove_methf = remove_methf, 
        sort_jets = sort_jets, compute_mjjj = compute_mjjj)

        scaler = StandardScaler()
        if scaleData:
            X[X.columns] = pd.DataFrame(scaler.fit_transform(X))

        X_train = X.sample(frac=trainFrac, random_state=3).dropna()
        Y_train = Y.loc[X_train.index]

        # predict values for efficiency
        modelFile = self.input()["model"].path
        Xp = X.drop(X_train.index)
        with tf.device(self.device):
            model = keras.models.load_model(modelFile)
            Yp = model.predict(Xp)

        # Background
        Xp_bkg, _ = self.get_data(self.background_dataset, nfiles=50, output_y=False)
        if scaleData:
            Xp_bkg[Xp_bkg.columns] = pd.DataFrame(scaler.fit_transform(Xp_bkg))
        with tf.device(self.device):
            model = keras.models.load_model(modelFile)
            Yp_bkg = model.predict(Xp_bkg)

        self.get_performance_plots(X, Y, X_train, Y_train, Xp, Yp, Xp_bkg, Yp_bkg)


class MLValidationWorkflow(MLTrainingWorkflow):
    def requires(self):
        return MLValidation.vreq(self, **self.matching_branch_data(MLValidation))


class BDTTrainingTask(ConfigTask):
    # training_config_name = luigi.Parameter(default="default", description="the name of the "
        # "training configuration, default: default")
    # training_category_name = luigi.Parameter(default="baseline_os_odd", description="the name of "
        # "a category whose selection rules are applied to data to train on, default: "
        # "baseline_os_odd")
    feature_tag = luigi.Parameter(default="default", description="the tag of features to use for "
        "the training, default: default")
    n_estimators = luigi.IntParameter(default=10, description="an integer "
        "describing the number of estimators, default: 10")
    max_depth = luigi.IntParameter(default=6, description="an integer "
        "describing the maximum depth of each estimator, default: 6")
    objective = luigi.Parameter(default="reg:linear", description="the name of "
        "the loss function, default: neg_mean_absolute_error")
    random_seed = luigi.IntParameter(default=1, description="random seed for weight "
        "initialization, 0 means non-deterministic, default: 1")
    signal_dataset_name = luigi.Parameter(default="signal_nopum_new", description="name of the signal dataset, "
        "default: signal_nopum_new")

    training_id = luigi.IntParameter(default=law.NO_INT, description="when given, overwrite "
        "training parameters from the training with this branch in the training_workflow_file, "
        "default: empty")
    training_workflow_file = luigi.Parameter(description="filename with training parameters",
        default="hyperopt_bdt")
    use_gpu = luigi.BoolParameter(default=False, significant=False, description="whether to launch "
        "jobs to a gpu, default: False")

    training_hash_params = [
        "feature_tag", "n_estimators", "max_depth", "objective", "random_seed", "signal_dataset_name",
        # "training_category_name", "training_config_name", "l2_norm", "event_weights",
    ]

    @classmethod
    def modify_param_values(cls, params):
        if "training_workflow_file" not in params or "training_id" not in params:
            return params
        if params["training_id"] == law.NO_INT:
            return params

        branch_map = parse_workflow_file(
            self.retrieve_file(f"config/{self.training_workflow_file}.yaml"))[1]

        param_names = cls.get_param_names()
        for name, value in branch_map[params["training_id"]].items():
            if name in param_names:
                params[name] = value

        params["training_id"] = law.NO_INT

        return params

    @classmethod
    def create_training_hash(cls, **kwargs):
        def fmt(key, prefix, fn):
            if key not in kwargs:
                return None
            value = fn(kwargs[key])
            if value in ("", None):
                return None
            return prefix + "_" + str(value)
        def num(n, tmpl="{}", skip_empty=False):
            if skip_empty and n in (law.NO_INT, law.NO_FLOAT):
                return None
            else:
                return tmpl.format(n).replace(".", "p")
        parts = [
            #fmt("training_category_name", "TC", str),
            fmt("feature_tag", "FT", lambda v: v.replace("*", "X").replace("?", "Y")),
            fmt("n_estimators", "N", int),
            fmt("max_depth", "MD", int),
            fmt("objective", "OBJ", lambda v: v.replace(":", "_")),
            fmt("random_seed", "RS", int),
            fmt("signal_dataset_name", "SIG", str),
        ]
        return "__".join(part for part in parts if part)

    def __init__(self, *args, **kwargs):
        super(BDTTrainingTask, self).__init__(*args, **kwargs)

        # save training features, without minimum feature score filtering applied
        self.training_features = [
            feature for feature in self.config.features
            if feature.has_tag(self.feature_tag)
        ]

        # compute the storage hash
        self.training_hash = self.create_training_hash(**self.get_training_hash_data())

    def get_training_hash_data(self):
        return {p: getattr(self, p) for p in self.training_hash_params}

    def store_parts(self):
        parts = super(BDTTrainingTask, self).store_parts()
        parts["training_hash"] = self.training_hash
        return parts


class BDTTraining(BDTTrainingTask):
    def __init__(self, *args, **kwargs):
        super(BDTTraining, self).__init__(*args, **kwargs)
        self.signal_dataset = self.config.datasets.get(self.signal_dataset_name)

    def requires(self):
        return InputData.req(self, dataset_name=self.signal_dataset_name, file_index=0)
        # I don't really need it, it's just a placeholder

    def output(self):
        return {
            "json": self.local_target("model.json"),
            "model": self.local_target("model.model"),
            "loss": self.local_target("loss.pdf"),
        }

    def generate_model(self):
        import xgboost
        def custom_turnon_loss(y_pred, y_val):
            grad = np.where(np.logical_or(y_val < 50, y_val > 180), (y_val-y_pred) ** 2, (y_val-y_pred) ** 4)
            hess = np.where(np.logical_or(y_val < 50, y_val > 180), 2 * (y_val-y_pred), 4 * (y_val-y_pred) ** 3)
            return grad, hess

        objective = self.objective
        if self.objective.startswith("custom"):
            objective = eval(self.objective)

        return xgboost.XGBRegressor(objective=objective, n_estimators=self.n_estimators,
            seed=self.random_seed, max_depth=self.max_depth)

    def run(self):
        from sklearn.preprocessing import StandardScaler
        from matplotlib import pyplot as plt

        feature_params = self.config.training_feature_groups()[self.feature_tag]
        trainFrac = feature_params.get("trainFrac", 0.5)
        scaleData = feature_params.get("scaleData", False)
        min_puppi_pt = feature_params.get("min_puppi_pt", -1)
        remove_saturated = feature_params.get("remove_saturated", False)

        X, Y = MLTraining.get_data(self, nfiles=-1, min_puppi_pt=min_puppi_pt,
            remove_saturated=remove_saturated)
        # X, Y = MLTraining.get_data(self, nfiles=2)

        scaler = StandardScaler()
        if scaleData:
            X[X.columns] = pd.DataFrame(scaler.fit_transform(X))
        X_train_test = X.sample(frac=trainFrac, random_state=3).dropna()
        # Y_train_test = Y.loc[X_train.index]
        X_train = X_train_test.sample(frac=0.7, random_state=3).dropna()
        Y_train = Y.loc[X_train.index]
        X_test = X_train_test.drop(X_train.index)
        Y_test = Y.loc[X_test.index]

        model = self.generate_model()
        evalset = [(X_train, Y_train), (X_test, Y_test)]
        model.fit(X_train, Y_train, eval_metric='rmse', eval_set=evalset)

        # retrieve performance metrics
        results = model.evals_result()
        model.save_model(create_file_dir(self.output()["json"].path))
        model.save_model(create_file_dir(self.output()["model"].path))

        # loss plotting
        plt.plot(results['validation_0']['rmse'], label="Train")
        plt.plot(results['validation_1']['rmse'], label="Test")
        plt.ylabel('rmse')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(create_file_dir(self.output()["loss"].path))
        plt.close()


class BDTTrainingWorkflowBase(MLTrainingWorkflowBase):
    training_workflow_file = BDTTrainingTask.training_workflow_file


class BDTTrainingWorkflow(BDTTrainingWorkflowBase):
    def requires(self):
        return BDTTraining.vreq(self, **self.matching_branch_data(BDTTraining))

    def output(self):
        return self.requires().output()

    def run(self):
        pass


class BDTValidation(BaseValidationTask, BDTTraining):
    background_dataset_name = luigi.Parameter(default="background_nopum_new", description="name of the "
        "background dataset, default: background_nopum_new")
    reference_background_dataset_name = luigi.Parameter(default="", description="name of the "
        "reference background dataset, default: none")
    l1_met_threshold = luigi.IntParameter(default=80, description="value of the MET threshold, "
        "default: 80")
    puppi_met_cut = luigi.IntParameter(default=0, description="minimum cut on the PuppiMET value, "
        "default: 0")

    def __init__(self, *args, **kwargs):
        super(BDTValidation, self).__init__(*args, **kwargs)
        self.background_dataset = self.config.datasets.get(self.background_dataset_name)

    def requires(self):
        return BDTTraining.vreq(self)

    def run(self):
        from sklearn.preprocessing import StandardScaler
        import xgboost
        from scipy.special import expit

        feature_params = self.config.training_feature_groups()[self.feature_tag]
        scaleData = feature_params.get("scaleData", False)
        trainFrac = feature_params.get("trainFrac", 0.5)

        # X, Y = MLTraining.get_data(self, nfiles=50)
        X, Y = MLTraining.get_data(self, nfiles=-1)
        # X, Y = MLTraining.get_data(self, nfiles=2)

        scaler = StandardScaler()
        if scaleData:
            X[X.columns] = pd.DataFrame(scaler.fit_transform(X))
        X_train = X.sample(frac=trainFrac, random_state=3).dropna()
        Y_train = Y.loc[X_train.index]

        # predict values for efficiency
        modelFile = self.input()["model"].path
        Xp = X.drop(X_train.index)

        model = xgboost.XGBRegressor()
        model.load_model(modelFile)

        Yp = model.predict(Xp)

        Xp_bkg, _ = MLTraining.get_data(self, self.background_dataset, output_y=False, nfiles=100)
        # Xp_bkg, _ = MLTraining.get_data(self, self.background_dataset, output_y=False, nfiles=1)
        if scaleData:
            Xp_bkg[Xp_bkg.columns] = pd.DataFrame(scaler.fit_transform(Xp_bkg))

        Yp_bkg = model.predict(Xp_bkg)

        X_bkg_ref = ""
        if self.reference_background_dataset_name:
            reference_background_dataset = self.config.datasets.get(self.reference_background_dataset_name)
            X_bkg_ref, _ = MLTraining.get_data(self, reference_background_dataset, output_y=False, nfiles=100, useEmu=False)

        self.get_performance_plots(X, Y, X_train, Y_train, Xp, Yp, Xp_bkg, Yp_bkg, X_bkg_ref)


class BDTValidationWorkflow(BDTTrainingWorkflow):
    def requires(self):
        return BDTValidation.vreq(self, **self.matching_branch_data(BDTValidation))


# BDT Quality

class BDTQualityTraining(BDTTraining):

    def generate_model(self):
        import xgboost
        return xgboost.XGBClassifier(objective=self.objective, n_estimators=self.n_estimators,
            seed=self.random_seed)

    def run(self):
        from sklearn.preprocessing import StandardScaler

        feature_params = self.config.training_feature_groups()[self.feature_tag]
        trainFrac = feature_params.get("trainFrac", 0.5)
        scaleData = feature_params.get("scaleData", False)
        min_puppi_pt = feature_params.get("min_puppi_pt", -1)

        # X, Y = MLTraining.get_data(self, nfiles=-1, min_puppi_pt=min_puppi_pt)
        X, Y = MLTraining.get_data(self, nfiles=5)

        qual = []
        for l1met, pupmet in zip(X['methf_0_pt'], Y["PuppiMET_pt"]):
            if pupmet - l1met < 20:
                qual.append(1)
            # elif abs(l1met - pupmet) < 30:
                # qual.append(2)
            # elif abs(l1met - pupmet) < 40:
                # qual.append(1)
            else:
                qual.append(0)
        qual = pd.DataFrame(qual)

        scaler = StandardScaler()
        if scaleData:
            X[X.columns] = pd.DataFrame(scaler.fit_transform(X))
        X_train = X.sample(frac=trainFrac, random_state=3).dropna()
        Y_train = qual.loc[X_train.index]

        model = self.generate_model()
        history = model.fit(X_train, Y_train)
        model.save_model(create_file_dir(self.output()["model"].path))

        Xp = X.drop(X_train.index)
        Yp = model.predict(Xp)
        Ytrue = qual.drop(X_train.index)

        # from sklearn.metrics import confusion_matrix, accuracy_score
        # c = confusion_matrix(Ytrue, Yp)
        # print(c)
        # accuracy = accuracy_score(Ytrue, Yp)
        # print("Accuracy: %.2f%%" % (accuracy * 100.0))


class BDTQualityTrainingWorkflowBase(MLTrainingWorkflowBase):
    #training_workflow_file = BDTTrainingTask.training_workflow_file
    training_workflow_file = "hyperopt_bdt_q"


class BDTQualityTrainingWorkflow(BDTQualityTrainingWorkflowBase):
    def requires(self):
        return BDTQualityTraining.vreq(self, **self.matching_branch_data(BDTQualityTraining))

    def output(self):
        return self.requires().output()

    def run(self):
        pass


# BDT Signal vs Background

class BDTSigBkgTraining(BDTTraining):

    def generate_model(self):
        import xgboost
        return xgboost.XGBClassifier(objective=self.objective, n_estimators=self.n_estimators,
            seed=self.random_seed)

    def run(self):
        from sklearn.preprocessing import StandardScaler

        feature_params = self.config.training_feature_groups()[self.feature_tag]
        trainFrac = feature_params.get("trainFrac", 0.5)
        scaleData = feature_params.get("scaleData", False)
        min_puppi_pt = feature_params.get("min_puppi_pt", -1)

        # X, Y = MLTraining.get_data(self, nfiles=-1, min_puppi_pt=min_puppi_pt)
        X, _ = MLTraining.get_data(self, output_y=False, nfiles=10)
        X_bkg, _ = MLTraining.get_data(self, nfiles=5, dataset=self.background_dataset,
            output_y=False)

        qual = []
        for l1met, pupmet in zip(X['methf_0_pt'], Y["PuppiMET_pt"]):
            if pupmet - l1met < 20:
                qual.append(1)
            # elif abs(l1met - pupmet) < 30:
                # qual.append(2)
            # elif abs(l1met - pupmet) < 40:
                # qual.append(1)
            else:
                qual.append(0)
        qual = pd.DataFrame(qual)

        scaler = StandardScaler()
        if scaleData:
            X[X.columns] = pd.DataFrame(scaler.fit_transform(X))
        X_train = X.sample(frac=trainFrac, random_state=3).dropna()
        Y_train = qual.loc[X_train.index]

        model = self.generate_model()
        history = model.fit(X_train, Y_train)
        model.save_model(create_file_dir(self.output()["model"].path))

        Xp = X.drop(X_train.index)
        Yp = model.predict(Xp)
        Ytrue = qual.drop(X_train.index)

        # from sklearn.metrics import confusion_matrix, accuracy_score
        # c = confusion_matrix(Ytrue, Yp)
        # print(c)
        # accuracy = accuracy_score(Ytrue, Yp)
        # print("Accuracy: %.2f%%" % (accuracy * 100.0))


class BDTQualityTrainingWorkflowBase(MLTrainingWorkflowBase):
    #training_workflow_file = BDTTrainingTask.training_workflow_file
    training_workflow_file = "hyperopt_bdt_q"


class BDTQualityTrainingWorkflow(BDTQualityTrainingWorkflowBase):
    def requires(self):
        return BDTQualityTraining.vreq(self, **self.matching_branch_data(BDTQualityTraining))

    def output(self):
        return self.requires().output()

    def run(self):
        pass


class BDTInputPrinter(BDTTrainingTask):
    def __init__(self, *args, **kwargs):
        super(BDTInputPrinter, self).__init__(*args, **kwargs)
        self.signal_dataset = self.config.datasets.get(self.signal_dataset_name)
        # self.device = "GPU: 0" if self.use_gpu else "CPU: 0"

    def requires(self):
        return InputData.req(self, dataset_name=self.signal_dataset_name, file_index=0)
        # I don't really need it, it's just a placeholder

    def output(self):
        return {
            "x_train": self.local_target("x_train.npy"),
            "y_train": self.local_target("y_train.npy"),
            "x_valid": self.local_target("x_valid.npy"),
            "y_valid": self.local_target("y_valid.npy"),
        }

    def run(self):
        from sklearn.preprocessing import StandardScaler

        feature_params = self.config.training_feature_groups()[self.feature_tag]
        trainFrac = feature_params.get("trainFrac", 0.5)
        scaleData = feature_params.get("scaleData", False)
        min_puppi_pt = feature_params.get("min_puppi_pt", -1)

        #X, Y = MLTraining.get_data(self, nfiles=-1, min_puppi_pt=min_puppi_pt)
        X, Y = MLTraining.get_data(self, nfiles=2)

        scaler = StandardScaler()
        if scaleData:
            X[X.columns] = pd.DataFrame(scaler.fit_transform(X))
        x_train = X.sample(frac=trainFrac, random_state=3).dropna()
        y_train = Y.loc[x_train.index]

        x_valid = X.drop(x_train.index)
        y_valid = Y.drop(y_train.index)

        outputs = self.output()
        for cat in ["x_train", "y_train", "x_valid", "y_valid"]:
            with open(create_file_dir(outputs[cat].path), "wb+") as f:
                np.save(f, eval(cat))


class BaseComparisonTask():
    def output(self):
        return {
            "rate": self.local_target("rate.pdf"),
            "resolution": self.local_target("resolution.pdf"),
            "dist": self.local_target("distributions.pdf"),
            "efficiency": self.local_target("efficiency.pdf"),
        }

    def store_parts(self):
        parts = super(BaseComparisonTask, self).store_parts()
        del parts["training_hash"]
        return parts

    def get_comparison_plots(self, inputs):
        from matplotlib import pyplot as plt

        # Rate, resolution, and distributions
        x_labels = {
            "rate": "L1 NET MET [GeV]",
            "dist": "L1 NET MET [GeV]",
            "resolution": "L1 NET MET - Puppi MET No Mu [GeV]",
        }
        for plot in ["rate", "dist", "resolution"]:
            ax = plt.subplot()
            for key, vals in inputs.items():
                with open(vals[f'x_{plot}'].path, 'rb') as f:
                    x = np.load(f)
                with open(vals[f'y_{plot}'].path, 'rb') as f:
                    y = np.load(f)
                # if plot == "rate":
                    # print(key)
                    # print(y)
                if plot == "resolution":
                    y /= sum(y)
                plt.stairs(y, x, label=f"Branch {key}")
            ax.set_xlabel(x_labels[plot])
            ax.set_ylabel('Events')
            plt.legend()
            if plot == "rate":
                plt.yscale('log')
            plt.savefig(create_file_dir(self.output()[plot].path))
            plt.close('all')

        # Efficiency
        ax = plt.subplot()
        for key, vals in inputs.items():
            with open(vals[f'x_efficiency'].path, 'rb') as f:
                x = np.load(f)
            with open(vals[f'y_efficiency'].path, 'rb') as f:
                y = np.load(f)
            with open(vals[f'y_error_efficiency'].path, 'rb') as f:
                y_errors = np.load(f)
            with open(vals[f'netMET_thresh'].path, 'r') as f:
                netMET_thresh = f.readlines()[0].strip()
            plt.errorbar(x, y, y_errors, label=f"Branch {key}, L1 NETMET > " + str(netMET_thresh),
                marker='o', capsize=7, linestyle='none')
            print(key)
            print(y)
        ax.set_xlabel('PUPPI MET No Mu [GeV]')
        ax.set_ylabel('Efficiency')
        plt.legend()
        plt.savefig(create_file_dir(self.output()["efficiency"].path))
        plt.close('all')

    def run(self):
        inputs = self.input()
        inputs = {key: vals for key, vals in inputs["collection"].targets.items()}
        self.get_comparison_plots(inputs)


class MLComparison(BaseComparisonTask, MLValidation):
    def requires(self):
        return MLValidationWorkflow.vreq(self)


class BDTComparison(BaseComparisonTask, BDTValidation):
    def requires(self):
        return BDTValidationWorkflow.vreq(self)
